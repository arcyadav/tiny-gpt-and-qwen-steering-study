"""A small decoder-only transformer (GPT) where every architecture choice is a config switch.

Switches (see GPTConfig):
    pos         learned | sinusoidal | rope | alibi | none
    norm        layernorm | rmsnorm
    norm_place  pre | post
    ffn         gelu | swiglu
    n_kv_head   n_head (MHA) | 1 < k < n_head (GQA) | 1 (MQA)
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

POS_TYPES = ("learned", "sinusoidal", "rope", "alibi", "none")
NORM_TYPES = ("layernorm", "rmsnorm")
NORM_PLACES = ("pre", "post")
FFN_TYPES = ("gelu", "swiglu")


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256  # training context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    n_kv_head: int | None = None  # None -> same as n_head (MHA)
    pos: str = "rope"
    norm: str = "rmsnorm"
    norm_place: str = "pre"
    ffn: str = "swiglu"
    dropout: float = 0.0
    rope_base: float = 10000.0

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.pos in POS_TYPES, f"pos must be one of {POS_TYPES}"
        assert self.norm in NORM_TYPES, f"norm must be one of {NORM_TYPES}"
        assert self.norm_place in NORM_PLACES, f"norm_place must be one of {NORM_PLACES}"
        assert self.ffn in FFN_TYPES, f"ffn must be one of {FFN_TYPES}"
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

    @property
    def head_dim(self):
        return self.n_embd // self.n_head

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- normalisation

class RMSNorm(nn.Module):
    """Scale by the root-mean-square only: no mean subtraction, no bias (Zhang & Sennrich, 2019)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


def make_norm(cfg):
    return RMSNorm(cfg.n_embd) if cfg.norm == "rmsnorm" else nn.LayerNorm(cfg.n_embd)


# ---------------------------------------------------------------- positions

def sinusoidal_table(T, dim, device):
    """The fixed sin/cos position table of Vaswani et al. (2017)."""
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(T, dim, device=device)
    table[:, 0::2] = torch.sin(pos * freq)
    table[:, 1::2] = torch.cos(pos * freq)
    return table


def rope_cos_sin(T, head_dim, base, device):
    """cos/sin tables for rotary embeddings, shape (T, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv_freq)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate pairs (x_i, x_{i+d/2}) of every head by a position-dependent angle (Su et al., 2021)."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rotated * sin).type_as(x)


def alibi_slopes(n_head):
    """Per-head slopes from the ALiBi paper (Press et al., 2022), incl. non-power-of-2 head counts."""
    def power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(n_head).is_integer():
        return power_of_2(n_head)
    closest = 2 ** math.floor(math.log2(n_head))
    return power_of_2(closest) + alibi_slopes(2 * closest)[0::2][: n_head - closest]


# ---------------------------------------------------------------- blocks

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        if cfg.pos == "alibi":
            self.register_buffer("slopes", torch.tensor(alibi_slopes(cfg.n_head)), persistent=False)

    def forward(self, x, rope=None, return_attn=False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # GQA / MQA: each group of query heads shares one key/value head
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)

        mask = None  # additive float mask (B or 1, heads, T, T), only needed for ALiBi
        if self.cfg.pos == "alibi":
            idx = torch.arange(T, device=x.device)
            distance = (idx.view(-1, 1) - idx.view(1, -1)).clamp(min=0).to(q.dtype)  # i - j for j <= i
            mask = -self.slopes.to(q.dtype).view(1, -1, 1, 1) * distance
            future = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
            mask = mask.masked_fill(future, float("-inf"))

        if return_attn:
            # explicit path so the attention weights can be inspected
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            else:
                future = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
                scores = scores.masked_fill(future, float("-inf"))
            attn = scores.softmax(dim=-1)
            y = attn @ v
        else:
            attn = None
            drop = self.dropout if self.training else 0.0
            if mask is not None:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y), attn


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kind = cfg.ffn
        d = cfg.n_embd
        if cfg.ffn == "gelu":
            self.up = nn.Linear(d, 4 * d, bias=False)
            self.down = nn.Linear(4 * d, d, bias=False)
        else:
            # SwiGLU: 3 matrices, hidden = 8/3 * d keeps the parameter count equal to the GELU MLP
            hidden = 64 * round(8 * d / 3 / 64)
            self.gate = nn.Linear(d, hidden, bias=False)
            self.up = nn.Linear(d, hidden, bias=False)
            self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        if self.kind == "gelu":
            return self.down(F.gelu(self.up(x)))
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pre = cfg.norm_place == "pre"
        self.norm1, self.norm2 = make_norm(cfg), make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ffn = FeedForward(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope=None, return_attn=False):
        if self.pre:
            a, attn = self.attn(self.norm1(x), rope, return_attn)
            x = x + self.drop(a)
            x = x + self.drop(self.ffn(self.norm2(x)))
        else:  # post-norm, as in the original transformer
            a, attn = self.attn(x, rope, return_attn)
            x = self.norm1(x + self.drop(a))
            x = self.norm2(x + self.drop(self.ffn(x)))
        return x, attn


# ---------------------------------------------------------------- model

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.pos == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        # pre-norm needs a final norm before the output; post-norm output is already normalised
        self.final_norm = make_norm(cfg) if cfg.norm_place == "pre" else nn.Identity()
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        # GPT-2: shrink the init of projections that write into the residual stream
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # tied weights are counted once

    def kv_cache_bytes_per_token(self, bytes_per_value=2):
        """Memory a KV cache needs per generated token (keys + values, all layers), fp16 by default."""
        c = self.cfg
        return 2 * c.n_layer * c.n_kv_head * c.head_dim * bytes_per_value

    def forward(self, idx, targets=None, return_attn=False):
        B, T = idx.shape
        cfg = self.cfg
        x = self.tok_emb(idx)
        if cfg.pos == "learned":
            if T > cfg.block_size:
                raise ValueError(f"learned positions only exist up to {cfg.block_size} tokens, got {T}")
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        elif cfg.pos == "sinusoidal":
            # Vaswani et al. scale token embeddings by sqrt(d) so they are not swamped by the table
            x = x * math.sqrt(cfg.n_embd) + sinusoidal_table(T, cfg.n_embd, idx.device).to(x.dtype)
        rope = rope_cos_sin(T, cfg.head_dim, cfg.rope_base, idx.device) if cfg.pos == "rope" else None
        x = self.drop(x)

        attns = []
        for block in self.blocks:
            x, attn = block(x, rope, return_attn)
            attns.append(attn)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        if return_attn:
            return logits, loss, attns  # attns: list over layers of (B, heads, T, T)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        for _ in range(max_new_tokens):
            context = idx[:, -self.cfg.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            next_id = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx

