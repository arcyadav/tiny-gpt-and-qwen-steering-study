"""Activation steering: build concept directions from contrastive pairs and add them to the
residual stream while the model generates. The model weights are never changed.

"Layer l" always means the residual stream right after transformer block l (0-indexed).
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def _hidden(output):
    """A block's output is either the hidden-state tensor or a tuple starting with it."""
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, new):
    return (new, *output[1:]) if isinstance(output, tuple) else new


class Steerer:
    """Shared steering logic. Subclasses provide: device, dtype, model, layers, n_layers, pad_id,
    _forward_batch(texts) -> (forward kwargs, mask of tokens to average), _ids(text),
    _logits(input_ids, attention_mask) and generate()."""

    @torch.no_grad()
    def _token_sums(self, texts, batch_size=32):
        """Per text: sum of residual-stream vectors after every block over its tokens, and the
        token count. Pad tokens and each text's first token (unusually large activations) are
        skipped. Returns sums (n_texts, n_layers, d_model) float32 and counts (n_texts,)."""
        all_sums, all_counts = [], []
        for i in range(0, len(texts), batch_size):
            inputs, mask = self._forward_batch(texts[i : i + batch_size])
            captured = []
            hooks = [layer.register_forward_hook(lambda m, inp, out: captured.append(_hidden(out)))
                     for layer in self.layers]
            try:
                self.model(**inputs)
            finally:
                for h in hooks:
                    h.remove()
            m = mask.unsqueeze(-1).float()
            all_sums.append(torch.stack([(h.float() * m).sum(dim=1) for h in captured], dim=1).cpu())
            all_counts.append(mask.sum(dim=1).cpu())
        return torch.cat(all_sums), torch.cat(all_counts)

    def layer_means(self, texts, batch_size=32):
        """Mean residual-stream vector after every block over all tokens of all texts.
        Returns a (n_layers, d_model) float32 tensor."""
        sums, counts = self._token_sums(texts, batch_size)
        return sums.sum(0) / counts.sum()

    def text_means(self, texts, batch_size=32):
        """Mean residual-stream vector of each text separately: (n_texts, n_layers, d_model)."""
        sums, counts = self._token_sums(texts, batch_size)
        return sums / counts.clamp(min=1).view(-1, 1, 1)

    def concept_vectors(self, pairs, batch_size=32):
        """One steering vector per layer: mean(positive) - mean(negative). Shape (n_layers, d_model)."""
        pos = self.layer_means([p for p, _ in pairs], batch_size)
        neg = self.layer_means([n for _, n in pairs], batch_size)
        return pos - neg

    @contextmanager
    def steering(self, layer, vector, alpha):
        """While active, add alpha * vector to the residual stream after block `layer`, at every token."""
        if layer is None or alpha == 0:
            yield
            return
        add = (alpha * vector).to(self.device, self.dtype)
        hook = self.layers[layer].register_forward_hook(
            lambda m, inp, out: _replace_hidden(out, _hidden(out) + add))
        try:
            yield
        finally:
            hook.remove()

    @torch.no_grad()
    def continuation_nll(self, prompts, continuations, batch_size=16):
        """Mean negative log-likelihood per token of each continuation given its prompt, under the
        unsteered model. Rows are right-padded, which is safe for causal attention."""
        nlls = []
        for i in range(0, len(prompts), batch_size):
            rows = []
            for p, c in zip(prompts[i : i + batch_size], continuations[i : i + batch_size]):
                p_ids = self._ids(p)
                c_ids = self._ids(" " + c) if c else []
                rows.append((p_ids + c_ids, len(p_ids)))
            width = max(len(ids) for ids, _ in rows)
            input_ids = torch.full((len(rows), width), self.pad_id)
            attn = torch.zeros((len(rows), width), dtype=torch.long)
            targets = torch.full((len(rows), width), -100)  # -100 = ignored by cross_entropy
            for r, (ids, n_prompt) in enumerate(rows):
                input_ids[r, : len(ids)] = torch.tensor(ids)
                attn[r, : len(ids)] = 1
                targets[r, n_prompt : len(ids)] = torch.tensor(ids[n_prompt:])
            logits = self._logits(input_ids.to(self.device), attn.to(self.device)).float()
            # position t predicts token t+1
            targets = targets[:, 1:].to(self.device)
            loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), targets, reduction="none", ignore_index=-100)
            nlls += (loss.sum(1) / (targets != -100).sum(1).clamp(min=1)).tolist()
        return nlls


class QwenSteerer(Steerer):
    def __init__(self, name="Qwen/Qwen2.5-0.5B", device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"  # so every prompt ends at the same position when batched
        self.dtype = dtype
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device).eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)

    def encode(self, texts):
        return self.tok(texts, return_tensors="pt", padding=True).to(self.device)

    @property
    def pad_id(self):
        return self.tok.pad_token_id

    def _ids(self, text):
        return self.tok(text).input_ids

    def _logits(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def _forward_batch(self, texts):
        batch = self.encode(texts)
        mask = batch["attention_mask"].clone()
        first = mask.argmax(dim=1)  # with left padding, the first real token of each row
        mask[torch.arange(len(mask)), first] = 0
        return batch, mask

    @torch.no_grad()
    def generate(self, prompts, layer=None, vector=None, alpha=0.0, max_new_tokens=40,
                 temperature=0.7, top_p=0.9, seed=0, **generate_kwargs):
        """Continue each prompt, optionally steered. Returns only the new text for each prompt.
        Extra keyword arguments go to HF generate (the demo uses suppress_tokens)."""
        batch = self.encode(prompts)
        torch.manual_seed(seed)
        with self.steering(layer, vector, alpha):
            out = self.model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                top_p=top_p, pad_token_id=self.tok.pad_token_id, **generate_kwargs,
            )
        new = out[:, batch["input_ids"].shape[1]:]
        return [t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True)]


class TinyGPTSteerer(Steerer):
    """Our from-scratch GPT. It has no padding mask, so texts are right-padded for reading
    activations (with causal attention, real tokens never see the pads after them) and
    generation batches together prompts that have the same token length."""

    def __init__(self, ckpt_path, tokenizer_path, device=None):
        from tokenizers import Tokenizer

        from tinygpt.model import GPT, GPTConfig

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model = GPT(GPTConfig(**ckpt["config"]))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.step = ckpt.get("step")
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.eos_id = self.tok.token_to_id("<|endoftext|>")
        self.layers = self.model.blocks
        self.n_layers = len(self.layers)

    @property
    def pad_id(self):
        return self.eos_id

    def _ids(self, text):
        return self.tok.encode(text).ids

    def _logits(self, input_ids, attention_mask):
        return self.model(input_ids)[0]

    def _forward_batch(self, texts):
        seqs = [self._ids(t) for t in texts]
        width = max(len(s) for s in seqs)
        idx = torch.full((len(seqs), width), self.eos_id)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for r, ids in enumerate(seqs):
            idx[r, : len(ids)] = torch.tensor(ids)
            mask[r, 1 : len(ids)] = 1  # skip the first token, as for Qwen
        return {"idx": idx.to(self.device)}, mask.to(self.device)

    @torch.no_grad()
    def generate(self, prompts, layer=None, vector=None, alpha=0.0, max_new_tokens=40,
                 temperature=0.7, top_p=0.9, seed=0, end_marker=None, **generate_kwargs):
        """Same interface as QwenSteerer.generate (sampling with temperature + top-p).
        By default the text stops at the end-of-story token; with `end_marker` it keeps everything
        and shows each end-of-story token as that marker (the demo uses this)."""
        torch.manual_seed(seed)
        encoded = [self._ids(p) for p in prompts]
        groups = {}
        for i, ids in enumerate(encoded):
            groups.setdefault(len(ids), []).append(i)
        results = [None] * len(prompts)
        block = self.model.cfg.block_size
        with self.steering(layer, vector, alpha):
            for length in sorted(groups):
                rows = groups[length]
                idx = torch.tensor([encoded[i] for i in rows], device=self.device)
                for _ in range(max_new_tokens):
                    logits = self.model(idx[:, -block:])[0][:, -1, :] / temperature
                    probs = logits.softmax(-1)
                    # top-p: keep the smallest set of tokens whose probability adds up to top_p
                    sorted_p, order = probs.sort(dim=-1, descending=True)
                    drop = sorted_p.cumsum(-1) - sorted_p > top_p
                    sorted_p[drop] = 0
                    choice = torch.multinomial(sorted_p, 1)
                    idx = torch.cat([idx, order.gather(-1, choice)], dim=1)
                for r, i in enumerate(rows):
                    new = idx[r, length:].tolist()
                    if end_marker is not None:  # decode each story separately, joined by the marker
                        parts, cur = [], []
                        for t in new + [self.eos_id]:
                            if t == self.eos_id:
                                parts.append(self.tok.decode(cur).strip())
                                cur = []
                            else:
                                cur.append(t)
                        results[i] = end_marker.join(parts).strip()
                        continue
                    if self.eos_id in new:  # stop at the end of the story
                        new = new[: new.index(self.eos_id)]
                    results[i] = self.tok.decode(new).strip()
        return results


def random_like(vector, seed=0):
    """A random direction with the same norm as `vector` (the control for every experiment)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(vector.shape, generator=g)
    return r * (vector.norm() / r.norm())
