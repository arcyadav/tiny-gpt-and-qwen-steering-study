"""Activation steering: build concept directions from contrastive pairs and add them to the
residual stream while the model generates. The model weights are never changed.

"Layer l" always means the residual stream right after transformer block l (0-indexed).
"""
from contextlib import contextmanager

import torch


def _hidden(output):
    """A block's output is either the hidden-state tensor or a tuple starting with it."""
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, new):
    return (new, *output[1:]) if isinstance(output, tuple) else new


class QwenSteerer:
    def __init__(self, name="Qwen/Qwen2.5-0.5B", device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"  # so every prompt ends at the same position when batched
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device).eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)

    def encode(self, texts):
        return self.tok(texts, return_tensors="pt", padding=True).to(self.device)

    @torch.no_grad()
    def _token_sums(self, texts, batch_size=32):
        """Per text: sum of residual-stream vectors after every block over its tokens, and the
        token count. Pad tokens and each text's first token (unusually large activations) are
        skipped. Returns sums (n_texts, n_layers, d_model) float32 and counts (n_texts,)."""
        all_sums, all_counts = [], []
        for i in range(0, len(texts), batch_size):
            batch = self.encode(texts[i : i + batch_size])
            mask = batch["attention_mask"].clone()
            first = mask.argmax(dim=1)  # with left padding, the first real token of each row
            mask[torch.arange(len(mask)), first] = 0
            captured = []
            hooks = [layer.register_forward_hook(lambda m, inp, out: captured.append(_hidden(out)))
                     for layer in self.layers]
            try:
                self.model(**batch)
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
        add = (alpha * vector).to(self.device, self.model.dtype)
        hook = self.layers[layer].register_forward_hook(
            lambda m, inp, out: _replace_hidden(out, _hidden(out) + add))
        try:
            yield
        finally:
            hook.remove()

    @torch.no_grad()
    def generate(self, prompts, layer=None, vector=None, alpha=0.0, max_new_tokens=40,
                 temperature=0.7, top_p=0.9, seed=0):
        """Continue each prompt, optionally steered. Returns only the new text for each prompt."""
        batch = self.encode(prompts)
        torch.manual_seed(seed)
        with self.steering(layer, vector, alpha):
            out = self.model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                top_p=top_p, pad_token_id=self.tok.pad_token_id,
            )
        new = out[:, batch["input_ids"].shape[1]:]
        return [t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True)]


def random_like(vector, seed=0):
    """A random direction with the same norm as `vector` (the control for every experiment)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(vector.shape, generator=g)
    return r * (vector.norm() / r.norm())
