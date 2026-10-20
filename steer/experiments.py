"""Steering experiments (see spec.md section 5).

Qwen2.5-0.5B, E1-E4:
    python -m steer.experiments --exp e1 e2 e3 e4 --out results/qwen      # Kaggle T4
    python -m steer.experiments --exp e1 --out results/quick --quick      # CPU check
TinyGPT, E5 (= layer + strength sweep, i.e. e1 + e2 on TinyGPT) and E6 (over training checkpoints):
    python -m steer.experiments --model tinygpt --run-dir results/tinygpt --tokenizer data/tokenizer.json         --exp e1 e2 e6 --out results/tinygpt_steer

Every steered setting is paired with a random vector of the same norm (the control).
Each experiment writes <out>/<exp>.json (summaries) and <out>/<exp>_texts.jsonl (every generation).
"""
import argparse
import json
import os
import time

import torch

from steer.core import QwenSteerer, TinyGPTSteerer, random_like
from steer import pairs as P
from steer.score import SentimentScorer, score

# prompt prefixes for E3 (Qwen2.5-0.5B is a base model, so "asking" = setting up the context)
PROMPT_PREFIX = {
    +1: "The following is a happy, cheerful story full of joy.\n\n",
    -1: "The following is a sad, gloomy story full of sorrow.\n\n",
}


class Runner:
    def __init__(self, args):
        self.args = args
        self.sentiment = SentimentScorer()
        if args.model == "qwen":
            steerer = QwenSteerer()
            self.prompts, self.pairs = P.NEUTRAL_PROMPTS, P.sentiment_pairs("train")
            self.probe_pos, self.probe_neg = P.PROBE_POSITIVE, P.PROBE_NEGATIVE
        else:
            steerer = TinyGPTSteerer(os.path.join(args.run_dir, "ckpt.pt"), args.tokenizer)
            self.prompts, self.pairs = P.TINY_PROMPTS, P.tiny_sentiment_pairs()
            self.probe_pos, self.probe_neg = P.TINY_PROBE_POSITIVE, P.TINY_PROBE_NEGATIVE
        self.prompts = self.prompts[: args.n_prompts]
        n = steerer.n_layers
        if args.layers is None:
            args.layers = list(range(n))
        elif args.layers == "first_last":
            args.layers = [0, n - 1]
        os.makedirs(args.out, exist_ok=True)
        self.use(steerer)
        torch.save(self.vectors, os.path.join(args.out, "sentiment_vectors.pt"))

    def use(self, steerer):
        """Switch to a (new) model: recompute its steering vectors and forget the old baseline."""
        t = time.time()
        self.steerer = steerer
        self.vectors = steerer.concept_vectors(self.pairs)
        self.baseline = None
        print(f"vectors from {len(self.pairs)} pairs in {time.time() - t:.0f}s")

    def run(self, layer=None, vector=None, alpha=0.0, prefix="", **info):
        """Generate `samples` continuations per prompt for one setting and score them.
        Fluency is always measured given the plain neutral prompt (without any prefix)."""
        a = self.args
        texts, prompts = [], []
        for seed in range(a.samples):
            texts += self.steerer.generate([prefix + p for p in self.prompts], layer, vector, alpha,
                                           max_new_tokens=a.max_new_tokens, seed=seed)
            prompts += self.prompts
        summary, per_text = score(self.steerer, self.sentiment, prompts, texts)
        row = {**info, "layer": layer, "alpha": alpha, **summary}
        print("  " + " ".join(f"{k}={v}" for k, v in info.items()),
              f"L={layer} a={alpha:+g} | sent {summary['sentiment']:+.2f} "
              f"ppl {summary['perplexity']:.1f} d2 {summary['distinct_2']:.2f}")
        return row, [{**info, "layer": layer, "alpha": alpha, **t} for t in per_text]

    def get_baseline(self):
        if self.baseline is None:
            self.baseline = self.run(kind="none")
        row, texts = self.baseline
        return row, list(texts)  # a copy: callers append their own texts

    def save(self, name, result, texts):
        with open(os.path.join(self.args.out, f"{name}.json"), "w") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(self.args.out, f"{name}_texts.jsonl"), "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    def steer_and_control(self, layer, alpha, vector, **info):
        """The concept vector and a same-norm random vector at the same layer and strength."""
        r1, t1 = self.run(layer, vector, alpha, kind="concept", **info)
        r2, t2 = self.run(layer, random_like(vector, seed=layer), alpha, kind="random", **info)
        return [r1, r2], t1 + t2

    # ------------------------------------------------------------ E1: layer sweep + probe

    def probe_auc(self):
        """Per layer: how well the steering direction separates naturally written positive and
        negative sentences (none of the pair words). AUC = chance that a random positive sentence
        projects higher than a random negative one; 0.5 = chance, 1.0 = perfect."""
        direction = self.vectors / self.vectors.norm(dim=-1, keepdim=True)
        pos = (self.steerer.text_means(self.probe_pos) * direction).sum(-1)  # (n, n_layers)
        neg = (self.steerer.text_means(self.probe_neg) * direction).sum(-1)
        wins = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean(dim=(0, 1))
        return wins.tolist()

    def e1(self):
        base_row, texts = self.get_baseline()
        rows = []
        for layer in self.args.layers:
            for alpha in self.args.e1_alphas:
                r, t = self.steer_and_control(layer, alpha, self.vectors[layer])
                rows += r
                texts += t
        result = {"baseline": base_row, "rows": rows, "probe_auc": self.probe_auc(),
                  "vector_norms": self.vectors.norm(dim=-1).tolist(),
                  }
        result["best_layer"], result["best_alpha"] = best_setting(rows, base_row)
        self.save("e1", result, texts)
        print(f"E1 best: layer {result['best_layer']}, strength {result['best_alpha']}")
        return result

    # ------------------------------------------------------------ E2: strength sweep

    def e2(self, layer):
        base_row, texts = self.get_baseline()
        rows = []
        for alpha in self.args.e2_alphas:
            r, t = self.steer_and_control(layer, alpha, self.vectors[layer])
            rows += r
            texts += t
        self.save("e2", {"baseline": base_row, "layer": layer, "rows": rows}, texts)

    # ------------------------------------------------------------ E3: steering vs prompting

    def e3(self, layer):
        base_row, texts = self.get_baseline()
        rows = [base_row]
        alpha = self.args.e3_alpha
        for sign, mood in ((+1, "happy"), (-1, "sad")):
            v = self.vectors[layer]
            for method, kw in (
                ("prompt", dict(prefix=PROMPT_PREFIX[sign])),
                ("steer", dict(layer=layer, vector=v, alpha=sign * alpha)),
                ("prompt+steer", dict(prefix=PROMPT_PREFIX[sign], layer=layer, vector=v, alpha=sign * alpha)),
            ):
                r, t = self.run(**kw, kind=method, mood=mood)
                rows.append(r)
                texts += t
        self.save("e3", {"layer": layer, "alpha": alpha, "rows": rows}, texts)

    # ------------------------------------------------------------ E4: how many pairs

    def e4(self, layer):
        base_row, texts = self.get_baseline()
        rows = []
        full = self.vectors[layer]
        for n in self.args.e4_sizes:
            for subset in range(1 if n >= len(self.pairs) else 3):
                v = self.steerer.concept_vectors(P.sample_pairs(self.pairs, n, seed=subset))[layer]
                cos = torch.nn.functional.cosine_similarity(v, full, dim=0).item()
                for alpha in (self.args.e3_alpha, -self.args.e3_alpha):
                    r, t = self.run(layer, v, alpha, kind="concept", n_pairs=n, subset=subset)
                    r["cosine_to_full"] = cos
                    rows.append(r)
                    texts += t
        self.save("e4", {"baseline": base_row, "layer": layer, "rows": rows}, texts)

    # ------------------------------------------------------------ E6: during training (TinyGPT)

    def e6(self, layer):
        """For every saved training checkpoint: rebuild the vectors from that checkpoint, measure
        probe AUC per layer and the steering effect at `layer` (vs that checkpoint's own baseline)."""
        ckpt_dir = os.path.join(self.args.run_dir, "checkpoints")
        rows, texts = [], []
        alpha = self.args.e3_alpha
        for name in sorted(os.listdir(ckpt_dir)):
            steerer = TinyGPTSteerer(os.path.join(ckpt_dir, name), self.args.tokenizer)
            self.use(steerer)
            print(f"  checkpoint {name} (step {steerer.step})")
            base_row, t = self.get_baseline()
            texts += t
            row = {"step": steerer.step, "probe_auc": self.probe_auc(), "baseline": base_row}
            for a in (alpha, -alpha):
                r, t = self.steer_and_control(layer, a, self.vectors[layer], step=steerer.step)
                row[f"alpha{a:+g}"] = r
                texts += t
            rows.append(row)
        self.save("e6", {"layer": layer, "alpha": alpha, "rows": rows}, texts)


def best_setting(rows, base_row, max_ppl_ratio=3.0):
    """The (layer, strength) where concept steering moves sentiment furthest in both directions,
    i.e. the largest sent(+a) - sent(-a), among settings whose perplexity stays below
    3x the unsteered one. Returns (None, None) if no setting qualifies."""
    concept = {(r["layer"], r["alpha"]): r for r in rows if r["kind"] == "concept"}
    limit = max_ppl_ratio * base_row["perplexity"]
    best, best_score = (None, None), float("-inf")
    for (layer, a), up in concept.items():
        down = concept.get((layer, -a))
        if a <= 0 or down is None or max(up["perplexity"], down["perplexity"]) > limit:
            continue
        if up["sentiment"] - down["sentiment"] > best_score:
            best, best_score = (layer, a), up["sentiment"] - down["sentiment"]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["e1", "e2", "e3", "e4"])
    ap.add_argument("--out", default="results/qwen")
    ap.add_argument("--layer", type=int, default=None, help="layer for E2-E6 (default: best from E1)")
    ap.add_argument("--alpha", type=float, default=None, help="strength for E2-E6 (default: best from E1)")
    ap.add_argument("--quick", action="store_true", help="tiny version for a CPU check")
    ap.add_argument("--model", choices=["qwen", "tinygpt"], default="qwen")
    ap.add_argument("--run-dir", default="results/tinygpt", help="TinyGPT: folder with ckpt.pt + checkpoints/")
    ap.add_argument("--tokenizer", default="data/tokenizer.json", help="TinyGPT tokenizer")
    args = ap.parse_args()

    args.n_prompts, args.samples, args.max_new_tokens = 50, 2, 40
    args.e4_sizes = [5, 10, 25, 50, 150]
    if args.model == "qwen":
        args.layers, strengths = list(range(0, 24, 2)), [4, 8]
    else:  # TinyGPT's scale is unknown in advance, so E1 tries a wide range
        args.layers, strengths = None, [2, 4, 8, 16, 32]  # None = all of its layers
    if args.quick:
        args.n_prompts, args.samples, args.max_new_tokens = 6, 1, 20
        args.layers = [6, 14] if args.model == "qwen" else "first_last"
        strengths, args.e4_sizes = [8], [5, 150]
    args.e1_alphas = sorted([-a for a in strengths] + strengths)

    runner = Runner(args)
    t0 = time.time()
    layer, alpha = args.layer, args.alpha
    if "e1" in args.exp:
        print("== E1 layer sweep")
        result = runner.e1()
        layer = layer if layer is not None else result["best_layer"]
        alpha = alpha if alpha is not None else result["best_alpha"]
    path = os.path.join(args.out, "e1.json")
    if (layer is None or alpha is None) and os.path.exists(path):
        saved = json.load(open(path))
        layer = layer if layer is not None else saved["best_layer"]
        alpha = alpha if alpha is not None else saved["best_alpha"]
    if layer is None or alpha is None:
        raise SystemExit("no layer/strength: run e1 first, or pass --layer and --alpha")
    # E2 sweeps around the chosen strength; E3, E4 and E6 use it directly
    args.e2_alphas = [sign * alpha * f for sign in (-1, 1) for f in (0.5, 1, 1.5, 2)]
    args.e3_alpha = alpha
    for name in ("e2", "e3", "e4", "e6"):
        if name in args.exp:
            print(f"== {name.upper()} at layer {layer}, strength {alpha:g}")
            getattr(runner, name)(layer)
    print(f"all done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
