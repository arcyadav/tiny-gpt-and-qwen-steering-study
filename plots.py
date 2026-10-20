"""Make the result figures from results/*/e*.json  ->  results/plots/*.png

    python plots.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/plots"
HAPPY, SAD, GREY = "#e8a33d", "#3b6fb6", "#9a9a9a"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def pick(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def layer_sweep(e1, title, name, alphas):
    """Sentiment vs layer for concept and random vectors, and probe AUC per layer."""
    rows, base = e1["rows"], e1["baseline"]
    layers = sorted({r["layer"] for r in rows})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    for a, ls in zip(alphas, ["-", "--"]):
        for sign, col in ((1, HAPPY), (-1, SAD)):
            y = [pick(rows, kind="concept", layer=l, alpha=sign * a)[0]["sentiment"] for l in layers]
            ax.plot(layers, y, ls, marker="o", ms=4, color=col, label=f"steer α={sign * a:+g}")
    a = max(alphas)
    for sign in (1, -1):
        y = [pick(rows, kind="random", layer=l, alpha=sign * a)[0]["sentiment"] for l in layers]
        ax.plot(layers, y, ":", marker="x", ms=4, color=GREY, label=f"random α=±{a:g}" if sign > 0 else None)
    ax.axhline(base["sentiment"], color="k", lw=1, label="no steering")
    ax.axvline(e1["best_layer"], color="k", lw=0.8, alpha=0.3)
    ax.set(xlabel="layer steered", ylabel="sentiment  P(pos) − P(neg)", ylim=(-1, 1),
           title=f"{title}: steering effect by layer")
    ax.legend(fontsize=8, ncol=2, loc="lower right")

    auc = e1["probe_auc"]
    bx.plot(range(len(auc)), auc, marker="o", ms=4, color="#5a9e6f")
    bx.axvline(e1["best_layer"], color="k", lw=0.8, alpha=0.3, label=f"best steering layer ({e1['best_layer']})")
    bx.set(xlabel="layer", ylabel="probe AUC (happy vs sad sentences)", ylim=(0.3, 1.02),
           title="where the concept is readable")
    bx.axhline(0.5, color=GREY, lw=0.8, ls=":")
    bx.legend(fontsize=8, loc="lower right")
    save(fig, name)


def strength_sweep(models):
    """Sentiment and fluency vs steering strength at the best layer (E2 + the α=0 baseline)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4))
    for label, e2, col in models:
        base = e2["baseline"]
        for kind, ls, alpha in (("concept", "-", 1.0), ("random", ":", 0.45)):
            rs = sorted(pick(e2["rows"], kind=kind) + [dict(base, alpha=0.0)], key=lambda r: r["alpha"])
            x = [r["alpha"] for r in rs]
            ax.plot(x, [r["sentiment"] for r in rs], ls, marker="o", ms=4, color=col, alpha=alpha,
                    label=f"{label} {'steer' if kind == 'concept' else 'random'} (layer {e2['layer']})")
            bx.plot(x, [r["perplexity"] / base["perplexity"] for r in rs], ls, marker="o", ms=4, color=col, alpha=alpha,
                    label=f"{label} {'steer' if kind == 'concept' else 'random'}")
    ax.set(xlabel="strength α  (− = sad, + = happy)", ylabel="sentiment", ylim=(-1, 1),
           title="how hard to push: sentiment")
    bx.set(xlabel="strength α", ylabel="perplexity ÷ unsteered perplexity", yscale="log",
           title="…and what it costs: fluency")
    bx.axhline(3, color="r", lw=0.8, ls="--", label="3× (usable limit)")
    ax.legend(fontsize=8)
    bx.legend(fontsize=8)
    save(fig, "e2_strength.png")


def prompting(e3):
    rows = e3["rows"]
    none = pick(rows, kind="none")[0]
    kinds = ["prompt", "steer", "prompt+steer"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.8))
    w = 0.38
    for i, (mood, col) in enumerate((("happy", HAPPY), ("sad", SAD))):
        rs = [none] + [pick(rows, kind=k, mood=mood)[0] for k in kinds]
        xs = [j + (i - 0.5) * w for j in range(4)]
        ax.bar(xs, [r["sentiment"] for r in rs], w, color=col, label=f"towards {mood}")
        bx.bar(xs, [r["perplexity"] for r in rs], w, color=col, label=f"towards {mood}")
    for z in (ax, bx):
        z.set_xticks(range(4), ["none", "prompt\n(\"write a happy…\")", f"steer\n(α=±{e3['alpha']:g})", "prompt\n+ steer"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set(ylabel="sentiment", ylim=(-1, 1), title=f"Qwen: steering vs prompting (layer {e3['layer']})")
    bx.set(ylabel="perplexity (lower = more fluent)", title="fluency")
    ax.legend(fontsize=8)
    save(fig, "e3_prompting.png")


def n_pairs(e4):
    rows = e4["rows"]
    sizes = sorted({r["n_pairs"] for r in rows})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.8))
    for sign, col, lab in ((1, HAPPY, "happy (α>0)"), (-1, SAD, "sad (α<0)")):
        ys = [[r["sentiment"] for r in pick(rows, n_pairs=n) if r["alpha"] * sign > 0] for n in sizes]
        mean = [sum(y) / len(y) for y in ys]
        ax.errorbar(sizes, mean, yerr=[[m - min(y) for m, y in zip(mean, ys)], [max(y) - m for m, y in zip(mean, ys)]],
                    marker="o", ms=4, capsize=3, color=col, label=lab)
    ax.axhline(e4["baseline"]["sentiment"], color="k", lw=1, label="no steering")
    ax.set(xscale="log", xlabel="contrastive pairs used", ylabel="sentiment", ylim=(-1, 1),
           title=f"Qwen: how many pairs are needed (layer {e4['layer']})")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.legend(fontsize=8)
    cos = [[r["cosine_to_full"] for r in pick(rows, n_pairs=n)] for n in sizes]
    bx.plot(sizes, [sum(c) / len(c) for c in cos], marker="o", ms=4, color="#5a9e6f")
    bx.set(xscale="log", xlabel="contrastive pairs used", ylabel="cosine to the 150-pair vector",
           ylim=(0.7, 1.01), title="vector stability")
    bx.set_xticks(sizes, [str(s) for s in sizes])
    save(fig, "e4_pairs.png")


def training(e6_runs, logs):
    """E6: probe AUC and steering effect over TinyGPT training checkpoints, plus the loss curve."""
    by_step = {}
    for e6 in e6_runs:  # later runs (dense early checkpoints) fill in / override
        for r in e6["rows"]:
            by_step[r["step"]] = r
    steps = sorted(by_step)
    layer, alpha = e6_runs[0]["layer"], e6_runs[0]["alpha"]
    x = [max(s, 1) for s in steps]  # log x-axis: show step 0 at 1
    rs = [by_step[s] for s in steps]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.9))

    ax = axs[0]
    ax.plot(x, [r["probe_auc"][layer] for r in rs], marker="o", ms=4, color="#5a9e6f", label=f"layer {layer}")
    ax.plot(x, [max(r["probe_auc"]) for r in rs], ":", color="#5a9e6f", label="best layer")
    ax.axhline(0.5, color=GREY, lw=0.8, ls=":")
    ax.set(xscale="log", xlabel="training step", ylabel="probe AUC", ylim=(0.3, 1.02),
           title="is happy/sad readable?")
    ax.legend(fontsize=8)

    ax = axs[1]
    for a, col in ((alpha, HAPPY), (-alpha, SAD)):
        key = f"alpha{a:+g}"
        ax.plot(x, [r[key][0]["sentiment"] - r["baseline"]["sentiment"] for r in rs], marker="o", ms=4,
                color=col, label=f"steer α={a:+g}")
        ax.plot(x, [r[key][1]["sentiment"] - r["baseline"]["sentiment"] for r in rs], ":", marker="x", ms=4,
                color=col, alpha=0.5, label=f"random α={a:+g}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xscale="log", xlabel="training step", ylabel="sentiment change vs no steering",
           title=f"does steering work? (layer {layer})")
    ax.legend(fontsize=8)

    ax = axs[2]
    for label, log, col in logs:
        v = log["val"]
        ax.plot([max(p["step"], 1) for p in v], [p["loss"] for p in v], marker="o", ms=3, color=col, label=label)
    ax.set(xscale="log", xlabel="training step", ylabel="validation loss", title="TinyGPT training")
    ax.legend(fontsize=8)
    save(fig, "e6_training.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    q = {e: load(f"results/qwen/{e}.json") for e in ("e1", "e2", "e3", "e4")}
    t = {e: load(f"results/tinygpt_steer/{e}.json") for e in ("e1", "e2", "e6")}
    if q["e1"]:
        layer_sweep(q["e1"], "Qwen2.5-0.5B", "e1_qwen_layers.png", [4, 8])
    if t["e1"]:
        layer_sweep(t["e1"], "TinyGPT 12M", "e1_tinygpt_layers.png", [4, 8])
    models = [m for m in (("Qwen", q["e2"], "#c0504d"), ("TinyGPT", t["e2"], "#7b5ea7")) if m[1]]
    if models:
        strength_sweep(models)
    if q["e3"]:
        prompting(q["e3"])
    if q["e4"]:
        n_pairs(q["e4"])
    if t["e6"]:
        early = load("results/tinygpt_early_steer/e6.json")
        logs = [("full run", load("results/tinygpt/log.json"), "#7b5ea7")]
        if load("results/tinygpt_early/log.json"):
            logs.append(("first 600 steps", load("results/tinygpt_early/log.json"), HAPPY))
        training([t["e6"]] + ([early] if early else []), [l for l in logs if l[1]])


if __name__ == "__main__":
    main()
