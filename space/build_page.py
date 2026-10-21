"""Build the static project page (free Hugging Face static Space) from the experiment results.

    python space/build_page.py      # -> space/site/  (index.html, replay.json, plots/)
    python space/deploy.py          # upload space/site to the Space

replay.json holds real generations from the experiments (first sample of every prompt), so the page
can replay steering without running a model. The live demo (python app.py --share) runs on a laptop
and announces its link in live.json.
"""
import json
import os
import shutil

SITE = "space/site"


def read_texts(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def short(x):
    return round(x, 2)


def settings_table(rows, prompts, key):
    """{setting key: [[text, sentiment] per prompt]} keeping the first sample of each prompt."""
    out = {}
    for r in rows:
        k = key(r)
        if k is None:
            continue
        slot = out.setdefault(k, [None] * len(prompts))
        i = prompts.index(r["prompt"])
        if slot[i] is None:
            slot[i] = [r["text"], short(r["sentiment"])]
    return out


def sweep(model_dir):
    """Layer sweep (E1) + strength sweep at the best layer (E2) for one model."""
    rows = read_texts(f"{model_dir}/e1_texts.jsonl") + read_texts(f"{model_dir}/e2_texts.jsonl")
    prompts = list(dict.fromkeys(r["prompt"] for r in rows))
    e1 = json.load(open(f"{model_dir}/e1.json"))
    table = settings_table(rows, prompts, lambda r: "none" if r["kind"] == "none"
                           else f"{r['kind']}|{r['layer']}|{r['alpha']:g}")
    layers = sorted({int(k.split("|")[1]) for k in table if k != "none"})
    alphas = {}
    for k in table:
        if k.startswith("concept|"):
            _, l, a = k.split("|")
            alphas.setdefault(l, set()).add(float(a))
    return {
        "prompts": prompts,
        "layers": layers,
        "alphas": {l: sorted(v) for l, v in alphas.items()},
        "best": [e1["best_layer"], e1["best_alpha"]],
        "probe_auc": [short(x) for x in e1["probe_auc"]],
        "texts": table,
    }


def prompting():
    """E3 (Qwen): no steering / prompt / steer / prompt + steer, towards happy and towards sad."""
    rows = read_texts("results/qwen/e3_texts.jsonl")
    prompts = list(dict.fromkeys(r["prompt"] for r in rows))
    table = settings_table(rows, prompts, lambda r: "none" if r["kind"] == "none" else f"{r['kind']}|{r['mood']}")
    e3 = json.load(open("results/qwen/e3.json"))
    return {"prompts": prompts, "layer": e3["layer"], "alpha": e3["alpha"], "texts": table}


def training():
    """E6 (TinyGPT): the same prompts steered at every training checkpoint. Baseline rows carry no
    step, but each checkpoint writes its baseline right before its steered rows."""
    runs = ["results/tinygpt_steer"]
    if os.path.exists("results/tinygpt_early_steer/e6_texts.jsonl"):
        runs.append("results/tinygpt_early_steer")  # dense checkpoints over the first 600 steps
    texts, metrics, prompts = {}, {}, None
    for run in runs:
        rows = read_texts(f"{run}/e6_texts.jsonl")
        prompts = prompts or list(dict.fromkeys(r["prompt"] for r in rows))
        pending = []
        for r in rows:
            if "step" not in r:
                pending.append(r)
                continue
            for p in pending:
                p["step"] = r["step"]
            pending = []
        table = settings_table(rows, prompts, lambda r: f"{r['step']}|none" if r["kind"] == "none"
                               else f"{r['step']}|{r['kind']}|{r['alpha']:g}")
        texts.update(table)
        e6 = json.load(open(f"{run}/e6.json"))
        for row in e6["rows"]:
            m = {"auc": short(row["probe_auc"][e6["layer"]]), "base": short(row["baseline"]["sentiment"])}
            for k in ("alpha+8", "alpha-8"):
                m[k[5:]] = short(row[k][0]["sentiment"])
            metrics[row["step"]] = m
    logs = {}
    for name in ("tinygpt", "tinygpt_early"):
        path = f"results/{name}/log.json"
        if os.path.exists(path):
            for v in json.load(open(path))["val"]:
                logs[v["step"]] = round(v["loss"], 3)
    steps = sorted(metrics)
    for s in steps:
        near = min(logs, key=lambda x: abs(x - s)) if logs else None
        metrics[s]["loss"] = logs.get(near)
    return {"prompts": prompts, "steps": steps, "metrics": {str(s): metrics[s] for s in steps}, "texts": texts}


def main():
    os.makedirs(f"{SITE}/plots", exist_ok=True)
    data = {
        "qwen": sweep("results/qwen"),
        "tiny": sweep("results/tinygpt_steer"),
        "prompting": prompting(),
        "training": training(),
    }
    with open(f"{SITE}/replay.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    for name in os.listdir("results/plots"):
        shutil.copy(f"results/plots/{name}", f"{SITE}/plots/{name}")
    shutil.copy("space/index.html", f"{SITE}/index.html")
    shutil.copy("space/concepts.json", f"{SITE}/concepts.json")  # from space/concept_gallery.py
    shutil.copy("space/README.md", f"{SITE}/README.md")
    size = os.path.getsize(f"{SITE}/replay.json") / 1e6
    print(f"wrote {SITE}: replay.json {size:.1f} MB, steps {data['training']['steps']}")


if __name__ == "__main__":
    main()
