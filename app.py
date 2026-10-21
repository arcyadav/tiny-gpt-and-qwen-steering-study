"""Gradio demo: steer Qwen2.5-0.5B or our TinyGPT with a slider, see the concept inside the model,
and browse the experiment results.

    python app.py            # this computer + other devices on the same network (intranet)
    python app.py --share    # also a public gradio.live link, announced on the project's static page
"""
import argparse
import datetime
import html
import json
import math
import os
import re
import socket
import threading

import gradio as gr
import torch

from steer import pairs as P
from steer.core import QwenSteerer, TinyGPTSteerer, random_like
from steer.score import SentimentScorer

QWEN, TINY = "Qwen2.5-0.5B (pretrained)", "TinyGPT 12M (built from scratch)"
MODELS = {
    QWEN: dict(load=lambda: QwenSteerer(), pairs=P.sentiment_pairs("train"), e1="results/qwen/e1.json"),
    TINY: dict(load=lambda: TinyGPTSteerer("results/tinygpt/ckpt.pt", "data/tokenizer.json"),
               pairs=P.tiny_sentiment_pairs(), e1="results/tinygpt_steer/e1.json"),
}
BUILTIN = "happy ↔ sad (built-in pairs)"
CUSTOM = "my own concept (type examples below)"
CUSTOM_POS = "\n".join([
    "I would be grateful if you could kindly review the attached document.",
    "We regret to inform you that the meeting has been postponed.",
    "Please find enclosed the requested information.",
    "The committee will convene at your earliest convenience.",
])
CUSTOM_NEG = "\n".join([
    "hey can u check this thing i sent lol",
    "yo the meeting's off, my bad",
    "here's the stuff u wanted",
    "let's hang out whenever, no rush dude",
])
HAPPY, SAD = (232, 163, 61), (59, 111, 182)

_lock = threading.Lock()
_models, _concepts, _scorer = {}, {}, []


def get_model(name):
    with _lock:
        if name not in _models:
            _models[name] = MODELS[name]["load"]()
        return _models[name]


# Qwen2.5 base often drifts into Chinese exam questions ("I ____ the door. A. … 答案: A"), with or
# without steering. The demo (not the experiments) stops it from producing CJK text and "___" blanks.
_UNWANTED = re.compile(r"[　-〿㐀-鿿＀-￯]|_{2,}")
_suppress = {}


def gen_kwargs(name):
    if name != QWEN:
        return {}
    if name not in _suppress:
        tok = get_model(name).tok
        _suppress[name] = [i for i in range(len(tok)) if _UNWANTED.search(tok.decode([i]))]
    return {"suppress_tokens": _suppress[name]}


def scorer():
    with _lock:
        if not _scorer:
            _scorer.append(SentimentScorer())
        return _scorer[0]


def best_layer(name):
    with open(MODELS[name]["e1"]) as f:
        e1 = json.load(f)
    return e1["best_layer"], e1["best_alpha"]


class Concept:
    """Mean residual stream of the positive and of the negative examples, at every layer.
    The steering vector is their difference; the midpoint is the 'neutral' reference for the meter."""

    def __init__(self, steerer, pos_texts, neg_texts):
        pos, neg = steerer.layer_means(pos_texts), steerer.layer_means(neg_texts)
        self.vector, self.mid = pos - neg, (pos + neg) / 2


def get_concept(name, choice, pos_text, neg_text):
    if choice == BUILTIN:
        pairs = MODELS[name]["pairs"]
        pos, neg = [p for p, _ in pairs], [n for _, n in pairs]
    else:
        pos = [l.strip() for l in pos_text.splitlines() if l.strip()]
        neg = [l.strip() for l in neg_text.splitlines() if l.strip()]
        if len(pos) < 2 or len(neg) < 2:
            raise gr.Error("Give at least 2 examples on each side (one per line).")
    key = (name, tuple(pos), tuple(neg))
    if key not in _concepts:
        _concepts[key] = Concept(get_model(name), pos, neg)
    return _concepts[key]


# ------------------------------------------------------------------------------------ Steer tab

def steer(name, prompt, choice, pos_text, neg_text, layer, alpha, n_tokens, seed, show_random):
    if not prompt.strip():
        raise gr.Error("Type a prompt first.")
    model = get_model(name)
    layer = min(int(layer), model.n_layers - 1)
    v = get_concept(name, choice, pos_text, neg_text).vector[layer]
    if choice != BUILTIN:  # same length as the happy/sad vector, so α means the same for any concept
        v = v * (get_concept(name, BUILTIN, "", "").vector[layer].norm() / v.norm())
    # one batch: row 0 unsteered, row 1 steered, row 2 a random vector of the same norm (control).
    # The hook adds alpha * vectors[row] to each row, so the rows differ only in what is added.
    rows = [torch.zeros_like(v), v] + ([random_like(v, seed=layer)] if show_random else [])
    vectors = torch.stack(rows).unsqueeze(1)  # (rows, 1, d_model) broadcasts over positions
    texts = model.generate([prompt] * len(rows), layer, vectors, float(alpha),
                           max_new_tokens=int(n_tokens), seed=int(seed), **gen_kwargs(name))
    scores = scorer()(texts)
    out = [f"{t}\n\n— sentiment {s:+.2f}" for t, s in zip(texts, scores)]
    return out[0], out[1], out[2] if show_random else ""


def on_model_change(name):
    layer, alpha = best_layer(name)
    n = 24 if name == QWEN else 6
    return gr.update(maximum=n - 1, value=layer, label=f"layer to steer (0–{n - 1}, best = {layer})"), \
        gr.update(value=alpha)


# ------------------------------------------------------------------------------ Concept meter tab

@torch.no_grad()
def token_scores(model, concept, text, layer):
    """Projection of every token's residual stream (after block `layer`) on the concept direction,
    relative to the midpoint, in units of half the pos-neg distance: about +1 = like the positive
    examples, -1 = like the negative ones."""
    inputs, mask = model._forward_batch([text])
    ids = (inputs.get("input_ids") if "input_ids" in inputs else inputs["idx"])[0].tolist()
    captured = []
    hook = model.layers[layer].register_forward_hook(
        lambda m, i, o: captured.append(o[0] if isinstance(o, tuple) else o))
    try:
        model.model(**inputs)
    finally:
        hook.remove()
    h = captured[0][0].float().cpu()  # (tokens, d_model)
    v, mid = concept.vector[layer], concept.mid[layer]
    s = ((h - mid) @ v) / (v.norm() ** 2 / 2)
    s[~mask[0].bool().cpu()] = float("nan")  # first token (huge activations) and pads: no score
    tokens = [model.tok.decode([i]) for i in ids]
    return tokens, s.tolist()


def meter_html(tokens, scores):
    spans = []
    for tok, s in zip(tokens, scores):
        if s != s:  # nan
            style, tip = "", "not scored"
        else:
            r, g, b = HAPPY if s > 0 else SAD
            style = f"background: rgba({r},{g},{b},{math.tanh(abs(s) / 3) * 0.9:.2f}); border-radius: 3px;"
            tip = f"{s:+.2f}"
        spans.append(f'<span title="{tip}" style="{style}">{html.escape(tok)}</span>')
    return ('<div style="font-size: 1.15em; line-height: 2; white-space: pre-wrap;">'
            + "".join(spans) + "</div>")


def meter(name, text, choice, pos_text, neg_text, layer):
    if not text.strip():
        raise gr.Error("Type some text first.")
    model = get_model(name)
    layer = min(int(layer), model.n_layers - 1)
    concept = get_concept(name, choice, pos_text, neg_text)
    tokens, scores = token_scores(model, concept, text, layer)
    valid = [s for s in scores if s == s]
    mean = sum(valid) / max(1, len(valid))
    side = "positive side (e.g. happy)" if mean > 0 else "negative side (e.g. sad)"
    summary = f"**Average: {mean:+.2f}** → the model reads this text on the **{side}** of the direction."
    legend = ('<div style="margin-top: 8px; font-size: 0.9em;">'
              f'<span style="background: rgba({SAD[0]},{SAD[1]},{SAD[2]},0.8); padding: 2px 8px; border-radius: 3px;">− negative / sad</span> '
              f'<span style="background: rgba({HAPPY[0]},{HAPPY[1]},{HAPPY[2]},0.8); padding: 2px 8px; border-radius: 3px;">+ positive / happy</span>'
              " &nbsp; hover a token to see its score</div>")
    return meter_html(tokens, scores) + legend, summary


# ------------------------------------------------------------------------------------ Results tab

RESULTS = [
    ("e1_qwen_layers.png", "**E1 – Which layer? (Qwen)** Steering works at every layer and random vectors of the "
     "same size do nothing. The concept is *most readable* in the middle layers (probe AUC 0.99 at layers 9–14), but "
     "steering in *both* directions works best early, at layer 4: reading a concept and pushing it are not the same."),
    ("e1_tinygpt_layers.png", "**E5 – TinyGPT.** Our 12M model has a happy/sad direction too (probe AUC 0.90–0.97 at "
     "all layers). Unsteered it is already positive (+0.48: TinyStories mostly end happily). Layer 4 steers best: "
     "+0.93 (happy) / −0.56 (sad)."),
    ("e2_strength.png", "**E2 – How hard to push?** Sentiment follows an S-curve with strength; fluency drops "
     "quickly beyond |α| ≈ 8 (perplexity > 3× the unsteered text). Pushed too hard, the text falls into loops."),
    ("e3_prompting.png", "**E3 – Steering vs prompting (Qwen).** Steering moves sentiment more than writing "
     "\"a happy/sad story\" into the prompt (+0.85 vs +0.50, −0.46 vs −0.14); both together are strongest. "
     "The price is fluency."),
    ("e4_pairs.png", "**E4 – How many examples?** Five contrastive pairs already give almost the full effect; "
     "the vector itself stabilises by ~25 pairs (cosine 0.97 to the 150-pair vector)."),
    ("e6_training.png", "**E6 – When does the direction appear in training?** In TinyGPT the happy/sad direction "
     "is readable and steerable early in training, long before the loss has converged; the sad side "
     "keeps getting stronger until about step 2,000."),
]

ABOUT = """
### How it works
1. **Contrastive examples:** sentences that differ only in the concept ("I loved the trip" / "I hated the trip").
2. **Steering vector:** run both sets through the model and take the difference of the mean hidden state
   (the *residual stream*) after one transformer block.
3. **Steering:** while the model writes, a PyTorch forward hook adds **α × vector** to the residual stream at
   that layer, at every token. The weights are never changed and the prompt is untouched.
4. **Control:** a random vector of the same length is added the same way, so we can tell the concept
   apart from "any push changes the text".

**Models:** Qwen2.5-0.5B (base, 24 layers) and **TinyGPT**, a 12.2M-parameter GPT written from scratch for this
project (RoPE, RMSNorm, pre-norm, SwiGLU, 6 layers, BPE vocab 4,096), trained for 18 minutes on 80M tokens of
TinyStories. In this demo Qwen is stopped from writing Chinese characters and "____" blanks (as a base
model it often drifts into exam questions); the experiments used no such filter. The sentiment score is P(positive) − P(negative) from `cardiffnlp/twitter-roberta-base-sentiment-latest`.

UMC301 Project 4 · Archana Yadav · code: [GitHub](https://github.com/arcyadav/tiny-gpt-and-qwen-steering-study)
"""


def build():
    layer0, alpha0 = best_layer(QWEN)
    with gr.Blocks(title="Steering Transformers") as demo:
        gr.Markdown("# Steering transformers from the inside\n"
                    "Change *what a language model feels* by adding a vector to its hidden states — "
                    "no prompt change, no fine-tuning.")
        with gr.Row():
            model = gr.Radio([QWEN, TINY], value=QWEN, label="model")
            concept = gr.Radio([BUILTIN, CUSTOM], value=BUILTIN, label="concept")
        with gr.Accordion("my own concept: positive vs negative examples (one per line)", open=False):
            gr.Markdown("Any contrast works: formal ↔ casual, polite ↔ rude, about food ↔ about sport… "
                        "Steering with α > 0 moves towards the left box, α < 0 towards the right box. "
                        "(The sentiment score below only measures happy/sad.)")
            with gr.Row():
                pos_text = gr.Textbox(CUSTOM_POS, lines=5, label="positive (+)")
                neg_text = gr.Textbox(CUSTOM_NEG, lines=5, label="negative (−)")

        with gr.Tab("Steer"):
            prompt = gr.Textbox("I went to the market today and", label="prompt")
            with gr.Row():
                alpha = gr.Slider(-16, 16, value=alpha0, step=1,
                                  label="strength α  (− = sad / negative, + = happy / positive)")
                layer = gr.Slider(0, 23, value=layer0, step=1, label=f"layer to steer (0–23, best = {layer0})")
            with gr.Accordion("more settings", open=False):
                with gr.Row():
                    n_tokens = gr.Slider(10, 80, value=40, step=5, label="new tokens")
                    seed = gr.Number(0, precision=0, label="seed")
                    show_random = gr.Checkbox(True, label="also show a random vector (control)")
            go = gr.Button("Generate", variant="primary")
            with gr.Row():
                out_none = gr.Textbox(label="no steering", lines=7)
                out_steer = gr.Textbox(label="steered", lines=7)
                out_rand = gr.Textbox(label="random vector, same size (control)", lines=7)
            gr.Examples([[QWEN, p] for p in P.NEUTRAL_PROMPTS[:4]] + [[TINY, p] for p in P.TINY_PROMPTS[:4]],
                        [model, prompt], label="example prompts")
            go.click(steer, [model, prompt, concept, pos_text, neg_text, layer, alpha, n_tokens, seed, show_random],
                     [out_none, out_steer, out_rand])

        with gr.Tab("Concept meter"):
            gr.Markdown("How happy or sad does the model *think* each word is? Every token is coloured by the "
                        "projection of its hidden state on the concept direction at the chosen layer.")
            text = gr.Textbox("The party was wonderful, but then my dog got sick and I cried all night. "
                              "In the morning he was fine again and we played in the sun.", lines=3, label="text")
            meter_layer = gr.Slider(0, 23, value=12, step=1, label="layer to read")
            read = gr.Button("Read the model", variant="primary")
            meter_out = gr.HTML()
            meter_summary = gr.Markdown()
            read.click(meter, [model, text, concept, pos_text, neg_text, meter_layer], [meter_out, meter_summary])

        with gr.Tab("Results"):
            for fname, caption in RESULTS:
                path = os.path.join("results/plots", fname)
                if os.path.exists(path):
                    gr.Markdown(caption)
                    gr.Image(path, show_label=False, container=False)

        with gr.Tab("How it works"):
            gr.Markdown(ABOUT)

        model.change(on_model_change, model, [layer, alpha])
        model.change(lambda name: gr.update(maximum=23 if name == QWEN else 5,
                                            value=12 if name == QWEN else 3), model, meter_layer)
    return demo


def warm_up():
    """Load both models, the classifier and the happy/sad vectors before the first visitor."""
    for name in MODELS:
        get_concept(name, BUILTIN, "", "")
        gen_kwargs(name)
    scorer()


SPACE = "Arcyadav/steering-transformers"  # the static project page that links to the live demo


def announce(url):
    """Write live.json to the static project page, so its 'Live demo' button points at this run."""
    from huggingface_hub import HfApi

    info = {"url": url, "since": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    try:
        HfApi().upload_file(path_or_fileobj=json.dumps(info).encode(), path_in_repo="live.json", repo_id=SPACE,
                            repo_type="space", commit_message="live demo online" if url else "live demo offline")
        print(f"project page now links to: {url}")
    except Exception as e:  # no login / no network: the demo itself still works
        print(f"could not update {SPACE}/live.json: {e}")


def lan_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))  # no packet is sent; this just picks the outgoing interface
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public gradio.live link, announced on the project page")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    warm_up()
    demo = build().queue(default_concurrency_limit=1)
    _, _, share_url = demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share,
                                  prevent_thread_lock=True)
    print(f"\nthis computer:      http://127.0.0.1:{args.port}")
    print(f"same network:       http://{lan_ip()}:{args.port}")
    if share_url:
        print(f"public link:        {share_url}")
        announce(share_url)
    try:
        demo.block_thread()
    finally:
        if share_url:
            announce(None)
