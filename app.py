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
from steer.presets import QWEN_PRESETS, TINY_PRESETS
from steer.score import SentimentScorer

QWEN, TINY = "Qwen2.5-0.5B (pretrained)", "TinyGPT 12M (built from scratch)"
MODELS = {
    # read_layer: where the concept score and the meter read the text (the happy/sad probe peaks there)
    QWEN: dict(load=lambda: QwenSteerer(), pairs=P.sentiment_pairs("train"), e1="results/qwen/e1.json",
               read_layer=12),
    TINY: dict(load=lambda: TinyGPTSteerer("results/tinygpt/ckpt.pt", "data/tokenizer.json"),
               pairs=P.tiny_sentiment_pairs(), e1="results/tinygpt_steer/e1.json", read_layer=3),
}
BUILTIN = "happy ↔ sad (built-in pairs)"
CUSTOM = "my own concept (type examples below)"
PRESETS = {QWEN: QWEN_PRESETS, TINY: TINY_PRESETS}
WRITING_TIP = ("Write 4–8 short sentences per side, one per line, that differ mainly in the concept. "
               "Matched pairs work best (line 1 on the left ↔ line 1 on the right). "
               "α > 0 steers towards the left box, α < 0 towards the right box.")
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


# TinyGPT ends each story with an end-of-story token, and steering towards happy makes that more
# likely (happy endings). The demo shows the marker and what follows instead of cutting the text.
END_MARKER = "\n[end of story — the model starts a new one]\n"


# Some TinyStories texts contain UTF-8 quotes decoded as Windows-1252 ("â€œHiâ€"), and TinyGPT
# learned to write them. The demo shows the intended characters.
_MOJIBAKE = [("â€œ", "“"), ("â€", "”"), ("â€™", "’"), ("â€˜", "‘"), ("â€”", "—"), ("â€“", "–"),
             ("â€¦", "…"), ("â€", "”")]


def fix_quotes(text):
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    return text


def gen_kwargs(name):
    if name != QWEN:
        return {"end_marker": END_MARKER}
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


def concept_choices(name):
    return [BUILTIN, *PRESETS[name], CUSTOM]


def sides(name, choice):
    """Names of the positive and negative side of a concept, for labels and scores."""
    if choice == BUILTIN:
        return "happy", "sad"
    if choice in PRESETS[name]:
        return PRESETS[name][choice][:2]
    return "left box", "right box"


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

def steering_vector(name, choice, pos_text, neg_text, layer):
    v = get_concept(name, choice, pos_text, neg_text).vector[layer]
    if choice != BUILTIN:  # same length as the happy/sad vector, so α means the same for any concept
        v = v * (get_concept(name, BUILTIN, "", "").vector[layer].norm() / v.norm())
    return v


def steer(name, prompt, choice, pos_text, neg_text, layer, alpha, n_tokens, seed, show_random):
    if not prompt.strip():
        raise gr.Error("Type a prompt first.")
    model = get_model(name)
    layer = min(int(layer), model.n_layers - 1)
    v = steering_vector(name, choice, pos_text, neg_text, layer)
    # one batch: row 0 unsteered, row 1 steered, row 2 a random vector of the same norm (control).
    # The hook adds alpha * vectors[row] to each row, so the rows differ only in what is added.
    rows = [torch.zeros_like(v), v] + ([random_like(v, seed=layer)] if show_random else [])
    vectors = torch.stack(rows).unsqueeze(1)  # (rows, 1, d_model) broadcasts over positions
    texts = [fix_quotes(t) for t in model.generate([prompt] * len(rows), layer, vectors, float(alpha),
                                                   max_new_tokens=int(n_tokens), seed=int(seed), **gen_kwargs(name))]
    # concept score: where the unsteered model places each text on the concept direction
    concept, (a, b) = get_concept(name, choice, pos_text, neg_text), sides(name, choice)
    plain = [t.replace(END_MARKER, "\n") for t in texts]  # scores read the text without the marker
    scores = [concept_score(name, concept, t) for t in plain]
    if choice == BUILTIN:  # the independent sentiment classifier as well
        out = [f"{t}\n\n— sentiment {s:+.2f} · happy ↔ sad score {c:+.2f}"
               for t, s, c in zip(texts, scorer()(plain), scores)]
    else:
        out = [f"{t}\n\n— {a} ↔ {b} score {c:+.2f}  (+ = {a}, − = {b})" for t, c in zip(texts, scores)]
    return out[0], out[1], out[2] if show_random else ""


# ------------------------------------------------------------------------------- Transition tab

class SharedNoise:
    """Sampling with temperature + top-p through the Gumbel-max trick, with the SAME noise for
    every row of the batch: argmax(logits / T + Gumbel noise) is a sample from softmax(logits / T),
    and sharing the noise means rows with similar distributions pick the same token. Lets Qwen
    generate all strengths in one batch while keeping 'rows differ only by the steering'."""

    def __init__(self, temperature, top_p, seed):
        self.t, self.top_p = temperature, top_p
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, input_ids, scores):
        scores = scores.float() / self.t
        ordered, order = scores.sort(dim=-1, descending=True)
        probs = ordered.softmax(-1)
        ordered[probs.cumsum(-1) - probs > self.top_p] = float("-inf")  # top-p
        scores = torch.full_like(scores, float("-inf")).scatter(-1, order, ordered)
        u = torch.rand(scores.shape[-1], generator=self.gen).clamp(1e-10, 1 - 1e-10)
        return scores - torch.log(-torch.log(u)).to(scores.device)  # one noise row, broadcast


@torch.no_grad()
def generate_coupled(name, prompt, layer, vectors, n_tokens, seed):
    """One continuation per added vector (already scaled), all sharing their random draws."""
    model = get_model(name)
    if name != QWEN:  # TinyGPT is fast: separate runs with the same seed share the draws too
        return [fix_quotes(model.generate([prompt], layer, v, 1.0, max_new_tokens=n_tokens, seed=seed,
                                          **gen_kwargs(name))[0]) for v in vectors]
    from transformers import LogitsProcessorList

    vectors = torch.stack(vectors).unsqueeze(1)
    batch = model.encode([prompt] * len(vectors))
    with model.steering(layer, vectors, 1.0):
        out = model.model.generate(**batch, max_new_tokens=n_tokens, do_sample=False,
                                   logits_processor=LogitsProcessorList([SharedNoise(0.7, 0.9, seed)]),
                                   pad_token_id=model.tok.pad_token_id, **gen_kwargs(name))
    new = out[:, batch["input_ids"].shape[1]:]
    return [t.strip() for t in model.tok.batch_decode(new, skip_special_tokens=True)]


def transition(name, prompt, choice, pos_text, neg_text, layer, max_alpha, n_tokens, seed):
    """The same prompt at five strengths from -max to +max. The rows share their random draws, so
    they differ only by the steering: they often agree for a few words and then branch where the
    push changes a word choice."""
    if not prompt.strip():
        raise gr.Error("Type a prompt first.")
    model = get_model(name)
    layer = min(int(layer), model.n_layers - 1)
    v = steering_vector(name, choice, pos_text, neg_text, layer)
    m = float(max_alpha)
    alphas = [-m, -m / 2, 0.0, m / 2, m]
    texts = generate_coupled(name, prompt, layer, [v * a for a in alphas], int(n_tokens), int(seed))
    concept, (a, b) = get_concept(name, choice, pos_text, neg_text), sides(name, choice)
    scores = [concept_score(name, concept, t.replace(END_MARKER, "\n")) for t in texts]
    rows = "".join(transition_row(al, sc, t, texts[2], prompt) for al, sc, t in zip(alphas, scores, texts))
    head = (f'<div style="font-size: 0.9em; opacity: 0.75; margin-bottom: 6px;">{html.escape(b)} ← → '
            f'{html.escape(a)} · layer {layer} · seed {int(seed)} · bar = {html.escape(a)} ↔ {html.escape(b)} '
            "score · <b>coloured text</b> = where the row branches off from the unsteered one</div>")
    return head + f'<table style="width: 100%; border-collapse: collapse;">{rows}</table>'


def transition_row(alpha, score, text, base, prompt):
    """One row: strength, score bar, and the text with the part that differs from α = 0 coloured."""
    r, g, b = HAPPY if alpha > 0 else SAD
    words, base_words = re.split(r"(\s+)", text), re.split(r"(\s+)", base)
    same = 0
    while same < min(len(words), len(base_words)) and words[same] == base_words[same]:
        same += 1
    kept, changed = "".join(words[:same]), "".join(words[same:])
    body = html.escape(kept)
    if changed and alpha != 0:
        body += f'<span style="background: rgba({r},{g},{b},0.28); border-radius: 3px;">{html.escape(changed)}</span>'
    elif changed:
        body += html.escape(changed)
    body = body.replace("\n", "<br>")
    fill = max(-1.0, min(1.0, score / 1.5))  # bar: ±1.5 fills one half
    br, bg, bb = HAPPY if fill > 0 else SAD
    left = 50 if fill > 0 else 50 + fill * 50
    bar = (f'<div style="position: relative; width: 120px; height: 10px; border-radius: 5px; '
           f'background: rgba(128,128,128,0.18);"><div style="position: absolute; left: {left:.1f}%; '
           f'width: {abs(fill) * 50:.1f}%; height: 100%; border-radius: 5px; background: rgb({br},{bg},{bb});">'
           '</div><div style="position: absolute; left: 50%; top: -2px; width: 1px; height: 14px; '
           'background: rgba(128,128,128,0.7);"></div></div>')
    label = "α 0 (none)" if alpha == 0 else f"α {alpha:+g}"
    return ('<tr style="border-top: 1px solid rgba(128,128,128,0.25); vertical-align: top;">'
            f'<td style="padding: 10px 8px; white-space: nowrap; font-weight: 600;">{label}</td>'
            f'<td style="padding: 12px 8px; white-space: nowrap;">{bar}<div style="font-size: 0.85em; '
            f'margin-top: 4px; font-family: monospace;">{score:+.2f}</div></td>'
            f'<td style="padding: 10px 8px; line-height: 1.6;"><span style="opacity: 0.6;">{html.escape(prompt)} '
            f'</span>{body}</td></tr>')


# ------------------------------------------------------------------------------------- Mix tab

MIX_DEFAULTS = {QWEN: ("rich ↔ poor", "old ↔ young"), TINY: ("huge ↔ tiny", "magic ↔ everyday")}


def mix_choices(name):
    return [BUILTIN, *PRESETS[name]]


def preset_texts(name, choice):
    """The two example boxes of a preset as text ('' for the built-in pairs)."""
    if choice == BUILTIN:
        return "", ""
    pairs = PRESETS[name][choice][3]
    return "\n".join(p for p, _ in pairs), "\n".join(n for _, n in pairs)


def mix(name, prompt, choice_a, choice_b, strength, n_tokens, seed):
    """Two concepts at once: a 3×3 grid of (concept A: +, 0, −) × (concept B: −, 0, +). Each cell
    adds s_a·A + s_b·B, so the centre is the unsteered text and the corners combine both sides."""
    if not prompt.strip():
        raise gr.Error("Type a prompt first.")
    if choice_a == choice_b:
        raise gr.Error("Pick two different concepts.")
    layer = best_layer(name)[0]
    s = float(strength)
    va = steering_vector(name, choice_a, *preset_texts(name, choice_a), layer)
    vb = steering_vector(name, choice_b, *preset_texts(name, choice_b), layer)
    grid = [(sa, sb) for sa in (1, 0, -1) for sb in (-1, 0, 1)]
    texts = generate_coupled(name, prompt, layer, [s * (sa * va + sb * vb) for sa, sb in grid],
                             int(n_tokens), int(seed))
    (a1, a2), (b1, b2) = sides(name, choice_a), sides(name, choice_b)
    cos = float(va @ vb / (va.norm() * vb.norm()))
    e = html.escape
    cell = "padding: 10px; vertical-align: top; line-height: 1.5; border: 1px solid rgba(128,128,128,0.25);"
    head = f'<th style="{cell} text-align: center;">'
    rows = [f'<tr><th style="{cell}"></th>{head}{e(b2)}</th>{head}— {e(b1)} / {e(b2)} 0 —</th>{head}{e(b1)}</th></tr>']
    for i, (row_name, sa) in enumerate([(a1, 1), (f"— {a1} / {a2} 0 —", 0), (a2, -1)]):
        tds = []
        for j in range(3):
            sb = grid[3 * i + j][1]
            t = e(texts[3 * i + j].replace(END_MARKER, " [end of story] ")).replace("\n", "<br>")
            bg = "rgba(128,128,128,0.12)" if sa == 0 and sb == 0 else ("rgba(232,163,61,0.10)" if sa and sb else "transparent")
            label = "unsteered" if sa == 0 and sb == 0 else " + ".join(x for x, k in [(a1 if sa > 0 else a2, sa), (b1 if sb > 0 else b2, sb)] if k)
            tds.append(f'<td style="{cell} background: {bg};"><div style="font-size: 0.8em; opacity: 0.6; '
                       f'margin-bottom: 4px;">{e(label)}</div>{t}</td>')
        rows.append(f'<tr><th style="{cell} text-align: right; white-space: nowrap;">{e(row_name)}</th>{"".join(tds)}</tr>')
    info = (f'<div style="font-size: 0.9em; opacity: 0.75; margin-bottom: 6px;">prompt: <b>{e(prompt)}</b> · layer '
            f'{layer} · strength {s:g} per concept · seed {int(seed)} · cosine between the two directions '
            f'{cos:+.2f} (0 = independent) · shaded corners = both concepts at once</div>')
    return info + f'<div style="overflow-x: auto;"><table style="border-collapse: collapse; width: 100%;">{"".join(rows)}</table></div>'


def on_model_change(name):
    layer, alpha = best_layer(name)
    n = 24 if name == QWEN else 6
    return gr.update(maximum=n - 1, value=layer, label=f"layer to steer (0–{n - 1}, best = {layer})"), \
        gr.update(value=alpha), gr.update(choices=concept_choices(name), value=BUILTIN)


def on_concept_change(name, choice):
    """A preset fills the two example boxes; 'my own' keeps what is typed; built-in hides them."""
    if choice == BUILTIN:
        return gr.update(visible=False), gr.update(), gr.update(), gr.update()
    if choice in PRESETS[name]:
        a, b, tip, pairs = PRESETS[name][choice]
        return (gr.update(visible=True), gr.update(value="\n".join(p for p, _ in pairs), label=f"{a} (α > 0)"),
                gr.update(value="\n".join(n for _, n in pairs), label=f"{b} (α < 0)"),
                f"**Tried and tested:** {tip}  \n{WRITING_TIP}")
    return (gr.update(visible=True), gr.update(label="positive side (α > 0)"),
            gr.update(label="negative side (α < 0)"), WRITING_TIP)


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


def concept_score(name, concept, text):
    """How far a text sits along the concept direction, as the unsteered model reads it: the mean
    token score at the model's read layer (about +1 = like the positive examples, -1 = negative)."""
    if not text.strip():
        return 0.0
    _, scores = token_scores(get_model(name), concept, text, MODELS[name]["read_layer"])
    valid = [s for s in scores if s == s]
    return sum(valid) / max(1, len(valid))


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
    a, b = sides(name, choice)
    summary = f"**Average: {mean:+.2f}** → the model reads this text on the **{a if mean > 0 else b}** side of the direction."
    legend = ('<div style="margin-top: 8px; font-size: 0.9em;">'
              f'<span style="background: rgba({SAD[0]},{SAD[1]},{SAD[2]},0.8); padding: 2px 8px; border-radius: 3px;">− {html.escape(b)}</span> '
              f'<span style="background: rgba({HAPPY[0]},{HAPPY[1]},{HAPPY[2]},0.8); padding: 2px 8px; border-radius: 3px;">+ {html.escape(a)}</span>'
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
    ("e6_training.png", "**E6 – When does the direction appear in training?** Re-running the first 600 steps with "
     "a checkpoint every 50 steps shows a sudden switch: until step 150 TinyGPT has no happy/sad direction (probe AUC "
     "below 0.5, steering does nothing); between steps 150 and 250 (about 3M tokens, 4% of training, once validation "
     "loss drops below ~3.1) it appears and steering starts to work. Steering towards sad keeps strengthening until "
     "about step 2,000."),
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


# Looks: a cool "lab" style for Qwen, and a pastel storybook style while TinyGPT (trained on children's
# stories) is selected. The model radio toggles a `tiny` class on <body>; the CSS swaps Gradio's
# theme variables under it.
HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600&family=Nunito:wght@400;700'
        '&display=swap" rel="stylesheet">')
TOGGLE_JS = "(name) => { document.body.classList.toggle('tiny', name.startsWith('TinyGPT')); }"
THEME_CSS = """
body, .gradio-container { transition: background-color 0.5s ease; }
.gradio-container {
  --button-primary-background-fill: linear-gradient(120deg, #4f46e5, #7c3aed);
  --button-primary-background-fill-hover: linear-gradient(120deg, #4338ca, #6d28d9);
  --button-primary-text-color: #fff;
  --color-accent: #6366f1; --slider-color: #6366f1; --color-accent-soft: #e0e7ff;
  --block-radius: 12px;
  background: linear-gradient(135deg, #eef2ff 0%, #f0f9ff 50%, #faf5ff 100%) !important;
}
.dark .gradio-container, .gradio-container.dark, body.dark .gradio-container {
  --color-accent-soft: #312e81;
  background: linear-gradient(135deg, #0b1020 0%, #111836 55%, #1e1b4b 100%) !important;
}
#hero h1 { display: inline-block; font-weight: 800; color: transparent; -webkit-background-clip: text;
           background-clip: text; background-image: linear-gradient(90deg, #4f46e5, #0ea5e9, #7c3aed); }

/* ---- storybook mode (TinyGPT) ---- */
body.tiny .gradio-container {
  --font: 'Nunito', ui-rounded, sans-serif;
  --button-primary-background-fill: linear-gradient(120deg, #f472b6, #fb923c);
  --button-primary-background-fill-hover: linear-gradient(120deg, #ec4899, #f97316);
  --color-accent: #ec4899; --slider-color: #ec4899; --color-accent-soft: #fce7f3;
  --block-radius: 20px; --block-border-color: #f9c4dc; --border-color-primary: #f9c4dc;
  --block-background-fill: rgba(255, 255, 255, 0.82); --input-background-fill: #fffdf7;
  --block-label-background-fill: #fef3c7; --block-label-text-color: #9a3412;
  --block-title-text-color: #9d174d; --body-text-color: #3b2a4a;
  --block-shadow: 0 6px 18px rgba(236, 72, 153, 0.12);
  --table-even-background-fill: #fff7ed; --table-odd-background-fill: #fdf2f8; --table-border-color: #f9c4dc;
  background:
    radial-gradient(circle at 10% 15%, #fde2e4 0, transparent 35%),
    radial-gradient(circle at 90% 10%, #fef9c3 0, transparent 30%),
    radial-gradient(circle at 85% 85%, #d1fae5 0, transparent 35%),
    radial-gradient(circle at 15% 90%, #e0f2fe 0, transparent 35%),
    #fff7ed !important;
}
body.tiny .dark .gradio-container, body.tiny .gradio-container.dark, body.tiny.dark .gradio-container {
  --block-background-fill: rgba(40, 30, 80, 0.78); --input-background-fill: #2a2050;
  --block-border-color: #6d28d9; --border-color-primary: #6d28d9; --color-accent-soft: #4c1d95;
  --block-label-background-fill: #4c1d95; --block-label-text-color: #fde68a;
  --block-title-text-color: #f9a8d4; --body-text-color: #f5f3ff;
  --table-even-background-fill: #2a2050; --table-odd-background-fill: #241b47; --table-border-color: #6d28d9;
  background:
    radial-gradient(1.5px 1.5px at 20% 30%, #fff 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 70% 20%, #fde68a 50%, transparent 51%),
    radial-gradient(1px 1px at 40% 70%, #fff 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 85% 60%, #f9a8d4 50%, transparent 51%),
    radial-gradient(circle at 80% 10%, #4c1d95 0, transparent 40%),
    linear-gradient(160deg, #1e1b4b, #2e1065 60%, #312e81) !important;
}
body.tiny #hero h1 { font-family: 'Fredoka', 'Nunito', sans-serif; font-weight: 600;
  background-image: linear-gradient(90deg, #ec4899, #f97316, #eab308, #22c55e, #3b82f6, #a855f7) !important; }
body.tiny #hero h1::after { content: " 📖✨"; -webkit-text-fill-color: initial; color: initial; }
body.tiny .tab-nav button, body.tiny button[role="tab"] { font-family: 'Fredoka', sans-serif; }
"""


def build():
    layer0, alpha0 = best_layer(QWEN)
    with gr.Blocks(title="Steering Transformers") as demo:
        gr.Markdown("# Steering transformers from the inside\n"
                    "Change *what a language model feels* by adding a vector to its hidden states — "
                    "no prompt change, no fine-tuning.", elem_id="hero")
        with gr.Row():
            model = gr.Radio([QWEN, TINY], value=QWEN, label="model")
            concept = gr.Dropdown(concept_choices(QWEN), value=BUILTIN, label="concept",
                                  info="the built-in happy ↔ sad, a tested example, or your own")
        with gr.Group(visible=False) as custom_box:
            tip = gr.Markdown(WRITING_TIP)
            with gr.Row():
                pos_text = gr.Textbox(lines=6, label="positive side (α > 0)")
                neg_text = gr.Textbox(lines=6, label="negative side (α < 0)")

        with gr.Tab("Steer"):
            prompt = gr.Textbox("I went to the market today and", label="prompt")
            with gr.Row():
                alpha = gr.Slider(-16, 16, value=alpha0, step=1,
                                  label="strength α  (+ = happy / first side, − = sad / second side)")
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

        with gr.Tab("Transition"):
            gr.Markdown("The same prompt at five strengths, from one side of the concept to the other: the text "
                        "version of morphing between faces in a GAN's latent space. Every row uses the same random "
                        "seed, so rows differ only by the steering; the coloured part of each row is where it "
                        "branches off from the unsteered text.")
            t_prompt = gr.Textbox("When I opened the door,", label="prompt")
            with gr.Row():
                t_max = gr.Slider(2, 16, value=8, step=1, label="max strength (rows: −max, −max/2, 0, +max/2, +max)")
                t_layer = gr.Slider(0, 23, value=layer0, step=1, label=f"layer to steer (best = {layer0})")
            with gr.Row():
                t_tokens = gr.Slider(10, 80, value=40, step=5, label="new tokens")
                t_seed = gr.Number(0, precision=0, label="seed")
            t_go = gr.Button("Show transition", variant="primary")
            t_out = gr.HTML()
            gr.Examples([[QWEN, "When I opened the door,"], [QWEN, "The weather this morning was"],
                         [TINY, "Tom had a red ball. He"], [TINY, "One day, a big dog went to the park. The dog"]],
                        [model, t_prompt], label="example prompts")
            t_go.click(transition, [model, t_prompt, concept, pos_text, neg_text, t_layer, t_max, t_tokens, t_seed],
                       t_out)

        with gr.Tab("Mix"):
            gr.Markdown("Two concepts at once: add **A + B** together and see whether the model combines them "
                        "(\"rich + old\" → a wealthy old man?), like the *king − man + woman* arithmetic of word "
                        "vectors. All nine cells share the same random draws; the centre is the unsteered text. "
                        "Uses the model's best layer; the concept picker above is not used here.")
            m_prompt = gr.Textbox("My neighbour", label="prompt")
            with gr.Row():
                m_a = gr.Dropdown(mix_choices(QWEN), value=MIX_DEFAULTS[QWEN][0], label="concept A (rows)")
                m_b = gr.Dropdown(mix_choices(QWEN), value=MIX_DEFAULTS[QWEN][1], label="concept B (columns)")
            with gr.Row():
                m_strength = gr.Slider(2, 12, value=6, step=1, label="strength per concept (6 works best; "
                                                                      "8+ starts to break when two are added)")
                m_tokens = gr.Slider(10, 80, value=40, step=5, label="new tokens")
                m_seed = gr.Number(0, precision=0, label="seed")
            m_go = gr.Button("Mix", variant="primary")
            m_out = gr.HTML()
            # prompts that introduce a person, place or thing without describing it, so the two
            # concepts decide the description
            gr.Examples([[QWEN, "My neighbour"], [QWEN, "The house at the end of our street"],
                         [QWEN, "Last weekend, my uncle"],
                         [TINY, "Tom went for a walk in the woods. Suddenly, he saw a"],
                         [TINY, "At the bottom of the garden lived a"], [TINY, "Lily opened the box. Inside there was a"]],
                        [model, m_prompt], label="example prompts")
            m_go.click(mix, [model, m_prompt, m_a, m_b, m_strength, m_tokens, m_seed], m_out)

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

        model.change(None, model, None, js=TOGGLE_JS)
        model.change(on_model_change, model, [layer, alpha, concept])
        concept.change(on_concept_change, [model, concept], [custom_box, pos_text, neg_text, tip])
        model.change(lambda name: gr.update(maximum=23 if name == QWEN else 5, value=best_layer(name)[0],
                                            label=f"layer to steer (best = {best_layer(name)[0]})"), model, t_layer)
        model.change(lambda name: gr.update(maximum=23 if name == QWEN else 5,
                                            value=12 if name == QWEN else 3), model, meter_layer)
        model.change(lambda name: (gr.update(choices=mix_choices(name), value=MIX_DEFAULTS[name][0]),
                                   gr.update(choices=mix_choices(name), value=MIX_DEFAULTS[name][1])),
                     model, [m_a, m_b])
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
                                  prevent_thread_lock=True, css=THEME_CSS, head=HEAD)
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
