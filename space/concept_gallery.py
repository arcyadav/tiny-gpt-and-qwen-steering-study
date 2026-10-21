"""Generate the 'other concepts' and 'mixing' examples for the project page (E7, E8), with the same
code and settings as the live demo's Transition and Mix tabs, so the page shows real generations.

    python space/concept_gallery.py     # -> space/concepts.json (then build_page.py copies it)

Runs on CPU in a few minutes. The examples are the ones in DEMO_SHOWCASE.md.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402

Q, T = app.QWEN, app.TINY

# (model, concept preset, prompt, seed, layer, max strength, what to notice)
STRIPS = [
    (Q, "rich ↔ poor", "For dinner tonight, my mother", 1, 4, 8,
     "The same dinner, from not enough rice to a five-star hotel."),
    (Q, "old ↔ young", "The person sitting next to me on the bus was", 0, 4, 8,
     "The stranger ages from a bullied schoolkid to an 86-year-old with a weak leg."),
    (Q, "formal ↔ casual", "i asked the retailer", 0, 4, 8,
     "Tone shifts step by step: rambling speech, a plain request, contract language. At −8 the text breaks into a run of zeros."),
    (Q, "English ↔ French", "When I looked from my window i", 0, 4, 8,
     "Most French at −4. At −8 the push is too strong: only French function words survive (\"indique\", \"et\") and the text loops."),
    (Q, "Indian ↔ American", "On Sunday morning, my family", 1, 4, 8,
     "A backyard picnic becomes a temple visit. The American side hardly moves: the unsteered text is already American."),
    (Q, "about food ↔ about sports", "i asked my brother", 0, 4, 8,
     "Basketball at one end, breakfast at the other; the middle rows barely change. A topic is more of a switch than a slider."),
    (T, "huge ↔ tiny", "Lily went to the zoo and saw a", 1, 4, 8,
     "The animal itself changes: a little bird, a lion, elephants, an enormous lion. \"Tiny\" also brings in \"gently\"."),
    (T, "sharing ↔ greedy", "Lily found a shiny red ball in the park. Her friend asked to play. Lily", 0, 4, 8,
     "The same moment ends four ways: refuses, won't share, plays alone, plays together."),
    (T, "magic ↔ everyday", "Tom went for a walk in the woods. Suddenly, he saw a", 0, 4, 8,
     "An apple on a bench, a bear, a monster, a brave knight, then \"something magical happened!\""),
    (T, "dogs ↔ cats", "Lily had a pet. Her pet was a", 1, 3, 8,
     "Works after rewriting the example pairs so they differ only in the animal word (the first version steered towards owls and tea)."),
]

# (model, concept A, concept B, prompt, seed, strength, what to notice)
MIXES = [
    (T, "huge ↔ tiny", "magic ↔ everyday", "Tom went for a walk in the woods. Suddenly, he saw a", 1, 6,
     "huge + magic gives a dragon and tiny + magic a fairy, though no example sentence mentions either."),
    (Q, "rich ↔ poor", "old ↔ young", "The house at the end of our street", 0, 6,
     "Wealth and age show through the house: heirlooms and landscaped gardens, or a dilapidated, cold home."),
    (Q, "rich ↔ poor", "old ↔ young", "My neighbour", 1, 6,
     "Single concepts are clear (old: can't see; young: wins every game); poor + old combines both. Rich + old reads mostly as old."),
]

MODEL_KEY = {Q: "qwen", T: "tiny"}


def strip(name, choice, prompt, seed, layer, m, note):
    pos, neg = app.preset_texts(name, choice)
    v = app.steering_vector(name, choice, pos, neg, layer)
    concept = app.get_concept(name, choice, pos, neg)
    alphas = [-m, -m / 2, 0.0, m / 2, m]
    texts = app.generate_coupled(name, prompt, layer, [v * a for a in alphas], 40, seed)
    a, b = app.sides(name, choice)
    rows = [[al, round(app.concept_score(name, concept, t.replace(app.END_MARKER, "\n")), 2),
             t.replace(app.END_MARKER, " [end of story] ")] for al, t in zip(alphas, texts)]
    return {"model": MODEL_KEY[name], "concept": choice, "pos": a, "neg": b, "prompt": prompt, "seed": seed,
            "layer": layer, "note": note, "rows": rows}


def mix(name, ca, cb, prompt, seed, s, note):
    layer = app.best_layer(name)[0]
    va = app.steering_vector(name, ca, *app.preset_texts(name, ca), layer)
    vb = app.steering_vector(name, cb, *app.preset_texts(name, cb), layer)
    grid = [(x, y) for x in (1, 0, -1) for y in (-1, 0, 1)]
    texts = app.generate_coupled(name, prompt, layer, [s * (x * va + y * vb) for x, y in grid], 40, seed)
    return {"model": MODEL_KEY[name], "a": app.sides(name, ca), "b": app.sides(name, cb), "prompt": prompt,
            "seed": seed, "strength": s, "layer": layer, "note": note,
            "cos": round(float(va @ vb / (va.norm() * vb.norm())), 2),
            "cells": [t.replace(app.END_MARKER, " [end of story] ") for t in texts]}


def main():
    out = {"strips": [], "mixes": []}
    for args in STRIPS:
        out["strips"].append(strip(*args))
        print("strip", args[1], "|", args[2])
    for args in MIXES:
        out["mixes"].append(mix(*args))
        print("mix", args[1], "+", args[2], "|", args[3])
    with open("space/concepts.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
