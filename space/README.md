---
title: Steering Transformers
emoji: 🎚️
colorFrom: yellow
colorTo: blue
sdk: static
app_file: index.html
pinned: false
short_description: Steer Qwen2.5-0.5B and a from-scratch GPT with a slider
---

# Steering transformers from the inside

Activation steering on **Qwen2.5-0.5B** and on **TinyGPT**, a 12M-parameter GPT written from scratch and
trained on TinyStories. A "happy − sad" direction (or your own concept) is computed from contrastive
examples and added to the residual stream during generation: no prompt change, no fine-tuning.

The live slider demo (`python app.py --share`) runs on the author's computer; this page shows the method,
the results, and a replay of real generations from the experiments, and links to the live demo when it is online.

UMC301 Project 4 · Archana Yadav · code and experiments:
[github.com/arcyadav/tiny-gpt-and-qwen-steering-study](https://github.com/arcyadav/tiny-gpt-and-qwen-steering-study)
