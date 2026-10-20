# Spec — Steering Transformers from the Inside: Activation Steering on Qwen and on a GPT Built from Scratch (UMC301 Project 4)

Author: Archana Yadav
Deadline: 27 Oct 2026, 2 pm (viva + MS Teams submission)
Status: Draft v2 (20 Oct 2026) — replaces the TinyGPT-ablations draft

---

## 1. Problem
We usually control a language model from the *outside*: by changing the prompt or by fine-tuning
the weights. But inside a transformer, ideas like "happy" or "formal" seem to be stored as
**directions** in the hidden states (the residual stream). If that is true, we can control the model
by adding a vector to its hidden states while it generates — no prompt change, no training. This is
called **activation steering**.

## 2. Goal and research questions
Build an interactive tool that steers a language model with sliders, and measure honestly when
steering works and when it breaks, on two models:

- **Qwen2.5-0.5B** — a real pretrained LLM (main, reliable part)
- **TinyGPT (12M)** — a GPT written from scratch in this project and trained on TinyStories
  (original part: we own its whole training history)

Questions:
1. **Does it work?** Does adding a "happy − sad" vector really change sentiment — more than a random
   vector of the same size?
2. **Where?** Which layer is the best place to steer?
3. **How hard?** How does steering strength trade off against fluency?
4. **vs prompting:** Is steering as good as just asking ("write happily")?
5. **Small vs big, and when:** Does steering work in a 12M model, and *when during training* does the
   happy/sad direction appear?

## 3. Method
1. **Contrastive pairs:** ~100 pairs of short texts that differ only in the concept
   (e.g. "I loved the trip, it was wonderful" / "I hated the trip, it was awful").
   Written with templates + hand-checked; separate pair sets for Qwen (general English) and
   TinyGPT (TinyStories-style English).
2. **Steering vector** at layer ℓ: mean hidden state (residual stream after block ℓ) over positive
   texts − mean over negative texts.
3. **Steering:** during generation, add `α · v` (v = the raw mean difference; for Qwen layer 14,
   |v| ≈ 1.7 vs a typical hidden-state norm ≈ 16) to the residual stream at layer ℓ, at every token. Implemented with a PyTorch forward
   hook — the model weights are never changed.
4. **Concept meter (for the demo):** project each token's hidden state onto v → colour every token by
   how "happy" or "sad" the model currently represents it.

Concepts: **sentiment** (sad ↔ happy) = core; **formality** (formal ↔ casual) = second concept if
time allows.

## 4. Models
| Model | Size | Layers / d_model | Role |
|---|---|---|---|
| Qwen2.5-0.5B (base) | 494M | 24 / 896 | main experiments + demo |
| TinyGPT (`tinygpt/model.py`, baseline config: RoPE, RMSNorm, pre-norm, SwiGLU) | 12.2M | 6 / 384 | built from scratch, trained once on TinyStories |

TinyGPT training: **one run only** (baseline config), 80M tokens (~5,000 steps, ~10–15 min on a Kaggle
T4), checkpoint saved every 500 steps (11 checkpoints) for question 5. No architecture ablations are
run; the other switches in `model.py` (learned / sinusoidal / ALiBi / no positions, LayerNorm,
post-norm, GELU, GQA / MQA) stay as tested options for explaining design choices.

**TinyGPT components (and why):**

| Component | What it does | Why |
|---|---|---|
| BPE tokenizer, vocab 4,096 | trained on TinyStories | GPT-2's 50k vocab would put ~19M params in the embedding alone |
| RoPE | rotates query/key vectors by a position-dependent angle | attention depends on relative distance |
| RMSNorm | rescales each vector to unit RMS | simpler than LayerNorm, same quality |
| Pre-norm | `x + f(norm(x))` | clean residual path, stable training |
| SwiGLU FFN (hidden 1,024) | gated feed-forward | standard in modern LLMs; same params as a 4× GELU MLP |
| 6-head attention, context 256 | causal self-attention | standard |
| Tied embeddings | output layer = input embedding matrix | saves 1.6M params |
| **Residual stream** | every block *adds* its output to a running vector | **this is the vector steering edits** |

## 5. Experiments
Fixed for all: 50 neutral prompts (e.g. "I went to the market today and"), 40 new tokens,
sampling with temperature 0.7, fixed seed. Every setting also run with a **random vector of the same
norm** (control).

| # | Experiment | Settings | Generations |
|---|---|---|---|
| E1 | **Layer sweep** (Qwen) | every 2nd layer (12) × α ∈ {4, 8} × {steering, random} | ~2,400 |
| E2 | **Strength sweep** (Qwen, best layer) | α ∈ {−12 … +12} (9 values) | ~450 |
| E3 | **Steering vs prompting** (Qwen) | none / prompt / steering / prompt + steering | ~200 |
| E4 | **How many pairs are needed** (Qwen) | vector from 5 / 10 / 25 / 50 / 100 pairs | ~250 |
| E5 | **TinyGPT steering** | 6 layers × 2 strengths + strength sweep at best layer | ~1,000 |
| E6 | **When does the direction appear** (TinyGPT) | the 11 training checkpoints: probe accuracy (happy vs sad) + steering effect | ~550 |

All generations are short and batched (50 prompts per batch), so each experiment is minutes on a T4.

## 6. Metrics
- **Steering effect:** sentiment score P(positive) − P(negative) ∈ [−1, 1] from a 3-class classifier
  (`cardiffnlp/twitter-roberta-base-sentiment-latest`), plus % clearly positive / negative; change vs
  no steering. (A 2-class SST-2 classifier was tried first and rejected: it labels neutral text
  "negative" — "The train leaves at 5 pm" → 63% negative — which hides steering towards sad.)
- **Fluency:** perplexity of the generated text under the *unsteered* model (higher = more broken),
  and **distinct-2** (share of unique word bigrams), because repetitive loops can have low perplexity
- **Probe AUC** (E1, E6): how well the steering direction separates 20 + 20 *naturally written*
  positive / negative sentences that contain none of the pair words (e.g. "My dog died yesterday").
  (Held-out template pairs were tried first and rejected: they differ by one word, so every layer —
  even layer 0 — scored 100%.)
- **Cosine similarity** between steering vectors across layers / pair counts / checkpoints
- Qualitative: fixed example prompts shown side by side in the report

## 7. Compute and time budget (kept deliberately small)
| Item | Where | Estimate |
|---|---|---|
| TinyStories download + tokenizer + tokenise | Kaggle CPU/GPU | ~15 min |
| TinyGPT training (1 run) | Kaggle T4 | ~10–15 min |
| E1–E6 generation + scoring | Kaggle T4 | ~1 h |
| Development / debugging | laptop CPU (few generations at a time) | — |
| **Total GPU** | | **< 2 GPU-hours** (Kaggle gives 30 h/week) |

Estimates are measured and updated after the first Kaggle run.

## 8. Deliverables
- `tinygpt/model.py`, `tinygpt/data.py`, `train.py` — the from-scratch GPT (tests in `tests/`)
- `steer/` — steering vectors, hooks, generation, evaluation (works for both models)
- `data/pairs_*.json` — contrastive pairs; `prompts.json` — neutral prompts
- Kaggle notebook running E1–E6 → `results/*.json` + plots
- **Deployed demo (Gradio on a free Hugging Face Space, CPU):**
  - *Steer* — prompt, model (Qwen / TinyGPT), concept slider(s), layer → normal vs steered text
  - *Concept meter* — tokens coloured by their projection on the happy/sad direction
  - *Results* — plots and the written finding for each experiment
- Report: method, results per experiment, findings, limitations
- Transcript of this Claude Code session → standalone HTML (Turn 1, 2, … with collapsed responses)

## 9. Timeline
| Date | Work |
|---|---|
| 20 Oct | Spec; contrastive pairs + prompts; steering hooks on Qwen working locally (first slider demo) |
| 21 Oct | Evaluation (classifier, perplexity); E1 + E2 on Kaggle; TinyGPT data + training run |
| 22 Oct | E3, E4; E5 + E6 on TinyGPT |
| 23 Oct | Gradio app (steer + concept meter); deploy to HF Space |
| 24 Oct | Plots, results tab, report draft |
| 25–26 Oct | Report, transcript HTML, buffer |
| 27 Oct, 2 pm | Submit |

**If behind schedule, cut in this order:** formality concept → E4 → E6 → E5 (Qwen part alone is a
complete project).

## 10. Risks
| Risk | Mitigation |
|---|---|
| Steering makes text incoherent | strength sweep finds the usable range; fluency is a reported metric |
| Effect is only "random noise pushes text around" | random-vector control in every setting |
| TinyGPT too small to steer | still a finding (steerability vs scale); Qwen part is independent |
| Demo slow on free CPU Space | 0.5B model, ≤ 60 new tokens, vectors precomputed; local `--share` fallback |
| Sentiment classifier misjudges odd texts | spot-check samples by hand; report examples, not only numbers |

## 11. Acceptance criteria
- Steering implemented with hooks on both models; random-vector control included
- E1–E3 and E5 done, each with a plot/table and a written finding (E4, E6 if time)
- Public demo URL that works from another computer
- Report + transcript HTML submitted before 27 Oct 2026, 2 pm

## 12. Related work (to cite)
- Vaswani et al., *Attention Is All You Need* (2017)
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021) — the residual stream view
- Turner et al., *Activation Addition: Steering Language Models Without Optimization* (2023)
- Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition* (2024)
- Zou et al., *Representation Engineering* (2023)
- Park et al., *The Linear Representation Hypothesis* (2023)
- Eldan & Li, *TinyStories* (2023); Qwen2.5 technical report (2024)
