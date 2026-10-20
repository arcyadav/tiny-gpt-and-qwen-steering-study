# Steering Transformers from the Inside

Activation steering on **Qwen2.5-0.5B** and on **TinyGPT**, a 12M-parameter GPT written from scratch
and trained on TinyStories. A "happy − sad" direction is computed from contrastive sentence pairs and
added to the residual stream during generation — no prompt change, no fine-tuning.

UMC301 Project 4 · Archana Yadav · full plan in [`spec.md`](spec.md)

## Layout
| Path | What |
|---|---|
| `steer/core.py` | steering vectors, forward hooks, generation |
| `steer/pairs.py` | contrastive pairs, neutral prompts, probe sentences |
| `steer/score.py` | sentiment, perplexity, distinct-2 |
| `steer/experiments.py` | experiments E1–E4 on Qwen |
| `tinygpt/model.py` | the from-scratch GPT (RoPE, RMSNorm, SwiGLU, …) |
| `tinygpt/data.py`, `train.py` | TinyStories tokenizer + training |
| `kaggle/` | notebooks for running on a Kaggle T4 |
| `tests/` | model tests (`pytest`) |

## Run
```bash
pip install -r requirements.txt
pytest                                                   # model tests
python -m steer.experiments --quick --out results/quick  # small CPU check of E1–E4
python -m steer.experiments --out results/qwen           # full run (GPU)
```
