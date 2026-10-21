# Steering Transformers from the Inside

Activation steering on **Qwen2.5-0.5B** and on **TinyGPT**, a 12M-parameter GPT written from scratch
and trained on TinyStories. A "happy − sad" direction is computed from contrastive sentence pairs and
added to the residual stream during generation — no prompt change, no fine-tuning.

UMC301 Project 4 · Archana Yadav · report in [`REPORT.md`](REPORT.md) · original plan in [`spec.md`](spec.md)

**Project page:** https://arcyadav-steering-transformers.static.hf.space (results, plots, replay of real
generations, and a link to the live demo while it is running)

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
| `plots.py` | result figures → `results/plots/` |
| `app.py` | Gradio demo: steer, custom concepts, concept meter |
| `space/` | static project page (Hugging Face Space) + build/deploy scripts |
| `tests/` | model tests (`pytest`) |

## Run
```bash
pip install -r requirements.txt
pytest                                                   # model tests
python -m steer.experiments --quick --out results/quick  # small CPU check of E1–E4
python -m steer.experiments --out results/qwen           # full run (GPU)
python plots.py                                          # figures
python app.py --share                                    # live demo: local, same network, public link
python space/build_page.py && python space/deploy.py     # update the project page
```
