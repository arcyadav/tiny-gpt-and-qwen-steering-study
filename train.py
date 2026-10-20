"""Train TinyGPT (baseline config) on TinyStories, saving checkpoints along the way.

    python train.py --data data --out results/tinygpt                      # Kaggle T4
    python train.py --data data_small --out results/smoke --tokens 2e6 --tiny   # CPU smoke test
    python train.py --data data --out results/tinygpt_early --stop-step 600 --ckpt-every 50 --eval-every 50
        # first 600 steps of the same run (same seed and LR schedule), with dense checkpoints

Intermediate checkpoints (every --ckpt-every steps) are used to study when the
happy/sad direction appears during training (spec E6).
"""
import argparse
import contextlib
import json
import math
import os
import time

import numpy as np
import torch

from tinygpt.model import GPT, GPTConfig

# a small model for quick CPU checks of the whole pipeline
TINY = dict(n_layer=2, n_head=4, n_embd=128)


def get_batch(data, batch_size, block_size, rng, device):
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[s : s + block_size] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1 : s + 1 + block_size] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def evaluate(model, data, batch_size, block_size, n_batches, device, amp_ctx):
    """Mean loss on a fixed set of validation windows."""
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, block_size, rng, device)
        with amp_ctx():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def lr_at(step, total, peak, warmup, min_ratio=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/tinygpt")
    ap.add_argument("--tokens", type=float, default=80e6, help="training token budget")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-step", type=int, default=None,
                    help="stop after this step; the LR schedule still follows the full --tokens budget")
    ap.add_argument("--tiny", action="store_true", help="2-layer model for CPU smoke tests")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    train_data = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")

    overrides = dict(vocab_size=meta["vocab_size"], block_size=args.block, **(TINY if args.tiny else {}))
    cfg = GPTConfig(**overrides)
    model = GPT(cfg).to(device)
    n_params = model.num_params()

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=device == "cuda",
    )
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp else contextlib.nullcontext
    train_model = torch.compile(model) if args.compile else model

    tokens_per_step = args.batch * args.block
    total_steps = int(args.tokens // tokens_per_step)
    last_step = total_steps - 1 if args.stop_step is None else min(args.stop_step, total_steps - 1)
    run_dir = args.out
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    def save(path, extra=None):
        torch.save({"config": cfg.to_dict(), "model": model.state_dict(), **(extra or {})}, path)

    print(f"TinyGPT: {n_params/1e6:.2f}M params, {total_steps} steps x {tokens_per_step} tokens, device={device}")

    log = {"train": [], "val": []}
    status = "ok"
    t0 = time.time()
    t_last, tokens_since = t0, 0
    for step in range(last_step + 1):
        lr = lr_at(step, total_steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.batch, args.block, rng, device)
        with amp_ctx():
            _, loss = train_model(x, y)
        if not torch.isfinite(loss):
            status = "diverged"
            print(f"step {step}: loss is {loss.item()} -> stopping")
            break
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        scaler.step(opt)
        scaler.update()
        tokens_since += tokens_per_step

        if step % 50 == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            tok_s = tokens_since / (now - t_last)
            t_last, tokens_since = now, 0
            log["train"].append({"step": step, "loss": loss.item(), "lr": lr, "grad_norm": grad_norm,
                                 "tokens_per_s": tok_s})
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | gnorm {grad_norm:.2f} | {tok_s:,.0f} tok/s")

        if step % args.eval_every == 0 or step == last_step:
            val = evaluate(model, val_data, args.batch, args.block, args.eval_batches, device, amp_ctx)
            log["val"].append({"step": step, "tokens": (step + 1) * tokens_per_step, "loss": val})
            print(f"step {step:5d} | val loss {val:.4f}")

        if step % args.ckpt_every == 0 or step == last_step:
            save(os.path.join(run_dir, "checkpoints", f"step{step:05d}.pt"), {"step": step})

    train_time = time.time() - t0
    final_val = log["val"][-1]["loss"] if log["val"] else float("nan")
    summary = {
        "seed": args.seed,
        "status": status,
        "config": cfg.to_dict(),
        "params": n_params,
        "kv_cache_bytes_per_token": model.kv_cache_bytes_per_token(),
        "steps": last_step + 1,
        "tokens": (last_step + 1) * tokens_per_step,
        "final_val_loss": final_val,
        "final_val_ppl": math.exp(final_val) if math.isfinite(final_val) else None,
        "train_time_s": train_time,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump(log, f)
    save(os.path.join(run_dir, "ckpt.pt"), {"summary": summary})
    print(f"done in {train_time/60:.1f} min | final val loss {final_val:.4f} | saved to {run_dir}")


if __name__ == "__main__":
    main()
