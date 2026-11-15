"""Train the GPT on TinyStories tokens.

    python train.py                      # default ~12M-param model
    python train.py --iters 200 --eval-every 50    # quick smoke test

Training recipe (standard for small GPTs):
- AdamW (decoupled weight decay; no decay on biases/LayerNorm/embeddings)
- linear warmup, then cosine decay of the learning rate
- gradient clipping at 1.0
- periodic validation loss; the best checkpoint is kept
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from gpt.model import GPT, GPTConfig
from gpt.tokenizer import BPETokenizer

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "checkpoints"


def get_batch(data: np.ndarray, batch: int, block: int, device: str):
    ix = np.random.randint(0, len(data) - block - 1, batch)
    x = torch.from_numpy(np.stack([data[i : i + block].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1 : i + 1 + block].astype(np.int64) for i in ix]))
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, args, device) -> dict:
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, args.batch, args.block, device))[1].item() for _ in range(args.eval_iters)]
        out[name] = float(np.mean(losses))
    model.train()
    return out


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.iters - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(DATA / "tokenizer.json")
    train = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block, n_layer=args.layers,
                    n_head=args.heads, n_embd=args.embd, dropout=args.dropout)
    model = GPT(cfg).to(device)
    print(f"device={device}  params={model.num_params() / 1e6:.2f}M  train tokens={len(train):,}")
    tokens_per_step = args.batch * args.block
    print(f"tokens/step={tokens_per_step:,}  total tokens seen={tokens_per_step * args.iters / 1e6:.1f}M "
          f"(~{tokens_per_step * args.iters / len(train):.1f} epochs)")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    OUT.mkdir(exist_ok=True)
    log, best_val, t0 = [], float("inf"), time.time()
    for step in range(args.iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)

        if step % args.eval_every == 0 or step == args.iters:
            losses = estimate_loss(model, {"train": train, "val": val}, args, device)
            elapsed = time.time() - t0
            log.append({"step": step, **losses, "lr": lr_at(step, args), "minutes": round(elapsed / 60, 2)})
            print(f"step {step:>5}  train {losses['train']:.3f}  val {losses['val']:.3f}  "
                  f"ppl {math.exp(losses['val']):6.1f}  lr {lr_at(step, args):.2e}  {elapsed / 60:5.1f} min")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": step, "val_loss": best_val}, OUT / "best.pt")
            (OUT / "train_log.json").write_text(json.dumps(log, indent=1))
        if step == args.iters:
            break

        x, y = get_batch(train, args.batch, args.block, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"done: best val loss {best_val:.3f} (perplexity {math.exp(best_val):.1f}) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
