"""Generate a story from the trained model.

    python generate.py "Once upon a time, a little robot"
    python generate.py --temperature 1.0 --top-k 0 --top-p 0.9 "Lily found a key"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from gpt.model import GPT, GPTConfig
from gpt.tokenizer import BPETokenizer

ROOT = Path(__file__).parent


def load(checkpoint: Path = ROOT / "checkpoints" / "best.pt", device: str | None = None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(checkpoint, map_location=device)
    model = GPT(GPTConfig(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, BPETokenizer.load(ROOT / "data" / "tokenizer.json"), device


def story(model, tok, device, prompt: str, max_new_tokens=300, temperature=0.8, top_k=50, top_p=0.95) -> str:
    ids = torch.tensor([tok.encode(prompt)], device=device)
    out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k or None,
                         top_p=top_p or None, stop_id=tok.eot_id)
    return tok.decode(out[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tok, device = load()
    t0 = time.time()
    text = story(model, tok, device, args.prompt, args.tokens, args.temperature, args.top_k, args.top_p)
    n = len(tok.encode(text))
    print(text)
    print(f"\n[{n} tokens in {time.time() - t0:.1f}s on {device}]")


if __name__ == "__main__":
    main()
