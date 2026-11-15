"""Download TinyStories, train the BPE tokenizer, and encode everything to token files.

TinyStories (Eldan & Li, 2023) is a dataset of short stories written in the
vocabulary of a 3-4 year old. It was designed to show that SMALL language
models (a few million parameters) can learn grammar, coherence and simple
reasoning when the data distribution is narrow enough.

    python prepare.py                 # ~120k stories, vocab 4096
"""

import argparse
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset

from gpt.tokenizer import BPETokenizer

DATA = Path(__file__).parent / "data"
EOT = "<|endoftext|>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", type=int, default=120_000)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--tokenizer-sample-mb", type=float, default=8)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    t0 = time.time()
    print("Streaming TinyStories from Hugging Face…")
    stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    stories = [row["text"].strip() for _, row in zip(range(args.stories), stream)]
    val_stream = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    val = [row["text"].strip() for _, row in zip(range(4000), val_stream)]
    print(f"  {len(stories):,} train / {len(val):,} val stories in {time.time() - t0:.0f}s")

    # train the tokenizer on a sample (BPE merges converge quickly)
    sample, size = [], 0
    for s in stories:
        sample.append(s)
        size += len(s)
        if size > args.tokenizer_sample_mb * 1e6:
            break
    t1 = time.time()
    print(f"Training BPE tokenizer (vocab {args.vocab}) on {size / 1e6:.1f} MB…")
    tok = BPETokenizer.train("\n".join(sample), args.vocab)
    tok.save(DATA / "tokenizer.json")
    print(f"  done in {time.time() - t1:.0f}s")

    for name, texts in [("train", stories), ("val", val)]:
        t2 = time.time()
        ids: list[int] = []
        for s in texts:
            ids.extend(tok.encode(s))
            ids.append(tok.eot_id)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(DATA / f"{name}.bin")
        chars = sum(len(s) for s in texts)
        print(f"  {name}: {len(arr):,} tokens ({chars / len(arr):.2f} chars/token) in {time.time() - t2:.0f}s")


if __name__ == "__main__":
    main()
