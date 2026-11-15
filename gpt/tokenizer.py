"""Byte-level BPE tokenizer, written from scratch (the scheme GPT-2/3/4 use).

How it works
1. Start from bytes, so the base vocabulary is 256 tokens and ANY text can be
   encoded (no "unknown token").
2. Split text into chunks with a regex first (words, numbers, punctuation,
   spaces). Merges never cross chunk boundaries, so "dog." and "dog!" share "dog".
3. Repeatedly find the most frequent adjacent pair of tokens and merge it into
   a new token. Do this `vocab_size - 256` times. The list of merges IS the model.

Training trick: we count each unique chunk once with its frequency instead of
scanning the whole corpus every step. That makes training on tens of MB fast
enough in pure Python.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# GPT-2's pre-tokenization pattern, simplified to avoid the `regex` dependency
SPLIT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")

SPECIAL = {"<|endoftext|>": None}  # id assigned after training


def _pair_counts(words: dict[tuple[int, ...], int]) -> Counter:
    counts: Counter = Counter()
    for word, freq in words.items():
        for a, b in zip(word, word[1:]):
            counts[a, b] += freq
    return counts


def _merge(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out, i = [], 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, pair in enumerate(merges or []):
            self._add_merge(pair, 256 + i)
        self.eot_id = 256 + len(self.merges)
        self._cache: dict[str, list[int]] = {}

    def _add_merge(self, pair: tuple[int, int], new_id: int) -> None:
        self.merges[pair] = new_id
        self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + 1  # +1 for <|endoftext|>

    # ── training ──────────────────────────────────────────────────────────
    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = True) -> "BPETokenizer":
        n_merges = vocab_size - 256 - 1
        chunk_freq = Counter(SPLIT.findall(text))
        words = {tuple(c.encode("utf-8")): f for c, f in chunk_freq.items()}
        tok = cls()
        for step in range(n_merges):
            counts = _pair_counts(words)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + step
            tok._add_merge(pair, new_id)
            # only rebuild words that actually contain the pair
            words = {(_merge(w, pair, new_id) if pair[0] in w else w): f for w, f in words.items()}
            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{n_merges}: {tok.vocab[new_id]!r}")
        tok.eot_id = 256 + len(tok.merges)
        return tok

    # ── encode / decode ───────────────────────────────────────────────────
    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # apply the earliest-learned merge available (lowest new id), like GPT-2
            pairs = {(a, b) for a, b in zip(ids, ids[1:])}
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = list(_merge(tuple(ids), best, self.merges[best]))
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in SPLIT.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids if i != self.eot_id)
        return data.decode("utf-8", errors="replace")

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
        Path(path).write_text(json.dumps({"merges": [list(p) for p, _ in ordered]}))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        merges = [tuple(p) for p in json.loads(Path(path).read_text())["merges"]]
        return cls(merges)
