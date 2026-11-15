"""Correctness checks.   python tests.py"""

import torch

from gpt.model import GPT, GPTConfig
from gpt.tokenizer import BPETokenizer


def test_tokenizer_roundtrip():
    tok = BPETokenizer.train("the cat sat on the mat. " * 40 + "Hello, wörld! 123", 300, verbose=False)
    for text in ["the cat sat on the mat.", "Hello, wörld! 123 😀", "  spaces\tand\nnewlines  ", ""]:
        assert tok.decode(tok.encode(text)) == text, text


def _tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)).eval()


def test_kv_cache_matches_full_forward():
    m, x = _tiny(), torch.randint(0, 300, (1, 12))
    full, _ = m(x)
    caches = [{} for _ in m.blocks]
    m(x[:, :11], kv_caches=caches)
    inc, _ = m(x[:, 11:], kv_caches=caches, start_pos=11)
    assert torch.allclose(full[:, -1], inc[:, -1], atol=1e-5)


def test_fused_attention_matches_manual():
    m, x = _tiny(), torch.randint(0, 300, (2, 20))
    fused, _ = m(x)
    for b in m.blocks:
        b.attn.fused = False
    manual, _ = m(x)
    assert torch.allclose(fused, manual, atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
