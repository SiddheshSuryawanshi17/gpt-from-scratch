"""A decoder-only Transformer (GPT), written from scratch in PyTorch.

Architecture (same family as GPT-2):
    token embedding + learned position embedding
    N × [ LayerNorm → causal multi-head self-attention → residual
          LayerNorm → MLP (4× wider, GELU)            → residual ]
    final LayerNorm → linear head (weights tied to the token embedding)

Extras that matter in practice:
- Pre-LayerNorm (more stable training than the original post-LN Transformer)
- Weight tying between input embedding and output head (fewer params, better LM)
- Scaled init for residual projections (keeps activations stable as depth grows)
- KV cache at inference: each new token attends to cached keys/values instead
  of recomputing the whole sequence, so generation is O(n) per token, not O(n²).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256      # max context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)       # one matmul for Q, K and V
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj.SCALE_INIT = True
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size)
        self.register_buffer("mask", mask, persistent=False)
        self.fused = True   # set False to run the explicit attention maths everywhere

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))

        if kv_cache is not None:                      # inference: append to cache
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v

        if kv_cache is None and self.fused:
            # training: PyTorch's fused kernel computes exactly the maths below, faster
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attn_drop.p if self.training else 0.0)
            return self.resid_drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))

        S = k.size(2)                                  # total length incl. cache
        # attention(Q, K, V) = softmax(Q Kᵀ / √d) V, with a causal mask
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, S - T : S, :S] == 0, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.proj.SCALE_INIT = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.ln2, self.mlp = nn.LayerNorm(cfg.n_embd), MLP(cfg)

    def forward(self, x, kv_cache=None):
        x = x + self.attn(self.ln1(x), kv_cache)       # pre-LN + residual
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "SCALE_INIT", False):            # GPT-2: scale by 1/√(2·layers)
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()

    def forward(self, idx, targets=None, kv_caches=None, start_pos: int = 0):
        B, T = idx.shape
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for i, block in enumerate(self.blocks):
            x = block(x, None if kv_caches is None else kv_caches[i])
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int | None = 50,
                 top_p: float | None = 0.95, stop_id: int | None = None):
        """Autoregressive sampling with a KV cache, temperature, top-k and nucleus (top-p)."""
        self.eval()
        idx = idx[:, -self.cfg.block_size :]
        caches = [{} for _ in self.blocks]
        logits, _ = self(idx, kv_caches=caches)            # prefill the cache with the prompt
        pos = idx.size(1)
        for _ in range(max_new_tokens):
            if pos >= self.cfg.block_size:                  # context full: restart cache on a sliding window
                idx_cond = idx[:, -self.cfg.block_size + 1 :]
                caches = [{} for _ in self.blocks]
                logits, _ = self(idx_cond, kv_caches=caches)
                pos = idx_cond.size(1)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            if top_p:
                sorted_p, sorted_i = torch.sort(probs, descending=True)
                drop = sorted_p.cumsum(-1) - sorted_p > top_p     # keep the smallest set with mass ≥ p
                sorted_p[drop] = 0
                probs = torch.zeros_like(probs).scatter(-1, sorted_i, sorted_p)
                probs = probs / probs.sum(-1, keepdim=True)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_id is not None and nxt.item() == stop_id:
                break
            logits, _ = self(nxt, kv_caches=caches, start_pos=pos)
            pos += 1
        return idx
