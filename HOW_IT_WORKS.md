# How it works (and how to explain it in an interview)

## 1. What a language model actually does
A GPT predicts the **next token** given the previous ones. Training minimises cross-entropy:
how surprised the model is by the real next token. Generation repeats "predict, sample, append".
Everything else (chat, reasoning, code) is this one objective at scale.

## 2. Tokenizer: byte-level BPE (`gpt/tokenizer.py`)
Models read integers, not text. **Byte Pair Encoding** builds a vocabulary bottom-up:
1. Start with the 256 possible bytes, so any text, emoji or language can be encoded.
2. Split text into chunks with a regex (words, numbers, punctuation) so merges don't glue words to punctuation.
3. Count every adjacent pair of tokens; merge the most frequent pair into a new token; repeat
   until the vocabulary reaches 4,096.

Encoding applies the learned merges in the order they were learned. Result: ~3.9 characters per
token on this data, versus 1 for character-level, so the model sees 4× more text in the same context.

*Speed trick*: count each unique word once with its frequency instead of rescanning the corpus
every merge, so 3,839 merges train in about 20 seconds in pure Python.

## 3. The model (`gpt/model.py`)
```
tokens → token embedding + position embedding
       → 6 × Block:  x = x + Attention(LayerNorm(x))
                     x = x + MLP(LayerNorm(x))
       → LayerNorm → Linear → logits over 4,096 tokens
```
- **Self-attention**: each token makes a query (what am I looking for?), key (what do I contain?)
  and value (what do I pass on?). `softmax(QKᵀ/√d)·V`. Dividing by √d keeps the dot products
  from growing with dimension, which would saturate softmax and kill gradients.
- **Causal mask**: a token may only attend to earlier tokens; otherwise training would cheat by looking ahead.
- **Multi-head** (6 heads × 64 dims): different heads learn different relations (who did what, which noun a pronoun refers to).
- **MLP** (4× wider, GELU): per-token processing; where much of the "knowledge" is stored.
- **Residual connections + pre-LayerNorm**: gradients flow straight through 6 layers; pre-LN is more stable than the original post-LN.
- **Weight tying**: the output layer reuses the input embedding matrix (saves 1.6M params, often improves perplexity).
- **Scaled init**: residual output layers start smaller (÷√(2·layers)) so activations don't grow with depth.

Attention is written out explicitly (and verified equal to PyTorch's fused kernel in a test);
the fused kernel is used during training for speed.

## 4. Training (`train.py`)
- **Data**: 120k TinyStories (27M tokens). Random 256-token windows; the target is the same window shifted by one.
- **AdamW** with weight decay 0.1 on matrices only (not biases/LayerNorm), betas (0.9, 0.95).
- **Learning-rate schedule**: linear warmup for 200 steps (avoids early instability), then cosine decay to 10%.
- **Gradient clipping** at 1.0 guards against loss spikes.
- **Validation** every 250 steps on held-out stories; the best checkpoint is saved.
- **Perplexity** = e^(val loss): roughly "how many tokens the model is choosing between".

## 5. Generation (`model.generate`)
- **KV cache**: keys/values for past tokens are stored, so each new token costs one forward
  pass of length 1 instead of re-running the whole sequence. A test checks cached output equals full recompute.
- **Temperature** divides logits: <1 is safer and more repetitive, >1 is more creative and chaotic.
- **Top-k**: sample only from the k most likely tokens.
- **Top-p (nucleus)**: sample from the smallest set of tokens whose probabilities sum to p.

## 6. Likely interview questions

**Why does a 12M-parameter model write coherent stories?**
TinyStories is a narrow distribution (simple vocabulary, simple plots). The paper behind it showed
small models can learn grammar and consistency when the data isn't the whole internet.
The same model trained on Wikipedia would produce fluent-looking nonsense.

**Why BPE over characters or whole words?**
Characters make sequences long (attention cost grows with length squared). Words create a huge
vocabulary and can't handle new words. BPE balances both and never meets an unknown token.

**What limits the model?**
Context (256 tokens), size (12M parameters) and data (27M tokens). Scaling laws say loss
improves predictably with all three.

**Why is attention O(n²) and how is that addressed in practice?**
Every token attends to every earlier token. Real systems use FlashAttention (memory-efficient
exact attention), sliding windows, or KV-cache compression.

**What changes between this and GPT-4-class models?**
Scale (billions of parameters, trillions of tokens), RoPE instead of learned positions,
RMSNorm, SwiGLU MLPs, grouped-query attention, mixed precision, then instruction tuning and RLHF/DPO.

## 7. Glossary
- **Logits**: raw scores before softmax.
- **Cross-entropy**: −log probability assigned to the correct next token.
- **Epoch**: one pass over the training data.
- **Checkpoint**: saved weights plus config.
