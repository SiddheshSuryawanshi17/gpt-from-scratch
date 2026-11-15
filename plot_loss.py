"""Plot the training curve from checkpoints/train_log.json -> docs/loss.png"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
log = json.loads((ROOT / "checkpoints" / "train_log.json").read_text())
steps = [r["step"] for r in log]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
ax.plot(steps, [r["train"] for r in log], label="train", lw=2)
ax.plot(steps, [r["val"] for r in log], label="validation", lw=2)
ax.set_xlabel("step")
ax.set_ylabel("cross-entropy loss")
ax.set_yscale("log")
best = min(log, key=lambda r: r["val"])
ax.set_title(f"GPT from scratch on TinyStories · best val loss {best['val']:.2f} (perplexity {math.exp(best['val']):.1f})")
ax.grid(alpha=0.3)
ax.legend()
(ROOT / "docs").mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(ROOT / "docs" / "loss.png")
print("saved docs/loss.png")
