"""
Generate training result visualizations for the README.
Outputs: assets/val_bpb_progression.png, assets/eval_comparison.png, assets/inference_benchmark.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ASSETS = Path(__file__).parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "axes.titlecolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.8,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

BLUE   = "#58a6ff"
GREEN  = "#3fb950"
ORANGE = "#f78166"
PURPLE = "#d2a8ff"
YELLOW = "#e3b341"

# ── 1. Val BPB Progression ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor("#0d1117")

stages = ["Base\n(2000 steps)", "SFT\n(500 steps)", "SFT\n(2000 steps)"]
bpb    = [1.2991,              0.8698,              0.6729]
colors = [ORANGE,              BLUE,                GREEN]

bars = ax.bar(stages, bpb, color=colors, width=0.45, zorder=3,
              edgecolor="#0d1117", linewidth=1.5)

for bar, val in zip(bars, bpb):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
            f"{val:.4f}", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#e6edf3")

# improvement annotation arrow
ax.annotate("", xy=(2, bpb[2] + 0.04), xytext=(0, bpb[0] + 0.04),
            arrowprops=dict(arrowstyle="-|>", color=GREEN,
                            lw=1.8, mutation_scale=16))
ax.text(1, bpb[0] + 0.065, "−48% BPB", ha="center",
        fontsize=10, color=GREEN, fontweight="bold")

ax.set_ylabel("Validation BPB  (lower is better)", fontsize=11)
ax.set_title("Val BPB Progression — d12 (286M params)", fontsize=13, fontweight="bold", pad=14)
ax.set_ylim(0, 1.55)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)

legend_handles = [
    mpatches.Patch(color=ORANGE, label="Base pretrain"),
    mpatches.Patch(color=BLUE,   label="SFT 500 steps"),
    mpatches.Patch(color=GREEN,  label="SFT 2000 steps"),
]
ax.legend(handles=legend_handles, loc="upper right",
          framealpha=0.15, edgecolor="#30363d", fontsize=10)

plt.tight_layout(pad=1.4)
out = ASSETS / "val_bpb_progression.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved {out}")

# ── 2. Eval Comparison ─────────────────────────────────────────────────────────
tasks   = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "SpellingBee"]
sft500  = [28.5,  23.5,  27.5,  1.0,   63.5]
sft2000 = [25.5,  20.0,  27.0,  2.5,   84.0]
rl240   = [24.5,  20.5,  27.0,  1.0,   82.5]

x     = np.arange(len(tasks))
width = 0.24

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor("#0d1117")

b1 = ax.bar(x - width,     sft500,  width, label="SFT 500 steps",  color=ORANGE, zorder=3, edgecolor="#0d1117", linewidth=1)
b2 = ax.bar(x,             sft2000, width, label="SFT 2000 steps", color=GREEN,  zorder=3, edgecolor="#0d1117", linewidth=1)
b3 = ax.bar(x + width,     rl240,   width, label="RL 240 steps",   color=PURPLE, zorder=3, edgecolor="#0d1117", linewidth=1)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}", ha="center", va="bottom",
                fontsize=8.5, color="#e6edf3")

ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.set_ylabel("Accuracy  (%)", fontsize=11)
ax.set_title("Eval Results — SFT 500 vs SFT 2000 vs RL 240 steps  (d12, 200 problems/task, greedy)",
             fontsize=12, fontweight="bold", pad=14)
ax.set_ylim(0, 100)
ax.yaxis.grid(True, zorder=0, alpha=0.6)
ax.set_axisbelow(True)

ax.legend(loc="upper left", framealpha=0.15, edgecolor="#30363d", fontsize=10)

# highlight best scores
best_vals   = [max(a, b, c) for a, b, c in zip(sft500, sft2000, rl240)]
best_labels = ["+20.5pp SpellingBee", "+1.5pp GSM8K"]
for i, task in enumerate(tasks):
    if task in ("SpellingBee", "GSM8K"):
        ax.axvline(x=i, color="#ffffff", alpha=0.04, linewidth=22, zorder=0)

plt.tight_layout(pad=1.4)
out = ASSETS / "eval_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved {out}")

# ── 3. Inference Benchmark ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Inference Benchmark — d6 (73M) vs d12 (286M)  [30 requests, 128 max tokens]",
             fontsize=12, fontweight="bold", y=1.01)

models = ["d6 (73M)", "d12 (286M)"]
clrs   = [BLUE, GREEN]

metrics = [
    ("Avg Latency (ms)",       [761.8,  1499.5], "lower is better"),
    ("Throughput (tok/sec)",   [165.4,   79.7],  "higher is better"),
    ("Peak VRAM (MB)",         [220.9,  820.4],  "lower is better"),
]

for ax, (title, vals, note) in zip(axes, metrics):
    ax.set_facecolor("#161b22")
    bars = ax.bar(models, vals, color=clrs, width=0.4, zorder=3,
                  edgecolor="#0d1117", linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                f"{v:,.1f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#e6edf3")
    ax.set_title(f"{title}\n({note})", fontsize=10, pad=8)
    ax.set_ylim(0, max(vals) * 1.28)
    ax.yaxis.grid(True, zorder=0, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

plt.tight_layout(pad=1.8)
out = ASSETS / "inference_benchmark.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved {out}")

print("\nAll charts saved to assets/")
