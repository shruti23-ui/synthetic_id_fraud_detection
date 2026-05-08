"""Quick plot of the 4 models that produced results in the sweep."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research_outputs"

rows = [
    ("ResNet50",        0.921, 0.928, 0.9885, 251, 1723.6, 31.5, "trains cleanly"),
    ("ConvNeXt-Tiny",   0.440, 0.000, 0.4833, 216, 2412.7, 31.2, "collapsed (F1=0)"),
    ("EfficientNetV2-S",0.659, 0.672, 0.6993, 780, 2779.7, 43.2, "underfit"),
    ("ViT-S/16",        0.457, 0.000, 0.5169, 0,   0.0,    0.0,  "collapsed after 6 ep"),
]

names = [r[0] for r in rows]
acc   = [r[1] for r in rows]
f1    = [r[2] for r in rows]
auc   = [r[3] for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
fig.suptitle("Architecture sweep on template-aware split (10 epochs each, lr=1e-4 AdamW, AMP)\n"
             "ResNet50 is the only architecture that converges under uniform hyperparameters on this small dataset",
             fontsize=12, fontweight="bold")

cmap = plt.colormaps.get_cmap("tab10")
colors = [cmap(i) for i in range(len(names))]

for ax, vals, title in zip(axes, [acc, f1, auc], ["Accuracy", "F1", "ROC-AUC"]):
    bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.5, linestyle=":", color="gray", lw=1, alpha=0.4)
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

axes[0].set_ylabel("Test set metric")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "07_architecture_sweep.png", dpi=150, bbox_inches="tight")
print("saved", OUT / "07_architecture_sweep.png")
