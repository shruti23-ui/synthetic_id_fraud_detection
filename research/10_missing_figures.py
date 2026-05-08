"""
research/10_missing_figures.py
------------------------------
PhD-/presentation-grade figures missing from the previous Stage-09 set.

The Stage-09 script reported headline numbers like ROC-AUC=0.999 and
PR-AUC=0.999 but never *rendered* the actual curves. This script fixes
that and adds the figures examiners actively look for in a fraud-
detection thesis chapter:

    10_roc_curves.png                ROC for ResNet50 + Two-Stream + chance,
                                     with shaded 95 % bootstrap band, AUC and
                                     operating-point markers.
    10_pr_curves.png                 Precision-recall for both models with
                                     average precision and operating point.
    10_confusion_matrix_resnet50.png Confusion matrix (counts + normalised)
                                     for the single-stream baseline.
    10_confusion_matrix_two_stream.png   Same for the Two-Stream model.
    10_per_ctype_recall.png          Per-forgery-type recall bar chart for
                                     both models (Real / Inpaint_and_Rewrite /
                                     Crop_and_Replace).
    10_training_curves.png           Train loss + val ROC-AUC + val recall
                                     across the 20 fine-tuning epochs for
                                     both models, with the best-epoch marker.
    10_lr_schedule.png               Cosine + warm-up LR schedule for the
                                     Two-Stream training.
    10_threshold_sweep.png           Precision / recall / F1 vs decision
                                     threshold for the Two-Stream model.
    10_fft_spectrum.png              Real vs fake log-magnitude FFT
                                     spectra (the architectural rationale
                                     for the FFT branch — visualised).
    10_gradcam_correct_per_ctype.png Grad-CAM on correctly-classified fakes
                                     of each forgery type (positive cases,
                                     not just failures).
    10_gradcam_resnet_vs_two_stream.png  Side-by-side Grad-CAM:
                                     ResNet50 vs Two-Stream (RGB branch
                                     layer4) on the *same* failure cases —
                                     the strongest visual argument for the
                                     two-stream architecture.

All figures use a consistent palette, large readable fonts, gridlines, and
informative titles with sample sizes / metric values baked into the title.

This script intentionally does NOT retrain anything — it loads the existing
checkpoints from models/ and the metric CSVs from outputs/metrics/ and
research_outputs/.
"""

from __future__ import annotations

# IMPORTANT: sklearn before torch on Windows
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_ASCII", " 123456789#")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from local_dataset_loader import LocalEmbeddingDataset                    # noqa: E402
from supervised_finetune import (  # noqa: E402
    CONFIG as SUP_CFG,
    SupervisedResNet50,
    get_simple_eval_transform,
)
from template_split import template_aware_split                            # noqa: E402
from two_stream_model import (  # noqa: E402
    TwoStreamForgeryNet,
    _compute_fft_magnitude,
)
from utils import get_device, seed_everything                              # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ── Vibrant presentation-style figure conventions ────────────────────────
# - DejaVu Sans (matplotlib default), 11 pt body / 12 pt bold titles
# - white background, light grey grid, all four spines
# - tab10 palette (vivid blue / vivid red) for high-contrast comparison
# - presentation-friendly sizes (not IEEE-column-constrained)
plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["DejaVu Sans", "Arial", "Helvetica"],
    "mathtext.fontset":    "dejavusans",
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.titleweight":    "bold",
    "axes.labelsize":      11,
    "axes.labelweight":    "normal",
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "xtick.major.size":    4.0,
    "ytick.major.size":    4.0,
    "xtick.major.width":   1.0,
    "ytick.major.width":   1.0,
    "axes.linewidth":      1.0,
    "axes.edgecolor":      "black",
    "axes.spines.top":     True,
    "axes.spines.right":   True,
    "axes.facecolor":      "white",
    "figure.facecolor":    "white",
    "axes.grid":           True,
    "grid.linestyle":      "-",
    "grid.linewidth":      0.5,
    "grid.color":          "0.85",
    "grid.alpha":          0.6,
    "legend.fontsize":     10,
    "legend.frameon":      True,
    "legend.fancybox":     True,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "0.7",
    "legend.borderpad":    0.5,
    "legend.handlelength": 2.4,
    "lines.linewidth":     2.0,
    "lines.markersize":    7.0,
    "lines.markeredgewidth": 0.8,
    "savefig.dpi":         200,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
    "figure.dpi":          120,
})

# Presentation-friendly figure widths (inches)
COL_W = 6.0      # single-panel figure
DBL_W = 12.0     # multi-panel figure

# Vibrant tab10-style palette
COLOR_RESNET     = "#1f77b4"   # tab:blue
COLOR_TWOSTREAM  = "#d62728"   # tab:red
COLOR_AVGFFT     = "#2ca02c"   # tab:green
COLOR_CHANCE     = "#7f7f7f"   # tab:grey

LS_RESNET    = "-"
LS_TWOSTREAM = "-"
MK_RESNET    = "o"
MK_TWOSTREAM = "s"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

CTYPE_LABELS = {0: "Real", 1: "Inpaint_and_Rewrite", 2: "Crop_and_Replace"}


# ===========================================================================
# Loaders
# ===========================================================================

def make_test_loader(image_size: int, batch_size: int, seed: int):
    eval_tf = get_simple_eval_transform(image_size)
    ds = LocalEmbeddingDataset(transform=eval_tf)
    cfg = dict(SUP_CFG)
    cfg["seed"] = seed
    cfg["val_size"] = 0.20
    cfg["test_size"] = 0.20
    _, _, test_idx = template_aware_split(ds.records, cfg)
    test_idx = sorted(test_idx)
    loader = DataLoader(
        Subset(ds, test_idx), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    return loader, ds.records, test_idx


@torch.no_grad()
def predict_test(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs, preds, labels_all = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type="cuda"):
            logits = model(x).float()
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels_all.append(y.numpy())
    return (np.concatenate(labels_all),
            np.concatenate(probs),
            np.concatenate(preds))


def bootstrap_roc_band(y_true: np.ndarray, y_prob: np.ndarray,
                       n_iter: int = 500, seed: int = 42):
    """Return fpr-grid, mean tpr, lo/hi 95% band (linear interpolation)."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 201)
    tprs = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[idx], y_prob[idx])
        tprs.append(np.interp(grid, fpr, tpr))
    tprs = np.asarray(tprs)
    return grid, tprs.mean(axis=0), np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0)


# ===========================================================================
# 1. ROC curves with bootstrap band + operating point
# ===========================================================================

def figure_roc(y_true, prob_a, pred_a, prob_b, pred_b,
               name_a: str, name_b: str):
    fig, ax = plt.subplots(figsize=(COL_W, COL_W))
    for prob, pred, name, color, ls in [
        (prob_a, pred_a, name_a, COLOR_RESNET,    LS_RESNET),
        (prob_b, pred_b, name_b, COLOR_TWOSTREAM, LS_TWOSTREAM),
    ]:
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        grid, _, lo, hi = bootstrap_roc_band(y_true, prob, n_iter=500)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(fpr, tpr, color=color, linewidth=2.2, linestyle=ls,
                label=f"{name} (AUC = {auc:.4f})")
        op_fpr = float(((pred == 1) & (y_true == 0)).sum()) / float((y_true == 0).sum())
        op_tpr = float(((pred == 1) & (y_true == 1)).sum()) / float((y_true == 1).sum())
        ax.scatter([op_fpr], [op_tpr], color=color, s=80, zorder=5,
                   edgecolors="black", linewidths=1.0)

    ax.plot([0, 1], [0, 1], color=COLOR_CHANCE, linestyle="--",
            linewidth=1.2, label="Chance (AUC = 0.500)")
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.005)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "ROC curves on held-out test set (n = %d)" % len(y_true)
    )
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(OUT / "10_roc_curves.png")
    plt.close(fig)
    logger.info("Saved: 10_roc_curves.png")


# ===========================================================================
# 2. Precision-Recall curves
# ===========================================================================

def figure_pr(y_true, prob_a, pred_a, prob_b, pred_b,
              name_a: str, name_b: str):
    fig, ax = plt.subplots(figsize=(COL_W, COL_W))
    base_rate = float((y_true == 1).mean())
    for prob, pred, name, color, ls in [
        (prob_a, pred_a, name_a, COLOR_RESNET,    LS_RESNET),
        (prob_b, pred_b, name_b, COLOR_TWOSTREAM, LS_TWOSTREAM),
    ]:
        precision, recall, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        ax.plot(recall, precision, color=color, linewidth=2.2, linestyle=ls,
                label=f"{name} (AP = {ap:.4f})")
        op_p = precision_score(y_true, pred, zero_division=0)
        op_r = recall_score(y_true, pred, zero_division=0)
        ax.scatter([op_r], [op_p], color=color, s=80, zorder=5,
                   edgecolors="black", linewidths=1.0)

    ax.axhline(base_rate, color=COLOR_CHANCE, linestyle="--", linewidth=1.2,
               label=f"Chance (base rate = {base_rate:.3f})")
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(0.45, 1.005)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves on held-out test set (n = %d)" % len(y_true))
    ax.legend(loc="lower left")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(OUT / "10_pr_curves.png")
    plt.close(fig)
    logger.info("Saved: 10_pr_curves.png")


# ===========================================================================
# 3. Confusion matrices (counts + normalised, one figure per model)
# ===========================================================================

def figure_confusion(y_true, y_pred, name: str, slug: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    cmap_name = "Reds" if slug == "two_stream" else "Blues"

    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 4.2))
    titles = ["(a) counts", "(b) row-normalised"]
    matrices = [cm, cm_norm]
    fmts = ["d", ".3f"]
    label_names = ["Real", "Fake"]

    for ax, m, ttl, fmt in zip(axes, matrices, titles, fmts):
        vmax = cm.max() if "counts" in ttl else 1.0
        im = ax.imshow(m, cmap=cmap_name, vmin=0, vmax=vmax)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(label_names); ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(ttl)
        for i in range(2):
            for j in range(2):
                txt = format(m[i, j], fmt)
                color = "white" if m[i, j] > 0.55 * vmax else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontsize=14, fontweight="bold")
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_linewidth(0.8)
        cbar.ax.tick_params(width=0.8, labelsize=9)

    n = len(y_true)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    fig.suptitle(
        f"Confusion matrix: {name} (n = {n}). "
        f"acc = {acc:.4f}, precision = {prec:.4f}, "
        f"recall = {rec:.4f}, F1 = {f1:.4f}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / f"10_confusion_matrix_{slug}.png")
    plt.close(fig)
    logger.info("Saved: 10_confusion_matrix_%s.png", slug)


# ===========================================================================
# 4. Per-forgery-type recall (multi-class breakdown)
# ===========================================================================

def figure_per_ctype_recall(y_true, pred_a, pred_b, ctypes,
                            name_a: str, name_b: str):
    classes = [0, 1, 2]
    counts  = {c: int((ctypes == c).sum()) for c in classes}

    def recall_per_class(pred):
        out = {}
        for c in classes:
            mask = (ctypes == c)
            if not mask.any():
                out[c] = float("nan"); continue
            if c == 0:
                # recall on Real = TN rate = pred 0 given true 0
                out[c] = float((pred[mask] == 0).mean())
            else:
                # recall on this fake type = pred 1 given true 1 (label = 1) AND ctype matches
                out[c] = float((pred[mask] == 1).mean())
        return out

    rec_a = recall_per_class(pred_a)
    rec_b = recall_per_class(pred_b)

    width = 0.36
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(DBL_W, 5.0))
    bars_a = ax.bar(x - width/2, [rec_a[c] for c in classes],
                    width, color=COLOR_RESNET, edgecolor="black",
                    linewidth=1.0, label=name_a)
    bars_b = ax.bar(x + width/2, [rec_b[c] for c in classes],
                    width, color=COLOR_TWOSTREAM, edgecolor="black",
                    linewidth=1.0, label=name_b)

    for bars, vals in [(bars_a, rec_a), (bars_b, rec_b)]:
        for b, c in zip(bars, classes):
            v = vals[c]
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, v + 0.012,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

    xlabels = [f"{CTYPE_LABELS[c]}\n(n = {counts[c]})" for c in classes]
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Per-class recall on test set")
    ax.set_title(
        "Per-forgery-type recall (Real column is specificity, the TN rate)"
    )
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, color="0.85", alpha=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "10_per_ctype_recall.png")
    plt.close(fig)
    logger.info("Saved: 10_per_ctype_recall.png")


# ===========================================================================
# 5. Training curves (loss + val ROC-AUC + val recall)
# ===========================================================================

def _read_train_csv(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, list] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                out.setdefault(k, []).append(v)
    return {k: np.array([float(x) for x in v]) for k, v in out.items()}


def figure_training_curves():
    rn_csv = ROOT / "outputs" / "metrics" / "supervised_train.csv"
    ts_csv = ROOT / "research_outputs" / "08_two_stream_train.csv"
    if not rn_csv.exists() or not ts_csv.exists():
        logger.warning("Training CSVs missing — skipping training curves figure")
        return
    rn = _read_train_csv(rn_csv)
    ts = _read_train_csv(ts_csv)

    def _smooth(y: np.ndarray, w: int = 3) -> np.ndarray:
        """Centered moving-average with edge padding (preserves length)."""
        if w <= 1 or len(y) < w:
            return np.asarray(y, dtype=float)
        pad = w // 2
        ypad = np.pad(np.asarray(y, dtype=float), pad, mode="edge")
        kernel = np.ones(w) / w
        return np.convolve(ypad, kernel, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 4.8))

    best_ts  = int(ts["epoch"][int(np.argmax(ts["val_roc_auc"]))])
    best_rn  = int(rn["epoch"][int(np.argmax(rn["val_roc_auc"]))])
    n_epochs = int(max(rn["epoch"].max(), ts["epoch"].max()))

    def _plot_pair(ax, ykey, smooth_w=3):
        # ResNet50: faint raw + bold smoothed
        ax.plot(rn["epoch"], rn[ykey], color=COLOR_RESNET,
                linestyle="-", linewidth=1.0, alpha=0.30, zorder=2)
        ax.scatter(rn["epoch"], rn[ykey], color=COLOR_RESNET,
                   s=18, alpha=0.45, zorder=3, edgecolors="none")
        ax.plot(rn["epoch"], _smooth(rn[ykey], smooth_w),
                color=COLOR_RESNET, linestyle="-", linewidth=2.4,
                marker=MK_RESNET, markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.6,
                label="ResNet50", zorder=5)
        # Two-Stream
        ax.plot(ts["epoch"], ts[ykey], color=COLOR_TWOSTREAM,
                linestyle="-", linewidth=1.0, alpha=0.30, zorder=2)
        ax.scatter(ts["epoch"], ts[ykey], color=COLOR_TWOSTREAM,
                   s=18, alpha=0.45, zorder=3, edgecolors="none")
        ax.plot(ts["epoch"], _smooth(ts[ykey], smooth_w),
                color=COLOR_TWOSTREAM, linestyle="-", linewidth=2.4,
                marker=MK_TWOSTREAM, markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.6,
                label="Two-Stream", zorder=5)

    # (a) train loss
    ax = axes[0]
    _plot_pair(ax, "train_loss", smooth_w=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("(a) Training loss")
    ax.legend(loc="upper right")
    ax.set_xlim(0.5, n_epochs + 0.5)

    # (b) val ROC-AUC (zoomed to show convergence shape)
    ax = axes[1]
    _plot_pair(ax, "val_roc_auc", smooth_w=3)
    ax.axvspan(best_ts, n_epochs + 0.5, color="0.85", alpha=0.35,
               zorder=1, label="converged region")
    ax.axvline(best_ts, linestyle="--", color="black", linewidth=1.2,
               label=f"best epoch = {best_ts}", zorder=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation ROC-AUC")
    ax.set_title("(b) Validation ROC-AUC")
    ax.legend(loc="lower right")
    ax.set_xlim(0.5, n_epochs + 0.5)
    ax.set_ylim(0.55, 1.005)

    # (c) val recall
    ax = axes[2]
    _plot_pair(ax, "val_recall", smooth_w=3)
    ax.axvspan(best_ts, n_epochs + 0.5, color="0.85", alpha=0.35, zorder=1)
    ax.axvline(best_ts, linestyle="--", color="black", linewidth=1.2, zorder=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation recall on fakes")
    ax.set_title("(c) Validation recall on fakes")
    ax.legend(loc="lower right")
    ax.set_xlim(0.5, n_epochs + 0.5)
    ax.set_ylim(0.25, 1.03)

    fig.tight_layout()
    fig.savefig(OUT / "10_training_curves.png")
    plt.close(fig)
    logger.info("Saved: 10_training_curves.png")


def figure_lr_schedule():
    ts_csv = ROOT / "research_outputs" / "08_two_stream_train.csv"
    if not ts_csv.exists():
        logger.warning("Two-stream train CSV missing — skipping LR figure")
        return
    ts = _read_train_csv(ts_csv)
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.7))
    ax.plot(ts["epoch"], ts["lr"], color=COLOR_TWOSTREAM,
            linewidth=2.0, marker=MK_TWOSTREAM, markersize=7.0,
            markerfacecolor=COLOR_TWOSTREAM, markeredgecolor="black",
            markeredgewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title(
        f"Cosine LR schedule (peak {ts['lr'].max():.2e}, 20 epochs)"
    )
    fig.tight_layout()
    fig.savefig(OUT / "10_lr_schedule.png")
    plt.close(fig)
    logger.info("Saved: 10_lr_schedule.png")


# ===========================================================================
# 6. Threshold sweep (precision/recall/F1 vs decision threshold)
# ===========================================================================

def figure_threshold_sweep(y_true, y_prob, name: str):
    thresholds = np.linspace(0.01, 0.99, 99)
    p_, r_, f_ = [], [], []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        p_.append(precision_score(y_true, pred, zero_division=0))
        r_.append(recall_score(y_true, pred, zero_division=0))
        f_.append(f1_score(y_true, pred, zero_division=0))
    p_, r_, f_ = map(np.asarray, (p_, r_, f_))
    best_t = float(thresholds[int(np.argmax(f_))])
    best_f = float(f_.max())

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.78))
    ax.plot(thresholds, p_, label="Precision", color="#1f77b4",
            linestyle="-",  linewidth=2.0)
    ax.plot(thresholds, r_, label="Recall",    color="#2ca02c",
            linestyle="-", linewidth=2.0)
    ax.plot(thresholds, f_, label="F1",        color="#9467bd",
            linestyle="-",  linewidth=2.4)
    ax.axvline(0.5, color="0.6", linestyle="--", linewidth=1.0,
               label="default = 0.5")
    ax.axvline(best_t, color=COLOR_TWOSTREAM, linestyle="-", linewidth=1.4,
               label=f"best F1 = {best_t:.2f} ({best_f:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.85, 1.005)
    ax.set_xlabel("Decision threshold on p(fake)")
    ax.set_ylabel("Test-set metric")
    ax.set_title("Threshold sweep, Two-Stream model (y-axis zoomed)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "10_threshold_sweep.png")
    plt.close(fig)
    logger.info("Saved: 10_threshold_sweep.png")


# ===========================================================================
# 7. FFT spectrum visualisation (real vs fake log-magnitude)
# ===========================================================================

def _load_image_tensor(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    eval_tf = get_simple_eval_transform(image_size)
    with Image.open(path).convert("RGB") as im:
        t = eval_tf(im).unsqueeze(0).to(device)
    return t


def figure_fft_spectrum(records: list[dict], test_idx: list[int],
                        device: torch.device, image_size: int):
    # Pick 3 reals, 3 inpaint, 3 crop_and_replace
    grouped: dict[str, list[int]] = {"real": [], "Inpaint_and_Rewrite": [], "Crop_and_Replace": []}
    for i in test_idx:
        ct = records[i].get("ctype") or "real"
        if ct in grouped and len(grouped[ct]) < 3:
            grouped[ct].append(i)

    cols = max(len(v) for v in grouped.values())
    fig, axes = plt.subplots(6, cols, figsize=(DBL_W, DBL_W * 1.10))
    fig.suptitle(
        "Frequency-domain signature: log(1 + |FFT|), DC centred (channel mean)"
    )

    for row_group, (ct, idxs) in enumerate(grouped.items()):
        for col, gi in enumerate(idxs):
            rec = records[gi]
            t = _load_image_tensor(rec["path"], image_size, device)
            with torch.no_grad():
                spec = _compute_fft_magnitude(t)  # (1, 3, H, W)
            spec_np = spec.squeeze(0).mean(dim=0).cpu().numpy()
            rgb = (t.squeeze(0).cpu().numpy() *
                   IMAGENET_STD.reshape(3, 1, 1) + IMAGENET_MEAN.reshape(3, 1, 1))
            rgb = np.clip(rgb.transpose(1, 2, 0), 0, 1)

            r_top = 2 * row_group
            r_bot = 2 * row_group + 1
            ax_top = axes[r_top, col] if cols > 1 else axes[r_top]
            ax_bot = axes[r_bot, col] if cols > 1 else axes[r_bot]

            ax_top.imshow(rgb)
            ax_top.axis("off")
            if col == 0:
                ax_top.text(-0.12, 0.5, f"{ct}\n(image)",
                            transform=ax_top.transAxes,
                            ha="right", va="center", fontsize=10,
                            fontweight="bold", rotation=90)
            ax_top.set_title(rec["path"].stem, fontsize=9)

            ax_bot.imshow(spec_np, cmap="viridis")
            ax_bot.axis("off")
            if col == 0:
                ax_bot.text(-0.12, 0.5, f"{ct}\n(FFT)",
                            transform=ax_bot.transAxes,
                            ha="right", va="center", fontsize=10,
                            fontweight="bold", rotation=90)

        # Hide unused columns in this group
        for col in range(len(idxs), cols):
            axes[2*row_group, col].axis("off")
            axes[2*row_group + 1, col].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "10_fft_spectrum.png")
    plt.close(fig)
    logger.info("Saved: 10_fft_spectrum.png")


# ===========================================================================
# 8. Grad-CAM helpers — single-stream and two-stream variants
# ===========================================================================

class GradCAM:
    """Grad-CAM hooked on a chosen nn.Module that produces a 4D feature map."""
    def __init__(self, target: nn.Module):
        self.target = target
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd = target.register_forward_hook(self._save_act)
        self._bwd = target.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, output):
        self.activations = output.detach()

    def _save_grad(self, _, gin, gout):
        self.gradients = gout[0].detach()

    def remove(self):
        self._fwd.remove(); self._bwd.remove()

    def heatmap(self, model: nn.Module, x: torch.Tensor, target_class: int) -> np.ndarray:
        model.eval()
        x = x.requires_grad_(True)
        score = model(x)[:, target_class].sum()
        model.zero_grad()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze(1).cpu().numpy()
        cmin = cam.reshape(cam.shape[0], -1).min(axis=1)[:, None, None]
        cmax = cam.reshape(cam.shape[0], -1).max(axis=1)[:, None, None]
        return (cam - cmin) / (cmax - cmin + 1e-8)


def denormalise(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().numpy() * IMAGENET_STD.reshape(3, 1, 1) + IMAGENET_MEAN.reshape(3, 1, 1)
    return np.clip(img.transpose(1, 2, 0), 0, 1)


# ===========================================================================
# 8a. Grad-CAM on correctly-classified positives, one example per ctype
# ===========================================================================

def figure_gradcam_positives_per_ctype(
    resnet: SupervisedResNet50, two_stream: TwoStreamForgeryNet,
    records: list[dict], test_idx: list[int],
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    prob_a: np.ndarray, prob_b: np.ndarray,
    device: torch.device, image_size: int, n_per: int = 3,
):
    """For each forgery type, pick up to `n_per` test fakes that BOTH models
    predict correctly (TPs), and render Grad-CAM from each model side-by-side.
    """
    eval_tf = get_simple_eval_transform(image_size)
    targets = {"Inpaint_and_Rewrite": [], "Crop_and_Replace": []}
    for sub, gi in enumerate(test_idx):
        rec = records[gi]
        ct = rec.get("ctype") or "real"
        if ct not in targets:
            continue
        if y_true[sub] == 1 and pred_a[sub] == 1 and pred_b[sub] == 1:
            if len(targets[ct]) < n_per:
                targets[ct].append((sub, rec))

    cam_rn = GradCAM(resnet.net.layer4)
    # Two-Stream: hook on RGB-branch ResNet50 layer4 (the deepest spatial map
    # before transformer fusion). That is `two_stream.rgb_features[-1]`.
    cam_ts = GradCAM(two_stream.rgb_features[-1])

    rows = sum(len(v) for v in targets.values())
    if rows == 0:
        logger.warning("No positives common to both models — skipping figure")
        cam_rn.remove(); cam_ts.remove()
        return
    fig, axes = plt.subplots(rows, 3, figsize=(DBL_W, rows * (DBL_W / 3.0) * 1.05))
    if rows == 1:
        axes = axes.reshape(1, 3)

    col_titles = ["Input", "ResNet50 (layer4)", "Two-Stream RGB branch (layer4)"]
    for c, ct_title in enumerate(col_titles):
        axes[0, c].set_title(ct_title, fontsize=11, fontweight="bold")

    r = 0
    for ct, lst in targets.items():
        for sub, rec in lst:
            t = _load_image_tensor(rec["path"], image_size, device)
            heat_a = cam_rn.heatmap(resnet, t.clone(), target_class=1)[0]
            heat_b = cam_ts.heatmap(two_stream, t.clone(), target_class=1)[0]
            rgb = denormalise(t.squeeze(0))

            axes[r, 0].imshow(rgb)
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
            axes[r, 0].set_ylabel(f"{ct}\n{rec['path'].stem}\n p(fake) = {prob_b[sub]:.3f}",
                                  fontsize=9, rotation=0, labelpad=46, va="center", ha="right")
            for spine in axes[r, 0].spines.values():
                spine.set_linewidth(0.8)
            axes[r, 1].imshow(rgb)
            axes[r, 1].imshow(heat_a, cmap="jet", alpha=0.45)
            axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
            for spine in axes[r, 1].spines.values():
                spine.set_linewidth(0.8)
            axes[r, 2].imshow(rgb)
            axes[r, 2].imshow(heat_b, cmap="jet", alpha=0.45)
            axes[r, 2].set_xticks([]); axes[r, 2].set_yticks([])
            for spine in axes[r, 2].spines.values():
                spine.set_linewidth(0.8)
            r += 1

    fig.suptitle(
        "Grad-CAM on correctly-classified fakes (both models agree)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "10_gradcam_correct_per_ctype.png")
    plt.close(fig)
    cam_rn.remove(); cam_ts.remove()
    logger.info("Saved: 10_gradcam_correct_per_ctype.png")


# ===========================================================================
# 8b. Side-by-side Grad-CAM on cases ResNet50 misses but Two-Stream catches
# ===========================================================================

def figure_gradcam_resnet_vs_two_stream(
    resnet: SupervisedResNet50, two_stream: TwoStreamForgeryNet,
    records: list[dict], test_idx: list[int],
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    prob_a: np.ndarray, prob_b: np.ndarray,
    device: torch.device, image_size: int, n_max: int = 5,
):
    """Pick test fakes that ResNet50 wrongly calls real but Two-Stream
    correctly catches. Render Grad-CAM from each model side-by-side."""
    target_subs = []
    for sub in range(len(y_true)):
        if y_true[sub] == 1 and pred_a[sub] == 0 and pred_b[sub] == 1:
            target_subs.append(sub)
    if not target_subs:
        logger.info("No A-wrong / B-right cases — drawing simple text figure")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5,
                "No cases where ResNet50 missed but Two-Stream caught.\n"
                "(McNemar b-cell empty: skip side-by-side Grad-CAM.)",
                ha="center", va="center", fontsize=12, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / "10_gradcam_resnet_vs_two_stream.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        return

    # rank by how strongly ResNet50 was wrong (lowest p(fake) first)
    target_subs.sort(key=lambda s: prob_a[s])
    target_subs = target_subs[:n_max]

    cam_rn = GradCAM(resnet.net.layer4)
    cam_ts = GradCAM(two_stream.rgb_features[-1])

    n = len(target_subs)
    fig, axes = plt.subplots(n, 3, figsize=(DBL_W, n * (DBL_W / 3.0) * 1.05))
    if n == 1:
        axes = axes.reshape(1, 3)

    col_titles = [
        "Input",
        "ResNet50: predicted REAL (wrong)",
        "Two-Stream: predicted FAKE (correct)",
    ]
    for c, ct_title in enumerate(col_titles):
        axes[0, c].set_title(ct_title, fontsize=11, fontweight="bold")

    for r, sub in enumerate(target_subs):
        rec = records[test_idx[sub]]
        t = _load_image_tensor(rec["path"], image_size, device)
        heat_a = cam_rn.heatmap(resnet, t.clone(), target_class=1)[0]
        heat_b = cam_ts.heatmap(two_stream, t.clone(), target_class=1)[0]
        rgb = denormalise(t.squeeze(0))
        ct = rec.get("ctype") or "real"

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"{ct}\n{rec['path'].stem}",
                              fontsize=9, rotation=0, labelpad=46,
                              va="center", ha="right")
        axes[r, 1].imshow(rgb); axes[r, 1].imshow(heat_a, cmap="jet", alpha=0.45)
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
        axes[r, 1].text(0.02, 0.96, f"p(fake) = {prob_a[sub]:.3f}",
                        transform=axes[r, 1].transAxes,
                        ha="left", va="top", fontsize=9, color="white",
                        bbox=dict(facecolor="black", edgecolor="none",
                                  alpha=0.6, pad=2.5))
        axes[r, 2].imshow(rgb); axes[r, 2].imshow(heat_b, cmap="jet", alpha=0.45)
        axes[r, 2].set_xticks([]); axes[r, 2].set_yticks([])
        axes[r, 2].text(0.02, 0.96, f"p(fake) = {prob_b[sub]:.3f}",
                        transform=axes[r, 2].transAxes,
                        ha="left", va="top", fontsize=9, color="white",
                        bbox=dict(facecolor="black", edgecolor="none",
                                  alpha=0.6, pad=2.5))
        for c in range(3):
            for spine in axes[r, c].spines.values():
                spine.set_linewidth(0.8)

    fig.suptitle(
        "Cases where the FFT branch makes the difference (ResNet50 missed; Two-Stream caught)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "10_gradcam_resnet_vs_two_stream.png")
    plt.close(fig)
    cam_rn.remove(); cam_ts.remove()
    logger.info("Saved: 10_gradcam_resnet_vs_two_stream.png")


# ===========================================================================
# Driver
# ===========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()
    logger.info("Device: %s", device)

    test_loader, records, test_idx = make_test_loader(
        args.image_size, args.batch_size, args.seed)
    n = len(test_idx)
    logger.info("Test set size: %d", n)

    # ── Load models ──────────────────────────────────────────────────────
    resnet = SupervisedResNet50(num_classes=2, dropout=0.2).to(device)
    rn_ckpt = ROOT / "models" / "supervised_resnet50_best.pth"
    rn_state = torch.load(rn_ckpt, map_location=device)
    resnet.load_state_dict(rn_state["model_state_dict"], strict=False)
    logger.info("Loaded ResNet50 from %s", rn_ckpt)

    two_stream = TwoStreamForgeryNet(num_classes=2, use_transformer=True).to(device)
    ts_ckpt = ROOT / "models" / "two_stream_best.pth"
    ts_state = torch.load(ts_ckpt, map_location=device)
    two_stream.load_state_dict(ts_state["model_state_dict"], strict=False)
    logger.info("Loaded Two-Stream from %s", ts_ckpt)

    # ── Predict on test set ──────────────────────────────────────────────
    y_true, prob_a, pred_a = predict_test(resnet, test_loader, device)
    _,      prob_b, pred_b = predict_test(two_stream, test_loader, device)

    # build ctype array aligned to test_idx
    ctype_map = {"real": 0, "Inpaint_and_Rewrite": 1, "Crop_and_Replace": 2}
    ctypes = np.array([ctype_map.get(records[i].get("ctype") or "real", 0)
                       for i in test_idx])

    name_a = "ResNet50 (single-stream)"
    name_b = "Two-Stream RGB+FFT + Transformer"

    # ── 1. ROC ───────────────────────────────────────────────────────────
    figure_roc(y_true, prob_a, pred_a, prob_b, pred_b, name_a, name_b)
    # ── 2. PR ────────────────────────────────────────────────────────────
    figure_pr(y_true, prob_a, pred_a, prob_b, pred_b, name_a, name_b)
    # ── 3. Confusion matrices ────────────────────────────────────────────
    figure_confusion(y_true, pred_a, name_a, "resnet50")
    figure_confusion(y_true, pred_b, name_b, "two_stream")
    # ── 4. Per-ctype recall ──────────────────────────────────────────────
    figure_per_ctype_recall(y_true, pred_a, pred_b, ctypes, name_a, name_b)
    # ── 5. Training curves + LR ─────────────────────────────────────────
    figure_training_curves()
    figure_lr_schedule()
    # ── 6. Threshold sweep on Two-Stream ────────────────────────────────
    figure_threshold_sweep(y_true, prob_b, name_b)
    # ── 7. FFT spectrum panel ───────────────────────────────────────────
    figure_fft_spectrum(records, test_idx, device, args.image_size)
    # ── 8. Grad-CAM positives + comparison ──────────────────────────────
    figure_gradcam_positives_per_ctype(
        resnet, two_stream, records, test_idx,
        y_true, pred_a, pred_b, prob_a, prob_b,
        device, args.image_size, n_per=2,
    )
    figure_gradcam_resnet_vs_two_stream(
        resnet, two_stream, records, test_idx,
        y_true, pred_a, pred_b, prob_a, prob_b,
        device, args.image_size, n_max=5,
    )

    logger.info("All Stage-10 figures rendered to %s", OUT)


if __name__ == "__main__":
    main()
