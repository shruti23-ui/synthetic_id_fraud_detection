"""
generate_supervised_plots.py
----------------------------
Produces thesis-ready plots specifically for the supervised ResNet50 model:

  outputs/plots/sup_training_curve.png    train loss + val ROC-AUC over 15 epochs
  outputs/plots/sup_roc_curve.png         ROC curve on the held-out test set
  outputs/plots/sup_pr_curve.png          Precision-recall curve on the test set
  outputs/plots/sup_confusion_matrix.png  raw + normalised confusion matrices

Run after the supervised stage has produced:
  models/supervised_resnet50_best.pth
  outputs/metrics/supervised_train.csv
"""

# IMPORTANT: sklearn before torch on Windows to avoid libomp DLL crash
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.amp.autocast_mode import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent))

from supervised_finetune import (
    CONFIG as SUP_CFG,
    SupervisedResNet50,
    make_loaders,
)
from utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PLOTS_DIR   = Path("outputs/plots")
METRICS_DIR = Path("outputs/metrics")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Training curve (loss + val ROC-AUC over epochs)
# ---------------------------------------------------------------------------

def plot_training_curve() -> Path | None:
    csv_path = METRICS_DIR / "supervised_train.csv"
    if not csv_path.exists():
        logger.warning("Supervised train CSV not found at %s — skipping.", csv_path)
        return None

    df = pd.read_csv(csv_path)
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    # Left axis: train loss
    color_loss = "#1565C0"
    ax1.plot(df["epoch"], df["train_loss"], color=color_loss, linewidth=2.5,
             marker="o", markersize=5, label="Train loss")
    ax1.fill_between(df["epoch"], df["train_loss"], alpha=0.10, color=color_loss)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Train loss (cross-entropy)", color=color_loss, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Right axis: val ROC-AUC
    ax2 = ax1.twinx()
    color_auc = "#D32F2F"
    ax2.plot(df["epoch"], df["val_roc_auc"], color=color_auc, linewidth=2.5,
             marker="s", markersize=5, label="Val ROC-AUC")
    ax2.set_ylabel("Validation ROC-AUC", color=color_auc, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_auc)
    ax2.set_ylim(0.4, 1.02)

    # Mark best epoch
    best_idx = df["val_roc_auc"].idxmax()
    best_ep  = int(df.loc[best_idx, "epoch"])
    best_auc = df.loc[best_idx, "val_roc_auc"]
    ax2.axvline(best_ep, linestyle=":", color="black", linewidth=1)
    ax2.annotate(
        f"best ckpt\nepoch {best_ep}\nROC-AUC {best_auc:.4f}",
        xy=(best_ep, best_auc),
        xytext=(best_ep - 4, best_auc - 0.10),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )

    fig.suptitle("Supervised ResNet50 — Training Convergence",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = PLOTS_DIR / "sup_training_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# 2-4. Test-set evaluation: ROC, PR, confusion matrix
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_test_predictions(config: dict = SUP_CFG):
    """Run the saved best checkpoint over the test loader and return
    (y_true, y_prob_fraud, y_pred)."""
    device = get_device()
    _, _, test_loader, _ = make_loaders(config)

    model = SupervisedResNet50(num_classes=2, dropout=0.2).to(device)
    ckpt_path = Path(config["best_path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best checkpoint missing: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint epoch=%s val ROC-AUC=%.4f",
                ckpt.get("epoch", "?"),
                ckpt.get("val_metrics", {}).get("roc_auc", float("nan")))

    use_amp = bool(config.get("use_amp", True)) and device.type == "cuda"
    all_probs, all_preds, all_labels = [], [], []
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        if use_amp:
            with autocast(device_type="cuda"):
                logits = model(images).float()
        else:
            logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return (
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        np.concatenate(all_preds),
    )


def plot_test_roc(y_true: np.ndarray, y_prob: np.ndarray) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1565C0", lw=2.8,
            label=f"Supervised ResNet50  (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.10, color="#1565C0")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Supervised ResNet50 — Test Set ROC Curve",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.01)

    out = PLOTS_DIR / "sup_roc_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s  (test ROC-AUC = %.4f)", out, roc_auc)
    return out


def plot_test_pr(y_true: np.ndarray, y_prob: np.ndarray) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = auc(recall[::-1], precision[::-1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.step(recall, precision, color="#D32F2F", lw=2.8, where="post",
            label=f"Supervised ResNet50  (AP = {ap:.4f})")
    ax.fill_between(recall, precision, step="post", alpha=0.10, color="#D32F2F")
    # Random baseline = positive rate
    pos_rate = float((y_true == 1).mean())
    ax.axhline(pos_rate, linestyle="--", color="gray", lw=1,
               label=f"Random classifier (AP = {pos_rate:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Supervised ResNet50 — Test Set Precision-Recall Curve",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.01)

    out = PLOTS_DIR / "sup_pr_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s  (test AP = %.4f)", out, ap)
    return out


def plot_test_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    class_names = ["Real (0)", "Fake (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, title, fmt in [
        (axes[0], cm,      "Confusion Matrix (Counts)",     "d"),
        (axes[1], cm_norm, "Confusion Matrix (Normalised)", ".3f"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.6, cbar_kws={"shrink": 0.85},
            annot_kws={"fontsize": 13},
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True",      fontsize=12)

    fig.suptitle("Supervised ResNet50 — Test Set Confusion Matrix",
                 fontsize=14, fontweight="bold", y=1.02)
    out = PLOTS_DIR / "sup_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Master entry-point
# ---------------------------------------------------------------------------

def main():
    plot_training_curve()
    y_true, y_prob, y_pred = collect_test_predictions()
    plot_test_roc(y_true, y_prob)
    plot_test_pr(y_true, y_prob)
    plot_test_confusion(y_true, y_pred)
    logger.info("Supervised plots done.")


if __name__ == "__main__":
    main()
