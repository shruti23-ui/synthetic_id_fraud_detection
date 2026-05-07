"""
visualization.py
----------------
Publication-quality visualisations for the thesis:

  1. Training loss curve
  2. t-SNE embedding plot
  3. UMAP embedding plot
  4. PCA embedding plot
  5. ROC curve
  6. Precision-Recall curve
  7. Confusion matrix (raw + normalised)
  8. Class distribution bar chart

Usage (standalone):
    python src/visualization.py

Outputs saved to:  outputs/plots/
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — safe on servers without a display

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import load_processed_arrays

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

PLOTS_DIR = Path("outputs/plots")
PALETTE   = {0: "#2196F3", 1: "#F44336"}   # blue=legit, red=fraud
LABEL_MAP = {0: "Legitimate", 1: "Fraud"}

def _set_style():
    plt.rcParams.update({
        "figure.dpi":        150,
        "figure.facecolor":  "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "legend.fontsize":   11,
    })

_set_style()


# ---------------------------------------------------------------------------
# 1. Training loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(metrics_csv: str = "outputs/metrics/train_loss.csv") -> Path:
    """Plot the contrastive training loss over epochs."""
    csv_path = Path(metrics_csv)
    if not csv_path.exists():
        logger.warning("Loss CSV not found at %s — skipping loss curve.", csv_path)
        return None

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["epoch"], df["loss"], color="#1565C0", linewidth=2.5, label="Train Loss")
    ax.fill_between(df["epoch"], df["loss"], alpha=0.12, color="#1565C0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent Loss")
    ax.set_title("SimCLR Contrastive Training Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    out = PLOTS_DIR / "loss_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# 2. t-SNE embedding plot
# ---------------------------------------------------------------------------

def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, n_samples: int = 2000) -> Path:
    """2-D t-SNE scatter plot of the learned embedding space."""
    if len(embeddings) > n_samples:
        idx = np.random.default_rng(0).choice(len(embeddings), n_samples, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]

    logger.info("Running t-SNE on %d samples …", len(embeddings))
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl, name in LABEL_MAP.items():
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=PALETTE[lbl], label=name, s=18, alpha=0.7, edgecolors="none",
        )
    ax.set_title("t-SNE: SimCLR Embedding Space")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=2)

    out = PLOTS_DIR / "tsne_embeddings.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# 3. UMAP embedding plot
# ---------------------------------------------------------------------------

def plot_umap(embeddings: np.ndarray, labels: np.ndarray, n_samples: int = 2000) -> Path:
    """2-D UMAP scatter plot of the learned embedding space."""
    if not UMAP_AVAILABLE:
        logger.warning("umap-learn not installed — skipping UMAP plot.")
        return None

    if len(embeddings) > n_samples:
        idx = np.random.default_rng(1).choice(len(embeddings), n_samples, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]

    logger.info("Running UMAP on %d samples …", len(embeddings))
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl, name in LABEL_MAP.items():
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=PALETTE[lbl], label=name, s=18, alpha=0.7, edgecolors="none",
        )
    ax.set_title("UMAP: SimCLR Embedding Space")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(markerscale=2)

    out = PLOTS_DIR / "umap_embeddings.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# 4. PCA embedding plot
# ---------------------------------------------------------------------------

def plot_pca(embeddings: np.ndarray, labels: np.ndarray) -> Path:
    """2-D PCA scatter + explained-variance subplot."""
    pca = PCA(n_components=min(50, embeddings.shape[1]), random_state=42)
    pca.fit(embeddings)
    coords = pca.transform(embeddings)[:, :2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter in PC1 / PC2
    ax = axes[0]
    for lbl, name in LABEL_MAP.items():
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1], c=PALETTE[lbl], label=name, s=18, alpha=0.7, edgecolors="none")
    ax.set_title("PCA: Principal Components 1 & 2")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(markerscale=2)

    # Right: cumulative explained variance
    ax2 = axes[1]
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
    ax2.plot(range(1, len(cum_var) + 1), cum_var, color="#1565C0", linewidth=2)
    ax2.axhline(90, linestyle="--", color="gray", linewidth=1, label="90% threshold")
    ax2.set_title("Cumulative Explained Variance")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    out = PLOTS_DIR / "pca_embeddings.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# 5. ROC Curve
# ---------------------------------------------------------------------------

def plot_roc(roc_csv: str = "outputs/metrics/roc_curve_data.csv") -> Path:
    """Plot ROC curve from pre-computed data or from raw scores."""
    csv_path = Path(roc_csv)
    if not csv_path.exists():
        logger.warning("ROC data not found at %s — skipping.", csv_path)
        return None

    df = pd.read_csv(csv_path)
    roc_auc = auc(df["fpr"], df["tpr"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(df["fpr"], df["tpr"], color="#1565C0", lw=2.5, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier")
    ax.fill_between(df["fpr"], df["tpr"], alpha=0.10, color="#1565C0")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)

    out = PLOTS_DIR / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# 6. Precision-Recall Curve
# ---------------------------------------------------------------------------

def plot_pr_curve(pr_csv: str = "outputs/metrics/pr_curve_data.csv") -> Path:
    """Plot Precision-Recall curve from pre-computed data."""
    csv_path = Path(pr_csv)
    if not csv_path.exists():
        logger.warning("PR data not found at %s — skipping.", csv_path)
        return None

    df = pd.read_csv(csv_path)
    ap = auc(df["recall"].iloc[::-1], df["precision"].iloc[::-1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.step(df["recall"], df["precision"], color="#D32F2F", lw=2.5, where="post",
            label=f"PR curve (AP = {ap:.4f})")
    ax.fill_between(df["recall"], df["precision"], step="post", alpha=0.10, color="#D32F2F")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    out = PLOTS_DIR / "pr_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# 7. Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm_npy: str = "outputs/metrics/confusion_matrix.npy") -> Path:
    """Plot raw and normalised confusion matrices side-by-side."""
    npy_path = Path(cm_npy)
    norm_path = Path(cm_npy.replace(".npy", "_norm.npy"))

    if not npy_path.exists():
        logger.warning("Confusion matrix not found at %s — skipping.", npy_path)
        return None

    cm      = np.load(npy_path)
    cm_norm = np.load(norm_path) if norm_path.exists() else cm.astype(float) / cm.sum(axis=1, keepdims=True)
    class_names = ["Legitimate", "Fraud"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, title, fmt in [
        (axes[0], cm,      "Confusion Matrix (Counts)",       "d"),
        (axes[1], cm_norm, "Confusion Matrix (Normalised)",   ".2f"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    out = PLOTS_DIR / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# 8. Class distribution bar chart
# ---------------------------------------------------------------------------

def plot_class_distribution(labels: np.ndarray) -> Path:
    """Bar chart of class distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    names  = [LABEL_MAP.get(int(u), str(u)) for u in unique]
    colors = [PALETTE.get(int(u), "#9E9E9E") for u in unique]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.8)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{count:,}\n({count/len(labels)*100:.1f}%)",
            ha="center", va="bottom", fontsize=11,
        )

    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts) * 1.2)

    out = PLOTS_DIR / "class_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# 9. Embedding similarity heatmap
# ---------------------------------------------------------------------------

def plot_embedding_similarity_heatmap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_per_class: int = 30,
) -> Path:
    """Cosine-similarity heatmap of n_per_class samples from each class.

    Strong block-diagonal structure (high intra-class similarity, low cross-class)
    indicates the contrastive embedding has learned a discriminative representation.
    """
    rng = np.random.default_rng(7)
    legit_idx = np.where(labels == 0)[0]
    fraud_idx = np.where(labels == 1)[0]

    n0 = min(n_per_class, len(legit_idx))
    n1 = min(n_per_class, len(fraud_idx))
    if n0 == 0 or n1 == 0:
        logger.warning("Need both classes for similarity heatmap — skipping.")
        return None

    sel = np.concatenate([
        rng.choice(legit_idx, n0, replace=False),
        rng.choice(fraud_idx, n1, replace=False),
    ])
    emb = embeddings[sel]
    norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb_n = emb / norm
    sim = emb_n @ emb_n.T

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    ax.axhline(n0 - 0.5, color="black", linewidth=1)
    ax.axvline(n0 - 0.5, color="black", linewidth=1)
    ax.set_xticks([n0 / 2, n0 + n1 / 2])
    ax.set_xticklabels(["Legitimate", "Fraud"])
    ax.set_yticks([n0 / 2, n0 + n1 / 2])
    ax.set_yticklabels(["Legitimate", "Fraud"], rotation=90, va="center")
    ax.set_title("Embedding Cosine Similarity Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")

    out = PLOTS_DIR / "embedding_similarity_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# 10. Baseline vs SimCLR comparison bar chart
# ---------------------------------------------------------------------------

def plot_baseline_comparison(
    csv_path: str = "outputs/metrics/baseline_vs_simclr.csv",
) -> Path:
    """Grouped bar chart comparing raw ResNet vs SimCLR embeddings."""
    p = Path(csv_path)
    if not p.exists():
        logger.warning("Baseline comparison CSV not found at %s — skipping.", p)
        return None

    df = pd.read_csv(p)
    metrics = ["test_accuracy", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc"]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        logger.warning("No test_* metric columns in %s — skipping baseline plot.", p)
        return None
    model_pivot = df.pivot_table(index="model", columns="features", values=metrics)

    if model_pivot.empty:
        logger.warning("Empty baseline comparison — skipping plot.")
        return None

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = model_pivot[metric]
        sub.plot(kind="bar", ax=ax, color=["#90A4AE", "#1565C0"], edgecolor="white")
        ax.set_title(metric.replace("test_", "").upper())
        ax.set_xlabel("")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Baseline (Raw ResNet) vs. SimCLR Contrastive Embeddings",
                 fontsize=14, fontweight="bold", y=1.02)
    out = PLOTS_DIR / "baseline_vs_simclr.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Master function: generate all plots
# ---------------------------------------------------------------------------

def generate_all_plots(split: str = "train") -> None:
    """Generate all visualisations in one call."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    embeddings, labels = load_processed_arrays(split=split)
    logger.info("Generating all plots for split '%s' (%d samples) …", split, len(embeddings))

    plot_loss_curve()
    plot_class_distribution(labels)
    plot_pca(embeddings, labels)
    plot_tsne(embeddings, labels)
    plot_umap(embeddings, labels)
    plot_roc()
    plot_pr_curve()
    plot_confusion_matrix()
    plot_embedding_similarity_heatmap(embeddings, labels)
    plot_baseline_comparison()

    logger.info("All visualisations saved to %s/", PLOTS_DIR)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all_plots()
    logger.info("visualization.py finished.")
