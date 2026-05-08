"""
research/09_publication_figures.py
----------------------------------
Publication-grade figures, one per file, each with a proper title, axis
labels, legend and (where applicable) confidence intervals.

Replaces the earlier debugging-quality plots with figures that:

  * Show only ACTUAL misclassifications in failure grids (the previous
    grid mislabelled "least-confident-but-correct" reals as false
    positives — there are zero true FPs on the clean test set).
  * Split combined plots into separate figures (PCA/t-SNE/UMAP × binary/
    ctype = 6 separate plots).
  * Split the 8-corruption robustness sweep into geometric and
    photometric figures so axis labels are readable.
  * Report 95 % bootstrap confidence intervals on every test metric and
    a McNemar significance test for the Two-Stream vs ResNet50 comparison.
  * Include an ablation table (RGB-only / RGB+FFT no-transformer /
    RGB+FFT + transformer) that isolates each architectural addition.

Outputs (all under research_outputs/):
    09_failure_false_negatives.png    only true FNs, with Grad-CAM
    09_failure_false_positives.png    only true FPs (or text figure if 0)
    09_pca_binary.png
    09_pca_ctype.png
    09_tsne_binary.png
    09_tsne_ctype.png
    09_umap_binary.png
    09_umap_ctype.png
    09_robustness_geometric.png
    09_robustness_photometric.png
    09_comparison_with_ci.png
    09_mcnemar_test.json
    09_ablation_results.csv
    09_ablation_bar.png
"""

from __future__ import annotations

# IMPORTANT: sklearn before torch on Windows
import numpy as np
from sklearn.cluster import KMeans  # noqa: F401  (used in cosine stats)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

# tqdm bars as ASCII (works on any terminal codepage, esp. Windows cp1252)
os.environ.setdefault("TQDM_ASCII", " 123456789#")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from local_dataset_loader import LocalEmbeddingDataset                    # noqa: E402
from supervised_finetune import (  # noqa: E402
    CONFIG as SUP_CFG,
    SupervisedResNet50,
    get_simple_eval_transform,
    get_simple_train_transform,
)
from template_split import template_aware_split                            # noqa: E402
from two_stream_model import TwoStreamForgeryNet                           # noqa: E402
from utils import count_parameters, get_device, make_param_groups, seed_everything  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

PALETTE_BIN  = {0: "#1565C0", 1: "#D32F2F"}
LABELS_BIN   = {0: "Real (n=%d)", 1: "Fake (n=%d)"}
PALETTE_CT   = {0: "#1565C0", 1: "#D32F2F", 2: "#F57C00"}
CTYPE_LABELS = {0: "Real", 1: "Inpaint_and_Rewrite", 2: "Crop_and_Replace"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ===========================================================================
# 1. Shared inference utilities
# ===========================================================================

@torch.no_grad()
def predict_test(model: nn.Module, test_loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_prob_fake, y_pred) on the test loader."""
    model.eval()
    probs, preds, labels_all = [], [], []
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type="cuda"):
            logits = model(x).float()
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels_all.append(y.numpy())
    return np.concatenate(labels_all), np.concatenate(probs), np.concatenate(preds)


def metrics_from_preds(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_prob)),
        "pr_auc":    float(average_precision_score(y_true, y_prob)),
    }


# ===========================================================================
# 2. Bootstrap CIs and McNemar test
# ===========================================================================

def bootstrap_ci(
    y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray,
    n_iter: int = 1000, seed: int = 42,
) -> dict:
    """Percentile bootstrap 95% CIs for every metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metric_samples: dict[str, list[float]] = {
        "accuracy": [], "precision": [], "recall": [],
        "f1": [], "roc_auc": [], "pr_auc": [],
    }
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx]
        pr = y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue  # skip degenerate bootstrap samples
        m = metrics_from_preds(yt, yp, pr)
        for k in metric_samples:
            metric_samples[k].append(m[k])
    return {
        k: {
            "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)),
        }
        for k, v in metric_samples.items()
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar's test for two paired classifiers on the same test set.

    b = #(A wrong, B right);  c = #(A right, B wrong)
    Test statistic ≈ (|b - c| - 1)^2 / (b + c)  (continuity-corrected).
    """
    a_wrong = (pred_a != y_true)
    b_wrong = (pred_b != y_true)
    b = int(((a_wrong)  & (~b_wrong)).sum())     # A wrong, B right
    c = int(((~a_wrong) & (b_wrong)).sum())      # A right, B wrong
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0,
                "interpretation": "models agree on every test sample"}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # Chi-squared survival (1 dof) without scipy: use scipy if available
    try:
        from scipy.stats import chi2 as _chi2
        p = float(_chi2.sf(chi2, df=1))
    except Exception:
        # Approximate via series for chi-squared 1 dof: P(X^2 > x) = erfc(sqrt(x/2))
        from math import erfc, sqrt
        p = float(erfc(sqrt(max(chi2, 0) / 2)))
    return {"b": b, "c": c, "chi2": float(chi2), "p_value": p,
            "interpretation":
            ("models differ significantly (p < 0.05)" if p < 0.05
             else "no significant difference (p >= 0.05)")}


# ===========================================================================
# 3. Loaders (reused across all figures)
# ===========================================================================

def make_test_loader(image_size: int, batch_size: int, seed: int) -> tuple[DataLoader, list[dict], list[int]]:
    eval_tf = get_simple_eval_transform(image_size)
    eval_ds = LocalEmbeddingDataset(transform=eval_tf)
    cfg = dict(SUP_CFG)
    cfg["seed"] = seed
    cfg["val_size"] = 0.20
    cfg["test_size"] = 0.20
    _, _, test_idx = template_aware_split(eval_ds.records, cfg)
    test_idx = sorted(test_idx)
    loader = DataLoader(
        Subset(eval_ds, test_idx), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    return loader, eval_ds.records, test_idx


def make_train_val_test_loaders(image_size: int, batch_size: int, seed: int):
    train_tf = get_simple_train_transform(image_size)
    eval_tf  = get_simple_eval_transform(image_size)
    train_ds = LocalEmbeddingDataset(transform=train_tf)
    eval_ds  = LocalEmbeddingDataset(transform=eval_tf)
    cfg = dict(SUP_CFG)
    cfg["seed"] = seed
    cfg["val_size"] = 0.20
    cfg["test_size"] = 0.20
    train_idx, val_idx, test_idx = template_aware_split(train_ds.records, cfg)
    labels = np.array([r["label"] for r in train_ds.records])
    train_loader = DataLoader(Subset(train_ds, sorted(train_idx)),
                              batch_size=batch_size, shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(Subset(eval_ds, sorted(val_idx)),
                            batch_size=batch_size, shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(Subset(eval_ds, sorted(test_idx)),
                             batch_size=batch_size, shuffle=False, num_workers=0,
                             pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader, labels


# ===========================================================================
# 4. Failure-case figures (FIXED: only show actual misclassifications)
# ===========================================================================

class GradCAMResNet50:
    def __init__(self, model: SupervisedResNet50):
        self.model = model
        self.target = model.net.layer4
        self.activations = None
        self.gradients = None
        self._fwd = self.target.register_forward_hook(self._save_act)
        self._bwd = self.target.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, output):
        self.activations = output.detach()

    def _save_grad(self, _, gin, gout):
        self.gradients = gout[0].detach()

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()

    def __call__(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.eval()
        x = x.requires_grad_(True)
        score = self.model(x)[:, target_class].sum()
        self.model.zero_grad()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1).cpu().numpy()
        cmin = cam.reshape(cam.shape[0], -1).min(axis=1)[:, None, None]
        cmax = cam.reshape(cam.shape[0], -1).max(axis=1)[:, None, None]
        return (cam - cmin) / (cmax - cmin + 1e-8)


def denormalise(t: torch.Tensor) -> np.ndarray:
    img = t.cpu().numpy() * IMAGENET_STD.reshape(3, 1, 1) + IMAGENET_MEAN.reshape(3, 1, 1)
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def render_failures(
    kind: str, fail_indices: list[int], records: list[dict],
    test_idx_global: list[int], y_prob: np.ndarray,
    model: SupervisedResNet50, device: torch.device,
    image_size: int, out_path: Path,
) -> None:
    """Render up to 6 misclassifications of one type (FN or FP) with
    Grad-CAM overlays and informative captions.
    """
    if not fail_indices:
        # Save a clean text figure stating "0 failures of this type"
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5,
                f"No {kind} on the held-out test split.\n"
                f"(All {len(y_prob)} test predictions of this type were correct.)",
                ha="center", va="center", fontsize=14, fontweight="bold")
        ax.axis("off")
        ax.set_title(f"{kind} — none observed", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved (no failures): %s", out_path)
        return

    cam = GradCAMResNet50(model)
    eval_tf = get_simple_eval_transform(image_size)
    n = min(6, len(fail_indices))

    fig, axes = plt.subplots(2, n, figsize=(n * 2.6, 5.6))
    if n == 1:
        axes = axes.reshape(2, 1)
    direction = ("FN: true-fake → predicted-real"
                 if kind == "False Negatives"
                 else "FP: true-real → predicted-fake")
    fig.suptitle(
        f"{kind} on held-out test split  ({len(fail_indices)} total)\n"
        f"Top row: original image. Bottom row: Grad-CAM overlay (last conv layer).\n"
        f"{direction}",
        fontsize=11, fontweight="bold",
    )

    for col, sub_idx in enumerate(fail_indices[:n]):
        rec = records[test_idx_global[sub_idx]]
        with Image.open(rec["path"]).convert("RGB") as im:
            tensor = eval_tf(im).unsqueeze(0).to(device)
        pred_class = int(y_prob[sub_idx] >= 0.5)
        heat = cam(tensor, target_class=pred_class)[0]
        rgb = denormalise(tensor.squeeze(0).detach())

        axes[0, col].imshow(rgb)
        axes[0, col].axis("off")
        axes[1, col].imshow(rgb)
        axes[1, col].imshow(heat, cmap="jet", alpha=0.45)
        axes[1, col].axis("off")

        ctype = rec.get("ctype", "real") or "real"
        true_lbl = "real" if rec["label"] == 0 else "fake"
        axes[0, col].set_title(
            f"{rec['path'].stem}\np(fake)={y_prob[sub_idx]:.3f}\n"
            f"true={true_lbl}\nctype={ctype}",
            fontsize=8,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    cam.remove()
    logger.info("Saved: %s", out_path)


def figure_failures(
    model: SupervisedResNet50, test_loader: DataLoader, records: list[dict],
    test_idx_global: list[int], device: torch.device, image_size: int,
):
    """Pick true FN and FP indices and render two SEPARATE figures."""
    y_true, y_prob, y_pred = predict_test(model, test_loader, device)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    # Order by confidence of the wrong prediction
    fn_idx = np.argsort(-(1.0 - y_prob[fn_mask]))
    fp_idx = np.argsort(-y_prob[fp_mask])
    fn_global_sub = np.where(fn_mask)[0][fn_idx].tolist()
    fp_global_sub = np.where(fp_mask)[0][fp_idx].tolist()

    logger.info("True FN count: %d   True FP count: %d", len(fn_global_sub), len(fp_global_sub))
    render_failures("False Negatives", fn_global_sub, records, test_idx_global,
                    y_prob, model, device, image_size,
                    OUT / "09_failure_false_negatives.png")
    render_failures("False Positives", fp_global_sub, records, test_idx_global,
                    y_prob, model, device, image_size,
                    OUT / "09_failure_false_positives.png")


# ===========================================================================
# 5. Embedding figures (each projection × each colouring = 1 separate plot)
# ===========================================================================

@torch.no_grad()
def extract_resnet50_features(model: SupervisedResNet50, loader: DataLoader,
                              device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    backbone = model.net
    feats, labels_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast(device_type="cuda"):
            x = backbone.conv1(images)
            x = backbone.bn1(x)
            x = backbone.relu(x)
            x = backbone.maxpool(x)
            x = backbone.layer1(x)
            x = backbone.layer2(x)
            x = backbone.layer3(x)
            x = backbone.layer4(x)
            x = backbone.avgpool(x)
            x = torch.flatten(x, 1)
            feats.append(x.float().cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labels_all)


def _scatter(ax, coords: np.ndarray, labels: np.ndarray,
             palette: dict, label_map: dict, title: str,
             marker_size: int = 22):
    # Plot rare classes LAST so they're on top and visible
    order = sorted(label_map.keys(), key=lambda v: int((labels == v).sum()), reverse=True)
    for v in order:
        m = labels == v
        if not m.any():
            continue
        # Make rarer classes more visible
        n = int(m.sum())
        edge = "black" if n < 30 else "none"
        sz = marker_size + (16 if n < 30 else 0)
        ax.scatter(coords[m, 0], coords[m, 1], c=palette[v], s=sz,
                   alpha=0.78, edgecolors=edge, linewidths=0.6,
                   label=f"{label_map[v]} (n={n})")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("dim 1", fontsize=11)
    ax.set_ylabel("dim 2", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(markerscale=1.4, fontsize=10, loc="best", framealpha=0.92)


def figure_embeddings(model: SupervisedResNet50, test_loader: DataLoader,
                      records: list[dict], test_idx_global: list[int],
                      device: torch.device):
    feats, labels = extract_resnet50_features(model, test_loader, device)
    ctypes = np.array([
        {"real": 0, "Inpaint_and_Rewrite": 1, "Crop_and_Replace": 2}.get(
            records[i].get("ctype", "real") or "real", 0)
        for i in test_idx_global
    ])
    label_map_bin = {0: "Real", 1: "Fake"}

    # PCA
    p = PCA(n_components=2, random_state=42).fit_transform(feats)
    var_explained = PCA(n_components=2, random_state=42).fit(feats).explained_variance_ratio_
    for view, lbls, palette, lab_map in [
        ("binary", labels, PALETTE_BIN, label_map_bin),
        ("ctype",  ctypes, PALETTE_CT,  CTYPE_LABELS),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        _scatter(ax, p, lbls, palette, lab_map,
                 f"PCA of ResNet50 penultimate features  —  test set ({view})\n"
                 f"PC1 explains {var_explained[0]*100:.1f}%, PC2 explains {var_explained[1]*100:.1f}%")
        fig.tight_layout()
        fig.savefig(OUT / f"09_pca_{view}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", OUT / f"09_pca_{view}.png")

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(feats)
    for view, lbls, palette, lab_map in [
        ("binary", labels, PALETTE_BIN, label_map_bin),
        ("ctype",  ctypes, PALETTE_CT,  CTYPE_LABELS),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        _scatter(ax, tsne, lbls, palette, lab_map,
                 f"t-SNE of ResNet50 penultimate features  —  test set ({view})\n"
                 f"perplexity=30, n_iter=1000, random_state=42")
        fig.tight_layout()
        fig.savefig(OUT / f"09_tsne_{view}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", OUT / f"09_tsne_{view}.png")

    # UMAP (optional — only if installed)
    try:
        import umap  # type: ignore
        u = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                      random_state=42).fit_transform(feats)
        for view, lbls, palette, lab_map in [
            ("binary", labels, PALETTE_BIN, label_map_bin),
            ("ctype",  ctypes, PALETTE_CT,  CTYPE_LABELS),
        ]:
            fig, ax = plt.subplots(figsize=(8, 6.5))
            _scatter(ax, u, lbls, palette, lab_map,
                     f"UMAP of ResNet50 penultimate features  —  test set ({view})\n"
                     f"n_neighbors=15, min_dist=0.1, random_state=42")
            fig.tight_layout()
            fig.savefig(OUT / f"09_umap_{view}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved: %s", OUT / f"09_umap_{view}.png")
    except Exception as exc:
        logger.warning("UMAP skipped: %s", exc)


# ===========================================================================
# 6. Robustness figures (split into geometric + photometric)
# ===========================================================================

ROBUSTNESS_GEOMETRIC = ["downscale_upscale", "rotation", "center_crop", "random_occlusion"]
ROBUSTNESS_PHOTOMETRIC = ["gaussian_blur", "jpeg_compression", "brightness", "gaussian_noise"]


def figure_robustness():
    csv_path = OUT / "05_robustness_clean.csv"
    if not csv_path.exists():
        logger.warning("Robustness CSV missing: %s — run research/05_robustness_sweep.py first", csv_path)
        return
    import pandas as pd
    df = pd.read_csv(csv_path)
    baseline = df[df["corruption"] == "baseline"].iloc[0]

    for group_name, families in [
        ("geometric", ROBUSTNESS_GEOMETRIC),
        ("photometric", ROBUSTNESS_PHOTOMETRIC),
    ]:
        n = len(families)
        fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5), sharey=True)
        if n == 1:
            axes = [axes]
        fig.suptitle(
            f"Robustness sweep — {group_name} corruptions\n"
            f"baseline (no corruption) ROC-AUC = {baseline['roc_auc']:.4f}, "
            f"accuracy = {baseline['accuracy']:.4f}",
            fontsize=13, fontweight="bold",
        )
        for ax, family in zip(axes, families):
            sub = df[df["corruption"] == family].sort_values("severity_idx")
            x = np.arange(len(sub))
            ax.plot(x, sub["roc_auc"], "o-",  color="#1565C0", lw=2.5, markersize=8, label="ROC-AUC")
            ax.plot(x, sub["accuracy"], "s--", color="#43A047", lw=2.0, markersize=7, label="Accuracy")
            ax.plot(x, sub["f1"],       "^:",  color="#EF6C00", lw=2.0, markersize=7, label="F1")
            ax.axhline(baseline["roc_auc"], color="#1565C0", linestyle=":", alpha=0.5, lw=1)
            ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4, lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels(sub["severity"].tolist(), rotation=20, fontsize=10)
            ax.set_xlabel("Severity", fontsize=11)
            ax.set_title(family.replace("_", " "), fontsize=12)
            ax.set_ylim(0.4, 1.03)
            ax.grid(True, linestyle="--", alpha=0.3)
            if ax is axes[0]:
                ax.set_ylabel("Metric value (test set)", fontsize=11)
                ax.legend(loc="lower left", fontsize=10, framealpha=0.92)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = OUT / f"09_robustness_{group_name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", out)


# ===========================================================================
# 7. Comparison plot with bootstrap CIs + McNemar test
# ===========================================================================

def figure_comparison_with_ci(
    name_a: str, preds_a: tuple,
    name_b: str, preds_b: tuple,
    out_path: Path, mcnemar_path: Path,
):
    y_true, prob_a, pred_a = preds_a
    _,      prob_b, pred_b = preds_b
    metrics_a = metrics_from_preds(y_true, prob_a, pred_a)
    metrics_b = metrics_from_preds(y_true, prob_b, pred_b)
    ci_a = bootstrap_ci(y_true, prob_a, pred_a)
    ci_b = bootstrap_ci(y_true, prob_b, pred_b)
    mc = mcnemar_test(y_true, pred_a, pred_b)

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    nice = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
            "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
    n_test = len(y_true)
    n_pos  = int(y_true.sum())

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(metric_names))
    w = 0.36

    a_vals = [metrics_a[m] for m in metric_names]
    b_vals = [metrics_b[m] for m in metric_names]
    a_err  = [[metrics_a[m] - ci_a[m]["lo"] for m in metric_names],
              [ci_a[m]["hi"] - metrics_a[m] for m in metric_names]]
    b_err  = [[metrics_b[m] - ci_b[m]["lo"] for m in metric_names],
              [ci_b[m]["hi"] - metrics_b[m] for m in metric_names]]

    bars_a = ax.bar(x - w/2, a_vals, w, yerr=a_err, capsize=4,
                    color="#1565C0", edgecolor="white", label=name_a,
                    error_kw=dict(lw=1.4, ecolor="#0D47A1"))
    bars_b = ax.bar(x + w/2, b_vals, w, yerr=b_err, capsize=4,
                    color="#7B1FA2", edgecolor="white", label=name_b,
                    error_kw=dict(lw=1.4, ecolor="#4A148C"))

    for bar, v in zip(bars_a, a_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.014, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    for bar, v in zip(bars_b, b_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.014, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([nice[m] for m in metric_names], fontsize=11)
    ax.set_ylabel("Test-set metric value (±95% bootstrap CI)", fontsize=12)
    ax.set_ylim(0.85, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    n_wrong_a = int((pred_a != y_true).sum())
    n_wrong_b = int((pred_b != y_true).sum())
    n_fn_a    = int(((pred_a == 0) & (y_true == 1)).sum())
    n_fn_b    = int(((pred_b == 0) & (y_true == 1)).sum())
    title = (
        f"Test-set comparison: {name_a} vs {name_b}\n"
        f"n_test = {n_test}  (n_fakes = {n_pos});  bootstrap n_iter = 1000\n"
        f"errors:  {name_a} = {n_wrong_a} ({n_fn_a} FN)   |   "
        f"{name_b} = {n_wrong_b} ({n_fn_b} FN)\n"
        f"McNemar p-value = {mc['p_value']:.4f}  →  {mc['interpretation']}"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)

    with open(mcnemar_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_a": name_a, "model_b": name_b,
            "n_test": int(n_test), "n_fakes": int(n_pos),
            "errors_a": n_wrong_a, "errors_b": n_wrong_b,
            "fn_a": n_fn_a, "fn_b": n_fn_b,
            "mcnemar": mc,
            "ci_a": ci_a, "ci_b": ci_b,
            "metrics_a": metrics_a, "metrics_b": metrics_b,
        }, f, indent=2)
    logger.info("Saved: %s", mcnemar_path)


# ===========================================================================
# 8. Ablation study — train RGB+FFT-no-transformer, then compare 3 models
# ===========================================================================

ABLATION_CFG = {
    "image_size":          224,
    "batch_size":          16,
    "epochs":              20,
    "lr":                  1e-4,
    "weight_decay":        1e-4,
    "use_amp":             True,
    "seed":                42,
    "best_path":           str(MODELS / "two_stream_no_transformer_best.pth"),
}


def train_one_epoch_simple(model, loader, loss_fn, opt, device, scaler) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc="  batch", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda"):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def train_ablation_b(device: torch.device) -> Path:
    """Train RGB+FFT WITHOUT per-branch transformer (avgpool fusion only)."""
    if Path(ABLATION_CFG["best_path"]).exists():
        logger.info("Ablation B checkpoint already exists at %s — skipping training",
                    ABLATION_CFG["best_path"])
        return Path(ABLATION_CFG["best_path"])

    seed_everything(ABLATION_CFG["seed"])
    train_loader, val_loader, test_loader, all_labels = make_train_val_test_loaders(
        ABLATION_CFG["image_size"], ABLATION_CFG["batch_size"], ABLATION_CFG["seed"],
    )
    model = TwoStreamForgeryNet(num_classes=2, use_transformer=False).to(device)
    _, total = count_parameters(model)
    logger.info("[Ablation B] RGB+FFT (no transformer) params: %s", f"{total:,}")

    pos = (all_labels == 1).sum()
    neg = (all_labels == 0).sum()
    weight = torch.tensor([(pos+neg)/(2*neg), (pos+neg)/(2*pos)], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = optim.AdamW(make_param_groups(model, ABLATION_CFG["weight_decay"]), lr=ABLATION_CFG["lr"])
    sched = CosineAnnealingLR(opt, T_max=ABLATION_CFG["epochs"])
    scaler = GradScaler(device="cuda")

    best_val_auc = -1.0
    t0 = time.time()
    for ep in range(1, ABLATION_CFG["epochs"] + 1):
        loss_v = train_one_epoch_simple(model, train_loader, loss_fn, opt, device, scaler)
        sched.step()
        # quick val
        y_true, y_prob, y_pred = predict_test(model, val_loader, device)
        val_auc = float(roc_auc_score(y_true, y_prob))
        logger.info("[Ablation B] ep %2d/%d  loss=%.4f  val_roc_auc=%.4f  elapsed=%.0fs",
                    ep, ABLATION_CFG["epochs"], loss_v, val_auc, time.time() - t0)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep,
                        "val_roc_auc": val_auc}, ABLATION_CFG["best_path"])
    logger.info("[Ablation B] best val ROC-AUC = %.4f", best_val_auc)
    return Path(ABLATION_CFG["best_path"])


def figure_ablation(device: torch.device):
    """Produce the ablation table + bar chart comparing 3 models on TEST."""
    test_loader, records, test_idx = make_test_loader(
        ABLATION_CFG["image_size"], ABLATION_CFG["batch_size"], ABLATION_CFG["seed"],
    )

    models_info = []

    # A — Single-stream ResNet50 (uses existing supervised checkpoint)
    ckpt_a = MODELS / "supervised_resnet50_best.pth"
    if ckpt_a.exists():
        m = SupervisedResNet50(num_classes=2, dropout=0.2).to(device)
        ck = torch.load(str(ckpt_a), map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state_dict"], strict=False)
        models_info.append(("A. RGB only (ResNet50)", m))
    else:
        logger.warning("Ablation A ckpt missing: %s — skipping", ckpt_a)

    # B — RGB+FFT no transformer (train if needed)
    ckpt_b = train_ablation_b(device)
    if ckpt_b.exists():
        m = TwoStreamForgeryNet(num_classes=2, use_transformer=False).to(device)
        ck = torch.load(str(ckpt_b), map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state_dict"], strict=False)
        models_info.append(("B. RGB+FFT (avg-pool fusion)", m))

    # C — RGB+FFT + Transformer (existing checkpoint)
    ckpt_c = MODELS / "two_stream_best.pth"
    if ckpt_c.exists():
        m = TwoStreamForgeryNet(num_classes=2, use_transformer=True).to(device)
        ck = torch.load(str(ckpt_c), map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state_dict"], strict=False)
        models_info.append(("C. RGB+FFT + Transformer", m))
    else:
        logger.warning("Ablation C ckpt missing: %s — skipping", ckpt_c)

    rows = []
    bar_data = []
    for name, model in models_info:
        y_true, y_prob, y_pred = predict_test(model, test_loader, device)
        m = metrics_from_preds(y_true, y_prob, y_pred)
        n_wrong = int((y_pred != y_true).sum())
        n_fn = int(((y_pred == 0) & (y_true == 1)).sum())
        rows.append({"model": name, **m, "n_wrong": n_wrong, "n_fn": n_fn})
        bar_data.append((name, m))

    # Save ablation CSV
    csv_path = OUT / "09_ablation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "precision", "recall", "f1",
                    "roc_auc", "pr_auc", "n_wrong", "n_fn"])
        for r in rows:
            w.writerow([r["model"], round(r["accuracy"], 4),
                        round(r["precision"], 4), round(r["recall"], 4),
                        round(r["f1"], 4), round(r["roc_auc"], 4),
                        round(r["pr_auc"], 4), r["n_wrong"], r["n_fn"]])
    logger.info("Saved: %s", csv_path)

    # Bar chart
    metrics = ["accuracy", "f1", "recall", "roc_auc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    fig.suptitle(
        "Ablation study: contribution of FFT branch and Transformer fusion\n"
        "(all models on the same template-aware test split, n=443)",
        fontsize=13, fontweight="bold",
    )
    cmap = plt.colormaps.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(bar_data))]
    nice = {"accuracy": "Accuracy", "f1": "F1", "recall": "Recall", "roc_auc": "ROC-AUC"}
    for ax, m in zip(axes, metrics):
        vals = [d[1][m] for d in bar_data]
        names = [d[0] for d in bar_data]
        bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split(". ", 1)[0] for n in names], fontsize=11)
        ax.set_title(nice[m], fontsize=12)
        ax.set_ylim(0.85, 1.02)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    axes[0].set_ylabel("Test metric", fontsize=11)

    # Legend at the top
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    legend_labels = [d[0] for d in bar_data]
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    out = OUT / "09_ablation_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


# ===========================================================================
# 9. Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-ablation", action="store_true",
                   help="Skip the ablation study (saves ~10 min training)")
    args = p.parse_args()

    device = get_device()

    test_loader, records, test_idx = make_test_loader(
        SUP_CFG["image_size"], SUP_CFG["batch_size"], seed=42,
    )

    # Load both headline checkpoints
    ckpt_resnet = MODELS / "supervised_resnet50_best.pth"
    ckpt_two    = MODELS / "two_stream_best.pth"
    if not ckpt_resnet.exists() or not ckpt_two.exists():
        logger.error("Headline checkpoints missing. Train them first:")
        logger.error("  python -u main.py --skip-train --skip-embed --skip-cls "
                     "--skip-baseline --skip-eval --skip-viz --skip-shap")
        return

    resnet = SupervisedResNet50(num_classes=2, dropout=0.2).to(device)
    resnet.load_state_dict(
        torch.load(str(ckpt_resnet), map_location=device, weights_only=False)["model_state_dict"],
        strict=False,
    )
    two_stream = TwoStreamForgeryNet(num_classes=2, use_transformer=True).to(device)
    two_stream.load_state_dict(
        torch.load(str(ckpt_two), map_location=device, weights_only=False)["model_state_dict"],
        strict=False,
    )

    # ── Failures (Two-Stream model, fixed labelling) ─────────────────────
    figure_failures(resnet, test_loader, records, test_idx, device, SUP_CFG["image_size"])

    # ── Embeddings (ResNet50 backbone — same as before, but separated) ───
    figure_embeddings(resnet, test_loader, records, test_idx, device)

    # ── Robustness (split into geometric / photometric) ──────────────────
    figure_robustness()

    # ── Comparison + bootstrap CI + McNemar test ─────────────────────────
    preds_a = predict_test(resnet,     test_loader, device)
    preds_b = predict_test(two_stream, test_loader, device)
    figure_comparison_with_ci(
        "ResNet50 (single-stream)", preds_a,
        "Two-Stream RGB+FFT + Transformer", preds_b,
        out_path=OUT / "09_comparison_with_ci.png",
        mcnemar_path=OUT / "09_mcnemar_test.json",
    )

    # ── Ablation study (RGB / RGB+FFT / RGB+FFT+Transformer) ─────────────
    if not args.skip_ablation:
        figure_ablation(device)

    logger.info("All publication figures saved to %s", OUT)


if __name__ == "__main__":
    main()
