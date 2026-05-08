"""
research/06_embedding_geometry.py
---------------------------------
Geometric analysis of the supervised ResNet50's penultimate-layer features
(2048-d) on the test set.

What we measure:
  - Cluster purity (binary fake/real and per-ctype) via k-means
  - Silhouette score          (higher = better-separated clusters)
  - Davies-Bouldin index      (lower  = better-separated)
  - Calinski-Harabasz index   (higher = better-separated)
  - Mean intra-class cosine similarity (compactness)
  - Mean inter-class cosine similarity (separation)
  - 2-D visualisations: PCA, t-SNE, UMAP coloured by binary label and by ctype

Run for both the leaky and clean checkpoints to see whether the cluster
structure that emerges is genuinely about *forgery* or about *templates*.

Outputs:
    research_outputs/06_embedding_metrics_<run>.json
    research_outputs/06_embedding_pca_<run>.png
    research_outputs/06_embedding_tsne_<run>.png
    research_outputs/06_embedding_umap_<run>.png
    research_outputs/06_embedding_similarity_<run>.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from local_dataset_loader import LocalEmbeddingDataset  # noqa: E402
from supervised_finetune import (  # noqa: E402
    CONFIG as LEAKY_CFG,
    SupervisedResNet50,
    get_simple_eval_transform,
)
from utils import get_device  # noqa: E402

from template_split import template_aware_split  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Extract penultimate (2048-d) features from ResNet50
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model: SupervisedResNet50, loader: DataLoader, device: torch.device,
                     use_amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Strip the FC head and run the backbone to get 2048-d features."""
    model.eval()
    backbone = model.net  # the torchvision ResNet50

    feats, labels_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if use_amp:
            with autocast(device_type="cuda"):
                # forward up to the global average pool (layer4 -> avgpool)
                x = backbone.conv1(images);  x = backbone.bn1(x);  x = backbone.relu(x)
                x = backbone.maxpool(x)
                x = backbone.layer1(x); x = backbone.layer2(x)
                x = backbone.layer3(x); x = backbone.layer4(x)
                x = backbone.avgpool(x)
                x = torch.flatten(x, 1)
                feats.append(x.float().cpu().numpy())
        else:
            x = backbone.conv1(images);  x = backbone.bn1(x);  x = backbone.relu(x)
            x = backbone.maxpool(x)
            x = backbone.layer1(x); x = backbone.layer2(x)
            x = backbone.layer3(x); x = backbone.layer4(x)
            x = backbone.avgpool(x)
            x = torch.flatten(x, 1)
            feats.append(x.cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels_all, axis=0)


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------

def cluster_purity(true_labels: np.ndarray, kmeans_labels: np.ndarray) -> float:
    """Standard cluster-purity metric in [0, 1]."""
    n = len(true_labels)
    purity = 0
    for k in np.unique(kmeans_labels):
        mask = kmeans_labels == k
        if mask.sum() == 0:
            continue
        # Most common true label in this cluster
        vals, counts = np.unique(true_labels[mask], return_counts=True)
        purity += counts.max()
    return float(purity) / n


def cosine_stats(features: np.ndarray, labels: np.ndarray) -> dict:
    """Mean intra/inter-class cosine similarity. Subsample to keep tractable."""
    rng = np.random.default_rng(0)
    n = len(features)
    if n > 800:
        idx = rng.choice(n, 800, replace=False)
        f = features[idx]; y = labels[idx]
    else:
        f = features; y = labels

    f_norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
    sim = f_norm @ f_norm.T

    # mask diagonal
    np.fill_diagonal(sim, np.nan)
    same_class = y[:, None] == y[None, :]

    intra = float(np.nanmean(sim[same_class]))
    inter = float(np.nanmean(sim[~same_class]))
    return {"intra_class_cosine": intra, "inter_class_cosine": inter,
            "separation_gap": intra - inter}


def compute_geometry(features: np.ndarray, labels: np.ndarray) -> dict:
    metrics: dict = {}

    # Cluster purity (k=2 binary)
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(features)
    metrics["cluster_purity_binary"] = cluster_purity(labels, km.labels_)

    # Silhouette / DB / CH (binary)
    metrics["silhouette_binary"]      = float(silhouette_score(features, labels, metric="cosine", sample_size=min(800, len(features)), random_state=42))
    metrics["davies_bouldin_binary"]  = float(davies_bouldin_score(features, labels))
    metrics["calinski_harabasz_binary"] = float(calinski_harabasz_score(features, labels))

    metrics.update(cosine_stats(features, labels))
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

PALETTE_BIN = {0: "#1565C0", 1: "#D32F2F"}
LABELS_BIN  = {0: "Real", 1: "Fake"}

CTYPE_LABELS = {0: "Real", 1: "Inpaint_and_Rewrite", 2: "Crop_and_Replace"}
PALETTE_CT   = {0: "#1565C0", 1: "#D32F2F", 2: "#F57C00"}


def scatter(coords: np.ndarray, labels: np.ndarray, title: str, ax,
            palette: dict, label_map: dict):
    for v, name in label_map.items():
        m = labels == v
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], c=palette[v], s=18, alpha=0.7,
                       edgecolors="none", label=f"{name} (n={int(m.sum())})")
    ax.set_title(title); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(markerscale=1.5, fontsize=10)


def plot_2d(features: np.ndarray, labels: np.ndarray, ctypes: np.ndarray,
            run_name: str, out_dir: Path):
    # PCA
    pca = PCA(n_components=2, random_state=42).fit(features)
    p_coords = pca.transform(features)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    scatter(p_coords, labels, f"PCA (binary) — {run_name}", axes[0], PALETTE_BIN, LABELS_BIN)
    scatter(p_coords, ctypes, f"PCA (forgery type) — {run_name}", axes[1], PALETTE_CT, CTYPE_LABELS)
    fig.tight_layout(); fig.savefig(out_dir / f"06_embedding_pca_{run_name}.png", dpi=140); plt.close(fig)
    logger.info("Saved: %s", out_dir / f"06_embedding_pca_{run_name}.png")

    # t-SNE
    n = len(features)
    if n > 1500:
        idx = np.random.default_rng(0).choice(n, 1500, replace=False)
        feats_s = features[idx]; lbl_s = labels[idx]; ct_s = ctypes[idx]
    else:
        feats_s = features; lbl_s = labels; ct_s = ctypes
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(feats_s)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    scatter(tsne, lbl_s, f"t-SNE (binary) — {run_name}", axes[0], PALETTE_BIN, LABELS_BIN)
    scatter(tsne, ct_s,  f"t-SNE (forgery type) — {run_name}", axes[1], PALETTE_CT, CTYPE_LABELS)
    fig.tight_layout(); fig.savefig(out_dir / f"06_embedding_tsne_{run_name}.png", dpi=140); plt.close(fig)
    logger.info("Saved: %s", out_dir / f"06_embedding_tsne_{run_name}.png")

    # UMAP
    try:
        import umap  # type: ignore
        u_coords = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(feats_s)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        scatter(u_coords, lbl_s, f"UMAP (binary) — {run_name}", axes[0], PALETTE_BIN, LABELS_BIN)
        scatter(u_coords, ct_s,  f"UMAP (forgery type) — {run_name}", axes[1], PALETTE_CT, CTYPE_LABELS)
        fig.tight_layout(); fig.savefig(out_dir / f"06_embedding_umap_{run_name}.png", dpi=140); plt.close(fig)
        logger.info("Saved: %s", out_dir / f"06_embedding_umap_{run_name}.png")
    except Exception as e:
        logger.warning("UMAP failed: %s", e)


def plot_similarity_heatmap(features: np.ndarray, labels: np.ndarray, run_name: str, out_dir: Path):
    rng = np.random.default_rng(0)
    real_idx = np.where(labels == 0)[0]; fake_idx = np.where(labels == 1)[0]
    n = min(40, len(real_idx), len(fake_idx))
    sel = np.concatenate([rng.choice(real_idx, n, replace=False),
                          rng.choice(fake_idx, n, replace=False)])
    f = features[sel]
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
    sim = f @ f.T

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    ax.axhline(n - 0.5, color="black", lw=1.2); ax.axvline(n - 0.5, color="black", lw=1.2)
    ax.set_xticks([n / 2, n + n / 2]); ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticks([n / 2, n + n / 2]); ax.set_yticklabels(["Real", "Fake"])
    ax.set_title(f"Embedding cosine-similarity heatmap — {run_name}", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.85, label="cosine similarity")
    fig.tight_layout(); fig.savefig(out_dir / f"06_embedding_similarity_{run_name}.png", dpi=140)
    plt.close(fig)
    logger.info("Saved: %s", out_dir / f"06_embedding_similarity_{run_name}.png")


# ---------------------------------------------------------------------------
# Run for one checkpoint
# ---------------------------------------------------------------------------

def analyse_run(name: str, ckpt_path: Path, get_loader, device: torch.device) -> dict | None:
    if not ckpt_path.exists():
        logger.warning("Skipping %s: %s missing", name, ckpt_path); return None
    logger.info("=== Embedding analysis: %s ===", name)
    loader, records, test_indices = get_loader()

    model = SupervisedResNet50(num_classes=2, dropout=0.2).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    features, labels = extract_features(model, loader, device)
    logger.info("Extracted features: %s", features.shape)

    metrics = compute_geometry(features, labels)
    logger.info("%s metrics: %s", name, json.dumps(metrics, indent=2))

    ctypes = np.array([
        {"real": 0, "Inpaint_and_Rewrite": 1, "Crop_and_Replace": 2}.get(
            records[i].get("ctype", "real") or "real", 0)
        for i in test_indices
    ])
    plot_2d(features, labels, ctypes, name, OUT)
    plot_similarity_heatmap(features, labels, name, OUT)
    return metrics


def get_leaky_loader():
    eval_tf = get_simple_eval_transform(LEAKY_CFG["image_size"])
    base_ds = LocalEmbeddingDataset(transform=eval_tf)
    labels = np.array([r["label"] for r in base_ds.records])
    rng = np.random.default_rng(LEAKY_CFG["seed"])
    train_idx, val_idx, test_idx = [], [], []
    for cls in (0, 1):
        cls_idx = np.where(labels == cls)[0]; rng.shuffle(cls_idx)
        n = len(cls_idx); n_test = int(round(n * LEAKY_CFG["test_size"])); n_val = int(round(n * LEAKY_CFG["val_size"]))
        test_idx.extend(cls_idx[:n_test].tolist())
        val_idx.extend(cls_idx[n_test:n_test + n_val].tolist())
        train_idx.extend(cls_idx[n_test + n_val:].tolist())
    test_idx = sorted(test_idx)
    loader = DataLoader(Subset(base_ds, test_idx), batch_size=LEAKY_CFG["batch_size"],
                        shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    return loader, base_ds.records, test_idx


def get_clean_loader():
    eval_tf = get_simple_eval_transform(LEAKY_CFG["image_size"])
    base_ds = LocalEmbeddingDataset(transform=eval_tf)
    cfg = dict(LEAKY_CFG); cfg["val_size"] = 0.20; cfg["test_size"] = 0.20
    _, _, test_idx = template_aware_split(base_ds.records, cfg)
    test_idx = sorted(test_idx)
    loader = DataLoader(Subset(base_ds, test_idx), batch_size=LEAKY_CFG["batch_size"],
                        shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    return loader, base_ds.records, test_idx


def main():
    device = get_device()
    summary = {}
    leaky = analyse_run("leaky", ROOT / "models" / "supervised_resnet50_best.pth",
                        get_leaky_loader, device)
    if leaky: summary["leaky"] = leaky
    clean = analyse_run("clean", ROOT / "models" / "supervised_resnet50_clean_best.pth",
                        get_clean_loader, device)
    if clean: summary["clean"] = clean
    with open(OUT / "06_embedding_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", OUT / "06_embedding_metrics.json")


if __name__ == "__main__":
    main()
