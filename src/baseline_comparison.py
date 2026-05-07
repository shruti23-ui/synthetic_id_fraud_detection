"""
baseline_comparison.py
----------------------
Demonstrates the value of contrastive pre-training:

  Baseline  : raw ImageNet-pretrained ResNet18 features (no SimCLR fine-tuning)
  SimCLR    : features from the encoder *after* contrastive pre-training

Both feature sets are fed into the same downstream classifiers (XGBoost, RF,
LR, MLP) and their metrics are compared side-by-side. This is the central
empirical claim of the thesis — that self-supervised contrastive pre-training
produces representations that improve downstream fraud classification.

Outputs:
    outputs/metrics/baseline_vs_simclr.csv
    outputs/metrics/baseline_features.npy / .npy labels (cached)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmentations import get_eval_transform
from local_dataset_loader import get_local_embedding_loader
from contrastive_model import SimCLRModel
from classifier import build_classifiers, evaluate_classifier
from preprocess import load_processed_arrays, split_train_val_test
from utils import seed_everything, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "image_size":     224,
    "batch_size":     32,
    "val_size":       0.2,
    "test_size":      0.2,
    "random_state":   42,
    "metrics_dir":    "outputs/metrics",
    "processed_dir":  "data/processed",
    "split":          "train",
    "seed":           42,
}


# ---------------------------------------------------------------------------
# Feature extractor for baseline (raw ImageNet ResNet18)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_baseline_features(loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with an *untrained* (ImageNet-only) SimCLR encoder."""
    model = SimCLRModel(embedding_dim=128, pretrained=True).to(device)
    model.eval()

    feats, labels = [], []
    for images, lbl in tqdm(loader, desc="Baseline features"):
        images = images.to(device, non_blocking=True)
        f = model.get_features(images)
        feats.append(f.cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(feats), np.concatenate(labels)


# ---------------------------------------------------------------------------
# Run all classifiers on a (X, y) feature set
# ---------------------------------------------------------------------------

def _train_eval_all(features: np.ndarray, labels: np.ndarray, source_tag: str, cfg: dict) -> list[dict]:
    """Train every classifier on (features, labels) using a 60/20/20 split."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        features, labels,
        val_size=cfg["val_size"],
        test_size=cfg["test_size"],
        seed=cfg["random_state"],
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    rows = []
    for name, clf in build_classifiers().items():
        logger.info("[%s] Training %s ...", source_tag, name)
        clf.fit(X_train_s, y_train)
        val_m  = evaluate_classifier(clf, X_val_s,  y_val)
        test_m = evaluate_classifier(clf, X_test_s, y_test)
        logger.info(
            "  %s [%s VAL ] acc=%.3f recall=%.3f f1=%.3f roc_auc=%.3f pr_auc=%.3f",
            name, source_tag,
            val_m["accuracy"], val_m["recall"], val_m["f1"], val_m["roc_auc"], val_m["pr_auc"],
        )
        logger.info(
            "  %s [%s TEST] acc=%.3f recall=%.3f f1=%.3f roc_auc=%.3f pr_auc=%.3f",
            name, source_tag,
            test_m["accuracy"], test_m["recall"], test_m["f1"], test_m["roc_auc"], test_m["pr_auc"],
        )
        rows.append({
            "features": source_tag,
            "model": name,
            **{f"val_{k}":  v for k, v in val_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
        })
    return rows


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

def run_baseline_comparison(config: dict = CONFIG) -> pd.DataFrame:
    """Compare raw ImageNet ResNet18 vs SimCLR-pretrained encoder."""
    Path(config["metrics_dir"]).mkdir(parents=True, exist_ok=True)
    seed_everything(config.get("seed", 42))
    device = get_device()

    # ── 1. Baseline features (raw ImageNet ResNet18, no SimCLR) ──────────────
    logger.info("Extracting baseline features (raw ImageNet ResNet18) …")
    transform = get_eval_transform(image_size=config["image_size"])
    loader = get_local_embedding_loader(
        transform=transform,
        batch_size=config["batch_size"],
        num_workers=0,
    )
    baseline_feats, baseline_labels = extract_baseline_features(loader, device)
    logger.info("Baseline features: %s", baseline_feats.shape)

    # ── 2. SimCLR features (loaded from data/processed/) ─────────────────────
    logger.info("Loading SimCLR embeddings from %s …", config["processed_dir"])
    try:
        simclr_feats, simclr_labels = load_processed_arrays(split=config["split"])
        logger.info("SimCLR features: %s", simclr_feats.shape)
    except FileNotFoundError:
        logger.error(
            "SimCLR embeddings missing — run generate_embeddings.py first. "
            "Baseline comparison aborted."
        )
        return pd.DataFrame()

    # ── 3. Train downstream classifiers on each ──────────────────────────────
    rows = []
    rows.extend(_train_eval_all(baseline_feats, baseline_labels, "raw_resnet",  config))
    rows.extend(_train_eval_all(simclr_feats,  simclr_labels,    "simclr",      config))

    df = pd.DataFrame(rows)
    out = Path(config["metrics_dir"]) / "baseline_vs_simclr.csv"
    df.to_csv(out, index=False)
    logger.info("Saved comparison → %s\n%s", out, df.to_string(index=False))

    # ── 4. Summary delta on TEST set ─────────────────────────────────────────
    pivot = df.pivot_table(index="model", columns="features",
                            values=["test_accuracy", "test_f1", "test_roc_auc", "test_pr_auc"])
    if "raw_resnet" in df["features"].unique() and "simclr" in df["features"].unique():
        for metric in ["test_accuracy", "test_f1", "test_roc_auc", "test_pr_auc"]:
            delta = pivot[metric]["simclr"] - pivot[metric]["raw_resnet"]
            logger.info("delta %s (SimCLR - Raw): %s", metric, dict(delta.round(4)))

    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_baseline_comparison()
    logger.info("baseline_comparison.py finished.")
