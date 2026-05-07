"""
train_contrastive.py
--------------------
Self-supervised SimCLR contrastive pre-training loop.

Usage (standalone):
    python src/train_contrastive.py

Key outputs:
    models/simclr_encoder.pth        – best encoder checkpoint
    outputs/metrics/train_loss.csv   – per-epoch loss log
"""

import logging
import os
import sys
import csv
import time
from pathlib import Path

# IMPORTANT: sklearn must be imported before torch to avoid libomp DLL crash
# on Windows + Python 3.13 (sklearn-pulled libomp conflicts with torch's).
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Allow running from repo root or from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_loader import load_hf_dataset, get_contrastive_loader
from local_dataset_loader import get_local_contrastive_loader, get_local_embedding_loader
from augmentations import get_contrastive_transform, get_eval_transform
from contrastive_model import SimCLRModel, NTXentLoss
from utils import seed_everything, get_device, count_parameters, EarlyStopping

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (all hyper-parameters in one place)
# ---------------------------------------------------------------------------

CONFIG = {
    "image_size":          224,
    "batch_size":          32,         # tuned for 4 GB VRAM
    "epochs":              5,          # quick-test default; override via CLI
    "lr":                  1e-3,
    "weight_decay":        1e-4,
    "temperature":         0.5,
    "embedding_dim":       128,
    "pretrained":          True,
    "num_workers":         0,          # Windows: 0 is safest; cache makes it fast anyway
    "checkpoint_dir":      "models",
    "metrics_dir":         "outputs/metrics",
    "save_every":          5,
    "data_source":         "local",
    "use_amp":             True,        # mixed-precision (cuts VRAM ~½, faster)
    "early_stop_patience": 8,
    "seed":                42,
    "linear_probe":        True,        # evaluate linear probe accuracy each epoch
    "probe_test_size":     0.2,
    "probe_max_samples":   500,         # cap probe set size for speed (sampled stratified)
}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler=None) -> float:
    """Run one epoch of contrastive training and return mean loss.

    If `scaler` is provided, uses torch.cuda.amp mixed-precision.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for view1, view2, _ in tqdm(loader, desc="  batch", leave=False):
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                _, z_i = model(view1)
                _, z_j = model(view2)
                loss = loss_fn(z_i, z_j)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            _, z_i = model(view1)
            _, z_j = model(view2)
            loss = loss_fn(z_i, z_j)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def linear_probe_eval(
    model: SimCLRModel,
    probe_loader,
    device: torch.device,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Freeze the encoder, extract features, fit a quick logistic regression,
    and return validation metrics. Used for per-epoch monitoring of
    representation quality during self-supervised training.

    Returns a dict with keys: accuracy, precision, recall, f1, roc_auc.
    """
    model.eval()
    feats_all, labels_all = [], []
    for images, labels in probe_loader:
        images = images.to(device, non_blocking=True)
        f = model.get_features(images).cpu().numpy()
        feats_all.append(f)
        labels_all.append(labels.numpy())
    X = np.concatenate(feats_all)
    y = np.concatenate(labels_all)

    if len(np.unique(y)) < 2:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    clf = LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs", random_state=seed)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    yprob = clf.predict_proba(Xte)[:, 1]

    return {
        "accuracy":  accuracy_score(yte, yp),
        "precision": precision_score(yte, yp, zero_division=0),
        "recall":    recall_score(yte, yp, zero_division=0),
        "f1":        f1_score(yte, yp, zero_division=0),
        "roc_auc":   roc_auc_score(yte, yprob),
    }


def save_checkpoint(model: SimCLRModel, path: str, epoch: int, loss: float) -> None:
    """Persist model state dict and metadata."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": loss,
        },
        path,
    )
    logger.info("Checkpoint saved → %s  (epoch %d, loss %.4f)", path, epoch, loss)


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def train(config: dict = CONFIG) -> SimCLRModel:
    """Full contrastive pre-training loop.

    Args:
        config: Dictionary of hyper-parameters (see CONFIG above).

    Returns:
        Trained SimCLRModel with best weights loaded.
    """
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["metrics_dir"]).mkdir(parents=True, exist_ok=True)

    seed_everything(config.get("seed", 42))
    device = get_device()

    # ── Data ─────────────────────────────────────────────────────────────────
    transform = get_contrastive_transform(image_size=config["image_size"])
    use_local = config.get("data_source", "local") == "local"

    if use_local:
        logger.info("Loading local fake/real dataset (templates/) …")
        loader = get_local_contrastive_loader(
            transform=transform,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
    else:
        logger.info("Loading HuggingFace dataset …")
        raw = load_hf_dataset()
        split_name = "train" if "train" in raw else list(raw.keys())[0]
        logger.info("Using split: %s  (%d samples)", split_name, len(raw[split_name]))
        loader = get_contrastive_loader(
            raw[split_name],
            transform=transform,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )

    logger.info("DataLoader ready: %d batches × batch_size %d", len(loader), config["batch_size"])

    # ── Probe loader (single-view, deterministic) for per-epoch monitoring ────
    probe_loader = None
    do_probe = bool(config.get("linear_probe", True)) and use_local
    if do_probe:
        from local_dataset_loader import LocalEmbeddingDataset
        probe_ds = LocalEmbeddingDataset(transform=get_eval_transform(image_size=config["image_size"]))

        # Stratified subsample for speed (probe is for monitoring, not final eval)
        max_n = int(config.get("probe_max_samples", 500))
        if max_n and len(probe_ds) > max_n:
            rng = np.random.default_rng(config.get("seed", 42))
            labels_all = np.array([r["label"] for r in probe_ds.records])
            per_class = max_n // 2
            idx0 = rng.choice(np.where(labels_all == 0)[0], min(per_class, (labels_all == 0).sum()), replace=False)
            idx1 = rng.choice(np.where(labels_all == 1)[0], min(per_class, (labels_all == 1).sum()), replace=False)
            sub_idx = np.concatenate([idx0, idx1])
            probe_ds = Subset(probe_ds, sub_idx.tolist())
            logger.info("Probe set: %d images (stratified subsample for speed)", len(probe_ds))
        else:
            logger.info("Probe set: %d images (full)", len(probe_ds))

        probe_loader = DataLoader(
            probe_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )
        logger.info("Linear probe enabled - per-epoch acc / recall / f1 / ROC-AUC will be reported.")

    # ── Model, Loss, Optimiser ───────────────────────────────────────────────
    model = SimCLRModel(
        embedding_dim=config["embedding_dim"],
        pretrained=config["pretrained"],
    ).to(device)

    trainable, total = count_parameters(model)
    logger.info("Model params — trainable: %s / total: %s", f"{trainable:,}", f"{total:,}")

    loss_fn = NTXentLoss(temperature=config["temperature"])

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Mixed-precision scaler (CUDA only)
    use_amp = bool(config.get("use_amp", True)) and device.type == "cuda"
    amp_scaler = torch.cuda.amp.GradScaler() if use_amp else None
    logger.info("Mixed precision (AMP): %s", "enabled" if use_amp else "disabled")

    early_stop = EarlyStopping(patience=config.get("early_stop_patience", 8), mode="min")

    # ── Metrics log ──────────────────────────────────────────────────────────
    metrics_path = Path(config["metrics_dir"]) / "train_loss.csv"
    header = ["epoch", "loss", "lr", "elapsed_s"]
    if do_probe:
        header += ["probe_accuracy", "probe_precision", "probe_recall", "probe_f1", "probe_roc_auc"]
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    best_loss = float("inf")
    best_model_path = Path(config["checkpoint_dir"]) / "simclr_encoder_best.pth"

    # ── Training Loop ────────────────────────────────────────────────────────
    logger.info("Starting contrastive training for %d epochs …", config["epochs"])
    t0 = time.time()

    for epoch in range(1, config["epochs"] + 1):
        epoch_loss = train_one_epoch(model, loader, loss_fn, optimizer, device, scaler=amp_scaler)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        logger.info(
            "Epoch [%3d/%d]  loss=%.4f  lr=%.2e  elapsed=%.0fs",
            epoch, config["epochs"], epoch_loss, current_lr, elapsed,
        )

        # ── Linear probe monitoring (acc / recall / f1 / ROC-AUC) ───────────
        probe_metrics = {}
        if do_probe and probe_loader is not None:
            probe_t = time.time()
            probe_metrics = linear_probe_eval(
                model, probe_loader, device,
                test_size=config.get("probe_test_size", 0.2),
                seed=config.get("seed", 42),
            )
            logger.info(
                "             [linear probe] acc=%.4f  precision=%.4f  recall=%.4f  f1=%.4f  roc_auc=%.4f  (probe %.1fs)",
                probe_metrics["accuracy"], probe_metrics["precision"],
                probe_metrics["recall"], probe_metrics["f1"], probe_metrics["roc_auc"],
                time.time() - probe_t,
            )

        # Append to CSV
        row = [epoch, round(epoch_loss, 6), round(current_lr, 8), round(elapsed, 1)]
        if do_probe:
            row += [
                round(probe_metrics.get("accuracy",  0.0), 4),
                round(probe_metrics.get("precision", 0.0), 4),
                round(probe_metrics.get("recall",    0.0), 4),
                round(probe_metrics.get("f1",        0.0), 4),
                round(probe_metrics.get("roc_auc",   0.0), 4),
            ]
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # Save periodic checkpoint
        if epoch % config["save_every"] == 0:
            ckpt_path = Path(config["checkpoint_dir"]) / f"simclr_epoch_{epoch}.pth"
            save_checkpoint(model, str(ckpt_path), epoch, epoch_loss)

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, str(best_model_path), epoch, epoch_loss)

        # Early stopping check
        if early_stop.step(epoch_loss):
            logger.info("Stopping early at epoch %d (best loss %.4f)", epoch, best_loss)
            break

    logger.info("Training complete. Best loss: %.4f", best_loss)
    logger.info("Best model saved at: %s", best_model_path)

    # Load best weights before returning
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    return model


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trained_model = train()
    logger.info("train_contrastive.py finished.")
