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
# on Windows when both pull conflicting OpenMP runtimes.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Allow running from repo root or from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_loader import load_hf_dataset, get_contrastive_loader
from local_dataset_loader import (
    get_local_contrastive_loader,
    get_local_embedding_loader,
    LocalContrastiveDataset,
    LocalEmbeddingDataset,
    NUM_CTYPES,
    compute_ctype_class_weights,
)
from augmentations import (
    get_contrastive_transform,
    get_forgery_aware_transform,
    get_eval_transform,
)
from contrastive_model import SimCLRModel, NTXentLoss, SupConLoss
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
    "batch_size":          32,
    "epochs":              5,
    "lr":                  3e-4,         # was 1e-3 (too high for pretrained init)
    "warmup_epochs":       5,            # linear warmup 0 -> lr over first 5 epochs
    "weight_decay":        1e-4,
    "temperature":         0.5,
    "embedding_dim":       128,
    "pretrained":          True,
    "num_workers":         0,
    "checkpoint_dir":      "models",
    "metrics_dir":         "outputs/metrics",
    "save_every":          5,
    "data_source":         "local",
    "use_amp":             True,
    "early_stop_patience": 8,
    "seed":                42,
    "linear_probe":        True,
    "probe_test_size":     0.2,
    "probe_max_samples":   500,

    # ── A: forgery-aware augmentations ────────────────────────────────────
    "forgery_aware_aug":   True,    # uses get_forgery_aware_transform when True

    # ── B: SupCon loss (fake/real labels guide the contrastive objective) ─
    "use_supcon":          True,    # if True, replace NT-Xent with SupCon
    "supcon_temperature":  0.2,     # 0.1 was too peaky; 0.2-0.5 is more stable
    "supcon_label_source": "label", # "label" (binary) - more balanced batches
                                    # than "ctype" with the heavily skewed
                                    # 1077 / 1000 / 145 distribution

    # ── C: multi-task ctype head (auxiliary CE loss) ──────────────────────
    "use_multitask":       True,    # if True, encoder predicts ctype too
    "ctype_loss_weight":   0.5,     # 1.0 was destabilising; 0.5 keeps the
                                    # supervised signal strong without
                                    # dominating the contrastive objective
    "num_ctypes":          NUM_CTYPES,
}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    contrastive_fn,
    optimizer,
    device,
    scaler=None,
    *,
    use_supcon: bool = False,
    supcon_label_source: str = "label",
    use_multitask: bool = False,
    ctype_loss_fn=None,
    ctype_loss_weight: float = 0.0,
) -> dict:
    """Run one epoch and return mean loss components.

    Returns dict with keys: total, contrastive, multitask (if enabled).
    """
    model.train()
    n_batches = 0
    sum_total = 0.0
    sum_contr = 0.0
    sum_mt    = 0.0

    for view1, view2, label, ctype in tqdm(loader, desc="  batch", leave=False):
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        ctype = ctype.to(device, non_blocking=True)
        sup_labels = ctype if supcon_label_source == "ctype" else label

        optimizer.zero_grad(set_to_none=True)

        def compute():
            if use_multitask:
                _, z_i, ctype_logits_i = model.forward_multitask(view1)
                _, z_j, ctype_logits_j = model.forward_multitask(view2)
            else:
                _, z_i = model(view1)
                _, z_j = model(view2)
                ctype_logits_i = ctype_logits_j = None

            l_contr = contrastive_fn(z_i, z_j, sup_labels) if use_supcon \
                      else contrastive_fn(z_i, z_j)

            l_mt = torch.zeros((), device=device)
            if use_multitask and ctype_loss_fn is not None:
                l_mt = 0.5 * (ctype_loss_fn(ctype_logits_i, ctype) +
                              ctype_loss_fn(ctype_logits_j, ctype))

            l_total = l_contr + ctype_loss_weight * l_mt
            return l_total, l_contr, l_mt

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                l_total, l_contr, l_mt = compute()
            scaler.scale(l_total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            l_total, l_contr, l_mt = compute()
            l_total.backward()
            optimizer.step()

        sum_total += l_total.item()
        sum_contr += l_contr.item()
        sum_mt    += l_mt.item()
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "total":       sum_total / n,
        "contrastive": sum_contr / n,
        "multitask":   sum_mt    / n,
    }


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
    use_local = config.get("data_source", "local") == "local"
    use_forgery_aug = bool(config.get("forgery_aware_aug", True)) and use_local

    if use_forgery_aug:
        transform = get_forgery_aware_transform(image_size=config["image_size"])
        logger.info("Augmentation: forgery-aware (JPEG re-compress, patch erase, mild color)")
    else:
        transform = get_contrastive_transform(image_size=config["image_size"])
        logger.info("Augmentation: generic SimCLR (heavy color jitter, grayscale)")

    if use_local:
        logger.info("Loading local fake/real dataset (templates/) ...")
        loader = get_local_contrastive_loader(
            transform=transform,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
    else:
        logger.info("Loading HuggingFace dataset ...")
        raw = load_hf_dataset()
        split_name = "train" if "train" in raw else list(raw.keys())[0]
        logger.info("Using split: %s  (%d samples)", split_name, len(raw[split_name]))
        loader = get_contrastive_loader(
            raw[split_name],
            transform=transform,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )

    logger.info("DataLoader ready: %d batches x batch_size %d", len(loader), config["batch_size"])

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
    use_multitask = bool(config.get("use_multitask", False)) and use_local
    use_supcon    = bool(config.get("use_supcon",    False)) and use_local

    model = SimCLRModel(
        embedding_dim=config["embedding_dim"],
        pretrained=config["pretrained"],
        num_ctypes=int(config["num_ctypes"]) if use_multitask else 0,
    ).to(device)

    trainable, total = count_parameters(model)
    logger.info("Model params - trainable: %s / total: %s", f"{trainable:,}", f"{total:,}")

    if use_supcon:
        contrastive_fn = SupConLoss(temperature=config.get("supcon_temperature", 0.1))
        logger.info("Contrastive loss: SupCon (label_source=%s, T=%.2f)",
                    config.get("supcon_label_source", "label"),
                    config.get("supcon_temperature", 0.1))
    else:
        contrastive_fn = NTXentLoss(temperature=config["temperature"])
        logger.info("Contrastive loss: NT-Xent (T=%.2f)", config["temperature"])

    ctype_loss_fn = None
    if use_multitask:
        # Inverse-frequency class weights computed once from the loader's records
        weights = compute_ctype_class_weights(loader.dataset.records)
        weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
        ctype_loss_fn = torch.nn.CrossEntropyLoss(weight=weight_t)
        logger.info(
            "Multi-task ctype head enabled (weights=%s, alpha=%.2f)",
            [round(w, 3) for w in weights], config["ctype_loss_weight"],
        )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    # Linear warmup -> cosine decay (standard SimCLR/MoCo recipe)
    warmup_e  = max(int(config.get("warmup_epochs", 0)), 0)
    total_e   = config["epochs"]
    if warmup_e > 0 and warmup_e < total_e:
        warmup    = LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / warmup_e)
        cosine    = CosineAnnealingLR(optimizer, T_max=max(total_e - warmup_e, 1))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_e])
        logger.info("LR schedule: linear warmup (%d ep) -> cosine over %d ep, peak lr=%.1e",
                    warmup_e, total_e - warmup_e, config["lr"])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_e)
        logger.info("LR schedule: cosine over %d ep, peak lr=%.1e", total_e, config["lr"])

    # Mixed-precision scaler (CUDA only)
    use_amp = bool(config.get("use_amp", True)) and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler(device="cuda") if use_amp else None
    logger.info("Mixed precision (AMP): %s", "enabled" if use_amp else "disabled")

    early_stop = EarlyStopping(patience=config.get("early_stop_patience", 8), mode="min")

    # ── Metrics log ──────────────────────────────────────────────────────────
    metrics_path = Path(config["metrics_dir"]) / "train_loss.csv"
    header = ["epoch", "loss_total", "loss_contrastive", "loss_multitask", "lr", "elapsed_s"]
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
        losses = train_one_epoch(
            model, loader, contrastive_fn, optimizer, device,
            scaler=amp_scaler,
            use_supcon=use_supcon,
            supcon_label_source=config.get("supcon_label_source", "label"),
            use_multitask=use_multitask,
            ctype_loss_fn=ctype_loss_fn,
            ctype_loss_weight=float(config.get("ctype_loss_weight", 0.0)) if use_multitask else 0.0,
        )
        epoch_loss = losses["total"]
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        logger.info(
            "Epoch [%3d/%d]  total=%.4f  contrastive=%.4f  multitask=%.4f  lr=%.2e  elapsed=%.0fs",
            epoch, config["epochs"],
            losses["total"], losses["contrastive"], losses["multitask"],
            current_lr, elapsed,
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
        row = [
            epoch,
            round(losses["total"],       6),
            round(losses["contrastive"], 6),
            round(losses["multitask"],   6),
            round(current_lr, 8),
            round(elapsed, 1),
        ]
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
    ckpt = torch.load(str(best_model_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    return model


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trained_model = train()
    logger.info("train_contrastive.py finished.")
