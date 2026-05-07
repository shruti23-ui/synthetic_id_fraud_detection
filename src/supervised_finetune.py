"""
supervised_finetune.py
----------------------
End-to-end supervised fine-tuning of a ResNet50 classifier on the local
fake / real ID dataset. This is the *strong baseline* for the thesis.

Design (deliberately simple and proven):

  - ResNet50 with ImageNet-pretrained weights, full network unfrozen
  - Single-stage end-to-end fine-tune (no head-only warmup phase)
  - lr = 1e-4 with cosine decay (standard sweet spot for ResNet fine-tune)
  - Light augmentation only: horizontal flip + small random crop
    (heavier augmentations belong to the contrastive pipeline; here they
    only confuse a model that's still learning the task)
  - Class-balanced binary CE loss to handle the 1222 / 1000 imbalance
  - Same 60 / 20 / 20 stratified split as classifier.py / evaluate.py
  - Best checkpoint selected on validation ROC-AUC (test never seen during
    selection)
  - Mixed-precision (AMP) for VRAM headroom on the 4 GB card

Why ResNet50 (not ResNet18, not Xception):
  - ResNet18 is the canonical SimCLR backbone (see contrastive_model.py),
    so we keep it for the self-supervised vs supervised comparison.
  - ResNet50 has materially stronger ImageNet features and is the
    standard "modern strong CNN baseline" in transfer-learning thesis work.
  - Xception (tested) collapsed under the two-stage + heavy-aug recipe.

Outputs:
  models/supervised_resnet50_best.pth      best checkpoint (val ROC-AUC)
  outputs/metrics/supervised_train.csv     per-epoch metrics
  outputs/metrics/supervised_results.csv   final test row for thesis table
"""

# IMPORTANT: sklearn before torch on Windows to avoid libomp DLL crash.
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import csv
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_dataset_loader import LocalEmbeddingDataset
from utils import EarlyStopping, count_parameters, get_device, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "image_size":          224,
    "batch_size":          32,
    "num_workers":         0,
    "seed":                42,

    # Single-stage end-to-end fine-tune
    "epochs":              15,
    "lr":                  1e-4,         # proven sweet spot for ResNet fine-tune
    "weight_decay":        1e-4,

    # 60 / 20 / 20 stratified split (matches classifier.py)
    "val_size":            0.20,
    "test_size":           0.20,

    "use_amp":             True,
    "early_stop_patience": 6,            # patience on val ROC-AUC

    # Outputs
    "checkpoint_dir":      "models",
    "metrics_dir":         "outputs/metrics",
    "best_path":           "models/supervised_resnet50_best.pth",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Augmentations (deliberately light)
# ---------------------------------------------------------------------------

def get_simple_train_transform(image_size: int = 224) -> transforms.Compose:
    """Light, stable augmentations. Just HFlip + small RandomResizedCrop.

    No colour jitter, no patch erasing, no JPEG re-compression. We want the
    model to learn the easy signal first; heavier regularisation can come
    in a follow-up experiment once we know it works.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_simple_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model: ResNet50 with a 2-class head
# ---------------------------------------------------------------------------

class SupervisedResNet50(nn.Module):
    """ImageNet-pretrained ResNet50 with a 2-class binary head."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2  # the better V2 weights
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features            # 2048
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loaders (same 60 / 20 / 20 stratified split as classifier.py)
# ---------------------------------------------------------------------------

def make_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    train_tf = get_simple_train_transform(config["image_size"])
    eval_tf  = get_simple_eval_transform(config["image_size"])

    train_ds = LocalEmbeddingDataset(transform=train_tf)
    eval_ds  = LocalEmbeddingDataset(transform=eval_tf)

    labels = np.array([r["label"] for r in train_ds.records])

    rng = np.random.default_rng(config["seed"])
    train_idx, val_idx, test_idx = [], [], []
    for cls in (0, 1):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_test = int(round(n * config["test_size"]))
        n_val  = int(round(n * config["val_size"]))
        test_idx.extend(cls_idx[:n_test].tolist())
        val_idx.extend(cls_idx[n_test:n_test + n_val].tolist())
        train_idx.extend(cls_idx[n_test + n_val:].tolist())

    train_idx, val_idx, test_idx = sorted(train_idx), sorted(val_idx), sorted(test_idx)

    logger.info(
        "Stratified 60/20/20: train=%d  val=%d  test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        ys = labels[idx]
        logger.info("  %s: real=%d  fake=%d", name, int((ys == 0).sum()), int((ys == 1).sum()))

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx),
        batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Subset(eval_ds, test_idx),
        batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, labels


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler=None) -> float:
    model.train()
    total = 0.0
    n = 0
    for images, labels in tqdm(loader, desc="  batch", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast(device_type="cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict:
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
        "pr_auc":    average_precision_score(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_supervised(config: dict = CONFIG) -> dict:
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["metrics_dir"]).mkdir(parents=True, exist_ok=True)

    seed_everything(config["seed"])
    device = get_device()

    train_loader, val_loader, test_loader, all_labels = make_loaders(config)
    logger.info(
        "Loaders: train=%d batches  val=%d batches  test=%d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = SupervisedResNet50(num_classes=2, dropout=0.2).to(device)
    _, total = count_parameters(model)
    logger.info("ResNet50 — total params: %s", f"{total:,}")

    # Class-balanced CE — slight 1222/1000 imbalance
    pos = (all_labels == 1).sum()
    neg = (all_labels == 0).sum()
    weight = torch.tensor(
        [(pos + neg) / (2 * neg), (pos + neg) / (2 * pos)],
        dtype=torch.float32, device=device,
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    logger.info("CE class weights = [real=%.3f, fake=%.3f]", weight[0].item(), weight[1].item())

    use_amp = bool(config.get("use_amp", True)) and device.type == "cuda"
    scaler = GradScaler(device="cuda") if use_amp else None
    logger.info("Mixed precision (AMP): %s", "enabled" if use_amp else "disabled")

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
    logger.info("Optimiser: AdamW lr=%.1e weight_decay=%.1e | cosine over %d epochs",
                config["lr"], config["weight_decay"], config["epochs"])

    metrics_csv = Path(config["metrics_dir"]) / "supervised_train.csv"
    with open(metrics_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "lr", "train_loss",
            "val_accuracy", "val_precision", "val_recall", "val_f1",
            "val_roc_auc", "val_pr_auc", "elapsed_s",
        ])

    best_val_auc = -1.0
    best_path = Path(config["best_path"])
    early_stop = EarlyStopping(patience=config["early_stop_patience"], mode="max")
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("End-to-end supervised fine-tune (ResNet50, %d epochs)", config["epochs"])
    logger.info("=" * 60)

    for ep in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        scheduler.step()
        val_m = evaluate_loader(model, val_loader, device)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch [%2d/%d] loss=%.4f  val: acc=%.3f recall=%.3f precision=%.3f f1=%.3f roc_auc=%.4f  lr=%.1e  elapsed=%.0fs",
            ep, config["epochs"], train_loss,
            val_m["accuracy"], val_m["recall"], val_m["precision"],
            val_m["f1"], val_m["roc_auc"], current_lr, elapsed,
        )

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, round(current_lr, 8), round(train_loss, 6),
                round(val_m["accuracy"],  4), round(val_m["precision"], 4),
                round(val_m["recall"],    4), round(val_m["f1"],        4),
                round(val_m["roc_auc"],   4), round(val_m["pr_auc"],    4),
                round(elapsed, 1),
            ])

        if val_m["roc_auc"] > best_val_auc:
            best_val_auc = val_m["roc_auc"]
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "epoch": ep, "val_metrics": val_m},
                str(best_path),
            )
            logger.info("  -> new best val ROC-AUC %.4f saved", best_val_auc)

        if early_stop.step(val_m["roc_auc"]):
            logger.info("Early stopping at epoch %d (no val improvement for %d epochs)",
                        ep, config["early_stop_patience"])
            break

    # ── Final TEST evaluation ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading best checkpoint and evaluating on TEST set")
    logger.info("=" * 60)
    ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate_loader(model, test_loader, device)

    logger.info(
        "TEST | acc=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f pr_auc=%.4f",
        test_m["accuracy"], test_m["precision"], test_m["recall"],
        test_m["f1"], test_m["roc_auc"], test_m["pr_auc"],
    )

    results_csv = Path(config["metrics_dir"]) / "supervised_results.csv"
    with open(results_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + list(test_m.keys()))
        w.writerow(["SupervisedResNet50_e2e"] + [round(v, 4) for v in test_m.values()])
    logger.info("Final test metrics -> %s", results_csv)
    logger.info("Best checkpoint    -> %s (val ROC-AUC %.4f)", best_path, best_val_auc)

    return {"val_roc_auc_best": best_val_auc, "test_metrics": test_m}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_supervised()
    logger.info("supervised_finetune.py finished.")
