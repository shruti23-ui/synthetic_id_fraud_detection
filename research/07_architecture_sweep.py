"""
research/07_architecture_sweep.py
---------------------------------
Train and compare 5 modern backbones under the *same* template-aware split,
the *same* hyperparameters and the *same* light augmentation recipe.

Backbones (all ImageNet-pretrained, all 2-class binary head appended):
  1. ResNet50           (control / our existing supervised baseline)
  2. ConvNeXt-Tiny      (modern CNN, 2022)
  3. EfficientNetV2-S   (scaling-efficient CNN, 2021)
  4. ViT-S/16           (small Vision Transformer)
  5. Swin-T             (Swin Transformer Tiny)

Each model is trained for 10 epochs (with --quick) or 15 (default), best
checkpoint saved on val ROC-AUC, final test metrics + parameter count + GPU
peak memory + per-image inference latency reported.

Outputs:
    research_outputs/07_architecture_sweep.csv     publication-style table
    research_outputs/07_architecture_sweep.png     bar chart
    models/sweep_<name>_best.pth                   per-model checkpoint
"""

from __future__ import annotations

# IMPORTANT: sklearn before torch on Windows
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from local_dataset_loader import LocalEmbeddingDataset  # noqa: E402
from supervised_finetune import (  # noqa: E402
    CONFIG as BASE_CFG,
    SupervisedResNet50,
    get_simple_eval_transform,
    get_simple_train_transform,
)
from utils import EarlyStopping, count_parameters, get_device, seed_everything  # noqa: E402

from template_split import template_aware_split  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

def build_resnet50(num_classes: int = 2) -> nn.Module:
    return SupervisedResNet50(num_classes=num_classes, dropout=0.2)


def build_timm(name: str, num_classes: int = 2, dropout: float = 0.2) -> nn.Module:
    return timm.create_model(name, pretrained=True, num_classes=num_classes, drop_rate=dropout)


BACKBONES: dict[str, dict] = {
    "ResNet50":          {"builder": build_resnet50,                                     "img_size": 224},
    "ConvNeXt-Tiny":     {"builder": lambda: build_timm("convnext_tiny",  num_classes=2),"img_size": 224},
    "EfficientNetV2-S":  {"builder": lambda: build_timm("tf_efficientnetv2_s", num_classes=2), "img_size": 224},
    "ViT-S/16":          {"builder": lambda: build_timm("vit_small_patch16_224", num_classes=2), "img_size": 224},
    "Swin-T":            {"builder": lambda: build_timm("swin_tiny_patch4_window7_224", num_classes=2), "img_size": 224},
}


# ---------------------------------------------------------------------------
# Data: template-aware split (reuse from script 02)
# ---------------------------------------------------------------------------

def make_loaders(image_size: int, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    train_tf = get_simple_train_transform(image_size)
    eval_tf  = get_simple_eval_transform(image_size)
    train_ds = LocalEmbeddingDataset(transform=train_tf)
    eval_ds  = LocalEmbeddingDataset(transform=eval_tf)

    cfg = dict(BASE_CFG); cfg["seed"] = seed; cfg["val_size"] = 0.20; cfg["test_size"] = 0.20
    train_idx, val_idx, test_idx = template_aware_split(train_ds.records, cfg)
    labels = np.array([r["label"] for r in train_ds.records])

    train_loader = DataLoader(
        Subset(train_ds, sorted(train_idx)), batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        Subset(eval_ds, sorted(val_idx)), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Subset(eval_ds, sorted(test_idx)), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, labels


# ---------------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, opt, device, scaler) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True).long()
        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda"):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, preds, labels_all = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type="cuda"):
            logits = model(x).float()
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels_all.append(y.numpy())
    yp = np.concatenate(probs); pr = np.concatenate(preds); yt = np.concatenate(labels_all)
    return {
        "accuracy":  float(accuracy_score(yt, pr)),
        "precision": float(precision_score(yt, pr, zero_division=0)),
        "recall":    float(recall_score(yt, pr, zero_division=0)),
        "f1":        float(f1_score(yt, pr, zero_division=0)),
        "roc_auc":   float(roc_auc_score(yt, yp)),
        "pr_auc":    float(average_precision_score(yt, yp)),
    }


# ---------------------------------------------------------------------------
# Per-image inference latency
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_latency(model, device, image_size: int, n_iter: int = 100) -> float:
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    # warm up
    for _ in range(5):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0 / n_iter  # ms / image


# ---------------------------------------------------------------------------
# Train one model end-to-end
# ---------------------------------------------------------------------------

def train_one_model(name: str, builder, image_size: int, epochs: int, seed: int,
                    device: torch.device) -> dict:
    logger.info("=" * 60); logger.info("=== Training %s (epochs=%d) ===", name, epochs)
    seed_everything(seed)

    # Use small batch for transformers to fit 4 GB; ConvNets at 32
    is_transformer = name.startswith("ViT") or name.startswith("Swin")
    bs = 16 if is_transformer else 32

    train_loader, val_loader, test_loader, all_labels = make_loaders(image_size, bs, seed)
    model = builder().to(device)
    _, total = count_parameters(model)
    logger.info("%s params: %s", name, f"{total:,}")

    pos = (all_labels == 1).sum(); neg = (all_labels == 0).sum()
    weight = torch.tensor([(pos + neg) / (2 * neg), (pos + neg) / (2 * pos)],
                          dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler(device="cuda")
    es = EarlyStopping(patience=5, mode="max")

    best_val_auc = -1.0
    best_path = MODELS / f"sweep_{name.replace('/', '-').replace(' ', '_')}_best.pth"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for ep in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, opt, device, scaler)
        sched.step()
        val_m = evaluate(model, val_loader, device)
        logger.info(
            "[%s] ep %2d/%d  loss=%.3f  val: acc=%.3f f1=%.3f roc_auc=%.4f  elapsed=%.0fs",
            name, ep, epochs, train_loss, val_m["accuracy"], val_m["f1"], val_m["roc_auc"],
            time.time() - t0,
        )
        if val_m["roc_auc"] > best_val_auc:
            best_val_auc = val_m["roc_auc"]
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": ep, "val_metrics": val_m,
                        "model_name": name}, str(best_path))
        if es.step(val_m["roc_auc"]):
            logger.info("[%s] early stop at epoch %d", name, ep); break

    # Load best and evaluate on TEST
    ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate(model, test_loader, device)

    train_time = time.time() - t0
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0
    latency_ms = measure_latency(model, device, image_size, n_iter=50)

    row = {
        "model":         name,
        "params":        total,
        "best_val_roc":  best_val_auc,
        "test_accuracy": test_m["accuracy"],
        "test_precision": test_m["precision"],
        "test_recall":   test_m["recall"],
        "test_f1":       test_m["f1"],
        "test_roc_auc":  test_m["roc_auc"],
        "test_pr_auc":   test_m["pr_auc"],
        "train_time_s":  train_time,
        "peak_vram_mb":  peak_vram_mb,
        "latency_ms":    latency_ms,
    }
    logger.info("[%s] TEST acc=%.3f roc_auc=%.4f f1=%.3f | %.0fs train | %.1f MB VRAM | %.2f ms/img",
                name, test_m["accuracy"], test_m["roc_auc"], test_m["f1"],
                train_time, peak_vram_mb, latency_ms)
    # Free up memory before next model
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sweep(rows: list[dict], out_path: Path):
    rows_sorted = sorted(rows, key=lambda r: -r["test_roc_auc"])
    names  = [r["model"] for r in rows_sorted]
    metrics = ["test_accuracy", "test_f1", "test_roc_auc", "test_pr_auc"]
    metric_labels = ["Test Accuracy", "Test F1", "Test ROC-AUC", "Test PR-AUC"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
    fig.suptitle("Architecture sweep — template-aware test split",
                 fontsize=14, fontweight="bold")

    cmap = plt.colormaps.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(names))]

    for ax, m, lab in zip(axes, metrics, metric_labels):
        vals = [r[m] for r in rows_sorted]
        ax.bar(range(len(names)), vals, color=colors, edgecolor="white")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(lab if ax is axes[0] else "")
        ax.set_title(lab)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Train each model for 10 epochs instead of 15 (saves ~1/3 the time)")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated list of model names to run (subset for testing)")
    args = p.parse_args()

    epochs = 10 if args.quick else 15
    device = get_device()

    only = set(args.only.split(",")) if args.only else None
    rows = []
    for name, info in BACKBONES.items():
        if only and name not in only: continue
        try:
            row = train_one_model(name, info["builder"], info["img_size"],
                                  epochs=epochs, seed=42, device=device)
            rows.append(row)
        except Exception as exc:
            logger.exception("%s failed: %s", name, exc)
            rows.append({"model": name, "error": str(exc)})

    # CSV table
    csv_path = OUT / "07_architecture_sweep.csv"
    fieldnames = ["model", "params", "best_val_roc", "test_accuracy", "test_precision",
                  "test_recall", "test_f1", "test_roc_auc", "test_pr_auc",
                  "train_time_s", "peak_vram_mb", "latency_ms"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if "error" in r:
                w.writerow({"model": r["model"], "params": -1})
            else:
                row_out = dict(r)
                for k in row_out:
                    if isinstance(row_out[k], float):
                        row_out[k] = round(row_out[k], 4)
                w.writerow(row_out)
    logger.info("Saved: %s", csv_path)

    valid_rows = [r for r in rows if "error" not in r]
    if valid_rows:
        plot_sweep(valid_rows, OUT / "07_architecture_sweep.png")


if __name__ == "__main__":
    main()
