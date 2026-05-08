"""
src/train_two_stream.py
-----------------------
Training driver for the TwoStreamForgeryNet (RGB + FFT, per-branch
Transformer fusion). Importable from main.py as Stage 9 of the master
pipeline; also wrapped by research/08_two_stream_train.py for standalone
research runs.

Recipe (matches the supervised ResNet50 baseline so comparisons are fair):
  * AdamW lr=1e-4, cosine decay, weight_decay=1e-4
  * 15 epochs, batch=16, AMP, gradient clipping max-norm 1.0
  * Class-balanced binary cross-entropy
  * Best checkpoint selected on validation ROC-AUC; test set untouched
    until the final report
  * **Template-aware** 60/20/20 split (no template appears across splits)

Outputs:
    research_outputs/08_two_stream_train.csv          per-epoch curves
    research_outputs/08_two_stream_test_metrics.csv   final test row
    research_outputs/08_two_stream_summary.json       full numerical record
    research_outputs/08_two_stream_vs_resnet50.png    headline comparison plot
    models/two_stream_best.pth                        best checkpoint
"""

from __future__ import annotations

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
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from local_dataset_loader import LocalEmbeddingDataset  # noqa: E402
from supervised_finetune import (  # noqa: E402
    get_simple_eval_transform,
    get_simple_train_transform,
)
from template_split import template_aware_split  # noqa: E402
from two_stream_model import TwoStreamForgeryNet  # noqa: E402
from utils import (  # noqa: E402
    EarlyStopping,
    count_parameters,
    get_device,
    make_param_groups,
    seed_everything,
)

logger = logging.getLogger(__name__)


CONFIG = {
    "image_size":          224,
    "batch_size":          16,
    "num_workers":         0,
    "seed":                42,
    "epochs":              20,            # bumped from 15: previous best ckpt
                                          # landed at epoch 14/15, i.e. cosine
                                          # LR had not finished annealing
    "lr":                  1e-4,
    "weight_decay":        1e-4,
    "use_amp":             True,
    # Early stopping disabled (patience=0); cosine LR runs to completion.
    "early_stop_patience": 0,
    "best_path":           str(ROOT / "models" / "two_stream_best.pth"),
    "val_size":            0.20,
    "test_size":           0.20,
    "embed_dim":           256,
    "num_heads":           4,
    "num_transformer_layers": 2,
    "dropout":             0.2,
    "out_dir":             str(ROOT / "research_outputs"),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_loaders(cfg: dict):
    train_tf = get_simple_train_transform(cfg["image_size"])
    eval_tf  = get_simple_eval_transform(cfg["image_size"])
    train_ds = LocalEmbeddingDataset(transform=train_tf)
    eval_ds  = LocalEmbeddingDataset(transform=eval_tf)

    train_idx, val_idx, test_idx = template_aware_split(train_ds.records, cfg)
    labels = np.array([r["label"] for r in train_ds.records])

    summary = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        ys = labels[idx]
        summary[name] = {"n": len(idx), "real": int((ys == 0).sum()),
                         "fake": int((ys == 1).sum())}
        logger.info("%s: n=%d (real=%d, fake=%d)", name, len(idx),
                    int((ys == 0).sum()), int((ys == 1).sum()))

    train_loader = DataLoader(
        Subset(train_ds, sorted(train_idx)),
        batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        Subset(eval_ds, sorted(val_idx)),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Subset(eval_ds, sorted(test_idx)),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, labels, summary


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(model, loader, loss_fn, opt, device, scaler) -> float:
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


@torch.no_grad()
def _evaluate(model, loader, device) -> dict:
    model.eval()
    probs, preds, labels_all = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type="cuda"):
            logits = model(x).float()
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels_all.append(y.numpy())
    yp = np.concatenate(probs)
    pr = np.concatenate(preds)
    yt = np.concatenate(labels_all)
    return {
        "accuracy":  float(accuracy_score(yt, pr)),
        "precision": float(precision_score(yt, pr, zero_division=0)),
        "recall":    float(recall_score(yt, pr, zero_division=0)),
        "f1":        float(f1_score(yt, pr, zero_division=0)),
        "roc_auc":   float(roc_auc_score(yt, yp)),
        "pr_auc":    float(average_precision_score(yt, yp)),
    }


def _plot_two_stream_vs_resnet50(two_stream_test: dict, out_path: Path) -> None:
    r50_csv = ROOT / "research_outputs" / "02_template_test_metrics.csv"
    if not r50_csv.exists():
        logger.warning("ResNet50 baseline CSV missing; skipping comparison plot")
        return
    import pandas as pd
    r50 = pd.read_csv(r50_csv).iloc[0]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    r50_v = [float(r50[m]) for m in metrics]
    ts_v  = [two_stream_test[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax.bar(x - w / 2, r50_v, w, color="#1565C0", edgecolor="white",
                label="ResNet50 (single-stream)")
    b2 = ax.bar(x + w / 2, ts_v, w, color="#7B1FA2", edgecolor="white",
                label="Two-Stream RGB+FFT + Transformer (this work)")
    for b, v in zip(b1, r50_v):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    for b, v in zip(b2, ts_v):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    delta = [t - r for r, t in zip(r50_v, ts_v)]
    avg_delta = sum(delta) / len(delta)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "-").upper() for m in metrics], fontsize=11)
    ax.set_ylabel("Test-set metric value", fontsize=12)
    ax.set_ylim(0.85, 1.02)
    ax.set_title(
        "Two-Stream RGB+FFT (Transformer fusion) vs single-stream ResNet50\n"
        f"both on template-aware 60/20/20 split,  average delta = {avg_delta:+.4f}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train_two_stream(config: dict = CONFIG) -> dict:
    """Train TwoStreamForgeryNet end-to-end on the template-aware split.

    Returns a dict with keys: best_val_roc_auc, test_metrics, params_total,
    train_time_s, peak_vram_mb.
    """
    out_dir = Path(config.get("out_dir", ROOT / "research_outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(config["best_path"]).parent.mkdir(parents=True, exist_ok=True)

    seed_everything(config["seed"])
    device = get_device()
    train_loader, val_loader, test_loader, all_labels, split_summary = make_loaders(config)

    model = TwoStreamForgeryNet(
        num_classes=2,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_transformer_layers=config["num_transformer_layers"],
        dropout=config["dropout"],
    ).to(device)
    _, total_params = count_parameters(model)
    logger.info("TwoStreamForgeryNet params: %s", f"{total_params:,}")

    pos = (all_labels == 1).sum()
    neg = (all_labels == 0).sum()
    weight = torch.tensor(
        [(pos + neg) / (2 * neg), (pos + neg) / (2 * pos)],
        dtype=torch.float32, device=device,
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    logger.info("CE class weights = [real=%.3f, fake=%.3f]",
                weight[0].item(), weight[1].item())

    use_amp = bool(config["use_amp"]) and device.type == "cuda"
    scaler = GradScaler(device="cuda") if use_amp else None
    # Standard AdamW recipe: weight decay applied to weight matrices but NOT
    # to BatchNorm/LayerNorm gammas-betas or to biases.
    optimizer = optim.AdamW(
        make_param_groups(model, weight_decay=config["weight_decay"]),
        lr=config["lr"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
    # Early stopping is opt-in; default disabled so cosine LR fully decays.
    early_stop = (
        EarlyStopping(patience=config["early_stop_patience"], mode="max")
        if config.get("early_stop_patience", 0) > 0 else None
    )

    csv_path = out_dir / "08_two_stream_train.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss",
            "val_accuracy", "val_recall", "val_precision", "val_f1",
            "val_roc_auc", "val_pr_auc", "lr", "elapsed_s",
        ])

    best_val_auc = -1.0
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Training TwoStreamForgeryNet on template-aware split")
    logger.info("=" * 60)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for ep in range(1, config["epochs"] + 1):
        train_loss = _train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        scheduler.step()
        val_m = _evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch [%2d/%d] loss=%.4f  val: acc=%.3f recall=%.3f precision=%.3f f1=%.3f roc_auc=%.4f  lr=%.1e  elapsed=%.0fs",
            ep, config["epochs"], train_loss,
            val_m["accuracy"], val_m["recall"], val_m["precision"],
            val_m["f1"], val_m["roc_auc"], cur_lr, elapsed,
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                ep, round(train_loss, 6),
                round(val_m["accuracy"], 4),  round(val_m["recall"], 4),
                round(val_m["precision"], 4), round(val_m["f1"], 4),
                round(val_m["roc_auc"], 4),   round(val_m["pr_auc"], 4),
                round(cur_lr, 8), round(elapsed, 1),
            ])

        if val_m["roc_auc"] > best_val_auc:
            best_val_auc = val_m["roc_auc"]
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": ep,
                 "val_metrics": val_m, "config": config},
                config["best_path"],
            )
            logger.info("  -> new best val ROC-AUC %.4f saved", best_val_auc)

        if early_stop is not None and early_stop.step(val_m["roc_auc"]):
            logger.info("Early stopping at epoch %d", ep)
            break

    # ── Final TEST eval ────────────────────────────────────────────────────
    ckpt = torch.load(config["best_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = _evaluate(model, test_loader, device)

    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0
    train_time_s = time.time() - t0

    logger.info("=" * 60)
    logger.info("TwoStream TEST set (template-aware, no leakage)")
    logger.info("=" * 60)
    logger.info(
        "TEST | acc=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f pr_auc=%.4f",
        test_m["accuracy"], test_m["precision"], test_m["recall"],
        test_m["f1"], test_m["roc_auc"], test_m["pr_auc"],
    )
    logger.info("Best ckpt at epoch %d | best val ROC-AUC %.4f", ckpt["epoch"], best_val_auc)
    logger.info("Total training time: %.1fs | peak VRAM: %.0f MB", train_time_s, peak_vram_mb)

    # Persist artefacts
    out_csv = out_dir / "08_two_stream_test_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model"] + list(test_m.keys()))
        w.writerow(["TwoStreamForgeryNet"] + [round(v, 4) for v in test_m.values()])

    summary_json = {
        "config":           config,
        "split_summary":    split_summary,
        "best_val_roc_auc": best_val_auc,
        "best_epoch":       int(ckpt["epoch"]),
        "test_metrics":     test_m,
        "params_total":     total_params,
        "peak_vram_mb":     peak_vram_mb,
        "train_time_s":     train_time_s,
    }
    with open(out_dir / "08_two_stream_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    _plot_two_stream_vs_resnet50(test_m, out_dir / "08_two_stream_vs_resnet50.png")

    return summary_json


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    train_two_stream()
