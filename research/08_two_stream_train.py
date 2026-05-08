"""
research/08_two_stream_train.py
-------------------------------
Train the TwoStreamForgeryNet (RGB + FFT, per-branch transformer fusion)
on the same template-aware 60/20/20 split as the ResNet50 clean run.

Same recipe as research/02_template_split_retrain.py:
  * AdamW lr=1e-4, cosine decay, weight_decay=1e-4
  * 15 epochs (was enough for ResNet50 to converge)
  * AMP, batch 16 (the two-stream model is bigger than R50)
  * Class-balanced binary CE
  * Best checkpoint selected on val ROC-AUC; test untouched until end

Outputs:
    research_outputs/08_two_stream_train.csv
    research_outputs/08_two_stream_test_metrics.csv
    research_outputs/08_two_stream_summary.json
    research_outputs/08_two_stream_vs_resnet50.png
    models/two_stream_best.pth
"""

from __future__ import annotations

# IMPORTANT: sklearn before torch on Windows
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
import sys
import time
from pathlib import Path

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
sys.path.insert(0, str(ROOT / "research"))

from local_dataset_loader import LocalEmbeddingDataset  # noqa: E402
from supervised_finetune import (  # noqa: E402
    get_simple_eval_transform,
    get_simple_train_transform,
)
from two_stream_model import TwoStreamForgeryNet  # noqa: E402
from utils import EarlyStopping, count_parameters, get_device, seed_everything  # noqa: E402

import importlib.util as _ilu  # noqa: E402
_t_spec = _ilu.spec_from_file_location("tmpl", ROOT / "research" / "02_template_split_retrain.py")
_t = _ilu.module_from_spec(_t_spec)  # type: ignore[arg-type]
_t_spec.loader.exec_module(_t)  # type: ignore[union-attr]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


CONFIG = {
    "image_size":          224,
    "batch_size":          16,            # two-stream is 38M params; 16 fits in 4 GB
    "num_workers":         0,
    "seed":                42,
    "epochs":              15,
    "lr":                  1e-4,
    "weight_decay":        1e-4,
    "use_amp":             True,
    "early_stop_patience": 6,
    "best_path":           str(MODELS / "two_stream_best.pth"),
    "val_size":            0.20,
    "test_size":           0.20,
    "embed_dim":           256,
    "num_heads":           4,
    "num_transformer_layers": 2,
    "dropout":             0.2,
}


def make_loaders(cfg: dict):
    train_tf = get_simple_train_transform(cfg["image_size"])
    eval_tf  = get_simple_eval_transform(cfg["image_size"])
    train_ds = LocalEmbeddingDataset(transform=train_tf)
    eval_ds  = LocalEmbeddingDataset(transform=eval_tf)

    splitter_cfg = dict(cfg)
    splitter_cfg["seed"] = cfg["seed"]
    train_idx, val_idx, test_idx = _t.template_aware_split(train_ds.records, splitter_cfg)
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


def train_one_epoch(model, loader, loss_fn, opt, device, scaler) -> float:
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
        # Gradient clip — transformers benefit a lot
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        total += loss.item()
        n += 1
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


def plot_two_stream_vs_resnet50(two_stream_test: dict, out_path: Path):
    """Side-by-side bar chart of TwoStream vs the ResNet50 clean baseline."""
    # Pull the ResNet50 clean numbers from the saved CSV
    r50_csv = OUT / "02_template_test_metrics.csv"
    if not r50_csv.exists():
        logger.warning("ResNet50 baseline CSV missing; skipping comparison plot")
        return
    import pandas as pd
    r50 = pd.read_csv(r50_csv).iloc[0]
    r50_metrics = {k: float(r50[k]) for k in
                   ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]}

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    r50_v = [r50_metrics[m] for m in metrics]
    ts_v  = [two_stream_test[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax.bar(x - w/2, r50_v, w, color="#1565C0", edgecolor="white",
                label="ResNet50 (single-stream)")
    b2 = ax.bar(x + w/2, ts_v,  w, color="#7B1FA2", edgecolor="white",
                label="Two-Stream RGB+FFT + Transformer (this work)")

    for b, v in zip(b1, r50_v):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    for b, v in zip(b2, ts_v):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
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


def main():
    seed_everything(CONFIG["seed"])
    device = get_device()
    train_loader, val_loader, test_loader, all_labels, split_summary = make_loaders(CONFIG)

    model = TwoStreamForgeryNet(
        num_classes=2,
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_transformer_layers=CONFIG["num_transformer_layers"],
        dropout=CONFIG["dropout"],
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

    use_amp = bool(CONFIG["use_amp"]) and device.type == "cuda"
    scaler = GradScaler(device="cuda") if use_amp else None
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    early_stop = EarlyStopping(patience=CONFIG["early_stop_patience"], mode="max")

    csv_path = OUT / "08_two_stream_train.csv"
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

    for ep in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        scheduler.step()
        val_m = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch [%2d/%d] loss=%.4f  val: acc=%.3f recall=%.3f precision=%.3f f1=%.3f roc_auc=%.4f  lr=%.1e  elapsed=%.0fs",
            ep, CONFIG["epochs"], train_loss,
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
                 "val_metrics": val_m, "config": CONFIG},
                CONFIG["best_path"],
            )
            logger.info("  -> new best val ROC-AUC %.4f saved", best_val_auc)

        if early_stop.step(val_m["roc_auc"]):
            logger.info("Early stopping at epoch %d", ep)
            break

    # ── Final TEST eval ─────────────────────────────────────────────────
    ckpt = torch.load(CONFIG["best_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate(model, test_loader, device)

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

    # ── Save artefacts ──────────────────────────────────────────────────
    out_csv = OUT / "08_two_stream_test_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model"] + list(test_m.keys()))
        w.writerow(["TwoStreamForgeryNet"] + [round(v, 4) for v in test_m.values()])

    summary_json = {
        "config":           CONFIG,
        "split_summary":    split_summary,
        "best_val_roc_auc": best_val_auc,
        "best_epoch":       int(ckpt["epoch"]),
        "test_metrics":     test_m,
        "params_total":     total_params,
        "peak_vram_mb":     peak_vram_mb,
        "train_time_s":     train_time_s,
    }
    with open(OUT / "08_two_stream_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    plot_two_stream_vs_resnet50(test_m, OUT / "08_two_stream_vs_resnet50.png")


if __name__ == "__main__":
    main()
