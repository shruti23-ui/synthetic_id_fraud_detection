"""
research/02_template_split_retrain.py
-------------------------------------
Re-train the supervised ResNet50 on a **template-aware** train/val/test split
that eliminates the data leakage discovered in 01_data_leakage_audit.py.

Each of the 374 source templates (e.g. `alb_id_00`) is assigned to exactly
one of {train, val, test}, and EVERY image (real + all fakes derived from
that template) goes into that single split. This kills both the source-
template-leakage channel and the near-duplicate-leakage channel that
inflated the original ResNet50 result to 0.9941 ROC-AUC.

Outputs:
    research_outputs/02_template_split_summary.json
    research_outputs/02_template_split_retrain.log
    research_outputs/02_template_test_metrics.csv
    research_outputs/02_clean_vs_leaky_comparison.png
    models/supervised_resnet50_clean_best.pth
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
    SupervisedResNet50,
    get_simple_train_transform,
    get_simple_eval_transform,
)
from utils import EarlyStopping, count_parameters, get_device, seed_everything  # noqa: E402

# Reuse the exact template_id() logic from the audit script
import importlib.util as _ilu  # noqa: E402
_audit_spec = _ilu.spec_from_file_location(
    "leakage_audit", ROOT / "research" / "01_data_leakage_audit.py"
)
_audit = _ilu.module_from_spec(_audit_spec)  # type: ignore[arg-type]
_audit_spec.loader.exec_module(_audit)        # type: ignore[union-attr]
template_id = _audit.template_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


CONFIG = {
    "image_size":          224,
    "batch_size":          32,
    "num_workers":         0,
    "seed":                42,
    "epochs":              15,
    "lr":                  1e-4,
    "weight_decay":        1e-4,
    "use_amp":             True,
    "early_stop_patience": 6,
    "best_path":           str(MODELS / "supervised_resnet50_clean_best.pth"),
    # Template-aware split: 60/20/20 of TEMPLATES, not images
    "val_size":            0.20,
    "test_size":           0.20,
}


# ---------------------------------------------------------------------------
# Template-aware split
# ---------------------------------------------------------------------------

def template_aware_split(records: list[dict], cfg: dict) -> tuple[list[int], list[int], list[int]]:
    """Group all images by their source template, then assign whole groups
    to train / val / test. This GUARANTEES no template appears across splits.
    """
    # Bucket records by template id
    tpl_to_idx: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        tid = template_id(r)
        tpl_to_idx.setdefault(tid, []).append(i)

    template_ids = sorted(tpl_to_idx.keys())
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(template_ids)

    # Try to balance class proportions per split, not just template count.
    # We do a greedy assignment: walk shuffled templates, place each into the
    # split that currently has the fewest records (capped by target sizes).
    n_total = sum(len(v) for v in tpl_to_idx.values())
    target_test  = int(round(n_total * cfg["test_size"]))
    target_val   = int(round(n_total * cfg["val_size"]))
    target_train = n_total - target_test - target_val

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    targets = {"train": target_train, "val": target_val, "test": target_test}

    for tid in template_ids:
        # Place into the split that is most under its target
        deficits = {k: targets[k] - len(splits[k]) for k in splits}
        chosen = max(deficits, key=lambda k: deficits[k])
        splits[chosen].extend(tpl_to_idx[tid])

    # Verify: zero template overlap
    train_tpl = {template_id(records[i]) for i in splits["train"]}
    val_tpl   = {template_id(records[i]) for i in splits["val"]}
    test_tpl  = {template_id(records[i]) for i in splits["test"]}
    assert not (train_tpl & val_tpl), "leak between train and val"
    assert not (train_tpl & test_tpl), "leak between train and test"
    assert not (val_tpl   & test_tpl), "leak between val and test"

    return splits["train"], splits["val"], splits["test"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def make_loaders(cfg: dict):
    train_tf = get_simple_train_transform(cfg["image_size"])
    eval_tf  = get_simple_eval_transform(cfg["image_size"])
    train_ds = LocalEmbeddingDataset(transform=train_tf)
    eval_ds  = LocalEmbeddingDataset(transform=eval_tf)

    records = train_ds.records
    train_idx, val_idx, test_idx = template_aware_split(records, cfg)
    labels = np.array([r["label"] for r in records])

    summary = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        ys = labels[idx]
        summary[name] = {
            "n":            len(idx),
            "real":         int((ys == 0).sum()),
            "fake":         int((ys == 1).sum()),
            "n_templates":  len({template_id(records[i]) for i in idx}),
        }
        logger.info(
            "%s: n=%d (real=%d, fake=%d) over %d unique templates",
            name, summary[name]["n"], summary[name]["real"],
            summary[name]["fake"], summary[name]["n_templates"],
        )

    train_loader = DataLoader(
        Subset(train_ds, sorted(train_idx)),
        batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
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
# Train / eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, opt, device, scaler=None) -> float:
    model.train()
    total, n = 0.0, 0
    for images, labels in tqdm(loader, desc="  batch", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast(device_type="cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict:
    model.eval()
    probs, preds, labels_all = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()
        probs.append(p)
        preds.append(y_pred)
        labels_all.append(labels.numpy())
    y_prob = np.concatenate(probs)
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels_all)
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

def main():
    seed_everything(CONFIG["seed"])
    device = get_device()
    train_loader, val_loader, test_loader, all_labels, split_summary = make_loaders(CONFIG)

    model = SupervisedResNet50(num_classes=2, dropout=0.2).to(device)
    _, total_params = count_parameters(model)
    logger.info("ResNet50 params: %s", f"{total_params:,}")

    pos = (all_labels == 1).sum()
    neg = (all_labels == 0).sum()
    weight = torch.tensor(
        [(pos + neg) / (2 * neg), (pos + neg) / (2 * pos)],
        dtype=torch.float32, device=device,
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    logger.info("CE class weights = [real=%.3f, fake=%.3f]", weight[0].item(), weight[1].item())

    use_amp = bool(CONFIG["use_amp"]) and device.type == "cuda"
    scaler = GradScaler(device="cuda") if use_amp else None
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    early_stop = EarlyStopping(patience=CONFIG["early_stop_patience"], mode="max")

    csv_path = OUT / "02_template_train.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss",
            "val_accuracy", "val_recall", "val_precision", "val_f1",
            "val_roc_auc", "val_pr_auc", "lr", "elapsed_s",
        ])

    best_val_auc = -1.0
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("CLEAN training run (template-aware 60/20/20 split)")
    logger.info("=" * 60)

    for ep in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        scheduler.step()
        val_m = evaluate_loader(model, val_loader, device)
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch [%2d/%d] loss=%.4f  val: acc=%.3f recall=%.3f precision=%.3f f1=%.3f roc_auc=%.4f  lr=%.1e  elapsed=%.0fs",
            ep, CONFIG["epochs"], train_loss,
            val_m["accuracy"], val_m["recall"], val_m["precision"],
            val_m["f1"], val_m["roc_auc"], cur_lr, elapsed,
        )

        with open(csv_path, "a", newline="") as f:
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
                {"model_state_dict": model.state_dict(), "epoch": ep, "val_metrics": val_m},
                CONFIG["best_path"],
            )
            logger.info("  -> new best val ROC-AUC %.4f saved", best_val_auc)

        if early_stop.step(val_m["roc_auc"]):
            logger.info("Early stopping at epoch %d", ep)
            break

    # ── Final TEST eval ─────────────────────────────────────────────────────
    ckpt = torch.load(CONFIG["best_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate_loader(model, test_loader, device)

    logger.info("=" * 60)
    logger.info("CLEAN test set (no template leakage)")
    logger.info("=" * 60)
    logger.info(
        "TEST | acc=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f pr_auc=%.4f",
        test_m["accuracy"], test_m["precision"], test_m["recall"],
        test_m["f1"], test_m["roc_auc"], test_m["pr_auc"],
    )

    # ── Save artefacts ──────────────────────────────────────────────────────
    out_csv = OUT / "02_template_test_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + list(test_m.keys()))
        w.writerow(["SupervisedResNet50_clean"] + [round(v, 4) for v in test_m.values()])
    logger.info("Saved: %s", out_csv)

    summary_json = {
        "config":         CONFIG,
        "split_summary":  split_summary,
        "best_val_roc_auc": best_val_auc,
        "best_epoch":     int(ckpt["epoch"]),
        "test_metrics":   test_m,
    }
    with open(OUT / "02_template_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    logger.info("Saved: %s", OUT / "02_template_split_summary.json")

    # ── Comparison plot: leaky vs clean ─────────────────────────────────────
    leaky_csv = ROOT / "outputs" / "metrics" / "supervised_results.csv"
    if leaky_csv.exists():
        import pandas as pd
        leaky = pd.read_csv(leaky_csv).iloc[0]
        leaky_metrics = {
            k: float(leaky[k]) for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        }
        plot_clean_vs_leaky(leaky_metrics, test_m, OUT / "02_clean_vs_leaky_comparison.png")


def plot_clean_vs_leaky(leaky: dict, clean: dict, out_path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    leaky_vals = [leaky[m] for m in metrics]
    clean_vals = [clean[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(metrics))
    w = 0.36
    bars1 = ax.bar(x - w/2, leaky_vals, w, color="#D32F2F", edgecolor="white",
                   label="Image-level split (leaky baseline)")
    bars2 = ax.bar(x + w/2, clean_vals, w, color="#1565C0", edgecolor="white",
                   label="Template-aware split (clean)")

    for b, v in zip(bars1, leaky_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    for b, v in zip(bars2, clean_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)

    delta = [c - lk for lk, c in zip(leaky_vals, clean_vals)]
    avg_delta = sum(delta) / len(delta)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "-").upper() for m in metrics], fontsize=11)
    ax.set_ylabel("Test-set metric value", fontsize=12)
    ax.set_ylim(0, 1.10)
    ax.axhline(0.5, linestyle=":", color="gray", lw=1, alpha=0.5)
    ax.set_title(
        "Supervised ResNet50: leaky vs clean test split\n"
        f"average drop on going to template-aware split: {avg_delta:+.3f}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
