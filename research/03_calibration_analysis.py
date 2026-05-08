"""
research/03_calibration_analysis.py
-----------------------------------
Beyond accuracy: is the supervised ResNet50 well-calibrated?

A model that hits 94 % accuracy can still be *systematically over- or
under-confident*. For a fraud-detection system this matters: a 99.9 %
confidence prediction should actually be wrong only 0.1 % of the time.
We evaluate this with three artefacts:

  1. Reliability diagram + Expected Calibration Error (ECE)
  2. Confidence histogram (predicted-probability distribution)
  3. Per-confidence-bucket accuracy table

We do this for BOTH the leaky and the clean checkpoint so the calibration
picture is paired with the headline number it sits next to.

Outputs:
    research_outputs/03_calibration_<run>.png
    research_outputs/03_confidence_histogram_<run>.png
    research_outputs/03_calibration_metrics.json
where <run> in {leaky, clean}.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score

import json
import logging
import sys
from pathlib import Path
from typing import Optional

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
    make_loaders as leaky_make_loaders,
)
from utils import get_device  # noqa: E402

import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("tmpl_retrain", ROOT / "research" / "02_template_split_retrain.py")
_t = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_t)        # type: ignore[union-attr]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Standard binned ECE on the *predicted-class confidence*.

    Confidence here = max(p, 1-p) since this is a 2-class problem and we want
    to know "when the model is X% confident, is it actually right X% of the
    time?".
    """
    confidences = np.maximum(y_prob, 1.0 - y_prob)
    predictions = (y_prob >= 0.5).astype(int)
    accuracies  = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if hi < 1.0 else (confidences > lo) & (confidences <= hi + 1e-9)
        c = mask.sum()
        if c > 0:
            acc  = accuracies[mask].mean()
            conf = confidences[mask].mean()
            bin_acc.append(acc)
            bin_conf.append(conf)
            bin_count.append(c)
            ece += (c / n) * abs(acc - conf)
        else:
            bin_acc.append(0.0)
            bin_conf.append((lo + hi) / 2)
            bin_count.append(0)
    return ece, np.array(bin_acc), np.array(bin_conf), np.array(bin_count)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                        use_amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if use_amp:
            with autocast(device_type="cuda"):
                logits = model(images).float()
        else:
            logits = model(images)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.append(p)
        labels_all.append(labels.numpy())
    return np.concatenate(labels_all), np.concatenate(probs)


def build_clean_test_loader() -> Optional[DataLoader]:
    """Reproduce the template-aware test loader from script 02."""
    eval_tf = get_simple_eval_transform(LEAKY_CFG["image_size"])
    eval_ds = LocalEmbeddingDataset(transform=eval_tf)
    cfg = dict(LEAKY_CFG); cfg["val_size"] = 0.20; cfg["test_size"] = 0.20
    try:
        _, _, test_idx = _t.template_aware_split(eval_ds.records, cfg)
    except Exception as exc:
        logger.warning("Could not build clean split: %s", exc)
        return None
    return DataLoader(
        Subset(eval_ds, sorted(test_idx)),
        batch_size=LEAKY_CFG["batch_size"], shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reliability(
    bin_conf: np.ndarray, bin_acc: np.ndarray, bin_count: np.ndarray,
    ece: float, brier: float, run_name: str, out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [3, 1]})

    bar_width = 0.5 / max(len(bin_conf), 1)
    valid = bin_count > 0
    ax1.bar(bin_conf[valid], bin_acc[valid], width=bar_width,
            color="#1565C0", edgecolor="black", linewidth=0.5,
            label="Observed accuracy")
    ax1.bar(bin_conf[valid], bin_conf[valid] - bin_acc[valid],
            bottom=bin_acc[valid], width=bar_width,
            color="#FFCDD2", edgecolor="#D32F2F", linewidth=0.5,
            alpha=0.6, label="Calibration gap")
    ax1.plot([0.5, 1.0], [0.5, 1.0], "k--", lw=1.5, label="Perfect calibration")
    ax1.set_xlim(0.5, 1.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("Predicted confidence (max class probability)", fontsize=11)
    ax1.set_ylabel("Observed accuracy in bin", fontsize=11)
    ax1.set_title(f"Reliability diagram — {run_name}", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.text(0.52, 0.92, f"ECE = {ece:.4f}\nBrier = {brier:.4f}",
             fontsize=11, bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9))

    ax2.bar(bin_conf, bin_count, width=bar_width, color="#7B1FA2",
            edgecolor="white")
    ax2.set_xlim(0.5, 1.0)
    ax2.set_xlabel("Confidence bin", fontsize=10)
    ax2.set_ylabel("Sample count", fontsize=10)
    ax2.set_title("Bin populations", fontsize=11)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_confidence_histogram(
    y_true: np.ndarray, y_prob: np.ndarray, run_name: str, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    correct_mask = ((y_prob >= 0.5).astype(int) == y_true)
    confidences = np.maximum(y_prob, 1.0 - y_prob)

    bins = np.linspace(0.5, 1.0, 26)
    ax.hist(confidences[correct_mask],   bins=bins, color="#2E7D32", alpha=0.75,
            label=f"Correct  ({correct_mask.sum()})", edgecolor="white")
    ax.hist(confidences[~correct_mask],  bins=bins, color="#C62828", alpha=0.85,
            label=f"Wrong    ({(~correct_mask).sum()})", edgecolor="white")
    ax.set_xlabel("Predicted confidence (max class probability)", fontsize=11)
    ax.set_ylabel("Number of test images", fontsize=11)
    ax.set_title(f"Confidence distribution — {run_name}",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Run one calibration analysis
# ---------------------------------------------------------------------------

def analyse_run(name: str, ckpt_path: Path, test_loader: DataLoader, device: torch.device) -> dict:
    if not ckpt_path.exists():
        logger.warning("Checkpoint missing: %s — skipping", ckpt_path)
        return {}
    logger.info("=== Analysing run: %s (ckpt=%s) ===", name, ckpt_path)
    model = SupervisedResNet50(num_classes=2, dropout=0.2).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    y_true, y_prob = collect_predictions(model, test_loader, device)

    ece, bin_acc, bin_conf, bin_count = expected_calibration_error(y_true, y_prob)
    brier = brier_score(y_true, y_prob)
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    confidences = np.maximum(y_prob, 1.0 - y_prob)

    metrics = {
        "n_test":               int(len(y_true)),
        "accuracy":             float(acc),
        "ece":                  float(ece),
        "brier":                float(brier),
        "mean_confidence":      float(confidences.mean()),
        "frac_above_0.99":      float((confidences >= 0.99).mean()),
        "frac_above_0.90":      float((confidences >= 0.90).mean()),
        "frac_below_0.60":      float((confidences <  0.60).mean()),
        "wrong_at_high_conf":   int(((y_prob >= 0.5).astype(int) != y_true) & (confidences >= 0.95)).sum() if False else int(((((y_prob >= 0.5).astype(int)) != y_true) & (confidences >= 0.95)).sum()),  # noqa
    }
    plot_reliability(bin_conf, bin_acc, bin_count, ece, brier, name,
                     OUT / f"03_calibration_{name}.png")
    plot_confidence_histogram(y_true, y_prob, name, OUT / f"03_confidence_histogram_{name}.png")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()

    # Leaky run uses the original by-image split test loader
    _, _, leaky_test_loader, _ = leaky_make_loaders(LEAKY_CFG)
    leaky_ckpt = ROOT / "models" / "supervised_resnet50_best.pth"
    leaky_metrics = analyse_run("leaky", leaky_ckpt, leaky_test_loader, device)

    # Clean run uses the template-aware split test loader
    clean_test_loader = build_clean_test_loader()
    clean_ckpt = ROOT / "models" / "supervised_resnet50_clean_best.pth"
    clean_metrics = analyse_run("clean", clean_ckpt, clean_test_loader, device) if clean_test_loader else {}

    out_json = OUT / "03_calibration_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"leaky": leaky_metrics, "clean": clean_metrics}, f, indent=2)
    logger.info("Saved: %s", out_json)


if __name__ == "__main__":
    main()
