"""
research/05_robustness_sweep.py
-------------------------------
Robustness analysis: how does the model degrade under realistic corruptions
that an adversary or normal acquisition pipeline would impose?

We sweep 8 corruption families at 5 severity levels each, evaluate the
test set under each, and produce one curve per corruption.

Corruptions
  - gaussian_blur      sigma  in [0.5, 1.0, 2.0, 4.0, 8.0]
  - jpeg_compression   quality in [80, 60, 40, 25, 10]
  - downscale_upscale  factor  in [0.9, 0.75, 0.5, 0.33, 0.2]
  - rotation           degrees in [2, 5, 10, 20, 45]
  - brightness         delta   in [-0.4, -0.2, 0.2, 0.4, 0.6]
  - gaussian_noise     sigma   in [0.02, 0.05, 0.10, 0.15, 0.20]
  - center_crop        keep    in [0.95, 0.85, 0.7, 0.5, 0.3]
  - random_occlusion   area    in [0.05, 0.10, 0.20, 0.30, 0.40]

For each (corruption, severity) we report accuracy and ROC-AUC on the test
set so the curve communicates "this method holds up well to X but breaks
on Y at severity Z".

Outputs:
    research_outputs/05_robustness_<run>.csv
    research_outputs/05_robustness_<run>.png
"""

from __future__ import annotations

import io
import logging
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

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
_t_spec = _ilu.spec_from_file_location("tmpl", ROOT / "research" / "02_template_split_retrain.py")
_t = _ilu.module_from_spec(_t_spec)  # type: ignore[arg-type]
_t_spec.loader.exec_module(_t)  # type: ignore[union-attr]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Corruption families. Each is a function(PIL.Image, severity) -> PIL.Image.
# ---------------------------------------------------------------------------

def corr_gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))

def corr_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def corr_downscale_upscale(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    return img.resize((nw, nh), Image.BILINEAR).resize((w, h), Image.BILINEAR)

def corr_rotation(img: Image.Image, degrees: float) -> Image.Image:
    return img.rotate(float(degrees), resample=Image.BILINEAR, fillcolor=(0, 0, 0))

def corr_brightness(img: Image.Image, delta: float) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = np.clip(a + float(delta), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))

def corr_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = np.clip(a + np.random.randn(*a.shape).astype(np.float32) * float(sigma), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))

def corr_center_crop(img: Image.Image, keep: float) -> Image.Image:
    w, h = img.size
    nw, nh = int(w * keep), int(h * keep)
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BILINEAR)

def corr_random_occlusion(img: Image.Image, area_frac: float, rng: np.random.Generator | None = None) -> Image.Image:
    rng = rng or np.random.default_rng(0)
    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    box_area = int(area_frac * h * w)
    box_h = int(math.sqrt(box_area))
    box_w = int(math.sqrt(box_area))
    if box_h <= 0 or box_w <= 0: return img
    y = int(rng.integers(0, max(1, h - box_h)))
    x = int(rng.integers(0, max(1, w - box_w)))
    a[y:y+box_h, x:x+box_w] = 0
    return Image.fromarray(a)


CORRUPTIONS = {
    "gaussian_blur":     ("σ",   [0.5, 1.0, 2.0, 4.0, 8.0],            corr_gaussian_blur),
    "jpeg_compression":  ("Q",   [80, 60, 40, 25, 10],                 corr_jpeg),
    "downscale_upscale": ("f",   [0.9, 0.75, 0.5, 0.33, 0.2],          corr_downscale_upscale),
    "rotation":          ("°",   [2, 5, 10, 20, 45],                   corr_rotation),
    "brightness":        ("Δ",   [-0.4, -0.2, 0.2, 0.4, 0.6],          corr_brightness),
    "gaussian_noise":    ("σ",   [0.02, 0.05, 0.10, 0.15, 0.20],       corr_gaussian_noise),
    "center_crop":       ("keep",[0.95, 0.85, 0.7, 0.5, 0.3],          corr_center_crop),
    "random_occlusion":  ("area",[0.05, 0.10, 0.20, 0.30, 0.40],       corr_random_occlusion),
}


# ---------------------------------------------------------------------------
# Wrapped Dataset that applies a corruption before the eval transform
# ---------------------------------------------------------------------------

class CorruptedView(torch.utils.data.Dataset):
    """Wrap a LocalEmbeddingDataset so that each image is loaded from the
    pre-resized 256-px-short-side cache (the training distribution), then
    corrupted, then passed through the eval transform.

    Reading from the resized cache (not the 2167x1360 original) is critical:
    the model was trained on 256-px images squashed to 224x224, so the
    baseline-no-corruption accuracy must match the training-time accuracy.
    """

    def __init__(self, base: LocalEmbeddingDataset, indices: list[int],
                 corruption_fn, severity, eval_transform):
        self.base = base
        self.indices = sorted(indices)
        self.fn = corruption_fn
        self.severity = severity
        self.eval_tf = eval_transform

    def __len__(self): return len(self.indices)

    def __getitem__(self, i: int):
        rec = self.base.records[self.indices[i]]
        # Use the resized cache when available (training distribution).
        # _pil_load() handles the fallback to the original.
        from local_dataset_loader import _pil_load
        im = _pil_load(rec)
        corrupted = self.fn(im, self.severity)
        x = self.eval_tf(corrupted)
        return x, rec["label"]


@torch.no_grad()
def evaluate_corrupted(model, device, base_ds, test_indices, corruption_fn,
                       severity, eval_tf, batch_size: int = 32) -> dict:
    ds = CorruptedView(base_ds, test_indices, corruption_fn, severity, eval_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    probs, preds, labels_all = [], [], []
    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type="cuda"):
            logits = model(x).float()
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels_all.append(y.numpy())
    y_prob = np.concatenate(probs)
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels_all)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":  float(roc_auc_score(y_true, y_prob)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_robustness(results: list[dict], baseline: dict, run_name: str, out_path: Path) -> None:
    """Plot 8 small subplots, one per corruption."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
    fig.suptitle(f"Robustness sweep — {run_name} (baseline test ROC-AUC = {baseline['roc_auc']:.3f})",
                 fontsize=14, fontweight="bold")
    by_corr = {}
    for r in results:
        by_corr.setdefault(r["corruption"], []).append(r)

    for ax, (name, items) in zip(axes.flatten(), by_corr.items()):
        items_sorted = sorted(items, key=lambda r: r["severity_idx"])
        sev_labels = [str(r["severity"]) for r in items_sorted]
        accs = [r["accuracy"] for r in items_sorted]
        aucs = [r["roc_auc"]  for r in items_sorted]
        f1s  = [r["f1"]       for r in items_sorted]
        x = np.arange(len(items_sorted))
        ax.plot(x, aucs, "o-", color="#1565C0", lw=2.2, label="ROC-AUC")
        ax.plot(x, accs, "s--", color="#43A047", lw=1.8, label="Accuracy")
        ax.plot(x, f1s,  "^:",  color="#EF6C00", lw=1.8, label="F1")
        ax.axhline(baseline["roc_auc"], color="#1565C0", linestyle=":", alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels(sev_labels, rotation=20)
        ax.set_title(name, fontsize=11)
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.3)
        if ax is axes[0, 0]:
            ax.legend(fontsize=9, loc="lower left")
    axes[1, 0].set_ylabel("Metric"); axes[0, 0].set_ylabel("Metric")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Run a robustness sweep for one checkpoint + split
# ---------------------------------------------------------------------------

def run_sweep(name: str, ckpt_path: Path, get_loader, device: torch.device):
    if not ckpt_path.exists():
        logger.warning("Skipping %s: %s missing", name, ckpt_path); return
    logger.info("=== Robustness sweep: %s ===", name)
    base_ds, test_indices, baseline_loader = get_loader()

    model = SupervisedResNet50(num_classes=2, dropout=0.2).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_tf = transforms.Compose([
        transforms.Resize((LEAKY_CFG["image_size"], LEAKY_CFG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Baseline (no corruption)
    base_metrics = evaluate_corrupted(
        model, device, base_ds, test_indices,
        lambda im, _: im, severity=None, eval_tf=eval_tf,
        batch_size=LEAKY_CFG["batch_size"],
    )
    logger.info("baseline: acc=%.4f f1=%.4f roc_auc=%.4f",
                base_metrics["accuracy"], base_metrics["f1"], base_metrics["roc_auc"])

    rows = []
    rows.append({"corruption": "baseline", "severity": "—", "severity_idx": 0, **base_metrics})

    for corr_name, (sym, severities, fn) in CORRUPTIONS.items():
        for i, sev in enumerate(severities):
            m = evaluate_corrupted(
                model, device, base_ds, test_indices,
                fn, sev, eval_tf, batch_size=LEAKY_CFG["batch_size"],
            )
            rows.append({
                "corruption":   corr_name,
                "severity":     f"{sym}={sev}",
                "severity_idx": i + 1,
                **m,
            })
            logger.info("  %s %s=%s -> acc=%.3f f1=%.3f roc_auc=%.3f",
                        corr_name, sym, sev, m["accuracy"], m["f1"], m["roc_auc"])

    # Save CSV
    import csv
    csv_path = OUT / f"05_robustness_{name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "severity", "severity_idx", "accuracy", "f1", "roc_auc"])
        for r in rows:
            w.writerow([r["corruption"], r["severity"], r["severity_idx"],
                        round(r["accuracy"], 4), round(r["f1"], 4), round(r["roc_auc"], 4)])
    logger.info("Saved: %s", csv_path)

    # Plot
    sweep = [r for r in rows if r["corruption"] != "baseline"]
    plot_robustness(sweep, base_metrics, name, OUT / f"05_robustness_{name}.png")


def get_leaky_loader_setup():
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
    _, _, baseline_loader, _ = leaky_make_loaders(LEAKY_CFG)
    return base_ds, test_idx, baseline_loader


def get_clean_loader_setup():
    eval_tf = get_simple_eval_transform(LEAKY_CFG["image_size"])
    base_ds = LocalEmbeddingDataset(transform=eval_tf)
    cfg = dict(LEAKY_CFG); cfg["val_size"] = 0.20; cfg["test_size"] = 0.20
    _, _, test_idx = _t.template_aware_split(base_ds.records, cfg)
    test_idx = sorted(test_idx)
    baseline_loader = DataLoader(
        Subset(base_ds, test_idx), batch_size=LEAKY_CFG["batch_size"], shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    return base_ds, test_idx, baseline_loader


def main():
    device = get_device()
    np.random.seed(42)  # for the random_occlusion corruption
    run_sweep("leaky", ROOT / "models" / "supervised_resnet50_best.pth",
              get_leaky_loader_setup, device)
    run_sweep("clean", ROOT / "models" / "supervised_resnet50_clean_best.pth",
              get_clean_loader_setup, device)


if __name__ == "__main__":
    main()
