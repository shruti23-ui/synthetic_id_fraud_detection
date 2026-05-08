"""
research/04_failure_analysis.py
-------------------------------
Where does the model fail and what does it look at when it fails?

For both the leaky and the clean checkpoints we:
  1. Run the test set
  2. Pick the worst false positives and worst false negatives
     (highest confidence wrong predictions)
  3. Render a contact sheet of those images alongside Grad-CAM overlays
  4. Cross-reference each failure with its forgery type / source template

This is the diagnostic that tells you whether the failures are:
  - random noise        -> model is just learning a bit imperfectly
  - one specific ctype  -> the model has a systematic blind spot
  - one specific src    -> the model just memorised some templates

Outputs:
    research_outputs/04_failures_<run>_grid.png
    research_outputs/04_failures_<run>.csv
    research_outputs/04_failures_summary.json
"""

from __future__ import annotations

import numpy as np

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp.autocast_mode import autocast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from local_dataset_loader import LocalEmbeddingDataset, _build_records, _TEMPLATES_ROOT  # noqa: E402
from supervised_finetune import (  # noqa: E402
    CONFIG as LEAKY_CFG,
    SupervisedResNet50,
    get_simple_eval_transform,
    make_loaders as leaky_make_loaders,
)
from utils import get_device  # noqa: E402

from template_split import template_aware_split, template_id  # noqa: E402

# Backwards-compat shim for the old `_audit.template_id` reference still
# used elsewhere in this module.
class _AuditShim:
    template_id = staticmethod(template_id)
_audit = _AuditShim()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Grad-CAM (last-conv-layer) for ResNet50
# ---------------------------------------------------------------------------

class GradCAMResNet50:
    def __init__(self, model: SupervisedResNet50):
        self.model = model
        # ResNet50's last conv block lives at .net.layer4
        self.target = model.net.layer4
        self.activations = None
        self.gradients   = None
        self._fwd = self.target.register_forward_hook(self._save_act)
        self._bwd = self.target.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, output): self.activations = output.detach()
    def _save_grad(self, _, gin, gout): self.gradients = gout[0].detach()
    def remove(self):
        self._fwd.remove(); self._bwd.remove()

    def __call__(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)
        score = logits[:, target_class].sum()
        self.model.zero_grad()
        score.backward()
        # Gradients-weighted activations
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1).cpu().numpy()
        cam_min = cam.reshape(cam.shape[0], -1).min(axis=1)[:, None, None]
        cam_max = cam.reshape(cam.shape[0], -1).max(axis=1)[:, None, None]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


def denormalise(t: torch.Tensor) -> np.ndarray:
    img = t.cpu().numpy()
    img = img * IMAGENET_STD.reshape(3, 1, 1) + IMAGENET_MEAN.reshape(3, 1, 1)
    return np.clip(img.transpose(1, 2, 0), 0, 1)


# ---------------------------------------------------------------------------
# Inference + failure picking
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_with_indices(model, loader, device, use_amp: bool = True):
    model.eval()
    probs_all, preds_all, labels_all, idx_all = [], [], [], []
    for batch in loader:
        images, labels = batch
        images = images.to(device, non_blocking=True)
        if use_amp:
            with autocast(device_type="cuda"):
                logits = model(images).float()
        else:
            logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        probs_all.append(probs); preds_all.append(preds); labels_all.append(labels.numpy())
    return np.concatenate(probs_all), np.concatenate(preds_all), np.concatenate(labels_all)


def pick_top_failures(y_true: np.ndarray, y_prob: np.ndarray, k: int = 6):
    """Return indices of the k worst false-positives (highest prob, true=0)
    and the k worst false-negatives (lowest prob, true=1)."""
    fp_mask = (y_true == 0)
    fn_mask = (y_true == 1)
    fp_scores = np.where(fp_mask, y_prob, -np.inf)            # high prob & true real
    fn_scores = np.where(fn_mask, 1.0 - y_prob, -np.inf)      # high (1-prob) & true fake
    fp_top = np.argsort(-fp_scores)[:k]
    fn_top = np.argsort(-fn_scores)[:k]
    return fp_top.tolist(), fn_top.tolist()


# ---------------------------------------------------------------------------
# Visualisation: contact sheet of failures with Grad-CAM
# ---------------------------------------------------------------------------

def render_failures(
    model: SupervisedResNet50, device: torch.device, loader_subset_records: list[dict],
    test_indices_in_subset: list[int], y_prob: np.ndarray, y_true: np.ndarray,
    fp_idx: list[int], fn_idx: list[int], run_name: str, out_path: Path,
):
    cam = GradCAMResNet50(model)
    eval_tf = get_simple_eval_transform(LEAKY_CFG["image_size"])

    rows, cols = 4, 6
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    fig.suptitle(f"Worst failure cases — {run_name}\n"
                 f"Top: high-confidence FALSE POSITIVES  (real -> predicted fake)\n"
                 f"Bot: high-confidence FALSE NEGATIVES  (fake -> predicted real)",
                 fontsize=12, fontweight="bold")

    def draw(ax, sub_idx: int, kind: str, panel_row: int):
        rec_idx = test_indices_in_subset[sub_idx]
        rec = loader_subset_records[rec_idx]
        with Image.open(rec["path"]).convert("RGB") as im:
            tensor = eval_tf(im).unsqueeze(0).to(device)
        pred_class = int(y_prob[sub_idx] >= 0.5)
        target = pred_class
        heat = cam(tensor, target_class=target)[0]
        rgb = denormalise(tensor.squeeze(0).detach())

        if panel_row == 0:
            ax.imshow(rgb)
        else:
            ax.imshow(rgb)
            ax.imshow(heat, cmap="jet", alpha=0.45)

        true_lbl = "real" if rec["label"] == 0 else "fake"
        ctype = rec.get("ctype", "real") or "real"
        title = f"{kind}\n{rec['path'].stem}\np(fake)={y_prob[sub_idx]:.3f}\ntrue={true_lbl} ({ctype})"
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    for col, idx in enumerate(fp_idx[:cols]):
        draw(axes[0, col], idx, "FP", 0)
        draw(axes[1, col], idx, "FP", 1)
    for col, idx in enumerate(fn_idx[:cols]):
        draw(axes[2, col], idx, "FN", 0)
        draw(axes[3, col], idx, "FN", 1)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    cam.remove()
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Aggregate failure stats by ctype / template
# ---------------------------------------------------------------------------

def failure_breakdown(records: list[dict], test_idx_global: list[int],
                      y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    pred = (y_prob >= 0.5).astype(int)
    wrong = pred != y_true
    out = {"by_ctype": {}, "by_template": {}}
    for sub_i, glob_i in enumerate(test_idx_global):
        rec = records[glob_i]
        ct = rec.get("ctype", "real") or "real"
        tpl = _audit.template_id(rec)
        out["by_ctype"].setdefault(ct, {"total": 0, "wrong": 0})
        out["by_ctype"][ct]["total"] += 1
        out["by_ctype"][ct]["wrong"] += int(wrong[sub_i])
        out["by_template"].setdefault(tpl, {"total": 0, "wrong": 0})
        out["by_template"][tpl]["total"] += 1
        out["by_template"][tpl]["wrong"] += int(wrong[sub_i])
    # error rate per ctype
    for ct, v in out["by_ctype"].items():
        v["error_rate"] = v["wrong"] / max(v["total"], 1)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyse_run(name: str, ckpt_path: Path, get_loader_and_indices, device: torch.device):
    if not ckpt_path.exists():
        logger.warning("Skipping %s: %s missing", name, ckpt_path)
        return None
    logger.info("=== %s ===", name)
    test_loader, records, test_indices_global = get_loader_and_indices()

    model = SupervisedResNet50(num_classes=2, dropout=0.2).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    y_prob, y_pred, y_true = predict_with_indices(model, test_loader, device)
    fp, fn = pick_top_failures(y_true, y_prob, k=6)

    # Save failure CSVs
    import csv as _csv
    csv_path = OUT / f"04_failures_{name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["sub_idx", "kind", "name", "true_label", "ctype", "p_fake", "predicted"])
        for sub_idx in fp:
            rec = records[test_indices_global[sub_idx]]
            w.writerow([sub_idx, "FP", rec["path"].stem, "real",
                        rec.get("ctype", "real"), float(y_prob[sub_idx]), 1])
        for sub_idx in fn:
            rec = records[test_indices_global[sub_idx]]
            w.writerow([sub_idx, "FN", rec["path"].stem, "fake",
                        rec.get("ctype", "fake"), float(y_prob[sub_idx]), 0])
    logger.info("Saved: %s", csv_path)

    # Contact sheet
    render_failures(model, device, records, test_indices_global, y_prob, y_true,
                    fp, fn, name, OUT / f"04_failures_{name}_grid.png")

    breakdown = failure_breakdown(records, test_indices_global, y_true, y_prob)
    logger.info("Per-ctype error rate: %s",
                {k: round(v["error_rate"], 4) for k, v in breakdown["by_ctype"].items()})
    return {"name": name, "n_test": int(len(y_true)),
            "n_wrong": int((y_pred != y_true).sum()),
            "by_ctype": breakdown["by_ctype"]}


def get_leaky_loader_and_indices():
    train_ds = LocalEmbeddingDataset(transform=get_simple_eval_transform(LEAKY_CFG["image_size"]))
    records = train_ds.records
    _, _, test_loader, _ = leaky_make_loaders(LEAKY_CFG)
    # Reproduce the indices that leaky_make_loaders used
    labels = np.array([r["label"] for r in records])
    rng = np.random.default_rng(LEAKY_CFG["seed"])
    train_idx, val_idx, test_idx = [], [], []
    for cls in (0, 1):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_test = int(round(n * LEAKY_CFG["test_size"]))
        n_val  = int(round(n * LEAKY_CFG["val_size"]))
        test_idx.extend(cls_idx[:n_test].tolist())
        val_idx.extend(cls_idx[n_test:n_test + n_val].tolist())
        train_idx.extend(cls_idx[n_test + n_val:].tolist())
    return test_loader, records, sorted(test_idx)


def get_clean_loader_and_indices():
    eval_ds = LocalEmbeddingDataset(transform=get_simple_eval_transform(LEAKY_CFG["image_size"]))
    cfg = dict(LEAKY_CFG); cfg["val_size"] = 0.20; cfg["test_size"] = 0.20
    _, _, test_idx = template_aware_split(eval_ds.records, cfg)
    test_idx = sorted(test_idx)
    from torch.utils.data import DataLoader, Subset
    test_loader = DataLoader(
        Subset(eval_ds, test_idx),
        batch_size=LEAKY_CFG["batch_size"], shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    return test_loader, eval_ds.records, test_idx


def main():
    device = get_device()
    summary = {}
    leaky_ckpt = ROOT / "models" / "supervised_resnet50_best.pth"
    clean_ckpt = ROOT / "models" / "supervised_resnet50_clean_best.pth"

    s = analyse_run("leaky", leaky_ckpt, get_leaky_loader_and_indices, device)
    if s: summary["leaky"] = s
    s = analyse_run("clean", clean_ckpt, get_clean_loader_and_indices, device)
    if s: summary["clean"] = s

    with open(OUT / "04_failures_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", OUT / "04_failures_summary.json")


if __name__ == "__main__":
    main()
