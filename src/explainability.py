"""
explainability.py
-----------------
SHAP-based explainability for the fraud classifier.

Generates:
  - SHAP summary plot (beeswarm)
  - SHAP bar plot (mean absolute impact)
  - SHAP waterfall for a single fraud prediction

Usage (standalone):
    python src/explainability.py

Outputs saved to:  outputs/plots/
"""

import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import load_processed_arrays, split_train_val_test
from contrastive_model import SimCLRModel
from augmentations import get_eval_transform
from local_dataset_loader import LocalEmbeddingDataset
from utils import get_device

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
    "split":            "train",
    "val_size":         0.2,
    "test_size":        0.2,
    "random_state":     42,
    "models_dir":       "models",
    "plots_dir":        "outputs/plots",
    "shap_samples":     200,
    "n_features_show":  20,
    "gradcam_samples":  6,        # number of fake/real images to explain visually
    "image_size":       224,
    "encoder_ckpt":     "models/simclr_encoder_best.pth",
}


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def run_shap_explainability(config: dict = CONFIG) -> None:
    """Compute SHAP values and generate explanation plots.

    Supports tree-based models (XGBoost, RandomForest) via TreeExplainer
    and falls back to KernelExplainer for linear models.

    Args:
        config: Configuration dictionary.
    """
    plots_dir  = Path(config["plots_dir"])
    models_dir = Path(config["models_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data; use the SAME 60/20/20 split as classifier.py ──────────────
    embeddings, labels = load_processed_arrays(split=config["split"])
    _, _, X_test, _, _, y_test = split_train_val_test(
        embeddings, labels,
        val_size=config["val_size"],
        test_size=config["test_size"],
        seed=config["random_state"],
    )

    scaler_path = models_dir / "feature_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_test = scaler.transform(X_test)

    clf_path = models_dir / "best_classifier.pkl"
    if not clf_path.exists():
        raise FileNotFoundError(f"Classifier not found at {clf_path}. Run classifier.py first.")

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    logger.info("Loaded classifier: %s", type(clf).__name__)

    # Use a subset for SHAP (speed)
    n = min(config["shap_samples"], len(X_test))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), n, replace=False)
    X_explain = X_test[idx]
    y_explain = y_test[idx]
    logger.info("Computing SHAP values for %d samples …", n)

    # Feature names based on embedding dimension
    feat_names = [f"feat_{i}" for i in range(X_test.shape[1])]

    # ── Choose explainer based on model type ──────────────────────────────────
    clf_type = type(clf).__name__
    if clf_type in ("XGBClassifier", "RandomForestClassifier"):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_explain)
        # RandomForest TreeExplainer returns list [class0, class1]; take class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        # Fallback: kernel SHAP (slower)
        background = shap.kmeans(X_test, 50)
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        shap_values = explainer.shap_values(X_explain, nsamples=100)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    logger.info("SHAP values shape: %s", shap_values.shape)

    # ── Plot 1: SHAP Summary (beeswarm) ──────────────────────────────────────
    logger.info("Generating SHAP summary (beeswarm) plot …")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feat_names,
        max_display=config["n_features_show"],
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Feature Impact on Fraud Prediction (Beeswarm)", fontsize=13, pad=12)
    plt.tight_layout()
    out = plots_dir / "shap_summary_beeswarm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)

    # ── Plot 2: SHAP Bar (mean |SHAP|) ────────────────────────────────────────
    logger.info("Generating SHAP bar plot …")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feat_names,
        max_display=config["n_features_show"],
        show=False,
        plot_type="bar",
    )
    plt.title("SHAP Mean Absolute Feature Importance", fontsize=13, pad=12)
    plt.tight_layout()
    out = plots_dir / "shap_summary_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)

    # ── Plot 3: Waterfall for one fraud sample ────────────────────────────────
    fraud_idx_local = np.where(y_explain == 1)[0]
    if len(fraud_idx_local) > 0:
        sample_idx = fraud_idx_local[0]
        logger.info("Generating SHAP waterfall plot for sample %d (fraud) …", sample_idx)

        try:
            # SHAP ≥ 0.40 API
            exp = shap.Explanation(
                values=shap_values[sample_idx],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                            else explainer.expected_value[1],
                data=X_explain[sample_idx],
                feature_names=feat_names,
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(exp, max_display=15, show=False)
            plt.title("SHAP Waterfall: Single Fraud Prediction", fontsize=13, pad=12)
            plt.tight_layout()
            out = plots_dir / "shap_waterfall_fraud.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Saved: %s", out)
        except Exception as exc:
            logger.warning("Waterfall plot failed (%s) — skipping.", exc)
    else:
        logger.warning("No fraud samples in explain subset — waterfall plot skipped.")

    logger.info("SHAP explainability complete.")


# ---------------------------------------------------------------------------
# Grad-CAM (visual explainability over the SimCLR encoder)
# ---------------------------------------------------------------------------

class GradCAM:
    """Grad-CAM hook on the last conv block of the SimCLR ResNet18 encoder.

    Targets the embedding norm so we get attribution for "what the encoder
    looked at" without needing a classification head fused into the same graph.
    """

    def __init__(self, model: SimCLRModel, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, output):
        self.activations = output.detach()

    def _save_grad(self, _, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self.model.eval()
        x = x.requires_grad_(True)
        feats, _ = self.model(x)
        score = feats.norm(dim=1).sum()
        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)         # (B, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1).cpu().numpy()
        # Normalise per-image to [0,1]
        cam_min = cam.reshape(cam.shape[0], -1).min(axis=1)[:, None, None]
        cam_max = cam.reshape(cam.shape[0], -1).max(axis=1)[:, None, None]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation for display."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = tensor.cpu().numpy() * std + mean
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def run_gradcam(config: dict = CONFIG) -> None:
    """Generate Grad-CAM overlays for a few fake and real images."""
    plots_dir = Path(config["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model  = SimCLRModel(embedding_dim=128, pretrained=False).to(device)

    ckpt_path = Path(config["encoder_ckpt"])
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Grad-CAM: loaded encoder from %s", ckpt_path)
    else:
        logger.warning("Grad-CAM: no encoder checkpoint at %s — using ImageNet weights only.", ckpt_path)
        model = SimCLRModel(embedding_dim=128, pretrained=True).to(device)

    transform = get_eval_transform(image_size=config["image_size"])
    ds = LocalEmbeddingDataset(transform=transform)

    # Collect a handful of fake & real samples
    fake_idx = [i for i, r in enumerate(ds.records) if r["label"] == 1][: config["gradcam_samples"] // 2]
    real_idx = [i for i, r in enumerate(ds.records) if r["label"] == 0][: config["gradcam_samples"] // 2]
    pick = fake_idx + real_idx
    if not pick:
        logger.warning("No samples available for Grad-CAM — skipping.")
        return

    # Hook on the last ResNet block (model.encoder[-2] is layer4 since avgpool is [-1])
    target = list(model.encoder.children())[-2]
    cam = GradCAM(model, target)

    n = len(pick)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4))
    if n == 1:
        axes = axes[:, None]

    for col, idx in enumerate(pick):
        img_t, lbl = ds[idx]
        x = img_t.unsqueeze(0).to(device)
        heatmap = cam(x)[0]                                  # (H, W) in [0,1]
        rgb = _denorm(img_t)

        axes[0, col].imshow(rgb)
        axes[0, col].set_title("Fake" if lbl == 1 else "Real",
                               color="#D32F2F" if lbl == 1 else "#1565C0",
                               fontsize=11)
        axes[0, col].axis("off")

        axes[1, col].imshow(rgb)
        axes[1, col].imshow(heatmap, cmap="jet", alpha=0.45)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original",  fontsize=11)
    axes[1, 0].set_ylabel("Grad-CAM",  fontsize=11)
    fig.suptitle("Grad-CAM: Encoder Attention on Fake vs Real IDs",
                 fontsize=13, fontweight="bold")
    out = plots_dir / "gradcam_overlays.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    cam.remove()
    logger.info("Saved Grad-CAM overlays → %s", out)


# ---------------------------------------------------------------------------
# Combined runner used by main.py
# ---------------------------------------------------------------------------

def run_explainability(config: dict = CONFIG) -> None:
    """Run SHAP + Grad-CAM explainability."""
    run_shap_explainability(config)
    try:
        run_gradcam(config)
    except Exception as exc:
        logger.warning("Grad-CAM failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_explainability()
    logger.info("explainability.py finished.")
