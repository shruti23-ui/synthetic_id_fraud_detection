"""
evaluate.py
-----------
Compute and persist comprehensive evaluation artefacts:
  - ROC curve data
  - Precision-Recall curve data
  - Confusion matrix (raw + normalised)
  - Per-threshold F1 / accuracy sweep

Usage (standalone):
    python src/evaluate.py
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import load_processed_arrays, split_train_val_test

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
    "split":        "train",
    "val_size":     0.2,
    "test_size":    0.2,
    "random_state": 42,
    "metrics_dir":  "outputs/metrics",
    "models_dir":   "models",
}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_roc(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute ROC curve data and AUC.

    Returns:
        Dict with keys: fpr, tpr, thresholds, auc_score.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    logger.info("ROC-AUC: %.4f", auc_score)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc_score": auc_score}


def compute_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute Precision-Recall curve data and average precision.

    Returns:
        Dict with keys: precision, recall, thresholds, avg_precision.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    logger.info("Average Precision: %.4f", ap)
    return {"precision": precision, "recall": recall, "thresholds": thresholds, "avg_precision": ap}


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute raw and row-normalised confusion matrices.

    Returns:
        Dict with keys: raw, normalised.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    logger.info("Confusion matrix (raw):\n%s", cm)
    return {"raw": cm, "normalised": cm_norm}


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(config: dict = CONFIG) -> dict:
    """Load the best classifier and produce all evaluation artefacts.

    Args:
        config: Configuration dictionary.

    Returns:
        Dict containing roc, pr, confusion, and threshold sweep data.
    """
    metrics_dir = Path(config["metrics_dir"])
    models_dir  = Path(config["models_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data and use the SAME 60/20/20 split as classifier.py ───────────
    embeddings, labels = load_processed_arrays(split=config["split"])
    _, _, X_test, _, _, y_test = split_train_val_test(
        embeddings, labels,
        val_size=config["val_size"],
        test_size=config["test_size"],
        seed=config["random_state"],
    )

    # Apply the same scaler used during training
    scaler_path = models_dir / "feature_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_test = scaler.transform(X_test)
    else:
        logger.warning("Scaler not found at %s. Using unscaled features.", scaler_path)

    # ── Load best classifier ──────────────────────────────────────────────────
    clf_path = models_dir / "best_classifier.pkl"
    if not clf_path.exists():
        raise FileNotFoundError(
            f"Best classifier not found at {clf_path}. Run classifier.py first."
        )
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    logger.info("Loaded classifier: %s", type(clf).__name__)

    y_pred  = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)

    # ── Compute metrics ───────────────────────────────────────────────────────
    roc_data        = compute_roc(y_test, y_score)
    pr_data         = compute_pr_curve(y_test, y_score)
    confusion_data  = compute_confusion(y_test, y_pred)

    # ── Threshold sweep ───────────────────────────────────────────────────────
    thresholds = np.linspace(0.1, 0.9, 17)
    sweep_rows = []
    for t in thresholds:
        y_t = (y_score >= t).astype(int)
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        sweep_rows.append({
            "threshold": round(t, 2),
            "accuracy":  round(accuracy_score(y_test, y_t), 4),
            "precision": round(precision_score(y_test, y_t, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_t, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_t, zero_division=0), 4),
        })
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = metrics_dir / "threshold_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    logger.info("Threshold sweep saved → %s", sweep_path)

    # ── Persist ROC / PR data ─────────────────────────────────────────────────
    pd.DataFrame({"fpr": roc_data["fpr"], "tpr": roc_data["tpr"]}).to_csv(
        metrics_dir / "roc_curve_data.csv", index=False
    )
    pd.DataFrame({"precision": pr_data["precision"], "recall": pr_data["recall"]}).to_csv(
        metrics_dir / "pr_curve_data.csv", index=False
    )
    np.save(metrics_dir / "confusion_matrix.npy", confusion_data["raw"])
    np.save(metrics_dir / "confusion_matrix_norm.npy", confusion_data["normalised"])

    logger.info("All evaluation artefacts saved to %s.", metrics_dir)

    return {
        "roc":       roc_data,
        "pr":        pr_data,
        "confusion": confusion_data,
        "sweep":     sweep_df,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
    logger.info("evaluate.py finished.")
