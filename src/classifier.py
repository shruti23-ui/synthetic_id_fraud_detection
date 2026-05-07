"""
classifier.py
-------------
Train and evaluate fraud classifiers (XGBoost, Random Forest, Logistic Regression)
on top of the contrastive embeddings extracted by the SimCLR encoder.

Usage (standalone):
    python src/classifier.py

Outputs:
    outputs/metrics/classifier_results.csv   – per-model metrics
    models/best_classifier.pkl               – best-performing model
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    recall_score,
    precision_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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
    "split":         "train",
    "val_size":      0.2,        # 20% validation
    "test_size":     0.2,        # 20% held-out test  (=> 60% train)
    "random_state":  42,
    "metrics_dir":   "outputs/metrics",
    "models_dir":    "models",
}

# ---------------------------------------------------------------------------
# Classifier definitions
# ---------------------------------------------------------------------------

def build_classifiers() -> dict:
    """Return a dict of {name: sklearn-compatible estimator}."""
    return {
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate="adaptive",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
        ),
    }


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute classification metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_test, y_prob), 4),
    }
    return metrics


def train_classifiers(config: dict = CONFIG) -> dict:
    """Fit all classifiers on the embedding matrix and return results.

    Args:
        config: Configuration dictionary.

    Returns:
        Dict mapping model name → {metrics, fitted_model}.
    """
    Path(config["metrics_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["models_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Load embeddings ───────────────────────────────────────────────────────
    embeddings, labels = load_processed_arrays(split=config["split"])
    logger.info("Embeddings: %s  Labels: %s", embeddings.shape, labels.shape)
    logger.info("Class distribution: %s", dict(zip(*np.unique(labels, return_counts=True))))

    # ── Stratified 60/20/20 split (train / val / test) ────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        embeddings, labels,
        val_size=config["val_size"],
        test_size=config["test_size"],
        seed=config["random_state"],
    )

    # Standardise features (important for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # Save scaler for inference
    scaler_path = Path(config["models_dir"]) / "feature_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved → %s", scaler_path)

    # ── Train each classifier ─────────────────────────────────────────────────
    classifiers = build_classifiers()
    results_rows = []
    fitted_models = {}

    for name, clf in classifiers.items():
        logger.info("Training %s ...", name)
        clf.fit(X_train_scaled, y_train)
        val_metrics  = evaluate_classifier(clf, X_val_scaled,  y_val)
        test_metrics = evaluate_classifier(clf, X_test_scaled, y_test)
        fitted_models[name] = clf

        logger.info(
            "  %s [VAL ] acc=%.3f recall=%.3f precision=%.3f f1=%.3f roc_auc=%.3f pr_auc=%.3f",
            name,
            val_metrics["accuracy"], val_metrics["recall"], val_metrics["precision"],
            val_metrics["f1"], val_metrics["roc_auc"], val_metrics["pr_auc"],
        )
        logger.info(
            "  %s [TEST] acc=%.3f recall=%.3f precision=%.3f f1=%.3f roc_auc=%.3f pr_auc=%.3f",
            name,
            test_metrics["accuracy"], test_metrics["recall"], test_metrics["precision"],
            test_metrics["f1"], test_metrics["roc_auc"], test_metrics["pr_auc"],
        )

        # Detailed classification report on test set
        y_pred = clf.predict(X_test_scaled)
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

        row = {
            "model": name,
            **{f"val_{k}":  v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        results_rows.append(row)

    # ── Save metrics CSV ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results_rows)
    csv_path = Path(config["metrics_dir"]) / "classifier_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Metrics saved → %s\n%s", csv_path, results_df.to_string(index=False))

    # ── Persist best model (selected on VAL ROC-AUC, never test) ─────────────
    best_idx  = results_df["val_roc_auc"].idxmax()
    best_name = results_df.loc[best_idx, "model"]
    best_clf  = fitted_models[best_name]
    best_path = Path(config["models_dir"]) / "best_classifier.pkl"
    with open(best_path, "wb") as f:
        pickle.dump(best_clf, f)
    logger.info(
        "Best model (%s, val_roc_auc=%.4f, test_roc_auc=%.4f) saved -> %s",
        best_name,
        results_df.loc[best_idx, "val_roc_auc"],
        results_df.loc[best_idx, "test_roc_auc"],
        best_path,
    )

    return {
        name: {"metrics": r, "model": fitted_models[name]}
        for name, r in zip(results_df["model"], results_rows)
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = train_classifiers()
    logger.info("classifier.py finished.")
