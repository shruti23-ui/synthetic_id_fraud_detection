"""
preprocess.py
-------------
Utility functions for inspecting, cleaning, and saving processed data.

Responsibilities:
- Analyse the raw Hugging Face dataset (column types, class distribution)
- Convert the HF dataset to a flat structure suitable for downstream tasks
- Persist processed numpy arrays to disk for fast reloading
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import Dataset as HFDataset

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------------------------

def inspect_dataset(hf_dataset) -> pd.DataFrame:
    """Print a concise summary of a Hugging Face dataset split.

    Args:
        hf_dataset: A single HF dataset split.

    Returns:
        A pandas DataFrame with one row per column showing dtype and
        a few sample values.
    """
    cols = hf_dataset.column_names
    rows = []
    for col in cols:
        samples = [hf_dataset[i][col] for i in range(min(3, len(hf_dataset)))]
        dtype = type(samples[0]).__name__
        rows.append({"column": col, "dtype": dtype, "sample_values": str(samples)})

    df = pd.DataFrame(rows)
    logger.info("\nDataset inspection:\n%s", df.to_string(index=False))
    return df


def get_label_distribution(hf_dataset, label_col: str = "label") -> pd.Series:
    """Return value counts for the label column.

    Args:
        hf_dataset: A single HF dataset split.
        label_col:  Name of the label column.

    Returns:
        pd.Series with label → count mapping, or empty Series if column absent.
    """
    if label_col not in hf_dataset.column_names:
        logger.warning("Label column '%s' not found.", label_col)
        return pd.Series(dtype=int)

    labels = hf_dataset[label_col]
    series = pd.Series(labels).value_counts().sort_index()
    logger.info("Label distribution:\n%s", series.to_string())
    return series


# ---------------------------------------------------------------------------
# Save / load processed arrays
# ---------------------------------------------------------------------------

def save_processed_arrays(
    embeddings: np.ndarray,
    labels: np.ndarray,
    split: str = "train",
) -> None:
    """Save embedding matrix and labels to disk.

    Files are stored in data/processed/<split>_embeddings.npy and
    data/processed/<split>_labels.npy.

    Args:
        embeddings: Float array of shape (N, embedding_dim).
        labels:     Integer array of shape (N,).
        split:      Dataset split name ('train', 'test', …).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / f"{split}_embeddings.npy", embeddings)
    np.save(PROCESSED_DIR / f"{split}_labels.npy", labels)
    logger.info(
        "Saved processed arrays → %s_embeddings.npy (%s), %s_labels.npy (%s)",
        split, embeddings.shape, split, labels.shape,
    )


def load_processed_arrays(split: str = "train"):
    """Load embedding matrix and labels from disk.

    Args:
        split: Dataset split name.

    Returns:
        Tuple (embeddings np.ndarray, labels np.ndarray).

    Raises:
        FileNotFoundError if the files do not exist.
    """
    emb_path = PROCESSED_DIR / f"{split}_embeddings.npy"
    lbl_path = PROCESSED_DIR / f"{split}_labels.npy"

    if not emb_path.exists() or not lbl_path.exists():
        raise FileNotFoundError(
            f"Processed files not found for split '{split}'. "
            "Run generate_embeddings.py first."
        )

    embeddings = np.load(emb_path)
    labels = np.load(lbl_path)
    logger.info("Loaded embeddings %s and labels %s for split '%s'.", embeddings.shape, labels.shape, split)
    return embeddings, labels


# ---------------------------------------------------------------------------
# Train / val / test splitter (60/20/20 by default, stratified)
# ---------------------------------------------------------------------------

def split_train_val_test(
    X: "np.ndarray",
    y: "np.ndarray",
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Stratified 60/20/20 (default) split using two passes of train_test_split.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split

    # First peel off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    # Then split the remainder into train + val
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=rel_val, random_state=seed, stratify=y_temp,
    )
    logger.info(
        "Stratified split: train=%d  val=%d  test=%d  (train_frac=%.2f val_frac=%.2f test_frac=%.2f)",
        len(X_train), len(X_val), len(X_test),
        1 - val_size - test_size, val_size, test_size,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Label engineering (synthetic fraud labels when absent)
# ---------------------------------------------------------------------------

def create_synthetic_fraud_labels(n_samples: int, fraud_ratio: float = 0.15, seed: int = 42) -> np.ndarray:
    """Generate pseudo fraud labels for datasets that lack explicit fraud/legit annotations.

    In real deployments labels come from business logic or manual review.
    Here we assign the minority class randomly to simulate an imbalanced fraud scenario.

    Args:
        n_samples:   Total number of samples.
        fraud_ratio: Fraction of samples labelled as fraud (class 1).
        seed:        Random seed for reproducibility.

    Returns:
        Binary integer array of shape (n_samples,).
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros(n_samples, dtype=int)
    n_fraud = max(1, int(n_samples * fraud_ratio))
    fraud_idx = rng.choice(n_samples, size=n_fraud, replace=False)
    labels[fraud_idx] = 1
    logger.info(
        "Synthetic labels created: %d legitimate, %d fraud (%.1f%%).",
        n_samples - n_fraud, n_fraud, 100 * fraud_ratio,
    )
    return labels


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from dataset_loader import load_hf_dataset

    raw = load_hf_dataset()
    split_name = list(raw.keys())[0]
    ds = raw[split_name]

    inspect_dataset(ds)
    get_label_distribution(ds)

    # Demonstrate save / load round-trip
    dummy_emb = np.random.randn(10, 128).astype(np.float32)
    dummy_lbl = np.random.randint(0, 2, 10)
    save_processed_arrays(dummy_emb, dummy_lbl, split="smoke")
    emb, lbl = load_processed_arrays(split="smoke")
    assert emb.shape == dummy_emb.shape
    logger.info("preprocess.py smoke-test passed.")
