"""
dataset_loader.py
-----------------
Loads the sugiv/synthetic_cards dataset from Hugging Face and wraps it
in a PyTorch Dataset that returns two augmented views of each image
(required for SimCLR-style contrastive learning).
"""

import os
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk

logger = logging.getLogger(__name__)

# Local Arrow dataset saved by download_dataset.py
_LOCAL_HF_DATASET = Path("data/raw/dataset/hf_dataset")
_LOCAL_HF_CACHE   = Path("data/raw/hf_cache")


# ---------------------------------------------------------------------------
# Raw dataset loading
# ---------------------------------------------------------------------------

def load_hf_dataset(cache_dir: str = "data/raw") -> dict:
    """Load the synthetic_cards dataset.

    Priority:
      1. Local Arrow dataset (data/raw/dataset/hf_dataset/) — instant, no network
      2. HuggingFace download via API (cached in data/raw/hf_cache/)
    """
    # Prefer locally saved Arrow dataset (created by download_dataset.py)
    if _LOCAL_HF_DATASET.exists():
        logger.info("Loading dataset from local Arrow cache: %s", _LOCAL_HF_DATASET)
        dataset = load_from_disk(str(_LOCAL_HF_DATASET))
        # load_from_disk returns DatasetDict or Dataset; normalise to dict
        if hasattr(dataset, "keys"):
            pass  # already a DatasetDict
        else:
            dataset = {"train": dataset}
    else:
        # Fall back to HuggingFace download
        cache = str(_LOCAL_HF_CACHE) if _LOCAL_HF_CACHE.exists() else cache_dir
        logger.info("Loading dataset sugiv/synthetic_cards from HuggingFace (cache: %s) …", cache)
        Path(cache).mkdir(parents=True, exist_ok=True)
        dataset = load_dataset("sugiv/synthetic_cards", cache_dir=cache)

    logger.info("Dataset ready. Splits: %s", list(dataset.keys()))
    for split, ds in dataset.items():
        logger.info("  %s → %d samples, columns: %s", split, len(ds), ds.column_names)

    return dataset


# ---------------------------------------------------------------------------
# Contrastive Dataset (two views per image)
# ---------------------------------------------------------------------------

class ContrastiveDataset(Dataset):
    """Returns two differently-augmented views of each image for SimCLR.

    Args:
        hf_dataset: A Hugging Face Dataset split (e.g. dataset['train']).
        transform:  A callable that takes a PIL image and returns a tensor.
                    Applied independently twice to produce view1, view2.
        image_col:  Column name that contains the PIL image.
        label_col:  Column name for the integer label (used at eval time).
    """

    def __init__(self, hf_dataset, transform, image_col: str = "image", label_col: str = "label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col

        # Validate column names against what actually exists
        available_cols = hf_dataset.column_names
        if image_col not in available_cols:
            # Try common alternatives
            for alt in ("img", "pixel_values", "image_path"):
                if alt in available_cols:
                    self.image_col = alt
                    logger.warning("Column '%s' not found; using '%s'.", image_col, alt)
                    break
            else:
                raise ValueError(f"No image column found. Available: {available_cols}")

        # The sugiv/synthetic_cards dataset uses 'annotations' not 'label'
        self.has_labels = label_col in available_cols
        if not self.has_labels:
            logger.warning("Label column '%s' not found; labels will be -1.", label_col)

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_label(self, sample: dict) -> int:
        """Extract an integer label from the sample, falling back to -1."""
        if not self.has_labels:
            return -1
        val = sample[self.label_col]
        # Handle annotations dict (e.g. {"card_type": "credit"}) by hashing to 0/1
        if isinstance(val, dict):
            return 0
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    def _load_image(self, raw) -> Image.Image:
        """Normalise any image representation to PIL RGB."""
        if isinstance(raw, np.ndarray):
            img = Image.fromarray(raw)
        elif isinstance(raw, Image.Image):
            img = raw
        else:
            img = Image.fromarray(np.array(raw))
        return img.convert("RGB")

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        img = self._load_image(sample[self.image_col])
        view1 = self.transform(img)
        view2 = self.transform(img)
        label = self._get_label(sample)
        return view1, view2, label


# ---------------------------------------------------------------------------
# Embedding Dataset (single view, used after training)
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """Single-view dataset used for generating embeddings after contrastive training."""

    def __init__(self, hf_dataset, transform, image_col: str = "image", label_col: str = "label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col

        available_cols = hf_dataset.column_names
        if image_col not in available_cols:
            for alt in ("img", "pixel_values", "image_path"):
                if alt in available_cols:
                    self.image_col = alt
                    break

        self.has_labels = label_col in available_cols

    def __len__(self) -> int:
        return len(self.dataset)

    def _load_image(self, raw) -> Image.Image:
        if isinstance(raw, np.ndarray):
            img = Image.fromarray(raw)
        elif isinstance(raw, Image.Image):
            img = raw
        else:
            img = Image.fromarray(np.array(raw))
        return img.convert("RGB")

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        img = self._load_image(sample[self.image_col])
        tensor = self.transform(img)
        label = -1
        if self.has_labels:
            val = sample[self.label_col]
            label = 0 if isinstance(val, dict) else int(val) if val is not None else -1
        return tensor, label


# ---------------------------------------------------------------------------
# DataLoader factory helpers
# ---------------------------------------------------------------------------

def get_contrastive_loader(
    hf_dataset,
    transform,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader for contrastive training."""
    ds = ContrastiveDataset(hf_dataset, transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,   # required for NT-Xent with full batches
    )


def get_embedding_loader(
    hf_dataset,
    transform,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for embedding extraction (no augmentation required)."""
    ds = EmbeddingDataset(hf_dataset, transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from augmentations import get_contrastive_transform, get_eval_transform

    raw = load_hf_dataset()
    split = list(raw.keys())[0]
    logger.info("Using split: %s", split)

    loader = get_contrastive_loader(
        raw[split],
        transform=get_contrastive_transform(image_size=224),
        batch_size=8,
    )

    v1, v2, labels = next(iter(loader))
    logger.info("view1 shape: %s  view2 shape: %s  labels: %s", v1.shape, v2.shape, labels)
    logger.info("dataset_loader.py smoke-test passed.")
