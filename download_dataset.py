"""
download_dataset.py
-------------------
Downloads sugiv/synthetic_cards from HuggingFace and saves:
  - All images as JPEGs  →  data/raw/dataset/images/<split>/<idx>.jpg
  - Metadata CSV         →  data/raw/dataset/<split>_metadata.csv
  - HF Arrow format      →  data/raw/dataset/hf_dataset/

Run once before main.py:
    python download_dataset.py
"""

import os
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent
DATASET_DIR   = PROJECT_ROOT / "data" / "raw" / "dataset"
HF_CACHE_DIR  = PROJECT_ROOT / "data" / "raw" / "hf_cache"


def save_split(ds, split_name: str):
    """Save all images and metadata for one split to disk."""
    img_dir = DATASET_DIR / "images" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in tqdm(range(len(ds)), desc=f"Saving {split_name}"):
        sample = ds[idx]

        # ── Save image ────────────────────────────────────────────────────────
        img = sample.get("image") or sample.get("img")
        if img is None:
            logger.warning("No image at index %d", idx)
            continue

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_path = img_dir / f"{idx:05d}.jpg"
        img.save(img_path, "JPEG", quality=95)

        # ── Collect metadata ──────────────────────────────────────────────────
        row = {"idx": idx, "image_path": str(img_path.relative_to(PROJECT_ROOT))}
        for col in ds.column_names:
            if col in ("image", "img"):
                continue
            val = sample[col]
            row[col] = json.dumps(val) if isinstance(val, (dict, list)) else val
        rows.append(row)

    # Save metadata CSV
    meta_df = pd.DataFrame(rows)
    meta_path = DATASET_DIR / f"{split_name}_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    logger.info("Saved %d images → %s", len(rows), img_dir)
    logger.info("Metadata CSV  → %s", meta_path)
    return meta_df


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from HuggingFace (cache: %s) …", HF_CACHE_DIR)
    raw = load_dataset("sugiv/synthetic_cards", cache_dir=str(HF_CACHE_DIR))
    logger.info("Splits: %s", list(raw.keys()))

    for split_name, ds in raw.items():
        logger.info("Processing split '%s' (%d samples, columns: %s)", split_name, len(ds), ds.column_names)
        save_split(ds, split_name)

    # Also save in HF Arrow format for fast reloading
    hf_out = DATASET_DIR / "hf_dataset"
    logger.info("Saving HF Arrow format → %s", hf_out)
    raw.save_to_disk(str(hf_out))

    logger.info("=" * 50)
    logger.info("Dataset ready in: %s", DATASET_DIR)
    logger.info("Run the pipeline: python main.py")


if __name__ == "__main__":
    main()
