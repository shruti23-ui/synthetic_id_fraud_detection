"""
local_dataset_loader.py
-----------------------
Loads the local fake/real document dataset from templates/.

Structure expected:
  templates/Images/fakes/<name>.jpg   → label 1 (fraud)
  templates/Images/reals/<name>.jpg   → label 0 (legitimate)
  templates/Annotations/fakes/<name>.json  (one JSON per fake image)
  templates/Annotations/reals/alb_id.json  (VIA format, covers all real images)

Returns PyTorch Datasets compatible with the rest of the pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

_TEMPLATES_ROOT = Path("templates")
_FAKE_IMAGES    = _TEMPLATES_ROOT / "Images"   / "fakes"
_REAL_IMAGES    = _TEMPLATES_ROOT / "Images"   / "reals"
_FAKE_ANNOTS    = _TEMPLATES_ROOT / "Annotations" / "fakes"
_REAL_ANNOTS    = _TEMPLATES_ROOT / "Annotations" / "reals"

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Multi-task forgery-type vocabulary
# 0 = real (no forgery), 1 = Inpaint_and_Rewrite, 2 = Crop_and_Replace
CTYPE_VOCAB: dict[str, int] = {
    "real":                0,
    "Inpaint_and_Rewrite": 1,
    "Crop_and_Replace":    2,
}
NUM_CTYPES = len(CTYPE_VOCAB)


def _ctype_to_int(ctype_str: str) -> int:
    return CTYPE_VOCAB.get(ctype_str, 0)


def compute_ctype_class_weights(records: list[dict]) -> list[float]:
    """Inverse-frequency class weights for the multi-task ctype head.

    Returned list is in CTYPE_VOCAB integer order (so it can be passed
    directly to torch.nn.CrossEntropyLoss(weight=...)).
    """
    counts = [0] * NUM_CTYPES
    for r in records:
        counts[r["ctype_label"]] += 1
    total = sum(counts)
    weights = [
        (total / (NUM_CTYPES * c)) if c > 0 else 1.0
        for c in counts
    ]
    return weights

# Disk-based pre-resize cache: avoids Python-process memory pressure
# (the in-memory cache caused libomp/numpy segfaults on Python 3.13 + Windows).
_RESIZED_ROOT = Path("data/processed/templates_256")


def _resized_path(orig: Path, label: int) -> Path:
    cls = "fakes" if label == 1 else "reals"
    return _RESIZED_ROOT / cls / f"{orig.stem}.jpg"


def _ensure_resized(records: list[dict], target: int = 256) -> None:
    """One-time: pre-resize all images to <=`target` short-side and save to disk.

    Skips files that already exist. Idempotent across runs. Greatly accelerates
    every later DataLoader iteration because PIL only has to decode small JPGs.
    """
    todo = []
    for rec in records:
        out = _resized_path(rec["path"], rec["label"])
        if not out.exists():
            todo.append((rec, out))
    if not todo:
        return

    (_RESIZED_ROOT / "fakes").mkdir(parents=True, exist_ok=True)
    (_RESIZED_ROOT / "reals").mkdir(parents=True, exist_ok=True)
    logger.info("Pre-resizing %d images to <=%dpx (one-time, saved to %s) ...",
                len(todo), target, _RESIZED_ROOT)

    for rec, out in tqdm(todo, desc="Pre-resize", leave=False):
        try:
            img = Image.open(rec["path"]).convert("RGB")
            w, h = img.size
            if min(w, h) > target:
                scale = target / min(w, h)
                img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
            img.save(out, "JPEG", quality=92)
        except Exception as exc:
            logger.warning("Skip %s (%s)", rec["path"], exc)
    logger.info("Pre-resize complete.")


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _load_fake_annotation(stem: str) -> dict:
    """Read per-image fake annotation JSON; return empty dict on failure."""
    path = _FAKE_ANNOTS / f"{stem}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Core record list builder
# ---------------------------------------------------------------------------

def _build_records(fake_dir: Path, real_dir: Path) -> list[dict]:
    """Return a list of {path, label, annotation} dicts for all images."""
    records = []

    # ── Fake images (label = 1) ───────────────────────────────────────────
    if fake_dir.exists():
        for img_path in sorted(fake_dir.iterdir()):
            if img_path.suffix.lower() in _IMAGE_EXTS:
                annot = _load_fake_annotation(img_path.stem)
                ctype = annot.get("ctype", "Inpaint_and_Rewrite")  # default for unparseable
                records.append({
                    "path":        img_path,
                    "label":       1,
                    "ctype":       ctype,
                    "ctype_label": _ctype_to_int(ctype),
                    "field":       annot.get("field", "unknown"),
                    "annotation":  annot,
                })
        logger.info("Loaded %d fake images from %s", sum(r["label"] == 1 for r in records), fake_dir)
    else:
        logger.warning("Fake images directory not found: %s", fake_dir)

    # ── Real images (label = 0) ───────────────────────────────────────────
    real_count_before = len(records)
    if real_dir.exists():
        for img_path in sorted(real_dir.iterdir()):
            if img_path.suffix.lower() in _IMAGE_EXTS:
                records.append({
                    "path":        img_path,
                    "label":       0,
                    "ctype":       "real",
                    "ctype_label": _ctype_to_int("real"),
                    "field":       "none",
                    "annotation":  {},
                })
        real_count = len(records) - real_count_before
        logger.info("Loaded %d real images from %s", real_count, real_dir)
    else:
        logger.warning("Real images directory not found: %s", real_dir)

    logger.info(
        "Total local dataset: %d images  (fake=%d, real=%d)",
        len(records),
        sum(r["label"] == 1 for r in records),
        sum(r["label"] == 0 for r in records),
    )
    return records


# ---------------------------------------------------------------------------
# Base dataset
# ---------------------------------------------------------------------------

def _pil_load(record: dict) -> Image.Image:
    """Open the resized JPG if it exists, else fall back to the original."""
    resized = _resized_path(record["path"], record["label"])
    return Image.open(resized if resized.exists() else record["path"]).convert("RGB")


class LocalFakeRealDataset(Dataset):
    """Flat dataset of all fake + real images with ground-truth labels."""

    def __init__(
        self,
        templates_root: Optional[Path] = None,
        transform=None,
        prepare: bool = True,
    ):
        root = Path(templates_root) if templates_root else _TEMPLATES_ROOT
        fake_dir = root / "Images" / "fakes"
        real_dir = root / "Images" / "reals"
        self.records   = _build_records(fake_dir, real_dir)
        self.transform = transform
        if prepare:
            _ensure_resized(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec   = self.records[idx]
        img   = _pil_load(rec)
        label = rec["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Contrastive Dataset (two augmented views)
# ---------------------------------------------------------------------------

class LocalContrastiveDataset(Dataset):
    """Two independently-augmented views of each image for SimCLR training."""

    def __init__(
        self,
        templates_root: Optional[Path] = None,
        transform=None,
        prepare: bool = True,
    ):
        root = Path(templates_root) if templates_root else _TEMPLATES_ROOT
        fake_dir = root / "Images" / "fakes"
        real_dir = root / "Images" / "reals"
        self.records   = _build_records(fake_dir, real_dir)
        self.transform = transform
        if prepare:
            _ensure_resized(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec   = self.records[idx]
        img   = _pil_load(rec)
        label = rec["label"]
        ctype = rec["ctype_label"]
        view1 = self.transform(img) if self.transform else img
        view2 = self.transform(img) if self.transform else img
        return view1, view2, label, ctype


# ---------------------------------------------------------------------------
# Embedding Dataset (single view)
# ---------------------------------------------------------------------------

class LocalEmbeddingDataset(Dataset):
    """Single-view dataset for embedding extraction after contrastive training."""

    def __init__(
        self,
        templates_root: Optional[Path] = None,
        transform=None,
        prepare: bool = True,
    ):
        root = Path(templates_root) if templates_root else _TEMPLATES_ROOT
        fake_dir = root / "Images" / "fakes"
        real_dir = root / "Images" / "reals"
        self.records   = _build_records(fake_dir, real_dir)
        self.transform = transform
        if prepare:
            _ensure_resized(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec   = self.records[idx]
        img   = _pil_load(rec)
        label = rec["label"]
        tensor = self.transform(img) if self.transform else img
        return tensor, label


# ---------------------------------------------------------------------------
# DataLoader factory helpers
# ---------------------------------------------------------------------------

def get_local_contrastive_loader(
    transform,
    templates_root: Optional[Path] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader for contrastive training on the local fake/real dataset."""
    ds = LocalContrastiveDataset(templates_root=templates_root, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def get_local_embedding_loader(
    transform,
    templates_root: Optional[Path] = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader for embedding extraction on the local fake/real dataset."""
    ds = LocalEmbeddingDataset(templates_root=templates_root, transform=transform)
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
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from augmentations import get_contrastive_transform, get_eval_transform

    loader = get_local_contrastive_loader(
        transform=get_contrastive_transform(image_size=224),
        batch_size=8,
    )
    v1, v2, labels = next(iter(loader))
    logger.info("view1: %s  view2: %s  labels: %s", v1.shape, v2.shape, labels.tolist())
    logger.info("local_dataset_loader.py smoke-test passed.")
