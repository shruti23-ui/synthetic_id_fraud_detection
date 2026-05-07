"""
augmentations.py
----------------
SimCLR augmentation pipelines.

Two contrastive variants are provided:

  get_contrastive_transform(...)
    Generic SimCLR pipeline (strong color jitter, grayscale, blur).
    Suitable as a baseline.

  get_forgery_aware_transform(...)
    Forgery-aware pipeline that PRESERVES the forgery cues we need:
      - Drops aggressive ColorJitter / RandomGrayscale (they erase the chromatic
        inconsistencies that distinguish inpainted regions).
      - Reduces rotation (rectified IDs lose information under heavy rotation).
      - Adds RandomJPEGCompression (re-compression artefacts mimic real-world
        forgery pipelines and force the encoder to be robust to them).
      - Adds RandomErasing patch-cutout (simulates inpainted regions).
      - Adds light Gaussian noise (simulates pixel-level inpaint residuals).

Each call produces a different random augmentation, so applying the transform
twice to the same image yields two distinct positive views.
"""

import io
import random

import torch
from PIL import Image
import torchvision.transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Custom forgery-aware augmentations (PIL-input)
# ---------------------------------------------------------------------------

class RandomJPEGCompression:
    """Re-encode the image as JPEG at a random quality and decode again.

    This injects real-world JPEG compression artefacts. Useful for forgery
    detection because:
      - real fraud pipelines re-compress images at upload time
      - inpainted regions interact with JPEG quantisation in distinctive ways
      - the encoder learns to be invariant to compression but sensitive to
        the *residual* artefacts that survive both compressions
    """

    def __init__(self, p: float = 0.5, quality_range: tuple[int, int] = (60, 95)):
        self.p = p
        self.q_lo, self.q_hi = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(self.q_lo, self.q_hi)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Custom forgery-aware augmentations (Tensor-input — applied after ToTensor)
# ---------------------------------------------------------------------------

class GaussianPixelNoise:
    """Add small i.i.d. Gaussian noise per pixel (simulates inpaint residuals)."""

    def __init__(self, p: float = 0.5, sigma_range: tuple[float, float] = (0.005, 0.02)):
        self.p = p
        self.s_lo, self.s_hi = sigma_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        sigma = random.uniform(self.s_lo, self.s_hi)
        return (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Generic SimCLR contrastive transform (baseline — strong color augs)
# ---------------------------------------------------------------------------

def get_contrastive_transform(image_size: int = 224) -> T.Compose:
    """Generic SimCLR augmentation pipeline (kept as a baseline).

    Uses the canonical SimCLR augmentation set: heavy color jitter, grayscale,
    Gaussian blur, large random resized crops. Good for natural-image
    self-supervision; less appropriate for forgery detection where chromatic
    inconsistency is itself a signal.
    """
    color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

    return T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomApply([color_jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=int(0.1 * image_size) | 1, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Forgery-aware contrastive transform (designed for this dataset)
# ---------------------------------------------------------------------------

def get_forgery_aware_transform(image_size: int = 224) -> T.Compose:
    """Augmentation pipeline tuned for inpaint / crop-and-replace forgeries.

    Key changes vs. generic SimCLR:
      - color jitter LIGHT (was 0.4 -> 0.15) and only for brightness/contrast
      - grayscale REMOVED (chromatic inconsistencies are diagnostic)
      - rotation REDUCED 15 deg -> 5 deg (IDs are pre-rectified)
      - crop TIGHTER (scale 0.7-1.0 vs 0.6-1.0) so we don't crop out the field
      - JPEG re-compression added (forgery artefacts interact with JPEG)
      - patch erasing added (simulates inpainted regions)
      - light Gaussian pixel noise added
    """
    light_jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0)

    return T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=5),
        T.RandomApply([light_jitter], p=0.5),
        RandomJPEGCompression(p=0.5, quality_range=(60, 95)),
        T.GaussianBlur(kernel_size=int(0.05 * image_size) | 1, sigma=(0.05, 1.0)),
        T.ToTensor(),
        GaussianPixelNoise(p=0.5, sigma_range=(0.005, 0.02)),
        T.RandomErasing(p=0.4, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Evaluation / embedding-extraction transform (deterministic)
# ---------------------------------------------------------------------------

def get_eval_transform(image_size: int = 224) -> T.Compose:
    """Deterministic preprocessing pipeline for inference."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    dummy = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))

    for name, tf in [
        ("generic SimCLR", get_contrastive_transform(224)),
        ("forgery-aware ", get_forgery_aware_transform(224)),
        ("eval (deter.)  ", get_eval_transform(224)),
    ]:
        v = tf(dummy)
        v2 = tf(dummy) if "generic" in name or "forgery" in name else v
        diff = float((v - v2).abs().mean()) if v is not v2 else 0.0
        print(f"{name}: shape={tuple(v.shape)} dtype={v.dtype} mean-abs-diff(views)={diff:.4f}")
    print("augmentations.py smoke-test passed.")
