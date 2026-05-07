"""
utils.py
--------
Shared utilities used across the pipeline:

  - seed_everything()    : reproducible training
  - get_device()         : pick CUDA / CPU consistently
  - load_yaml_config()   : read configs/<name>.yaml
  - count_parameters()   : print trainable param count
  - timestamp()          : human-readable timestamp tag
  - AverageMeter         : running-mean tracker for training loops
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Set every relevant RNG so a run is bit-for-bit reproducible (modulo CUDA non-determinism)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seeds fixed at %d", seed)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(prefer_gpu: bool = True, strict: bool = True) -> torch.device:
    """Return the best available torch device and log VRAM info.

    If strict=True (default) and CUDA is unavailable, raises RuntimeError
    instead of silently falling back to CPU. This project requires a GPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "CUDA device: %s  |  %.2f GB VRAM  |  CUDA %s  |  PyTorch %s",
            props.name, props.total_memory / 1e9, torch.version.cuda, torch.__version__,
        )
        return device

    if prefer_gpu and strict:
        raise RuntimeError(
            "GPU is required but CUDA is unavailable. "
            "Install a CUDA-enabled torch build (e.g. torch==2.6.0+cu124) "
            "or pass strict=False to allow CPU fallback."
        )
    logger.warning("Falling back to CPU (no CUDA available)")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    """Load a YAML config file. Falls back to {} on failure."""
    try:
        import yaml  # noqa: WPS433
    except ImportError:
        logger.warning("PyYAML not installed — returning empty config")
        return {}

    p = Path(path)
    if not p.exists():
        logger.warning("Config file not found: %s — returning empty config", p)
        return {}

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    logger.info("Loaded config from %s", p)
    return cfg


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Timing / tracking helpers
# ---------------------------------------------------------------------------

def timestamp() -> str:
    """Return a filesystem-safe local timestamp like '20260507_213045'."""
    return time.strftime("%Y%m%d_%H%M%S")


class AverageMeter:
    """Computes and stores the running mean of a scalar metric."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum   += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Epochs with no improvement before stopping.
        delta:    Minimum change to count as improvement.
        mode:     "min" (e.g. loss) or "max" (e.g. ROC-AUC).
    """

    def __init__(self, patience: int = 7, delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.delta    = delta
        self.mode     = mode
        self.counter  = 0
        self.best     = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Return True if the run should stop."""
        if self.best is None:
            self.best = score
            return False

        improved = (
            (self.mode == "min" and score < self.best - self.delta) or
            (self.mode == "max" and score > self.best + self.delta)
        )

        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("Early stopping: no improvement for %d epochs", self.patience)
        return self.should_stop
