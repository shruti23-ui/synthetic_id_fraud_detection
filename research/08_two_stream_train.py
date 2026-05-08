"""
research/08_two_stream_train.py
-------------------------------
Standalone research wrapper around src/train_two_stream.train_two_stream.

The actual training logic lives in src/train_two_stream.py so that it can
be imported by main.py as Stage 9 of the master pipeline. This script
just sets up logging and calls the training function with the default
CONFIG.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from train_two_stream import CONFIG, train_two_stream  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


if __name__ == "__main__":
    train_two_stream(CONFIG)
