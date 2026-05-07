"""
generate_embeddings.py
----------------------
Extract encoder embeddings from the trained SimCLR model and persist them
to disk for downstream classification and visualisation.

Usage (standalone):
    python src/generate_embeddings.py

Outputs:
    data/processed/train_embeddings.npy
    data/processed/train_labels.npy
    outputs/embeddings/embeddings_metadata.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_loader import load_hf_dataset, get_embedding_loader
from local_dataset_loader import get_local_embedding_loader
from augmentations import get_eval_transform
from contrastive_model import SimCLRModel
from preprocess import save_processed_arrays, create_synthetic_fraud_labels, inspect_dataset
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
    "image_size":     224,
    "batch_size":     64,
    "num_workers":    0,
    "embedding_dim":  128,
    "model_path":     "models/simclr_encoder_best.pth",
    "embeddings_dir": "outputs/embeddings",
    "processed_dir":  "data/processed",
    "fraud_ratio":    0.15,   # used only if dataset has no explicit fraud label
    "data_source":    "local", # "local" = templates/ fake/real, "hf" = HuggingFace
}


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: SimCLRModel,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over the full dataset and collect (features, labels).

    Args:
        model:  Trained SimCLRModel (must be in eval mode).
        loader: DataLoader returning (image_tensor, label) batches.
        device: Compute device.

    Returns:
        Tuple (embeddings, labels) as numpy arrays.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device, non_blocking=True)
        features = model.get_features(images)           # (B, 512)
        all_embeddings.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_embeddings(config: dict = CONFIG) -> tuple[np.ndarray, np.ndarray]:
    """Full embedding-generation pipeline.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple (embeddings np.ndarray, labels np.ndarray).
    """
    Path(config["embeddings_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["processed_dir"]).mkdir(parents=True, exist_ok=True)

    device = get_device()

    # ── Load model ───────────────────────────────────────────────────────────
    model = SimCLRModel(embedding_dim=config["embedding_dim"], pretrained=False).to(device)

    model_path = Path(config["model_path"])
    if model_path.exists():
        ckpt = torch.load(str(model_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint from %s (epoch %s)", model_path, ckpt.get("epoch", "?"))
    else:
        logger.warning(
            "No checkpoint found at '%s'. Using randomly initialized weights. "
            "Run train_contrastive.py first for meaningful embeddings.",
            model_path,
        )

    # ── Load dataset ─────────────────────────────────────────────────────────
    transform = get_eval_transform(image_size=config["image_size"])
    results = {}

    if config.get("data_source", "local") == "local":
        logger.info("Extracting embeddings from local fake/real dataset (templates/) …")
        loader = get_local_embedding_loader(
            transform=transform,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
        embeddings, labels = extract_embeddings(model, loader, device)
        save_processed_arrays(embeddings, labels, split="train")

        meta_df = pd.DataFrame(
            embeddings, columns=[f"feat_{i}" for i in range(embeddings.shape[1])]
        )
        meta_df.insert(0, "label", labels)
        meta_path = Path(config["embeddings_dir"]) / "train_embeddings_metadata.csv"
        meta_df.to_csv(meta_path, index=False)
        logger.info("Metadata CSV saved → %s", meta_path)
        results["train"] = (embeddings, labels)

    else:
        raw = load_hf_dataset()
        for split_name, ds in raw.items():
            logger.info("Processing split: %s (%d samples)", split_name, len(ds))
            inspect_dataset(ds)

            loader = get_embedding_loader(
                ds,
                transform=transform,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
            )
            embeddings, labels = extract_embeddings(model, loader, device)

            if (labels == -1).all():
                logger.warning(
                    "No explicit labels — generating synthetic fraud labels (%.0f%%).",
                    config["fraud_ratio"] * 100,
                )
                labels = create_synthetic_fraud_labels(
                    len(embeddings), fraud_ratio=config["fraud_ratio"]
                )

            save_processed_arrays(embeddings, labels, split=split_name)

            meta_df = pd.DataFrame(
                embeddings, columns=[f"feat_{i}" for i in range(embeddings.shape[1])]
            )
            meta_df.insert(0, "label", labels)
            meta_path = Path(config["embeddings_dir"]) / f"{split_name}_embeddings_metadata.csv"
            meta_df.to_csv(meta_path, index=False)
            logger.info("Metadata CSV saved → %s", meta_path)
            results[split_name] = (embeddings, labels)

    logger.info("Embedding generation complete.")
    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = generate_embeddings()
    for split, (emb, lbl) in results.items():
        logger.info("  %s → embeddings %s, labels %s", split, emb.shape, lbl.shape)
    logger.info("generate_embeddings.py finished.")
