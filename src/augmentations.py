"""
augmentations.py
----------------
Defines the SimCLR augmentation pipeline and a lightweight eval transform.

SimCLR augmentations are applied *independently* to each image to produce two
correlated-but-different views for contrastive learning.
"""

import torchvision.transforms as T


# ---------------------------------------------------------------------------
# SimCLR contrastive transform (strong augmentation)
# ---------------------------------------------------------------------------

def get_contrastive_transform(image_size: int = 224) -> T.Compose:
    """Return the SimCLR augmentation pipeline.

    Each call produces a different random augmentation, so applying this
    transform twice to the same image yields two distinct views.

    Args:
        image_size: Target spatial resolution (square) after crop.

    Returns:
        torchvision.transforms.Compose
    """
    color_jitter = T.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
    )

    return T.Compose([
        T.Resize((image_size + 32, image_size + 32)),        # slightly larger before crop
        T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),   # random crop
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomApply([color_jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=int(0.1 * image_size) | 1, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],              # ImageNet statistics
                    std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Evaluation / embedding-extraction transform (deterministic)
# ---------------------------------------------------------------------------

def get_eval_transform(image_size: int = 224) -> T.Compose:
    """Return a deterministic preprocessing pipeline for inference.

    No randomness — used for embedding extraction and downstream
    classifier training/evaluation.

    Args:
        image_size: Target spatial resolution (square).

    Returns:
        torchvision.transforms.Compose
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    dummy = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

    tf_contrast = get_contrastive_transform(224)
    tf_eval = get_eval_transform(224)

    v1 = tf_contrast(dummy)
    v2 = tf_contrast(dummy)
    ev = tf_eval(dummy)

    print(f"Contrastive view 1 shape : {v1.shape}")
    print(f"Contrastive view 2 shape : {v2.shape}")
    print(f"Eval view shape          : {ev.shape}")
    print(f"Views are different      : {not (v1 == v2).all()}")
    print("augmentations.py smoke-test passed.")
