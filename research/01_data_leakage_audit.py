"""
research/01_data_leakage_audit.py
---------------------------------
THE FIRST QUESTION any IEEE / NeurIPS reviewer will ask of a 0.9941 ROC-AUC
on 2222 images: "Is the train/test split *truly* independent, or is the
model memorising templates that appear on both sides?"

This script audits exactly that.

Two leakage channels are tested:

  1. SOURCE-TEMPLATE LEAKAGE (the big one).
     Each fake annotation in templates/Annotations/fakes/<name>.json
     contains a `src` field pointing to a real image. The 1222 fakes
     are derived from a small number of source templates (e.g.
     alb_id_00.jpg). If a real template lands in TRAIN and any of
     its derived fakes land in TEST, the model can score by template
     identity, not by forgery cues.

  2. NEAR-DUPLICATE LEAKAGE.
     Even within the same class, are there visually-near-duplicate
     images split across train and test? We compute perceptual hashes
     (pHash + dHash) and report the cross-split duplicate rate.

Outputs:
    research_outputs/01_leakage_audit.json     numerical findings
    research_outputs/01_leakage_audit.md       human-readable report
    research_outputs/01_overlap_matrix.png     visualisation
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from local_dataset_loader import _build_records, _TEMPLATES_ROOT  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR = Path("research_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Source-template leakage
# ---------------------------------------------------------------------------

def template_id(record: dict) -> str:
    """Return the underlying template ID for any image record.

    For real images the template ID is the file stem (e.g. 'alb_id_00').
    For fakes it is the 'src' field of the annotation JSON, with .jpg stripped
    (e.g. 'alb_id_00').
    """
    label = record["label"]
    stem = record["path"].stem

    if label == 0:
        # real image: the stem already IS the template id
        return stem

    # fake: read its annotation; fall back to heuristic
    annot = record.get("annotation", {}) or {}
    src = annot.get("src", "")
    if src:
        return Path(src).stem
    # heuristic fallback: 'alb_id_00_fake_6_25' -> 'alb_id_00'
    parts = stem.split("_fake_")
    return parts[0] if parts else stem


def audit_source_template_leakage(
    records: list[dict], seed: int = 42, val_size: float = 0.20, test_size: float = 0.20,
) -> dict:
    """Reproduce the *image-level* 60/20/20 split used by classifier.py /
    supervised_finetune.py and measure how many template IDs are shared
    across train/val/test.
    """
    labels = np.array([r["label"] for r in records])
    template_ids = [template_id(r) for r in records]

    # Replicate the exact split logic from supervised_finetune.make_loaders
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in (0, 1):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_test = int(round(n * test_size))
        n_val  = int(round(n * val_size))
        test_idx.extend(cls_idx[:n_test].tolist())
        val_idx.extend(cls_idx[n_test:n_test + n_val].tolist())
        train_idx.extend(cls_idx[n_test + n_val:].tolist())

    train_idx, val_idx, test_idx = sorted(train_idx), sorted(val_idx), sorted(test_idx)

    train_tpl = {template_ids[i] for i in train_idx}
    val_tpl   = {template_ids[i] for i in val_idx}
    test_tpl  = {template_ids[i] for i in test_idx}

    train_test_shared = train_tpl & test_tpl
    train_val_shared  = train_tpl & val_tpl
    val_test_shared   = val_tpl & test_tpl

    # How many TEST images sit on a template that also has SOMETHING in TRAIN?
    test_records_leaking = sum(1 for i in test_idx if template_ids[i] in train_tpl)
    # And of those leaks, how many are CROSS-CLASS leaks (real on one side, fake on other)?
    cross_class_leaks = 0
    train_template_classes: dict[str, set[int]] = defaultdict(set)
    for i in train_idx:
        train_template_classes[template_ids[i]].add(labels[i])
    for i in test_idx:
        tpl = template_ids[i]
        if tpl in train_template_classes:
            other_class = (1 - labels[i])
            if other_class in train_template_classes[tpl]:
                cross_class_leaks += 1

    return {
        "n_total":               len(records),
        "n_train":               len(train_idx),
        "n_val":                 len(val_idx),
        "n_test":                len(test_idx),
        "n_unique_templates":    len(set(template_ids)),
        "n_templates_train":     len(train_tpl),
        "n_templates_val":       len(val_tpl),
        "n_templates_test":      len(test_tpl),
        "n_templates_train_test_shared": len(train_test_shared),
        "n_templates_train_val_shared":  len(train_val_shared),
        "n_templates_val_test_shared":   len(val_test_shared),
        "n_test_records_with_template_in_train": test_records_leaking,
        "n_test_records_with_cross_class_leak":  cross_class_leaks,
        "fraction_test_leaking":           test_records_leaking / max(len(test_idx), 1),
        "fraction_test_cross_class_leak":  cross_class_leaks    / max(len(test_idx), 1),
        "shared_templates_sample": sorted(train_test_shared)[:10],
    }


# ---------------------------------------------------------------------------
# 2. Near-duplicate leakage via perceptual hashing
# ---------------------------------------------------------------------------

def phash(img: Image.Image, hash_size: int = 8) -> int:
    """Simple pHash via DCT of a small grey image. Returns 64-bit int."""
    img = img.convert("L").resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32)
    # 2-D DCT-II (use scipy if available, else manual)
    try:
        from scipy.fftpack import dct
        d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")
    except Exception:
        # fallback: just use the low-frequency block of the FFT magnitude
        d = np.abs(np.fft.fft2(a))
    block = d[:hash_size, :hash_size]
    med = np.median(block[1:].flatten())  # exclude DC component
    bits = (block > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def audit_near_duplicates(
    records: list[dict],
    seed: int = 42,
    val_size: float = 0.20,
    test_size: float = 0.20,
    hamming_threshold: int = 8,
) -> dict:
    """Compute pHashes for every image, then count near-duplicates that
    sit on opposite sides of the train/test split.
    Hamming <= 8 on a 64-bit pHash is the standard "perceptually similar"
    threshold (~12 % of bits differ).
    """
    labels = np.array([r["label"] for r in records])

    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in (0, 1):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_test = int(round(n * test_size))
        n_val  = int(round(n * val_size))
        test_idx.extend(cls_idx[:n_test].tolist())
        val_idx.extend(cls_idx[n_test:n_test + n_val].tolist())
        train_idx.extend(cls_idx[n_test + n_val:].tolist())

    logger.info("Computing pHash for %d images ...", len(records))
    hashes = []
    for i, r in enumerate(records):
        # Prefer the pre-resized 256-px version if it exists (much faster)
        cls_dir = "fakes" if r["label"] == 1 else "reals"
        resized = Path("data/processed/templates_256") / cls_dir / f"{r['path'].stem}.jpg"
        path = resized if resized.exists() else r["path"]
        try:
            with Image.open(path) as im:
                hashes.append(phash(im))
        except Exception:
            hashes.append(0)
        if (i + 1) % 500 == 0:
            logger.info("  hashed %d / %d", i + 1, len(records))

    train_hashes = [(i, hashes[i], int(labels[i])) for i in train_idx]
    test_hashes  = [(i, hashes[i], int(labels[i])) for i in test_idx]

    # For each test image, find its nearest train neighbour (same-class and overall)
    near_dup_total = 0
    near_dup_cross_class = 0
    nearest_distances = []

    for ti, th, tlbl in test_hashes:
        best = 64
        best_lbl = None
        for _, h, lbl in train_hashes:
            d = hamming(th, h)
            if d < best:
                best = d
                best_lbl = lbl
        nearest_distances.append(best)
        if best <= hamming_threshold:
            near_dup_total += 1
            if best_lbl != tlbl:
                near_dup_cross_class += 1

    return {
        "hamming_threshold":            hamming_threshold,
        "n_test":                       len(test_idx),
        "n_test_with_train_near_dup":   near_dup_total,
        "n_test_with_cross_class_near_dup": near_dup_cross_class,
        "fraction_test_near_dup":       near_dup_total / max(len(test_idx), 1),
        "fraction_test_cross_class_near_dup": near_dup_cross_class / max(len(test_idx), 1),
        "median_nearest_distance":      float(np.median(nearest_distances)),
        "mean_nearest_distance":        float(np.mean(nearest_distances)),
        "p10_nearest_distance":         float(np.percentile(nearest_distances, 10)),
        "nearest_distance_histogram":   np.histogram(
            nearest_distances, bins=list(range(0, 33))
        )[0].tolist(),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_template_overlap(template_audit: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    splits = ["Train", "Val", "Test"]
    n_total = template_audit["n_unique_templates"]
    counts = [
        template_audit["n_templates_train"],
        template_audit["n_templates_val"],
        template_audit["n_templates_test"],
    ]
    bars = ax.bar(splits, counts, color=["#1565C0", "#43A047", "#D32F2F"], edgecolor="white")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 1,
                f"{c}\n({c/n_total*100:.0f}% of {n_total})",
                ha="center", va="bottom", fontsize=10)

    title = (
        f"Source-template overlap across image-level 60/20/20 split\n"
        f"train ∩ test templates: {template_audit['n_templates_train_test_shared']}"
        f"   |   leaky test images: {template_audit['n_test_records_with_template_in_train']} / "
        f"{template_audit['n_test']}"
        f" ({template_audit['fraction_test_leaking']*100:.1f} %)\n"
        f"cross-class leaks (real & fake of same template across train/test): "
        f"{template_audit['n_test_records_with_cross_class_leak']}"
        f" ({template_audit['fraction_test_cross_class_leak']*100:.1f} %)"
    )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Unique templates appearing in split")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, n_total * 1.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_duplicate_histogram(dup_audit: dict, out_path: Path) -> None:
    hist = dup_audit["nearest_distance_histogram"]
    bins = list(range(0, len(hist) + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(bins[:-1], hist, color="#7B1FA2", edgecolor="white", width=0.9)
    ax.axvline(dup_audit["hamming_threshold"], color="red", linestyle="--", lw=1.5,
               label=f"near-dup threshold = {dup_audit['hamming_threshold']}")
    ax.set_xlabel("Hamming distance to nearest train pHash (lower = more similar)")
    ax.set_ylabel("Number of TEST images")
    ax.set_title(
        f"pHash distance from each TEST image to its nearest TRAIN neighbour\n"
        f"median = {dup_audit['median_nearest_distance']:.1f}, "
        f"p10 = {dup_audit['p10_nearest_distance']:.1f}, "
        f"<= {dup_audit['hamming_threshold']}: "
        f"{dup_audit['fraction_test_near_dup']*100:.1f} % of test ({dup_audit['n_test_with_train_near_dup']} images)",
        fontsize=11,
    )
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fake_dir = _TEMPLATES_ROOT / "Images" / "fakes"
    real_dir = _TEMPLATES_ROOT / "Images" / "reals"
    records = _build_records(fake_dir, real_dir)

    # Quick template-id sanity check: distribution of templates per fake
    tpl_counter = Counter(template_id(r) for r in records)
    logger.info("Total unique templates: %d", len(tpl_counter))
    logger.info("Top 5 most-used templates: %s", tpl_counter.most_common(5))

    template_audit = audit_source_template_leakage(records)
    logger.info("Template audit: %s", json.dumps(template_audit, indent=2))

    dup_audit = audit_near_duplicates(records)
    logger.info("Duplicate audit (subset): median=%.1f, near-dup rate=%.3f",
                dup_audit["median_nearest_distance"],
                dup_audit["fraction_test_near_dup"])

    # Save raw findings
    with open(OUT_DIR / "01_leakage_audit.json", "w", encoding="utf-8") as f:
        json.dump({"template": template_audit, "near_duplicate": dup_audit}, f, indent=2)

    plot_template_overlap(template_audit, OUT_DIR / "01_template_overlap.png")
    plot_duplicate_histogram(dup_audit,    OUT_DIR / "01_phash_distance_hist.png")

    # Human-readable summary
    md = []
    md.append("# Data-leakage audit\n")
    md.append("## 1. Source-template leakage\n")
    md.append(f"- Total records: {template_audit['n_total']}")
    md.append(f"- Total unique source templates: {template_audit['n_unique_templates']}")
    md.append(f"- Templates appearing in TRAIN: {template_audit['n_templates_train']}")
    md.append(f"- Templates appearing in TEST:  {template_audit['n_templates_test']}")
    md.append(f"- **Templates shared train ∩ test:** "
              f"{template_audit['n_templates_train_test_shared']}")
    md.append(f"- **TEST images whose template also appears in TRAIN:** "
              f"{template_audit['n_test_records_with_template_in_train']} / "
              f"{template_audit['n_test']} "
              f"({template_audit['fraction_test_leaking']*100:.1f} %)")
    md.append(f"- **Cross-class template leaks (real of template T in train, fake of T in test or vice versa):** "
              f"{template_audit['n_test_records_with_cross_class_leak']} "
              f"({template_audit['fraction_test_cross_class_leak']*100:.1f} %)\n")
    md.append("Sample of shared templates: " + ", ".join(template_audit["shared_templates_sample"]) + "\n")

    md.append("## 2. Near-duplicate (pHash) leakage\n")
    md.append(f"- Hamming threshold for 'near-duplicate': "
              f"{dup_audit['hamming_threshold']} bits out of 64")
    md.append(f"- Median nearest-train distance: {dup_audit['median_nearest_distance']:.1f}")
    md.append(f"- 10th percentile: {dup_audit['p10_nearest_distance']:.1f}")
    md.append(f"- Test images with a near-duplicate in train: "
              f"{dup_audit['n_test_with_train_near_dup']} / {dup_audit['n_test']} "
              f"({dup_audit['fraction_test_near_dup']*100:.1f} %)")
    md.append(f"- Of those, **cross-class** near-duplicates: "
              f"{dup_audit['n_test_with_cross_class_near_dup']} "
              f"({dup_audit['fraction_test_cross_class_near_dup']*100:.1f} %)\n")

    md.append("## Interpretation\n")
    if template_audit["fraction_test_leaking"] > 0.5:
        md.append("> **CRITICAL:** more than half of the test set sits on a template "
                  "that the model already saw during training. The 0.99 ROC-AUC "
                  "almost certainly includes a substantial 'template-memorisation' "
                  "component. The headline number is *not* a clean measurement of "
                  "the model's ability to detect forgery patterns on unseen IDs.\n")
    elif template_audit["fraction_test_leaking"] > 0.2:
        md.append("> **WARNING:** non-trivial template overlap between train and test. "
                  "Some of the high accuracy is from template familiarity rather than "
                  "forgery understanding. Re-split by template to get a clean number.\n")
    else:
        md.append("> Template overlap is below 20 %, but the cross-class leak rate "
                  "(same template in train as one class and in test as the other) "
                  "is the metric that actually matters. Inspect carefully.\n")

    md.append("## Recommended fix\n")
    md.append("Re-split the dataset by **template ID** rather than by image: assign "
              "each unique template to exactly one of train / val / test, then take "
              "every real and every fake derived from that template into the same "
              "split. This eliminates the channel by which the model can score via "
              "template memorisation.\n")
    md.append("See `research/02_template_split_retrain.py` for the implementation.\n")

    with open(OUT_DIR / "01_leakage_audit.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info("Saved: %s", OUT_DIR / "01_leakage_audit.md")


if __name__ == "__main__":
    main()
