"""
main.py
-------
Master pipeline runner for:
    "Detecting Synthetic Identity Fraud in Digital Payments
     Using Self-Supervised Contrastive Learning"

Executes all stages in order:
    1. Dataset loading & inspection
    2. Contrastive pre-training (SimCLR)
    3. Embedding generation
    4. Fraud classification
    5. Evaluation
    6. Visualisation
    7. SHAP explainability

Usage:
    python main.py                   # run full pipeline
    python main.py --skip-train      # skip contrastive training (use existing checkpoint)
    python main.py --skip-embed      # skip embedding generation
    python main.py --epochs 10       # override epoch count
    python main.py --batch-size 32   # override batch size
"""

import argparse
import io
import logging
import sys
import time
from pathlib import Path

# Ensure outputs dir exists before FileHandler writes to it
Path("outputs").mkdir(parents=True, exist_ok=True)

# ── UTF-8 console (Windows cp1252 default can't render arrows etc.) ────────
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/pipeline.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


# ── CLI arguments ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic Identity Fraud Detection – Full ML Pipeline"
    )
    parser.add_argument("--skip-train",    action="store_true", help="Skip contrastive training")
    parser.add_argument("--skip-embed",    action="store_true", help="Skip embedding generation")
    parser.add_argument("--skip-cls",      action="store_true", help="Skip classifier training")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip raw vs SimCLR comparison")
    parser.add_argument("--skip-eval",     action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-viz",      action="store_true", help="Skip visualisations")
    parser.add_argument("--skip-shap",     action="store_true", help="Skip SHAP / Grad-CAM explainability")
    parser.add_argument("--epochs",      type=int,   default=None, help="Override training epoch count")
    parser.add_argument("--batch-size",  type=int,   default=None, help="Override batch size")
    parser.add_argument("--split",       type=str,   default=None, help="Override HuggingFace split name")
    parser.add_argument(
        "--source",
        type=str,
        default="local",
        choices=["local", "hf"],
        help="Data source: 'local' = templates/fake+real (default), 'hf' = HuggingFace",
    )
    return parser.parse_args()


# ── Stage runner ─────────────────────────────────────────────────────────────
def run_stage(name: str, fn, *args, **kwargs):
    """Execute a pipeline stage with timing and error handling."""
    logger.info("=" * 60)
    logger.info("STAGE: %s", name)
    logger.info("=" * 60)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        logger.info("STAGE DONE: %s  (%.1f s)", name, time.time() - t0)
        return result
    except Exception as exc:
        logger.exception("STAGE FAILED: %s  — %s", name, exc)
        raise


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Add src/ to path so stages can import each other
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

    from train_contrastive    import train,              CONFIG as TRAIN_CFG
    from generate_embeddings  import generate_embeddings, CONFIG as EMBED_CFG
    from classifier           import train_classifiers,  CONFIG as CLS_CFG
    from evaluate             import run_evaluation,     CONFIG as EVAL_CFG
    from visualization        import generate_all_plots
    from explainability       import run_explainability, CONFIG as SHAP_CFG
    from baseline_comparison  import run_baseline_comparison, CONFIG as BASELINE_CFG

    pipeline_start = time.time()
    logger.info("Synthetic Identity Fraud Detection – Pipeline Starting")

    # ── Apply CLI overrides ───────────────────────────────────────────────────
    if args.epochs:
        TRAIN_CFG["epochs"] = args.epochs
    if args.batch_size:
        TRAIN_CFG["batch_size"] = args.batch_size
        EMBED_CFG["batch_size"] = args.batch_size
    TRAIN_CFG["data_source"] = args.source
    EMBED_CFG["data_source"] = args.source
    logger.info("Data source: %s", args.source)

    # ── Stage 1: Contrastive Pre-training ─────────────────────────────────────
    if not args.skip_train:
        model = run_stage("Contrastive Pre-training (SimCLR)", train, TRAIN_CFG)
    else:
        logger.info("Skipping contrastive training (--skip-train).")

    # ── Stage 2: Embedding Generation ────────────────────────────────────────
    if not args.skip_embed:
        run_stage("Embedding Generation", generate_embeddings, EMBED_CFG)
    else:
        logger.info("Skipping embedding generation (--skip-embed).")

    # ── Stage 3: Fraud Classification ─────────────────────────────────────────
    if not args.skip_cls:
        run_stage("Fraud Classification", train_classifiers, CLS_CFG)
    else:
        logger.info("Skipping classifier training (--skip-cls).")

    # ── Stage 4: Baseline (Raw ResNet) vs SimCLR Comparison ──────────────────
    if not args.skip_baseline:
        run_stage("Baseline vs SimCLR Comparison", run_baseline_comparison, BASELINE_CFG)
    else:
        logger.info("Skipping baseline comparison (--skip-baseline).")

    # ── Stage 5: Evaluation ───────────────────────────────────────────────────
    if not args.skip_eval:
        run_stage("Evaluation", run_evaluation, EVAL_CFG)
    else:
        logger.info("Skipping evaluation (--skip-eval).")

    # ── Stage 6: Visualisations ───────────────────────────────────────────────
    if not args.skip_viz:
        run_stage("Visualisations", generate_all_plots)
    else:
        logger.info("Skipping visualisations (--skip-viz).")

    # ── Stage 7: SHAP + Grad-CAM Explainability ───────────────────────────────
    if not args.skip_shap:
        run_stage("Explainability (SHAP + Grad-CAM)", run_explainability, SHAP_CFG)
    else:
        logger.info("Skipping explainability (--skip-shap).")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE  (total time: %.1f s = %.1f min)", elapsed, elapsed / 60)
    logger.info("=" * 60)
    logger.info("Key outputs:")
    logger.info("  models/simclr_encoder_best.pth   – trained encoder")
    logger.info("  data/processed/                  – embeddings + labels")
    logger.info("  outputs/plots/                   – all figures")
    logger.info("  outputs/metrics/                 – CSV metrics")
    logger.info("  models/best_classifier.pkl       – best fraud classifier")
    logger.info("  outputs/pipeline.log             – full execution log")


if __name__ == "__main__":
    main()
