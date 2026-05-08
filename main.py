"""
main.py
-------
Master pipeline runner for:
    "Detecting Synthetic Identity Fraud in Digital Payments
     Using Self-Supervised Contrastive Learning"

Executes all stages in order:
    1. Contrastive pre-training (SimCLR)
    2. Embedding generation
    3. Fraud classification (XGBoost / RandomForest / LR / MLP on embeddings)
    4. Baseline (raw ResNet) vs SimCLR comparison
    5. Evaluation
    6. Visualisation
    7. Explainability (SHAP + Grad-CAM)
    8. Supervised end-to-end fine-tune (ResNet50 single-stream baseline)
    9. Two-Stream RGB+FFT + Transformer fusion (the headline model)

Usage:
    python main.py                       # run full pipeline (all 9 stages)
    python main.py --skip-train          # skip SimCLR pre-training
    python main.py --skip-supervised     # skip the supervised ResNet50 fine-tune
    python main.py --skip-twostream      # skip the Two-Stream RGB+FFT model
    python main.py --epochs 10           # override SimCLR epoch count
    python main.py --batch-size 32       # override batch size
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
# .reconfigure() exists on TextIOWrapper (the concrete stream class) but isn't
# declared on the TextIO protocol — hence the type: ignore.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

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
    parser.add_argument("--skip-supervised", action="store_true", help="Skip supervised ResNet50 end-to-end fine-tune")
    parser.add_argument("--skip-twostream",  action="store_true", help="Skip Two-Stream RGB+FFT model (the headline)")
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
    from supervised_finetune  import train_supervised,   CONFIG as SUP_CFG
    from train_two_stream     import train_two_stream,   CONFIG as TS_CFG

    pipeline_start = time.time()
    logger.info("Synthetic Identity Fraud Detection – Pipeline Starting")

    # ── Apply CLI overrides ───────────────────────────────────────────────────
    # --epochs N applies UNIFORMLY to every training stage (1, 8, 9). If a
    # stage benefits from a different epoch count, edit its CONFIG dict
    # directly (e.g. src/train_two_stream.py:CONFIG).
    if args.epochs:
        TRAIN_CFG["epochs"] = args.epochs
        SUP_CFG["epochs"]   = args.epochs
        TS_CFG["epochs"]    = args.epochs
    if args.batch_size:
        TRAIN_CFG["batch_size"] = args.batch_size
        EMBED_CFG["batch_size"] = args.batch_size
        SUP_CFG["batch_size"]   = args.batch_size
        TS_CFG["batch_size"]    = args.batch_size
    TRAIN_CFG["data_source"] = args.source
    EMBED_CFG["data_source"] = args.source
    logger.info("Data source: %s", args.source)

    # ── Stage 1: Contrastive Pre-training ─────────────────────────────────────
    # Trained encoder is persisted to disk; downstream stages reload it.
    if not args.skip_train:
        run_stage("Contrastive Pre-training (SimCLR)", train, TRAIN_CFG)
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

    # ── Stage 8: Supervised End-to-End Fine-Tune (ResNet50) ───────────────────
    # The single-stream supervised baseline — used in the thesis comparison
    # against the Two-Stream RGB+FFT model. Image-level 60/20/20 split
    # (so the leakage-audit numbers match RESULTS.md).
    if not args.skip_supervised:
        run_stage("Supervised End-to-End Fine-Tune (ResNet50, single-stream)",
                  train_supervised, SUP_CFG)
    else:
        logger.info("Skipping supervised fine-tune (--skip-supervised).")

    # ── Stage 9: Two-Stream RGB+FFT with Transformer fusion (HEADLINE) ────────
    # The thesis's novel architecture. Trained on the *template-aware*
    # 60/20/20 split (no template appears across splits), so the resulting
    # numbers are clean of the data-leakage that inflated stage 8's image-
    # level baseline. This is the model RESEARCH_REPORT.md uses as the
    # final answer to "what generalises on this dataset".
    if not args.skip_twostream:
        run_stage("Two-Stream RGB+FFT + Transformer (template-aware split)",
                  train_two_stream, TS_CFG)
    else:
        logger.info("Skipping two-stream model (--skip-twostream).")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE  (total time: %.1f s = %.1f min)", elapsed, elapsed / 60)
    logger.info("=" * 60)
    logger.info("Key outputs:")
    logger.info("  models/simclr_encoder_best.pth        - SimCLR encoder")
    logger.info("  models/supervised_resnet50_best.pth   - supervised ResNet50 (single-stream)")
    logger.info("  models/two_stream_best.pth            - HEADLINE: Two-Stream RGB+FFT + Transformer")
    logger.info("  models/best_classifier.pkl            - best XGBoost/RF/LR/MLP on SimCLR features")
    logger.info("  data/processed/                       - embeddings + labels")
    logger.info("  outputs/plots/                        - all SimCLR-pipeline figures")
    logger.info("  outputs/metrics/                      - CSV metrics for the SimCLR pipeline")
    logger.info("  research_outputs/                     - thesis-grade artefacts:")
    logger.info("    -> 02_template_test_metrics.csv     ResNet50 on clean (template-aware) split")
    logger.info("    -> 08_two_stream_test_metrics.csv   Two-Stream RGB+FFT (the headline)")
    logger.info("    -> 08_two_stream_vs_resnet50.png    head-to-head comparison plot")
    logger.info("  RESEARCH_REPORT.md                    - publication-grade IEEE-style report")
    logger.info("  outputs/pipeline.log                  - full execution log")


if __name__ == "__main__":
    main()
