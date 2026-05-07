# Detecting Synthetic Identity Fraud in Digital Payments Using Self-Supervised Contrastive Learning

End-to-end ML pipeline (Python / PyTorch) for the bachelor thesis project.

The system learns visual representations of identity-document images via
**SimCLR** self-supervised contrastive learning and uses those embeddings to
train classical fraud classifiers — empirically outperforming raw
ImageNet-pretrained ResNet18 features.

---

## 1. Pipeline overview

```
  Raw images (real / fake IDs)
            │
            ▼
  ContrastiveDataset  ──►  SimCLR (ResNet18 + projection head, NT-Xent)
            │
            ▼
  Encoder features (512-d)  ──►  XGBoost / RandomForest / LR / MLP
            │
            ▼
  Evaluation, Visualisation, SHAP, Grad-CAM
```

| Stage | Module | Output |
|---|---|---|
| Contrastive pre-training | `src/train_contrastive.py` | `models/simclr_encoder_best.pth` |
| Embedding extraction | `src/generate_embeddings.py` | `data/processed/train_embeddings.npy` |
| Fraud classification | `src/classifier.py` | `models/best_classifier.pkl`, `outputs/metrics/classifier_results.csv` |
| Baseline vs SimCLR | `src/baseline_comparison.py` | `outputs/metrics/baseline_vs_simclr.csv` |
| Evaluation | `src/evaluate.py` | ROC / PR / confusion / threshold sweep |
| Visualisation | `src/visualization.py` | `outputs/plots/*.png` |
| Explainability | `src/explainability.py` | SHAP plots + Grad-CAM overlays |

---

## 2. Datasets

Two sources are supported:

1. **Local labeled** (default — `--source local`):
   `templates/Images/{fakes,reals}` — 1222 fake + 1000 real ID images with
   ground-truth labels. Per-image annotations live in
   `templates/Annotations/{fakes,reals}/`.

2. **HuggingFace** (`--source hf`):
   `sugiv/synthetic_cards` — fetched on demand via `download_dataset.py`.

---

## 3. Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU build (recommended — RTX 2050 / RTX 3060 / etc.):

```powershell
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

---

## 4. Run

```powershell
# Full pipeline (5 epochs, local dataset, batch 32)
python main.py --batch-size 32 --source local

# Only re-train the encoder for 50 epochs
python main.py --epochs 50 --skip-cls --skip-baseline --skip-eval --skip-viz --skip-shap

# Skip training and re-use existing checkpoint
python main.py --skip-train --batch-size 32
```

Useful flags: `--skip-train`, `--skip-embed`, `--skip-cls`, `--skip-baseline`,
`--skip-eval`, `--skip-viz`, `--skip-shap`, `--epochs N`, `--batch-size N`,
`--source {local,hf}`.

---

## 5. Project structure

```
.
├── main.py                       # master pipeline runner
├── download_dataset.py           # one-shot HF dataset fetcher
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/        ← HF cache + Arrow snapshots
│   └── processed/  ← embeddings.npy + labels.npy
├── templates/
│   ├── Images/{fakes,reals}/
│   └── Annotations/{fakes,reals}/
├── models/                       # checkpoints + best classifier
├── outputs/
│   ├── plots/      ← all PNG figures (thesis-ready)
│   ├── metrics/    ← CSV + npy artefacts
│   ├── embeddings/ ← per-split metadata CSVs
│   └── pipeline.log
└── src/
    ├── utils.py                  # seeds, device, AMP, EarlyStopping
    ├── augmentations.py          # SimCLR augmentations + eval transform
    ├── dataset_loader.py         # HF dataset loader + Contrastive/Embedding wrappers
    ├── local_dataset_loader.py   # local fake/real dataset loader
    ├── preprocess.py             # save/load arrays, label utilities
    ├── contrastive_model.py      # SimCLRModel + NTXentLoss
    ├── train_contrastive.py      # AMP-enabled training loop
    ├── generate_embeddings.py    # encoder feature extraction
    ├── classifier.py             # XGBoost / RF / LR / MLP
    ├── baseline_comparison.py    # raw ResNet vs SimCLR head-to-head
    ├── evaluate.py               # ROC / PR / confusion / thresholds
    ├── visualization.py          # 10+ thesis-quality plots
    └── explainability.py         # SHAP + Grad-CAM
```

---

## 6. Outputs (thesis-ready)

After a full run, expect the following under `outputs/`:

**Plots** (`outputs/plots/`)
- `loss_curve.png` — NT-Xent training loss
- `tsne_embeddings.png`, `umap_embeddings.png`, `pca_embeddings.png`
- `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`
- `class_distribution.png`
- `embedding_similarity_heatmap.png`
- `baseline_vs_simclr.png` — **the headline figure**
- `shap_summary_beeswarm.png`, `shap_summary_bar.png`, `shap_waterfall_fraud.png`
- `gradcam_overlays.png` — encoder attention on fake vs real IDs

**Metrics** (`outputs/metrics/`)
- `train_loss.csv`, `classifier_results.csv`
- `roc_curve_data.csv`, `pr_curve_data.csv`, `threshold_sweep.csv`
- `confusion_matrix.npy`, `confusion_matrix_norm.npy`
- `baseline_vs_simclr.csv`

---

## 7. Engineering notes

- **Mixed precision** (`torch.cuda.amp`) — enabled by default on CUDA, halves VRAM
- **Early stopping** — patience 8 epochs on training loss
- **Reproducibility** — `seed_everything(42)` fixes Python / NumPy / PyTorch / cuDNN
- **Modular** — every `src/*.py` runs standalone via `python src/<file>.py`
- **Config-driven** — defaults in `src/<file>.py:CONFIG`; YAML at `configs/default.yaml`
