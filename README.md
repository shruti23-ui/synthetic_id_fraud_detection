# Detecting Synthetic Identity Fraud with a Two-Stream RGB+FFT Network

End-to-end research-grade ML pipeline for detecting synthetic / forged
identity-document images on the local fake/real dataset (1222 forgeries +
1000 reals).

The pipeline contains **three tracks** that culminate in a novel architecture:

* **Self-supervised track** — SimCLR contrastive pre-training (ResNet18) with
  three forgery-domain additions: forgery-aware augmentations, a multi-task
  forgery-type head, and SupCon (Khosla et al. 2020). Encoder features are
  consumed by classical classifiers (XGBoost / RF / LR / MLP).
* **Supervised baseline track** — end-to-end fine-tune of a ResNet50 with
  binary cross-entropy. The single-stream baseline.
* **Headline architecture (this work)** — **Two-Stream RGB + FFT network with
  per-branch Transformer fusion**: parallel ResNet50 (RGB) and ResNet18
  (log-magnitude FFT) backbones, each followed by a 2-layer Transformer
  encoder over the 7×7 spatial token grid, fused at the CLS-token level.
  Trained on a **template-aware** 60/20/20 split that eliminates the data
  leakage we discovered in the original image-level split.

## TL;DR — final test-set numbers (template-aware 60/20/20 split)

| Method | Acc | F1 | Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Raw ImageNet ResNet18 + LogReg (control) | 0.726 | 0.747 | 0.735 | 0.759 | 0.759 |
| SimCLR ResNet18 + LogReg (5-ep, undertrained) | 0.647 | 0.675 | 0.665 | 0.694 | 0.688 |
| Supervised ResNet50 end-to-end (single-stream) | 0.964 | 0.967 | 0.936 | 0.998 | 0.999 |
| **Two-Stream RGB+FFT + Transformer (headline)** | **0.989** | **0.990** | **0.992** | **0.998** | **0.999** |

The Two-Stream model **misses only 2 of 248 test forgeries** (99.19 %
recall) versus 16 missed by the ResNet50 baseline (93.55 % recall) —
a 5.6 % absolute recall gain at the same near-saturated ROC-AUC. For a
fraud-detection deployment, the recall delta is what matters.

> The original `RESULTS.md` headline of 0.9941 ROC-AUC was computed on an
> *image-level* split that contained massive data leakage (65 % source-
> template overlap, 99 % near-duplicate rate). That number is *not* a
> measurement of generalisation; the leakage analysis and recovery are
> documented in `RESEARCH_REPORT.md`.

---

## 1. Self-supervised contrastive contributions

The SimCLR track adds three pieces to canonical SimCLR, each motivated by
the document-forgery domain:

1. **Forgery-aware augmentations** — JPEG re-compression, patch erasing, and
   pixel-noise instead of generic SimCLR colour jitter, which would erase the
   chromatic inconsistencies that distinguish inpainted regions.
2. **Multi-task forgery-type head** — an auxiliary cross-entropy head predicts
   the forgery method (`real` / `Inpaint_and_Rewrite` / `Crop_and_Replace`)
   alongside the contrastive objective, exploiting per-image annotations that
   plain SimCLR ignores.
3. **Supervised contrastive loss (SupCon, Khosla et al. 2020)** — replaces
   NT-Xent so that all real images cluster together and all forgeries cluster
   together in embedding space, rather than relying purely on augmentation
   pairs.

The downstream binary fraud classifier is trained on the frozen encoder
features and compared head-to-head against a raw ImageNet-pretrained ResNet18
baseline so the contribution of each design choice can be measured.

---

## 2. Pipeline overview

```
  Local fake/real ID images (templates/)
       |
       +-------------------- TRACK A: self-supervised --------------------+
       |                                                                   |
       v                                                                   |
  ContrastiveDataset       forgery-aware augs                              |
   (v1, v2, label, ctype)        |                                          |
       |                         v                                          |
       +---> SimCLR encoder (ResNet18 + ProjectionHead)                     |
       |                         |                                          |
       |       +-----------------+----------------+                         |
       |       v                                  v                         |
       |  contrastive loss                  ctype head                      |
       |  SupCon or NT-Xent          CrossEntropy(3-class forgery type)    |
       |       +-------- weighted sum (alpha=0.3-0.5) -----+               |
       |                         v                                          |
       |               Adam + warmup + cosine LR                            |
       v                                                                   |
  encoder features (512-d) -> 60/20/20 split -> XGB / RF / LR / MLP       |
       |                                                                   |
       v                                                                   |
  Evaluation, Visualisation, SHAP, Grad-CAM                                |
                                                                            |
  +---------------------- TRACK B: supervised baseline -------------------+
  |                                                                        |
  ResNet50 (ImageNet V2 weights) -> 2-class head                           |
  end-to-end binary CE, AdamW lr=1e-4, cosine, AMP                        |
  60/20/20 split, best on val ROC-AUC -> final TEST evaluation            |
```

| Stage | Module | Output |
|---|---|---|
| 1. Contrastive pre-training (forgery-aware aug + multi-task + SupCon) | `src/train_contrastive.py` | `models/simclr_encoder_best.pth` |
| 2. Embedding extraction | `src/generate_embeddings.py` | `data/processed/train_embeddings.npy` |
| 3. Fraud classification (4 models on SimCLR features) | `src/classifier.py` | `models/best_classifier.pkl`, `outputs/metrics/classifier_results.csv` |
| 4. Baseline (raw ResNet) vs SimCLR | `src/baseline_comparison.py` | `outputs/metrics/baseline_vs_simclr.csv` |
| 5. Evaluation | `src/evaluate.py` | ROC / PR / confusion / threshold sweep |
| 6. Visualisation | `src/visualization.py` | `outputs/plots/*.png` |
| 7. Explainability | `src/explainability.py` | SHAP + Grad-CAM overlays |
| 8. Supervised ResNet50 (single-stream baseline) | `src/supervised_finetune.py` | `models/supervised_resnet50_best.pth` |
| **9. Two-Stream RGB+FFT + Transformer (headline)** | `src/two_stream_model.py` + `src/train_two_stream.py` | `models/two_stream_best.pth`, `research_outputs/08_two_stream_test_metrics.csv` |

### Standalone research suite

Beyond the 9 main-pipeline stages, the `research/` directory contains 8
diagnostic scripts (data-leakage audit, template-aware retrain, calibration,
failure analysis, robustness sweep, embedding geometry, architecture sweep,
two-stream training) that produce the publication-grade analysis in
`RESEARCH_REPORT.md`.

---

## 3. Dataset

**Local labeled** (default, `--source local`):

```
templates/
├── Images/
│   ├── fakes/                  1222 forged ID images
│   └── reals/                  1000 real ID images
└── Annotations/
    ├── fakes/<name>.json       per-image: ctype, field forged, source
    └── reals/<group>.json      VIA bbox annotations
```

Forgery-type distribution among the 1222 fakes:

| ctype | count |
|---|---:|
| `Inpaint_and_Rewrite` | 1077 |
| `Crop_and_Replace`    |  145 |

Annotations are exploited by the multi-task head; class imbalance is handled
via inverse-frequency CrossEntropy weights.

**HuggingFace** (`--source hf`): the `sugiv/synthetic_cards` dataset, fetched
on demand via `download_dataset.py`. Used as a comparison set; does not have
forgery-type annotations so multi-task is auto-disabled.

---

## 4. Setup (Python 3.11 + CUDA 12.4)

This project requires a CUDA-capable GPU. The codebase enforces this and
will refuse to fall back to CPU silently (`utils.get_device(strict=True)`).

```powershell
# 1. Create a Python 3.11 virtual environment
py -3.11 -m venv venv311
.\venv311\Scripts\python.exe -m pip install --upgrade pip

# 2. Install PyTorch with CUDA 12.4 wheels
.\venv311\Scripts\python.exe -m pip install `
    --index-url https://download.pytorch.org/whl/cu124 `
    torch==2.6.0+cu124 torchvision==0.21.0+cu124

# 3. Install the rest of the requirements
.\venv311\Scripts\python.exe -m pip install -r requirements.txt

# 4. (Optional) timm — required only by src/supervised_finetune.py for legacy backbones
.\venv311\Scripts\python.exe -m pip install timm
```

Verify the install:

```powershell
.\venv311\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: 2.6.0+cu124 True NVIDIA GeForce RTX 2050  (or your GPU)
```

---

## 5. Run

```powershell
# Full 9-stage pipeline (SimCLR + downstream + supervised + two-stream)
.\venv311\Scripts\python.exe -u main.py --batch-size 32 --source local --epochs 5

# Just the headline Two-Stream model (~8 min, 99% recall, 0.998 ROC-AUC)
.\venv311\Scripts\python.exe -u src\train_two_stream.py

# Just the single-stream ResNet50 baseline
.\venv311\Scripts\python.exe -u src\supervised_finetune.py

# Skip everything before the two new model stages
.\venv311\Scripts\python.exe -u main.py `
    --skip-train --skip-embed --skip-cls --skip-baseline `
    --skip-eval --skip-viz --skip-shap

# Standalone research suite (audits + diagnostics + arch sweep)
.\venv311\Scripts\python.exe -u research\01_data_leakage_audit.py
.\venv311\Scripts\python.exe -u research\02_template_split_retrain.py
.\venv311\Scripts\python.exe -u research\03_calibration_analysis.py
# ...etc through 07
```

CLI flags: `--skip-{train,embed,cls,baseline,eval,viz,shap,supervised,twostream}`,
`--epochs N`, `--batch-size N`, `--source {local,hf}`.

---

## 6. Per-epoch monitoring

During contrastive training the script reports, every epoch:

```
Epoch [  3/5]  total=4.31  contrastive=3.92  multitask=1.30  lr=4.5e-04  elapsed=98s
             [linear probe] acc=0.842 precision=0.853 recall=0.831 f1=0.842 roc_auc=0.918  (probe 7s)
```

The linear probe is a small logistic regression fit on encoder features at
each epoch end, evaluated on a held-out 20% subset of 500 stratified images.
This is the standard self-supervised representation-quality metric (see
SimCLR / MoCo papers) and lets you watch the encoder learn in real time.

---

## 7. Project structure

```
.
├── main.py                       # master pipeline runner (8 stages)
├── download_dataset.py           # one-shot HF dataset fetcher
├── requirements.txt
├── pyrightconfig.json            # Pylance config (extraPaths: src/)
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/        ← HF cache + Arrow snapshots
│   └── processed/  ← embeddings.npy + labels.npy + resized image cache
├── templates/                    # the local labeled dataset
│   ├── Images/{fakes,reals}/
│   └── Annotations/{fakes,reals}/
├── models/                       # checkpoints + best classifier
│   ├── simclr_encoder_best.pth        SimCLR ResNet18 encoder
│   ├── supervised_resnet50_best.pth   strong baseline
│   └── best_classifier.pkl            best XGB / RF / LR / MLP on SimCLR features
├── outputs/
│   ├── plots/      ← all PNG figures (thesis-ready)
│   ├── metrics/    ← CSV + npy artefacts
│   ├── embeddings/ ← per-split metadata CSVs
│   └── pipeline.log
└── src/
    ├── utils.py                  # seeds, strict GPU device, AMP, EarlyStopping
    ├── augmentations.py          # generic SimCLR + forgery-aware transforms
    ├── dataset_loader.py         # HF dataset loader + Contrastive/Embedding wrappers
    ├── local_dataset_loader.py   # local fake/real loader, ctype labels, class weights
    ├── preprocess.py             # save/load arrays, 60/20/20 splitter
    ├── contrastive_model.py      # SimCLRModel + ClassificationHead + NTXent + SupCon
    ├── train_contrastive.py      # AMP-enabled, multi-task aware contrastive training
    ├── supervised_finetune.py    # ResNet50 end-to-end binary fine-tune (single-stream baseline)
    ├── two_stream_model.py       # TwoStreamForgeryNet (RGB + FFT + Transformer) -- HEADLINE
    ├── train_two_stream.py       # training driver for the two-stream model
    ├── generate_embeddings.py    # encoder feature extraction
    ├── classifier.py             # XGBoost / RF / LR / MLP, 60/20/20 split
    ├── baseline_comparison.py    # raw ResNet vs SimCLR head-to-head
    ├── evaluate.py               # ROC / PR / confusion / threshold sweep
    ├── visualization.py          # 10 thesis-quality plots
    └── explainability.py         # SHAP (beeswarm/bar/waterfall) + Grad-CAM

research/                          # standalone diagnostic scripts (publication-grade)
├── 01_data_leakage_audit.py       # template overlap + pHash near-duplicate audit
├── 02_template_split_retrain.py   # clean retrain on template-aware split
├── 03_calibration_analysis.py     # ECE, Brier, reliability diagrams
├── 04_failure_analysis.py         # worst FP/FN with Grad-CAM overlays
├── 05_robustness_sweep.py         # 9 corruptions × 5 severities × 2 ckpts
├── 06_embedding_geometry.py       # silhouette, DB, CH, cluster purity, t-SNE/UMAP/PCA
├── 07_architecture_sweep.py       # ResNet50 / ConvNeXt-T / EfficientNetV2-S / ViT-S / Swin-T
└── 08_two_stream_train.py         # standalone wrapper for src/train_two_stream

research_outputs/                  # ~50 artefacts: JSON metrics, CSVs, PNG figures
RESEARCH_REPORT.md                 # IEEE-reviewer-style honest analysis
```

---

## 8. Outputs (thesis-ready)

After a full run:

**Plots** (`outputs/plots/`)
- `loss_curve.png` — total / contrastive / multi-task loss components
- `tsne_embeddings.png`, `umap_embeddings.png`, `pca_embeddings.png`
- `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`
- `class_distribution.png`
- `embedding_similarity_heatmap.png` — block-diagonal structure proves the encoder separates fake vs real
- `baseline_vs_simclr.png` — comparing raw ResNet to the SimCLR encoder across all 4 classifiers and 5 metrics
- `shap_summary_beeswarm.png`, `shap_summary_bar.png`, `shap_waterfall_fraud.png`
- `gradcam_overlays.png` — encoder attention on fake vs real IDs

**Metrics** (`outputs/metrics/`)
- `train_loss.csv` — SimCLR per-epoch: total / contrastive / multitask losses + linear-probe metrics
- `classifier_results.csv` — VAL + TEST metrics for XGB / RF / LR / MLP on SimCLR features
- `baseline_vs_simclr.csv` — same metrics for raw ResNet vs SimCLR features
- **`supervised_results.csv`** — final test row for the supervised ResNet50 fine-tune (the headline)
- `supervised_train.csv` — supervised per-epoch: train loss + val acc / recall / F1 / ROC-AUC / PR-AUC
- `roc_curve_data.csv`, `pr_curve_data.csv`, `threshold_sweep.csv`
- `confusion_matrix.npy`, `confusion_matrix_norm.npy`

---

## 9. Methodology and design choices

### 9.1 Why three changes on top of plain SimCLR

Plain SimCLR was designed for natural images where colour, lighting and
crop augmentations are class-preserving. For document forgery the situation
is the opposite — the forgery cue often *is* a chromatic inconsistency in
an inpainted region. Generic SimCLR augmentations therefore actively erase
the signal we want the encoder to learn. The three additions address this:

| Addition | What it changes | Why |
|---|---|---|
| Forgery-aware aug (A) | Drops grayscale + heavy ColorJitter, adds JPEG re-compression and patch erasing | Preserves chromatic forgery cues while still producing semantically equivalent positive pairs |
| Multi-task head (C) | Auxiliary CE loss predicting forgery type from encoder features | Forces the encoder to be discriminative w.r.t. how the image was forged, not just whether it was |
| SupCon (B) | All same-label samples treated as positives in the contrastive loss | Pulls all reals together and all fakes together, rather than only the two augmented views |

The combined loss is `L = L_contrastive + alpha * L_ctype` with `alpha = 0.3`.

### 9.2 Why ResNet50 for the supervised baseline (and not ResNet18 or Xception)

- **ResNet18** (11M params) is the canonical SimCLR backbone, kept for the
  self-supervised track to keep the comparison clean.
- **Xception** (20M, classic forgery-detection backbone in FaceForensics++)
  was tried first but did not converge under the two-stage + heavy-aug recipe
  on this small dataset.
- **ResNet50 with ImageNet V2 weights** (23.5M, AdamW lr=1e-4, cosine, AMP,
  light augmentations) is the proven supervised-fine-tune recipe. It trains
  in ~7 min on RTX 2050 and reaches **94 % test accuracy / 0.994 ROC-AUC**.

### 9.3 Data splits

- **SimCLR pre-training** uses all 2222 images (self-supervised; no held-out
  set is needed for the encoder).
- **Per-epoch monitoring** uses an internal 80/20 stratified linear-probe
  split on a 500-image subset purely for speed.
- **Downstream classifier** uses a stratified **60 / 20 / 20** train / val /
  test split. Best model is selected on **validation ROC-AUC**; test set is
  touched once for final reporting. The same split (seed = 42) is reused by
  `evaluate.py` and `baseline_comparison.py` so numbers are directly
  comparable across stages.

### 9.4 Engineering

- **Mixed precision** (`torch.amp.autocast('cuda')`) — halves VRAM, enables
  batch 32 on a 4 GB card without spilling
- **Early stopping** — patience 8 epochs on training loss
- **Reproducibility** — `seed_everything(42)` fixes Python / NumPy /
  PyTorch / cuDNN
- **Strict GPU** — `get_device(strict=True)` raises if CUDA is missing
- **Disk-based image cache** — images pre-resized once to 256-px short-side
  in `data/processed/templates_256/`, ~10× speedup vs. decoding originals
- **Modular** — every `src/*.py` runs standalone via `python src/<file>.py`
- **Config-driven** — defaults in `src/<file>.py:CONFIG`; YAML at
  `configs/default.yaml`

---

## 10. References

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR), ICML 2020.
- Khosla et al., *Supervised Contrastive Learning* (SupCon), NeurIPS 2020.
- He et al., *Deep Residual Learning for Image Recognition* (ResNet), CVPR 2016.
- Chollet, *Xception: Deep Learning with Depthwise Separable Convolutions*, CVPR 2017.
- Rossler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019.
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP), NeurIPS 2017.
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- Bulatov et al., *MIDV-2020: A Comprehensive Benchmark Dataset for Identity Document Analysis*, 2020 — source of the document templates used here.

---

## 10. Author

Shruti Priya — bachelor thesis, 2026.
