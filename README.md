# Detecting Synthetic Identity Fraud in Digital Payments Using Self-Supervised Contrastive Learning

End-to-end research-grade ML pipeline for detecting synthetic / forged
identity-document images. The system combines self-supervised contrastive
representation learning with three additions tailored to the document-forgery
problem:

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

## 1. Pipeline overview

```
  Local fake/real ID images (templates/)
            |
            v
  ContrastiveDataset           Forgery-aware augmentations (A)
   returns (v1, v2, label, ctype)        |
            |                            v
            +------> SimCLR encoder (ResNet18 + ProjectionHead)
            |                            |
            |        +-------------------+-----------------------+
            |        |                                            |
            |        v                                            v
            |   contrastive loss                          ctype head (C)
            |   SupCon (B) or NT-Xent              CrossEntropy(forgery type)
            |        |                                            |
            |        +-------- weighted sum (alpha=0.3) ----------+
            |                            |
            |                            v
            |                       total loss -> Adam + Cosine LR
            v
  Encoder features (512-d)
            |
            v
  60/20/20 stratified split
   train -> XGBoost / RandomForest / LogReg / MLP
            |
            v
  Evaluation, Visualisation, SHAP, Grad-CAM
```

| Stage | Module | Output |
|---|---|---|
| Contrastive pre-training (with forgery-aware aug + multi-task + SupCon) | `src/train_contrastive.py` | `models/simclr_encoder_best.pth` |
| Embedding extraction | `src/generate_embeddings.py` | `data/processed/train_embeddings.npy` |
| Fraud classification (4 models) | `src/classifier.py` | `models/best_classifier.pkl`, `outputs/metrics/classifier_results.csv` |
| Baseline (raw ResNet) vs SimCLR | `src/baseline_comparison.py` | `outputs/metrics/baseline_vs_simclr.csv` |
| Evaluation | `src/evaluate.py` | ROC / PR / confusion / threshold sweep |
| Visualisation | `src/visualization.py` | `outputs/plots/*.png` |
| Explainability | `src/explainability.py` | SHAP + Grad-CAM overlays |

---

## 2. Dataset

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

## 3. Setup (Python 3.11 + CUDA 12.4)

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
```

Verify the install:

```powershell
.\venv311\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: 2.6.0+cu124 True NVIDIA GeForce RTX 2050  (or your GPU)
```

---

## 4. Run

```powershell
# Full pipeline with all novelties enabled (default config: A + C + SupCon)
.\venv311\Scripts\python.exe -u main.py --batch-size 32 --source local --epochs 5

# Pure SimCLR baseline (toggle off the additions in train_contrastive.CONFIG)
#   set use_supcon = False
#   set use_multitask = False
#   set forgery_aware_aug = False

# Re-run downstream stages without retraining the encoder
.\venv311\Scripts\python.exe -u main.py --skip-train --batch-size 32

# Train encoder for longer
.\venv311\Scripts\python.exe -u main.py --epochs 50 --batch-size 32
```

CLI flags: `--skip-{train,embed,cls,baseline,eval,viz,shap}`, `--epochs N`,
`--batch-size N`, `--source {local,hf}`.

---

## 5. Per-epoch monitoring

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

## 6. Project structure

```
.
├── main.py                       # master pipeline runner
├── download_dataset.py           # one-shot HF dataset fetcher
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/        ← HF cache + Arrow snapshots
│   └── processed/  ← embeddings.npy + labels.npy + resized image cache
├── templates/                    # the local labeled dataset
│   ├── Images/{fakes,reals}/
│   └── Annotations/{fakes,reals}/
├── models/                       # checkpoints + best classifier
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
    ├── train_contrastive.py      # AMP-enabled, multi-task aware training loop
    ├── generate_embeddings.py    # encoder feature extraction
    ├── classifier.py             # XGBoost / RF / LR / MLP, 60/20/20 split
    ├── baseline_comparison.py    # raw ResNet vs SimCLR head-to-head
    ├── evaluate.py               # ROC / PR / confusion / threshold sweep
    ├── visualization.py          # 10 thesis-quality plots
    └── explainability.py         # SHAP (beeswarm/bar/waterfall) + Grad-CAM
```

---

## 7. Outputs (thesis-ready)

After a full run:

**Plots** (`outputs/plots/`)
- `loss_curve.png` — total / contrastive / multi-task loss components
- `tsne_embeddings.png`, `umap_embeddings.png`, `pca_embeddings.png`
- `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`
- `class_distribution.png`
- `embedding_similarity_heatmap.png` — block-diagonal structure proves the encoder separates fake vs real
- `baseline_vs_simclr.png` — **the headline figure** comparing raw ResNet to the SimCLR encoder across all 4 classifiers and 5 metrics
- `shap_summary_beeswarm.png`, `shap_summary_bar.png`, `shap_waterfall_fraud.png`
- `gradcam_overlays.png` — encoder attention on fake vs real IDs

**Metrics** (`outputs/metrics/`)
- `train_loss.csv` — per epoch: total / contrastive / multitask losses, lr, linear-probe metrics
- `classifier_results.csv` — VAL + TEST metrics for XGB / RF / LR / MLP
- `baseline_vs_simclr.csv` — same metrics for raw ResNet vs SimCLR features
- `roc_curve_data.csv`, `pr_curve_data.csv`, `threshold_sweep.csv`
- `confusion_matrix.npy`, `confusion_matrix_norm.npy`

---

## 8. Methodology and design choices

### 8.1 Why three changes on top of plain SimCLR

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

### 8.2 Data splits

- **SimCLR pre-training** uses all 2222 images (self-supervised; no held-out
  set is needed for the encoder).
- **Per-epoch monitoring** uses an internal 80/20 stratified linear-probe
  split on a 500-image subset purely for speed.
- **Downstream classifier** uses a stratified **60 / 20 / 20** train / val /
  test split. Best model is selected on **validation ROC-AUC**; test set is
  touched once for final reporting. The same split (seed = 42) is reused by
  `evaluate.py` and `baseline_comparison.py` so numbers are directly
  comparable across stages.

### 8.3 Engineering

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

## 9. References

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR), ICML 2020.
- Khosla et al., *Supervised Contrastive Learning* (SupCon), NeurIPS 2020.
- He et al., *Deep Residual Learning for Image Recognition* (ResNet), CVPR 2016.
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP), NeurIPS 2017.
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- Bulatov et al., *MIDV-2020: A Comprehensive Benchmark Dataset for Identity Document Analysis*, 2020 — source of the document templates used here.

---

## 10. Author

Shruti Priya — bachelor thesis, 2026.
