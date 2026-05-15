# Melanoma Classification

A deep learning pipeline for binary skin lesion classification (benign vs. malignant) using the [ISIC 2020 Challenge](https://www.kaggle.com/c/siim-isic-melanoma-classification) dataset.

**Mean OOF ROC-AUC: 0.9062 ± 0.0116**

---

## Overview

The model fuses two branches:
- **Image branch** — EfficientNet-B5 backbone with Generalized Mean Pooling (GeM)
- **Metadata branch** — patient age, sex, anatomical site → 256-dim MLP

Both are concatenated and passed to a binary classification head.

Training uses 5-fold stratified cross-validation with out-of-fold (OOF) predictions aggregated for final evaluation.

---

## Results

| Metric | Value |
|---|---|
| Mean ROC-AUC | 0.9062 |
| Std ROC-AUC | ±0.0116 |
| PR-AUC | 0.1806 |
| Sensitivity | 0.8647 |
| Specificity | 0.7810 |
| Accuracy | 0.7825 |

Per-fold AUC: 0.9011 / 0.8886 / 0.9231 / 0.9130 / 0.9051

---

## Key Techniques

| Technique | Detail |
|---|---|
| Backbone | EfficientNet-B5 (pretrained, timm) |
| Image size | 384×384 |
| Loss | Focal Loss (α=0.25, γ=2.0) |
| Optimizer | AdamW + Cosine Annealing w/ Warmup |
| Augmentation | Flips, rotation, brightness, color jitter, Gaussian noise |
| Inference | Test-Time Augmentation (TTA, 11 passes) |
| Class imbalance | WeightedRandomSampler + Focal Loss |
| Precision | Mixed precision (AMP) |

---

## Project Structure

```
.
├── main.ipynb          # Full pipeline: data → training → inference
├── config.py           # All hyperparameters (CFG class)
├── data/
│   ├── train.csv       # 33,127 training samples with metadata
│   ├── test.csv        # 10,983 test samples
│   └── images/
│       ├── train/
│       └── test/
└── outputs/
    ├── fold0_best.pth … fold4_best.pth   # Model checkpoints
    ├── fold_metrics.csv                  # Per-fold AUC scores
    ├── oof_predictions.csv               # OOF predictions
    ├── submission.csv                    # Final test predictions
    ├── ablation_results.csv              # Ablation study results
    └── ablations/                        # Per-experiment checkpoints
```

---

## Setup

```bash
pip install torch torchvision timm albumentations opencv-python \
            pandas numpy scikit-learn matplotlib
```

Requires CUDA for reasonable training times.

---

## Usage

Open `main.ipynb` and run all cells. The notebook is organized into sections:

1. **Data loading & preprocessing** — metadata encoding, augmentation setup
2. **Model definition** — image + metadata fusion architecture
3. **Ablation studies** — compare loss functions, sampling strategies, LR schedules
4. **5-fold training** — with early stopping (patience=4), saves best checkpoint per fold
5. **Inference** — TTA over all folds, outputs `outputs/submission.csv`

All hyperparameters are centralized in `config.py`.

---

## Ablation Findings

| Experiment | ROC-AUC |
|---|---|
| Focal Loss (final) | 0.9062 |
| BCE with pos_weight | lower |
| Image only (no metadata) | PR-AUC 0.1001 vs 0.1183 |
| Uniform LR | 0.8793 (best single-config) |
| No WeightedSampler | worse recall |

---

## Data

- **Source**: ISIC Skin Imaging Collaboration (Kaggle SIIM-ISIC 2020)
- **Train**: 33,127 images (~3.4 GB), ~4.8% malignant
- **Test**: 10,983 images (unlabeled)
- **Metadata**: patient ID, sex, age, anatomical site, diagnosis
