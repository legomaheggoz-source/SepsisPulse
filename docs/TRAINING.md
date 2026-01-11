# SepsisPulse Model Training Guide

Complete guide for training XGBoost-TS and TFT-Lite models on the PhysioNet Challenge 2019 dataset.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Dataset Information](#dataset-information)
3. [Training Infrastructure](#training-infrastructure)
4. [Design Decisions](#design-decisions)
5. [CLI Reference](#cli-reference)
6. [Hyperparameter Optimization](#hyperparameter-optimization)
7. [Expected Results](#expected-results)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

```bash
# Install training dependencies
pip install -r requirements.txt

# Verify GPU availability (optional but recommended)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 1: Download PhysioNet Data (FREE)

```bash
python -m training.data.download_physionet --output-dir data/physionet
```

This downloads ~42MB and extracts to ~200MB (40,336 patients).

### Step 2: Quick Validation Run

```bash
# Test XGBoost with 100 patients
python -m training.scripts.train_xgboost --max-patients 100 --n-folds 3

# Test TFT with 100 patients
python -m training.scripts.train_tft --max-patients 100 --n-folds 3
```

### Step 3: Full Training

```bash
# XGBoost (~2 hours on RTX 4090)
python -m training.scripts.train_xgboost \
    --data-dir data/physionet \
    --n-folds 5 \
    --save-final

# TFT-Lite (~4 hours on RTX 4090)
python -m training.scripts.train_tft \
    --data-dir data/physionet \
    --n-folds 5 \
    --batch-size 1024 \
    --device cuda \
    --save-final
```

### Step 4: Verify and Deploy

```bash
# Check trained models exist
ls -lh models/xgboost_ts/weights/xgb_sepsis_v1.json
ls -lh models/tft_lite/weights/tft_lite_v1.pt

# Run app to verify
streamlit run app.py
```

---

## Dataset Information

### PhysioNet Computing in Cardiology Challenge 2019

| Property | Value |
|----------|-------|
| **Cost** | FREE (no payment required) |
| **License** | Open Database License (ODbL) v1.0 |
| **Commercial Use** | Allowed with attribution |
| **Registration** | Not required |
| **Size** | 42MB compressed, ~200MB extracted |
| **Patients** | 40,336 total |
| **Sepsis Rate** | ~7.3% of patients |
| **Features** | 41 (8 vital signs, 26 labs, 7 demographics) |

### Data Structure

```
data/physionet/
├── training_setA/          # 20,336 patients
│   ├── p000001.psv
│   ├── p000002.psv
│   └── ...
└── training_setB/          # 20,000 patients
    ├── p100001.psv
    └── ...
```

Each PSV (pipe-separated values) file contains hourly observations:
- **Vital Signs**: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2
- **Lab Values**: 26 measurements (often sparse)
- **Demographics**: Age, Gender, Unit, HospAdmTime, ICULOS
- **Label**: SepsisLabel (0 or 1)

---

## Training Infrastructure

### Directory Structure

```
training/
├── config/
│   ├── base.py              # Base configuration dataclass
│   ├── xgboost_config.py    # XGBoost hyperparameters
│   └── tft_config.py        # TFT-Lite hyperparameters
├── data/
│   ├── download_physionet.py # Data download script
│   ├── dataset.py            # PyTorch datasets
│   └── cross_validation.py   # Patient-level CV
├── trainers/
│   ├── base_trainer.py       # Abstract trainer interface
│   ├── xgboost_trainer.py    # XGBoost training loop
│   └── tft_trainer.py        # TFT PyTorch training
├── optimization/
│   └── optuna_study.py       # Hyperparameter search
├── callbacks/
│   ├── checkpointing.py      # Model checkpoints
│   ├── early_stopping.py     # Early stopping logic
│   └── logging_callbacks.py  # Metrics tracking
└── scripts/
    ├── train_xgboost.py      # XGBoost CLI
    ├── train_tft.py          # TFT CLI
    └── optimize_hyperparams.py # Optuna CLI
```

---

## Design Decisions

### 1. Patient-Level Cross-Validation

**Decision**: 5-fold stratified K-fold at patient level (NOT hour level)

**Rationale**:
- Splitting by hours would leak information (same patient in train+test)
- Model would learn patient-specific patterns, not generalizable sepsis signals
- Validation metrics would be artificially inflated

**Implementation**: All hours from a patient stay in the same fold. Stratification ensures each fold has similar sepsis prevalence (~7.3%).

### 2. Class Imbalance Handling

**Decision**: Cost-weighted loss (scale_pos_weight ~14:1 ratio)

**Rationale**:
- Sepsis is rare (~7.3% of patients, ~2.7% of hours)
- Resampling (SMOTE, etc.) can cause overfitting on clinical data
- Cost weighting directly optimizes for the clinical utility score
- Implemented via `scale_pos_weight` in XGBoost and `pos_weight` in BCEWithLogitsLoss

### 3. Primary Optimization Metric

**Decision**: Optimize for Clinical Utility Score, not AUC-ROC

**Rationale**:
- AUC-ROC doesn't capture the value of early detection
- Utility score directly rewards predictions 6-12 hours before onset
- Matches the PhysioNet Challenge 2019 official evaluation metric
- Penalizes alarm fatigue (false positives) appropriately

**Utility Score Weights**:
- True Positive (early detection): +1.0
- True Negative: 0.0
- False Positive: -0.05 (small penalty for alarm fatigue)
- False Negative: -2.0 (high penalty for missed sepsis)

### 4. Hyperparameter Search with Optuna

**Decision**: TPE sampler with MedianPruner, 50-100 trials

**Rationale**:
- TPE (Tree Parzen Estimator) is efficient for moderate-dimensional spaces
- Learns from previous trials to focus on promising regions
- MedianPruner stops unpromising trials early
- SQLite storage allows resuming interrupted studies
- 50-100 trials sufficient with 8-hour overnight budget

### 5. TFT-Lite Batch Size

**Decision**: batch_size=1024 for RTX 4090

**Rationale**:
- RTX 4090 has 24GB VRAM, can handle large batches
- Larger batches = better GPU utilization = faster training
- At batch_size=1024: ~10GB VRAM usage
- Leaves headroom for gradient computation and activations

### 6. Mixed Precision Training

**Decision**: Enable AMP (Automatic Mixed Precision)

**Rationale**:
- RTX 4090 Tensor Cores are optimized for FP16/BF16 operations
- ~2x memory reduction enables larger batch sizes
- ~1.5x training speedup
- No significant accuracy loss with GradScaler

### 7. Feature Engineering for XGBoost

**Decision**: Lag [1, 3, 6h], Rolling [3, 6, 12h], Delta features

**Rationale**:
- Lag features capture temporal dynamics at different scales
- 1 hour: Acute changes (new infection, treatment response)
- 3 hours: Short-term trends (within-shift patterns)
- 6 hours: Medium-term patterns (shift-to-shift dynamics)
- 12 hours: Long-term trends (circadian, day-to-day)
- Delta features detect rapid deterioration preceding sepsis
- Results in ~200-700 features from 41 raw variables

### 8. Model Weight Format

**Decision**: XGBoost=JSON, TFT=.pt state_dict

**Rationale**:
- JSON is portable, HuggingFace compatible, human-readable
- PyTorch state_dict is standard, efficient, version-agnostic
- Both formats work with existing `load_weights()` methods
- Total size expected: <10MB combined (fits HuggingFace free tier)

---

## CLI Reference

### Download Data

```bash
python -m training.data.download_physionet --help

Options:
  --output-dir PATH   Output directory (default: data/physionet)
  --sets [A B]        Which sets to download (default: both)
  --keep-zips         Keep zip files after extraction
  --force             Force re-download
```

### Train XGBoost

```bash
python -m training.scripts.train_xgboost --help

Options:
  --data-dir PATH           PhysioNet data directory
  --output-dir PATH         Output directory
  --config PATH             Load config from YAML
  --n-folds INT             CV folds (default: 5)
  --max-patients INT        Limit patients (for testing)
  --n-estimators INT        Max boosting rounds (default: 500)
  --max-depth INT           Tree depth (default: 6)
  --learning-rate FLOAT     Learning rate (default: 0.1)
  --device [cpu|cuda]       Training device
  --save-final              Save to models/xgboost_ts/weights/
```

### Train TFT

```bash
python -m training.scripts.train_tft --help

Options:
  --data-dir PATH           PhysioNet data directory
  --output-dir PATH         Output directory
  --config PATH             Load config from YAML
  --n-folds INT             CV folds (default: 5)
  --max-patients INT        Limit patients
  --hidden-size INT         Hidden size (default: 64)
  --n-heads INT             Attention heads (default: 4)
  --batch-size INT          Batch size (default: 1024)
  --n-epochs INT            Max epochs (default: 100)
  --device [cpu|cuda]       Training device
  --no-amp                  Disable mixed precision
  --save-final              Save to models/tft_lite/weights/
```

### Hyperparameter Optimization

```bash
python -m training.scripts.optimize_hyperparams --help

Options:
  --model [xgboost|tft]     Model to optimize (required)
  --data-dir PATH           PhysioNet data directory
  --output-dir PATH         Output directory
  --n-trials INT            Max trials (default: 100)
  --timeout-hours FLOAT     Time limit in hours
  --n-folds INT             CV folds per trial (default: 3)
  --max-patients INT        Limit patients per trial
```

---

## Hyperparameter Optimization

### Overnight Optimization Workflow

```bash
# Start XGBoost optimization (4 hours)
python -m training.scripts.optimize_hyperparams \
    --model xgboost \
    --n-trials 100 \
    --timeout-hours 4 \
    --output-dir training_outputs/optuna &

# Start TFT optimization (4 hours) - can run in parallel on second GPU
python -m training.scripts.optimize_hyperparams \
    --model tft \
    --n-trials 50 \
    --timeout-hours 4 \
    --output-dir training_outputs/optuna &
```

### Search Spaces

**XGBoost**:
| Parameter | Range | Scale |
|-----------|-------|-------|
| max_depth | 3-10 | int |
| learning_rate | 0.01-0.3 | log |
| min_child_weight | 1-10 | int |
| subsample | 0.6-1.0 | linear |
| colsample_bytree | 0.6-1.0 | linear |
| gamma | 0.0-5.0 | linear |
| reg_alpha | 1e-8 - 10.0 | log |
| reg_lambda | 1e-8 - 10.0 | log |

**TFT-Lite**:
| Parameter | Options/Range | Scale |
|-----------|---------------|-------|
| hidden_size | [32, 48, 64, 96, 128] | categorical |
| n_heads | [2, 4, 8] | categorical |
| n_encoder_layers | 1-4 | int |
| learning_rate | 1e-5 - 1e-2 | log |
| weight_decay | 1e-6 - 1e-2 | log |
| dropout | 0.0-0.3 | linear |
| batch_size | [256, 512, 768, 1024] | categorical |

### Final Training with Best Hyperparameters

```bash
# XGBoost with optimized config
python -m training.scripts.train_xgboost \
    --config training_outputs/optuna/xgboost_best.yaml \
    --data-dir data/physionet \
    --n-folds 5 \
    --save-final

# TFT with optimized config
python -m training.scripts.train_tft \
    --config training_outputs/optuna/tft_best.yaml \
    --data-dir data/physionet \
    --n-folds 5 \
    --save-final
```

---

## Expected Results

### Training Times (RTX 4090)

| Model | Patients | Folds | Time |
|-------|----------|-------|------|
| XGBoost-TS | 1,000 | 3 | ~5 min |
| XGBoost-TS | 40,000 | 5 | ~2 hours |
| TFT-Lite | 1,000 | 3 | ~10 min |
| TFT-Lite | 40,000 | 5 | ~4 hours |
| Optuna (XGBoost) | 100 trials | - | ~4 hours |
| Optuna (TFT) | 50 trials | - | ~4 hours |

### Expected Performance

Based on PhysioNet Challenge 2019 benchmarks:

| Model | AUC-ROC | Utility | Lead Time |
|-------|---------|---------|-----------|
| qSOFA | 0.70-0.75 | 0.30-0.35 | 4-5 hrs |
| XGBoost-TS | 0.82-0.88 | 0.38-0.42 | 5-6 hrs |
| TFT-Lite | 0.85-0.90 | 0.40-0.45 | 5-7 hrs |

### Output Files

After training:
```
models/xgboost_ts/weights/
├── xgb_sepsis_v1.json          # ~5MB trained model
└── xgb_sepsis_v1.features.json # Feature names

models/tft_lite/weights/
├── tft_lite_v1.pt              # ~2MB trained model
└── tft_lite_v1.config.json     # Model config

training_outputs/
├── xgboost/
│   ├── checkpoints/            # Per-fold checkpoints
│   ├── metrics.json            # Training metrics
│   └── config.json             # Training config
└── tft/
    ├── checkpoints/
    ├── metrics.json
    └── config.json
```

---

## Troubleshooting

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size: `--batch-size 512`
2. Enable gradient checkpointing (for TFT)
3. Use CPU for XGBoost: `--device cpu`

### PhysioNet Download Fails

```
HTTP Error 403: Forbidden
```

**Solutions**:
1. Check internet connection
2. Try alternative download:
   ```bash
   wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/
   ```

### XGBoost CUDA Errors

```
XGBoostError: No CUDA device detected
```

**Solutions**:
1. Verify CUDA installation: `nvidia-smi`
2. Use CPU: `--device cpu`
3. Reinstall XGBoost with CUDA: `pip install xgboost --upgrade`

### Low Validation Scores

**Possible causes**:
1. Too few patients: Use at least 1000 for meaningful results
2. Wrong metric: Ensure using `primary_metric="utility"`
3. Data leakage: Verify patient-level CV is working

---

## References

- [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/)
- [Sepsis-3 Consensus](https://jamanetwork.com/journals/jama/fullarticle/2526284)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)

---

*Last updated: January 2026*
