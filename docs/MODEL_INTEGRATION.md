# Model Integration Guide

This document explains how the trained sepsis prediction models are integrated into the SepsisPulse application.

## Overview

SepsisPulse uses three models for sepsis prediction:

| Model | Type | Status | Weights Location |
|-------|------|--------|------------------|
| qSOFA | Rule-based | Always available | N/A (no weights) |
| XGBoost-TS | ML (Gradient Boosting) | Trained | `models/xgboost_ts/weights/xgb_sepsis_v1.json` |
| TFT-Lite | Deep Learning (Transformer) | Trained | `models/tft_lite/weights/tft_lite_v1.pt` |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         app.py (Streamlit)                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              get_model_service() [cached]                │   │
│   │                         │                                │   │
│   │                         ▼                                │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │           ModelService (singleton)              │   │   │
│   │   │  ┌──────────┬──────────────┬──────────────┐    │   │   │
│   │   │  │  qSOFA   │  XGBoost-TS  │  TFT-Lite    │    │   │   │
│   │   │  │ (rules)  │  (trained)   │  (trained)   │    │   │   │
│   │   │  └──────────┴──────────────┴──────────────┘    │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Integration Components

### 1. Model Service (`src/services/model_service.py`)

The `ModelService` class provides a unified interface for all models:

```python
from src.services.model_service import ModelService, ModelStatus

# Initialize service (singleton pattern via Streamlit caching)
service = ModelService()
service.load_all_models()

# Check model status
status = service.get_status_summary()
# Returns: {
#     "total_models": 3,
#     "trained_count": 3,
#     "demo_count": 0,
#     "all_trained": True,
#     "models": [...]
# }

# Get specific model
xgb_model = service.get_model("XGBoost-TS")

# Check if model has trained weights
if service.is_model_trained("XGBoost-TS"):
    print("Using trained model")
```

### 2. Model Status Tracking

Each model has a status indicating its operational state:

| Status | Meaning |
|--------|---------|
| `trained` | Loaded with trained weights, ready for production use |
| `demo` | Running without trained weights, predictions are random/untrained |
| `error` | Failed to load, check error message |
| `not_loaded` | Not yet initialized |

### 3. Prediction Service (`src/services/prediction_service.py`)

For running full prediction pipelines:

```python
from src.services import ModelService, PredictionService

model_service = ModelService()
model_service.load_all_models()

pred_service = PredictionService(model_service)

# Single patient prediction
patient_data = load_patient("data/sample/p00001.psv")
result = pred_service.predict_patient(patient_data, "p00001")
# Returns: PatientPrediction(
#     patient_id="p00001",
#     has_sepsis=False,
#     qsofa_score=1,
#     xgboost_probability=0.23,
#     tft_probability=0.31,
#     ...
# )

# Dataset evaluation
dataset = load_dataset("data/sample/")
predictions, metrics = pred_service.predict_dataset(dataset)
```

## UI Integration

### Status Indicators

The app displays model status in several places:

1. **Sidebar**: Shows "Trained Models Active" or "Demo Mode"
2. **Dashboard Banner**: Green banner for trained models, yellow for demo
3. **Model Comparison**: Indicates data source (trained vs demo)

### Dynamic Metrics

Metrics displayed in the UI are dynamically sourced:

```python
def get_model_metrics():
    """
    Returns (metrics_dict, is_using_trained_models)
    - If trained models: Returns actual training performance
    - If demo mode: Returns hardcoded demo values
    """
```

## Training Performance

### XGBoost-TS Results (5-Fold CV on 40,311 patients)

| Metric | Mean | Std |
|--------|------|-----|
| AUROC | 0.8138 | 0.0058 |
| Clinical Utility | 0.7017 | 0.0022 |
| Sensitivity | 56.9% | 0.87% |
| Specificity | 84.7% | 0.74% |

**Top Features by Importance:**
1. FiO2 (Fraction of inspired oxygen)
2. Alkalinephos_lag3 (Liver enzyme, 3h lag)
3. AST_lag1 (Liver enzyme, 1h lag)
4. Lactate_lag6 (Key sepsis marker, 6h lag)
5. Lactate_mean12 (12h rolling mean)

### TFT-Lite Results (Partial Training)

| Metric | Value |
|--------|-------|
| Parameters | 149,645 |
| Architecture | 39 input, 64 hidden, 2 LSTM, 4 attention heads |
| Estimated AUROC | ~0.82 |
| Estimated Utility | ~0.68 |

## Weight Files

### XGBoost-TS
```
models/xgboost_ts/weights/
├── xgb_sepsis_v1.json          # Model weights (JSON format)
├── xgb_sepsis_v1.features.json # Feature names (429 features)
└── xgb_sepsis_v1.meta.json     # Training metadata
```

### TFT-Lite
```
models/tft_lite/weights/
└── tft_lite_v1.pt              # PyTorch state dict
```

## Feature Engineering

### XGBoost-TS Features (429 total)

The XGBoost model uses engineered time-series features:

```
Raw Features (39):
- 8 vital signs: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2
- 26 lab values: BaseExcess, HCO3, FiO2, pH, ...
- 5 demographics: Age, Gender, Unit1, Unit2, HospAdmTime

Engineered Features:
- Lag features: _lag1, _lag3, _lag6 (117 features)
- Rolling mean: _mean3, _mean6, _mean12 (117 features)
- Rolling std: _std3, _std6, _std12 (117 features)
- Delta/change features (39 features)
```

### TFT-Lite Input

The TFT model takes raw sequence data:
- Input shape: `(batch, seq_len, 39)`
- Max sequence length: 72 hours
- Features: Forward-filled, normalized

## Fallback Behavior

When trained weights are not available:

1. **XGBoost-TS**: Falls back to demo mode with random predictions
2. **TFT-Lite**: Falls back to demo mode with untrained network
3. **qSOFA**: Always available (rule-based, no weights needed)

The UI clearly indicates when running in demo mode vs with trained models.

## Adding New Models

To add a new model to the integration:

1. Create model wrapper in `models/new_model/`
2. Add loading logic to `ModelService._load_new_model()`
3. Add entry to `ModelService.WEIGHT_PATHS`
4. Update UI in `app.py` to display new model
5. Update `get_model_metrics()` to include new model metrics

## Troubleshooting

### Models Loading in Demo Mode

Check that weight files exist:
```bash
ls -la models/xgboost_ts/weights/xgb_sepsis_v1.json
ls -la models/tft_lite/weights/tft_lite_v1.pt
```

### XGBoost JSON Parse Error

Ensure model was saved in JSON format (not binary):
```python
# Correct
model.save_model("model.json", "json")

# Incorrect (saves binary despite .json extension)
model.save_model("model.json")
```

### TFT State Dict Mismatch

Ensure model architecture matches saved weights:
```python
# Check saved config
import torch
state = torch.load("tft_lite_v1.pt")
print(state.keys())  # Should include model layers
```

## References

- [Model Training Guide](TRAINING.md)
- [Architecture Overview](ARCHITECTURE.md)
- [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/)
