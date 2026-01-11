# SepsisPulse - System Architecture

## System Overview

SepsisPulse is a clinical utility auditor for sepsis prediction models. The system ingests patient time-series data, processes it through three distinct prediction models, and evaluates their performance using clinical utility metrics.

### High-Level Architecture Diagram

```
+------------------------------------------------------------------+
|                        SEPSISPULSE SYSTEM                        |
+------------------------------------------------------------------+
|                                                                  |
|  +------------------+     +------------------+     +----------+  |
|  |   DATA LAYER    |     |   MODEL LAYER    |     | UI LAYER |  |
|  +------------------+     +------------------+     +----------+  |
|  |                  |     |                  |     |          |  |
|  | PSV Files        |---->| qSOFA            |---->| Streamlit|  |
|  | (PhysioNet 2019) |     | (Heuristic)      |     | Dashboard|  |
|  |                  |     |                  |     |          |  |
|  | Preprocessor     |---->| XGBoost-TS       |---->| Charts   |  |
|  | (Imputation)     |     | (Gradient Boost) |     | (Plotly) |  |
|  |                  |     |                  |     |          |  |
|  | Feature Eng.     |---->| TFT-Lite         |---->| Metrics  |  |
|  | (Lags, Rolling)  |     | (Deep Learning)  |     | Cards    |  |
|  |                  |     |                  |     |          |  |
|  +------------------+     +------------------+     +----------+  |
|           |                       |                     |        |
|           v                       v                     v        |
|  +----------------------------------------------------------+   |
|  |                    EVALUATION LAYER                       |   |
|  +----------------------------------------------------------+   |
|  |  Clinical Utility Score  |  Lead Time  |  ROC/PR Curves  |   |
|  +----------------------------------------------------------+   |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Data Flow

### Complete Pipeline

```
                    DATA FLOW DIAGRAM

    +-------------+
    | PSV Files   |  PhysioNet 2019 format (pipe-delimited)
    | (.psv)      |  41 columns, 1 row per hour
    +------+------+
           |
           v
    +------+------+
    | Data Loader |  load_patient(), load_dataset()
    | loader.py   |  Validates columns, handles missing
    +------+------+
           |
           v
    +------+------+
    | Preprocessor|  Forward fill, backward fill, mean imputation
    | preproc.py  |  Z-score normalization (optional)
    +------+------+
           |
           +------------------+------------------+
           |                  |                  |
           v                  v                  v
    +------+------+    +------+------+    +------+------+
    |   qSOFA     |    | Feature Eng |    | Direct to   |
    | (raw vitals)|    | (for XGB)   |    | TFT-Lite    |
    +------+------+    +------+------+    +------+------+
           |                  |                  |
           v                  v                  v
    +------+------+    +------+------+    +------+------+
    | qSOFA Score |    | XGBoost-TS  |    | TFT-Lite    |
    | Threshold   |    | predict()   |    | predict()   |
    +------+------+    +------+------+    +------+------+
           |                  |                  |
           +------------------+------------------+
                              |
                              v
                    +---------+---------+
                    |    Predictions    |
                    | Dict[patient_id,  |
                    |      np.ndarray]  |
                    +---------+---------+
                              |
           +------------------+------------------+
           |                  |                  |
           v                  v                  v
    +------+------+    +------+------+    +------+------+
    | Clinical    |    | Lead Time   |    | ROC/PR     |
    | Utility     |    | Analysis    |    | Metrics    |
    +------+------+    +------+------+    +------+------+
           |                  |                  |
           +------------------+------------------+
                              |
                              v
                    +---------+---------+
                    |   Dashboard UI    |
                    |   (Streamlit)     |
                    +-------------------+
```

---

## Component Descriptions

### Data Layer

#### `src/data/loader.py`

**Purpose**: Load and validate PhysioNet 2019 PSV files.

**Key Functions**:
- `load_patient(file_path)`: Load single patient file
- `load_dataset(data_dir, max_patients)`: Load multiple patients
- `get_sample_subset()`: Load bundled sample data
- `get_column_groups()`: Get column categories (vitals, labs, demographics)

**Design Decisions**:
- Returns Dict[patient_id, DataFrame] for easy patient-level processing
- Validates all 41 expected columns on load
- Preserves NaN values for downstream imputation

#### `src/data/preprocessor.py`

**Purpose**: Handle missing values and normalize features.

**Imputation Strategy** (clinically appropriate):
1. **Forward fill**: Carry last observation forward (lab values remain valid until re-measured)
2. **Backward fill**: Fill initial NaNs with first available measurement
3. **Mean imputation**: Use population means for entirely missing columns

**Key Functions**:
- `handle_missing_values(df)`: Apply imputation pipeline
- `normalize_features(df)`: Z-score normalization
- `preprocess_patient(df, normalize=True)`: Complete pipeline

**Constants**:
- `POPULATION_MEANS`: Clinical reference values for 34 features
- `POPULATION_STDS`: Standard deviations for z-score normalization

#### `src/data/feature_engineering.py`

**Purpose**: Generate engineered features for XGBoost-TS model.

**Feature Types**:

| Type | Example | Count per Column |
|------|---------|------------------|
| Lag | HR_lag_1, HR_lag_3, HR_lag_6 | 3 |
| Rolling Mean | HR_roll_mean_3, HR_roll_mean_6, HR_roll_mean_12 | 3 |
| Rolling Std | HR_roll_std_3, HR_roll_std_6, HR_roll_std_12 | 3 |
| Rolling Min | HR_roll_min_3, HR_roll_min_6, HR_roll_min_12 | 3 |
| Rolling Max | HR_roll_max_3, HR_roll_max_6, HR_roll_max_12 | 3 |
| Delta | HR_delta | 1 |

**Total Features**: ~200 from 41 raw variables (17x expansion)

---

### Model Layer

#### qSOFA Model (`models/qsofa/qsofa_model.py`)

**Type**: Rule-based heuristic (no training required)

**Architecture**:
```
Input: Patient vitals (Resp, SBP, GCS)
       |
       v
+------+------+
| Criteria    |
| Evaluation  |
+------+------+
| - Resp >= 22 --> +1 |
| - SBP <= 100 --> +1 |
| - GCS < 15   --> +1 |
+------+------+
       |
       v
Score (0-3) --> Threshold (default: 2) --> Binary Prediction
```

**Key Methods**:
- `calculate_score(df)`: Get qSOFA score (0-3)
- `predict(df)`: Binary predictions
- `predict_proba(df)`: Probability estimates (score/3)
- `get_criteria_breakdown(df)`: Individual criterion values

**Strengths**: Simple, interpretable, no training, works at bedside
**Weaknesses**: Low sensitivity, triggers late, contributes to alarm fatigue

---

#### XGBoost-TS Model (`models/xgboost_ts/xgboost_model.py`)

**Type**: Gradient Boosted Decision Trees with time-series features

**Architecture**:
```
Input: Patient DataFrame (41 columns)
       |
       v
+------+------+
| Feature     |
| Engineering |
+------+------+
       |
       v (~200 features)
+------+------+
| XGBoost     |
| Booster     |
+------+------+
| - 500 trees          |
| - max_depth: 6       |
| - learning_rate: 0.05|
| - reg_lambda: 1.0    |
+------+------+
       |
       v
Probability (0-1) --> Threshold --> Binary Prediction
```

**Key Methods**:
- `load_weights(path)`: Load pre-trained JSON weights
- `save_weights(path)`: Save model weights
- `predict(X, threshold)`: Binary predictions
- `predict_proba(X)`: Probability matrix [P(0), P(1)]
- `get_feature_importance()`: Ranked feature importance

**Demo Mode**: When weights unavailable, generates reproducible pseudo-random predictions.

---

#### TFT-Lite Model (`models/tft_lite/`)

**Type**: Lightweight Temporal Fusion Transformer (Deep Learning)

**Architecture**:
```
Input: Tensor (batch, seq_len=24, features=41)
       |
       v
+------+------+
| Variable    |
| Selection   |
| Network     |
+------+------+
| - Feature embedding      |
| - GRN-based gating       |
| - Softmax weights        |
+------+------+
       |
       v (batch, seq_len, hidden=32)
+------+------+
| LSTM        |
| Encoder     |
+------+------+
| - 1 layer              |
| - hidden_size: 32      |
| - bidirectional: False |
+------+------+
       |
       v
+------+------+
| Multi-Head  |
| Attention   |
+------+------+
| - 2 attention heads    |
| - Self-attention       |
| - Returns weights      |
+------+------+
       |
       v
+------+------+
| Output GRN  |
| + Linear    |
+------+------+
       |
       v
Logit --> Sigmoid --> Probability (0-1)
```

**TFT-Lite vs Original TFT**:

| Parameter | TFT-Lite | Original TFT |
|-----------|----------|--------------|
| Hidden Size | 32 | 256 |
| LSTM Layers | 1 | 2 |
| Attention Heads | 2 | 4 |
| Max Sequence | 24h | 168h |
| Parameters | ~500K | ~10M |
| Memory | ~500MB | ~4GB |

**Key Components**:
- `GatedLinearUnit`: Feature gating mechanism
- `GatedResidualNetwork`: Non-linear processing with skip connections
- `VariableSelectionNetwork`: Learned feature importance
- `InterpretableMultiHeadAttention`: Attention with interpretable weights

**Key Methods**:
- `predict(X, threshold)`: Binary predictions
- `predict_proba(X)`: Probability estimates
- `get_attention_weights(X)`: Temporal attention weights
- `get_variable_importance(X)`: Feature importance weights

---

### Evaluation Layer

#### `src/evaluation/clinical_utility.py`

**Purpose**: Implement PhysioNet 2019 Clinical Utility Score.

**Utility Function**:
```python
def utility_function(t_pred, t_sepsis, has_sepsis):
    if has_sepsis:
        time_diff = t_sepsis - t_pred
        if 6 <= time_diff <= 12:    # Optimal window
            return 1.0
        elif 0 <= time_diff < 6:    # Late but acceptable
            return 0.5
        elif time_diff > 12:         # Too early
            return -0.05
        else:                        # Missed (after onset)
            return -2.0
    else:
        return -0.05  # False positive
```

**Normalization**:
```
normalized_score = (U - U_worst) / (U_best - U_worst)
```

#### `src/evaluation/lead_time.py`

**Purpose**: Measure prediction lead time before sepsis onset.

**Key Functions**:
- `compute_lead_time(predictions, t_sepsis)`: Single patient lead time
- `compute_average_lead_time(predictions, t_sepsis)`: Aggregate statistics
- `compute_lead_time_distribution(predictions, t_sepsis)`: For histogram plots
- `get_detection_rate_by_lead_time(predictions, t_sepsis, time_points)`: Detection rates

#### `src/evaluation/metrics.py`

**Purpose**: Standard ML evaluation metrics.

**Key Functions**:
- `compute_roc_auc(y_true, y_score)`: ROC curve and AUC
- `compute_pr_auc(y_true, y_score)`: Precision-Recall curve and AUC
- `compute_confusion_matrix(y_true, y_pred)`: TP, TN, FP, FN
- `compute_classification_metrics(y_true, y_pred)`: Sensitivity, specificity, PPV, NPV, F1

---

### UI Layer

#### `app.py`

**Purpose**: Streamlit dashboard entry point.

**Pages**:
1. **Dashboard**: Key metrics, model comparison cards, performance charts
2. **Model Comparison**: Detailed metrics table, model descriptions
3. **Patient Explorer**: Individual patient vitals and predictions
4. **Configuration**: Threshold settings, data source selection
5. **Documentation**: Embedded user guide

#### `src/visualization/theme.py`

**Purpose**: Aurora Solar-inspired dark theme.

**Color Palette**:
```python
COLORS = {
    "background": "#0d1117",
    "card_bg": "#161b22",
    "card_border": "#30363d",
    "primary": "#58a6ff",
    "secondary": "#8b949e",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f85149",
}
```

#### `src/visualization/charts.py`

**Purpose**: Plotly chart generators.

**Chart Types**:
- `create_roc_chart(results)`: Multi-model ROC comparison
- `create_lead_time_chart(lead_times)`: Lead time histogram
- `create_utility_chart(utilities)`: Utility score bar chart
- `create_patient_timeline(vitals, predictions)`: Vitals + predictions subplot

#### `src/visualization/components.py`

**Purpose**: Reusable UI components.

**Components**:
- `metric_card(title, value, delta)`: Styled metric display
- `alert_banner(message, severity)`: Alert/notification banner
- `model_comparison_table(metrics)`: Formatted comparison DataFrame

---

## Directory Structure

```
SepsisPulse/
+-- app.py                      # Streamlit entry point
+-- .streamlit/
|   +-- config.toml             # Streamlit configuration (theme, server)
+-- src/
|   +-- __init__.py
|   +-- data/
|   |   +-- __init__.py
|   |   +-- loader.py           # PSV file loading
|   |   +-- preprocessor.py     # Missing value handling, normalization
|   |   +-- feature_engineering.py  # Lag, rolling, delta features
|   +-- evaluation/
|   |   +-- __init__.py
|   |   +-- clinical_utility.py # PhysioNet utility score
|   |   +-- lead_time.py        # Prediction timing metrics
|   |   +-- metrics.py          # ROC, PR, classification metrics
|   +-- visualization/
|   |   +-- __init__.py
|   |   +-- theme.py            # Aurora color palette
|   |   +-- charts.py           # Plotly chart generators
|   |   +-- components.py       # UI components
|   +-- utils/
|       +-- __init__.py
|       +-- config.py           # Application configuration
|       +-- cache.py            # Streamlit caching utilities
+-- models/
|   +-- __init__.py             # Model exports
|   +-- qsofa/
|   |   +-- __init__.py
|   |   +-- qsofa_model.py      # Rule-based qSOFA
|   +-- xgboost_ts/
|   |   +-- __init__.py
|   |   +-- xgboost_model.py    # XGBoost wrapper
|   |   +-- weights/            # Pre-trained weights (JSON)
|   +-- tft_lite/
|       +-- __init__.py
|       +-- architecture.py     # TFT-Lite neural network
|       +-- tft_model.py        # High-level wrapper
|       +-- weights/            # Pre-trained weights (PyTorch)
+-- data/
|   +-- sample/
|       +-- patients/           # Synthetic PSV files
|       +-- metadata.json       # Dataset summary
|       +-- README.md           # Sample data documentation
+-- tests/
|   +-- __init__.py
+-- docs/
    +-- PRD.md
    +-- ARCHITECTURE.md
    +-- API.md
    +-- USER_GUIDE.md
    +-- DESIGN_DECISIONS.md
    +-- DATA_PREPARATION.md
```

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| Web Framework | Streamlit | 1.28+ | Interactive dashboard |
| Deep Learning | PyTorch | 2.0+ | TFT-Lite model |
| ML | XGBoost | 2.0+ | Gradient boosting model |
| ML Utilities | scikit-learn | 1.3+ | Metrics, preprocessing |
| Data | pandas | 2.0+ | DataFrame operations |
| Numerical | numpy | 1.24+ | Array operations |
| Visualization | Plotly | 5.17+ | Interactive charts |
| Deployment | HuggingFace Spaces | - | Free hosting |

---

## Deployment Architecture

### HuggingFace Spaces (Production)

```
+-------------------+
| User Browser      |
+---------+---------+
          |
          | HTTPS
          v
+---------+---------+
| HuggingFace       |
| Spaces            |
| (CPU, 2GB RAM)    |
+---------+---------+
          |
          | Streamlit
          v
+---------+---------+
| SepsisPulse       |
| Container         |
| - app.py          |
| - models/         |
| - data/sample/    |
+-------------------+
```

### Local Development

```bash
# Clone repository
git clone https://github.com/user/SepsisPulse.git
cd SepsisPulse

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

---

## Memory Optimization Strategies

To fit within the 2GB RAM constraint:

1. **TFT-Lite Architecture**: Reduced from 10M to 500K parameters
2. **Lazy Model Loading**: Models loaded on-demand via `@st.cache_resource`
3. **Data Streaming**: Process patients one at a time, don't load all into memory
4. **Feature Engineering**: Generate features per-patient, not dataset-wide
5. **Quantization Ready**: TFT-Lite can be quantized to INT8 if needed
