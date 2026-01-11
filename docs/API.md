# SepsisPulse - API Documentation

This document provides detailed API documentation for all public functions and classes in SepsisPulse.

---

## Table of Contents

1. [Data Loading Module](#data-loading-module)
2. [Preprocessing Module](#preprocessing-module)
3. [Feature Engineering Module](#feature-engineering-module)
4. [Models](#models)
   - [qSOFA Model](#qsofa-model)
   - [XGBoost-TS Model](#xgboost-ts-model)
   - [TFT-Lite Model](#tft-lite-model)
5. [Evaluation Module](#evaluation-module)
   - [Clinical Utility](#clinical-utility)
   - [Lead Time](#lead-time)
   - [Metrics](#metrics)
6. [Visualization Module](#visualization-module)
7. [Utilities](#utilities)

---

## Data Loading Module

**Location**: `src/data/loader.py`

### Constants

```python
VITAL_COLUMNS: List[str]
# ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']

LAB_COLUMNS: List[str]
# 26 laboratory value column names

DEMOGRAPHIC_COLUMNS: List[str]
# ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

LABEL_COLUMN: List[str]
# ['SepsisLabel']

PHYSIONET_COLUMNS: List[str]
# All 41 columns in order
```

### load_patient

```python
def load_patient(file_path: str) -> pd.DataFrame
```

Load a single patient's PSV file into a DataFrame.

**Parameters**:
- `file_path` (str): Path to the .psv file (pipe-delimited)

**Returns**:
- `pd.DataFrame`: Patient data with 41 columns. Each row is one hour of ICU stay.

**Raises**:
- `FileNotFoundError`: If the file does not exist
- `ValueError`: If the file format is invalid

**Example**:
```python
from src.data.loader import load_patient

df = load_patient("data/sample/patients/p00001.psv")
print(df.shape)  # (62, 41)
print(df.columns.tolist()[:5])  # ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP']
```

---

### load_dataset

```python
def load_dataset(
    data_dir: str,
    max_patients: Optional[int] = None
) -> Dict[str, pd.DataFrame]
```

Load multiple patient PSV files from a directory.

**Parameters**:
- `data_dir` (str): Directory containing .psv files
- `max_patients` (Optional[int]): Maximum patients to load. None loads all.

**Returns**:
- `Dict[str, pd.DataFrame]`: Maps patient IDs to DataFrames

**Raises**:
- `FileNotFoundError`: If directory does not exist
- `ValueError`: If no .psv files found

**Example**:
```python
from src.data.loader import load_dataset

# Load first 100 patients
dataset = load_dataset("data/training_setA", max_patients=100)
print(len(dataset))  # 100
print(list(dataset.keys())[:3])  # ['p00001', 'p00002', 'p00003']
```

---

### get_sample_subset

```python
def get_sample_subset() -> Dict[str, pd.DataFrame]
```

Load bundled sample data for testing and development.

**Returns**:
- `Dict[str, pd.DataFrame]`: Maps patient IDs to DataFrames. Empty dict if not found.

**Example**:
```python
from src.data.loader import get_sample_subset

sample = get_sample_subset()
print(len(sample))  # 20
```

---

### get_patient_ids

```python
def get_patient_ids(data_dir: str) -> List[str]
```

Get list of patient IDs without loading data.

**Parameters**:
- `data_dir` (str): Directory containing .psv files

**Returns**:
- `List[str]`: Sorted list of patient IDs

**Example**:
```python
from src.data.loader import get_patient_ids

ids = get_patient_ids("data/training_setA")
print(ids[:3])  # ['p00001', 'p00002', 'p00003']
```

---

### get_column_groups

```python
def get_column_groups() -> Dict[str, List[str]]
```

Get column names grouped by category.

**Returns**:
- `Dict[str, List[str]]`: Keys are 'vitals', 'labs', 'demographics', 'label'

**Example**:
```python
from src.data.loader import get_column_groups

groups = get_column_groups()
print(groups['vitals'])  # ['HR', 'O2Sat', 'Temp', ...]
```

---

## Preprocessing Module

**Location**: `src/data/preprocessor.py`

### handle_missing_values

```python
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame
```

Handle missing values using clinically appropriate imputation.

**Strategy**:
1. Forward fill (carry last observation forward)
2. Backward fill (for initial NaNs)
3. Mean imputation (for entirely missing columns)

**Parameters**:
- `df` (pd.DataFrame): Input DataFrame with clinical features

**Returns**:
- `pd.DataFrame`: DataFrame with all missing values imputed

**Example**:
```python
from src.data.preprocessor import handle_missing_values

# Before: df has NaN values
print(df.isna().sum().sum())  # 342

# After: no NaN values
df_clean = handle_missing_values(df)
print(df_clean.isna().sum().sum())  # 0
```

---

### normalize_features

```python
def normalize_features(df: pd.DataFrame) -> pd.DataFrame
```

Apply z-score normalization using population statistics.

**Formula**: `z = (x - mean) / std`

**Parameters**:
- `df` (pd.DataFrame): Input DataFrame (missing values should be handled first)

**Returns**:
- `pd.DataFrame`: DataFrame with normalized features

**Example**:
```python
from src.data.preprocessor import normalize_features

df_normalized = normalize_features(df_clean)
print(df_normalized['HR'].mean())  # ~0.0
print(df_normalized['HR'].std())   # ~1.0
```

---

### preprocess_patient

```python
def preprocess_patient(
    df: pd.DataFrame,
    normalize: bool = True
) -> pd.DataFrame
```

Full preprocessing pipeline for a single patient.

**Parameters**:
- `df` (pd.DataFrame): Raw patient data
- `normalize` (bool): Whether to apply z-score normalization. Default True.

**Returns**:
- `pd.DataFrame`: Preprocessed patient data

**Example**:
```python
from src.data.loader import load_patient
from src.data.preprocessor import preprocess_patient

raw_df = load_patient("data/sample/patients/p00001.psv")
processed = preprocess_patient(raw_df, normalize=True)
```

---

### get_missing_rate

```python
def get_missing_rate(df: pd.DataFrame) -> pd.Series
```

Calculate missing rate for each column.

**Returns**:
- `pd.Series`: Missing rates (0-1) indexed by column name

**Example**:
```python
from src.data.preprocessor import get_missing_rate

missing = get_missing_rate(df)
print(missing.sort_values(ascending=False).head())
# TroponinI     0.98
# Fibrinogen    0.97
# ...
```

---

## Feature Engineering Module

**Location**: `src/data/feature_engineering.py`

### create_lag_features

```python
def create_lag_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 3, 6],
    columns: Optional[List[str]] = None
) -> pd.DataFrame
```

Create lagged versions of each numeric column.

**Parameters**:
- `df` (pd.DataFrame): Input time series data
- `lags` (List[int]): Lag periods to create. Default [1, 3, 6].
- `columns` (Optional[List[str]]): Columns to process. None uses all numeric.

**Returns**:
- `pd.DataFrame`: Original + lagged features (e.g., HR_lag_1, HR_lag_3)

**Example**:
```python
from src.data.feature_engineering import create_lag_features

df_lags = create_lag_features(df, lags=[1, 2])
print([c for c in df_lags.columns if 'lag' in c][:4])
# ['HR_lag_1', 'HR_lag_2', 'O2Sat_lag_1', 'O2Sat_lag_2']
```

---

### create_rolling_features

```python
def create_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [3, 6, 12],
    columns: Optional[List[str]] = None,
    statistics: Optional[List[str]] = None
) -> pd.DataFrame
```

Create rolling window statistics.

**Parameters**:
- `df` (pd.DataFrame): Input time series data
- `windows` (List[int]): Window sizes. Default [3, 6, 12].
- `columns` (Optional[List[str]]): Columns to process
- `statistics` (Optional[List[str]]): Stats to compute. Default ['mean', 'std', 'min', 'max'].

**Returns**:
- `pd.DataFrame`: Original + rolling features (e.g., HR_roll_mean_3)

**Example**:
```python
from src.data.feature_engineering import create_rolling_features

df_rolling = create_rolling_features(df, windows=[3], statistics=['mean'])
print([c for c in df_rolling.columns if 'roll' in c][:3])
# ['HR_roll_mean_3', 'O2Sat_roll_mean_3', 'Temp_roll_mean_3']
```

---

### create_rate_of_change

```python
def create_rate_of_change(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    periods: int = 1
) -> pd.DataFrame
```

Create first difference (delta) features.

**Parameters**:
- `df` (pd.DataFrame): Input time series data
- `columns` (Optional[List[str]]): Columns to process
- `periods` (int): Difference periods. Default 1.

**Returns**:
- `pd.DataFrame`: Original + delta features (e.g., HR_delta)

**Example**:
```python
from src.data.feature_engineering import create_rate_of_change

df_delta = create_rate_of_change(df)
print(df_delta['HR_delta'].values[:4])
# [nan, 2.0, -1.0, 3.0]
```

---

### create_all_features

```python
def create_all_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 3, 6],
    windows: List[int] = [3, 6, 12],
    columns: Optional[List[str]] = None,
    include_lag: bool = True,
    include_rolling: bool = True,
    include_delta: bool = True
) -> pd.DataFrame
```

Create all engineered features.

**Returns**:
- `pd.DataFrame`: Original + all engineered features (~200 features from 41 raw)

**Example**:
```python
from src.data.feature_engineering import create_all_features

df_features = create_all_features(df)
print(len(df_features.columns))  # ~200+
```

---

## Models

### qSOFA Model

**Location**: `models/qsofa/qsofa_model.py`

```python
class QSOFAModel:
    """Rule-based qSOFA scoring for sepsis prediction."""

    def __init__(
        self,
        threshold: int = 2,
        resp_rate_threshold: float = 22.0,
        sbp_threshold: float = 100.0,
        gcs_threshold: float = 15.0
    )
```

**Parameters**:
- `threshold` (int): Score threshold for positive prediction (0-3). Default 2.
- `resp_rate_threshold` (float): Respiratory rate threshold. Default 22.
- `sbp_threshold` (float): Systolic BP threshold. Default 100.
- `gcs_threshold` (float): GCS threshold. Default 15.

#### Methods

```python
def calculate_score(self, df: pd.DataFrame) -> np.ndarray
```
Calculate qSOFA score (0-3) for each row.

```python
def predict(self, df: pd.DataFrame) -> np.ndarray
```
Binary predictions (0 or 1) based on threshold.

```python
def predict_proba(self, df: pd.DataFrame) -> np.ndarray
```
Probability estimates. Returns (n_samples, 2) array with [P(0), P(1)].

```python
def get_criteria_breakdown(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```
Get individual criterion values for interpretability.

```python
def get_feature_importance(self) -> dict
```
Returns equal importance (1/3) for each criterion.

**Example**:
```python
from models import QSOFAModel

model = QSOFAModel(threshold=2)

# Calculate scores
scores = model.calculate_score(patient_df)
print(scores)  # [0, 1, 1, 2, 2, 3, ...]

# Get predictions
predictions = model.predict(patient_df)
print(predictions)  # [0, 0, 0, 1, 1, 1, ...]

# Get probabilities
proba = model.predict_proba(patient_df)
print(proba[0])  # [0.667, 0.333]  # score=1 -> prob=1/3
```

---

### XGBoost-TS Model

**Location**: `models/xgboost_ts/xgboost_model.py`

```python
class XGBoostTSModel:
    """XGBoost model wrapper for sepsis prediction."""

    def __init__(self, weights_path: Optional[str] = None)
```

**Parameters**:
- `weights_path` (Optional[str]): Path to pre-trained weights. If None, tries default path. If not found, runs in demo mode.

**Attributes**:
- `model`: XGBoost Booster (None in demo mode)
- `is_demo_mode` (bool): Whether running without trained weights
- `weights_path` (str): Path to loaded weights

#### Methods

```python
def load_weights(self, path: str) -> None
```
Load model weights from JSON file.

```python
def save_weights(self, path: str) -> None
```
Save model weights to JSON file.

```python
def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray
```
Binary predictions.

```python
def predict_proba(self, X: pd.DataFrame) -> np.ndarray
```
Probability estimates. Returns (n_samples, 2) array.

```python
def get_feature_importance(self) -> pd.Series
```
Feature importance scores (gain-based).

```python
def get_model_info(self) -> Dict[str, Union[str, bool, int, None]]
```
Model status information.

**Example**:
```python
from models import XGBoostTSModel
from src.data.feature_engineering import create_all_features

model = XGBoostTSModel(weights_path="models/xgboost_ts/weights/xgb_v1.json")

# Check if weights loaded
print(model.is_demo_mode)  # False if weights loaded

# Create features
features = create_all_features(patient_df)

# Get predictions
proba = model.predict_proba(features)
predictions = model.predict(features, threshold=0.5)

# Get feature importance
importance = model.get_feature_importance()
print(importance.head())
```

---

### TFT-Lite Model

**Location**: `models/tft_lite/tft_model.py`

```python
class TFTLiteModel:
    """High-level wrapper for TFT-Lite model."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = 'cpu',
        input_size: int = 41,
        hidden_size: int = 32,
        lstm_layers: int = 1,
        attention_heads: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
        max_seq_length: int = 24
    )
```

**Parameters**:
- `weights_path` (Optional[str]): Path to pre-trained weights (.pt file)
- `device` (str): 'cpu' or 'cuda'
- `input_size` (int): Number of input features. Default 41.
- `hidden_size` (int): Hidden dimension. Default 32.
- `lstm_layers` (int): LSTM layers. Default 1.
- `attention_heads` (int): Attention heads. Default 2.

#### Methods

```python
def load_weights(self, path: str) -> None
```
Load pre-trained weights.

```python
def save_weights(self, path: str) -> None
```
Save model weights.

```python
def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray
```
Binary predictions. X shape: (batch, seq_len, features) or (seq_len, features).

```python
def predict_proba(self, X: np.ndarray) -> np.ndarray
```
Probability estimates.

```python
def get_attention_weights(self, X: np.ndarray) -> np.ndarray
```
Get attention weights for interpretability. Shape: (batch, heads, seq, seq).

```python
def get_variable_importance(self, X: np.ndarray) -> np.ndarray
```
Get variable selection weights. Shape: (batch, seq_len, input_size).

```python
def get_feature_importance_summary(
    self,
    X: np.ndarray,
    feature_names: Optional[list] = None
) -> dict
```
Aggregated feature importance across timesteps.

```python
def get_model_info(self) -> dict
```
Model configuration and status.

**Example**:
```python
from models import TFTLiteModel
import numpy as np

model = TFTLiteModel(weights_path="models/tft_lite/weights/tft_v1.pt")

# Prepare input: (batch, seq_len, features)
X = np.random.randn(1, 24, 41).astype(np.float32)

# Get predictions
proba = model.predict_proba(X)
print(proba)  # 0.42

# Get attention weights
attn = model.get_attention_weights(X)
print(attn.shape)  # (2, 24, 24) for single sample

# Get feature importance
importance = model.get_feature_importance_summary(
    X,
    feature_names=['HR', 'O2Sat', ...]  # 41 names
)
print(importance)  # {'HR': 0.15, 'O2Sat': 0.08, ...}
```

---

## Evaluation Module

### Clinical Utility

**Location**: `src/evaluation/clinical_utility.py`

#### utility_function

```python
def utility_function(
    t_pred: int,
    t_sepsis: int,
    has_sepsis: bool
) -> float
```

Compute utility for a single prediction.

**Parameters**:
- `t_pred` (int): Time step of prediction
- `t_sepsis` (int): Time step of sepsis onset
- `has_sepsis` (bool): Whether patient develops sepsis

**Returns**:
- `float`: Utility value (+1.0, +0.5, -0.05, or -2.0)

**Example**:
```python
from src.evaluation.clinical_utility import utility_function

# Optimal early (8 hours before)
print(utility_function(t_pred=42, t_sepsis=50, has_sepsis=True))  # 1.0

# Late (3 hours before)
print(utility_function(t_pred=47, t_sepsis=50, has_sepsis=True))  # 0.5

# False positive
print(utility_function(t_pred=20, t_sepsis=100, has_sepsis=False))  # -0.05
```

---

#### compute_utility_score

```python
def compute_utility_score(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    t_sepsis: Dict[str, Optional[int]]
) -> float
```

Compute normalized utility score across all patients.

**Parameters**:
- `predictions` (Dict[str, np.ndarray]): Patient ID -> binary predictions per hour
- `labels` (Dict[str, np.ndarray]): Patient ID -> true labels per hour
- `t_sepsis` (Dict[str, Optional[int]]): Patient ID -> sepsis onset hour (None if no sepsis)

**Returns**:
- `float`: Normalized score in [0, 1]

**Example**:
```python
from src.evaluation.clinical_utility import compute_utility_score
import numpy as np

predictions = {
    'p1': np.array([0, 0, 1, 1]),  # Predicts at t=2
    'p2': np.array([0, 0, 0, 0])   # No prediction
}
labels = {
    'p1': np.array([0, 0, 0, 1]),  # Sepsis at t=3
    'p2': np.array([0, 0, 0, 0])   # No sepsis
}
t_sepsis = {'p1': 3, 'p2': None}

score = compute_utility_score(predictions, labels, t_sepsis)
print(score)  # 0.xx
```

---

#### compute_utility_components

```python
def compute_utility_components(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    t_sepsis: Dict[str, Optional[int]]
) -> Dict[str, float]
```

Get detailed breakdown of utility components.

**Returns**:
- `Dict` with keys: 'total_utility', 'best_utility', 'worst_utility', 'normalized_score', 'num_sepsis_patients', 'num_non_sepsis_patients', 'optimal_early_count', 'optimal_late_count', 'too_early_count', 'missed_count', 'false_positive_count', 'true_negative_count'

---

### Lead Time

**Location**: `src/evaluation/lead_time.py`

#### compute_lead_time

```python
def compute_lead_time(
    predictions: np.ndarray,
    t_sepsis: int,
    threshold: float = 0.5
) -> Optional[float]
```

Compute lead time for a single patient.

**Parameters**:
- `predictions` (np.ndarray): Predictions (binary or probability)
- `t_sepsis` (int): Sepsis onset time
- `threshold` (float): Threshold for binary conversion. Default 0.5.

**Returns**:
- `Optional[float]`: Lead time in hours, or None if no positive prediction before onset

**Example**:
```python
from src.evaluation.lead_time import compute_lead_time
import numpy as np

predictions = np.array([0.2, 0.3, 0.6, 0.8, 0.9])
lead_time = compute_lead_time(predictions, t_sepsis=4, threshold=0.5)
print(lead_time)  # 2.0 (predicted at t=2, sepsis at t=4)
```

---

#### compute_average_lead_time

```python
def compute_average_lead_time(
    predictions: Dict[str, np.ndarray],
    t_sepsis: Dict[str, int],
    threshold: float = 0.5
) -> Dict[str, float]
```

Compute aggregate lead time statistics.

**Returns**:
- `Dict` with keys: 'mean', 'median', 'std', 'min', 'max', 'count'

**Example**:
```python
from src.evaluation.lead_time import compute_average_lead_time

stats = compute_average_lead_time(predictions, t_sepsis)
print(f"Mean lead time: {stats['mean']:.1f} hours")
print(f"Count: {stats['count']} patients")
```

---

#### get_detection_rate_by_lead_time

```python
def get_detection_rate_by_lead_time(
    predictions: Dict[str, np.ndarray],
    t_sepsis: Dict[str, int],
    time_points: List[int] = [1, 3, 6, 12],
    threshold: float = 0.5
) -> Dict[int, float]
```

Compute detection rates at various lead time thresholds.

**Returns**:
- `Dict[int, float]`: Time point -> detection rate (0.0-1.0)

**Example**:
```python
from src.evaluation.lead_time import get_detection_rate_by_lead_time

rates = get_detection_rate_by_lead_time(predictions, t_sepsis)
print(rates)  # {1: 0.95, 3: 0.82, 6: 0.65, 12: 0.31}
```

---

### Metrics

**Location**: `src/evaluation/metrics.py`

#### compute_roc_auc

```python
def compute_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]
```

Compute ROC AUC and curve data.

**Returns**:
- `Tuple[float, np.ndarray, np.ndarray]`: (auc_score, fpr_array, tpr_array)

---

#### compute_pr_auc

```python
def compute_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]
```

Compute Precision-Recall AUC and curve data.

**Returns**:
- `Tuple[float, np.ndarray, np.ndarray]`: (auc_score, precision_array, recall_array)

---

#### compute_classification_metrics

```python
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]
```

Compute standard classification metrics.

**Returns**:
- `Dict` with keys: 'sensitivity', 'specificity', 'ppv', 'npv', 'f1', 'accuracy'

**Example**:
```python
from src.evaluation.metrics import compute_classification_metrics
import numpy as np

y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0, 0, 1, 1])

metrics = compute_classification_metrics(y_true, y_pred)
print(metrics)
# {'sensitivity': 1.0, 'specificity': 1.0, 'ppv': 1.0, 'npv': 1.0, 'f1': 1.0, 'accuracy': 1.0}
```

---

## Visualization Module

**Location**: `src/visualization/`

### Charts (`charts.py`)

#### create_roc_chart

```python
def create_roc_chart(
    results: Dict[str, Tuple[float, np.ndarray, np.ndarray]]
) -> go.Figure
```

Create multi-model ROC comparison chart.

**Parameters**:
- `results`: Model name -> (AUC, FPR array, TPR array)

**Returns**:
- `go.Figure`: Plotly figure

---

#### create_lead_time_chart

```python
def create_lead_time_chart(
    lead_times: Dict[str, np.ndarray]
) -> go.Figure
```

Create lead time distribution histogram.

**Parameters**:
- `lead_times`: Model name -> array of lead times

---

#### create_utility_chart

```python
def create_utility_chart(
    utilities: Dict[str, float]
) -> go.Figure
```

Create utility score bar chart.

**Parameters**:
- `utilities`: Model name -> utility score

---

#### create_patient_timeline

```python
def create_patient_timeline(
    vitals: pd.DataFrame,
    predictions: Dict[str, np.ndarray]
) -> go.Figure
```

Create patient vitals + prediction timeline.

**Parameters**:
- `vitals`: DataFrame with vital sign columns
- `predictions`: Model name -> prediction probabilities

---

### Components (`components.py`)

#### metric_card

```python
def metric_card(
    title: str,
    value: str,
    delta: Optional[str] = None
) -> str
```

Create HTML metric card.

**Returns**:
- `str`: HTML string for rendering with `st.markdown(html, unsafe_allow_html=True)`

---

#### alert_banner

```python
def alert_banner(
    message: str,
    severity: str  # 'info', 'success', 'warning', 'danger'
) -> str
```

Create alert banner HTML.

---

#### model_comparison_table

```python
def model_comparison_table(
    metrics: Dict[str, Dict]
) -> pd.DataFrame
```

Create formatted comparison table.

**Parameters**:
- `metrics`: Model name -> {metric_name: value}

---

## Utilities

### Configuration (`src/utils/config.py`)

```python
@dataclass
class Config:
    DATA_DIR: Path
    MODELS_DIR: Path
    SAMPLE_DATA_DIR: Path
    PREDICTION_THRESHOLD: float = 0.5
    OPTIMAL_LEAD_TIME: int = 6
    MAX_LEAD_TIME: int = 12
    QSOFA_THRESHOLD: int = 2

def get_config() -> Config
```

Get singleton configuration instance.

```python
def load_env_config(config: Optional[Config] = None) -> Config
```

Load configuration from environment variables.

**Environment Variables**:
- `SEPSISPULSE_DATA_DIR`
- `SEPSISPULSE_MODELS_DIR`
- `SEPSISPULSE_PREDICTION_THRESHOLD`
- `SEPSISPULSE_OPTIMAL_LEAD_TIME`
- `SEPSISPULSE_MAX_LEAD_TIME`
- `SEPSISPULSE_QSOFA_THRESHOLD`

---

### Caching (`src/utils/cache.py`)

```python
@st.cache_data
def cached_load_data(
    data_dir: str,
    max_patients: Optional[int] = None
) -> Dict[str, pd.DataFrame]
```

Load patient data with Streamlit caching.

```python
@st.cache_data
def cached_load_sample_data() -> Dict[str, pd.DataFrame]
```

Load sample data with caching.

```python
@st.cache_data
def cached_model_predictions(
    model_name: str,  # 'qsofa', 'xgboost', 'tft'
    data: pd.DataFrame
) -> np.ndarray
```

Get cached model predictions.

```python
def clear_cache() -> None
```

Clear all Streamlit caches.
