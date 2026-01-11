"""Data loading and preprocessing modules for PhysioNet 2019 data."""

from .loader import load_patient, load_dataset, get_sample_subset
from .preprocessor import preprocess_patient, handle_missing_values
from .feature_engineering import (
    create_lag_features,
    create_rolling_features,
    create_rate_of_change,
    create_all_features,
    get_feature_names,
)

# HuggingFace dataset loading (optional, may not be available)
try:
    from .hf_dataset import (
        load_hf_dataset,
        get_patient_ids_from_hf,
        get_patient_from_hf,
        patient_record_to_dataframe,
        get_dataset_statistics,
        is_hf_dataset_available,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

__all__ = [
    "load_patient",
    "load_dataset",
    "get_sample_subset",
    "preprocess_patient",
    "handle_missing_values",
    "create_lag_features",
    "create_rolling_features",
    "create_rate_of_change",
    "create_all_features",
    "get_feature_names",
    # HuggingFace (conditional)
    "HF_AVAILABLE",
]
