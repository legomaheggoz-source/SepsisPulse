"""Streamlit caching utilities for SepsisPulse.

This module provides cached data loading and model prediction functions
using Streamlit's caching decorators to improve application performance.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader import load_dataset, get_sample_subset


@st.cache_data(show_spinner="Loading patient data...")
def cached_load_data(
    data_dir: str,
    max_patients: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Load patient data with Streamlit caching.

    Uses @st.cache_data to cache the loaded data across reruns,
    significantly improving load times for large datasets.

    Args:
        data_dir: Path to directory containing .psv patient files.
        max_patients: Maximum number of patients to load. If None, loads all.

    Returns:
        Dictionary mapping patient IDs to their DataFrames.
        Example: {"p00001": DataFrame, "p00002": DataFrame, ...}

    Example:
        >>> data = cached_load_data("data/training_setA")
        >>> len(data)
        1000
        >>> data["p00001"].shape
        (48, 41)
    """
    return load_dataset(data_dir, max_patients=max_patients)


@st.cache_data(show_spinner="Loading sample data...")
def cached_load_sample_data() -> Dict[str, pd.DataFrame]:
    """Load bundled sample data with Streamlit caching.

    Returns:
        Dictionary mapping patient IDs to their DataFrames.
        Returns empty dict if sample data not found.

    Example:
        >>> sample = cached_load_sample_data()
        >>> len(sample)
        5
    """
    return get_sample_subset()


@st.cache_resource(show_spinner="Loading model...")
def _get_model_instance(model_name: str):
    """Get a cached model instance by name.

    Uses @st.cache_resource for models since they are not serializable
    and should be shared across sessions.

    Args:
        model_name: Name of the model ("qsofa", "xgboost", or "tft").

    Returns:
        Model instance or None if model not found.
    """
    try:
        if model_name.lower() == "qsofa":
            from models import QSOFAModel
            return QSOFAModel()
        elif model_name.lower() in ("xgboost", "xgboost-ts", "xgboost_ts"):
            from models import XGBoostTSModel
            return XGBoostTSModel()
        elif model_name.lower() in ("tft", "tft-lite", "tft_lite"):
            from models import TFTLiteModel
            return TFTLiteModel()
        else:
            return None
    except ImportError:
        return None


@st.cache_data(show_spinner="Running predictions...")
def cached_model_predictions(
    model_name: str,
    data: pd.DataFrame,
) -> np.ndarray:
    """Get cached model predictions for patient data.

    Uses @st.cache_data to cache prediction results. The cache key
    includes the model name and a hash of the input data.

    Args:
        model_name: Name of the model to use for predictions.
            Options: "qsofa", "xgboost" / "xgboost-ts", "tft" / "tft-lite"
        data: DataFrame containing patient data in PhysioNet format.
            Expected columns: HR, Resp, SBP, MAP, etc.

    Returns:
        NumPy array of prediction probabilities with shape (n_samples,).
        Values are in range [0.0, 1.0] representing sepsis probability.

    Raises:
        ValueError: If the model name is not recognized.

    Example:
        >>> predictions = cached_model_predictions("xgboost", patient_df)
        >>> predictions.shape
        (48,)
        >>> predictions[0]
        0.23
    """
    model = _get_model_instance(model_name)

    if model is None:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Valid options: qsofa, xgboost, xgboost-ts, tft, tft-lite"
        )

    # Call the model's predict method
    predictions = model.predict(data)

    # Ensure output is a numpy array
    if isinstance(predictions, pd.Series):
        predictions = predictions.values
    elif not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    return predictions


def clear_cache() -> None:
    """Clear all Streamlit caches.

    Clears both data caches (@st.cache_data) and resource caches
    (@st.cache_resource). Useful when data has been updated or
    when models need to be reloaded.

    Example:
        >>> clear_cache()
        >>> # Subsequent calls will reload data from disk
    """
    st.cache_data.clear()
    st.cache_resource.clear()


def clear_data_cache() -> None:
    """Clear only the data cache, preserving model cache.

    Use this when data has changed but models don't need reloading.

    Example:
        >>> clear_data_cache()
    """
    st.cache_data.clear()


def clear_model_cache() -> None:
    """Clear only the model cache, preserving data cache.

    Use this when models need to be reloaded but data is unchanged.

    Example:
        >>> clear_model_cache()
    """
    st.cache_resource.clear()
