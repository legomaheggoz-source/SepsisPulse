"""
HuggingFace Dataset loader for PhysioNet Sepsis 2019 data.

This module provides functions to load patient data from a HuggingFace Dataset,
which allows the SepsisPulse app to access the full 40,311 patient dataset
without bundling the data in the repository.

The dataset is hosted at: huggingface.co/datasets/legomaheggo/physionet-sepsis-2019
"""

import logging
from typing import Dict, List, Optional, Any
from functools import lru_cache
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# HuggingFace Dataset repository ID
HF_DATASET_REPO = "legomaheggo/physionet-sepsis-2019"

# Cache for loaded dataset
_dataset_cache = None


def is_hf_dataset_available() -> bool:
    """Check if HuggingFace datasets library is available."""
    try:
        import datasets
        return True
    except ImportError:
        return False


def load_hf_dataset(force_reload: bool = False):
    """
    Load the PhysioNet Sepsis dataset from HuggingFace.

    Args:
        force_reload: If True, reload from HuggingFace even if cached.

    Returns:
        HuggingFace Dataset object, or None if unavailable.
    """
    global _dataset_cache

    if _dataset_cache is not None and not force_reload:
        return _dataset_cache

    if not is_hf_dataset_available():
        logger.warning("HuggingFace datasets library not installed")
        return None

    try:
        from datasets import load_dataset

        logger.info(f"Loading dataset from {HF_DATASET_REPO}...")
        dataset = load_dataset(HF_DATASET_REPO, split="train")
        _dataset_cache = dataset
        logger.info(f"Loaded {len(dataset)} patients from HuggingFace")
        return dataset

    except Exception as e:
        logger.warning(f"Failed to load HuggingFace dataset: {e}")
        return None


def get_patient_ids_from_hf() -> List[str]:
    """
    Get list of all patient IDs from HuggingFace dataset.

    Returns:
        Sorted list of patient IDs, or empty list if unavailable.
    """
    dataset = load_hf_dataset()
    if dataset is None:
        logger.warning("Dataset is None, cannot get patient IDs")
        return []

    try:
        # Access patient_id column directly - more efficient than iterating
        patient_ids = dataset["patient_id"]
        if patient_ids:
            sorted_ids = sorted(set(patient_ids))  # Use set to ensure uniqueness
            logger.info(f"Retrieved {len(sorted_ids)} patient IDs from HuggingFace")
            return sorted_ids
        else:
            logger.warning("patient_id column is empty")
            return []
    except Exception as e:
        logger.warning(f"Failed to get patient IDs: {e}")
        return []


# Index cache for fast patient lookups
_patient_index_cache = None


def _build_patient_index():
    """Build an index mapping patient_id to dataset row index for fast lookup."""
    global _patient_index_cache
    if _patient_index_cache is not None:
        return _patient_index_cache

    dataset = load_hf_dataset()
    if dataset is None:
        return {}

    try:
        # Build index: patient_id -> row index
        patient_ids = dataset["patient_id"]
        _patient_index_cache = {pid: idx for idx, pid in enumerate(patient_ids)}
        logger.info(f"Built patient index with {len(_patient_index_cache)} entries")
        return _patient_index_cache
    except Exception as e:
        logger.warning(f"Failed to build patient index: {e}")
        return {}


def get_patient_from_hf(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single patient's data from HuggingFace dataset.

    Args:
        patient_id: The patient ID (e.g., "p000001")

    Returns:
        Dictionary with patient data, or None if not found.
    """
    dataset = load_hf_dataset()
    if dataset is None:
        return None

    try:
        # Use index for fast lookup
        index = _build_patient_index()
        if patient_id not in index:
            logger.warning(f"Patient {patient_id} not found in dataset index")
            return None

        row_idx = index[patient_id]
        record = dataset[row_idx]
        return dict(record)

    except Exception as e:
        logger.warning(f"Failed to get patient {patient_id}: {e}")
        return None


def patient_record_to_dataframe(record: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a HuggingFace patient record to a pandas DataFrame.

    Args:
        record: Patient record dictionary from HuggingFace dataset.

    Returns:
        DataFrame with patient vitals, one row per hour.
    """
    if record is None:
        return pd.DataFrame()

    # Get the length from one of the vital sign arrays
    hr_data = record.get("HR", [])
    n_hours = len(hr_data) if hr_data else 0

    if n_hours == 0:
        return pd.DataFrame()

    # Define vital sign columns to extract
    vital_columns = ["HR", "SBP", "DBP", "MAP", "Resp", "Temp", "O2Sat", "SepsisLabel"]

    # Build DataFrame from vital signs
    data = {}
    for col in vital_columns:
        col_data = record.get(col, [])
        if col_data and len(col_data) == n_hours:
            # Convert to numpy array to properly handle NaN values
            data[col] = np.array(col_data, dtype=np.float64)
        else:
            # Fill with NaN if missing
            data[col] = np.full(n_hours, np.nan)

    df = pd.DataFrame(data)

    # Ensure SepsisLabel is integer (0 or 1)
    df["SepsisLabel"] = df["SepsisLabel"].fillna(0).astype(int)

    # Add demographics (constant across all rows)
    df["Age"] = record.get("age")
    df["Gender"] = record.get("gender")

    return df


def get_patient_summary_from_hf(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Get summary information for a patient without loading full vitals.

    Args:
        patient_id: The patient ID

    Returns:
        Dictionary with summary info (age, gender, icu_hours, has_sepsis, etc.)
    """
    record = get_patient_from_hf(patient_id)
    if record is None:
        return None

    return {
        "patient_id": record.get("patient_id"),
        "age": record.get("age"),
        "gender": record.get("gender"),
        "icu_hours": record.get("icu_hours"),
        "has_sepsis": record.get("has_sepsis"),
        "sepsis_onset_hour": record.get("sepsis_onset_hour"),
    }


def get_dataset_statistics() -> Dict[str, Any]:
    """
    Get statistics about the loaded dataset.

    Returns:
        Dictionary with dataset statistics.
    """
    dataset = load_hf_dataset()
    if dataset is None:
        return {"available": False}

    try:
        total_patients = len(dataset)
        sepsis_count = sum(1 for x in dataset if x["has_sepsis"])

        return {
            "available": True,
            "total_patients": total_patients,
            "sepsis_positive": sepsis_count,
            "sepsis_negative": total_patients - sepsis_count,
            "sepsis_rate": sepsis_count / total_patients if total_patients > 0 else 0,
            "source": HF_DATASET_REPO,
        }
    except Exception as e:
        logger.warning(f"Failed to get dataset statistics: {e}")
        return {"available": False, "error": str(e)}
