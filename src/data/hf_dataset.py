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
        return []

    try:
        patient_ids = sorted(dataset["patient_id"])
        return patient_ids
    except Exception as e:
        logger.warning(f"Failed to get patient IDs: {e}")
        return []


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
        # Filter dataset for the specific patient
        patient_data = dataset.filter(lambda x: x["patient_id"] == patient_id)

        if len(patient_data) == 0:
            logger.warning(f"Patient {patient_id} not found in dataset")
            return None

        # Get the first (should be only) matching record
        record = patient_data[0]
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
    n_hours = len(record.get("HR", []))

    if n_hours == 0:
        return pd.DataFrame()

    # Build DataFrame from vital signs
    data = {
        "HR": record.get("HR", [None] * n_hours),
        "SBP": record.get("SBP", [None] * n_hours),
        "DBP": record.get("DBP", [None] * n_hours),
        "MAP": record.get("MAP", [None] * n_hours),
        "Resp": record.get("Resp", [None] * n_hours),
        "Temp": record.get("Temp", [None] * n_hours),
        "O2Sat": record.get("O2Sat", [None] * n_hours),
        "SepsisLabel": record.get("SepsisLabel", [0] * n_hours),
    }

    df = pd.DataFrame(data)

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
