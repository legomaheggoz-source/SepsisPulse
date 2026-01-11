"""Data loading module for PhysioNet 2019 Sepsis Challenge PSV files.

This module provides functions to load patient data from the PhysioNet 2019
Sepsis Early Prediction Challenge. Data format: pipe-delimited .psv files,
one per patient, with 41 variables (8 vitals, 26 labs, 6 demographics, 1 label).
Each row represents 1 hour of ICU stay.

Reference: https://physionet.org/content/challenge-2019/
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# PhysioNet 2019 Challenge: All 41 column names
# 8 Vital Signs
VITAL_COLUMNS = [
    "HR",        # Heart rate (beats per minute)
    "O2Sat",     # Pulse oximetry (%)
    "Temp",      # Temperature (Deg C)
    "SBP",       # Systolic BP (mm Hg)
    "MAP",       # Mean arterial pressure (mm Hg)
    "DBP",       # Diastolic BP (mm Hg)
    "Resp",      # Respiration rate (breaths per minute)
    "EtCO2",     # End tidal carbon dioxide (mm Hg)
]

# 26 Laboratory Values
LAB_COLUMNS = [
    "BaseExcess",     # Measure of excess bicarbonate (mmol/L)
    "HCO3",           # Bicarbonate (mmol/L)
    "FiO2",           # Fraction of inspired oxygen (%)
    "pH",             # N/A
    "PaCO2",          # Partial pressure of carbon dioxide from arterial blood (mm Hg)
    "SaO2",           # Oxygen saturation from arterial blood (%)
    "AST",            # Aspartate transaminase (IU/L)
    "BUN",            # Blood urea nitrogen (mg/dL)
    "Alkalinephos",   # Alkaline phosphatase (IU/L)
    "Calcium",        # (mg/dL)
    "Chloride",       # (mmol/L)
    "Creatinine",     # (mg/dL)
    "Bilirubin_direct",  # Bilirubin direct (mg/dL)
    "Glucose",        # Serum glucose (mg/dL)
    "Lactate",        # Lactic acid (mg/dL)
    "Magnesium",      # (mmol/dL)
    "Phosphate",      # (mg/dL)
    "Potassium",      # (mmol/L)
    "Bilirubin_total",   # Total bilirubin (mg/dL)
    "TroponinI",      # Troponin I (ng/mL)
    "Hct",            # Hematocrit (%)
    "Hgb",            # Hemoglobin (g/dL)
    "PTT",            # Partial thromboplastin time (seconds)
    "WBC",            # Leukocyte count (count*10^3/uL)
    "Fibrinogen",     # (mg/dL)
    "Platelets",      # (count*10^3/uL)
]

# 6 Demographics
DEMOGRAPHIC_COLUMNS = [
    "Age",            # Years (100 for patients 90 or above)
    "Gender",         # Female (0) or Male (1)
    "Unit1",          # Administrative identifier for ICU unit (MICU)
    "Unit2",          # Administrative identifier for ICU unit (SICU)
    "HospAdmTime",    # Hours between hospital admit and ICU admit
    "ICULOS",         # ICU length-of-stay (hours since ICU admit)
]

# 1 Label
LABEL_COLUMN = ["SepsisLabel"]

# All 41 columns in order as they appear in PhysioNet 2019 data
PHYSIONET_COLUMNS: List[str] = (
    VITAL_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS + LABEL_COLUMN
)


def load_patient(file_path: str) -> pd.DataFrame:
    """Load a single patient's PSV file into a DataFrame.

    Args:
        file_path: Path to the .psv file (pipe-delimited).

    Returns:
        DataFrame with patient data. Each row is one hour of ICU stay.
        Missing values are preserved as NaN.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.

    Example:
        >>> df = load_patient("data/p00001.psv")
        >>> df.shape
        (48, 41)  # 48 hours of data, 41 variables
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Patient file not found: {file_path}")

    if not file_path.suffix.lower() == ".psv":
        raise ValueError(f"Expected .psv file, got: {file_path.suffix}")

    try:
        df = pd.read_csv(
            file_path,
            sep="|",
            na_values=["", "NaN", "nan", "NA", "N/A"],
            dtype={col: float for col in PHYSIONET_COLUMNS},
        )
    except Exception as e:
        raise ValueError(f"Failed to parse PSV file {file_path}: {e}") from e

    # Validate columns
    missing_cols = set(PHYSIONET_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing expected columns in {file_path}: {sorted(missing_cols)}"
        )

    # Ensure column order matches expected order
    df = df[PHYSIONET_COLUMNS]

    return df


def load_dataset(
    data_dir: str,
    max_patients: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Load multiple patient PSV files from a directory.

    Args:
        data_dir: Directory containing .psv files.
        max_patients: Maximum number of patients to load. If None, load all.
            Useful for testing or memory-constrained environments.

    Returns:
        Dictionary mapping patient IDs (filename without extension) to DataFrames.
        Example: {"p00001": DataFrame, "p00002": DataFrame, ...}

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no .psv files are found in the directory.

    Example:
        >>> dataset = load_dataset("data/training_setA", max_patients=100)
        >>> len(dataset)
        100
        >>> dataset["p00001"].shape
        (48, 41)
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if not data_dir.is_dir():
        raise ValueError(f"Path is not a directory: {data_dir}")

    # Find all PSV files
    psv_files = sorted(data_dir.glob("*.psv"))

    if not psv_files:
        raise ValueError(f"No .psv files found in directory: {data_dir}")

    # Limit number of patients if specified
    if max_patients is not None:
        psv_files = psv_files[:max_patients]

    # Load all patient files
    dataset: Dict[str, pd.DataFrame] = {}

    for psv_file in psv_files:
        patient_id = psv_file.stem  # Filename without extension
        try:
            dataset[patient_id] = load_patient(str(psv_file))
        except (ValueError, FileNotFoundError) as e:
            # Log warning but continue loading other patients
            print(f"Warning: Skipping {psv_file.name}: {e}")
            continue

    if not dataset:
        raise ValueError(f"Failed to load any patients from: {data_dir}")

    return dataset


def get_sample_subset() -> Dict[str, pd.DataFrame]:
    """Load bundled sample data for testing and development.

    This function loads a small subset of sample patient data that is
    bundled with the package for testing purposes.

    Returns:
        Dictionary mapping patient IDs to DataFrames.
        Returns empty dict if sample data directory is empty or not found.

    Example:
        >>> sample = get_sample_subset()
        >>> len(sample)
        5  # Number of sample patients included
    """
    # Determine the path to the sample data directory
    # This handles both installed package and development scenarios
    module_dir = Path(__file__).parent

    # Try multiple possible locations for sample data
    possible_paths = [
        module_dir.parent.parent / "data" / "sample" / "patients",  # Development: src/data -> data/sample/patients
        module_dir.parent.parent.parent / "data" / "sample" / "patients",  # Alternative structure
        Path(__file__).resolve().parent.parent.parent / "data" / "sample" / "patients",
    ]

    sample_dir = None
    for path in possible_paths:
        if path.exists() and path.is_dir():
            sample_dir = path
            break

    if sample_dir is None:
        # Return empty dict if sample data not found
        return {}

    # Check if there are any PSV files
    psv_files = list(sample_dir.glob("*.psv"))
    if not psv_files:
        return {}

    return load_dataset(str(sample_dir))


def get_patient_ids(data_dir: str) -> List[str]:
    """Get list of patient IDs available in a directory without loading data.

    Args:
        data_dir: Directory containing .psv files.

    Returns:
        Sorted list of patient IDs (filenames without .psv extension).

    Example:
        >>> ids = get_patient_ids("data/training_setA")
        >>> ids[:3]
        ['p00001', 'p00002', 'p00003']
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    psv_files = sorted(data_dir.glob("*.psv"))
    return [f.stem for f in psv_files]


def get_column_groups() -> Dict[str, List[str]]:
    """Get column names grouped by category.

    Returns:
        Dictionary with keys 'vitals', 'labs', 'demographics', 'label',
        each containing a list of column names.

    Example:
        >>> groups = get_column_groups()
        >>> groups['vitals']
        ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
    """
    return {
        "vitals": VITAL_COLUMNS.copy(),
        "labs": LAB_COLUMNS.copy(),
        "demographics": DEMOGRAPHIC_COLUMNS.copy(),
        "label": LABEL_COLUMN.copy(),
    }
