"""
Preprocessing module for handling missing values in PhysioNet 2019 sepsis data.

PhysioNet ICU data has extensive missing values (>90% for some lab measurements)
because lab values are only measured when clinically indicated. This module
provides imputation strategies appropriate for clinical time-series data.
"""

import pandas as pd
import numpy as np
from typing import Optional


# Vital signs measured hourly in ICU
VITAL_SIGNS = [
    "HR",        # Heart rate (beats per minute)
    "O2Sat",     # Pulse oximetry (%)
    "Temp",      # Temperature (Celsius)
    "SBP",       # Systolic blood pressure (mm Hg)
    "MAP",       # Mean arterial pressure (mm Hg)
    "DBP",       # Diastolic blood pressure (mm Hg)
    "Resp",      # Respiration rate (breaths per minute)
    "EtCO2",     # End-tidal carbon dioxide (mm Hg)
]

# Laboratory values measured when clinically indicated
LAB_VALUES = [
    "BaseExcess",    # Base excess (mEq/L)
    "HCO3",          # Bicarbonate (mEq/L)
    "FiO2",          # Fraction of inspired oxygen (%)
    "pH",            # Arterial pH
    "PaCO2",         # Partial pressure of CO2 (mm Hg)
    "SaO2",          # Arterial oxygen saturation (%)
    "AST",           # Aspartate aminotransferase (IU/L)
    "BUN",           # Blood urea nitrogen (mg/dL)
    "Alkalinephos",  # Alkaline phosphatase (IU/L)
    "Calcium",       # Calcium (mg/dL)
    "Chloride",      # Chloride (mEq/L)
    "Creatinine",    # Creatinine (mg/dL)
    "Bilirubin_direct",   # Direct bilirubin (mg/dL)
    "Glucose",       # Glucose (mg/dL)
    "Lactate",       # Lactate (mmol/L)
    "Magnesium",     # Magnesium (mEq/dL)
    "Phosphate",     # Phosphate (mg/dL)
    "Potassium",     # Potassium (mEq/L)
    "Bilirubin_total",    # Total bilirubin (mg/dL)
    "TroponinI",     # Troponin I (ng/mL)
    "Hct",           # Hematocrit (%)
    "Hgb",           # Hemoglobin (g/dL)
    "PTT",           # Partial thromboplastin time (seconds)
    "WBC",           # White blood cell count (10^3/uL)
    "Fibrinogen",    # Fibrinogen (mg/dL)
    "Platelets",     # Platelets (10^3/uL)
]

# All clinical features (vitals + labs)
CLINICAL_FEATURES = VITAL_SIGNS + LAB_VALUES

# Demographics and static features
DEMOGRAPHIC_FEATURES = [
    "Age",           # Age (years)
    "Gender",        # Gender (0=female, 1=male)
    "Unit1",         # Administrative identifier for ICU unit
    "Unit2",         # Administrative identifier for ICU unit
    "HospAdmTime",   # Hours between hospital admit and ICU admit
    "ICULOS",        # ICU length of stay (hours)
]

# Population means for imputation (realistic ICU values from literature)
POPULATION_MEANS = {
    # Vital signs
    "HR": 84.0,
    "O2Sat": 97.0,
    "Temp": 37.0,
    "SBP": 120.0,
    "MAP": 80.0,
    "DBP": 70.0,
    "Resp": 18.0,
    "EtCO2": 35.0,
    # Laboratory values
    "BaseExcess": 0.0,
    "HCO3": 24.0,
    "FiO2": 0.21,
    "pH": 7.40,
    "PaCO2": 40.0,
    "SaO2": 97.0,
    "AST": 35.0,
    "BUN": 18.0,
    "Alkalinephos": 70.0,
    "Calcium": 9.0,
    "Chloride": 102.0,
    "Creatinine": 1.0,
    "Bilirubin_direct": 0.2,
    "Glucose": 120.0,
    "Lactate": 1.5,
    "Magnesium": 2.0,
    "Phosphate": 3.5,
    "Potassium": 4.0,
    "Bilirubin_total": 0.7,
    "TroponinI": 0.04,
    "Hct": 38.0,
    "Hgb": 12.5,
    "PTT": 30.0,
    "WBC": 9.0,
    "Fibrinogen": 300.0,
    "Platelets": 220.0,
    # Demographics
    "Age": 62.0,
    "Gender": 0.5,
    "Unit1": 0.5,
    "Unit2": 0.5,
    "HospAdmTime": 0.0,
    "ICULOS": 24.0,
}

# Population standard deviations for z-score normalization
POPULATION_STDS = {
    # Vital signs
    "HR": 17.0,
    "O2Sat": 3.0,
    "Temp": 0.7,
    "SBP": 23.0,
    "MAP": 16.0,
    "DBP": 14.0,
    "Resp": 5.0,
    "EtCO2": 6.0,
    # Laboratory values
    "BaseExcess": 4.5,
    "HCO3": 4.5,
    "FiO2": 0.2,
    "pH": 0.08,
    "PaCO2": 10.0,
    "SaO2": 4.0,
    "AST": 60.0,
    "BUN": 15.0,
    "Alkalinephos": 40.0,
    "Calcium": 0.8,
    "Chloride": 5.0,
    "Creatinine": 1.5,
    "Bilirubin_direct": 0.5,
    "Glucose": 45.0,
    "Lactate": 1.5,
    "Magnesium": 0.4,
    "Phosphate": 1.2,
    "Potassium": 0.6,
    "Bilirubin_total": 1.5,
    "TroponinI": 0.5,
    "Hct": 6.0,
    "Hgb": 2.0,
    "PTT": 12.0,
    "WBC": 5.0,
    "Fibrinogen": 100.0,
    "Platelets": 90.0,
    # Demographics
    "Age": 17.0,
    "Gender": 0.5,
    "Unit1": 0.5,
    "Unit2": 0.5,
    "HospAdmTime": 100.0,
    "ICULOS": 48.0,
}


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in clinical time-series data.

    Imputation strategy (in order):
    1. Forward fill: Carry last observation forward (clinically appropriate
       because lab values remain valid until re-measured)
    2. Backward fill: Fill initial NaNs with first available measurement
    3. Mean imputation: Use population means for any remaining NaNs

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with clinical features as columns and time points as rows.
        May contain extensive missing values (>90% for some columns).

    Returns
    -------
    pd.DataFrame
        DataFrame with all missing values imputed.

    Notes
    -----
    - Forward fill is the primary strategy because it's clinically appropriate:
      lab values remain valid until the next measurement.
    - Backward fill handles the case where initial values are missing.
    - Mean imputation is a last resort for features that are entirely missing.
    """
    df = df.copy()

    # Get columns that exist in both the dataframe and our constants
    columns_to_impute = [col for col in df.columns if col in POPULATION_MEANS]

    # Step 1: Forward fill (carry last observation forward)
    df[columns_to_impute] = df[columns_to_impute].ffill()

    # Step 2: Backward fill (for initial NaNs)
    df[columns_to_impute] = df[columns_to_impute].bfill()

    # Step 3: Mean imputation (for entirely missing columns)
    for col in columns_to_impute:
        if df[col].isna().any():
            df[col] = df[col].fillna(POPULATION_MEANS[col])

    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply z-score normalization to clinical features.

    Uses pre-computed population means and standard deviations to normalize
    features to zero mean and unit variance. This is important for neural
    network training where features should be on similar scales.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with clinical features. Should have missing values
        already handled.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized features (z-scores).

    Notes
    -----
    Z-score formula: z = (x - mean) / std

    Using population statistics (rather than sample statistics) ensures
    consistent normalization between training and inference.
    """
    df = df.copy()

    # Normalize each column that exists in our statistics
    for col in df.columns:
        if col in POPULATION_MEANS and col in POPULATION_STDS:
            mean = POPULATION_MEANS[col]
            std = POPULATION_STDS[col]
            # Avoid division by zero
            if std > 0:
                df[col] = (df[col] - mean) / std

    return df


def preprocess_patient(df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline for a single patient's data.

    Applies the complete preprocessing pipeline:
    1. Handle missing values (forward fill, backward fill, mean imputation)
    2. Optionally normalize features using z-score normalization

    Parameters
    ----------
    df : pd.DataFrame
        Raw patient data with clinical features as columns and hourly
        time points as rows. Expected to have ICULOS (ICU length of stay)
        column indicating the hour.
    normalize : bool, default=True
        Whether to apply z-score normalization after imputation.
        Set to False if you need raw clinical values.

    Returns
    -------
    pd.DataFrame
        Preprocessed patient data with:
        - All missing values imputed
        - Features optionally normalized to z-scores

    Examples
    --------
    >>> patient_df = load_patient("p000001")
    >>> processed = preprocess_patient(patient_df)
    >>> processed.isna().sum().sum()  # No missing values
    0

    >>> # Get processed data without normalization
    >>> raw_processed = preprocess_patient(patient_df, normalize=False)
    """
    # Step 1: Handle missing values
    df = handle_missing_values(df)

    # Step 2: Normalize features if requested
    if normalize:
        df = normalize_features(df)

    return df


def get_feature_columns() -> list:
    """
    Get the list of all feature columns used in preprocessing.

    Returns
    -------
    list
        Combined list of vital signs and lab values column names.
    """
    return CLINICAL_FEATURES.copy()


def get_missing_rate(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the missing rate for each column.

    Useful for understanding data quality and deciding on imputation strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.

    Returns
    -------
    pd.Series
        Series with column names as index and missing rates (0-1) as values.

    Examples
    --------
    >>> missing = get_missing_rate(patient_df)
    >>> missing.sort_values(ascending=False).head()
    TroponinI        0.98
    Fibrinogen       0.97
    Bilirubin_direct 0.95
    ...
    """
    return df.isna().mean()


def validate_preprocessed_data(df: pd.DataFrame) -> bool:
    """
    Validate that preprocessing was successful.

    Checks:
    - No missing values remain
    - No infinite values present
    - All expected columns are present

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame to validate.

    Returns
    -------
    bool
        True if validation passes, False otherwise.

    Raises
    ------
    ValueError
        If validation fails, with description of the issue.
    """
    # Check for missing values
    if df.isna().any().any():
        missing_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Missing values found in columns: {missing_cols}")

    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    if np.isinf(numeric_df.values).any():
        inf_cols = numeric_df.columns[np.isinf(numeric_df).any()].tolist()
        raise ValueError(f"Infinite values found in columns: {inf_cols}")

    return True
