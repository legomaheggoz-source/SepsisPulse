"""
PyTorch Datasets for sepsis prediction training.

Design Decisions:

1. SepsisDataset (Hour-Level):
   - Returns individual hourly samples with features and labels
   - Suitable for XGBoost feature extraction
   - Memory efficient for large datasets

2. SepsisSequenceDataset (Sequence-Level):
   - Returns variable-length patient sequences
   - Padded/masked for batching
   - Required for TFT-Lite temporal modeling

3. Memory Optimization:
   - Lazy loading: Read files on-demand, not all at once
   - Caching: Optional memory cache for frequently accessed patients
   - Chunked iteration: Process patients in chunks for large datasets
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

# PhysioNet 2019 column definitions
VITAL_SIGNS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
LABEL_COL = "SepsisLabel"

# All feature columns (excluding label and ICULOS which is time index)
FEATURE_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime"
]


def load_patient_psv(file_path: Path) -> pd.DataFrame:
    """
    Load a single patient PSV file.

    Args:
        file_path: Path to PSV file

    Returns:
        DataFrame with patient time series data
    """
    df = pd.read_csv(file_path, sep="|", na_values=["NaN", "nan", ""])
    return df


def impute_features(df: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
    """
    Impute missing values in patient data.

    PhysioNet data has high missingness (~80% for lab values).

    Strategy:
    1. Forward fill: Carry last observation forward (clinical standard)
    2. Backward fill: Fill initial NaNs with first available value
    3. Global median: Fill remaining with feature medians

    Args:
        df: Patient DataFrame
        method: Imputation method

    Returns:
        DataFrame with imputed values
    """
    df = df.copy()

    if method == "forward_fill":
        # Forward fill first (carry last observation)
        df = df.ffill()
        # Backward fill for initial NaNs
        df = df.bfill()

    # Fill any remaining NaNs with column medians
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

    return df


class SepsisDataset(Dataset):
    """
    Hour-level dataset for sepsis prediction.

    Each sample is a single hourly observation with:
    - features: [41] tensor of vital signs and lab values
    - label: Binary sepsis label (0 or 1)
    - patient_id: String identifier
    - hour: Hour index within patient stay

    Suitable for XGBoost and simple neural networks.

    Example:
        >>> dataset = SepsisDataset(patient_ids, data_dir)
        >>> features, label, patient_id, hour = dataset[0]
        >>> print(features.shape)  # torch.Size([41])
    """

    def __init__(
        self,
        patient_ids: List[str],
        data_dir: Path,
        feature_cols: Optional[List[str]] = None,
        normalize: bool = True,
        cache_patients: bool = True,
    ):
        """
        Initialize hour-level dataset.

        Args:
            patient_ids: List of patient IDs to include
            data_dir: Path to PhysioNet data directory
            feature_cols: Columns to use as features (default: all 41)
            normalize: Whether to z-score normalize features
            cache_patients: Keep loaded patients in memory
        """
        self.patient_ids = list(patient_ids)
        self.data_dir = Path(data_dir)
        self.feature_cols = feature_cols or FEATURE_COLS
        self.normalize = normalize
        self.cache_patients = cache_patients

        # Build patient file mapping
        self._file_mapping = self._build_file_mapping()

        # Build hour-to-patient index
        self._hour_index = self._build_hour_index()

        # Cache for loaded patients
        self._cache: Dict[str, pd.DataFrame] = {}

        # Normalization statistics (computed lazily)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        logger.info(f"SepsisDataset: {len(self.patient_ids)} patients, {len(self)} hours")

    def _build_file_mapping(self) -> Dict[str, Path]:
        """Build mapping from patient ID to file path."""
        mapping = {}
        for set_dir in ["training_setA", "training_setB"]:
            set_path = self.data_dir / set_dir
            if set_path.exists():
                for psv_path in set_path.glob("*.psv"):
                    mapping[psv_path.stem] = psv_path
        return mapping

    def _build_hour_index(self) -> List[Tuple[str, int]]:
        """
        Build flat index mapping dataset index to (patient_id, hour).

        This allows random access by hour across all patients.
        """
        index = []
        for patient_id in self.patient_ids:
            if patient_id not in self._file_mapping:
                continue
            # Count hours for this patient
            df = load_patient_psv(self._file_mapping[patient_id])
            n_hours = len(df)
            for hour in range(n_hours):
                index.append((patient_id, hour))
        return index

    def _load_patient(self, patient_id: str) -> pd.DataFrame:
        """Load and preprocess a single patient."""
        if self.cache_patients and patient_id in self._cache:
            return self._cache[patient_id]

        file_path = self._file_mapping[patient_id]
        df = load_patient_psv(file_path)
        df = impute_features(df)

        if self.cache_patients:
            self._cache[patient_id] = df

        return df

    def compute_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute feature mean and std across all training data.

        Returns:
            Tuple of (mean, std) arrays of shape [n_features]
        """
        if self._mean is not None and self._std is not None:
            return self._mean, self._std

        logger.info("Computing normalization statistics...")
        all_features = []

        for patient_id in self.patient_ids:
            df = self._load_patient(patient_id)
            features = df[self.feature_cols].values
            all_features.append(features)

        all_features = np.vstack(all_features)
        self._mean = np.nanmean(all_features, axis=0)
        self._std = np.nanstd(all_features, axis=0)

        # Prevent division by zero
        self._std[self._std < 1e-8] = 1.0

        return self._mean, self._std

    def __len__(self) -> int:
        return len(self._hour_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        """
        Get a single hourly sample.

        Args:
            idx: Dataset index

        Returns:
            Tuple of (features, label, patient_id, hour)
        """
        patient_id, hour = self._hour_index[idx]
        df = self._load_patient(patient_id)

        # Extract features and label
        features = df.iloc[hour][self.feature_cols].values.astype(np.float32)
        label = df.iloc[hour][LABEL_COL]

        # Normalize if requested
        if self.normalize:
            if self._mean is None:
                self.compute_normalization_stats()
            features = (features - self._mean) / self._std

        return (
            torch.from_numpy(features),
            torch.tensor(label, dtype=torch.float32),
            patient_id,
            hour,
        )


class SepsisSequenceDataset(Dataset):
    """
    Sequence-level dataset for TFT-Lite training.

    Each sample is a full patient sequence:
    - features: [seq_len, 41] tensor of time series
    - labels: [seq_len] tensor of hourly sepsis labels
    - mask: [seq_len] boolean mask (True = valid, False = padding)
    - patient_id: String identifier

    Supports variable-length sequences with padding for batching.

    Example:
        >>> dataset = SepsisSequenceDataset(patient_ids, data_dir, max_seq_len=72)
        >>> features, labels, mask, patient_id = dataset[0]
        >>> print(features.shape)  # torch.Size([72, 41]) or shorter
    """

    def __init__(
        self,
        patient_ids: List[str],
        data_dir: Path,
        max_seq_len: int = 72,
        min_seq_len: int = 6,
        feature_cols: Optional[List[str]] = None,
        normalize: bool = True,
    ):
        """
        Initialize sequence dataset.

        Args:
            patient_ids: List of patient IDs
            data_dir: Path to PhysioNet data
            max_seq_len: Maximum sequence length (truncate longer)
            min_seq_len: Minimum sequence length (skip shorter)
            feature_cols: Feature columns to use
            normalize: Whether to normalize features
        """
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.feature_cols = feature_cols or FEATURE_COLS
        self.normalize = normalize

        # Build file mapping
        self._file_mapping = {}
        for set_dir in ["training_setA", "training_setB"]:
            set_path = self.data_dir / set_dir
            if set_path.exists():
                for psv_path in set_path.glob("*.psv"):
                    self._file_mapping[psv_path.stem] = psv_path

        # Filter patients by minimum length
        self.patient_ids = []
        for pid in patient_ids:
            if pid in self._file_mapping:
                df = load_patient_psv(self._file_mapping[pid])
                if len(df) >= min_seq_len:
                    self.patient_ids.append(pid)

        # Normalization stats
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        logger.info(
            f"SepsisSequenceDataset: {len(self.patient_ids)} patients "
            f"(max_seq={max_seq_len}, min_seq={min_seq_len})"
        )

    def set_normalization_stats(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization statistics (from training set)."""
        self._mean = mean
        self._std = std

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Get a patient sequence.

        Args:
            idx: Patient index

        Returns:
            Tuple of (features, labels, mask, patient_id)
        """
        patient_id = self.patient_ids[idx]
        df = load_patient_psv(self._file_mapping[patient_id])
        df = impute_features(df)

        # Extract features and labels
        features = df[self.feature_cols].values.astype(np.float32)
        labels = df[LABEL_COL].values.astype(np.float32)

        # Truncate if too long
        if len(features) > self.max_seq_len:
            features = features[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        seq_len = len(features)

        # Normalize
        if self.normalize and self._mean is not None:
            features = (features - self._mean) / self._std

        # Create mask (all valid for unpadded)
        mask = np.ones(seq_len, dtype=bool)

        return (
            torch.from_numpy(features),
            torch.from_numpy(labels),
            torch.from_numpy(mask),
            patient_id,
        )


def collate_sequences(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Collate function for variable-length sequences.

    Pads sequences to the maximum length in the batch.

    Args:
        batch: List of (features, labels, mask, patient_id) tuples

    Returns:
        Tuple of padded (features, labels, mask, patient_ids)
        - features: [batch, max_seq_len, n_features]
        - labels: [batch, max_seq_len]
        - mask: [batch, max_seq_len]
        - patient_ids: List of strings
    """
    features, labels, masks, patient_ids = zip(*batch)

    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0.0)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=False)

    return features_padded, labels_padded, masks_padded, list(patient_ids)


def create_data_loaders(
    train_patients: List[str],
    val_patients: List[str],
    data_dir: Path,
    batch_size: int = 1024,
    num_workers: int = 4,
    max_seq_len: int = 72,
    for_tft: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        train_patients: Training patient IDs
        val_patients: Validation patient IDs
        data_dir: Path to PhysioNet data
        batch_size: Batch size
        num_workers: DataLoader workers
        max_seq_len: Maximum sequence length (for TFT)
        for_tft: If True, use sequence dataset; else hour-level

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if for_tft:
        train_dataset = SepsisSequenceDataset(
            train_patients, data_dir, max_seq_len=max_seq_len
        )
        val_dataset = SepsisSequenceDataset(
            val_patients, data_dir, max_seq_len=max_seq_len
        )

        # Compute normalization from training set only
        # Create a temporary hour-level dataset for stats
        temp_dataset = SepsisDataset(train_patients, data_dir, normalize=False)
        mean, std = temp_dataset.compute_normalization_stats()
        train_dataset.set_normalization_stats(mean, std)
        val_dataset.set_normalization_stats(mean, std)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_sequences,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_sequences,
            pin_memory=True,
        )
    else:
        train_dataset = SepsisDataset(train_patients, data_dir)
        val_dataset = SepsisDataset(val_patients, data_dir)

        # Share normalization stats
        mean, std = train_dataset.compute_normalization_stats()
        val_dataset._mean = mean
        val_dataset._std = std

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
