"""
Patient-level cross-validation for sepsis prediction.

Design Decision: Patient-Level Splits (NOT Hour-Level)

CRITICAL: Splitting by hours would cause data leakage because:
- Same patient appears in both train and test sets
- Model learns patient-specific patterns, not generalizable sepsis signals
- Validation metrics are artificially inflated

Solution: Stratified K-Fold at patient level
- Each fold contains different patients
- Stratification ensures balanced sepsis prevalence per fold
- All hours from a patient stay in same fold

Example:
    Patient p00001 (62 hours, sepsis onset at hour 43)
    - ALL 62 hours go to either train OR test, never split
    - Stratification ensures ~7.3% sepsis patients in each fold
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Iterator
import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class CVSplit:
    """
    A single cross-validation split.

    Attributes:
        fold_idx: Fold number (0-indexed)
        train_patients: List of patient IDs for training
        val_patients: List of patient IDs for validation
        train_sepsis_count: Number of sepsis patients in training set
        val_sepsis_count: Number of sepsis patients in validation set
    """
    fold_idx: int
    train_patients: List[str]
    val_patients: List[str]
    train_sepsis_count: int
    val_sepsis_count: int

    @property
    def train_size(self) -> int:
        return len(self.train_patients)

    @property
    def val_size(self) -> int:
        return len(self.val_patients)

    @property
    def train_sepsis_rate(self) -> float:
        return self.train_sepsis_count / self.train_size if self.train_size > 0 else 0

    @property
    def val_sepsis_rate(self) -> float:
        return self.val_sepsis_count / self.val_size if self.val_size > 0 else 0

    def __repr__(self) -> str:
        return (
            f"CVSplit(fold={self.fold_idx}, "
            f"train={self.train_size} ({self.train_sepsis_rate:.1%} sepsis), "
            f"val={self.val_size} ({self.val_sepsis_rate:.1%} sepsis))"
        )


class PatientKFold:
    """
    Patient-level stratified K-Fold cross-validation.

    This ensures:
    1. No data leakage: All hours from a patient stay together
    2. Balanced folds: Each fold has similar sepsis prevalence
    3. Reproducibility: Same random seed gives same splits

    Rationale for 5-fold default:
    - 3 folds: Too few for reliable variance estimates
    - 5 folds: Good balance of reliability and training time
    - 10 folds: Diminishing returns, 2x training time vs 5-fold

    Example:
        >>> kfold = PatientKFold(n_splits=5, random_state=42)
        >>> for split in kfold.split(patient_ids, sepsis_labels):
        ...     train_patients = split.train_patients
        ...     val_patients = split.val_patients
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize patient-level K-Fold.

        Args:
            n_splits: Number of folds (default: 5)
            shuffle: Whether to shuffle patients before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        # Use sklearn's StratifiedKFold under the hood
        self._skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    def split(
        self,
        patient_ids: List[str],
        sepsis_labels: List[bool],
    ) -> Iterator[CVSplit]:
        """
        Generate cross-validation splits.

        Args:
            patient_ids: List of unique patient identifiers
            sepsis_labels: Boolean list indicating if each patient has sepsis

        Yields:
            CVSplit objects for each fold

        Raises:
            ValueError: If inputs have mismatched lengths
        """
        if len(patient_ids) != len(sepsis_labels):
            raise ValueError(
                f"patient_ids ({len(patient_ids)}) and sepsis_labels "
                f"({len(sepsis_labels)}) must have same length"
            )

        patient_ids = np.array(patient_ids)
        sepsis_labels = np.array(sepsis_labels, dtype=int)

        logger.info(f"Creating {self.n_splits}-fold patient-level splits")
        logger.info(f"  Total patients: {len(patient_ids)}")
        logger.info(f"  Sepsis patients: {sepsis_labels.sum()} ({sepsis_labels.mean():.1%})")

        for fold_idx, (train_idx, val_idx) in enumerate(
            self._skf.split(patient_ids, sepsis_labels)
        ):
            split = CVSplit(
                fold_idx=fold_idx,
                train_patients=patient_ids[train_idx].tolist(),
                val_patients=patient_ids[val_idx].tolist(),
                train_sepsis_count=int(sepsis_labels[train_idx].sum()),
                val_sepsis_count=int(sepsis_labels[val_idx].sum()),
            )
            logger.info(f"  Fold {fold_idx}: {split}")
            yield split


def create_cv_splits(
    data_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
    max_patients: Optional[int] = None,
) -> List[CVSplit]:
    """
    Create cross-validation splits from PhysioNet data directory.

    This function:
    1. Scans data directory for patient PSV files
    2. Determines sepsis status for each patient
    3. Creates stratified patient-level splits

    Args:
        data_dir: Path to PhysioNet data (contains training_setA/, training_setB/)
        n_splits: Number of CV folds
        random_state: Random seed
        max_patients: Limit number of patients (for testing)

    Returns:
        List of CVSplit objects

    Example:
        >>> splits = create_cv_splits(Path("data/physionet"), n_splits=5)
        >>> for split in splits:
        ...     print(f"Fold {split.fold_idx}: {split.train_size} train, {split.val_size} val")
    """
    data_dir = Path(data_dir)

    # Find all patient files
    patient_files = []
    for set_dir in ["training_setA", "training_setB"]:
        set_path = data_dir / set_dir
        if set_path.exists():
            patient_files.extend(set_path.glob("*.psv"))

    if not patient_files:
        raise FileNotFoundError(
            f"No PSV files found in {data_dir}. "
            "Run: python -m training.data.download_physionet --output-dir data/physionet"
        )

    # Limit patients if requested
    if max_patients is not None:
        patient_files = patient_files[:max_patients]

    logger.info(f"Found {len(patient_files)} patient files")

    # Determine sepsis status for each patient
    patient_ids = []
    sepsis_labels = []

    for psv_path in patient_files:
        patient_id = psv_path.stem  # e.g., "p000001"
        patient_ids.append(patient_id)

        # Read last column (SepsisLabel) to check if patient ever develops sepsis
        try:
            with open(psv_path) as f:
                # Skip header
                next(f)
                has_sepsis = False
                for line in f:
                    # SepsisLabel is last column
                    if line.strip().endswith("|1"):
                        has_sepsis = True
                        break
                sepsis_labels.append(has_sepsis)
        except Exception as e:
            logger.warning(f"Error reading {psv_path}: {e}")
            sepsis_labels.append(False)

    # Create splits
    kfold = PatientKFold(n_splits=n_splits, random_state=random_state)
    splits = list(kfold.split(patient_ids, sepsis_labels))

    return splits


def get_patient_file_mapping(data_dir: Path) -> dict:
    """
    Create mapping from patient ID to file path.

    Args:
        data_dir: Path to PhysioNet data directory

    Returns:
        Dictionary mapping patient_id -> Path to PSV file
    """
    data_dir = Path(data_dir)
    mapping = {}

    for set_dir in ["training_setA", "training_setB"]:
        set_path = data_dir / set_dir
        if set_path.exists():
            for psv_path in set_path.glob("*.psv"):
                patient_id = psv_path.stem
                mapping[patient_id] = psv_path

    return mapping
