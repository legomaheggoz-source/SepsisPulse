"""
qSOFA (Quick Sequential Organ Failure Assessment) Baseline Model.

This module implements the qSOFA scoring system from the Sepsis-3 guidelines
as a rule-based heuristic model for sepsis prediction. No training is required
as it uses fixed clinical criteria.

qSOFA Criteria (each worth 1 point, max score = 3):
    - Respiratory Rate >= 22 breaths/min
    - Systolic Blood Pressure <= 100 mmHg
    - Altered mentation (GCS < 15)

Reference:
    Singer M, et al. The Third International Consensus Definitions for Sepsis
    and Septic Shock (Sepsis-3). JAMA. 2016;315(8):801-810.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple


class QSOFAModel:
    """
    qSOFA (Quick SOFA) scoring model for sepsis prediction.

    This is a rule-based baseline model that implements the qSOFA criteria
    from the Sepsis-3 guidelines. It requires no training and provides
    interpretable predictions based on three clinical criteria.

    Attributes:
        threshold (int): Score threshold for positive prediction (default: 2).
        resp_rate_threshold (float): Respiratory rate threshold (default: 22).
        sbp_threshold (float): Systolic BP threshold (default: 100).
        gcs_threshold (float): GCS threshold for altered mentation (default: 15).

    Example:
        >>> model = QSOFAModel(threshold=2)
        >>> scores = model.calculate_score(patient_data)
        >>> predictions = model.predict(patient_data)
        >>> probabilities = model.predict_proba(patient_data)
    """

    # Column name mappings for flexibility
    RESP_RATE_COLUMNS = ['resp_rate', 'respiratory_rate', 'rr', 'Resp', 'RespRate']
    SBP_COLUMNS = ['sbp', 'systolic_bp', 'SBP', 'systolic', 'SysBP', 'NISysABP', 'SysABP']
    GCS_COLUMNS = ['gcs', 'GCS', 'glasgow_coma_scale', 'GcsTotal']

    def __init__(
        self,
        threshold: int = 2,
        resp_rate_threshold: float = 22.0,
        sbp_threshold: float = 100.0,
        gcs_threshold: float = 15.0
    ):
        """
        Initialize the qSOFA model.

        Args:
            threshold: Score threshold for positive prediction (0-3).
                      A score >= threshold results in a positive prediction.
                      Default is 2 as per Sepsis-3 guidelines.
            resp_rate_threshold: Respiratory rate threshold in breaths/min.
                                Default is 22.
            sbp_threshold: Systolic blood pressure threshold in mmHg.
                          Default is 100.
            gcs_threshold: GCS threshold for altered mentation.
                          Default is 15 (score < 15 indicates altered mentation).

        Raises:
            ValueError: If threshold is not between 0 and 3.
        """
        if not 0 <= threshold <= 3:
            raise ValueError(f"Threshold must be between 0 and 3, got {threshold}")

        self.threshold = threshold
        self.resp_rate_threshold = resp_rate_threshold
        self.sbp_threshold = sbp_threshold
        self.gcs_threshold = gcs_threshold

        # Track which columns were found during last calculation
        self._last_columns_found = {}

    def _find_column(
        self,
        df: pd.DataFrame,
        possible_names: List[str]
    ) -> Optional[str]:
        """
        Find the first matching column name from a list of possibilities.

        Args:
            df: Input DataFrame.
            possible_names: List of possible column names to search for.

        Returns:
            The first matching column name, or None if no match found.
        """
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    def _get_respiratory_criterion(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate respiratory rate criterion.

        Args:
            df: Input DataFrame with vital signs.

        Returns:
            Array of 0s and 1s indicating if criterion is met.
        """
        col = self._find_column(df, self.RESP_RATE_COLUMNS)
        self._last_columns_found['respiratory_rate'] = col

        if col is None:
            return np.zeros(len(df), dtype=np.int32)

        values = df[col].values
        # Handle NaN values - treat as criterion not met
        criterion = np.where(
            pd.isna(values),
            0,
            (values >= self.resp_rate_threshold).astype(np.int32)
        )
        return criterion

    def _get_sbp_criterion(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate systolic blood pressure criterion.

        Args:
            df: Input DataFrame with vital signs.

        Returns:
            Array of 0s and 1s indicating if criterion is met.
        """
        col = self._find_column(df, self.SBP_COLUMNS)
        self._last_columns_found['systolic_bp'] = col

        if col is None:
            return np.zeros(len(df), dtype=np.int32)

        values = df[col].values
        # Handle NaN values - treat as criterion not met
        criterion = np.where(
            pd.isna(values),
            0,
            (values <= self.sbp_threshold).astype(np.int32)
        )
        return criterion

    def _get_gcs_criterion(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate altered mentation (GCS) criterion.

        Args:
            df: Input DataFrame with vital signs.

        Returns:
            Array of 0s and 1s indicating if criterion is met.
        """
        col = self._find_column(df, self.GCS_COLUMNS)
        self._last_columns_found['gcs'] = col

        if col is None:
            return np.zeros(len(df), dtype=np.int32)

        values = df[col].values
        # Handle NaN values - treat as criterion not met
        criterion = np.where(
            pd.isna(values),
            0,
            (values < self.gcs_threshold).astype(np.int32)
        )
        return criterion

    def calculate_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate qSOFA score for each row in the DataFrame.

        The qSOFA score ranges from 0 to 3, with one point for each
        criterion met:
            - Respiratory Rate >= 22 breaths/min
            - Systolic BP <= 100 mmHg
            - Altered mentation (GCS < 15)

        Args:
            df: Input DataFrame containing vital signs data.
                Expected columns (flexible naming):
                - Respiratory rate: 'resp_rate', 'respiratory_rate', 'rr', etc.
                - Systolic BP: 'sbp', 'systolic_bp', 'SBP', etc.
                - GCS: 'gcs', 'GCS', 'glasgow_coma_scale', etc.

        Returns:
            numpy array of integer scores (0-3) for each row.

        Note:
            Missing columns are handled gracefully - the criterion
            contributes 0 points if its column is not found.
        """
        if len(df) == 0:
            return np.array([], dtype=np.int32)

        resp_criterion = self._get_respiratory_criterion(df)
        sbp_criterion = self._get_sbp_criterion(df)
        gcs_criterion = self._get_gcs_criterion(df)

        total_score = resp_criterion + sbp_criterion + gcs_criterion

        return total_score.astype(np.int32)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate binary predictions based on qSOFA score.

        A positive prediction (1) is made when the qSOFA score is
        greater than or equal to the threshold.

        Args:
            df: Input DataFrame containing vital signs data.

        Returns:
            numpy array of binary predictions (0 or 1) for each row.
        """
        scores = self.calculate_score(df)
        return (scores >= self.threshold).astype(np.int32)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate probability estimates based on qSOFA score.

        The probability is calculated as score/3, providing a simple
        linear mapping from the discrete score to a probability value.

        Args:
            df: Input DataFrame containing vital signs data.

        Returns:
            numpy array of shape (n_samples, 2) with probabilities
            for class 0 (no sepsis) and class 1 (sepsis).
            Format: [[P(class=0), P(class=1)], ...]
        """
        scores = self.calculate_score(df)
        prob_positive = scores / 3.0
        prob_negative = 1.0 - prob_positive

        return np.column_stack([prob_negative, prob_positive])

    def get_criteria_breakdown(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get individual criterion values for interpretability.

        Args:
            df: Input DataFrame containing vital signs data.

        Returns:
            Tuple of three arrays:
                - respiratory_criterion: 0/1 for each row
                - sbp_criterion: 0/1 for each row
                - gcs_criterion: 0/1 for each row
        """
        resp_criterion = self._get_respiratory_criterion(df)
        sbp_criterion = self._get_sbp_criterion(df)
        gcs_criterion = self._get_gcs_criterion(df)

        return resp_criterion, sbp_criterion, gcs_criterion

    def get_feature_importance(self) -> dict:
        """
        Return feature importance for the qSOFA model.

        Since qSOFA is a rule-based model with equal weights,
        each criterion has equal importance (1/3).

        Returns:
            Dictionary mapping criterion names to importance values.
        """
        return {
            'respiratory_rate': 1/3,
            'systolic_bp': 1/3,
            'gcs': 1/3
        }

    def get_column_mapping(self) -> dict:
        """
        Get the columns found during the last calculation.

        Returns:
            Dictionary mapping criterion names to the column names
            found in the data (or None if not found).
        """
        return self._last_columns_found.copy()

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f"QSOFAModel(threshold={self.threshold}, "
            f"resp_rate_threshold={self.resp_rate_threshold}, "
            f"sbp_threshold={self.sbp_threshold}, "
            f"gcs_threshold={self.gcs_threshold})"
        )
