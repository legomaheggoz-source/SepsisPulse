"""Prediction Service - Full pipeline for sepsis prediction.

This service orchestrates the complete prediction pipeline:
1. Load patient data from PSV files
2. Engineer features for XGBoost (lags, rolling stats)
3. Prepare sequences for TFT
4. Run predictions through all models
5. Calculate evaluation metrics

Integration Notes:
- Uses src/data/loader.py for data loading
- Uses src/data/feature_engineering.py for XGBoost features
- Uses src/evaluation/metrics.py for performance metrics
- Uses ModelService for model access
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from src.data.loader import (
    load_patient,
    load_dataset,
    get_sample_subset,
    VITAL_COLUMNS,
    LAB_COLUMNS,
    DEMOGRAPHIC_COLUMNS,
)
from src.evaluation.metrics import (
    compute_roc_auc,
    compute_classification_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class PatientPrediction:
    """Prediction results for a single patient."""
    patient_id: str
    has_sepsis: bool
    sepsis_onset_hour: Optional[int]
    hours_of_data: int
    qsofa_score: int
    qsofa_risk: str
    xgboost_probability: float
    xgboost_risk: str
    tft_probability: float
    tft_risk: str


@dataclass
class DatasetMetrics:
    """Evaluation metrics for a dataset."""
    total_patients: int
    sepsis_patients: int
    sepsis_rate: float
    qsofa_metrics: Dict[str, float]
    xgboost_metrics: Dict[str, float]
    tft_metrics: Dict[str, float]


class PredictionService:
    """
    Service for running the complete sepsis prediction pipeline.

    This service handles:
    - Loading and preprocessing patient data
    - Feature engineering for different model types
    - Running predictions across all models
    - Calculating evaluation metrics

    Example:
        >>> from src.services import ModelService, PredictionService
        >>> model_service = ModelService()
        >>> model_service.load_all_models()
        >>> pred_service = PredictionService(model_service)
        >>> results = pred_service.predict_patient("data/sample/p00001.psv")
        >>> metrics = pred_service.evaluate_dataset("data/sample/")
    """

    # Feature columns for models (excluding SepsisLabel)
    FEATURE_COLUMNS = VITAL_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS[:-1]  # Exclude ICULOS

    def __init__(self, model_service):
        """
        Initialize the prediction service.

        Args:
            model_service: Initialized ModelService instance
        """
        self.model_service = model_service
        self._feature_names = None

    def load_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Load bundled sample patient data."""
        return get_sample_subset()

    def load_patient_data(self, file_path: str) -> pd.DataFrame:
        """Load a single patient's data from PSV file."""
        return load_patient(file_path)

    def load_dataset_data(
        self,
        data_dir: str,
        max_patients: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple patients from a directory."""
        return load_dataset(data_dir, max_patients=max_patients)

    def _compute_qsofa(self, patient_data: pd.DataFrame) -> Tuple[int, str]:
        """
        Compute qSOFA score for a patient.

        qSOFA criteria:
        - Respiratory rate >= 22/min
        - Altered mentation (GCS < 15, not available in data)
        - Systolic BP <= 100 mmHg

        Returns:
            Tuple of (score, risk_level)
        """
        # Get last available values
        resp = patient_data["Resp"].dropna()
        sbp = patient_data["SBP"].dropna()

        score = 0

        if len(resp) > 0 and resp.iloc[-1] >= 22:
            score += 1

        if len(sbp) > 0 and sbp.iloc[-1] <= 100:
            score += 1

        # GCS not available in PhysioNet data, so max score is 2
        risk = "Low" if score < 2 else "High"

        return score, risk

    def _prepare_xgboost_features(self, patient_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for XGBoost model.

        Creates lag features, rolling statistics, and delta features
        from raw patient data.
        """
        try:
            from src.data.feature_engineering import engineer_features
            return engineer_features(patient_data)
        except ImportError:
            # Fallback: use raw features with forward fill
            features = patient_data[self.FEATURE_COLUMNS].copy()
            features = features.ffill().bfill().fillna(0)
            return features

    def _prepare_tft_sequence(
        self,
        patient_data: pd.DataFrame,
        max_seq_len: int = 72
    ) -> np.ndarray:
        """
        Prepare sequence data for TFT model.

        Args:
            patient_data: Raw patient DataFrame
            max_seq_len: Maximum sequence length (default 72 hours)

        Returns:
            Array of shape (1, seq_len, n_features)
        """
        # Select feature columns (exclude SepsisLabel and ICULOS)
        feature_cols = VITAL_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS[:-1]

        # Get features
        features = patient_data[feature_cols].copy()

        # Forward fill then backward fill missing values
        features = features.ffill().bfill().fillna(0)

        # Truncate or pad to max_seq_len
        if len(features) > max_seq_len:
            features = features.iloc[-max_seq_len:]  # Take most recent
        elif len(features) < max_seq_len:
            # Pad at the beginning with zeros
            padding = pd.DataFrame(
                np.zeros((max_seq_len - len(features), len(feature_cols))),
                columns=feature_cols
            )
            features = pd.concat([padding, features], ignore_index=True)

        # Convert to numpy and add batch dimension
        sequence = features.values.astype(np.float32)
        return sequence[np.newaxis, :, :]  # (1, seq_len, features)

    def _risk_level(self, probability: float) -> str:
        """Convert probability to risk level string."""
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Medium"
        else:
            return "High"

    def predict_patient(self, patient_data: pd.DataFrame, patient_id: str = "unknown") -> PatientPrediction:
        """
        Run all models on a single patient.

        Args:
            patient_data: DataFrame with patient's hourly data
            patient_id: Patient identifier

        Returns:
            PatientPrediction with results from all models
        """
        # Check for sepsis
        has_sepsis = patient_data["SepsisLabel"].sum() > 0
        sepsis_onset = None
        if has_sepsis:
            sepsis_onset = int(patient_data["SepsisLabel"].idxmax())

        # qSOFA
        qsofa_score, qsofa_risk = self._compute_qsofa(patient_data)

        # XGBoost
        try:
            xgb_features = self._prepare_xgboost_features(patient_data)
            xgb_model = self.model_service.get_model("XGBoost-TS")
            if xgb_model is not None:
                # Predict on last row (current state)
                xgb_proba = xgb_model.predict_proba(xgb_features.iloc[[-1]])
                xgb_prob = float(xgb_proba[0, 1]) if xgb_proba.ndim > 1 else float(xgb_proba[0])
            else:
                xgb_prob = 0.5
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
            xgb_prob = 0.5

        # TFT
        try:
            tft_sequence = self._prepare_tft_sequence(patient_data)
            tft_model = self.model_service.get_model("TFT-Lite")
            if tft_model is not None:
                tft_prob = float(tft_model.predict_proba(tft_sequence))
            else:
                tft_prob = 0.5
        except Exception as e:
            logger.warning(f"TFT prediction failed: {e}")
            tft_prob = 0.5

        return PatientPrediction(
            patient_id=patient_id,
            has_sepsis=has_sepsis,
            sepsis_onset_hour=sepsis_onset,
            hours_of_data=len(patient_data),
            qsofa_score=qsofa_score,
            qsofa_risk=qsofa_risk,
            xgboost_probability=xgb_prob,
            xgboost_risk=self._risk_level(xgb_prob),
            tft_probability=tft_prob,
            tft_risk=self._risk_level(tft_prob),
        )

    def predict_dataset(
        self,
        dataset: Dict[str, pd.DataFrame],
        threshold: float = 0.5
    ) -> Tuple[List[PatientPrediction], DatasetMetrics]:
        """
        Run predictions on a dataset and compute metrics.

        Args:
            dataset: Dictionary mapping patient IDs to DataFrames
            threshold: Classification threshold

        Returns:
            Tuple of (list of predictions, dataset metrics)
        """
        predictions = []
        y_true = []
        qsofa_scores = []
        xgb_probas = []
        tft_probas = []

        for patient_id, patient_data in dataset.items():
            pred = self.predict_patient(patient_data, patient_id)
            predictions.append(pred)

            y_true.append(1 if pred.has_sepsis else 0)
            qsofa_scores.append(pred.qsofa_score)
            xgb_probas.append(pred.xgboost_probability)
            tft_probas.append(pred.tft_probability)

        # Convert to arrays
        y_true = np.array(y_true)
        qsofa_scores = np.array(qsofa_scores)
        xgb_probas = np.array(xgb_probas)
        tft_probas = np.array(tft_probas)

        # Compute metrics
        qsofa_metrics = self._compute_model_metrics(y_true, qsofa_scores, threshold=2)
        xgb_metrics = self._compute_model_metrics(y_true, xgb_probas, threshold=threshold)
        tft_metrics = self._compute_model_metrics(y_true, tft_probas, threshold=threshold)

        sepsis_count = int(y_true.sum())
        metrics = DatasetMetrics(
            total_patients=len(predictions),
            sepsis_patients=sepsis_count,
            sepsis_rate=sepsis_count / len(predictions) if predictions else 0,
            qsofa_metrics=qsofa_metrics,
            xgboost_metrics=xgb_metrics,
            tft_metrics=tft_metrics,
        )

        return predictions, metrics

    def _compute_model_metrics(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute evaluation metrics for a model."""
        metrics = {}

        # ROC AUC
        try:
            auc, fpr, tpr = compute_roc_auc(y_true, y_score)
            metrics["auroc"] = auc
            metrics["fpr"] = fpr.tolist()
            metrics["tpr"] = tpr.tolist()
        except Exception:
            metrics["auroc"] = 0.5

        # Classification metrics
        y_pred = (y_score >= threshold).astype(int)
        try:
            class_metrics = compute_classification_metrics(y_true, y_pred)
            metrics.update(class_metrics)
        except Exception:
            pass

        return metrics

    def get_patient_timeline(
        self,
        patient_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get timeline data for patient visualization.

        Returns vitals and model predictions over time.
        """
        timeline = {
            "hours": list(range(len(patient_data))),
            "vitals": {},
            "predictions": {
                "xgboost": [],
                "tft": [],
            },
            "sepsis_label": patient_data["SepsisLabel"].tolist(),
        }

        # Vitals
        for col in VITAL_COLUMNS:
            timeline["vitals"][col] = patient_data[col].tolist()

        # Could compute rolling predictions here if needed

        return timeline
