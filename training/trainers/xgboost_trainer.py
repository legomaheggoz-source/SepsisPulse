"""
XGBoost-TS Trainer with time-series feature engineering.

Design Decisions:

1. Feature Engineering Strategy:
   - Lag features: [1, 3, 6] hours capture temporal dynamics
   - Rolling stats: [3, 6, 12] hour windows for trends
   - Delta features: Rate of change detection
   - Results in ~200-700 features from 41 raw variables

2. Training Strategy:
   - Native XGBoost API (not sklearn wrapper) for GPU support
   - Early stopping within XGBoost rounds
   - Scale_pos_weight for class imbalance

3. Evaluation:
   - PhysioNet Clinical Utility Score as primary metric
   - Also tracks AUC-ROC, AUPRC, F1, sensitivity, specificity
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

from training.config.xgboost_config import XGBoostConfig
from training.trainers.base_trainer import BaseTrainer
from training.data.dataset import load_patient_psv, impute_features, FEATURE_COLS
from training.data.cross_validation import get_patient_file_mapping

# Import evaluation utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class XGBoostTrainer(BaseTrainer):
    """
    XGBoost trainer with time-series feature engineering.

    Features:
    - Lag features for temporal patterns
    - Rolling statistics for trend detection
    - GPU acceleration (RTX 4090 optimized)
    - PhysioNet Clinical Utility Score optimization

    Example:
        >>> config = XGBoostConfig(data_dir="data/physionet")
        >>> trainer = XGBoostTrainer(config)
        >>> results = trainer.train()
        >>> best_model = trainer.get_best_model()
    """

    def __init__(
        self,
        config: XGBoostConfig,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize XGBoost trainer.

        Args:
            config: XGBoost training configuration
            output_dir: Output directory (overrides config)
        """
        super().__init__(config, output_dir)
        self.config: XGBoostConfig = config

        # File mapping for data loading
        self._file_mapping: Dict[str, Path] = {}

        # Feature names (set during training)
        self.feature_names: Optional[List[str]] = None

        # Normalization stats
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def _load_data(
        self,
        train_patients: List[str],
        val_patients: List[str],
    ) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        """
        Load and preprocess data for XGBoost.

        Creates time-series features (lags, rolling stats, deltas)
        and returns XGBoost DMatrix objects.

        Args:
            train_patients: Training patient IDs
            val_patients: Validation patient IDs

        Returns:
            Tuple of (train_dmatrix, val_dmatrix)
        """
        logger.info("Loading and preprocessing data...")

        # Build file mapping if needed
        if not self._file_mapping:
            self._file_mapping = get_patient_file_mapping(self.config.data_dir)

        # Load and process all patients
        train_features, train_labels = self._process_patients(train_patients, fit_scaler=True)
        val_features, val_labels = self._process_patients(val_patients, fit_scaler=False)

        logger.info(f"  Train: {train_features.shape[0]} samples, {train_features.shape[1]} features")
        logger.info(f"  Val: {val_features.shape[0]} samples, {val_features.shape[1]} features")
        logger.info(f"  Train sepsis rate: {train_labels.mean():.2%}")
        logger.info(f"  Val sepsis rate: {val_labels.mean():.2%}")

        # Compute class weight if needed
        if self.config.scale_pos_weight is None:
            n_pos = train_labels.sum()
            n_neg = len(train_labels) - n_pos
            self.config.scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            logger.info(f"  Computed scale_pos_weight: {self.config.scale_pos_weight:.2f}")

        # Create DMatrix objects
        dtrain = xgb.DMatrix(
            train_features,
            label=train_labels,
            feature_names=self.feature_names,
        )
        dval = xgb.DMatrix(
            val_features,
            label=val_labels,
            feature_names=self.feature_names,
        )

        return dtrain, dval

    def _process_patients(
        self,
        patient_ids: List[str],
        fit_scaler: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process patient data with feature engineering.

        Args:
            patient_ids: List of patient IDs
            fit_scaler: Whether to fit normalization scaler

        Returns:
            Tuple of (features, labels) arrays
        """
        all_features = []
        all_labels = []

        for patient_id in patient_ids:
            if patient_id not in self._file_mapping:
                continue

            # Load patient data
            df = load_patient_psv(self._file_mapping[patient_id])
            df = impute_features(df)

            # Extract raw features and labels
            raw_features = df[FEATURE_COLS].values
            labels = df["SepsisLabel"].values

            # Create time-series features
            ts_features = self._create_ts_features(raw_features)

            all_features.append(ts_features)
            all_labels.append(labels)

        # Stack all data
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        # Normalize
        if fit_scaler:
            self._mean = np.nanmean(features, axis=0)
            self._std = np.nanstd(features, axis=0)
            self._std[self._std < 1e-8] = 1.0

        if self._mean is not None:
            features = (features - self._mean) / self._std

        # Handle any remaining NaNs
        features = np.nan_to_num(features, nan=0.0)

        return features, labels

    def _create_ts_features(self, raw_features: np.ndarray) -> np.ndarray:
        """
        Create time-series features from raw patient data.

        Features created:
        - Lag features: value at t-1, t-3, t-6 hours
        - Rolling mean/std: 3, 6, 12 hour windows
        - Delta features: change from previous hour

        Args:
            raw_features: [seq_len, n_features] raw feature matrix

        Returns:
            [seq_len, n_engineered_features] engineered features
        """
        n_hours, n_raw = raw_features.shape
        feature_list = [raw_features]  # Start with raw features
        feature_names = list(FEATURE_COLS)

        # Lag features
        for lag in self.config.lag_hours:
            lagged = np.roll(raw_features, lag, axis=0)
            lagged[:lag] = raw_features[0]  # Fill with first value
            feature_list.append(lagged)
            feature_names.extend([f"{col}_lag{lag}" for col in FEATURE_COLS])

        # Rolling statistics
        if self.config.include_rolling_stats:
            for window in self.config.rolling_hours:
                # Rolling mean
                rolling_mean = self._rolling_stat(raw_features, window, "mean")
                feature_list.append(rolling_mean)
                feature_names.extend([f"{col}_mean{window}" for col in FEATURE_COLS])

                # Rolling std
                rolling_std = self._rolling_stat(raw_features, window, "std")
                feature_list.append(rolling_std)
                feature_names.extend([f"{col}_std{window}" for col in FEATURE_COLS])

        # Delta features (change from previous hour)
        if self.config.include_deltas:
            delta = np.diff(raw_features, axis=0, prepend=raw_features[:1])
            feature_list.append(delta)
            feature_names.extend([f"{col}_delta" for col in FEATURE_COLS])

        # Store feature names (only once)
        if self.feature_names is None:
            self.feature_names = feature_names

        return np.hstack(feature_list)

    def _rolling_stat(
        self,
        data: np.ndarray,
        window: int,
        stat: str,
    ) -> np.ndarray:
        """Compute rolling statistic."""
        n_hours, n_features = data.shape
        result = np.zeros_like(data)

        for t in range(n_hours):
            start = max(0, t - window + 1)
            window_data = data[start:t + 1]
            if stat == "mean":
                result[t] = np.nanmean(window_data, axis=0)
            elif stat == "std":
                result[t] = np.nanstd(window_data, axis=0)
            elif stat == "min":
                result[t] = np.nanmin(window_data, axis=0)
            elif stat == "max":
                result[t] = np.nanmax(window_data, axis=0)

        return result

    def _train_fold(
        self,
        train_data: xgb.DMatrix,
        val_data: xgb.DMatrix,
        fold: int,
    ) -> Tuple[xgb.Booster, Dict[str, float]]:
        """
        Train XGBoost model on a single fold.

        Args:
            train_data: Training DMatrix
            val_data: Validation DMatrix
            fold: Fold index

        Returns:
            Tuple of (trained_booster, metrics_dict)
        """
        logger.info(f"Training XGBoost fold {fold}...")

        # Get parameters
        params = self.config.get_xgb_params()

        # Early stopping callback
        evals = [(train_data, "train"), (val_data, "val")]
        evals_result = {}

        # Train
        model = xgb.train(
            params,
            train_data,
            num_boost_round=self.config.n_estimators,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=self.config.xgb_early_stopping_rounds,
            verbose_eval=50,
        )

        # Evaluate on validation set
        val_preds = model.predict(val_data)
        val_labels = val_data.get_label()
        metrics = self._compute_metrics(val_preds, val_labels)

        # Log metrics
        for name, value in metrics.items():
            self.logger.log_epoch(
                epoch=model.best_iteration,
                fold=fold,
                train_loss=evals_result["train"]["logloss"][-1],
                val_loss=evals_result["val"]["logloss"][-1],
                metrics={name: value},
            )

        # Save checkpoint
        self.checkpoint_manager.save_xgboost(
            model=model,
            epoch=model.best_iteration,
            fold=fold,
            metric_value=metrics[self.config.primary_metric],
            config=self.config.to_dict(),
            feature_names=self.feature_names,
        )

        logger.info(f"Fold {fold} complete: {metrics}")
        return model, metrics

    def _evaluate(
        self,
        model: xgb.Booster,
        data: xgb.DMatrix,
    ) -> Dict[str, float]:
        """Evaluate model on data."""
        preds = model.predict(data)
        labels = data.get_label()
        return self._compute_metrics(preds, labels)

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: Predicted probabilities
            labels: True labels
            threshold: Classification threshold

        Returns:
            Dictionary of metric names to values
        """
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
        )

        # Binary predictions
        binary_preds = (predictions >= threshold).astype(int)

        metrics = {}

        # AUC-ROC
        try:
            metrics["auroc"] = roc_auc_score(labels, predictions)
        except ValueError:
            metrics["auroc"] = 0.5

        # AUC-PR
        try:
            metrics["auprc"] = average_precision_score(labels, predictions)
        except ValueError:
            metrics["auprc"] = labels.mean()

        # Classification metrics
        metrics["f1"] = f1_score(labels, binary_preds, zero_division=0)
        metrics["precision"] = precision_score(labels, binary_preds, zero_division=0)
        metrics["recall"] = recall_score(labels, binary_preds, zero_division=0)
        metrics["sensitivity"] = metrics["recall"]

        # Specificity
        tn = ((binary_preds == 0) & (labels == 0)).sum()
        fp = ((binary_preds == 1) & (labels == 0)).sum()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Clinical Utility Score (simplified version for hour-level)
        # Full version requires patient-level predictions with onset times
        # This is an approximation based on sensitivity/specificity tradeoff
        metrics["utility"] = self._compute_approx_utility(
            predictions, labels, threshold
        )

        return metrics

    def _compute_approx_utility(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float,
    ) -> float:
        """
        Compute approximate clinical utility score.

        The full PhysioNet utility score requires patient-level
        predictions with onset times. This approximation uses
        the confusion matrix weights.

        Weights (from PhysioNet 2019):
        - TP (early detection): +1.0
        - TN: 0.0
        - FP: -0.05
        - FN: -2.0 (missed sepsis is costly)
        """
        binary_preds = (predictions >= threshold).astype(int)

        tp = ((binary_preds == 1) & (labels == 1)).sum()
        tn = ((binary_preds == 0) & (labels == 0)).sum()
        fp = ((binary_preds == 1) & (labels == 0)).sum()
        fn = ((binary_preds == 0) & (labels == 1)).sum()

        # Utility weights
        utility = (
            1.0 * tp +
            0.0 * tn +
            -0.05 * fp +
            -2.0 * fn
        )

        # Normalize to [0, 1] range approximately
        max_utility = tp + fn  # If we detected all sepsis perfectly
        min_utility = -0.05 * (tn + fp) - 2.0 * (tp + fn)

        if max_utility > min_utility:
            utility_normalized = (utility - min_utility) / (max_utility - min_utility)
        else:
            utility_normalized = 0.0

        return float(utility_normalized)

    def save_final_model(self, output_path: Optional[Path] = None) -> Path:
        """
        Save the best model for deployment.

        Args:
            output_path: Path to save model (default: models/xgboost_ts/weights/)

        Returns:
            Path to saved model
        """
        if output_path is None:
            output_path = Path("models/xgboost_ts/weights/xgb_sepsis_v1.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        best_model = self.get_best_model()
        if best_model is None:
            raise ValueError("No trained model available")

        best_model.save_model(str(output_path), "json")
        logger.info(f"Saved final model to {output_path} (JSON format)")

        # Save feature names
        features_path = output_path.with_suffix(".features.json")
        import json
        with open(features_path, "w") as f:
            json.dump(self.feature_names, f)
        logger.info(f"Saved feature names to {features_path}")

        return output_path
