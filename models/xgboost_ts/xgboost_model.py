"""
XGBoost Time-Series Model for Sepsis Prediction.

This module implements an XGBoost-based model wrapper for sepsis prediction
using engineered time-series features. The model supports loading pre-trained
weights for inference and includes a demo mode for use without trained weights.

The model expects engineered features as input (e.g., rolling statistics,
lag features, trend indicators) and outputs binary predictions or
probability scores for sepsis risk.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = logging.getLogger(__name__)

# Default path for pre-trained model weights
DEFAULT_WEIGHTS_PATH = str(
    Path(__file__).parent / "weights" / "xgb_sepsis_v1.json"
)


class XGBoostTSModel:
    """
    XGBoost Time-Series model for sepsis prediction.

    This class provides a wrapper around XGBoost for sepsis prediction
    using engineered time-series features. It supports loading pre-trained
    weights for inference and includes a demo mode that generates random
    predictions when no weights are available.

    Attributes:
        model: The underlying XGBoost Booster model (None in demo mode).
        weights_path: Path to the loaded weights file.
        is_demo_mode: Whether the model is running in demo mode.
        feature_names: List of feature names used by the model.

    Example:
        >>> model = XGBoostTSModel(weights_path="path/to/weights.json")
        >>> probabilities = model.predict_proba(feature_df)
        >>> predictions = model.predict(feature_df, threshold=0.5)
        >>> importance = model.get_feature_importance()
    """

    def __init__(self, weights_path: Optional[str] = None):
        """
        Initialize the XGBoost model.

        Args:
            weights_path: Path to pre-trained model weights in JSON format.
                         If None, attempts to load from DEFAULT_WEIGHTS_PATH.
                         If weights file doesn't exist, runs in demo mode.

        Raises:
            ImportError: If xgboost is not installed.
        """
        self.model: Optional["xgb.Booster"] = None
        self.weights_path: Optional[str] = None
        self.is_demo_mode: bool = False
        self.feature_names: Optional[List[str]] = None
        self._random_state = np.random.RandomState(42)

        if not HAS_XGBOOST:
            logger.warning(
                "XGBoost not installed. Running in demo mode with random predictions. "
                "Install with: pip install xgboost"
            )
            self.is_demo_mode = True
            return

        # Determine which weights path to use
        path_to_load = weights_path if weights_path is not None else DEFAULT_WEIGHTS_PATH

        # Try to load weights
        if path_to_load and os.path.exists(path_to_load):
            try:
                self.load_weights(path_to_load)
                logger.info(f"Loaded XGBoost model from {path_to_load}")
            except Exception as e:
                logger.warning(
                    f"Failed to load weights from {path_to_load}: {e}. "
                    "Running in demo mode."
                )
                self.is_demo_mode = True
        else:
            logger.info(
                f"No weights file found at {path_to_load}. "
                "Running in demo mode with random predictions."
            )
            self.is_demo_mode = True

    def load_weights(self, path: str) -> None:
        """
        Load model weights from a JSON file.

        Args:
            path: Path to the JSON file containing model weights.

        Raises:
            FileNotFoundError: If the weights file doesn't exist.
            xgboost.core.XGBoostError: If the file format is invalid.
            ImportError: If xgboost is not installed.
        """
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found: {path}")

        self.model = xgb.Booster()
        self.model.load_model(path)
        self.weights_path = path
        self.is_demo_mode = False

        # Try to extract feature names from the model
        try:
            self.feature_names = self.model.feature_names
        except AttributeError:
            self.feature_names = None

        logger.info(f"Successfully loaded model weights from {path}")

    def save_weights(self, path: str) -> None:
        """
        Save model weights to a JSON file.

        Args:
            path: Path where the model weights will be saved.

        Raises:
            ValueError: If the model is in demo mode (no weights to save).
            ImportError: If xgboost is not installed.
        """
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )

        if self.is_demo_mode or self.model is None:
            raise ValueError(
                "Cannot save weights: model is in demo mode with no trained weights."
            )

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        self.model.save_model(path)
        self.weights_path = path
        logger.info(f"Successfully saved model weights to {path}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions for sepsis.

        Args:
            X: Input DataFrame with engineered features.
               Each row represents a patient observation.

        Returns:
            numpy array of shape (n_samples, 2) with probabilities
            for class 0 (no sepsis) and class 1 (sepsis).
            Format: [[P(class=0), P(class=1)], ...]

        Note:
            In demo mode, returns random probabilities seeded for reproducibility.
        """
        if len(X) == 0:
            return np.array([]).reshape(0, 2)

        if self.is_demo_mode:
            # Demo mode: generate pseudo-random but reproducible predictions
            # Use hash of input features to create deterministic randomness
            prob_positive = self._generate_demo_predictions(X)
            prob_negative = 1.0 - prob_positive
            return np.column_stack([prob_negative, prob_positive])

        # Convert DataFrame to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(X)

        # Get raw predictions (probabilities for positive class)
        prob_positive = self.model.predict(dmatrix)

        # Ensure 1D array
        if prob_positive.ndim > 1:
            prob_positive = prob_positive.flatten()

        prob_negative = 1.0 - prob_positive

        return np.column_stack([prob_negative, prob_positive])

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Generate binary predictions for sepsis.

        Args:
            X: Input DataFrame with engineered features.
               Each row represents a patient observation.
            threshold: Decision threshold for classification.
                      Predictions with P(sepsis) >= threshold are classified as 1.
                      Default is 0.5.

        Returns:
            numpy array of binary predictions (0 or 1) for each row.

        Raises:
            ValueError: If threshold is not between 0 and 1.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        if len(X) == 0:
            return np.array([], dtype=np.int32)

        probabilities = self.predict_proba(X)
        prob_positive = probabilities[:, 1]

        return (prob_positive >= threshold).astype(np.int32)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores from the model.

        Returns:
            pandas Series mapping feature names to importance scores,
            sorted in descending order by importance.

        Note:
            In demo mode, returns an empty Series.
            Importance type is 'gain' (average gain of splits using the feature).
        """
        if self.is_demo_mode or self.model is None:
            logger.warning(
                "Cannot get feature importance: model is in demo mode."
            )
            return pd.Series(dtype=float)

        try:
            importance_dict = self.model.get_score(importance_type='gain')

            if not importance_dict:
                # Try 'weight' if 'gain' returns empty
                importance_dict = self.model.get_score(importance_type='weight')

            importance = pd.Series(importance_dict)
            importance = importance.sort_values(ascending=False)

            return importance

        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
            return pd.Series(dtype=float)

    def _generate_demo_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate demo predictions using a simple heuristic.

        This creates semi-realistic predictions based on input data
        characteristics for demonstration purposes when no trained
        model is available.

        Args:
            X: Input DataFrame with features.

        Returns:
            numpy array of probabilities for the positive class.
        """
        n_samples = len(X)

        # Use a combination of:
        # 1. Random baseline
        # 2. Data-dependent component based on feature values

        # Random component (seeded for reproducibility within session)
        random_component = self._random_state.beta(2, 5, size=n_samples)

        # Data-dependent component: higher values in some columns -> higher risk
        data_component = np.zeros(n_samples)

        if len(X.columns) > 0:
            # Use normalized mean of numeric columns as a signal
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Normalize each column to 0-1 range and take mean
                normalized = X[numeric_cols].apply(
                    lambda col: (col - col.min()) / (col.max() - col.min() + 1e-10)
                    if col.max() != col.min() else 0.5
                )
                data_component = normalized.mean(axis=1).values
                # Handle NaN values
                data_component = np.nan_to_num(data_component, nan=0.3)

        # Combine components: 70% random, 30% data-dependent
        probabilities = 0.7 * random_component + 0.3 * data_component

        # Clip to valid probability range
        probabilities = np.clip(probabilities, 0.0, 1.0)

        return probabilities

    def get_model_info(self) -> Dict[str, Union[str, bool, int, None]]:
        """
        Get information about the current model state.

        Returns:
            Dictionary containing:
                - is_demo_mode: Whether running in demo mode
                - weights_path: Path to loaded weights (if any)
                - n_features: Number of features (if available)
                - xgboost_installed: Whether XGBoost is installed
        """
        info = {
            "is_demo_mode": self.is_demo_mode,
            "weights_path": self.weights_path,
            "n_features": None,
            "xgboost_installed": HAS_XGBOOST,
        }

        if self.feature_names is not None:
            info["n_features"] = len(self.feature_names)
        elif self.model is not None and HAS_XGBOOST:
            try:
                info["n_features"] = self.model.num_features()
            except Exception:
                pass

        return info

    def __repr__(self) -> str:
        """Return string representation of the model."""
        mode = "demo" if self.is_demo_mode else "trained"
        weights_info = f", weights='{self.weights_path}'" if self.weights_path else ""
        return f"XGBoostTSModel(mode={mode}{weights_info})"
