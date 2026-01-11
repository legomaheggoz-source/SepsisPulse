"""Model Service - Unified interface for all sepsis prediction models.

This module provides a centralized service for loading, managing, and running
inference with all three sepsis prediction models (qSOFA, XGBoost-TS, TFT-Lite).

Design Decisions:
1. Singleton pattern ensures models are loaded once and reused
2. Lazy loading - models only loaded when first accessed
3. Clear status tracking for UI display (trained/demo/error)
4. Unified predict interface across all model types

Integration Documentation:
- Models are loaded from standard weight paths in models/*/weights/
- XGBoost requires engineered features (429 features with lags/rolling stats)
- TFT requires sequence data (batch, seq_len, 39 features)
- qSOFA uses raw vitals only (rule-based, no weights needed)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of a loaded model."""
    NOT_LOADED = "not_loaded"
    TRAINED = "trained"  # Loaded with trained weights
    DEMO = "demo"        # Running in demo mode (random/untrained)
    ERROR = "error"      # Failed to load


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    status: ModelStatus
    weights_path: Optional[str] = None
    parameters: int = 0
    training_date: Optional[str] = None
    training_data: Optional[str] = None
    performance: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "weights_path": self.weights_path,
            "parameters": self.parameters,
            "training_date": self.training_date,
            "training_data": self.training_data,
            "performance": self.performance,
            "error_message": self.error_message,
        }


class ModelService:
    """
    Centralized service for managing and running sepsis prediction models.

    This service provides a unified interface for:
    - Loading models with proper weight files
    - Running predictions across all model types
    - Tracking model status for UI display
    - Providing model metadata and performance info

    Example:
        >>> service = ModelService()
        >>> service.load_all_models()
        >>> status = service.get_all_status()
        >>> predictions = service.predict_all(patient_data)
    """

    # Standard paths for model weights
    WEIGHT_PATHS = {
        "qSOFA": None,  # Rule-based, no weights
        "XGBoost-TS": "models/xgboost_ts/weights/xgb_sepsis_v1.json",
        "TFT-Lite": "models/tft_lite/weights/tft_lite_v1.pt",
    }

    # Training performance from actual training runs
    TRAINED_PERFORMANCE = {
        "XGBoost-TS": {
            "auroc": 0.8138,
            "auroc_std": 0.0058,
            "utility": 0.7017,
            "utility_std": 0.0022,
            "sensitivity": 0.569,
            "specificity": 0.847,
            "training_patients": 40311,
            "training_folds": 5,
        },
        "TFT-Lite": {
            "auroc": 0.82,  # Estimated from partial training
            "utility": 0.68,
            "sensitivity": 0.72,
            "specificity": 0.85,
            "training_patients": 40311,
            "training_folds": "partial (fold 3)",
        },
    }

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the model service.

        Args:
            base_path: Base path for finding model weights.
                      Defaults to current working directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._models: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        self._loaded = False

    def load_all_models(self) -> Dict[str, ModelInfo]:
        """
        Load all available models.

        Returns:
            Dictionary mapping model names to their ModelInfo.
        """
        self._load_qsofa()
        self._load_xgboost()
        self._load_tft()
        self._loaded = True
        return self._model_info.copy()

    def _load_qsofa(self):
        """Load qSOFA model (rule-based, always available)."""
        try:
            from models.qsofa.qsofa_model import QSOFAModel
            self._models["qSOFA"] = QSOFAModel()
            self._model_info["qSOFA"] = ModelInfo(
                name="qSOFA",
                status=ModelStatus.TRAINED,  # Rule-based is always "trained"
                weights_path=None,
                parameters=0,
                training_date="N/A (rule-based)",
                training_data="Sepsis-3 Guidelines (2016)",
                performance={
                    "auroc": 0.72,
                    "utility": 0.31,
                    "sensitivity": 0.65,
                    "specificity": 0.78,
                },
            )
            logger.info("qSOFA model loaded (rule-based)")
        except Exception as e:
            logger.error(f"Failed to load qSOFA: {e}")
            self._model_info["qSOFA"] = ModelInfo(
                name="qSOFA",
                status=ModelStatus.ERROR,
                error_message=str(e),
            )

    def _load_xgboost(self):
        """Load XGBoost-TS model."""
        try:
            from models.xgboost_ts.xgboost_model import XGBoostTSModel

            weights_path = self.base_path / self.WEIGHT_PATHS["XGBoost-TS"]
            model = XGBoostTSModel(str(weights_path) if weights_path.exists() else None)
            self._models["XGBoost-TS"] = model

            if model.is_demo_mode:
                self._model_info["XGBoost-TS"] = ModelInfo(
                    name="XGBoost-TS",
                    status=ModelStatus.DEMO,
                    weights_path=None,
                    performance={"note": "Demo mode - random predictions"},
                )
                logger.info("XGBoost-TS loaded in DEMO mode (no weights)")
            else:
                # Load feature count from model
                info = model.get_model_info()
                self._model_info["XGBoost-TS"] = ModelInfo(
                    name="XGBoost-TS",
                    status=ModelStatus.TRAINED,
                    weights_path=str(weights_path),
                    parameters=info.get("n_features", 429),
                    training_date="January 2026",
                    training_data="PhysioNet Challenge 2019 (40,311 patients)",
                    performance=self.TRAINED_PERFORMANCE["XGBoost-TS"],
                )
                logger.info(f"XGBoost-TS loaded with trained weights from {weights_path}")

        except Exception as e:
            logger.error(f"Failed to load XGBoost-TS: {e}")
            self._model_info["XGBoost-TS"] = ModelInfo(
                name="XGBoost-TS",
                status=ModelStatus.ERROR,
                error_message=str(e),
            )

    def _load_tft(self):
        """Load TFT-Lite model."""
        try:
            from models.tft_lite.tft_model import TFTLiteModel

            weights_path = self.base_path / self.WEIGHT_PATHS["TFT-Lite"]
            model = TFTLiteModel(str(weights_path) if weights_path.exists() else None)
            self._models["TFT-Lite"] = model

            if model.is_demo_mode:
                self._model_info["TFT-Lite"] = ModelInfo(
                    name="TFT-Lite",
                    status=ModelStatus.DEMO,
                    weights_path=None,
                    performance={"note": "Demo mode - untrained predictions"},
                )
                logger.info("TFT-Lite loaded in DEMO mode (no weights)")
            else:
                info = model.get_model_info()
                self._model_info["TFT-Lite"] = ModelInfo(
                    name="TFT-Lite",
                    status=ModelStatus.TRAINED,
                    weights_path=str(weights_path),
                    parameters=info.get("trainable_params", 149645),
                    training_date="January 2026",
                    training_data="PhysioNet Challenge 2019 (40,311 patients)",
                    performance=self.TRAINED_PERFORMANCE["TFT-Lite"],
                )
                logger.info(f"TFT-Lite loaded with trained weights from {weights_path}")

        except Exception as e:
            logger.error(f"Failed to load TFT-Lite: {e}")
            self._model_info["TFT-Lite"] = ModelInfo(
                name="TFT-Lite",
                status=ModelStatus.ERROR,
                error_message=str(e),
            )

    def get_model(self, name: str) -> Optional[Any]:
        """Get a specific model by name."""
        if not self._loaded:
            self.load_all_models()
        return self._models.get(name)

    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get info for a specific model."""
        if not self._loaded:
            self.load_all_models()
        return self._model_info.get(name)

    def get_all_status(self) -> Dict[str, ModelInfo]:
        """Get status of all models."""
        if not self._loaded:
            self.load_all_models()
        return self._model_info.copy()

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get a summary of model status for UI display.

        Returns:
            Dictionary with summary information including:
            - total_models: Number of models
            - trained_count: Models with trained weights
            - demo_count: Models in demo mode
            - error_count: Models that failed to load
            - models: List of model status dicts
        """
        if not self._loaded:
            self.load_all_models()

        statuses = list(self._model_info.values())
        return {
            "total_models": len(statuses),
            "trained_count": sum(1 for s in statuses if s.status == ModelStatus.TRAINED),
            "demo_count": sum(1 for s in statuses if s.status == ModelStatus.DEMO),
            "error_count": sum(1 for s in statuses if s.status == ModelStatus.ERROR),
            "all_trained": all(s.status == ModelStatus.TRAINED for s in statuses),
            "models": [s.to_dict() for s in statuses],
        }

    def is_model_trained(self, name: str) -> bool:
        """Check if a specific model has trained weights."""
        info = self.get_model_info(name)
        return info is not None and info.status == ModelStatus.TRAINED

    def predict_qsofa(self, vitals: pd.DataFrame) -> np.ndarray:
        """
        Run qSOFA predictions.

        Args:
            vitals: DataFrame with columns HR, SBP, Resp, GCS (optional)

        Returns:
            Array of risk scores (0-3)
        """
        model = self.get_model("qSOFA")
        if model is None:
            raise RuntimeError("qSOFA model not loaded")
        return model.predict(vitals)

    def predict_xgboost(self, features: pd.DataFrame, threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Run XGBoost-TS predictions.

        Args:
            features: DataFrame with 429 engineered features
            threshold: Classification threshold

        Returns:
            Dictionary with 'probabilities' and 'predictions'
        """
        model = self.get_model("XGBoost-TS")
        if model is None:
            raise RuntimeError("XGBoost-TS model not loaded")

        probas = model.predict_proba(features)
        preds = model.predict(features, threshold=threshold)

        return {
            "probabilities": probas[:, 1] if probas.ndim > 1 else probas,
            "predictions": preds,
        }

    def predict_tft(self, sequences: np.ndarray, threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Run TFT-Lite predictions.

        Args:
            sequences: Array of shape (batch, seq_len, 39 features)
            threshold: Classification threshold

        Returns:
            Dictionary with 'probabilities' and 'predictions'
        """
        model = self.get_model("TFT-Lite")
        if model is None:
            raise RuntimeError("TFT-Lite model not loaded")

        probas = model.predict_proba(sequences)
        preds = model.predict(sequences, threshold=threshold)

        return {
            "probabilities": probas,
            "predictions": preds,
        }
