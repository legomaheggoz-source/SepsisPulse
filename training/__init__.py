"""
SepsisPulse Training Module.

This module provides comprehensive training infrastructure for sepsis prediction models:
- XGBoost-TS: Gradient boosting with time-series feature engineering
- TFT-Lite: Lightweight Temporal Fusion Transformer

Key Components:
- config/: Training configuration dataclasses
- data/: Data loading, preprocessing, and cross-validation
- trainers/: Model-specific training loops
- optimization/: Hyperparameter search with Optuna
- callbacks/: Checkpointing, early stopping, logging
- scripts/: CLI entry points for training

Training Philosophy:
1. Patient-level cross-validation (no data leakage)
2. Clinical Utility Score as primary metric (not just AUC-ROC)
3. Cost-weighted loss for class imbalance (~7.3% sepsis)
4. Mixed precision training for GPU efficiency
"""

from training.config import TrainingConfig, XGBoostConfig, TFTConfig

__version__ = "1.0.0"
__all__ = ["TrainingConfig", "XGBoostConfig", "TFTConfig"]
