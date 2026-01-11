"""
XGBoost-TS training configuration.

Design Decisions:

1. Feature Engineering Strategy:
   - Lag features [1, 3, 6 hours]: Capture short/medium/shift-length dynamics
   - Rolling windows [3, 6, 12 hours]: Detect trends and volatility
   - Delta features: Detect deterioration/improvement rates
   - Results in ~200-700 features from 41 raw variables

2. Hyperparameter Defaults:
   - max_depth=6: Balanced complexity, avoids overfitting
   - learning_rate=0.1: Standard for moderate-sized datasets
   - n_estimators=500: Sufficient with early stopping
   - scale_pos_weight: Auto-computed from class ratio (~14:1)

3. GPU Acceleration:
   - tree_method="hist": Fastest for large datasets
   - device="cuda" when available: RTX 4090 provides 10-50x speedup
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal

from training.config.base import TrainingConfig


@dataclass
class XGBoostConfig(TrainingConfig):
    """
    XGBoost-TS specific training configuration.

    Inherits all base config options plus XGBoost-specific hyperparameters.

    Feature Engineering Rationale:
    - lag_hours=[1, 3, 6]: Short-term (1h), medium (3h), shift-length (6h) dynamics
    - rolling_hours=[3, 6, 12]: Trend detection over different time windows
    - Deltas detect rapid deterioration that precedes sepsis onset

    Hyperparameter Rationale:
    - max_depth=6: Deep enough for feature interactions, shallow enough to generalize
    - learning_rate=0.1: Standard value, can lower for final training
    - n_estimators=500: Early stopping will find optimal count
    - scale_pos_weight=auto: Computed from training data class ratio
    """

    # Feature engineering
    # Rationale: These window sizes capture clinically meaningful time scales:
    # - 1 hour: Acute changes (new infection, treatment response)
    # - 3 hours: Short-term trends (shift-level assessment)
    # - 6 hours: Medium-term patterns (half-shift dynamics)
    # - 12 hours: Long-term trends (full-shift or circadian patterns)
    lag_hours: List[int] = field(default_factory=lambda: [1, 3, 6])
    rolling_hours: List[int] = field(default_factory=lambda: [3, 6, 12])
    include_deltas: bool = True
    include_rolling_stats: bool = True  # mean, std, min, max

    # XGBoost hyperparameters
    # Rationale: Conservative defaults that work well across datasets.
    # Optuna will optimize these for better performance.
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    # Class imbalance
    # Rationale: Auto-compute from data. Typical sepsis ratio is ~14:1 (non-sepsis:sepsis)
    # at hour level, ~13:1 at patient level.
    scale_pos_weight: Optional[float] = None  # Auto-compute if None

    # GPU settings
    # Rationale: RTX 4090 provides massive speedup for tree construction.
    # hist method is fastest for datasets > 10K samples.
    tree_method: Literal["auto", "hist", "approx", "exact"] = "hist"
    device: Literal["cpu", "cuda"] = "cuda"

    # Early stopping within XGBoost (separate from epoch-level early stopping)
    xgb_early_stopping_rounds: int = 50

    # Output format
    # Rationale: JSON is portable, HuggingFace compatible, human-readable.
    # Smaller than binary formats and works across XGBoost versions.
    save_format: Literal["json", "ubj"] = "json"

    def get_xgb_params(self) -> dict:
        """
        Get XGBoost parameter dictionary for training.

        Returns:
            Dictionary of XGBoost parameters ready for xgb.train()
        """
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "tree_method": self.tree_method,
            "device": self.device,
            "seed": self.random_seed,
            "verbosity": 1 if self.log_level == "DEBUG" else 0,
        }

        if self.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.scale_pos_weight

        return params

    def get_optuna_search_space(self) -> dict:
        """
        Get Optuna hyperparameter search space.

        Returns:
            Dictionary mapping parameter names to (low, high) tuples or categorical lists.

        Rationale for ranges:
        - max_depth [3-10]: Shallow=underfit, deep=overfit
        - learning_rate [0.01-0.3]: Lower=more trees needed, higher=faster convergence
        - subsample/colsample [0.6-1.0]: Regularization via feature/sample dropout
        - min_child_weight [1-10]: Controls leaf node size, higher=more conservative
        """
        return {
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float", 0.01, 0.3, "log"),
            "min_child_weight": ("int", 1, 10),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "gamma": ("float", 0.0, 5.0),
            "reg_alpha": ("float", 1e-8, 10.0, "log"),
            "reg_lambda": ("float", 1e-8, 10.0, "log"),
        }

    def validate(self) -> List[str]:
        """Validate XGBoost-specific configuration."""
        warnings = super().validate()

        if self.device == "cuda":
            try:
                import xgboost as xgb
                # Check if CUDA is available
                # This will be validated at runtime
            except ImportError:
                warnings.append("XGBoost not installed. Run: pip install xgboost")

        if self.max_depth > 10:
            warnings.append(
                f"max_depth={self.max_depth} is high. Risk of overfitting. "
                "Consider max_depth <= 10."
            )

        if self.learning_rate > 0.3:
            warnings.append(
                f"learning_rate={self.learning_rate} is high. May cause instability. "
                "Consider learning_rate <= 0.3."
            )

        if not self.lag_hours:
            warnings.append(
                "No lag features configured. Time-series patterns may be missed."
            )

        return warnings
