"""
Base trainer interface.

Design Decisions:

1. Abstract Interface:
   - Defines common training workflow
   - Concrete implementations for XGBoost and TFT

2. Cross-Validation Loop:
   - Built-in patient-level CV
   - Aggregates results across folds

3. Callback Integration:
   - Checkpointing, early stopping, logging
   - Extensible for custom callbacks
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from training.config.base import TrainingConfig
from training.data.cross_validation import CVSplit, create_cv_splits
from training.callbacks import CheckpointManager, EarlyStopping, TrainingLogger

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.

    Provides:
    - Cross-validation loop
    - Callback management
    - Metric aggregation

    Subclasses must implement:
    - _train_fold(): Train model on a single fold
    - _evaluate(): Evaluate model and compute metrics
    - _load_data(): Load and preprocess data for training
    """

    def __init__(
        self,
        config: TrainingConfig,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            output_dir: Directory for outputs (overrides config)
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.early_stopping: Optional[EarlyStopping] = None
        self.logger: Optional[TrainingLogger] = None

        # Results storage
        self.fold_results: List[Dict[str, Any]] = []
        self.best_models: List[Any] = []

        # Validate config
        warnings = config.validate()
        for warning in warnings:
            logger.warning(warning)

    def setup_callbacks(self):
        """Initialize training callbacks."""
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.output_dir / "checkpoints",
            metric_name=self.config.primary_metric,
            mode="max" if self.config.primary_metric in ["utility", "auroc", "auprc", "f1"] else "min",
            keep_best_only=self.config.keep_best_only,
        )

        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            mode="max" if self.config.primary_metric in ["utility", "auroc", "auprc", "f1"] else "min",
        )

        self.logger = TrainingLogger(
            output_dir=self.output_dir,
            experiment_name=self.__class__.__name__,
            use_tensorboard=self.config.use_tensorboard,
            use_wandb=self.config.use_wandb,
            wandb_project=self.config.wandb_project,
        )

    @abstractmethod
    def _load_data(
        self,
        train_patients: List[str],
        val_patients: List[str],
    ) -> Tuple[Any, Any]:
        """
        Load and preprocess training data.

        Args:
            train_patients: Patient IDs for training
            val_patients: Patient IDs for validation

        Returns:
            Tuple of (train_data, val_data) in model-specific format
        """
        pass

    @abstractmethod
    def _train_fold(
        self,
        train_data: Any,
        val_data: Any,
        fold: int,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train model on a single fold.

        Args:
            train_data: Training data
            val_data: Validation data
            fold: Fold index

        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        pass

    @abstractmethod
    def _evaluate(
        self,
        model: Any,
        data: Any,
    ) -> Dict[str, float]:
        """
        Evaluate model and compute metrics.

        Args:
            model: Trained model
            data: Evaluation data

        Returns:
            Dictionary of metric names to values
        """
        pass

    def train(self) -> Dict[str, Any]:
        """
        Run full cross-validation training.

        Returns:
            Dictionary with training results and aggregated metrics
        """
        logger.info("=" * 60)
        logger.info(f"Starting {self.__class__.__name__}")
        logger.info(f"  Data: {self.config.data_dir}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Folds: {self.config.n_folds}")
        logger.info(f"  Metric: {self.config.primary_metric}")
        logger.info("=" * 60)

        # Setup callbacks
        self.setup_callbacks()

        # Log config
        self.logger.log_config(self.config.to_dict())

        # Create CV splits
        splits = create_cv_splits(
            data_dir=self.config.data_dir,
            n_splits=self.config.n_folds,
            random_state=self.config.random_seed,
            max_patients=self.config.max_patients,
        )

        # Train each fold
        for split in splits:
            self._train_single_fold(split)

        # Aggregate results
        results = self._aggregate_results()

        # Finalize logging
        self.logger.finish()

        logger.info("Training complete!")
        return results

    def _train_single_fold(self, split: CVSplit):
        """Train a single CV fold."""
        fold = split.fold_idx
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold + 1}/{self.config.n_folds}")
        logger.info(f"  Train: {split.train_size} patients ({split.train_sepsis_rate:.1%} sepsis)")
        logger.info(f"  Val: {split.val_size} patients ({split.val_sepsis_rate:.1%} sepsis)")
        logger.info("=" * 60)

        # Reset early stopping for new fold
        self.early_stopping.reset()

        # Load data
        train_data, val_data = self._load_data(
            split.train_patients,
            split.val_patients,
        )

        # Train
        model, metrics = self._train_fold(train_data, val_data, fold)

        # Store results
        self.fold_results.append({
            "fold": fold,
            "train_size": split.train_size,
            "val_size": split.val_size,
            "train_sepsis_rate": split.train_sepsis_rate,
            "val_sepsis_rate": split.val_sepsis_rate,
            "metrics": metrics,
        })
        self.best_models.append(model)

        # Log fold summary
        self.logger.log_fold_summary(fold)

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all folds."""
        if not self.fold_results:
            return {}

        # Collect all metrics
        all_metrics: Dict[str, List[float]] = {}
        for result in self.fold_results:
            for name, value in result["metrics"].items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(value)

        # Compute mean and std
        aggregated = {}
        for name, values in all_metrics.items():
            aggregated[f"{name}_mean"] = np.mean(values)
            aggregated[f"{name}_std"] = np.std(values)
            aggregated[f"{name}_values"] = values

        # Best fold
        primary_values = all_metrics.get(self.config.primary_metric, [])
        if primary_values:
            best_idx = np.argmax(primary_values)
            aggregated["best_fold"] = int(best_idx)
            aggregated["best_value"] = primary_values[best_idx]

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Cross-Validation Results")
        logger.info("=" * 60)
        for name, values in all_metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            logger.info(f"  {name}: {mean:.4f} +/- {std:.4f}")
        logger.info("=" * 60)

        return {
            "n_folds": len(self.fold_results),
            "fold_results": self.fold_results,
            "aggregated": aggregated,
            "config": self.config.to_dict(),
        }

    def get_best_model(self) -> Optional[Any]:
        """Get the best model across all folds."""
        if not self.best_models or not self.fold_results:
            return None

        # Find best fold
        primary_metric = self.config.primary_metric
        best_idx = 0
        best_value = float("-inf")

        for i, result in enumerate(self.fold_results):
            value = result["metrics"].get(primary_metric, float("-inf"))
            if value > best_value:
                best_value = value
                best_idx = i

        return self.best_models[best_idx]
