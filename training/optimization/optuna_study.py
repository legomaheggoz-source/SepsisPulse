"""
Hyperparameter optimization with Optuna.

Design Decisions:

1. TPE Sampler (Tree-Parzen Estimator):
   - Efficient for moderate-dimensional spaces
   - Learns from previous trials to focus on promising regions
   - Better than random/grid for 8+ hyperparameters

2. MedianPruner:
   - Stops unpromising trials early
   - Compares to running median of completed trials
   - Saves significant compute time

3. SQLite Storage:
   - Allows resuming interrupted studies
   - Enables parallel optimization
   - Persists results for analysis

4. Trial Configuration:
   - 50-100 trials typically sufficient with TPE
   - ~4 hours budget per model overnight
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from training.config.base import TrainingConfig
from training.config.xgboost_config import XGBoostConfig
from training.config.tft_config import TFTConfig
from training.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


def create_study(
    study_name: str,
    storage_path: Optional[Path] = None,
    direction: str = "maximize",
    n_startup_trials: int = 10,
    n_warmup_steps: int = 5,
) -> optuna.Study:
    """
    Create an Optuna study for hyperparameter optimization.

    Args:
        study_name: Name for the study (used in storage)
        storage_path: Path to SQLite database for persistence
        direction: "maximize" or "minimize"
        n_startup_trials: Random trials before TPE kicks in
        n_warmup_steps: Steps before pruning starts

    Returns:
        Optuna Study object

    Rationale:
    - n_startup_trials=10: Enough random exploration before TPE
    - n_warmup_steps=5: Let trials run a bit before pruning
    """
    # Storage
    if storage_path:
        storage_path = Path(storage_path)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{storage_path}"
    else:
        storage = None

    # Sampler
    sampler = TPESampler(
        seed=42,
        n_startup_trials=n_startup_trials,
    )

    # Pruner
    pruner = MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True,
    )

    logger.info(f"Created study: {study_name}")
    logger.info(f"  Direction: {direction}")
    logger.info(f"  Sampler: TPE (startup={n_startup_trials})")
    logger.info(f"  Pruner: Median (warmup={n_warmup_steps})")
    if storage:
        logger.info(f"  Storage: {storage}")

    return study


class OptunaOptimizer:
    """
    Hyperparameter optimizer for sepsis prediction models.

    Wraps Optuna for XGBoost and TFT-Lite optimization.

    Example:
        >>> optimizer = OptunaOptimizer(
        ...     model_type="xgboost",
        ...     data_dir="data/physionet",
        ...     output_dir="optuna_results",
        ... )
        >>> best_config = optimizer.optimize(n_trials=100, timeout_hours=4)
    """

    def __init__(
        self,
        model_type: str,  # "xgboost" or "tft"
        data_dir: Path,
        output_dir: Path,
        n_folds: int = 3,  # Fewer folds for faster optimization
        max_patients: Optional[int] = None,
    ):
        """
        Initialize optimizer.

        Args:
            model_type: "xgboost" or "tft"
            data_dir: Path to PhysioNet data
            output_dir: Directory for optimization outputs
            n_folds: Number of CV folds (3 for speed during optimization)
            max_patients: Limit patients for faster trials
        """
        self.model_type = model_type
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.n_folds = n_folds
        self.max_patients = max_patients

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get config class and search space
        if model_type == "xgboost":
            self.config_class = XGBoostConfig
            self.search_space = XGBoostConfig().get_optuna_search_space()
        elif model_type == "tft":
            self.config_class = TFTConfig
            self.search_space = TFTConfig().get_optuna_search_space()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        logger.info(f"OptunaOptimizer: {model_type}")
        logger.info(f"  Search space: {list(self.search_space.keys())}")

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """Suggest hyperparameters for a trial."""
        params = {}

        for name, spec in self.search_space.items():
            param_type = spec[0]

            if param_type == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif param_type == "float":
                if len(spec) > 3 and spec[3] == "log":
                    params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
                else:
                    params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec[1])

        return params

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.

        Trains model with suggested hyperparameters and returns
        the mean validation metric across folds.
        """
        # Suggest hyperparameters
        suggested_params = self._suggest_params(trial)
        logger.info(f"Trial {trial.number}: {suggested_params}")

        # Create config with suggested params
        config = self.config_class(
            data_dir=self.data_dir,
            output_dir=self.output_dir / f"trial_{trial.number}",
            n_folds=self.n_folds,
            max_patients=self.max_patients,
            **suggested_params,
        )

        # Create trainer
        if self.model_type == "xgboost":
            from training.trainers.xgboost_trainer import XGBoostTrainer
            trainer = XGBoostTrainer(config)
        else:
            from training.trainers.tft_trainer import TFTTrainer
            trainer = TFTTrainer(config)

        try:
            # Train
            results = trainer.train()

            # Get mean of primary metric
            primary_metric = config.primary_metric
            mean_value = results["aggregated"].get(f"{primary_metric}_mean", 0.0)

            logger.info(f"Trial {trial.number} complete: {primary_metric}={mean_value:.4f}")

            # Report intermediate values for pruning
            for i, fold_result in enumerate(results["fold_results"]):
                trial.report(fold_result["metrics"].get(primary_metric, 0), step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return mean_value

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

    def optimize(
        self,
        n_trials: int = 100,
        timeout_hours: Optional[float] = None,
        n_jobs: int = 1,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Maximum number of trials
            timeout_hours: Time limit in hours
            n_jobs: Parallel trials (1 for GPU models)

        Returns:
            Dictionary with best parameters and study results
        """
        study_name = f"{self.model_type}_optimization"
        storage_path = self.output_dir / f"{study_name}.db"

        study = create_study(
            study_name=study_name,
            storage_path=storage_path,
            direction="maximize",
        )

        timeout = timeout_hours * 3600 if timeout_hours else None

        logger.info(f"Starting optimization: {n_trials} trials, timeout={timeout_hours}h")

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )

        # Results
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        logger.info("=" * 60)
        logger.info("Optimization Complete")
        logger.info("=" * 60)
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best value: {best_value:.4f}")
        logger.info(f"Best params: {best_params}")
        logger.info("=" * 60)

        # Save best config
        best_config = self.config_class(
            data_dir=self.data_dir,
            output_dir=self.output_dir / "best",
            **best_params,
        )
        config_path = self.output_dir / f"{self.model_type}_best.yaml"
        best_config.save(config_path)
        logger.info(f"Saved best config to {config_path}")

        return {
            "best_params": best_params,
            "best_value": best_value,
            "best_trial": best_trial.number,
            "n_trials": len(study.trials),
            "config_path": str(config_path),
        }
