"""
Training logging and metrics tracking.

Design Decisions:

1. Structured Logging:
   - JSON format for machine parsing
   - Human-readable console output
   - Configurable verbosity

2. Metrics Tracking:
   - Rolling window for smoothing
   - Per-epoch and per-batch stats
   - Export to CSV/JSON for analysis

3. Optional Integrations:
   - TensorBoard (local)
   - Weights & Biases (cloud)
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int
    fold: int
    train_loss: float
    val_loss: Optional[float] = None
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    learning_rate: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


class MetricsTracker:
    """
    Track and aggregate training metrics across epochs and folds.

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.log_epoch(epoch=1, fold=0, train_loss=0.5, val_loss=0.4)
        >>> tracker.log_metric("utility", 0.35, phase="val")
        >>> summary = tracker.get_summary()
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize metrics tracker.

        Args:
            output_dir: Directory to save metrics (optional)
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.epochs: List[EpochMetrics] = []
        self.current_epoch: Optional[EpochMetrics] = None
        self.fold_summaries: Dict[int, Dict] = {}

        # Timing
        self.epoch_start_time: Optional[float] = None
        self.training_start_time: float = time.time()

    def start_epoch(self, epoch: int, fold: int):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
        self.current_epoch = EpochMetrics(epoch=epoch, fold=fold, train_loss=0.0)

    def end_epoch(self, train_loss: float, val_loss: Optional[float] = None):
        """Mark the end of an epoch."""
        if self.current_epoch is None:
            return

        self.current_epoch.train_loss = train_loss
        self.current_epoch.val_loss = val_loss
        self.current_epoch.duration_seconds = time.time() - (self.epoch_start_time or time.time())

        self.epochs.append(self.current_epoch)

        # Log summary
        metrics_str = f"loss={train_loss:.4f}"
        if val_loss is not None:
            metrics_str += f", val_loss={val_loss:.4f}"
        for name, value in self.current_epoch.val_metrics.items():
            metrics_str += f", val_{name}={value:.4f}"

        logger.info(
            f"Epoch {self.current_epoch.epoch} [{self.current_epoch.duration_seconds:.1f}s]: {metrics_str}"
        )

    def log_metric(
        self,
        name: str,
        value: float,
        phase: str = "val",  # "train" or "val"
    ):
        """Log a metric for the current epoch."""
        if self.current_epoch is None:
            return

        if phase == "train":
            self.current_epoch.train_metrics[name] = value
        else:
            self.current_epoch.val_metrics[name] = value

    def log_learning_rate(self, lr: float):
        """Log current learning rate."""
        if self.current_epoch:
            self.current_epoch.learning_rate = lr

    def end_fold(self, fold: int):
        """Mark the end of a CV fold and compute summary."""
        fold_epochs = [e for e in self.epochs if e.fold == fold]
        if not fold_epochs:
            return

        # Find best epoch
        val_losses = [e.val_loss for e in fold_epochs if e.val_loss is not None]
        best_val_loss = min(val_losses) if val_losses else None

        # Aggregate metrics
        all_metrics = defaultdict(list)
        for e in fold_epochs:
            for name, value in e.val_metrics.items():
                all_metrics[name].append(value)

        self.fold_summaries[fold] = {
            "n_epochs": len(fold_epochs),
            "best_val_loss": best_val_loss,
            "total_time": sum(e.duration_seconds for e in fold_epochs),
            "final_metrics": {
                name: values[-1] for name, values in all_metrics.items()
            },
            "best_metrics": {
                name: max(values) for name, values in all_metrics.items()
            },
        }

        logger.info(f"Fold {fold} complete: {self.fold_summaries[fold]}")

    def get_summary(self) -> dict:
        """Get overall training summary."""
        total_time = time.time() - self.training_start_time

        # Aggregate across folds
        all_best = defaultdict(list)
        for fold_summary in self.fold_summaries.values():
            for name, value in fold_summary.get("best_metrics", {}).items():
                all_best[name].append(value)

        return {
            "total_epochs": len(self.epochs),
            "n_folds": len(self.fold_summaries),
            "total_time_seconds": total_time,
            "fold_summaries": self.fold_summaries,
            "mean_best_metrics": {
                name: sum(values) / len(values)
                for name, values in all_best.items()
                if values
            },
            "std_best_metrics": {
                name: (sum((v - sum(values)/len(values))**2 for v in values) / len(values)) ** 0.5
                for name, values in all_best.items()
                if len(values) > 1
            },
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """Save all metrics to JSON."""
        if path is None:
            path = self.output_dir / "metrics.json" if self.output_dir else Path("metrics.json")

        data = {
            "epochs": [e.to_dict() for e in self.epochs],
            "fold_summaries": self.fold_summaries,
            "summary": self.get_summary(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved metrics to {path}")
        return path

    def to_dataframe(self):
        """Convert epoch metrics to pandas DataFrame."""
        import pandas as pd

        records = []
        for e in self.epochs:
            record = {
                "epoch": e.epoch,
                "fold": e.fold,
                "train_loss": e.train_loss,
                "val_loss": e.val_loss,
                "duration": e.duration_seconds,
                "lr": e.learning_rate,
            }
            record.update({f"train_{k}": v for k, v in e.train_metrics.items()})
            record.update({f"val_{k}": v for k, v in e.val_metrics.items()})
            records.append(record)

        return pd.DataFrame(records)


class TrainingLogger:
    """
    High-level training logger with optional integrations.

    Provides a unified interface for logging to:
    - Console (via Python logging)
    - File (JSON)
    - TensorBoard (optional)
    - Weights & Biases (optional)

    Example:
        >>> logger = TrainingLogger(output_dir, use_tensorboard=True)
        >>> logger.log_config(config)
        >>> for epoch in range(100):
        ...     logger.log_epoch(epoch, train_loss, val_loss, metrics)
        >>> logger.finish()
    """

    def __init__(
        self,
        output_dir: Path,
        experiment_name: str = "sepsis_training",
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
    ):
        """
        Initialize training logger.

        Args:
            output_dir: Directory for log files
            experiment_name: Name for experiment tracking
            use_tensorboard: Enable TensorBoard logging
            use_wandb: Enable Weights & Biases logging
            wandb_project: W&B project name
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Metrics tracker
        self.metrics = MetricsTracker(output_dir)

        # TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.output_dir / "tensorboard")
                logger.info(f"TensorBoard logging enabled: {self.output_dir / 'tensorboard'}")
            except ImportError:
                logger.warning("TensorBoard not available. Install with: pip install tensorboard")

        # Weights & Biases
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project or "sepsis-prediction",
                    name=experiment_name,
                    dir=str(self.output_dir),
                )
                logger.info(f"W&B logging enabled: {self.wandb_run.url}")
            except ImportError:
                logger.warning("Weights & Biases not available. Install with: pip install wandb")

    def log_config(self, config: dict):
        """Log training configuration."""
        # Save to file
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        # W&B
        if self.wandb_run:
            import wandb
            wandb.config.update(config)

        logger.info(f"Logged config to {config_path}")

    def log_epoch(
        self,
        epoch: int,
        fold: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None,
    ):
        """Log epoch metrics."""
        # Metrics tracker
        self.metrics.start_epoch(epoch, fold)
        if metrics:
            for name, value in metrics.items():
                self.metrics.log_metric(name, value, phase="val")
        if learning_rate:
            self.metrics.log_learning_rate(learning_rate)
        self.metrics.end_epoch(train_loss, val_loss)

        # Global step for tensorboard
        global_step = epoch + fold * 1000

        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar(f"Loss/train_fold{fold}", train_loss, epoch)
            if val_loss is not None:
                self.tb_writer.add_scalar(f"Loss/val_fold{fold}", val_loss, epoch)
            if metrics:
                for name, value in metrics.items():
                    self.tb_writer.add_scalar(f"Metrics/{name}_fold{fold}", value, epoch)
            if learning_rate:
                self.tb_writer.add_scalar(f"LR/fold{fold}", learning_rate, epoch)

        # W&B
        if self.wandb_run:
            import wandb
            log_dict = {
                "epoch": epoch,
                "fold": fold,
                "train_loss": train_loss,
            }
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
            if metrics:
                log_dict.update({f"val_{k}": v for k, v in metrics.items()})
            if learning_rate:
                log_dict["learning_rate"] = learning_rate
            wandb.log(log_dict, step=global_step)

    def log_fold_summary(self, fold: int):
        """Log fold completion."""
        self.metrics.end_fold(fold)

    def finish(self):
        """Finalize logging and save summaries."""
        # Save metrics
        self.metrics.save()

        # Close TensorBoard
        if self.tb_writer:
            self.tb_writer.close()

        # Finish W&B
        if self.wandb_run:
            import wandb
            wandb.finish()

        # Print summary
        summary = self.metrics.get_summary()
        logger.info("=" * 60)
        logger.info("Training Complete")
        logger.info("=" * 60)
        logger.info(f"Total epochs: {summary['total_epochs']}")
        logger.info(f"Total time: {summary['total_time_seconds']:.1f}s")
        if summary.get("mean_best_metrics"):
            logger.info("Mean best metrics across folds:")
            for name, value in summary["mean_best_metrics"].items():
                std = summary.get("std_best_metrics", {}).get(name, 0)
                logger.info(f"  {name}: {value:.4f} +/- {std:.4f}")
        logger.info("=" * 60)
