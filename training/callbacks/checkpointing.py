"""
Model checkpointing for training.

Design Decisions:

1. Best-Only vs All Checkpoints:
   - Default: Keep only best checkpoint (saves disk space)
   - Optional: Keep all for debugging/analysis

2. Checkpoint Contents:
   - Model state dict (weights)
   - Optimizer state (for resume)
   - Training config (for reproducibility)
   - Metrics history (for analysis)

3. Atomic Writes:
   - Write to temp file, then rename
   - Prevents corruption on crash

4. Format:
   - XGBoost: JSON (portable, human-readable)
   - PyTorch: .pt state_dict (standard, efficient)
"""

import json
import logging
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata saved with each checkpoint."""
    epoch: int
    fold: int
    metric_name: str
    metric_value: float
    timestamp: str
    config: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointMetadata":
        return cls(**data)


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Features:
    - Saves best model based on monitored metric
    - Optional: saves all epoch checkpoints
    - Atomic writes prevent corruption
    - Tracks training history

    Example:
        >>> manager = CheckpointManager(output_dir, metric_name="utility")
        >>> # During training loop
        >>> if manager.should_save(epoch, utility_score):
        ...     manager.save(model, optimizer, epoch, fold, utility_score, config)
    """

    def __init__(
        self,
        output_dir: Path,
        metric_name: str = "utility",
        mode: str = "max",
        save_every_epoch: bool = False,
        keep_best_only: bool = True,
        model_name: str = "model",
    ):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Directory to save checkpoints
            metric_name: Name of metric to monitor
            mode: "max" if higher is better, "min" if lower is better
            save_every_epoch: Save checkpoint every epoch
            keep_best_only: Delete old best when new best found
            model_name: Prefix for checkpoint files
        """
        self.output_dir = Path(output_dir)
        self.metric_name = metric_name
        self.mode = mode
        self.save_every_epoch = save_every_epoch
        self.keep_best_only = keep_best_only
        self.model_name = model_name

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track best metric
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_path: Optional[Path] = None

        # History
        self.history: list = []

        logger.info(f"CheckpointManager: {output_dir}, metric={metric_name}, mode={mode}")

    def is_better(self, value: float) -> bool:
        """Check if value is better than current best."""
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value
        else:
            return value < self.best_value

    def should_save(self, epoch: int, metric_value: float) -> bool:
        """
        Determine if checkpoint should be saved.

        Args:
            epoch: Current epoch
            metric_value: Current metric value

        Returns:
            True if checkpoint should be saved
        """
        if self.save_every_epoch:
            return True
        return self.is_better(metric_value)

    def save_xgboost(
        self,
        model: Any,  # xgb.Booster
        epoch: int,
        fold: int,
        metric_value: float,
        config: dict,
        feature_names: Optional[list] = None,
    ) -> Path:
        """
        Save XGBoost model checkpoint.

        Args:
            model: XGBoost Booster object
            epoch: Training epoch (or num_boost_round)
            fold: CV fold index
            metric_value: Current metric value
            config: Training configuration dict
            feature_names: List of feature names

        Returns:
            Path to saved checkpoint
        """
        is_best = self.is_better(metric_value)
        suffix = "_best" if is_best else f"_epoch{epoch}"

        # Generate filename
        filename = f"{self.model_name}_fold{fold}{suffix}.json"
        checkpoint_path = self.output_dir / filename
        temp_path = checkpoint_path.with_suffix(".tmp")

        # Save model in JSON format (XGBoost default is binary unless format specified)
        model.save_model(str(temp_path), "json")

        # Atomic rename
        shutil.move(str(temp_path), str(checkpoint_path))

        # Save metadata
        metadata = CheckpointMetadata(
            epoch=epoch,
            fold=fold,
            metric_name=self.metric_name,
            metric_value=metric_value,
            timestamp=datetime.now().isoformat(),
            config=config,
        )
        metadata_path = checkpoint_path.with_suffix(".meta.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)

        # Save feature names if provided
        if feature_names:
            features_path = checkpoint_path.with_suffix(".features.json")
            with open(features_path, "w") as f:
                json.dump(feature_names, f)

        # Update best tracking
        if is_best:
            # Delete old best if keep_best_only
            if self.keep_best_only and self.best_path and self.best_path.exists():
                self.best_path.unlink()
                meta_old = self.best_path.with_suffix(".meta.json")
                if meta_old.exists():
                    meta_old.unlink()

            self.best_value = metric_value
            self.best_epoch = epoch
            self.best_path = checkpoint_path
            logger.info(f"New best {self.metric_name}: {metric_value:.4f} at epoch {epoch}")

        # Track history
        self.history.append({
            "epoch": epoch,
            "fold": fold,
            self.metric_name: metric_value,
            "path": str(checkpoint_path),
            "is_best": is_best,
        })

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def save_pytorch(
        self,
        model: Any,  # nn.Module
        optimizer: Any,  # torch.optim.Optimizer
        epoch: int,
        fold: int,
        metric_value: float,
        config: dict,
        scheduler: Optional[Any] = None,
    ) -> Path:
        """
        Save PyTorch model checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Training epoch
            fold: CV fold index
            metric_value: Current metric value
            config: Training configuration dict
            scheduler: Optional LR scheduler

        Returns:
            Path to saved checkpoint
        """
        import torch

        is_best = self.is_better(metric_value)
        suffix = "_best" if is_best else f"_epoch{epoch}"

        # Generate filename
        filename = f"{self.model_name}_fold{fold}{suffix}.pt"
        checkpoint_path = self.output_dir / filename
        temp_path = checkpoint_path.with_suffix(".tmp")

        # Build checkpoint dict
        checkpoint = {
            "epoch": epoch,
            "fold": fold,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric_name": self.metric_name,
            "metric_value": metric_value,
            "config": config,
            "timestamp": datetime.now().isoformat(),
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Save atomically
        torch.save(checkpoint, temp_path)
        shutil.move(str(temp_path), str(checkpoint_path))

        # Update best tracking
        if is_best:
            if self.keep_best_only and self.best_path and self.best_path.exists():
                self.best_path.unlink()

            self.best_value = metric_value
            self.best_epoch = epoch
            self.best_path = checkpoint_path
            logger.info(f"New best {self.metric_name}: {metric_value:.4f} at epoch {epoch}")

        # Track history
        self.history.append({
            "epoch": epoch,
            "fold": fold,
            self.metric_name: metric_value,
            "path": str(checkpoint_path),
            "is_best": is_best,
        })

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def load_best(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        return self.best_path

    def save_history(self, path: Optional[Path] = None) -> Path:
        """Save training history to JSON."""
        if path is None:
            path = self.output_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        return path

    def get_summary(self) -> dict:
        """Get summary of checkpoint manager state."""
        return {
            "output_dir": str(self.output_dir),
            "metric_name": self.metric_name,
            "mode": self.mode,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "best_path": str(self.best_path) if self.best_path else None,
            "n_checkpoints": len(self.history),
        }
