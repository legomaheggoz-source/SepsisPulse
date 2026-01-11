"""
Early stopping callback for training.

Design Decisions:

1. Patience-Based:
   - Stop after N epochs without improvement
   - Default: 10 epochs (balances thoroughness vs efficiency)

2. Minimum Delta:
   - Ignore tiny improvements (noise)
   - Default: 0.001 (0.1% improvement required)

3. Restore Best:
   - Option to restore best weights when stopping
   - Useful for final model selection
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping based on validation metric.

    Stops training when the monitored metric stops improving for a
    specified number of epochs (patience).

    Example:
        >>> early_stop = EarlyStopping(patience=10, min_delta=0.001)
        >>> for epoch in range(100):
        ...     val_score = train_epoch()
        ...     if early_stop(val_score):
        ...         print(f"Early stopping at epoch {epoch}")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        verbose: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Epochs without improvement before stopping
            min_delta: Minimum change to count as improvement
            mode: "max" if higher is better, "min" if lower is better
            verbose: Log early stopping events

        Rationale for defaults:
        - patience=10: Allows recovery from local minima
        - min_delta=0.001: 0.1% improvement threshold filters noise
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        # State
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.wait: int = 0
        self.stopped_epoch: Optional[int] = None

        # Adjust delta sign based on mode
        if mode == "min":
            self.min_delta *= -1

        logger.info(
            f"EarlyStopping: patience={patience}, min_delta={abs(min_delta):.4f}, mode={mode}"
        )

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement over best."""
        if self.best_value is None:
            return True

        if self.mode == "max":
            return current > (self.best_value + self.min_delta)
        else:
            return current < (self.best_value - self.min_delta)

    def __call__(self, value: float, epoch: Optional[int] = None) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value
            epoch: Current epoch (for logging)

        Returns:
            True if training should stop, False otherwise
        """
        if self._is_improvement(value):
            self.best_value = value
            self.best_epoch = epoch or self.best_epoch + 1
            self.wait = 0
            if self.verbose:
                logger.debug(f"EarlyStopping: New best {value:.4f}")
        else:
            self.wait += 1
            if self.verbose:
                logger.debug(
                    f"EarlyStopping: No improvement for {self.wait}/{self.patience} epochs"
                )

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best: {self.best_value:.4f} at epoch {self.best_epoch}"
                )
            return True

        return False

    def reset(self):
        """Reset early stopping state for new fold."""
        self.best_value = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = None

    def get_state(self) -> dict:
        """Get current state for checkpointing."""
        return {
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "wait": self.wait,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state(self, state: dict):
        """Load state from checkpoint."""
        self.best_value = state["best_value"]
        self.best_epoch = state["best_epoch"]
        self.wait = state["wait"]
        self.stopped_epoch = state["stopped_epoch"]
