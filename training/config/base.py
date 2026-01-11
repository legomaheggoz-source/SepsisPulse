"""
Base training configuration dataclass.

Design Decisions:
1. Dataclass-based for type safety and IDE support
2. YAML serialization for experiment reproducibility
3. Patient-level CV by default (prevents data leakage)
4. Clinical Utility Score as primary metric (matches PhysioNet Challenge)
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Literal
import yaml


@dataclass
class TrainingConfig:
    """
    Base configuration for model training.

    Attributes:
        data_dir: Path to PhysioNet data directory
        output_dir: Path to save trained models and logs
        n_folds: Number of cross-validation folds (5 recommended for reliability)
        random_seed: Seed for reproducibility
        max_patients: Limit patients for quick testing (None = use all)

        # Class Imbalance
        # Rationale: Sepsis is rare (~7.3% of patients). Cost-weighted loss
        # directly optimizes for clinical utility without resampling artifacts.
        use_class_weights: Whether to weight classes inversely to frequency

        # Metric Selection
        # Rationale: AUC-ROC doesn't capture early detection benefit.
        # Clinical Utility Score rewards 6-12 hour lead time and penalizes
        # false alarms, matching the PhysioNet Challenge 2019 official metric.
        primary_metric: Metric to optimize during training

        # Checkpointing
        checkpoint_every_epoch: Save model every N epochs
        keep_best_only: Only keep best checkpoint (saves disk space)

        # Early Stopping
        early_stopping_patience: Epochs without improvement before stopping
        early_stopping_min_delta: Minimum improvement to count as progress
    """

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("data/physionet"))
    output_dir: Path = field(default_factory=lambda: Path("training_outputs"))

    # Cross-validation
    # Rationale: 5-fold balances validation reliability with training time.
    # Patient-level splits prevent data leakage (same patient can't be in train+test).
    n_folds: int = 5
    random_seed: int = 42
    max_patients: Optional[int] = None

    # Class imbalance handling
    use_class_weights: bool = True

    # Metric selection
    primary_metric: Literal["utility", "auroc", "auprc", "f1"] = "utility"

    # Checkpointing
    checkpoint_every_epoch: int = 1
    keep_best_only: bool = True

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: Optional[str] = None

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        d = asdict(self)
        d["data_dir"] = str(d["data_dir"])
        d["output_dir"] = str(d["output_dir"])
        return d

    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings.

        Returns:
            List of warning messages (empty if all valid)
        """
        warnings = []

        if self.n_folds < 3:
            warnings.append(
                f"n_folds={self.n_folds} is low. 3-5 folds recommended for "
                "reliable validation estimates."
            )

        if self.n_folds > 10:
            warnings.append(
                f"n_folds={self.n_folds} is high. Training time scales linearly "
                "with folds. 5 folds typically sufficient."
            )

        if self.max_patients is not None and self.max_patients < 1000:
            warnings.append(
                f"max_patients={self.max_patients} is very low. Results may not "
                "generalize. Use at least 1000 for meaningful validation."
            )

        if self.primary_metric != "utility":
            warnings.append(
                f"primary_metric='{self.primary_metric}' differs from PhysioNet "
                "Challenge metric. Consider 'utility' for clinical relevance."
            )

        return warnings
