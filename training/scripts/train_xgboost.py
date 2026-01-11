"""
CLI script for XGBoost-TS training.

Usage:
    # Quick test on sample data
    python -m training.scripts.train_xgboost --max-patients 100 --n-folds 3

    # Full training on PhysioNet data
    python -m training.scripts.train_xgboost --data-dir data/physionet --n-folds 5

    # Load config from file (e.g., from Optuna optimization)
    python -m training.scripts.train_xgboost --config training_outputs/optuna/xgboost_best.yaml
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.config.xgboost_config import XGBoostConfig
from training.trainers.xgboost_trainer import XGBoostTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("xgboost_training.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost-TS model for sepsis prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick validation run
    python -m training.scripts.train_xgboost --max-patients 100 --n-folds 3

    # Full training
    python -m training.scripts.train_xgboost --data-dir data/physionet --n-folds 5

    # With custom hyperparameters
    python -m training.scripts.train_xgboost --max-depth 8 --learning-rate 0.05
        """,
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/physionet"),
        help="Path to PhysioNet data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_outputs/xgboost"),
        help="Output directory for models and logs",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=Path,
        help="Load configuration from YAML file",
    )

    # Cross-validation
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        help="Limit number of patients (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # XGBoost hyperparameters
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Maximum number of boosting rounds",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (eta)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Subsample ratio of training instances",
    )

    # GPU
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device for training",
    )

    # Early stopping
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Epochs without improvement before stopping",
    )

    # Output
    parser.add_argument(
        "--save-final",
        action="store_true",
        help="Save final model to models/xgboost_ts/weights/",
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = XGBoostConfig.load(args.config)
        # Override with CLI args
        config.data_dir = args.data_dir
        config.output_dir = args.output_dir
        if args.max_patients:
            config.max_patients = args.max_patients
    else:
        config = XGBoostConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_folds=args.n_folds,
            random_seed=args.seed,
            max_patients=args.max_patients,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            device=args.device,
            early_stopping_patience=args.early_stopping_patience,
        )

    # Validate
    warnings = config.validate()
    for warning in warnings:
        logger.warning(warning)

    # Train
    trainer = XGBoostTrainer(config)
    results = trainer.train()

    # Save final model if requested
    if args.save_final:
        trainer.save_final_model()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    for metric, value in results["aggregated"].items():
        if not metric.endswith("_values"):
            logger.info(f"  {metric}: {value:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
