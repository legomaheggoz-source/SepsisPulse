"""
CLI script for TFT-Lite training.

Usage:
    # Quick test
    python -m training.scripts.train_tft --max-patients 100 --n-folds 3

    # Full training on RTX 4090
    python -m training.scripts.train_tft --data-dir data/physionet --device cuda

    # Load config from Optuna
    python -m training.scripts.train_tft --config training_outputs/optuna/tft_best.yaml
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.config.tft_config import TFTConfig
from training.trainers.tft_trainer import TFTTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tft_training.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train TFT-Lite model for sepsis prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick validation run
    python -m training.scripts.train_tft --max-patients 100 --n-folds 3

    # Full training on RTX 4090
    python -m training.scripts.train_tft --data-dir data/physionet --device cuda --batch-size 1024

    # With custom architecture
    python -m training.scripts.train_tft --hidden-size 96 --n-heads 8
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
        default=Path("training_outputs/tft"),
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

    # Architecture
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden layer size",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--n-encoder-layers",
        type=int,
        default=2,
        help="Number of encoder layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size (1024 for RTX 4090)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Maximum epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    # GPU
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device for training",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
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
        help="Save final model to models/tft_lite/weights/",
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = TFTConfig.load(args.config)
        config.data_dir = args.data_dir
        config.output_dir = args.output_dir
        if args.max_patients:
            config.max_patients = args.max_patients
    else:
        config = TFTConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_folds=args.n_folds,
            random_seed=args.seed,
            max_patients=args.max_patients,
            hidden_size=args.hidden_size,
            n_heads=args.n_heads,
            n_encoder_layers=args.n_encoder_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            use_amp=not args.no_amp,
            early_stopping_patience=args.early_stopping_patience,
        )

    # Validate
    warnings = config.validate()
    for warning in warnings:
        logger.warning(warning)

    # Estimate memory
    mem_gb, mem_desc = config.estimate_memory_usage()
    logger.info(f"Estimated memory usage: {mem_desc}")

    # Train
    trainer = TFTTrainer(config)
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
