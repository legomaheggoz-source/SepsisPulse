"""
CLI script for Optuna hyperparameter optimization.

Usage:
    # XGBoost optimization (4 hours)
    python -m training.scripts.optimize_hyperparams --model xgboost --n-trials 100 --timeout-hours 4

    # TFT optimization (4 hours)
    python -m training.scripts.optimize_hyperparams --model tft --n-trials 50 --timeout-hours 4

    # Quick test
    python -m training.scripts.optimize_hyperparams --model xgboost --n-trials 5 --max-patients 500
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.optimization.optuna_study import OptunaOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("optuna_optimization.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # XGBoost optimization overnight
    python -m training.scripts.optimize_hyperparams --model xgboost --n-trials 100 --timeout-hours 4

    # TFT optimization overnight
    python -m training.scripts.optimize_hyperparams --model tft --n-trials 50 --timeout-hours 4

    # Quick test run
    python -m training.scripts.optimize_hyperparams --model xgboost --n-trials 5 --max-patients 500 --n-folds 2
        """,
    )

    # Model selection
    parser.add_argument(
        "--model",
        choices=["xgboost", "tft"],
        required=True,
        help="Model type to optimize",
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
        default=Path("training_outputs/optuna"),
        help="Output directory for optimization results",
    )

    # Optimization
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Maximum number of trials",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        help="Time limit in hours",
    )

    # Speed vs accuracy
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of CV folds per trial (3 for speed)",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        help="Limit patients per trial (for faster optimization)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Optuna Hyperparameter Optimization")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Max trials: {args.n_trials}")
    logger.info(f"Timeout: {args.timeout_hours}h" if args.timeout_hours else "Timeout: None")
    logger.info(f"CV folds: {args.n_folds}")
    logger.info(f"Max patients: {args.max_patients or 'All'}")
    logger.info("=" * 60)

    # Create optimizer
    optimizer = OptunaOptimizer(
        model_type=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        max_patients=args.max_patients,
    )

    # Run optimization
    results = optimizer.optimize(
        n_trials=args.n_trials,
        timeout_hours=args.timeout_hours,
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Optimization Complete")
    logger.info("=" * 60)
    logger.info(f"Best trial: {results['best_trial']}")
    logger.info(f"Best value: {results['best_value']:.4f}")
    logger.info(f"Best params: {results['best_params']}")
    logger.info(f"Config saved: {results['config_path']}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
