"""Application configuration for SepsisPulse.

This module provides centralized configuration management with support for
environment variable overrides and a singleton pattern for consistent access.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Determine the project root directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Config:
    """Application configuration settings.

    Attributes:
        DATA_DIR: Path to the main data directory containing patient files.
        MODELS_DIR: Path to directory containing trained model weights.
        SAMPLE_DATA_DIR: Path to sample data subset for testing/demos.
        PREDICTION_THRESHOLD: Default probability threshold for predictions.
        OPTIMAL_LEAD_TIME: Target lead time for predictions (hours before sepsis).
        MAX_LEAD_TIME: Maximum useful lead time for predictions (hours).
        QSOFA_THRESHOLD: Minimum qSOFA score to trigger alert (Sepsis-3 standard).

    Example:
        >>> config = get_config()
        >>> config.DATA_DIR
        PosixPath('/path/to/project/data')
        >>> config.PREDICTION_THRESHOLD
        0.5
    """

    DATA_DIR: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")
    MODELS_DIR: Path = field(default_factory=lambda: _PROJECT_ROOT / "models")
    SAMPLE_DATA_DIR: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "sample" / "patients"
    )

    # Prediction thresholds
    PREDICTION_THRESHOLD: float = 0.5

    # Lead time settings (hours)
    OPTIMAL_LEAD_TIME: int = 6  # Ideal: 6 hours before sepsis onset
    MAX_LEAD_TIME: int = 12  # Maximum useful prediction window

    # Clinical thresholds
    QSOFA_THRESHOLD: int = 2  # Score >= 2 indicates high sepsis risk (Sepsis-3)


# Singleton instance storage
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the singleton configuration instance.

    Returns the same Config instance on subsequent calls, ensuring
    consistent configuration across the application.

    Returns:
        Config: The singleton configuration instance.

    Example:
        >>> config1 = get_config()
        >>> config2 = get_config()
        >>> config1 is config2
        True
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config()
        load_env_config(_config_instance)

    return _config_instance


def load_env_config(config: Optional[Config] = None) -> Config:
    """Load configuration from environment variables if present.

    Environment variables override default values. Supported variables:
        - SEPSISPULSE_DATA_DIR: Override DATA_DIR path
        - SEPSISPULSE_MODELS_DIR: Override MODELS_DIR path
        - SEPSISPULSE_SAMPLE_DATA_DIR: Override SAMPLE_DATA_DIR path
        - SEPSISPULSE_PREDICTION_THRESHOLD: Override default threshold (0.0-1.0)
        - SEPSISPULSE_OPTIMAL_LEAD_TIME: Override optimal lead time (hours)
        - SEPSISPULSE_MAX_LEAD_TIME: Override max lead time (hours)
        - SEPSISPULSE_QSOFA_THRESHOLD: Override qSOFA threshold (1-3)

    Args:
        config: Optional Config instance to update. If None, creates a new one.

    Returns:
        Config: The updated configuration instance.

    Example:
        >>> import os
        >>> os.environ["SEPSISPULSE_PREDICTION_THRESHOLD"] = "0.6"
        >>> config = load_env_config()
        >>> config.PREDICTION_THRESHOLD
        0.6
    """
    if config is None:
        config = Config()

    # Path overrides
    if data_dir := os.environ.get("SEPSISPULSE_DATA_DIR"):
        config.DATA_DIR = Path(data_dir)

    if models_dir := os.environ.get("SEPSISPULSE_MODELS_DIR"):
        config.MODELS_DIR = Path(models_dir)

    if sample_dir := os.environ.get("SEPSISPULSE_SAMPLE_DATA_DIR"):
        config.SAMPLE_DATA_DIR = Path(sample_dir)

    # Numeric overrides
    if threshold := os.environ.get("SEPSISPULSE_PREDICTION_THRESHOLD"):
        try:
            value = float(threshold)
            if 0.0 <= value <= 1.0:
                config.PREDICTION_THRESHOLD = value
        except ValueError:
            pass  # Keep default if invalid

    if optimal_lead := os.environ.get("SEPSISPULSE_OPTIMAL_LEAD_TIME"):
        try:
            value = int(optimal_lead)
            if value > 0:
                config.OPTIMAL_LEAD_TIME = value
        except ValueError:
            pass

    if max_lead := os.environ.get("SEPSISPULSE_MAX_LEAD_TIME"):
        try:
            value = int(max_lead)
            if value > 0:
                config.MAX_LEAD_TIME = value
        except ValueError:
            pass

    if qsofa := os.environ.get("SEPSISPULSE_QSOFA_THRESHOLD"):
        try:
            value = int(qsofa)
            if 1 <= value <= 3:
                config.QSOFA_THRESHOLD = value
        except ValueError:
            pass

    return config


def reset_config() -> None:
    """Reset the configuration singleton to None.

    Useful for testing or when configuration needs to be reloaded.

    Example:
        >>> reset_config()
        >>> config = get_config()  # Creates fresh instance
    """
    global _config_instance
    _config_instance = None
