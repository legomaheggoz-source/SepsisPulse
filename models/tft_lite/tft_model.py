"""
TFT-Lite Model Wrapper for Sepsis Prediction.

This module provides a high-level interface for the TFT-Lite model,
handling weight management, inference, and interpretability features.

The wrapper supports:
    - Weight loading/saving with validation
    - Demo mode when weights are unavailable
    - Batch prediction with threshold control
    - Attention weight extraction for interpretability
"""

import os
import logging
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union
from pathlib import Path

from .architecture import TFTLite


# Configure logging
logger = logging.getLogger(__name__)


class TFTLiteModel:
    """
    High-level wrapper for the TFT-Lite model.

    Provides a scikit-learn-like interface for the TFT-Lite architecture
    with support for weight management and interpretability.

    Attributes:
        model: The underlying TFTLite neural network.
        device: Device for computation ('cpu' or 'cuda').
        weights_loaded: Whether pre-trained weights have been loaded.
        demo_mode: Whether running in demo mode (random predictions).

    Example:
        >>> model = TFTLiteModel(weights_path='models/tft_lite/weights/tft_lite_v1.pt')
        >>> predictions = model.predict(patient_data)
        >>> probabilities = model.predict_proba(patient_data)
        >>> attention = model.get_attention_weights(patient_data)
    """

    # Default paths
    DEFAULT_WEIGHTS_PATH = 'models/tft_lite/weights/tft_lite_v1.pt'

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = 'cpu',
        input_size: int = 39,  # PhysioNet 2019 features
        hidden_size: int = 64,  # Matches trained model
        lstm_layers: int = 2,  # Matches trained model
        attention_heads: int = 4,  # Matches trained model
        output_size: int = 1,
        dropout: float = 0.1,
        max_seq_length: int = 72  # Matches trained model
    ):
        """
        Initialize the TFT-Lite model wrapper.

        Args:
            weights_path: Path to pre-trained weights file.
                         If None, attempts to load from default path.
                         If weights don't exist, runs in demo mode.
            device: Device for computation ('cpu' or 'cuda').
            input_size: Number of input features.
            hidden_size: Hidden dimension size.
            lstm_layers: Number of LSTM layers.
            attention_heads: Number of attention heads.
            output_size: Number of output units.
            dropout: Dropout probability.
            max_seq_length: Maximum sequence length.
        """
        self.device = self._validate_device(device)
        self.weights_loaded = False
        self.demo_mode = False

        # Model configuration
        self.config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'lstm_layers': lstm_layers,
            'attention_heads': attention_heads,
            'output_size': output_size,
            'dropout': dropout,
            'max_seq_length': max_seq_length
        }

        # Initialize the model
        self.model = TFTLite(**self.config)
        self.model.to(self.device)
        self.model.eval()

        # Store for last inference results
        self._last_attention_weights = None
        self._last_variable_weights = None

        # Attempt to load weights
        if weights_path is not None:
            self.load_weights(weights_path)
        else:
            # Try default path
            self._try_load_default_weights()

    def _validate_device(self, device: str) -> str:
        """
        Validate and return the computation device.

        Args:
            device: Requested device ('cpu' or 'cuda').

        Returns:
            Validated device string.
        """
        if device == 'cuda' and not torch.cuda.is_available():
            warnings.warn(
                "CUDA requested but not available. Falling back to CPU.",
                RuntimeWarning
            )
            return 'cpu'
        return device

    def _try_load_default_weights(self):
        """Attempt to load weights from the default path."""
        # Try multiple potential locations
        potential_paths = [
            self.DEFAULT_WEIGHTS_PATH,
            Path(__file__).parent / 'weights' / 'tft_lite_v1.pt',
            Path.cwd() / 'models' / 'tft_lite' / 'weights' / 'tft_lite_v1.pt'
        ]

        for path in potential_paths:
            if Path(path).exists():
                try:
                    self.load_weights(str(path))
                    return
                except Exception as e:
                    logger.warning(f"Failed to load weights from {path}: {e}")

        # No weights found - enable demo mode
        self._enable_demo_mode()

    def _enable_demo_mode(self):
        """Enable demo mode with random predictions."""
        self.demo_mode = True
        self.weights_loaded = False
        logger.info(
            "TFT-Lite running in demo mode (no pre-trained weights). "
            "Predictions will be based on uninitialized weights."
        )

    def load_weights(self, path: str):
        """
        Load pre-trained weights from a file.

        Args:
            path: Path to the weights file (.pt or .pth).

        Raises:
            FileNotFoundError: If the weights file doesn't exist.
            RuntimeError: If the weights are incompatible with the model.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")

        try:
            # Load state dict
            state_dict = torch.load(path, map_location=self.device, weights_only=True)

            # Handle wrapped state dicts (e.g., from training checkpoints)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Load weights into model
            self.model.load_state_dict(state_dict)
            self.model.eval()

            self.weights_loaded = True
            self.demo_mode = False
            logger.info(f"Successfully loaded weights from {path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {path}: {e}")

    def save_weights(self, path: str):
        """
        Save model weights to a file.

        Args:
            path: Path to save the weights file.

        Raises:
            RuntimeError: If saving fails.
        """
        path = Path(path)

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Successfully saved weights to {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save weights to {path}: {e}")

    def _preprocess_input(self, X: np.ndarray) -> torch.Tensor:
        """
        Preprocess numpy input for the model.

        Args:
            X: Input array of shape (batch, seq_len, features) or
               (seq_len, features) for single sample.

        Returns:
            Preprocessed tensor on the correct device.

        Raises:
            ValueError: If input shape is invalid.
        """
        # Handle single sample (2D input)
        if X.ndim == 2:
            X = X[np.newaxis, :, :]

        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq_len, features), got {X.ndim}D"
            )

        # Convert to tensor
        tensor = torch.from_numpy(X.astype(np.float32))
        return tensor.to(self.device)

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Generate binary predictions.

        Args:
            X: Input array of shape (batch, seq_len, features) or
               (seq_len, features) for single sample.
            threshold: Probability threshold for positive class.

        Returns:
            Binary predictions of shape (batch,) or scalar for single sample.
        """
        probas = self.predict_proba(X)

        # Apply threshold
        predictions = (probas >= threshold).astype(np.int32)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions.

        Args:
            X: Input array of shape (batch, seq_len, features) or
               (seq_len, features) for single sample.

        Returns:
            Probability predictions of shape (batch,).
        """
        # Track if input was single sample
        single_sample = X.ndim == 2

        # Preprocess
        tensor = self._preprocess_input(X)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probas = torch.sigmoid(logits).cpu().numpy()

        # Store attention weights
        self._last_attention_weights = self.model.get_attention_weights()
        self._last_variable_weights = self.model.get_variable_weights()

        # Flatten output
        probas = probas.squeeze(-1)

        # Return scalar for single sample
        if single_sample and probas.size == 1:
            return probas.item()

        return probas

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Get attention weights for interpretability.

        Performs a forward pass and returns the attention weights
        from the multi-head attention layer.

        Args:
            X: Input array of shape (batch, seq_len, features) or
               (seq_len, features) for single sample.

        Returns:
            Attention weights of shape (batch, num_heads, seq_len, seq_len).
            For single sample, shape is (num_heads, seq_len, seq_len).
        """
        single_sample = X.ndim == 2

        # Run forward pass to get attention weights
        tensor = self._preprocess_input(X)

        with torch.no_grad():
            _ = self.model(tensor)

        # Get attention weights
        attn_weights = self.model.get_attention_weights()

        if attn_weights is None:
            raise RuntimeError("No attention weights available")

        attn_weights = attn_weights.cpu().numpy()

        # Remove batch dimension for single sample
        if single_sample:
            attn_weights = attn_weights.squeeze(0)

        return attn_weights

    def get_variable_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Get variable selection weights for feature importance.

        Args:
            X: Input array of shape (batch, seq_len, features) or
               (seq_len, features) for single sample.

        Returns:
            Variable weights of shape (batch, seq_len, input_size).
            For single sample, shape is (seq_len, input_size).
        """
        single_sample = X.ndim == 2

        # Run forward pass to get variable weights
        tensor = self._preprocess_input(X)

        with torch.no_grad():
            _ = self.model(tensor)

        # Get variable weights
        var_weights = self.model.get_variable_weights()

        if var_weights is None:
            raise RuntimeError("No variable weights available")

        var_weights = var_weights.cpu().numpy()

        # Remove batch dimension for single sample
        if single_sample:
            var_weights = var_weights.squeeze(0)

        return var_weights

    def get_feature_importance_summary(
        self,
        X: np.ndarray,
        feature_names: Optional[list] = None
    ) -> dict:
        """
        Get aggregated feature importance across all timesteps.

        Args:
            X: Input array of shape (batch, seq_len, features).
            feature_names: Optional list of feature names.

        Returns:
            Dictionary mapping feature indices/names to importance scores.
        """
        var_weights = self.get_variable_importance(X)

        # Average across batch and timesteps
        if var_weights.ndim == 3:
            importance = var_weights.mean(axis=(0, 1))
        else:
            importance = var_weights.mean(axis=0)

        # Create result dictionary
        if feature_names is not None:
            if len(feature_names) != len(importance):
                raise ValueError(
                    f"feature_names length ({len(feature_names)}) doesn't match "
                    f"number of features ({len(importance)})"
                )
            return dict(zip(feature_names, importance))

        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    @property
    def is_demo_mode(self) -> bool:
        """Check if model is running in demo mode."""
        return self.demo_mode

    def get_model_info(self) -> dict:
        """
        Get model information and configuration.

        Returns:
            Dictionary with model configuration and status.
        """
        return {
            'architecture': 'TFT-Lite',
            'config': self.config.copy(),
            'device': self.device,
            'weights_loaded': self.weights_loaded,
            'demo_mode': self.demo_mode,
            'trainable_params': self.model.count_parameters()
        }

    def __repr__(self) -> str:
        """Return string representation of the model wrapper."""
        status = "demo mode" if self.demo_mode else "weights loaded"
        return (
            f"TFTLiteModel(\n"
            f"  status='{status}',\n"
            f"  device='{self.device}',\n"
            f"  params={self.model.count_parameters():,}\n"
            f")"
        )
