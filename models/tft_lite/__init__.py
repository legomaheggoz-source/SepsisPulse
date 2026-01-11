"""
TFT-Lite: Lightweight Temporal Fusion Transformer for Sepsis Prediction.

This module provides a memory-efficient implementation of the Temporal Fusion
Transformer optimized for resource-constrained environments (2GB RAM).

Key Features:
    - ~500K parameters (vs ~10M in full TFT)
    - Variable Selection Network for feature importance
    - Multi-head attention for interpretability
    - Demo mode for testing without pre-trained weights

Components:
    - TFTLite: The neural network architecture
    - TFTLiteModel: High-level wrapper with inference utilities

Example:
    >>> from models.tft_lite import TFTLiteModel
    >>> model = TFTLiteModel()
    >>> predictions = model.predict(patient_data)
    >>> attention_weights = model.get_attention_weights(patient_data)
"""

from .architecture import (
    TFTLite,
    GatedLinearUnit,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
)
from .tft_model import TFTLiteModel


__all__ = [
    # Main classes
    "TFTLite",
    "TFTLiteModel",
    # Architecture components
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "VariableSelectionNetwork",
    "InterpretableMultiHeadAttention",
]

__version__ = "1.0.0"
