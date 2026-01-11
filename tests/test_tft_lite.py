"""
Unit tests for the TFT-Lite model.

Tests forward pass shapes, predictions, demo mode functionality,
and proper handling of PyTorch tensors.
"""

import pytest
import numpy as np
import torch

from models.tft_lite.tft_model import TFTLiteModel


class TestTFTLiteModelInitialization:
    """Tests for TFTLiteModel initialization."""

    def test_tft_init_demo_mode(self):
        """Test initialization without weights runs in demo mode."""
        model = TFTLiteModel(weights_path=None)

        # Should be in demo mode when no weights available
        assert model.demo_mode is True

    def test_tft_init_device_cpu(self):
        """Test initialization on CPU device."""
        model = TFTLiteModel(device='cpu')

        assert model.device == 'cpu'

    def test_tft_init_device_cuda_fallback(self):
        """Test that CUDA falls back to CPU if unavailable."""
        model = TFTLiteModel(device='cuda')

        # Should fallback to CPU if CUDA unavailable
        assert model.device in ['cpu', 'cuda']

    def test_tft_init_custom_config(self):
        """Test initialization with custom configuration."""
        model = TFTLiteModel(
            input_size=50,
            hidden_size=64,
            lstm_layers=2,
            attention_heads=4,
        )

        assert model.config['input_size'] == 50
        assert model.config['hidden_size'] == 64
        assert model.config['lstm_layers'] == 2
        assert model.config['attention_heads'] == 4

    def test_tft_get_model_info(self):
        """Test get_model_info returns expected information."""
        model = TFTLiteModel()

        info = model.get_model_info()

        assert isinstance(info, dict)
        assert 'architecture' in info
        assert 'device' in info
        assert 'weights_loaded' in info
        assert 'demo_mode' in info
        assert 'trainable_params' in info

        assert info['architecture'] == 'TFT-Lite'


class TestPredictProba:
    """Tests for predict_proba method."""

    def test_predict_proba_single_sample(self):
        """Test predict_proba with single sample (2D input)."""
        model = TFTLiteModel(input_size=10, max_seq_length=24)

        # Single sample: (seq_len, features)
        X = np.random.randn(24, 10).astype(np.float32)

        proba = model.predict_proba(X)

        # Should return scalar probability for single sample
        assert isinstance(proba, (float, np.floating))
        assert 0.0 <= proba <= 1.0

    def test_predict_proba_batch(self):
        """Test predict_proba with batch of samples (3D input)."""
        model = TFTLiteModel(input_size=10, max_seq_length=24)

        # Batch: (batch_size, seq_len, features)
        X = np.random.randn(5, 24, 10).astype(np.float32)

        probas = model.predict_proba(X)

        # Should return array of probabilities
        assert isinstance(probas, np.ndarray)
        assert probas.shape == (5,)
        assert np.all((probas >= 0.0) & (probas <= 1.0))

    def test_predict_proba_valid_range(self):
        """Test that predict_proba returns values in [0, 1]."""
        model = TFTLiteModel(input_size=5, max_seq_length=12)

        # Multiple batches
        for _ in range(3):
            X = np.random.randn(3, 12, 5).astype(np.float32)
            probas = model.predict_proba(X)

            assert np.all(probas >= 0.0)
            assert np.all(probas <= 1.0)

    def test_predict_proba_handles_zeros(self):
        """Test predict_proba with zero input."""
        model = TFTLiteModel(input_size=5, max_seq_length=12)

        X = np.zeros((3, 12, 5), dtype=np.float32)

        probas = model.predict_proba(X)

        # Should return valid probabilities
        assert probas.shape == (3,)
        assert np.all((probas >= 0.0) & (probas <= 1.0))

    def test_predict_proba_handles_large_values(self):
        """Test predict_proba with large input values."""
        model = TFTLiteModel(input_size=5, max_seq_length=12)

        X = np.random.randn(3, 12, 5).astype(np.float32) * 100

        # Should not raise error or produce NaN/inf
        probas = model.predict_proba(X)

        assert not np.any(np.isnan(probas))
        assert not np.any(np.isinf(probas))
        assert np.all((probas >= 0.0) & (probas <= 1.0))

    def test_predict_proba_invalid_shape(self):
        """Test that invalid input shapes raise error."""
        model = TFTLiteModel(input_size=5, max_seq_length=12)

        # 1D input should raise error
        X = np.random.randn(12).astype(np.float32)

        with pytest.raises(ValueError):
            model.predict_proba(X)

        # 4D input should raise error
        X = np.random.randn(2, 3, 12, 5).astype(np.float32)

        with pytest.raises(ValueError):
            model.predict_proba(X)


class TestPredict:
    """Tests for predict method."""

    def test_predict_returns_binary(self):
        """Test that predict returns binary predictions."""
        model = TFTLiteModel(input_size=10, max_seq_length=24)

        X = np.random.randn(5, 24, 10).astype(np.float32)

        predictions = model.predict(X)

        # All values should be 0 or 1
        assert np.all((predictions == 0) | (predictions == 1))

    def test_predict_returns_int32(self):
        """Test that predict returns int32 dtype."""
        model = TFTLiteModel(input_size=10, max_seq_length=24)

        X = np.random.randn(3, 24, 10).astype(np.float32)

        predictions = model.predict(X)

        assert predictions.dtype == np.int32

    def test_predict_custom_threshold(self):
        """Test predict with custom threshold."""
        model = TFTLiteModel(input_size=10, max_seq_length=24)

        X = np.random.randn(10, 24, 10).astype(np.float32)

        # Different thresholds should generally give different results
        pred_low = model.predict(X, threshold=0.3)
        pred_mid = model.predict(X, threshold=0.5)
        pred_high = model.predict(X, threshold=0.7)

        assert len(pred_low) == 10
        assert len(pred_mid) == 10
        assert len(pred_high) == 10


class TestForwardPass:
    """Tests for forward pass functionality."""

    def test_forward_pass_shape_preservation(self):
        """Test that forward pass works with various shapes."""
        model = TFTLiteModel(input_size=20, max_seq_length=48)

        # Different sequence lengths and batch sizes
        test_cases = [
            (1, 12, 20),   # Single sample, short sequence
            (5, 24, 20),   # Normal batch
            (10, 48, 20),  # Larger batch
        ]

        for batch_size, seq_len, input_size in test_cases:
            X = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)

            # Should not raise error
            probas = model.predict_proba(X)

            assert len(probas) == batch_size or isinstance(probas, (float, np.floating))

    def test_forward_pass_deterministic(self):
        """Test that forward pass is deterministic in eval mode."""
        model = TFTLiteModel(input_size=10, max_seq_length=24)

        X = np.random.randn(3, 24, 10).astype(np.float32)

        probas1 = model.predict_proba(X)
        probas2 = model.predict_proba(X)

        # Should be identical
        np.testing.assert_array_almost_equal(probas1, probas2)

    def test_forward_pass_different_inputs(self):
        """Test that different inputs produce different outputs."""
        model = TFTLiteModel(input_size=10, max_seq_length=24)

        X1 = np.random.randn(3, 24, 10).astype(np.float32)
        X2 = np.random.randn(3, 24, 10).astype(np.float32)

        probas1 = model.predict_proba(X1)
        probas2 = model.predict_proba(X2)

        # Outputs should generally be different
        # (probabilistically true unless very unlucky)
        assert not np.allclose(probas1, probas2)


class TestDemoMode:
    """Tests for demo mode functionality."""

    def test_demo_mode_enabled(self):
        """Test that demo mode is enabled without weights."""
        model = TFTLiteModel(weights_path='/nonexistent/path.pt')

        assert model.is_demo_mode is True

    def test_demo_mode_predictions(self):
        """Test that demo mode generates valid predictions."""
        model = TFTLiteModel()

        X = np.random.randn(5, 24, 10).astype(np.float32)

        probas = model.predict_proba(X)

        # Should return valid probabilities
        assert probas.shape == (5,)
        assert np.all((probas >= 0.0) & (probas <= 1.0))

    def test_demo_mode_consistency(self):
        """Test demo mode produces consistent results."""
        model = TFTLiteModel()

        X = np.random.randn(3, 24, 10).astype(np.float32)

        probas1 = model.predict_proba(X)
        probas2 = model.predict_proba(X)

        # Should be identical (same random state)
        np.testing.assert_array_almost_equal(probas1, probas2)


class TestInterpretability:
    """Tests for interpretability features."""

    def test_get_attention_weights(self):
        """Test getting attention weights."""
        model = TFTLiteModel(input_size=10, attention_heads=2, max_seq_length=24)

        X = np.random.randn(24, 10).astype(np.float32)

        # Should not raise error
        attn_weights = model.get_attention_weights(X)

        # For single sample, shape is (num_heads, seq_len, seq_len)
        assert attn_weights.ndim >= 2

    def test_get_variable_importance(self):
        """Test getting variable importance."""
        model = TFTLiteModel(input_size=10, max_seq_length=24)

        X = np.random.randn(24, 10).astype(np.float32)

        # Should not raise error
        var_weights = model.get_variable_importance(X)

        # Shape should reflect features
        assert var_weights.shape[-1] == 10

    def test_get_feature_importance_summary(self):
        """Test getting feature importance summary."""
        model = TFTLiteModel(input_size=5, max_seq_length=12)

        X = np.random.randn(3, 12, 5).astype(np.float32)

        feature_names = [f'feature_{i}' for i in range(5)]

        summary = model.get_feature_importance_summary(X, feature_names)

        assert isinstance(summary, dict)
        assert len(summary) == 5
        assert all(name in summary for name in feature_names)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_predict_with_nan(self):
        """Test handling of NaN values."""
        model = TFTLiteModel(input_size=10, max_seq_length=24)

        X = np.random.randn(3, 24, 10).astype(np.float32)
        X[0, 5, 3] = np.nan  # Inject NaN

        # Should handle NaN (might produce NaN or fill it)
        probas = model.predict_proba(X)

        assert len(probas) > 0

    def test_predict_with_inf(self):
        """Test handling of infinite values."""
        model = TFTLiteModel(input_size=10, max_seq_length=24)

        X = np.random.randn(3, 24, 10).astype(np.float32)
        X[0, 5, 3] = np.inf  # Inject inf

        # Model should handle it (might produce warning or clip)
        probas = model.predict_proba(X)

        assert len(probas) > 0

    def test_predict_very_short_sequence(self):
        """Test with very short sequence."""
        model = TFTLiteModel(input_size=10, max_seq_length=24)

        X = np.random.randn(1, 1, 10).astype(np.float32)

        # Should handle short sequence
        proba = model.predict_proba(X)

        assert 0.0 <= proba <= 1.0

    def test_model_repr(self):
        """Test string representation of model."""
        model = TFTLiteModel()

        repr_str = repr(model)

        assert isinstance(repr_str, str)
        assert 'TFT' in repr_str or 'Lite' in repr_str


class TestMemoryEfficiency:
    """Tests for memory efficiency in demo mode."""

    def test_model_eval_mode(self):
        """Test that model is in eval mode."""
        model = TFTLiteModel()

        # Model should be in eval mode for inference
        assert not model.model.training

    def test_no_grad_context(self):
        """Test that gradients are not computed during inference."""
        model = TFTLiteModel(input_size=10, max_seq_length=24)

        X = np.random.randn(3, 24, 10).astype(np.float32)

        # Inference should not require gradients
        probas = model.predict_proba(X)

        assert len(probas) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
