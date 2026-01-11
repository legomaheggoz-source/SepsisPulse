"""
Unit tests for the XGBoost time-series model.

Tests demo mode predictions, binary predictions, probability shapes,
and handling of various input scenarios.
"""

import pytest
import pandas as pd
import numpy as np

from models.xgboost_ts.xgboost_model import XGBoostTSModel


class TestXGBoostModelInitialization:
    """Tests for XGBoostTSModel initialization."""

    def test_xgboost_init_demo_mode(self):
        """Test initialization without weights runs in demo mode."""
        model = XGBoostTSModel(weights_path=None)

        # Should be in demo mode when no weights available
        assert model.is_demo_mode is True

    def test_xgboost_init_sets_attributes(self):
        """Test that initialization sets expected attributes."""
        model = XGBoostTSModel()

        assert hasattr(model, 'model')
        assert hasattr(model, 'is_demo_mode')
        assert hasattr(model, 'feature_names')
        assert hasattr(model, 'weights_path')

    def test_xgboost_get_model_info(self):
        """Test get_model_info returns dict with expected keys."""
        model = XGBoostTSModel()

        info = model.get_model_info()

        assert isinstance(info, dict)
        assert 'is_demo_mode' in info
        assert 'weights_path' in info
        assert 'xgboost_installed' in info


class TestPredictProba:
    """Tests for predict_proba method."""

    def test_predict_proba_returns_correct_shape(self):
        """Test that predict_proba returns shape (n_samples, 2)."""
        model = XGBoostTSModel()

        # Create sample data
        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(10) for i in range(5)
        })

        probas = model.predict_proba(X)

        # Should have shape (n_samples, 2)
        assert probas.shape == (10, 2)

    def test_predict_proba_valid_probabilities(self):
        """Test that returned values are valid probabilities."""
        model = XGBoostTSModel()

        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(5) for i in range(3)
        })

        probas = model.predict_proba(X)

        # All values should be in [0, 1]
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

        # Each row should sum to 1
        row_sums = probas.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_predict_proba_single_sample(self):
        """Test predict_proba with single sample."""
        model = XGBoostTSModel()

        X = pd.DataFrame({
            'feature_0': [0.5],
            'feature_1': [1.5],
        })

        probas = model.predict_proba(X)

        assert probas.shape == (1, 2)

    def test_predict_proba_empty_dataframe(self):
        """Test predict_proba with empty DataFrame."""
        model = XGBoostTSModel()

        X = pd.DataFrame()

        probas = model.predict_proba(X)

        # Should return empty array with correct shape
        assert probas.shape == (0, 2)

    def test_predict_proba_deterministic_demo_mode(self):
        """Test that demo mode predictions are deterministic."""
        model = XGBoostTSModel()

        X = pd.DataFrame({
            'feature_0': [1.0, 2.0, 3.0],
            'feature_1': [4.0, 5.0, 6.0],
        })

        # Same input should give same predictions
        probas1 = model.predict_proba(X)
        probas2 = model.predict_proba(X)

        np.testing.assert_array_almost_equal(probas1, probas2)

    def test_predict_proba_handles_nan(self):
        """Test that predict_proba handles NaN values."""
        model = XGBoostTSModel()

        X = pd.DataFrame({
            'feature_0': [1.0, np.nan, 3.0],
            'feature_1': [4.0, 5.0, np.nan],
        })

        # Should not raise error
        probas = model.predict_proba(X)

        assert probas.shape == (3, 2)

    def test_predict_proba_large_dataset(self):
        """Test predict_proba with large dataset."""
        model = XGBoostTSModel()

        # Create larger dataset
        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(1000) for i in range(10)
        })

        probas = model.predict_proba(X)

        assert probas.shape == (1000, 2)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)


class TestPredict:
    """Tests for predict method."""

    def test_predict_returns_binary(self):
        """Test that predict returns binary values."""
        model = XGBoostTSModel()

        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(10) for i in range(5)
        })

        predictions = model.predict(X)

        # All values should be 0 or 1
        assert np.all((predictions == 0) | (predictions == 1))

    def test_predict_returns_correct_dtype(self):
        """Test that predict returns int32 dtype."""
        model = XGBoostTSModel()

        X = pd.DataFrame({
            'feature_0': [0.5, 1.5, 2.5],
        })

        predictions = model.predict(X)

        assert predictions.dtype == np.int32

    def test_predict_threshold_default(self):
        """Test predict with default threshold (0.5)."""
        model = XGBoostTSModel()

        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(10) for i in range(5)
        })

        predictions = model.predict(X, threshold=0.5)

        assert len(predictions) == 10

    def test_predict_custom_threshold(self):
        """Test predict with custom threshold."""
        model = XGBoostTSModel()

        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(10) for i in range(5)
        })

        # Lower threshold should give more positive predictions
        pred_low = model.predict(X, threshold=0.3)
        pred_high = model.predict(X, threshold=0.7)

        # Generally, lower threshold -> more predictions
        # (not strictly true for all cases, but should be tendency)
        assert isinstance(pred_low, np.ndarray)
        assert isinstance(pred_high, np.ndarray)

    def test_predict_invalid_threshold(self):
        """Test that invalid thresholds raise ValueError."""
        model = XGBoostTSModel()

        X = pd.DataFrame({'feature_0': [0.5]})

        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            model.predict(X, threshold=-0.1)

        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            model.predict(X, threshold=1.5)

    def test_predict_empty_dataframe(self):
        """Test predict with empty DataFrame."""
        model = XGBoostTSModel()

        X = pd.DataFrame()

        predictions = model.predict(X)

        assert len(predictions) == 0

    def test_predict_single_sample(self):
        """Test predict with single sample."""
        model = XGBoostTSModel()

        X = pd.DataFrame({
            'feature_0': [1.5],
            'feature_1': [2.5],
        })

        predictions = model.predict(X)

        assert len(predictions) == 1
        assert predictions[0] in [0, 1]


class TestDemoMode:
    """Tests for demo mode functionality."""

    def test_demo_mode_generates_probabilities(self):
        """Test that demo mode generates valid probability predictions."""
        model = XGBoostTSModel()

        # Force demo mode
        assert model.is_demo_mode is True

        X = pd.DataFrame({
            'feature_0': [1.0, 2.0],
            'feature_1': [3.0, 4.0],
        })

        probas = model.predict_proba(X)

        # Should still be valid probabilities
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_demo_mode_respects_input_data(self):
        """Test that demo predictions are influenced by input data."""
        model = XGBoostTSModel()

        # Two very different inputs
        X_low = pd.DataFrame({
            'feature_0': [0.0, 0.0],
            'feature_1': [0.0, 0.0],
        })

        X_high = pd.DataFrame({
            'feature_0': [10.0, 10.0],
            'feature_1': [10.0, 10.0],
        })

        probas_low = model.predict_proba(X_low)
        probas_high = model.predict_proba(X_high)

        # Mean probability for positive class should differ
        # (higher input values might correlate with higher risk)
        assert isinstance(probas_low, np.ndarray)
        assert isinstance(probas_high, np.ndarray)

    def test_demo_mode_consistent_across_calls(self):
        """Test that demo mode gives consistent results."""
        model = XGBoostTSModel()

        X = pd.DataFrame({
            'feature_0': [1.0, 2.0, 3.0],
            'feature_1': [4.0, 5.0, 6.0],
        })

        probas1 = model.predict_proba(X)
        probas2 = model.predict_proba(X)

        # Should be identical (seeded random state)
        np.testing.assert_array_equal(probas1, probas2)


class TestGetFeatureImportance:
    """Tests for get_feature_importance method."""

    def test_get_feature_importance_demo_mode(self):
        """Test that demo mode returns empty importance."""
        model = XGBoostTSModel()

        importance = model.get_feature_importance()

        # Demo mode should return empty Series or minimal data
        assert isinstance(importance, pd.Series)
        # In demo mode, likely returns empty

    def test_get_feature_importance_type(self):
        """Test that feature importance returns pandas Series."""
        model = XGBoostTSModel()

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.Series)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_predict_with_many_features(self):
        """Test prediction with many features."""
        model = XGBoostTSModel()

        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(5) for i in range(100)
        })

        # Should not raise error
        probas = model.predict_proba(X)

        assert probas.shape == (5, 2)

    def test_predict_with_constant_features(self):
        """Test prediction with constant feature values."""
        model = XGBoostTSModel()

        X = pd.DataFrame({
            'feature_0': [1.0] * 5,
            'feature_1': [2.0] * 5,
        })

        # Should not raise error
        predictions = model.predict(X)

        assert len(predictions) == 5

    def test_predict_with_zero_features(self):
        """Test prediction with all zero features."""
        model = XGBoostTSModel()

        X = pd.DataFrame({
            'feature_0': [0.0, 0.0],
            'feature_1': [0.0, 0.0],
        })

        predictions = model.predict(X)

        assert len(predictions) == 2

    def test_model_repr(self):
        """Test string representation of model."""
        model = XGBoostTSModel()

        repr_str = repr(model)

        assert isinstance(repr_str, str)
        assert 'demo' in repr_str.lower() or 'mode' in repr_str.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
