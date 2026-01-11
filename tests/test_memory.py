"""
Unit tests for memory constraints and efficiency.

Tests that all models load and run inference within specified memory limits.
"""

import pytest
import psutil
import os
import numpy as np
import pandas as pd

from models.qsofa.qsofa_model import QSOFAModel
from models.xgboost_ts.xgboost_model import XGBoostTSModel
from models.tft_lite.tft_model import TFTLiteModel

# Memory limits in MB
MAX_SINGLE_MODEL_MB = 500
MAX_COMBINED_INFERENCE_MB = 1500


def get_process_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def measure_model_initialization_memory(model_init_func):
    """Measure memory used to initialize a model."""
    initial_memory = get_process_memory_mb()

    # Initialize model
    model = model_init_func()

    peak_memory = get_process_memory_mb()
    memory_used = peak_memory - initial_memory

    return model, memory_used


class TestQSOFAMemory:
    """Tests for qSOFA model memory usage."""

    def test_qsofa_initialization_memory(self):
        """Test qSOFA model initialization memory usage."""
        model, memory_used = measure_model_initialization_memory(QSOFAModel)

        # qSOFA is rule-based, should use minimal memory
        assert memory_used < MAX_SINGLE_MODEL_MB

    def test_qsofa_inference_memory(self):
        """Test qSOFA inference doesn't exceed memory limits."""
        model = QSOFAModel()
        initial_memory = get_process_memory_mb()

        # Create large batch of data
        df = pd.DataFrame({
            'Resp': np.random.randn(10000),
            'SBP': np.random.randn(10000),
            'GCS': np.random.randn(10000),
        })

        # Run inference
        predictions = model.predict(df)

        peak_memory = get_process_memory_mb()
        memory_used = peak_memory - initial_memory

        assert memory_used < MAX_SINGLE_MODEL_MB
        assert len(predictions) == 10000


class TestXGBoostMemory:
    """Tests for XGBoost model memory usage."""

    def test_xgboost_initialization_memory(self):
        """Test XGBoost model initialization memory usage."""
        model, memory_used = measure_model_initialization_memory(XGBoostTSModel)

        # XGBoost should load efficiently in demo mode
        assert memory_used < MAX_SINGLE_MODEL_MB

    def test_xgboost_demo_mode_memory(self):
        """Test XGBoost in demo mode uses reasonable memory."""
        model = XGBoostTSModel()

        # Should be in demo mode
        assert model.is_demo_mode is True

        initial_memory = get_process_memory_mb()

        # Create and predict on large dataset
        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(5000) for i in range(50)
        })

        predictions = model.predict(X)

        peak_memory = get_process_memory_mb()
        memory_used = peak_memory - initial_memory

        # Demo mode should be memory efficient
        assert memory_used < MAX_SINGLE_MODEL_MB
        assert len(predictions) == 5000

    def test_xgboost_batch_inference_memory(self):
        """Test XGBoost batch inference memory usage."""
        model = XGBoostTSModel()

        initial_memory = get_process_memory_mb()

        # Multiple large batches
        for batch in range(3):
            X = pd.DataFrame({
                f'feature_{i}': np.random.randn(1000) for i in range(20)
            })
            predictions = model.predict(X)

        peak_memory = get_process_memory_mb()
        memory_used = peak_memory - initial_memory

        assert memory_used < MAX_SINGLE_MODEL_MB


class TestTFTLiteMemory:
    """Tests for TFT-Lite model memory usage."""

    def test_tft_lite_initialization_memory(self):
        """Test TFT-Lite model initialization memory usage."""
        def init_tft():
            return TFTLiteModel(input_size=41, hidden_size=32)

        model, memory_used = measure_model_initialization_memory(init_tft)

        # TFT-Lite is optimized for low memory
        assert memory_used < MAX_SINGLE_MODEL_MB

    def test_tft_lite_inference_memory(self):
        """Test TFT-Lite inference doesn't exceed memory limits."""
        model = TFTLiteModel(input_size=41, max_seq_length=24)
        initial_memory = get_process_memory_mb()

        # Create batch predictions
        X = np.random.randn(100, 24, 41).astype(np.float32)

        predictions = model.predict(X)

        peak_memory = get_process_memory_mb()
        memory_used = peak_memory - initial_memory

        assert memory_used < MAX_SINGLE_MODEL_MB
        assert len(predictions) == 100

    def test_tft_lite_large_batch_memory(self):
        """Test TFT-Lite with large batch size."""
        model = TFTLiteModel(input_size=41, max_seq_length=24)

        initial_memory = get_process_memory_mb()

        # Large batch
        X = np.random.randn(500, 24, 41).astype(np.float32)

        predictions = model.predict(X)

        peak_memory = get_process_memory_mb()
        memory_used = peak_memory - initial_memory

        # Even with large batch, should be efficient
        assert memory_used < MAX_SINGLE_MODEL_MB
        assert len(predictions) == 500


class TestCombinedInference:
    """Tests for combined inference across all models."""

    def test_all_models_combined_memory(self):
        """Test loading and running all models together."""
        initial_memory = get_process_memory_mb()

        # Initialize all models
        qsofa_model = QSOFAModel()
        xgb_model = XGBoostTSModel()
        tft_model = TFTLiteModel(input_size=41, max_seq_length=24)

        after_init_memory = get_process_memory_mb()
        init_memory_used = after_init_memory - initial_memory

        # All models should fit in memory
        assert init_memory_used < MAX_COMBINED_INFERENCE_MB

        # Run inference on all models
        qsofa_data = pd.DataFrame({
            'Resp': np.random.randn(100),
            'SBP': np.random.randn(100),
        })

        xgb_data = pd.DataFrame({
            f'feature_{i}': np.random.randn(100) for i in range(30)
        })

        tft_data = np.random.randn(100, 24, 41).astype(np.float32)

        qsofa_pred = qsofa_model.predict(qsofa_data)
        xgb_pred = xgb_model.predict(xgb_data)
        tft_pred = tft_model.predict(tft_data)

        peak_memory = get_process_memory_mb()
        total_memory_used = peak_memory - initial_memory

        # Combined inference should stay within budget
        assert total_memory_used < MAX_COMBINED_INFERENCE_MB

        # Verify predictions were generated
        assert len(qsofa_pred) == 100
        assert len(xgb_pred) == 100
        assert len(tft_pred) == 100

    def test_sequential_inference_memory(self):
        """Test memory efficiency of sequential inference."""
        # Initialize all models
        qsofa_model = QSOFAModel()
        xgb_model = XGBoostTSModel()
        tft_model = TFTLiteModel(input_size=41, max_seq_length=24)

        initial_memory = get_process_memory_mb()

        # Sequential inference on multiple samples
        for i in range(10):
            qsofa_data = pd.DataFrame({
                'Resp': np.random.randn(50),
                'SBP': np.random.randn(50),
            })

            xgb_data = pd.DataFrame({
                f'feature_{j}': np.random.randn(50) for j in range(20)
            })

            tft_data = np.random.randn(50, 24, 41).astype(np.float32)

            qsofa_pred = qsofa_model.predict(qsofa_data)
            xgb_pred = xgb_model.predict(xgb_data)
            tft_pred = tft_model.predict(tft_data)

        peak_memory = get_process_memory_mb()
        memory_used = peak_memory - initial_memory

        # Should not accumulate memory over sequential calls
        assert memory_used < MAX_COMBINED_INFERENCE_MB

    def test_demo_mode_memory_efficiency(self):
        """Test that demo modes are memory efficient."""
        initial_memory = get_process_memory_mb()

        # All models in demo mode
        xgb_model = XGBoostTSModel()
        tft_model = TFTLiteModel()

        assert xgb_model.is_demo_mode is True
        assert tft_model.is_demo_mode is True

        peak_memory = get_process_memory_mb()
        memory_used = peak_memory - initial_memory

        # Demo modes should use minimal memory
        assert memory_used < 200  # Well under limit


class TestMemoryStress:
    """Stress tests for memory efficiency."""

    def test_large_qsofa_inference(self):
        """Test qSOFA with very large dataset."""
        model = QSOFAModel()
        initial_memory = get_process_memory_mb()

        # 100,000 samples
        df = pd.DataFrame({
            'Resp': np.random.randn(100000),
            'SBP': np.random.randn(100000),
        })

        predictions = model.predict(df)

        peak_memory = get_process_memory_mb()
        memory_used = peak_memory - initial_memory

        # Should handle large datasets efficiently
        assert memory_used < MAX_SINGLE_MODEL_MB
        assert len(predictions) == 100000

    def test_large_xgboost_inference(self):
        """Test XGBoost with very large feature set."""
        model = XGBoostTSModel()
        initial_memory = get_process_memory_mb()

        # Many features
        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(5000) for i in range(200)
        })

        predictions = model.predict(X)

        peak_memory = get_process_memory_mb()
        memory_used = peak_memory - initial_memory

        assert memory_used < MAX_SINGLE_MODEL_MB
        assert len(predictions) == 5000

    def test_large_tft_inference(self):
        """Test TFT-Lite with large batches and sequences."""
        model = TFTLiteModel(input_size=41, max_seq_length=48)
        initial_memory = get_process_memory_mb()

        # Large batch, longer sequences
        X = np.random.randn(200, 48, 41).astype(np.float32)

        predictions = model.predict(X)

        peak_memory = get_process_memory_mb()
        memory_used = peak_memory - initial_memory

        assert memory_used < MAX_SINGLE_MODEL_MB
        assert len(predictions) == 200


class TestMemoryLeaks:
    """Tests to detect potential memory leaks."""

    def test_repeated_qsofa_inference(self):
        """Test qSOFA doesn't leak memory over repeated calls."""
        model = QSOFAModel()

        initial_memory = get_process_memory_mb()
        memory_samples = []

        for i in range(50):
            df = pd.DataFrame({
                'Resp': np.random.randn(100),
                'SBP': np.random.randn(100),
            })

            predictions = model.predict(df)

            current_memory = get_process_memory_mb()
            memory_samples.append(current_memory)

        # Memory should stabilize, not grow continuously
        early_avg = np.mean(memory_samples[:10])
        late_avg = np.mean(memory_samples[-10:])

        # Should not grow significantly
        growth = late_avg - early_avg
        assert growth < 50  # Less than 50MB growth

    def test_repeated_xgboost_inference(self):
        """Test XGBoost doesn't leak memory over repeated calls."""
        model = XGBoostTSModel()

        initial_memory = get_process_memory_mb()
        memory_samples = []

        for i in range(50):
            X = pd.DataFrame({
                f'feature_{j}': np.random.randn(100) for j in range(20)
            })

            predictions = model.predict(X)

            current_memory = get_process_memory_mb()
            memory_samples.append(current_memory)

        # Memory should stabilize
        early_avg = np.mean(memory_samples[:10])
        late_avg = np.mean(memory_samples[-10:])

        growth = late_avg - early_avg
        assert growth < 50

    def test_repeated_tft_inference(self):
        """Test TFT-Lite doesn't leak memory over repeated calls."""
        model = TFTLiteModel(input_size=41, max_seq_length=24)

        initial_memory = get_process_memory_mb()
        memory_samples = []

        for i in range(50):
            X = np.random.randn(50, 24, 41).astype(np.float32)

            predictions = model.predict(X)

            current_memory = get_process_memory_mb()
            memory_samples.append(current_memory)

        # Memory should stabilize
        early_avg = np.mean(memory_samples[:10])
        late_avg = np.mean(memory_samples[-10:])

        growth = late_avg - early_avg
        assert growth < 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
