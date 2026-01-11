"""
Unit tests for the qSOFA model.

Tests score calculation, predictions, probabilities,
and handling of missing columns.
"""

import pytest
import pandas as pd
import numpy as np

from models.qsofa.qsofa_model import QSOFAModel


class TestQSOFAModelInitialization:
    """Tests for QSOFAModel initialization."""

    def test_qsofa_init_default_parameters(self):
        """Test initialization with default parameters."""
        model = QSOFAModel()

        assert model.threshold == 2
        assert model.resp_rate_threshold == 22.0
        assert model.sbp_threshold == 100.0
        assert model.gcs_threshold == 15.0

    def test_qsofa_init_custom_threshold(self):
        """Test initialization with custom threshold."""
        model = QSOFAModel(threshold=1)

        assert model.threshold == 1

    def test_qsofa_init_invalid_threshold(self):
        """Test that invalid thresholds raise ValueError."""
        with pytest.raises(ValueError, match="Threshold must be between 0 and 3"):
            QSOFAModel(threshold=-1)

        with pytest.raises(ValueError, match="Threshold must be between 0 and 3"):
            QSOFAModel(threshold=4)


class TestCalculateScore:
    """Tests for calculate_score method."""

    def test_calculate_score_returns_array(self):
        """Test that calculate_score returns a numpy array."""
        model = QSOFAModel()
        df = pd.DataFrame({
            'Resp': [20.0, 22.0, 25.0],
            'SBP': [100.0, 95.0, 110.0],
        })

        scores = model.calculate_score(df)

        assert isinstance(scores, np.ndarray)

    def test_calculate_score_returns_0_to_3(self):
        """Test that scores are in range 0-3."""
        model = QSOFAModel()
        df = pd.DataFrame({
            'Resp': [20.0, 22.0, 30.0],
            'SBP': [100.0, 95.0, 90.0],
        })

        scores = model.calculate_score(df)

        # All scores should be between 0 and 3
        assert np.all(scores >= 0)
        assert np.all(scores <= 3)

    def test_calculate_score_respiratory_criterion(self):
        """Test respiratory rate criterion (>= 22)."""
        model = QSOFAModel()

        # Below threshold
        df_low = pd.DataFrame({
            'Resp': [20.0],
        })
        score_low = model.calculate_score(df_low)[0]

        # At threshold
        df_at = pd.DataFrame({
            'Resp': [22.0],
        })
        score_at = model.calculate_score(df_at)[0]

        # Above threshold
        df_high = pd.DataFrame({
            'Resp': [25.0],
        })
        score_high = model.calculate_score(df_high)[0]

        # Score should increase when criterion is met
        assert score_at > score_low
        assert score_high >= score_at

    def test_calculate_score_sbp_criterion(self):
        """Test systolic BP criterion (<= 100)."""
        model = QSOFAModel()

        # Above threshold
        df_high = pd.DataFrame({
            'SBP': [110.0],
        })
        score_high = model.calculate_score(df_high)[0]

        # At threshold
        df_at = pd.DataFrame({
            'SBP': [100.0],
        })
        score_at = model.calculate_score(df_at)[0]

        # Below threshold
        df_low = pd.DataFrame({
            'SBP': [90.0],
        })
        score_low = model.calculate_score(df_low)[0]

        # Score should increase when criterion is met
        assert score_at > score_high
        assert score_low >= score_at

    def test_calculate_score_all_criteria_met(self):
        """Test score when all three criteria are met."""
        model = QSOFAModel()

        df = pd.DataFrame({
            'Resp': [25.0],      # >= 22
            'SBP': [90.0],       # <= 100
            'GCS': [14.0],       # < 15
        })

        scores = model.calculate_score(df)

        # All three criteria met, score should be 3
        assert scores[0] == 3

    def test_calculate_score_no_criteria_met(self):
        """Test score when no criteria are met."""
        model = QSOFAModel()

        df = pd.DataFrame({
            'Resp': [18.0],      # < 22
            'SBP': [110.0],      # > 100
        })

        scores = model.calculate_score(df)

        # No criteria met, score should be 0
        assert scores[0] == 0

    def test_calculate_score_handles_nan(self):
        """Test that NaN values are handled gracefully."""
        model = QSOFAModel()

        df = pd.DataFrame({
            'Resp': [np.nan],
            'SBP': [95.0],
        })

        # Should not raise error
        scores = model.calculate_score(df)

        # Score should not count NaN criterion
        assert scores[0] == 1  # Only SBP criterion met

    def test_calculate_score_empty_dataframe(self):
        """Test with empty DataFrame."""
        model = QSOFAModel()

        df = pd.DataFrame()

        scores = model.calculate_score(df)

        assert len(scores) == 0

    def test_calculate_score_missing_columns(self):
        """Test with missing column names."""
        model = QSOFAModel()

        # DataFrame with only some columns
        df = pd.DataFrame({
            'Resp': [25.0],
            'SBP': [95.0],
            # No GCS column
        })

        # Should not raise error, GCS criterion = 0
        scores = model.calculate_score(df)

        # Only Resp and SBP criteria can be met
        assert 0 <= scores[0] <= 2


class TestPredict:
    """Tests for predict method."""

    def test_predict_returns_binary(self):
        """Test that predict returns binary predictions."""
        model = QSOFAModel(threshold=2)
        df = pd.DataFrame({
            'Resp': [20.0, 25.0, 30.0],
            'SBP': [100.0, 95.0, 85.0],
        })

        predictions = model.predict(df)

        # All predictions should be 0 or 1
        assert np.all((predictions == 0) | (predictions == 1))

    def test_predict_threshold_2_default(self):
        """Test predict with default threshold=2."""
        model = QSOFAModel(threshold=2)

        # Score 2: should predict 1
        df_2 = pd.DataFrame({
            'Resp': [25.0],      # Criterion met
            'SBP': [95.0],       # Criterion met
            'GCS': [15.0],       # Not met
        })
        pred_2 = model.predict(df_2)[0]

        # Score 1: should predict 0
        df_1 = pd.DataFrame({
            'Resp': [25.0],      # Criterion met
            'SBP': [110.0],      # Not met
        })
        pred_1 = model.predict(df_1)[0]

        assert pred_2 == 1
        assert pred_1 == 0

    def test_predict_different_thresholds(self):
        """Test predict with different thresholds."""
        # Create a dataset with score=1
        df = pd.DataFrame({
            'Resp': [25.0],      # Criterion met
            'SBP': [110.0],      # Not met
        })

        # Threshold 0: score >= 0 -> predict 1
        model_0 = QSOFAModel(threshold=0)
        pred_0 = model_0.predict(df)[0]

        # Threshold 2: score < 2 -> predict 0
        model_2 = QSOFAModel(threshold=2)
        pred_2 = model_2.predict(df)[0]

        assert pred_0 == 1
        assert pred_2 == 0


class TestPredictProba:
    """Tests for predict_proba method."""

    def test_predict_proba_shape(self):
        """Test that predict_proba returns correct shape."""
        model = QSOFAModel()
        df = pd.DataFrame({
            'Resp': [20.0, 22.0, 25.0],
            'SBP': [100.0, 95.0, 85.0],
        })

        probas = model.predict_proba(df)

        # Should be (n_samples, 2)
        assert probas.shape == (3, 2)

    def test_predict_proba_valid_probabilities(self):
        """Test that probabilities are valid (sum to 1, in [0,1])."""
        model = QSOFAModel()
        df = pd.DataFrame({
            'Resp': [20.0, 22.0, 25.0],
            'SBP': [100.0, 95.0, 85.0],
        })

        probas = model.predict_proba(df)

        # Each row should sum to ~1
        row_sums = probas.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

        # All values should be in [0, 1]
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_predict_proba_ranges(self):
        """Test that probabilities range from 0 to 1."""
        model = QSOFAModel()

        # Score 0: prob_positive should be 0/3 = 0
        df_0 = pd.DataFrame({
            'Resp': [20.0],
            'SBP': [110.0],
        })
        probas_0 = model.predict_proba(df_0)[0]

        # Score 3: prob_positive should be 3/3 = 1
        df_3 = pd.DataFrame({
            'Resp': [25.0],
            'SBP': [95.0],
            'GCS': [14.0],
        })
        probas_3 = model.predict_proba(df_3)[0]

        # Score 1: prob_positive should be 1/3
        df_1 = pd.DataFrame({
            'Resp': [25.0],
            'SBP': [110.0],
        })
        probas_1 = model.predict_proba(df_1)[0]

        assert probas_0[1] == pytest.approx(0.0)
        assert probas_3[1] == pytest.approx(1.0)
        assert probas_1[1] == pytest.approx(1/3, rel=1e-5)

    def test_predict_proba_complementary(self):
        """Test that P(class 0) + P(class 1) = 1."""
        model = QSOFAModel()
        df = pd.DataFrame({
            'Resp': [20.0, 22.0, 25.0],
            'SBP': [100.0, 95.0, 85.0],
        })

        probas = model.predict_proba(df)

        # Check complementary property
        prob_sum = probas[:, 0] + probas[:, 1]
        assert np.allclose(prob_sum, 1.0)


class TestHandlingMissingColumns:
    """Tests for handling of missing columns."""

    def test_missing_resp_rate_column(self):
        """Test with missing respiratory rate column."""
        model = QSOFAModel()

        df = pd.DataFrame({
            'SBP': [95.0],
            'GCS': [14.0],
            # Missing Resp
        })

        # Should not raise error
        scores = model.calculate_score(df)
        predictions = model.predict(df)

        assert len(scores) == 1
        assert len(predictions) == 1

    def test_missing_sbp_column(self):
        """Test with missing SBP column."""
        model = QSOFAModel()

        df = pd.DataFrame({
            'Resp': [25.0],
            'GCS': [14.0],
            # Missing SBP
        })

        # Should not raise error
        scores = model.calculate_score(df)

        assert len(scores) == 1

    def test_all_columns_missing(self):
        """Test with all criterion columns missing."""
        model = QSOFAModel()

        df = pd.DataFrame({
            'OtherColumn': [1.0],
        })

        # Should return score 0
        scores = model.calculate_score(df)

        assert scores[0] == 0

    def test_alternative_column_names(self):
        """Test that alternative column name formats work."""
        model = QSOFAModel()

        # Test with 'respiratory_rate' instead of 'Resp'
        df = pd.DataFrame({
            'respiratory_rate': [25.0],
            'SBP': [95.0],
        })

        scores = model.calculate_score(df)

        # Should recognize 'respiratory_rate' as respiratory criterion
        assert scores[0] >= 1  # At least respiratory criterion met

    def test_get_criteria_breakdown(self):
        """Test getting individual criterion values."""
        model = QSOFAModel()

        df = pd.DataFrame({
            'Resp': [25.0],
            'SBP': [95.0],
            'GCS': [14.0],
        })

        resp_c, sbp_c, gcs_c = model.get_criteria_breakdown(df)

        # All criteria should be met
        assert resp_c[0] == 1
        assert sbp_c[0] == 1
        assert gcs_c[0] == 1

    def test_get_feature_importance(self):
        """Test feature importance scores."""
        model = QSOFAModel()

        importance = model.get_feature_importance()

        # qSOFA has equal weights
        assert len(importance) == 3
        assert all(v == pytest.approx(1/3) for v in importance.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
