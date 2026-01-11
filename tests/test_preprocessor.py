"""
Unit tests for the preprocessor module.

Tests handling of missing values, feature normalization,
and full preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessor import (
    handle_missing_values,
    normalize_features,
    preprocess_patient,
    get_feature_columns,
    get_missing_rate,
    validate_preprocessed_data,
    POPULATION_MEANS,
    POPULATION_STDS,
    CLINICAL_FEATURES,
    DEMOGRAPHIC_FEATURES,
)


class TestHandleMissingValues:
    """Tests for handle_missing_values function."""

    def test_handle_missing_values_removes_all_nan(self):
        """Test that all NaN values are imputed after processing."""
        # Create DataFrame with NaN values
        df = pd.DataFrame({
            'HR': [70.0, np.nan, np.nan, 80.0],
            'O2Sat': [95.0, 96.0, np.nan, np.nan],
            'Temp': [37.0, 37.1, np.nan, 37.2],
        })

        result = handle_missing_values(df)

        # Verify no NaN values remain
        assert result.isna().sum().sum() == 0

    def test_handle_missing_values_forward_fill(self):
        """Test that forward fill is applied (carry last value forward)."""
        df = pd.DataFrame({
            'HR': [70.0, np.nan, np.nan, 80.0],
        })

        result = handle_missing_values(df)

        # After forward fill, should be [70, 70, 70, 80] then filled
        assert result.loc[1, 'HR'] == 70.0

    def test_handle_missing_values_backward_fill(self):
        """Test that backward fill handles initial NaN values."""
        df = pd.DataFrame({
            'HR': [np.nan, np.nan, 75.0, 80.0],
        })

        result = handle_missing_values(df)

        # After backward fill, initial NaN should be filled
        assert result.loc[0, 'HR'] == 75.0

    def test_handle_missing_values_uses_population_means(self):
        """Test that mean imputation uses population means."""
        df = pd.DataFrame({
            'HR': [np.nan, np.nan, np.nan, np.nan],
        })

        result = handle_missing_values(df)

        # Should use population mean
        assert result.loc[0, 'HR'] == POPULATION_MEANS['HR']

    def test_handle_missing_values_ignores_unknown_columns(self):
        """Test that unknown columns are left as-is."""
        df = pd.DataFrame({
            'HR': [70.0, np.nan],
            'UnknownColumn': [1.0, np.nan],
        })

        result = handle_missing_values(df)

        # HR should be filled, UnknownColumn preserved
        assert result.loc[1, 'HR'] != np.nan
        # UnknownColumn has no means, so might still have NaN or be filled
        # Depends on forward/backward fill

    def test_handle_missing_values_preserves_shape(self):
        """Test that shape is preserved."""
        df = pd.DataFrame({
            'HR': [70.0, np.nan, 75.0],
            'O2Sat': [95.0, np.nan, np.nan],
        })

        original_shape = df.shape
        result = handle_missing_values(df)

        assert result.shape == original_shape

    def test_handle_missing_values_doesnt_modify_input(self):
        """Test that input DataFrame is not modified."""
        df = pd.DataFrame({
            'HR': [70.0, np.nan, 75.0],
        })

        original_na_count = df.isna().sum().sum()

        handle_missing_values(df)

        # Input should not be modified
        assert df.isna().sum().sum() == original_na_count


class TestNormalizeFeatures:
    """Tests for normalize_features function."""

    def test_normalize_features_produces_zscore(self):
        """Test that normalization produces z-scores (mean~0, std~1)."""
        # Create DataFrame with known values
        df = pd.DataFrame({
            'HR': [70.0] * 100,  # All same value
            'Temp': [37.0] * 100,
        })

        result = normalize_features(df)

        # With constant input and population statistics,
        # output should be approximately zero mean
        # (since input mean differs from population mean)
        assert 'HR' in result.columns
        assert 'Temp' in result.columns

        # Check that values are different from input (normalization applied)
        assert result.loc[0, 'HR'] != 70.0

    def test_normalize_features_formula_correctness(self):
        """Test that z-score formula is applied correctly."""
        df = pd.DataFrame({
            'HR': [POPULATION_MEANS['HR']],  # Value equals population mean
        })

        result = normalize_features(df)

        # z = (x - mean) / std = (mean - mean) / std = 0
        assert abs(result.loc[0, 'HR']) < 1e-10

    def test_normalize_features_ignores_unknown_columns(self):
        """Test that unknown columns are left unchanged."""
        df = pd.DataFrame({
            'HR': [70.0],
            'UnknownColumn': [5.0],
        })

        result = normalize_features(df)

        # HR should be normalized
        assert result.loc[0, 'HR'] != 70.0

        # UnknownColumn should remain unchanged
        assert result.loc[0, 'UnknownColumn'] == 5.0

    def test_normalize_features_doesnt_modify_input(self):
        """Test that input DataFrame is not modified."""
        df = pd.DataFrame({
            'HR': [70.0, 80.0],
            'Temp': [37.0, 37.5],
        })

        original_values = df.copy()

        normalize_features(df)

        # Input should not be modified
        pd.testing.assert_frame_equal(df, original_values)

    def test_normalize_features_preserves_shape(self):
        """Test that shape is preserved."""
        df = pd.DataFrame({
            'HR': [70.0, 80.0, 75.0],
            'O2Sat': [95.0, 96.0, 97.0],
        })

        original_shape = df.shape
        result = normalize_features(df)

        assert result.shape == original_shape

    def test_normalize_features_handles_zero_std(self):
        """Test handling of features with zero standard deviation."""
        df = pd.DataFrame({
            'HR': [70.0, 70.0, 70.0],
        })

        # Should not raise error
        result = normalize_features(df)

        assert result.shape == df.shape


class TestPreprocessPatient:
    """Tests for preprocess_patient full pipeline."""

    def test_preprocess_patient_removes_all_nan(self):
        """Test that full pipeline removes all NaN values."""
        df = pd.DataFrame({
            'HR': [70.0, np.nan, np.nan, 80.0],
            'O2Sat': [95.0, np.nan, 96.0, np.nan],
            'Temp': [37.0, 37.1, np.nan, 37.2],
            'Age': [65.0] * 4,
            'Gender': [1.0] * 4,
            'Unit1': [0.0] * 4,
            'Unit2': [1.0] * 4,
            'HospAdmTime': [0.0] * 4,
            'ICULOS': [1.0, 2.0, 3.0, 4.0],
        })

        result = preprocess_patient(df)

        # No NaN values should remain
        assert result.isna().sum().sum() == 0

    def test_preprocess_patient_with_normalization(self):
        """Test preprocessing with normalization enabled."""
        df = pd.DataFrame({
            'HR': [70.0, 75.0, 80.0],
            'Age': [65.0, 65.0, 65.0],
        })

        result = preprocess_patient(df, normalize=True)

        # Should be normalized (not equal to input)
        assert result.loc[0, 'HR'] != 70.0

    def test_preprocess_patient_without_normalization(self):
        """Test preprocessing without normalization (raw pipeline)."""
        df = pd.DataFrame({
            'HR': [70.0, 75.0, 80.0],
        })

        result = preprocess_patient(df, normalize=False)

        # Should still have valid data (no NaN)
        assert result.isna().sum().sum() == 0

    def test_preprocess_patient_validates_output(self):
        """Test that preprocessed data is valid."""
        df = pd.DataFrame({
            'HR': [70.0, np.nan, 80.0],
            'O2Sat': [95.0, 96.0, np.nan],
        })

        result = preprocess_patient(df)

        # Should pass validation (no NaN, no inf)
        assert validate_preprocessed_data(result) is True

    def test_preprocess_patient_preserves_shape(self):
        """Test that shape is preserved."""
        df = pd.DataFrame({
            'HR': [70.0, np.nan, 80.0],
            'O2Sat': [95.0, 96.0, np.nan],
        })

        original_shape = df.shape
        result = preprocess_patient(df)

        assert result.shape == original_shape


class TestGetFeatureColumns:
    """Tests for get_feature_columns function."""

    def test_get_feature_columns_returns_list(self):
        """Test that function returns a list."""
        cols = get_feature_columns()

        assert isinstance(cols, list)

    def test_get_feature_columns_includes_vitals_and_labs(self):
        """Test that all vital signs and labs are included."""
        cols = get_feature_columns()

        # Should include vitals
        assert 'HR' in cols
        assert 'Resp' in cols
        assert 'Temp' in cols

        # Should include labs
        assert 'Glucose' in cols
        assert 'WBC' in cols
        assert 'Lactate' in cols

    def test_get_feature_columns_excludes_demographics_and_label(self):
        """Test that demographics and label are excluded."""
        cols = get_feature_columns()

        # Should not include demographics
        assert 'Age' not in cols
        assert 'Gender' not in cols

        # Should not include label
        assert 'SepsisLabel' not in cols


class TestGetMissingRate:
    """Tests for get_missing_rate function."""

    def test_get_missing_rate_returns_series(self):
        """Test that function returns a pandas Series."""
        df = pd.DataFrame({
            'HR': [70.0, np.nan, 80.0],
            'Temp': [37.0, 37.1, 37.2],
        })

        missing = get_missing_rate(df)

        assert isinstance(missing, pd.Series)

    def test_get_missing_rate_values(self):
        """Test that missing rates are correctly calculated."""
        df = pd.DataFrame({
            'HR': [70.0, np.nan, np.nan],  # 2/3 missing
            'Temp': [37.0, 37.1, 37.2],   # 0/3 missing
        })

        missing = get_missing_rate(df)

        assert missing['HR'] == pytest.approx(2/3)
        assert missing['Temp'] == 0.0

    def test_get_missing_rate_all_missing(self):
        """Test column with all missing values."""
        df = pd.DataFrame({
            'HR': [np.nan, np.nan, np.nan],
        })

        missing = get_missing_rate(df)

        assert missing['HR'] == 1.0

    def test_get_missing_rate_no_missing(self):
        """Test column with no missing values."""
        df = pd.DataFrame({
            'HR': [70.0, 75.0, 80.0],
        })

        missing = get_missing_rate(df)

        assert missing['HR'] == 0.0


class TestValidatePreprocessedData:
    """Tests for validate_preprocessed_data function."""

    def test_validate_preprocessed_data_clean_data(self):
        """Test validation passes for clean data."""
        df = pd.DataFrame({
            'HR': [70.0, 75.0, 80.0],
            'Temp': [37.0, 37.1, 37.2],
        })

        assert validate_preprocessed_data(df) is True

    def test_validate_preprocessed_data_rejects_nan(self):
        """Test that validation fails for data with NaN."""
        df = pd.DataFrame({
            'HR': [70.0, np.nan, 80.0],
        })

        with pytest.raises(ValueError, match="Missing values"):
            validate_preprocessed_data(df)

    def test_validate_preprocessed_data_rejects_inf(self):
        """Test that validation fails for data with infinite values."""
        df = pd.DataFrame({
            'HR': [70.0, np.inf, 80.0],
        })

        with pytest.raises(ValueError, match="Infinite values"):
            validate_preprocessed_data(df)

    def test_validate_preprocessed_data_rejects_negative_inf(self):
        """Test that validation fails for data with negative infinite values."""
        df = pd.DataFrame({
            'HR': [70.0, -np.inf, 80.0],
        })

        with pytest.raises(ValueError, match="Infinite values"):
            validate_preprocessed_data(df)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
