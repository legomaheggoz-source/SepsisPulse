"""
Unit tests for the data loader module.

Tests loading of patient data from PSV files, dataset loading,
and sample data access.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.data.loader import (
    load_patient,
    load_dataset,
    get_sample_subset,
    get_patient_ids,
    get_column_groups,
    PHYSIONET_COLUMNS,
    VITAL_COLUMNS,
    LAB_COLUMNS,
    DEMOGRAPHIC_COLUMNS,
    LABEL_COLUMN,
)


class TestLoadPatient:
    """Tests for load_patient function."""

    def test_load_patient_with_valid_psv(self):
        """Test loading a valid PSV file."""
        # Create a temporary PSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.psv', delete=False) as f:
            # Write header
            f.write('|'.join(PHYSIONET_COLUMNS) + '\n')
            # Write one data row with all values
            values = [str(float(i)) for i in range(len(PHYSIONET_COLUMNS))]
            f.write('|'.join(values) + '\n')
            temp_file = f.name

        try:
            df = load_patient(temp_file)

            # Verify shape
            assert df.shape == (1, len(PHYSIONET_COLUMNS))

            # Verify all columns are present
            assert list(df.columns) == PHYSIONET_COLUMNS

            # Verify data types are float
            assert all(df.dtypes == float)

        finally:
            os.unlink(temp_file)

    def test_load_patient_handles_missing_values(self):
        """Test that missing values are preserved as NaN."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.psv', delete=False) as f:
            f.write('|'.join(PHYSIONET_COLUMNS) + '\n')
            # Write row with some missing values
            values = []
            for i, col in enumerate(PHYSIONET_COLUMNS):
                if i % 5 == 0:  # Every 5th column is NaN
                    values.append('')
                else:
                    values.append(str(float(i)))
            f.write('|'.join(values) + '\n')
            temp_file = f.name

        try:
            df = load_patient(temp_file)

            # Check that some NaN values are present
            assert df.isna().sum().sum() > 0

            # Check that non-NaN values are correct
            assert df.iloc[0, 1] == 1.0

        finally:
            os.unlink(temp_file)

    def test_load_patient_multiple_rows(self):
        """Test loading a patient file with multiple hours of data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.psv', delete=False) as f:
            f.write('|'.join(PHYSIONET_COLUMNS) + '\n')
            # Write 5 data rows
            for row_idx in range(5):
                values = [str(float(row_idx * 10 + i)) for i in range(len(PHYSIONET_COLUMNS))]
                f.write('|'.join(values) + '\n')
            temp_file = f.name

        try:
            df = load_patient(temp_file)

            # Verify shape
            assert df.shape == (5, len(PHYSIONET_COLUMNS))

            # Verify data is correctly loaded
            assert df.iloc[0, 0] == 0.0
            assert df.iloc[4, 0] == 40.0

        finally:
            os.unlink(temp_file)

    def test_load_patient_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_patient('/nonexistent/file.psv')

    def test_load_patient_invalid_extension(self):
        """Test that ValueError is raised for non-PSV files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('test')
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Expected .psv file"):
                load_patient(temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_patient_missing_columns(self):
        """Test that ValueError is raised when expected columns are missing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.psv', delete=False) as f:
            # Write header with missing columns
            cols = PHYSIONET_COLUMNS[:-5]  # Missing last 5 columns
            f.write('|'.join(cols) + '\n')
            values = [str(float(i)) for i in range(len(cols))]
            f.write('|'.join(values) + '\n')
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Missing expected columns"):
                load_patient(temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_patient_column_order(self):
        """Test that columns are reordered to match expected order."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.psv', delete=False) as f:
            # Write header in reversed order
            reversed_cols = PHYSIONET_COLUMNS[::-1]
            f.write('|'.join(reversed_cols) + '\n')
            # Write values in reversed order
            values = [str(float(len(PHYSIONET_COLUMNS) - 1 - i)) for i in range(len(PHYSIONET_COLUMNS))]
            f.write('|'.join(values) + '\n')
            temp_file = f.name

        try:
            df = load_patient(temp_file)

            # Verify columns are in expected order
            assert list(df.columns) == PHYSIONET_COLUMNS

        finally:
            os.unlink(temp_file)


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_dataset_from_directory(self):
        """Test loading multiple patient files from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 sample PSV files
            for patient_num in range(1, 4):
                filename = os.path.join(tmpdir, f'p0000{patient_num}.psv')
                with open(filename, 'w') as f:
                    f.write('|'.join(PHYSIONET_COLUMNS) + '\n')
                    values = [str(float(i)) for i in range(len(PHYSIONET_COLUMNS))]
                    f.write('|'.join(values) + '\n')

            # Load dataset
            dataset = load_dataset(tmpdir)

            # Verify number of patients
            assert len(dataset) == 3

            # Verify patient IDs
            assert 'p00001' in dataset
            assert 'p00002' in dataset
            assert 'p00003' in dataset

            # Verify each is a DataFrame
            for patient_id, df in dataset.items():
                assert isinstance(df, pd.DataFrame)
                assert df.shape[1] == len(PHYSIONET_COLUMNS)

    def test_load_dataset_max_patients(self):
        """Test max_patients parameter limits loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 10 sample PSV files
            for patient_num in range(1, 11):
                filename = os.path.join(tmpdir, f'p0000{patient_num:02d}.psv')
                with open(filename, 'w') as f:
                    f.write('|'.join(PHYSIONET_COLUMNS) + '\n')
                    values = [str(float(i)) for i in range(len(PHYSIONET_COLUMNS))]
                    f.write('|'.join(values) + '\n')

            # Load only 5 patients
            dataset = load_dataset(tmpdir, max_patients=5)

            assert len(dataset) == 5

    def test_load_dataset_directory_not_found(self):
        """Test that FileNotFoundError is raised for missing directory."""
        with pytest.raises(FileNotFoundError):
            load_dataset('/nonexistent/directory')

    def test_load_dataset_no_psv_files(self):
        """Test that ValueError is raised when no PSV files found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No .psv files found"):
                load_dataset(tmpdir)

    def test_load_dataset_skips_invalid_files(self):
        """Test that invalid files are skipped without stopping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 1 valid and 1 invalid file
            # Valid file
            valid_file = os.path.join(tmpdir, 'p00001.psv')
            with open(valid_file, 'w') as f:
                f.write('|'.join(PHYSIONET_COLUMNS) + '\n')
                values = [str(float(i)) for i in range(len(PHYSIONET_COLUMNS))]
                f.write('|'.join(values) + '\n')

            # Invalid file (missing columns)
            invalid_file = os.path.join(tmpdir, 'p00002.psv')
            with open(invalid_file, 'w') as f:
                f.write('invalid|data\n')
                f.write('1|2\n')

            # Load dataset - should skip invalid and still return valid
            dataset = load_dataset(tmpdir)

            assert len(dataset) == 1
            assert 'p00001' in dataset


class TestGetSampleSubset:
    """Tests for get_sample_subset function."""

    def test_get_sample_subset_returns_dict(self):
        """Test that get_sample_subset returns a dictionary."""
        sample = get_sample_subset()

        assert isinstance(sample, dict)

    def test_get_sample_subset_contains_patients(self):
        """Test that sample subset contains patient data."""
        sample = get_sample_subset()

        # Should have at least some patients
        if len(sample) > 0:
            # Verify structure
            for patient_id, df in sample.items():
                assert isinstance(patient_id, str)
                assert isinstance(df, pd.DataFrame)
                assert df.shape[1] == len(PHYSIONET_COLUMNS)

    def test_get_sample_subset_is_consistent(self):
        """Test that multiple calls return the same data."""
        sample1 = get_sample_subset()
        sample2 = get_sample_subset()

        if len(sample1) > 0:
            assert set(sample1.keys()) == set(sample2.keys())

            # Verify data is identical
            for patient_id in sample1.keys():
                pd.testing.assert_frame_equal(sample1[patient_id], sample2[patient_id])


class TestGetPatientIds:
    """Tests for get_patient_ids function."""

    def test_get_patient_ids_returns_list(self):
        """Test that get_patient_ids returns a sorted list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 sample PSV files in random order
            for patient_num in [3, 1, 2]:
                filename = os.path.join(tmpdir, f'p0000{patient_num}.psv')
                with open(filename, 'w') as f:
                    f.write('|'.join(PHYSIONET_COLUMNS) + '\n')
                    values = [str(float(i)) for i in range(len(PHYSIONET_COLUMNS))]
                    f.write('|'.join(values) + '\n')

            ids = get_patient_ids(tmpdir)

            # Verify type
            assert isinstance(ids, list)

            # Verify sorted
            assert ids == sorted(ids)

            # Verify content
            assert ids == ['p00001', 'p00002', 'p00003']

    def test_get_patient_ids_empty_directory(self):
        """Test that empty directory returns empty list or raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                get_patient_ids(tmpdir)


class TestGetColumnGroups:
    """Tests for get_column_groups function."""

    def test_get_column_groups_returns_dict(self):
        """Test that get_column_groups returns correct structure."""
        groups = get_column_groups()

        assert isinstance(groups, dict)
        assert 'vitals' in groups
        assert 'labs' in groups
        assert 'demographics' in groups
        assert 'label' in groups

    def test_get_column_groups_completeness(self):
        """Test that all columns are included in groups."""
        groups = get_column_groups()

        all_cols = []
        for col_list in groups.values():
            all_cols.extend(col_list)

        # Check all columns are present
        assert set(all_cols) == set(PHYSIONET_COLUMNS)

    def test_get_column_groups_specific_columns(self):
        """Test that specific columns are in expected groups."""
        groups = get_column_groups()

        # Check vital signs
        assert 'HR' in groups['vitals']
        assert 'Resp' in groups['vitals']

        # Check labs
        assert 'Glucose' in groups['labs']
        assert 'WBC' in groups['labs']

        # Check demographics
        assert 'Age' in groups['demographics']
        assert 'Gender' in groups['demographics']

        # Check label
        assert 'SepsisLabel' in groups['label']

    def test_get_column_groups_no_duplicates(self):
        """Test that columns don't appear in multiple groups."""
        groups = get_column_groups()

        all_cols = []
        for col_list in groups.values():
            all_cols.extend(col_list)

        # Check no duplicates
        assert len(all_cols) == len(set(all_cols))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
