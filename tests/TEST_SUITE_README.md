# SepsisPulse Comprehensive Unit Test Suite

## Overview

This comprehensive unit test suite covers all core modules of the SepsisPulse project, including data loading, preprocessing, model inference, clinical utility calculations, and memory constraints.

**Total Test Files:** 7
**Total Test Cases:** 150+
**Total Lines of Code:** 2,777

## Test Files

### 1. test_data_loader.py (364 lines)
Tests for the data loading module (`src/data/loader.py`)

**Coverage:**
- `test_data_loader.py::TestLoadPatient` - 7 tests
  - Loading valid PSV files
  - Handling missing values (NaN)
  - Multiple rows of patient data
  - File not found errors
  - Invalid file extensions
  - Missing columns validation
  - Column order preservation

- `test_data_loader.py::TestLoadDataset` - 5 tests
  - Loading multiple patients from directory
  - max_patients parameter limiting
  - Directory not found errors
  - No PSV files found handling
  - Invalid files skipping

- `test_data_loader.py::TestGetSampleSubset` - 3 tests
  - Returns dictionary structure
  - Contains patient data
  - Consistency across calls

- `test_data_loader.py::TestGetPatientIds` - 2 tests
  - Returns sorted list
  - Empty directory handling

- `test_data_loader.py::TestGetColumnGroups` - 3 tests
  - Dictionary structure validation
  - All columns included
  - No duplicate columns

### 2. test_preprocessor.py (395 lines)
Tests for the preprocessing module (`src/data/preprocessor.py`)

**Coverage:**
- `test_preprocessor.py::TestHandleMissingValues` - 7 tests
  - Removes all NaN values
  - Forward fill strategy
  - Backward fill for initial NaN
  - Population means imputation
  - Unknown columns handling
  - Shape preservation
  - Input immutability

- `test_preprocessor.py::TestNormalizeFeatures` - 6 tests
  - Z-score normalization
  - Formula correctness
  - Unknown columns handling
  - Input immutability
  - Shape preservation
  - Zero standard deviation handling

- `test_preprocessor.py::TestPreprocessPatient` - 5 tests
  - Full pipeline removes NaN
  - With normalization
  - Without normalization
  - Output validation
  - Shape preservation

- `test_preprocessor.py::TestGetFeatureColumns` - 2 tests
  - Returns list
  - Includes vitals and labs

- `test_preprocessor.py::TestGetMissingRate` - 4 tests
  - Returns pandas Series
  - Correct calculations
  - All missing columns
  - No missing columns

- `test_preprocessor.py::TestValidatePreprocessedData` - 3 tests
  - Clean data validation
  - Rejects NaN values
  - Rejects infinite values

### 3. test_qsofa.py (424 lines)
Tests for the qSOFA model (`models/qsofa/qsofa_model.py`)

**Coverage:**
- `test_qsofa.py::TestQSOFAModelInitialization` - 3 tests
  - Default parameters
  - Custom threshold
  - Invalid threshold validation

- `test_qsofa.py::TestCalculateScore` - 9 tests
  - Returns numpy array
  - Scores in 0-3 range
  - Respiratory criterion (>= 22 bpm)
  - Systolic BP criterion (<= 100 mmHg)
  - GCS criterion (< 15)
  - All criteria met (score=3)
  - No criteria met (score=0)
  - NaN handling
  - Empty DataFrame
  - Missing columns

- `test_qsofa.py::TestPredict` - 3 tests
  - Binary predictions (0 or 1)
  - Default threshold=2
  - Different thresholds

- `test_qsofa.py::TestPredictProba` - 4 tests
  - Shape (n_samples, 2)
  - Valid probabilities
  - Probability ranges 0-1
  - Complementary probabilities

- `test_qsofa.py::TestHandlingMissingColumns` - 3 tests
  - Missing respiratory rate
  - Missing SBP
  - Alternative column names
  - Criteria breakdown
  - Feature importance

### 4. test_xgboost.py (381 lines)
Tests for XGBoost model (`models/xgboost_ts/xgboost_model.py`)

**Coverage:**
- `test_xgboost.py::TestXGBoostModelInitialization` - 3 tests
  - Demo mode activation
  - Attribute initialization
  - Model info retrieval

- `test_xgboost.py::TestPredictProba` - 8 tests
  - Shape (n_samples, 2)
  - Valid probability range [0, 1]
  - Rows sum to 1
  - Single sample handling
  - Empty DataFrame
  - Deterministic demo mode
  - NaN handling
  - Large dataset efficiency

- `test_xgboost.py::TestPredict` - 7 tests
  - Binary output
  - Int32 dtype
  - Default threshold 0.5
  - Custom threshold
  - Invalid threshold validation
  - Empty DataFrame
  - Single sample

- `test_xgboost.py::TestDemoMode` - 3 tests
  - Valid probability generation
  - Input data influence
  - Consistent results

- `test_xgboost.py::TestGetFeatureImportance` - 2 tests
  - Demo mode returns empty Series
  - Returns pandas Series type

- `test_xgboost.py::TestEdgeCases` - 4 tests
  - Many features (100+)
  - Constant feature values
  - Zero features
  - Model representation

### 5. test_tft_lite.py (388 lines)
Tests for TFT-Lite model (`models/tft_lite/tft_model.py`)

**Coverage:**
- `test_tft_lite.py::TestTFTLiteModelInitialization` - 4 tests
  - Demo mode activation
  - CPU device
  - CUDA fallback to CPU
  - Custom configuration
  - Model info

- `test_tft_lite.py::TestPredictProba` - 7 tests
  - Single sample (2D input) -> scalar
  - Batch (3D input) -> 1D array
  - Probability range [0, 1]
  - Zero input handling
  - Large values handling
  - Invalid shapes
  - Shape validation

- `test_tft_lite.py::TestPredict` - 4 tests
  - Binary predictions
  - Int32 dtype
  - Custom threshold
  - Threshold differences

- `test_tft_lite.py::TestForwardPass` - 3 tests
  - Shape preservation
  - Deterministic in eval mode
  - Different inputs -> different outputs

- `test_tft_lite.py::TestDemoMode` - 3 tests
  - Demo mode enabled
  - Valid predictions
  - Consistency

- `test_tft_lite.py::TestInterpretability` - 3 tests
  - Attention weights extraction
  - Variable importance
  - Feature importance summary

- `test_tft_lite.py::TestEdgeCases` - 3 tests
  - NaN handling
  - Infinite value handling
  - Very short sequences

- `test_tft_lite.py::TestMemoryEfficiency` - 2 tests
  - Eval mode confirmation
  - No gradient computation

### 6. test_clinical_utility.py (422 lines)
Tests for clinical utility module (`src/evaluation/clinical_utility.py`)

**Coverage:**
- `test_clinical_utility.py::TestUtilityFunction` - 7 tests
  - Optimal early window (6-12 hours) = 1.0
  - Optimal late window (0-6 hours) = 0.5
  - Too early (>12 hours) = -0.05
  - Missed sepsis (after onset) = -2.0
  - False positive = -0.05
  - Boundary cases

- `test_clinical_utility.py::TestComputePerPatientUtility` - 6 tests
  - Optimal early prediction
  - Optimal late prediction
  - False positive
  - True negative
  - Missed sepsis
  - Too early prediction

- `test_clinical_utility.py::TestComputeUtilityScore` - 6 tests
  - Perfect predictions -> 1.0
  - Worst predictions -> 0.0
  - Mixed predictions
  - Single patient
  - Mismatched patient IDs error
  - Empty dictionaries

- `test_clinical_utility.py::TestComputeUtilityComponents` - 3 tests
  - Structure validation
  - Specific values
  - Count aggregation

- `test_clinical_utility.py::TestEdgeCases` - 4 tests
  - Zero time
  - Large time values
  - Long ICU stays
  - Single hour data

### 7. test_memory.py (403 lines)
Tests for memory constraints and efficiency (`/test_memory.py`)

**Coverage:**
- `test_memory.py::TestQSOFAMemory` - 2 tests
  - Initialization < 500MB
  - Large batch inference < 500MB

- `test_memory.py::TestXGBoostMemory` - 3 tests
  - Initialization < 500MB
  - Demo mode efficiency
  - Batch inference

- `test_memory.py::TestTFTLiteMemory` - 3 tests
  - Initialization < 500MB
  - Batch inference < 500MB
  - Large batches < 500MB

- `test_memory.py::TestCombinedInference` - 3 tests
  - All models combined < 1.5GB
  - Sequential inference stability
  - Demo mode efficiency

- `test_memory.py::TestMemoryStress` - 3 tests
  - qSOFA with 100k samples
  - XGBoost with 200 features
  - TFT-Lite with large batches

- `test_memory.py::TestMemoryLeaks` - 3 tests
  - qSOFA repeated calls
  - XGBoost repeated calls
  - TFT-Lite repeated calls

## Key Features

### Comprehensive Coverage
- **All core modules tested**: Data loading, preprocessing, models, utilities
- **150+ test cases** covering functionality, edge cases, and error handling
- **Boundary condition testing** for all numerical thresholds
- **Demo mode validation** for all models

### Memory Constraints
- Individual models must load < **500MB**
- Combined inference must stay < **1.5GB**
- Memory leak detection over repeated calls
- Stress tests with large datasets

### Demo Mode Testing
- All models tested without trained weights
- Deterministic random predictions
- Valid probability ranges
- Memory efficiency in demo mode

### Error Handling
- File not found scenarios
- Invalid input formats
- Missing columns/data
- Invalid thresholds
- Shape mismatches

## Running Tests

### Install Dependencies
```bash
pip install pytest pandas numpy torch scikit-learn psutil
```

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_data_loader.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_qsofa.py::TestCalculateScore -v
```

### Run with Coverage Report
```bash
pytest tests/ --cov=src --cov=models --cov-report=html
```

### Run Memory Tests Only
```bash
pytest tests/test_memory.py -v
```

## Test Organization

```
tests/
├── __init__.py
├── TEST_SUITE_README.md
├── test_data_loader.py       # Data loading tests
├── test_preprocessor.py      # Preprocessing tests
├── test_qsofa.py             # qSOFA model tests
├── test_xgboost.py           # XGBoost model tests
├── test_tft_lite.py          # TFT-Lite model tests
├── test_clinical_utility.py  # Clinical utility tests
└── test_memory.py            # Memory constraint tests
```

## Test Conventions

- **Test class naming**: `Test<ModuleName>` (e.g., `TestLoadPatient`)
- **Test method naming**: `test_<functionality>` (e.g., `test_calculate_score_returns_0_to_3`)
- **Assertion style**: Using pytest assertions for clarity
- **Fixtures**: Temporary files created and cleaned up with context managers
- **Mocking**: Uses real data when possible, synthetic when needed

## Expected Results

All tests should pass with:
- ✓ No errors
- ✓ Memory usage within limits
- ✓ All assertions passing
- ✓ Edge cases handled gracefully
- ✓ Demo modes working correctly

## Notes

- Tests use temporary files that are automatically cleaned up
- Memory tests measure process RSS (resident set size)
- All models tested in demo mode (no trained weights required)
- Sample data available in `data/sample/patients/`
- Tests are independent and can be run in any order
