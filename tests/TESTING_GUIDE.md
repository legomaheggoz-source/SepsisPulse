# SepsisPulse Testing Guide

## Quick Start

### Install Test Dependencies
```bash
pip install pytest pandas numpy torch xgboost psutil
```

### Run All Tests
```bash
cd /c/SepsisPulse
pytest tests/ -v
```

### Run Specific Module Tests
```bash
pytest tests/test_data_loader.py -v
pytest tests/test_preprocessor.py -v
pytest tests/test_qsofa.py -v
pytest tests/test_xgboost.py -v
pytest tests/test_tft_lite.py -v
pytest tests/test_clinical_utility.py -v
pytest tests/test_memory.py -v
```

## Test Suite Summary

### Files Created

| File | Lines | Tests | Purpose |
|------|-------|-------|---------|
| test_data_loader.py | 364 | 20 | PSV file loading, dataset handling |
| test_preprocessor.py | 395 | 23 | Missing value handling, normalization |
| test_qsofa.py | 424 | 22 | qSOFA scoring, predictions, probabilities |
| test_xgboost.py | 381 | 23 | XGBoost inference, demo mode |
| test_tft_lite.py | 388 | 25 | TFT-Lite forward pass, predictions |
| test_clinical_utility.py | 422 | 26 | Utility functions, normalization |
| test_memory.py | 403 | 25 | Memory constraints, leak detection |
| **Total** | **2,777** | **150+** | **Comprehensive coverage** |

## Detailed Test Coverage

### 1. test_data_loader.py

**Module**: `src/data/loader.py`

**Test Classes**:
- `TestLoadPatient` (7 tests)
- `TestLoadDataset` (5 tests)
- `TestGetSampleSubset` (3 tests)
- `TestGetPatientIds` (2 tests)
- `TestGetColumnGroups` (3 tests)

**Key Tests**:
```python
✓ test_load_patient_with_valid_psv()           # Load PSV from file
✓ test_load_patient_handles_missing_values()   # NaN preservation
✓ test_load_patient_multiple_rows()            # Multiple hours of data
✓ test_load_patient_file_not_found()           # Error handling
✓ test_load_patient_missing_columns()          # Column validation
✓ test_load_dataset_from_directory()           # Multiple patient loading
✓ test_load_dataset_max_patients()             # Limit loading
✓ test_load_dataset_skips_invalid_files()      # Fault tolerance
✓ test_get_sample_subset_returns_dict()        # Sample data access
✓ test_get_column_groups_completeness()        # Metadata validation
```

**Requirements Met**:
- ✓ Test load_patient with valid PSV file
- ✓ Test load_dataset with sample directory
- ✓ Test get_sample_subset returns data
- ✓ Test handling of missing files

---

### 2. test_preprocessor.py

**Module**: `src/data/preprocessor.py`

**Test Classes**:
- `TestHandleMissingValues` (7 tests)
- `TestNormalizeFeatures` (6 tests)
- `TestPreprocessPatient` (5 tests)
- `TestGetFeatureColumns` (2 tests)
- `TestGetMissingRate` (4 tests)
- `TestValidatePreprocessedData` (3 tests)

**Key Tests**:
```python
✓ test_handle_missing_values_removes_all_nan()    # No NaN after processing
✓ test_handle_missing_values_forward_fill()       # LOCF imputation
✓ test_handle_missing_values_backward_fill()      # Initial NaN handling
✓ test_handle_missing_values_uses_population_means()  # Mean imputation
✓ test_normalize_features_produces_zscore()       # Z-score calculation
✓ test_normalize_features_formula_correctness()   # (x - mean) / std
✓ test_preprocess_patient_removes_all_nan()       # Full pipeline
✓ test_preprocess_patient_with_normalization()    # Optional normalization
✓ test_validate_preprocessed_data_clean_data()    # Validation passes
✓ test_validate_preprocessed_data_rejects_nan()   # Rejects NaN
```

**Requirements Met**:
- ✓ Test handle_missing_values removes all NaN
- ✓ Test normalize_features produces z-scores
- ✓ Test preprocess_patient full pipeline

---

### 3. test_qsofa.py

**Module**: `models/qsofa/qsofa_model.py`

**Test Classes**:
- `TestQSOFAModelInitialization` (3 tests)
- `TestCalculateScore` (9 tests)
- `TestPredict` (3 tests)
- `TestPredictProba` (4 tests)
- `TestHandlingMissingColumns` (6 tests)

**Key Tests**:
```python
✓ test_calculate_score_returns_0_to_3()          # Score range validation
✓ test_calculate_score_respiratory_criterion()   # Resp >= 22
✓ test_calculate_score_sbp_criterion()           # SBP <= 100
✓ test_calculate_score_all_criteria_met()        # Score = 3
✓ test_calculate_score_no_criteria_met()         # Score = 0
✓ test_calculate_score_handles_nan()             # NaN resilience
✓ test_predict_returns_binary()                  # Binary output
✓ test_predict_threshold_2_default()             # Threshold behavior
✓ test_predict_proba_shape()                     # Shape (n, 2)
✓ test_predict_proba_valid_probabilities()       # Sum to 1, in [0,1]
```

**Requirements Met**:
- ✓ Test calculate_score returns 0-3
- ✓ Test predict with threshold=2
- ✓ Test predict_proba returns valid probabilities
- ✓ Test handling of missing columns

---

### 4. test_xgboost.py

**Module**: `models/xgboost_ts/xgboost_model.py`

**Test Classes**:
- `TestXGBoostModelInitialization` (3 tests)
- `TestPredictProba` (8 tests)
- `TestPredict` (7 tests)
- `TestDemoMode` (3 tests)
- `TestGetFeatureImportance` (2 tests)
- `TestEdgeCases` (4 tests)

**Key Tests**:
```python
✓ test_xgboost_init_demo_mode()                  # Demo mode activation
✓ test_predict_proba_returns_correct_shape()     # Shape (n, 2)
✓ test_predict_proba_valid_probabilities()       # Probabilities valid
✓ test_predict_proba_empty_dataframe()           # Empty handling
✓ test_predict_returns_binary()                  # Binary output
✓ test_predict_invalid_threshold()               # Threshold validation
✓ test_demo_mode_deterministic()                 # Reproducibility
✓ test_demo_mode_respects_input_data()           # Data influence
✓ test_predict_with_many_features()              # Scalability
✓ test_predict_with_nan()                        # NaN resilience
```

**Requirements Met**:
- ✓ Test demo mode predictions
- ✓ Test predict returns binary
- ✓ Test predict_proba shape is (n, 2)

---

### 5. test_tft_lite.py

**Module**: `models/tft_lite/tft_model.py`

**Test Classes**:
- `TestTFTLiteModelInitialization` (4 tests)
- `TestPredictProba` (7 tests)
- `TestPredict` (4 tests)
- `TestForwardPass` (3 tests)
- `TestDemoMode` (3 tests)
- `TestInterpretability` (3 tests)
- `TestEdgeCases` (3 tests)
- `TestMemoryEfficiency` (2 tests)

**Key Tests**:
```python
✓ test_tft_init_demo_mode()                      # Demo mode activation
✓ test_predict_proba_single_sample()             # 2D input -> scalar
✓ test_predict_proba_batch()                     # 3D input -> 1D array
✓ test_predict_proba_valid_range()               # [0, 1] range
✓ test_predict_proba_invalid_shape()             # Shape validation
✓ test_predict_returns_binary()                  # Binary output
✓ test_forward_pass_shape_preservation()         # Shape consistency
✓ test_forward_pass_deterministic()              # Reproducibility
✓ test_demo_mode_predictions()                   # Valid probabilities
✓ test_get_attention_weights()                   # Interpretability
✓ test_predict_with_nan()                        # NaN handling
```

**Requirements Met**:
- ✓ Test forward pass shape
- ✓ Test predict returns binary
- ✓ Test demo mode works without weights

---

### 6. test_clinical_utility.py

**Module**: `src/evaluation/clinical_utility.py`

**Test Classes**:
- `TestUtilityFunction` (7 tests)
- `TestComputePerPatientUtility` (6 tests)
- `TestComputeUtilityScore` (6 tests)
- `TestComputeUtilityComponents` (3 tests)
- `TestEdgeCases` (4 tests)

**Key Tests**:
```python
✓ test_utility_function_optimal_early()          # Reward 1.0 (6-12h)
✓ test_utility_function_optimal_late()           # Reward 0.5 (0-6h)
✓ test_utility_function_too_early()              # Penalty -0.05 (>12h)
✓ test_utility_function_missed()                 # Penalty -2.0 (after)
✓ test_utility_function_false_positive()         # Penalty -0.05
✓ test_compute_per_patient_utility_optimal_early()   # Patient utility
✓ test_compute_per_patient_utility_false_positive()  # FP utility
✓ test_compute_per_patient_utility_true_negative()   # TN utility
✓ test_compute_utility_score_perfect()           # Score 1.0
✓ test_compute_utility_score_worst()             # Score 0.0
```

**Requirements Met**:
- ✓ Test utility_function rewards and penalties
- ✓ Test compute_per_patient_utility
- ✓ Test compute_utility_score normalization

---

### 7. test_memory.py

**Module**: All models memory efficiency

**Test Classes**:
- `TestQSOFAMemory` (2 tests)
- `TestXGBoostMemory` (3 tests)
- `TestTFTLiteMemory` (3 tests)
- `TestCombinedInference` (3 tests)
- `TestMemoryStress` (3 tests)
- `TestMemoryLeaks` (3 tests)

**Key Tests**:
```python
✓ test_qsofa_initialization_memory()             # Init < 500MB
✓ test_qsofa_inference_memory()                  # Inference < 500MB
✓ test_xgboost_initialization_memory()           # Init < 500MB
✓ test_xgboost_demo_mode_memory()                # Demo < 500MB
✓ test_tft_lite_initialization_memory()          # Init < 500MB
✓ test_tft_lite_inference_memory()               # Inference < 500MB
✓ test_all_models_combined_memory()              # Combined < 1.5GB
✓ test_sequential_inference_memory()             # Sequential < 1.5GB
✓ test_large_qsofa_inference()                   # 100k samples < 500MB
✓ test_repeated_qsofa_inference()                # No memory leak
✓ test_repeated_xgboost_inference()              # No memory leak
✓ test_repeated_tft_inference()                  # No memory leak
```

**Requirements Met**:
- ✓ Test all models load under 500MB each
- ✓ Test combined inference under 1.5GB

## Memory Constraints

### Individual Model Limits
- **qSOFA**: < 500MB (rule-based, minimal memory)
- **XGBoost**: < 500MB (demo mode efficient)
- **TFT-Lite**: < 500MB (optimized lightweight architecture)

### Combined Operations
- **All models together**: < 1.5GB
- **Sequential inference**: No accumulation
- **Repeated calls**: No memory leaks

### Stress Testing
- 100,000+ samples with qSOFA
- 200+ features with XGBoost
- Large batches with TFT-Lite

## Test Execution

### Run All Tests
```bash
pytest tests/ -v
# Expected: 150+ tests passing
# Execution time: ~30-60 seconds
```

### Run with Output
```bash
pytest tests/ -v -s
# Includes print statements
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov=models --cov-report=html
# Generates coverage report
```

### Run Specific Tests
```bash
pytest tests/test_data_loader.py::TestLoadPatient::test_load_patient_with_valid_psv -v
```

### Run Memory Tests Only
```bash
pytest tests/test_memory.py -v
# Memory constraint validation
```

## Key Features of Test Suite

### 1. Comprehensive Coverage
- All core modules tested
- 150+ test cases
- Edge cases and error handling
- Boundary conditions

### 2. Demo Mode Validation
- All models tested without trained weights
- Deterministic predictions
- Valid probability ranges
- Memory efficiency

### 3. Error Handling
- File not found scenarios
- Invalid data formats
- Missing columns
- Invalid thresholds
- Shape mismatches

### 4. Memory Constraints
- Individual model limits < 500MB
- Combined inference < 1.5GB
- Memory leak detection
- Stress testing

### 5. Data Validation
- Input shape validation
- Output range validation
- Type checking
- Immutability tests

### 6. Real Data Testing
- Uses sample PSV files
- PhysioNet-format data
- Realistic dimensions
- Patient hour data

## Sample Test Invocations

### Quick Smoke Test
```bash
pytest tests/test_data_loader.py::TestLoadPatient::test_load_patient_with_valid_psv -v
```

### Model Inference Tests
```bash
pytest tests/test_qsofa.py tests/test_xgboost.py tests/test_tft_lite.py -v
```

### Utility Score Tests
```bash
pytest tests/test_clinical_utility.py -v
```

### Memory Tests
```bash
pytest tests/test_memory.py -v --tb=short
```

### All Tests with Summary
```bash
pytest tests/ -v --tb=short --co -q
```

## Expected Output

```
================================= test session starts =================================
collected 150+ items

tests/test_data_loader.py::TestLoadPatient::test_load_patient_with_valid_psv PASSED
tests/test_data_loader.py::TestLoadPatient::test_load_patient_handles_missing_values PASSED
...
tests/test_memory.py::TestMemoryLeaks::test_repeated_tft_inference PASSED

================================= 150+ passed in 45.23s =================================
```

## Notes

1. **Temporary Files**: Tests automatically clean up temporary files
2. **Memory Measurements**: Uses `psutil` for accurate RSS measurement
3. **Sample Data**: Located in `/c/SepsisPulse/data/sample/patients/`
4. **Model Modes**: All tests run with demo mode (no weights required)
5. **Independence**: Tests can run in any order

## Troubleshooting

### Tests Not Found
```bash
# Ensure you're in the project root
cd /c/SepsisPulse

# Check Python path
python -c "import sys; print(sys.path)"
```

### Import Errors
```bash
# Install dependencies
pip install pytest pandas numpy torch xgboost psutil

# Verify imports
python -c "from src.data.loader import load_patient"
```

### Memory Test Failures
```bash
# Memory limits are tight for stress tests
# Reduce stress test size or increase available RAM
# Edit test_memory.py and adjust sample sizes

# For development, skip memory tests
pytest tests/ -v --ignore=tests/test_memory.py
```

### PyTorch Warnings
```bash
# Ignore PyTorch warnings during testing
export TF_CPP_MIN_LOG_LEVEL=3
pytest tests/test_tft_lite.py -v
```

## Continuous Integration

Tests are designed to work in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    pip install pytest pandas numpy torch xgboost psutil
    pytest tests/ -v --tb=short --junit-xml=test-results.xml
```

## Success Criteria

✓ All 150+ tests passing
✓ Memory usage within limits
✓ No memory leaks detected
✓ Demo mode functioning correctly
✓ Edge cases handled gracefully
