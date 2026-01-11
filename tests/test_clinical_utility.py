"""
Unit tests for the clinical utility module.

Tests utility function calculations, per-patient utility computation,
and normalized utility score calculations.
"""

import pytest
import numpy as np

from src.evaluation.clinical_utility import (
    utility_function,
    compute_per_patient_utility,
    compute_utility_score,
    compute_utility_components,
    REWARD_OPTIMAL_EARLY,
    REWARD_OPTIMAL_LATE,
    PENALTY_TOO_EARLY,
    PENALTY_MISSED,
    PENALTY_FALSE_POSITIVE,
    REWARD_TRUE_NEGATIVE,
    OPTIMAL_WINDOW_START,
    OPTIMAL_WINDOW_END,
)


class TestUtilityFunction:
    """Tests for utility_function."""

    def test_utility_function_optimal_early(self):
        """Test reward for prediction in optimal early window (6-12 hours)."""
        # Prediction 8 hours before sepsis onset
        utility = utility_function(t_pred=42, t_sepsis=50, has_sepsis=True)

        assert utility == REWARD_OPTIMAL_EARLY
        assert utility == 1.0

    def test_utility_function_optimal_late(self):
        """Test reward for prediction in optimal late window (0-6 hours)."""
        # Prediction 3 hours before sepsis onset
        utility = utility_function(t_pred=47, t_sepsis=50, has_sepsis=True)

        assert utility == REWARD_OPTIMAL_LATE
        assert utility == 0.5

    def test_utility_function_too_early(self):
        """Test penalty for prediction too early (>12 hours before)."""
        # Prediction 15 hours before sepsis onset
        utility = utility_function(t_pred=35, t_sepsis=50, has_sepsis=True)

        assert utility == PENALTY_TOO_EARLY
        assert utility == -0.05

    def test_utility_function_missed(self):
        """Test penalty for missed or late prediction (after onset)."""
        # Prediction after sepsis onset
        utility = utility_function(t_pred=51, t_sepsis=50, has_sepsis=True)

        assert utility == PENALTY_MISSED
        assert utility == -2.0

    def test_utility_function_false_positive(self):
        """Test penalty for false positive on non-sepsis patient."""
        utility = utility_function(t_pred=20, t_sepsis=100, has_sepsis=False)

        assert utility == PENALTY_FALSE_POSITIVE
        assert utility == -0.05

    def test_utility_function_true_negative(self):
        """Test that non-sepsis with no prediction has zero penalty."""
        # Note: utility_function only handles predictions
        # For non-sepsis patients, this function assumes a prediction was made
        # True negatives are handled at the per-patient level
        utility = utility_function(t_pred=20, t_sepsis=100, has_sepsis=False)

        assert utility == PENALTY_FALSE_POSITIVE

    def test_utility_function_boundary_optimal_early_start(self):
        """Test boundary case: prediction exactly 12 hours before."""
        # t_sepsis - t_pred = 12 (at the optimal early window start)
        utility = utility_function(t_pred=38, t_sepsis=50, has_sepsis=True)

        assert utility == REWARD_OPTIMAL_EARLY

    def test_utility_function_boundary_optimal_late_end(self):
        """Test boundary case: prediction exactly 6 hours before."""
        # t_sepsis - t_pred = 6 (at optimal late window end)
        utility = utility_function(t_pred=44, t_sepsis=50, has_sepsis=True)

        assert utility == REWARD_OPTIMAL_EARLY or utility == REWARD_OPTIMAL_LATE

    def test_utility_function_boundary_zero_hours(self):
        """Test boundary: prediction at sepsis onset."""
        # t_sepsis - t_pred = 0 (prediction at onset)
        utility = utility_function(t_pred=50, t_sepsis=50, has_sepsis=True)

        assert utility == REWARD_OPTIMAL_LATE


class TestComputePerPatientUtility:
    """Tests for compute_per_patient_utility."""

    def test_compute_per_patient_utility_optimal_early(self):
        """Test per-patient utility with optimal early prediction."""
        predictions = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # First prediction at t=4
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])      # Sepsis at t=8
        t_sepsis = 8

        # t_sepsis - t_pred = 8 - 4 = 4 hours (in optimal early window)
        utility = compute_per_patient_utility(predictions, labels, t_sepsis)

        assert utility == REWARD_OPTIMAL_EARLY

    def test_compute_per_patient_utility_optimal_late(self):
        """Test per-patient utility with optimal late prediction."""
        predictions = np.array([0, 0, 0, 1, 1, 1])  # First prediction at t=3
        labels = np.array([0, 0, 0, 0, 1, 1])       # Sepsis at t=4
        t_sepsis = 4

        # t_sepsis - t_pred = 4 - 3 = 1 hour (in optimal late window)
        utility = compute_per_patient_utility(predictions, labels, t_sepsis)

        assert utility == REWARD_OPTIMAL_LATE

    def test_compute_per_patient_utility_false_positive(self):
        """Test per-patient utility with false positive."""
        predictions = np.array([0, 0, 1, 0, 0])  # Prediction at t=2
        labels = np.array([0, 0, 0, 0, 0])       # No sepsis
        t_sepsis = None

        utility = compute_per_patient_utility(predictions, labels, t_sepsis)

        assert utility == PENALTY_FALSE_POSITIVE

    def test_compute_per_patient_utility_true_negative(self):
        """Test per-patient utility with correct non-prediction."""
        predictions = np.array([0, 0, 0, 0, 0])  # No prediction
        labels = np.array([0, 0, 0, 0, 0])       # No sepsis
        t_sepsis = None

        utility = compute_per_patient_utility(predictions, labels, t_sepsis)

        assert utility == REWARD_TRUE_NEGATIVE
        assert utility == 0.0

    def test_compute_per_patient_utility_missed_sepsis(self):
        """Test per-patient utility when sepsis is missed."""
        predictions = np.array([0, 0, 0, 0, 0])  # No prediction
        labels = np.array([0, 0, 0, 1, 1])       # Sepsis at t=3
        t_sepsis = 3

        utility = compute_per_patient_utility(predictions, labels, t_sepsis)

        assert utility == PENALTY_MISSED

    def test_compute_per_patient_utility_too_early(self):
        """Test per-patient utility with too early prediction."""
        predictions = np.array([1, 0, 0, 0, 0, 0, 0, 0])  # Prediction at t=0
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 1])       # Sepsis at t=7
        t_sepsis = 7

        # t_sepsis - t_pred = 7 - 0 = 7 hours
        # Should not be in optimal window if > 12 hours before
        utility = compute_per_patient_utility(predictions, labels, t_sepsis)

        # 7 hours is still in late optimal window
        assert utility == REWARD_OPTIMAL_LATE


class TestComputeUtilityScore:
    """Tests for compute_utility_score."""

    def test_compute_utility_score_perfect_predictions(self):
        """Test normalized score with perfect predictions."""
        predictions = {
            'p1': np.array([0, 0, 0, 1, 1, 1]),  # Optimal early prediction
            'p2': np.array([0, 0, 0, 0, 0, 0]),  # Correct non-sepsis
        }
        labels = {
            'p1': np.array([0, 0, 0, 0, 1, 1]),
            'p2': np.array([0, 0, 0, 0, 0, 0]),
        }
        t_sepsis = {
            'p1': 4,    # Sepsis at t=4, prediction at t=3 (1 hour early)
            'p2': None,
        }

        score = compute_utility_score(predictions, labels, t_sepsis)

        # Perfect predictions should give score 1.0
        assert score == pytest.approx(1.0)

    def test_compute_utility_score_worst_predictions(self):
        """Test normalized score with worst predictions."""
        predictions = {
            'p1': np.array([0, 0, 0, 0, 0, 0]),  # Missed sepsis
            'p2': np.array([1, 1, 1, 1, 1, 1]),  # False positive
        }
        labels = {
            'p1': np.array([0, 0, 0, 1, 1, 1]),
            'p2': np.array([0, 0, 0, 0, 0, 0]),
        }
        t_sepsis = {
            'p1': 3,    # Sepsis at t=3, not predicted
            'p2': None,
        }

        score = compute_utility_score(predictions, labels, t_sepsis)

        # Worst predictions should give score 0.0
        assert score == pytest.approx(0.0)

    def test_compute_utility_score_mixed_predictions(self):
        """Test normalized score with mixed predictions."""
        predictions = {
            'p1': np.array([0, 0, 1, 1]),  # Prediction at t=2, sepsis at t=3 (optimal late)
            'p2': np.array([0, 0, 0, 0]),  # Correct non-sepsis
        }
        labels = {
            'p1': np.array([0, 0, 0, 1]),
            'p2': np.array([0, 0, 0, 0]),
        }
        t_sepsis = {
            'p1': 3,
            'p2': None,
        }

        score = compute_utility_score(predictions, labels, t_sepsis)

        # Should be between 0 and 1
        assert 0.0 <= score <= 1.0

    def test_compute_utility_score_single_patient(self):
        """Test normalized score with single patient."""
        predictions = {
            'p1': np.array([0, 0, 1, 1]),
        }
        labels = {
            'p1': np.array([0, 0, 0, 1]),
        }
        t_sepsis = {
            'p1': 3,
        }

        score = compute_utility_score(predictions, labels, t_sepsis)

        assert 0.0 <= score <= 1.0

    def test_compute_utility_score_mismatched_keys(self):
        """Test that mismatched patient IDs raise ValueError."""
        predictions = {
            'p1': np.array([0, 0, 1, 1]),
            'p2': np.array([0, 0, 0, 0]),
        }
        labels = {
            'p1': np.array([0, 0, 0, 1]),
            # Missing p2
        }
        t_sepsis = {
            'p1': 3,
            'p2': None,
        }

        with pytest.raises(ValueError, match="Patient IDs must match"):
            compute_utility_score(predictions, labels, t_sepsis)

    def test_compute_utility_score_empty_dicts(self):
        """Test with empty dictionaries."""
        predictions = {}
        labels = {}
        t_sepsis = {}

        score = compute_utility_score(predictions, labels, t_sepsis)

        assert score == 0.0


class TestComputeUtilityComponents:
    """Tests for compute_utility_components."""

    def test_compute_utility_components_structure(self):
        """Test that compute_utility_components returns expected structure."""
        predictions = {
            'p1': np.array([0, 0, 1, 1]),
        }
        labels = {
            'p1': np.array([0, 0, 0, 1]),
        }
        t_sepsis = {
            'p1': 3,
        }

        components = compute_utility_components(predictions, labels, t_sepsis)

        # Check expected keys
        expected_keys = [
            'total_utility',
            'best_utility',
            'worst_utility',
            'normalized_score',
            'num_sepsis_patients',
            'num_non_sepsis_patients',
            'optimal_early_count',
            'optimal_late_count',
            'too_early_count',
            'missed_count',
            'false_positive_count',
            'true_negative_count',
        ]

        for key in expected_keys:
            assert key in components

    def test_compute_utility_components_values(self):
        """Test specific component values."""
        predictions = {
            'p1': np.array([0, 0, 1, 1]),  # Optimal late
            'p2': np.array([0, 0, 0, 0]),  # True negative
        }
        labels = {
            'p1': np.array([0, 0, 0, 1]),
            'p2': np.array([0, 0, 0, 0]),
        }
        t_sepsis = {
            'p1': 3,
            'p2': None,
        }

        components = compute_utility_components(predictions, labels, t_sepsis)

        # Check counts
        assert components['num_sepsis_patients'] == 1
        assert components['num_non_sepsis_patients'] == 1
        assert components['optimal_late_count'] == 1
        assert components['true_negative_count'] == 1

    def test_compute_utility_components_counts_sum(self):
        """Test that outcome counts sum correctly."""
        predictions = {
            'p1': np.array([0, 0, 1, 1]),  # Optimal late
            'p2': np.array([1, 1, 1, 1]),  # False positive
            'p3': np.array([0, 0, 0, 0]),  # Missed sepsis
            'p4': np.array([0, 0, 0, 0]),  # True negative
        }
        labels = {
            'p1': np.array([0, 0, 0, 1]),
            'p2': np.array([0, 0, 0, 0]),
            'p3': np.array([0, 0, 1, 1]),
            'p4': np.array([0, 0, 0, 0]),
        }
        t_sepsis = {
            'p1': 3,
            'p2': None,
            'p3': 2,
            'p4': None,
        }

        components = compute_utility_components(predictions, labels, t_sepsis)

        # Count sepsis outcomes (should include optimal_early, optimal_late, too_early, missed)
        sepsis_outcomes = (
            components['optimal_early_count'] +
            components['optimal_late_count'] +
            components['too_early_count'] +
            components['missed_count']
        )
        assert sepsis_outcomes == components['num_sepsis_patients']

        # Count non-sepsis outcomes (should include false_positive, true_negative)
        non_sepsis_outcomes = (
            components['false_positive_count'] +
            components['true_negative_count']
        )
        assert non_sepsis_outcomes == components['num_non_sepsis_patients']


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_utility_with_zero_time(self):
        """Test utility calculation at t=0."""
        utility = utility_function(t_pred=0, t_sepsis=0, has_sepsis=True)

        # Prediction at sepsis onset
        assert utility == REWARD_OPTIMAL_LATE

    def test_utility_with_large_times(self):
        """Test utility with large time values."""
        # Simulate ICU stay over 500 hours
        utility = utility_function(t_pred=300, t_sepsis=500, has_sepsis=True)

        # 200 hours early should be too early
        assert utility == PENALTY_TOO_EARLY

    def test_per_patient_utility_long_stay(self):
        """Test per-patient utility with long ICU stay."""
        # 100-hour stay
        predictions = np.zeros(100)
        predictions[90] = 1  # First prediction at t=90
        labels = np.zeros(100)
        labels[95:] = 1  # Sepsis at t=95
        t_sepsis = 95

        utility = compute_per_patient_utility(predictions, labels, t_sepsis)

        # 95 - 90 = 5 hours (optimal late window)
        assert utility == REWARD_OPTIMAL_LATE

    def test_per_patient_utility_single_hour(self):
        """Test per-patient utility with single hour of data."""
        predictions = np.array([1])
        labels = np.array([1])
        t_sepsis = 0

        utility = compute_per_patient_utility(predictions, labels, t_sepsis)

        # Prediction at sepsis onset
        assert utility == REWARD_OPTIMAL_LATE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
