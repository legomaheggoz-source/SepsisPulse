"""Evaluation metrics for sepsis prediction models."""

from .clinical_utility import (
    compute_utility_score,
    utility_function,
    compute_per_patient_utility,
    compute_utility_components,
    OPTIMAL_WINDOW_START,
    OPTIMAL_WINDOW_END,
    REWARD_OPTIMAL_EARLY,
    REWARD_OPTIMAL_LATE,
    PENALTY_TOO_EARLY,
    PENALTY_MISSED,
    PENALTY_FALSE_POSITIVE,
    REWARD_TRUE_NEGATIVE,
)
from .lead_time import (
    compute_lead_time,
    compute_average_lead_time,
    compute_lead_time_distribution,
    get_detection_rate_by_lead_time,
)
from .metrics import (
    compute_roc_auc,
    compute_pr_auc,
    compute_confusion_matrix,
    compute_classification_metrics,
)

__all__ = [
    # Clinical utility functions
    "compute_utility_score",
    "utility_function",
    "compute_per_patient_utility",
    "compute_utility_components",
    # Clinical utility constants
    "OPTIMAL_WINDOW_START",
    "OPTIMAL_WINDOW_END",
    "REWARD_OPTIMAL_EARLY",
    "REWARD_OPTIMAL_LATE",
    "PENALTY_TOO_EARLY",
    "PENALTY_MISSED",
    "PENALTY_FALSE_POSITIVE",
    "REWARD_TRUE_NEGATIVE",
    # Lead time functions
    "compute_lead_time",
    "compute_average_lead_time",
    "compute_lead_time_distribution",
    "get_detection_rate_by_lead_time",
    # Standard metrics
    "compute_roc_auc",
    "compute_pr_auc",
    "compute_confusion_matrix",
    "compute_classification_metrics",
]
