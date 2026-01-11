"""PhysioNet 2019 Clinical Utility Score Calculator.

This module implements the utility function from the PhysioNet/Computing in
Cardiology Challenge 2019 for early prediction of sepsis. The utility function
rewards early predictions within an optimal window (6-12 hours before sepsis
onset) and penalizes false positives and late/missed detections.

Reference:
    Reyna, M. A., et al. "Early Prediction of Sepsis from Clinical Data:
    The PhysioNet/Computing in Cardiology Challenge 2019."
    Critical Care Medicine 48.2 (2020): 210-217.
"""

from typing import Dict, Optional

import numpy as np

# =============================================================================
# Constants for reward/penalty values (PhysioNet 2019 Challenge)
# =============================================================================

# Optimal window boundaries (hours before sepsis onset)
OPTIMAL_WINDOW_START = 12  # Hours before sepsis onset (early boundary)
OPTIMAL_WINDOW_END = 6     # Hours before sepsis onset (late boundary)

# Rewards for sepsis patients
REWARD_OPTIMAL_EARLY = 1.0   # Prediction in [t_sepsis-12, t_sepsis-6]
REWARD_OPTIMAL_LATE = 0.5    # Prediction in [t_sepsis-6, t_sepsis]

# Penalties for sepsis patients
PENALTY_TOO_EARLY = -0.05    # Prediction before t_sepsis-12
PENALTY_MISSED = -2.0        # Prediction after sepsis onset (missed/too late)

# Penalties for non-sepsis patients
PENALTY_FALSE_POSITIVE = -0.05  # False positive prediction
REWARD_TRUE_NEGATIVE = 0.0      # Correctly identified non-sepsis


def utility_function(t_pred: int, t_sepsis: int, has_sepsis: bool) -> float:
    """Compute utility value for a single prediction.

    This implements the PhysioNet 2019 utility function that rewards early
    predictions within an optimal window and penalizes false positives and
    late detections.

    Parameters
    ----------
    t_pred : int
        Time step (hour) at which the prediction was made.
    t_sepsis : int
        Time step (hour) of sepsis onset. For non-sepsis patients, this value
        is typically set to the length of their stay or a sentinel value.
    has_sepsis : bool
        Whether the patient actually develops sepsis.

    Returns
    -------
    float
        Utility value for the prediction:
        - For sepsis patients:
            * 1.0 if prediction is 6-12 hours before onset (optimal early)
            * 0.5 if prediction is 0-6 hours before onset (optimal late)
            * -0.05 if prediction is more than 12 hours before onset (too early)
            * -2.0 if prediction is after onset (missed/too late)
        - For non-sepsis patients:
            * -0.05 for false positive predictions
            * 0.0 for true negative (no prediction needed)

    Examples
    --------
    >>> # Optimal early prediction (8 hours before sepsis)
    >>> utility_function(t_pred=42, t_sepsis=50, has_sepsis=True)
    1.0

    >>> # Late prediction (3 hours before sepsis)
    >>> utility_function(t_pred=47, t_sepsis=50, has_sepsis=True)
    0.5

    >>> # False positive on non-sepsis patient
    >>> utility_function(t_pred=20, t_sepsis=100, has_sepsis=False)
    -0.05
    """
    if has_sepsis:
        # Time difference: positive means prediction is before sepsis
        time_diff = t_sepsis - t_pred

        if time_diff >= OPTIMAL_WINDOW_END and time_diff <= OPTIMAL_WINDOW_START:
            # Optimal window: 6-12 hours before sepsis onset
            return REWARD_OPTIMAL_EARLY
        elif time_diff >= 0 and time_diff < OPTIMAL_WINDOW_END:
            # Late optimal window: 0-6 hours before sepsis onset
            return REWARD_OPTIMAL_LATE
        elif time_diff > OPTIMAL_WINDOW_START:
            # Too early: more than 12 hours before onset
            return PENALTY_TOO_EARLY
        else:
            # Missed: prediction after sepsis onset (time_diff < 0)
            return PENALTY_MISSED
    else:
        # Non-sepsis patient: false positive penalty
        return PENALTY_FALSE_POSITIVE


def compute_per_patient_utility(
    predictions: np.ndarray,
    labels: np.ndarray,
    t_sepsis: Optional[int]
) -> float:
    """Compute utility score for a single patient.

    This function evaluates predictions for one patient based on the PhysioNet
    2019 utility function. For sepsis patients, it finds the first positive
    prediction and computes utility based on its timing relative to sepsis onset.
    For non-sepsis patients, any positive prediction incurs a false positive
    penalty.

    Parameters
    ----------
    predictions : np.ndarray
        Binary predictions (0 or 1) for each hour of the patient's stay.
        Shape: (num_hours,)
    labels : np.ndarray
        True binary labels for each hour. Shape: (num_hours,)
    t_sepsis : Optional[int]
        Hour of sepsis onset, or None if the patient never develops sepsis.

    Returns
    -------
    float
        Utility score for this patient:
        - For sepsis patients: utility based on first positive prediction timing
        - For non-sepsis patients: penalty if any false positive, else 0

    Examples
    --------
    >>> predictions = np.array([0, 0, 0, 1, 1, 1])
    >>> labels = np.array([0, 0, 0, 0, 1, 1])
    >>> # First prediction at t=3, sepsis at t=4 (1 hour early)
    >>> compute_per_patient_utility(predictions, labels, t_sepsis=4)
    0.5

    >>> # Non-sepsis patient with false positive
    >>> predictions = np.array([0, 0, 1, 0, 0])
    >>> labels = np.array([0, 0, 0, 0, 0])
    >>> compute_per_patient_utility(predictions, labels, t_sepsis=None)
    -0.05
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    has_sepsis = t_sepsis is not None

    # Find first positive prediction
    positive_indices = np.where(predictions == 1)[0]

    if len(positive_indices) == 0:
        # No positive prediction made
        if has_sepsis:
            # Missed sepsis entirely - use the worst penalty
            # The prediction is considered as happening after sepsis (missed)
            return PENALTY_MISSED
        else:
            # True negative - correctly did not predict sepsis
            return REWARD_TRUE_NEGATIVE

    # Get the first positive prediction time
    t_pred = int(positive_indices[0])

    if has_sepsis:
        return utility_function(t_pred, t_sepsis, has_sepsis=True)
    else:
        # Non-sepsis patient with at least one positive prediction
        return PENALTY_FALSE_POSITIVE


def compute_utility_score(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    t_sepsis: Dict[str, Optional[int]]
) -> float:
    """Compute overall normalized utility score across all patients.

    This function aggregates per-patient utilities and normalizes the result
    to a score in [0, 1]. The normalization is done by comparing the achieved
    utility against the best and worst possible utilities.

    Parameters
    ----------
    predictions : Dict[str, np.ndarray]
        Dictionary mapping patient_id to binary predictions per hour.
        Each array has shape (num_hours_for_patient,).
    labels : Dict[str, np.ndarray]
        Dictionary mapping patient_id to true labels per hour.
        Each array has shape (num_hours_for_patient,).
    t_sepsis : Dict[str, Optional[int]]
        Dictionary mapping patient_id to hour of sepsis onset.
        Value is None if the patient never develops sepsis.

    Returns
    -------
    float
        Normalized utility score in range [0, 1], where:
        - 1.0 represents optimal predictions for all patients
        - 0.0 represents worst-case predictions

    Raises
    ------
    ValueError
        If the patient IDs in predictions, labels, and t_sepsis don't match.

    Notes
    -----
    The normalization follows the PhysioNet 2019 challenge formula:

    normalized_score = (U - U_worst) / (U_best - U_worst)

    where:
    - U is the achieved utility sum
    - U_best is the utility from optimal predictions (1.0 for sepsis, 0.0 for non-sepsis)
    - U_worst is the utility from worst predictions (missed sepsis, false positives)

    Examples
    --------
    >>> predictions = {
    ...     'p1': np.array([0, 0, 1, 1]),  # Predicts at t=2
    ...     'p2': np.array([0, 0, 0, 0])   # No prediction
    ... }
    >>> labels = {
    ...     'p1': np.array([0, 0, 0, 1]),  # Sepsis at t=3
    ...     'p2': np.array([0, 0, 0, 0])   # No sepsis
    ... }
    >>> t_sepsis = {'p1': 3, 'p2': None}
    >>> score = compute_utility_score(predictions, labels, t_sepsis)
    """
    # Validate inputs
    pred_keys = set(predictions.keys())
    label_keys = set(labels.keys())
    t_sepsis_keys = set(t_sepsis.keys())

    if pred_keys != label_keys or pred_keys != t_sepsis_keys:
        raise ValueError(
            "Patient IDs must match across predictions, labels, and t_sepsis. "
            f"Got {len(pred_keys)} predictions, {len(label_keys)} labels, "
            f"{len(t_sepsis_keys)} t_sepsis entries."
        )

    if len(predictions) == 0:
        return 0.0

    # Compute achieved utility
    total_utility = 0.0
    best_utility = 0.0
    worst_utility = 0.0

    for patient_id in predictions.keys():
        patient_preds = predictions[patient_id]
        patient_labels = labels[patient_id]
        patient_t_sepsis = t_sepsis[patient_id]

        has_sepsis = patient_t_sepsis is not None

        # Compute achieved utility for this patient
        patient_utility = compute_per_patient_utility(
            patient_preds, patient_labels, patient_t_sepsis
        )
        total_utility += patient_utility

        # Compute best possible utility for this patient
        if has_sepsis:
            # Best case: optimal early prediction
            best_utility += REWARD_OPTIMAL_EARLY
            # Worst case: completely missed sepsis
            worst_utility += PENALTY_MISSED
        else:
            # Best case: true negative (no prediction)
            best_utility += REWARD_TRUE_NEGATIVE
            # Worst case: false positive
            worst_utility += PENALTY_FALSE_POSITIVE

    # Normalize to [0, 1]
    utility_range = best_utility - worst_utility

    if utility_range == 0:
        # Edge case: all patients have same best and worst utility
        return 1.0 if total_utility >= best_utility else 0.0

    normalized_score = (total_utility - worst_utility) / utility_range

    # Clamp to [0, 1] to handle numerical edge cases
    return float(np.clip(normalized_score, 0.0, 1.0))


def compute_utility_components(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    t_sepsis: Dict[str, Optional[int]]
) -> Dict[str, float]:
    """Compute detailed utility components for analysis.

    This function provides a breakdown of the utility score into its
    constituent components for debugging and analysis purposes.

    Parameters
    ----------
    predictions : Dict[str, np.ndarray]
        Dictionary mapping patient_id to binary predictions per hour.
    labels : Dict[str, np.ndarray]
        Dictionary mapping patient_id to true labels per hour.
    t_sepsis : Dict[str, Optional[int]]
        Dictionary mapping patient_id to hour of sepsis onset (None if no sepsis).

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'total_utility': Raw utility sum
        - 'best_utility': Best possible utility
        - 'worst_utility': Worst possible utility
        - 'normalized_score': Score in [0, 1]
        - 'num_sepsis_patients': Count of sepsis patients
        - 'num_non_sepsis_patients': Count of non-sepsis patients
        - 'optimal_early_count': Predictions in optimal early window
        - 'optimal_late_count': Predictions in optimal late window
        - 'too_early_count': Predictions too early
        - 'missed_count': Missed sepsis cases
        - 'false_positive_count': False positive predictions
        - 'true_negative_count': True negative cases
    """
    total_utility = 0.0
    best_utility = 0.0
    worst_utility = 0.0

    num_sepsis = 0
    num_non_sepsis = 0
    optimal_early = 0
    optimal_late = 0
    too_early = 0
    missed = 0
    false_positive = 0
    true_negative = 0

    for patient_id in predictions.keys():
        patient_preds = predictions[patient_id]
        patient_labels = labels[patient_id]
        patient_t_sepsis = t_sepsis[patient_id]

        has_sepsis = patient_t_sepsis is not None

        if has_sepsis:
            num_sepsis += 1
            best_utility += REWARD_OPTIMAL_EARLY
            worst_utility += PENALTY_MISSED
        else:
            num_non_sepsis += 1
            best_utility += REWARD_TRUE_NEGATIVE
            worst_utility += PENALTY_FALSE_POSITIVE

        # Find first positive prediction
        positive_indices = np.where(patient_preds == 1)[0]

        if len(positive_indices) == 0:
            if has_sepsis:
                missed += 1
                total_utility += PENALTY_MISSED
            else:
                true_negative += 1
                total_utility += REWARD_TRUE_NEGATIVE
        else:
            t_pred = int(positive_indices[0])

            if has_sepsis:
                time_diff = patient_t_sepsis - t_pred

                if time_diff >= OPTIMAL_WINDOW_END and time_diff <= OPTIMAL_WINDOW_START:
                    optimal_early += 1
                    total_utility += REWARD_OPTIMAL_EARLY
                elif time_diff >= 0 and time_diff < OPTIMAL_WINDOW_END:
                    optimal_late += 1
                    total_utility += REWARD_OPTIMAL_LATE
                elif time_diff > OPTIMAL_WINDOW_START:
                    too_early += 1
                    total_utility += PENALTY_TOO_EARLY
                else:
                    missed += 1
                    total_utility += PENALTY_MISSED
            else:
                false_positive += 1
                total_utility += PENALTY_FALSE_POSITIVE

    # Compute normalized score
    utility_range = best_utility - worst_utility
    if utility_range == 0:
        normalized_score = 1.0 if total_utility >= best_utility else 0.0
    else:
        normalized_score = (total_utility - worst_utility) / utility_range
        normalized_score = float(np.clip(normalized_score, 0.0, 1.0))

    return {
        'total_utility': total_utility,
        'best_utility': best_utility,
        'worst_utility': worst_utility,
        'normalized_score': normalized_score,
        'num_sepsis_patients': num_sepsis,
        'num_non_sepsis_patients': num_non_sepsis,
        'optimal_early_count': optimal_early,
        'optimal_late_count': optimal_late,
        'too_early_count': too_early,
        'missed_count': missed,
        'false_positive_count': false_positive,
        'true_negative_count': true_negative,
    }
