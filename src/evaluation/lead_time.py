"""
Lead Time Measurement Module for SepsisPulse.

This module computes lead time metrics for sepsis prediction models.
Lead time is defined as the hours between the first positive prediction
and the actual sepsis onset time. This is a critical metric for evaluating
the effectiveness of early warning systems.

Lead time is only meaningful for true positive cases - where the model
correctly predicts sepsis before it occurs.
"""

from typing import Dict, List, Optional

import numpy as np


def compute_lead_time(
    predictions: np.ndarray,
    t_sepsis: int,
    threshold: float = 0.5,
) -> Optional[float]:
    """
    Compute the lead time between first positive prediction and sepsis onset.

    Lead time is the number of hours between when the model first predicts
    sepsis (prediction >= threshold) and when sepsis actually occurs (t_sepsis).

    Parameters
    ----------
    predictions : np.ndarray
        Array of predictions for a single patient. Can be binary (0/1) or
        probability values (0.0 to 1.0). Each index represents one hour.
    t_sepsis : int
        The time index (hour) when sepsis onset occurs.
    threshold : float, optional
        Threshold for converting probabilities to binary predictions.
        Predictions >= threshold are considered positive. Default is 0.5.

    Returns
    -------
    Optional[float]
        The lead time in hours (t_sepsis - first_positive_time), or None if
        there is no positive prediction before sepsis onset.

    Notes
    -----
    - Only predictions at time indices < t_sepsis are considered (before onset).
    - If no positive prediction occurs before onset, returns None.
    - A larger lead time indicates earlier detection, which is desirable.

    Examples
    --------
    >>> predictions = np.array([0.2, 0.3, 0.6, 0.8, 0.9])
    >>> compute_lead_time(predictions, t_sepsis=4, threshold=0.5)
    2.0

    >>> predictions = np.array([0.1, 0.2, 0.3, 0.4])
    >>> compute_lead_time(predictions, t_sepsis=4, threshold=0.5) is None
    True
    """
    if t_sepsis <= 0:
        return None

    # Only consider predictions before sepsis onset
    predictions_before_onset = predictions[:t_sepsis]

    if len(predictions_before_onset) == 0:
        return None

    # Convert to binary predictions
    binary_predictions = (predictions_before_onset >= threshold).astype(int)

    # Find the first positive prediction
    positive_indices = np.where(binary_predictions == 1)[0]

    if len(positive_indices) == 0:
        return None

    first_positive_time = positive_indices[0]
    lead_time = float(t_sepsis - first_positive_time)

    return lead_time


def compute_average_lead_time(
    predictions: Dict[str, np.ndarray],
    t_sepsis: Dict[str, int],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute aggregate lead time statistics across multiple patients.

    This function calculates mean, median, standard deviation, minimum, and
    maximum lead times for all true positive patients (patients with sepsis
    who have at least one positive prediction before onset).

    Parameters
    ----------
    predictions : Dict[str, np.ndarray]
        Dictionary mapping patient IDs to their prediction arrays.
        Each array can be binary or probability values.
    t_sepsis : Dict[str, int]
        Dictionary mapping patient IDs to their sepsis onset time.
        Only patients present in this dictionary are considered (sepsis cases).
    threshold : float, optional
        Threshold for converting probabilities to binary predictions.
        Default is 0.5.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'mean': Mean lead time across all true positive patients
        - 'median': Median lead time
        - 'std': Standard deviation of lead times
        - 'min': Minimum lead time
        - 'max': Maximum lead time
        - 'count': Number of true positive patients included

        Returns empty dict with all values as NaN if no true positives exist.

    Notes
    -----
    - Only patients with valid lead times (true positives) are included.
    - Patients without positive predictions before onset are excluded.
    - If no true positive cases exist, all statistics are NaN.

    Examples
    --------
    >>> predictions = {
    ...     'P001': np.array([0.2, 0.6, 0.8, 0.9]),
    ...     'P002': np.array([0.1, 0.2, 0.7, 0.8]),
    ...     'P003': np.array([0.1, 0.2, 0.3, 0.4]),  # No detection
    ... }
    >>> t_sepsis = {'P001': 3, 'P002': 3, 'P003': 3}
    >>> stats = compute_average_lead_time(predictions, t_sepsis)
    >>> stats['mean']
    1.5
    """
    lead_times = []

    for patient_id, onset_time in t_sepsis.items():
        if patient_id not in predictions:
            continue

        patient_predictions = predictions[patient_id]
        lead_time = compute_lead_time(patient_predictions, onset_time, threshold)

        if lead_time is not None:
            lead_times.append(lead_time)

    if len(lead_times) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'count': 0,
        }

    lead_times_array = np.array(lead_times)

    return {
        'mean': float(np.mean(lead_times_array)),
        'median': float(np.median(lead_times_array)),
        'std': float(np.std(lead_times_array)),
        'min': float(np.min(lead_times_array)),
        'max': float(np.max(lead_times_array)),
        'count': len(lead_times),
    }


def compute_lead_time_distribution(
    predictions: Dict[str, np.ndarray],
    t_sepsis: Dict[str, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Compute array of all lead times for distribution analysis and plotting.

    This function returns all individual lead times as an array, which can
    be used to create histograms or other distribution visualizations.

    Parameters
    ----------
    predictions : Dict[str, np.ndarray]
        Dictionary mapping patient IDs to their prediction arrays.
        Each array can be binary or probability values.
    t_sepsis : Dict[str, int]
        Dictionary mapping patient IDs to their sepsis onset time.
        Only patients present in this dictionary are considered.
    threshold : float, optional
        Threshold for converting probabilities to binary predictions.
        Default is 0.5.

    Returns
    -------
    np.ndarray
        Array of lead times for all true positive patients.
        Empty array if no true positives exist.

    Notes
    -----
    - Only includes patients with valid lead times (true positives).
    - The returned array can be used directly with matplotlib.pyplot.hist().
    - Results are not sorted; order reflects iteration over patient dictionary.

    Examples
    --------
    >>> predictions = {
    ...     'P001': np.array([0.2, 0.6, 0.8, 0.9]),
    ...     'P002': np.array([0.1, 0.2, 0.7, 0.8]),
    ... }
    >>> t_sepsis = {'P001': 3, 'P002': 3}
    >>> lead_times = compute_lead_time_distribution(predictions, t_sepsis)
    >>> lead_times
    array([2., 1.])
    """
    lead_times = []

    for patient_id, onset_time in t_sepsis.items():
        if patient_id not in predictions:
            continue

        patient_predictions = predictions[patient_id]
        lead_time = compute_lead_time(patient_predictions, onset_time, threshold)

        if lead_time is not None:
            lead_times.append(lead_time)

    return np.array(lead_times)


def get_detection_rate_by_lead_time(
    predictions: Dict[str, np.ndarray],
    t_sepsis: Dict[str, int],
    time_points: List[int] = [1, 3, 6, 12],
    threshold: float = 0.5,
) -> Dict[int, float]:
    """
    Compute detection rates at various lead time thresholds.

    For each time point N, calculates what percentage of sepsis cases were
    detected with at least N hours of lead time before onset.

    Parameters
    ----------
    predictions : Dict[str, np.ndarray]
        Dictionary mapping patient IDs to their prediction arrays.
        Each array can be binary or probability values.
    t_sepsis : Dict[str, int]
        Dictionary mapping patient IDs to their sepsis onset time.
        Only patients present in this dictionary are considered.
    time_points : List[int], optional
        List of lead time thresholds to evaluate (in hours).
        Default is [1, 3, 6, 12].
    threshold : float, optional
        Threshold for converting probabilities to binary predictions.
        Default is 0.5.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping each time point to the detection rate (0.0 to 1.0).
        The detection rate is the proportion of sepsis patients detected
        with at least that many hours of lead time.

    Notes
    -----
    - Detection rate is calculated as: (patients with lead_time >= N) / total_sepsis_patients
    - A detection rate of 1.0 means all sepsis patients were detected at least N hours early.
    - A detection rate of 0.0 means no sepsis patients were detected that early.
    - Higher detection rates at larger time points indicate better early warning.

    Examples
    --------
    >>> predictions = {
    ...     'P001': np.array([0.6, 0.7, 0.8, 0.9]),  # Lead time = 4
    ...     'P002': np.array([0.1, 0.6, 0.7, 0.8]),  # Lead time = 3
    ...     'P003': np.array([0.1, 0.2, 0.6, 0.8]),  # Lead time = 2
    ...     'P004': np.array([0.1, 0.2, 0.3, 0.4]),  # No detection
    ... }
    >>> t_sepsis = {'P001': 4, 'P002': 4, 'P003': 4, 'P004': 4}
    >>> rates = get_detection_rate_by_lead_time(predictions, t_sepsis, [1, 2, 3, 4])
    >>> rates[3]  # 2 patients (P001, P002) have lead time >= 3
    0.5
    """
    total_sepsis_patients = len(t_sepsis)

    if total_sepsis_patients == 0:
        return {tp: 0.0 for tp in time_points}

    # Compute all lead times
    lead_times = compute_lead_time_distribution(predictions, t_sepsis, threshold)

    detection_rates = {}

    for time_point in time_points:
        if len(lead_times) == 0:
            detection_rates[time_point] = 0.0
        else:
            # Count patients with lead time >= time_point
            detected_count = np.sum(lead_times >= time_point)
            detection_rates[time_point] = float(detected_count) / total_sepsis_patients

    return detection_rates
