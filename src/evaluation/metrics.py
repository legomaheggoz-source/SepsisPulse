"""Standard ML evaluation metrics for sepsis prediction models.

This module provides standard machine learning evaluation metrics including
ROC curves, Precision-Recall curves, confusion matrices, and classification
metrics (sensitivity, specificity, PPV, NPV, F1, accuracy).

These metrics complement the clinical utility score by providing traditional
performance measures that are widely used in the ML community.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute ROC AUC score and curve data for plotting.

    The Receiver Operating Characteristic (ROC) curve plots the True Positive
    Rate (TPR/sensitivity) against the False Positive Rate (FPR/1-specificity)
    at various classification thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1). Shape: (n_samples,)
    y_score : np.ndarray
        Predicted probabilities or scores. Higher values indicate higher
        likelihood of the positive class. Shape: (n_samples,)

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        A tuple containing:
        - auc_score: Area under the ROC curve (0.0 to 1.0)
        - fpr: False positive rates at each threshold
        - tpr: True positive rates at each threshold

    Raises
    ------
    ValueError
        If inputs are empty or have mismatched lengths.

    Notes
    -----
    - Returns (0.5, [0, 1], [0, 1]) for edge cases with only one class.
    - An AUC of 0.5 indicates random performance.
    - An AUC of 1.0 indicates perfect classification.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_score = np.array([0.1, 0.4, 0.6, 0.9])
    >>> auc_score, fpr, tpr = compute_roc_auc(y_true, y_score)
    >>> auc_score
    1.0
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Validate inputs
    if len(y_true) == 0 or len(y_score) == 0:
        raise ValueError("Input arrays cannot be empty.")

    if len(y_true) != len(y_score):
        raise ValueError(
            f"Input arrays must have the same length. "
            f"Got y_true: {len(y_true)}, y_score: {len(y_score)}."
        )

    # Check for single-class edge case
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        # Cannot compute meaningful ROC when only one class is present
        # Return baseline (random classifier performance)
        return 0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0])

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Compute AUC
    auc_score = roc_auc_score(y_true, y_score)

    return float(auc_score), fpr, tpr


def compute_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute Precision-Recall AUC score and curve data for plotting.

    The Precision-Recall (PR) curve plots precision against recall at various
    classification thresholds. PR curves are particularly useful for imbalanced
    datasets where the positive class is rare (as is typical in sepsis prediction).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1). Shape: (n_samples,)
    y_score : np.ndarray
        Predicted probabilities or scores. Higher values indicate higher
        likelihood of the positive class. Shape: (n_samples,)

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        A tuple containing:
        - auc_score: Area under the PR curve (0.0 to 1.0)
        - precision: Precision values at each threshold
        - recall: Recall values at each threshold

    Raises
    ------
    ValueError
        If inputs are empty or have mismatched lengths.

    Notes
    -----
    - Returns baseline AUC (prevalence) for single-class edge cases.
    - A baseline PR-AUC equals the prevalence of the positive class.
    - Higher PR-AUC indicates better performance, especially on imbalanced data.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_score = np.array([0.1, 0.4, 0.6, 0.9])
    >>> auc_score, precision, recall = compute_pr_auc(y_true, y_score)
    >>> auc_score
    1.0
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Validate inputs
    if len(y_true) == 0 or len(y_score) == 0:
        raise ValueError("Input arrays cannot be empty.")

    if len(y_true) != len(y_score):
        raise ValueError(
            f"Input arrays must have the same length. "
            f"Got y_true: {len(y_true)}, y_score: {len(y_score)}."
        )

    # Check for single-class edge case
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        # Cannot compute meaningful PR curve when only one class is present
        # Return baseline (prevalence of positive class)
        prevalence = float(np.mean(y_true))
        return prevalence, np.array([prevalence, 1.0]), np.array([1.0, 0.0])

    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    # Compute AUC using the trapezoidal rule
    # Note: recall is in descending order from precision_recall_curve
    auc_score = auc(recall, precision)

    return float(auc_score), precision, recall


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, int]:
    """Compute confusion matrix components.

    The confusion matrix summarizes prediction results by counting
    true positives, true negatives, false positives, and false negatives.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1). Shape: (n_samples,)
    y_pred : np.ndarray
        Predicted binary labels (0 or 1). Shape: (n_samples,)

    Returns
    -------
    Dict[str, int]
        Dictionary containing:
        - 'tp': True positives (correctly predicted positive)
        - 'tn': True negatives (correctly predicted negative)
        - 'fp': False positives (incorrectly predicted positive)
        - 'fn': False negatives (incorrectly predicted negative)

    Raises
    ------
    ValueError
        If inputs are empty or have mismatched lengths.

    Notes
    -----
    - For empty inputs, raises ValueError.
    - All counts will be 0 if arrays have length 0.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 0, 1])
    >>> cm = compute_confusion_matrix(y_true, y_pred)
    >>> cm
    {'tp': 1, 'tn': 1, 'fp': 1, 'fn': 1}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Validate inputs
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays cannot be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Input arrays must have the same length. "
            f"Got y_true: {len(y_true)}, y_pred: {len(y_pred)}."
        )

    # Compute confusion matrix components
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard classification metrics.

    This function calculates sensitivity (recall), specificity, positive
    predictive value (PPV/precision), negative predictive value (NPV),
    F1 score, and accuracy from binary predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1). Shape: (n_samples,)
    y_pred : np.ndarray
        Predicted binary labels (0 or 1). Shape: (n_samples,)

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'sensitivity': TP / (TP + FN) - also known as recall or TPR
        - 'specificity': TN / (TN + FP) - also known as TNR
        - 'ppv': TP / (TP + FP) - positive predictive value (precision)
        - 'npv': TN / (TN + FN) - negative predictive value
        - 'f1': 2 * (PPV * sensitivity) / (PPV + sensitivity)
        - 'accuracy': (TP + TN) / (TP + TN + FP + FN)

    Raises
    ------
    ValueError
        If inputs are empty or have mismatched lengths.

    Notes
    -----
    - Returns 0.0 for metrics that cannot be computed due to division by zero.
    - For example, if there are no positive predictions, PPV will be 0.0.
    - All metrics are in the range [0.0, 1.0].

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 0, 1, 1])
    >>> metrics = compute_classification_metrics(y_true, y_pred)
    >>> metrics['sensitivity']
    1.0
    >>> metrics['specificity']
    1.0
    """
    # Get confusion matrix components
    cm = compute_confusion_matrix(y_true, y_pred)
    tp = cm['tp']
    tn = cm['tn']
    fp = cm['fp']
    fn = cm['fn']

    total = tp + tn + fp + fn

    # Compute sensitivity (recall, TPR)
    # TP / (TP + FN)
    if (tp + fn) > 0:
        sensitivity = tp / (tp + fn)
    else:
        sensitivity = 0.0

    # Compute specificity (TNR)
    # TN / (TN + FP)
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0

    # Compute positive predictive value (precision)
    # TP / (TP + FP)
    if (tp + fp) > 0:
        ppv = tp / (tp + fp)
    else:
        ppv = 0.0

    # Compute negative predictive value
    # TN / (TN + FN)
    if (tn + fn) > 0:
        npv = tn / (tn + fn)
    else:
        npv = 0.0

    # Compute F1 score
    # 2 * (PPV * sensitivity) / (PPV + sensitivity)
    if (ppv + sensitivity) > 0:
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity)
    else:
        f1 = 0.0

    # Compute accuracy
    # (TP + TN) / total
    if total > 0:
        accuracy = (tp + tn) / total
    else:
        accuracy = 0.0

    return {
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'ppv': float(ppv),
        'npv': float(npv),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }
