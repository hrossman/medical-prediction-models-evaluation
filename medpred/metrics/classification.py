"""
Classification performance measures for binary prediction models.

Classification measures require a decision threshold to classify individuals
into low-risk and high-risk groups. The paper discusses:

Summary measures (all improper at clinically relevant thresholds):
- Classification accuracy (semi-proper only at t=0.5)
- Balanced accuracy (semi-proper only at t=prevalence)
- Youden index (semi-proper only at t=prevalence)
- Diagnostic odds ratio (improper)
- Kappa (improper)
- F1 score (improper AND lacks clear focus - only measure with neither characteristic)
- MCC (improper)

Partial/descriptive measures (improper individually):
- Sensitivity (recall)
- Specificity
- PPV (precision)
- NPV

Reference: Van Calster et al. (2025), The Lancet Digital Health.
"""

import numpy as np


def _confusion_matrix(y_true, y_prob, threshold):
    """Compute confusion matrix components at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def _sensitivity(tp, fn):
    """Sensitivity (recall, true positive rate)."""
    denom = tp + fn
    return tp / denom if denom > 0 else np.nan


def _specificity(tn, fp):
    """Specificity (true negative rate)."""
    denom = tn + fp
    return tn / denom if denom > 0 else np.nan


def _ppv(tp, fp):
    """Positive predictive value (precision)."""
    denom = tp + fp
    return tp / denom if denom > 0 else np.nan


def _npv(tn, fn):
    """Negative predictive value."""
    denom = tn + fn
    return tn / denom if denom > 0 else np.nan


def _accuracy(tp, tn, fp, fn):
    """Classification accuracy."""
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else np.nan


def _balanced_accuracy(tp, tn, fp, fn):
    """Balanced accuracy: (sensitivity + specificity) / 2."""
    sens = _sensitivity(tp, fn)
    spec = _specificity(tn, fp)
    if np.isnan(sens) or np.isnan(spec):
        return np.nan
    return (sens + spec) / 2


def _youden_index(tp, tn, fp, fn):
    """Youden index: sensitivity + specificity - 1."""
    sens = _sensitivity(tp, fn)
    spec = _specificity(tn, fp)
    if np.isnan(sens) or np.isnan(spec):
        return np.nan
    return sens + spec - 1


def _diagnostic_odds_ratio(tp, tn, fp, fn):
    """Diagnostic odds ratio: (TP * TN) / (FP * FN)."""
    if fp == 0 or fn == 0:
        return np.inf if (tp > 0 and tn > 0) else np.nan
    return (tp * tn) / (fp * fn)


def _kappa(tp, tn, fp, fn):
    """Cohen's kappa coefficient."""
    n = tp + tn + fp + fn
    if n == 0:
        return np.nan
    po = (tp + tn) / n  # observed agreement
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    if pe == 1:
        return np.nan
    return (po - pe) / (1 - pe)


def _f1_score(tp, fp, fn):
    """
    F1 score: harmonic mean of precision and recall.

    NOTE: The paper identifies F1 as the ONLY measure that is both improper
    AND lacks clear focus. It conflates classification with clinical utility,
    ignores true negatives, has no intuitive interpretation, and its value
    changes when outcome labels are swapped.
    """
    precision = _ppv(tp, fp)
    recall = _sensitivity(tp, fn)
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        return np.nan
    return 2 * (precision * recall) / (precision + recall)


def _mcc(tp, tn, fp, fn):
    """
    Matthews Correlation Coefficient.

    Improper measure. Range [-1, 1]. Higher is better.
    """
    denom = np.sqrt(
        float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    if denom == 0:
        return np.nan
    return (tp * tn - fp * fn) / denom


def classification_metrics(y_true, y_prob, threshold=0.1):
    """
    Compute all classification performance measures at a given threshold.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities of the event.
    threshold : float, default=0.1
        Decision threshold for classifying high vs low risk.

    Returns
    -------
    dict
        Dictionary with keys for all 11 classification measures.

    Notes
    -----
    ALL classification summary measures are improper at clinically relevant
    thresholds (other than t=0.5 for accuracy or t=prevalence for balanced
    accuracy/Youden/F1). The paper recommends against using these for
    clinical validation, in favor of clinical utility measures.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    tp, tn, fp, fn = _confusion_matrix(y_true, y_prob, threshold)

    return {
        "threshold": threshold,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        # Summary measures
        "accuracy": _accuracy(tp, tn, fp, fn),
        "balanced_accuracy": _balanced_accuracy(tp, tn, fp, fn),
        "youden_index": _youden_index(tp, tn, fp, fn),
        "diagnostic_odds_ratio": _diagnostic_odds_ratio(tp, tn, fp, fn),
        "kappa": _kappa(tp, tn, fp, fn),
        "f1_score": _f1_score(tp, fp, fn),
        "mcc": _mcc(tp, tn, fp, fn),
        # Partial / descriptive measures
        "sensitivity": _sensitivity(tp, fn),
        "specificity": _specificity(tn, fp),
        "ppv": _ppv(tp, fp),
        "npv": _npv(tn, fn),
    }
