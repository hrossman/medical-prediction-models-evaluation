"""
Discrimination performance measures for binary prediction models.

Discrimination focuses on the extent to which the model assigns higher
probabilities of the event for individuals with the event than for those
without. Three measures are implemented:

1. AUROC (C-statistic) - Recommended. Semi-proper, clear focus.
2. AUPRC (Average Precision) - Semi-proper, but lacks clear focus.
3. pAUROC (Partial AUROC) - Semi-proper, but lacks clear focus.

Reference: Van Calster et al. (2025), The Lancet Digital Health.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc


def _fast_auc(y_true, y_prob):
    """Compute AUC using the rank-based (Mann-Whitney U) method."""
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = np.argsort(np.argsort(y_prob)) + 1  # 1-based ranks
    sum_ranks = np.sum(ranks[y_true == 1])
    return (sum_ranks - n1 * (n1 + 1) / 2) / (n1 * n0)


def _average_precision(y_true, y_prob):
    """Compute average precision (area under precision-recall curve)."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def _partial_auroc(y_true, y_prob, min_sensitivity=0.8):
    """
    Compute partial AUROC restricting to sensitivity >= min_sensitivity.

    This focuses on the region of the ROC curve where sensitivity
    is at least the specified minimum.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    # Filter to where sensitivity (tpr) >= min_sensitivity
    mask = tpr >= min_sensitivity
    if not np.any(mask):
        return np.nan

    fpr_partial = fpr[mask]
    tpr_partial = tpr[mask]

    if len(fpr_partial) < 2:
        return np.nan

    return auc(fpr_partial, tpr_partial)


def discrimination_metrics(y_true, y_prob, min_sensitivity=0.8):
    """
    Compute all discrimination performance measures.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities of the event.
    min_sensitivity : float, default=0.8
        Minimum sensitivity threshold for partial AUROC.

    Returns
    -------
    dict
        Dictionary with keys:
        - auroc: Area under ROC curve (C-statistic)
        - auprc: Area under precision-recall curve
        - pauroc: Partial AUROC (sensitivity >= min_sensitivity)

    Notes
    -----
    AUROC is the recommended discrimination measure (semi-proper, clear focus).
    AUPRC and pAUROC are semi-proper but lack clear decision-analytical focus
    and are considered inadvisable for clinical validation per the paper.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": _average_precision(y_true, y_prob),
        "pauroc": _partial_auroc(y_true, y_prob, min_sensitivity),
    }


def roc_curve_data(y_true, y_prob):
    """Return FPR, TPR, thresholds for ROC curve plotting."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def pr_curve_data(y_true, y_prob):
    """Return precision, recall, thresholds for PR curve plotting."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}
