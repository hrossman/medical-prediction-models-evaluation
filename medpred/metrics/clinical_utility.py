"""
Clinical utility measures for binary prediction models.

Clinical utility is the most important performance domain per the paper.
It explicitly incorporates misclassification costs when evaluating
classifications, following decision-analytical principles.

Measures:
- Net benefit (semi-proper, clear focus) - RECOMMENDED
- Standardized net benefit (semi-proper, clear focus) - RECOMMENDED
- Expected cost (semi-proper, clear focus) - RECOMMENDED

Reference: Van Calster et al. (2025), The Lancet Digital Health.
"""

import numpy as np


def _net_benefit(y_true, y_prob, threshold):
    """
    Net benefit at a given decision threshold.

    NB = (TP/n) - (FP/n) * (t / (1-t))

    where t is the decision threshold.

    The threshold encodes the relative cost of false positives to
    false negatives: using threshold t implies that the harm of a
    false negative is (1-t)/t times greater than the harm of a
    false positive.

    Maximum value equals the prevalence.
    """
    if threshold <= 0 or threshold >= 1:
        return np.nan

    n = len(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    return (tp / n) - (fp / n) * (threshold / (1 - threshold))


def _standardized_net_benefit(y_true, y_prob, threshold):
    """
    Standardized net benefit: NB / prevalence.

    Maximum value is 1. Easier to interpret across different prevalences.
    """
    nb = _net_benefit(y_true, y_prob, threshold)
    prevalence = np.mean(y_true)
    if prevalence == 0:
        return np.nan
    return nb / prevalence


def _treat_all_net_benefit(y_true, threshold):
    """Net benefit for the 'treat all' strategy."""
    if threshold <= 0 or threshold >= 1:
        return np.nan
    prevalence = np.mean(y_true)
    return prevalence - (1 - prevalence) * (threshold / (1 - threshold))


def _expected_cost(y_true, y_prob, cost_fn_ratio=0.9):
    """
    Expected cost at the optimal threshold for given cost ratio.

    cost_fn_ratio: normalized cost of a false negative (vs false positive).
    E.g., cost_fn_ratio=0.9 means FN is 9x more costly than FP.

    The function finds the threshold that minimizes expected cost.
    """
    cost_fp_ratio = 1 - cost_fn_ratio
    prevalence = np.mean(y_true)
    n = len(y_true)

    thresholds = np.sort(np.unique(y_prob))
    # Add boundary thresholds
    thresholds = np.concatenate([[0], thresholds, [1]])

    min_cost = np.inf
    best_threshold = None

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fn = np.sum((y_pred == 0) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        expected_cost = (
            cost_fn_ratio * prevalence * (fn / max(np.sum(y_true), 1))
            + cost_fp_ratio * (1 - prevalence) * (fp / max(np.sum(y_true == 0), 1))
        )

        if expected_cost < min_cost:
            min_cost = expected_cost
            best_threshold = t

    return min_cost, best_threshold


def clinical_utility_metrics(y_true, y_prob, threshold=0.1, cost_fn_ratio=0.9):
    """
    Compute all clinical utility performance measures.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities of the event.
    threshold : float, default=0.1
        Decision threshold for net benefit calculation.
        t=0.1 implies accepting up to 9 FP per TP.
    cost_fn_ratio : float, default=0.9
        Normalized cost of false negative for expected cost.
        0.9 means FN cost is 9x the FP cost.

    Returns
    -------
    dict
        Dictionary with keys:
        - net_benefit: Net benefit at the threshold
        - standardized_net_benefit: NB / prevalence
        - treat_all_nb: Net benefit for 'treat all' strategy
        - treat_none_nb: Net benefit for 'treat none' (always 0)
        - expected_cost: Minimum expected cost
        - expected_cost_threshold: Threshold minimizing expected cost
        - prevalence: Observed prevalence

    Notes
    -----
    Clinical utility is the most important performance domain.
    The key concern is whether the model has better utility than the
    reference strategies (treat all or treat none) and competing models.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    ec, ec_threshold = _expected_cost(y_true, y_prob, cost_fn_ratio)

    return {
        "net_benefit": _net_benefit(y_true, y_prob, threshold),
        "standardized_net_benefit": _standardized_net_benefit(y_true, y_prob, threshold),
        "treat_all_nb": _treat_all_net_benefit(y_true, threshold),
        "treat_none_nb": 0.0,
        "expected_cost": ec,
        "expected_cost_threshold": ec_threshold,
        "prevalence": float(np.mean(y_true)),
    }


def net_benefit_curve(y_true, y_prob, thresholds=None):
    """
    Compute net benefit across a range of thresholds for decision curve analysis.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes.
    y_prob : array-like
        Predicted probabilities.
    thresholds : array-like, optional
        Thresholds to evaluate. Default: 0.01 to 0.99 in steps of 0.01.

    Returns
    -------
    dict
        - thresholds: array of thresholds
        - model_nb: net benefit of the model at each threshold
        - treat_all_nb: net benefit of treating all
        - treat_none_nb: net benefit of treating none (always 0)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    model_nb = [_net_benefit(y_true, y_prob, t) for t in thresholds]
    treat_all_nb = [_treat_all_net_benefit(y_true, t) for t in thresholds]

    return {
        "thresholds": thresholds,
        "model_nb": np.array(model_nb),
        "treat_all_nb": np.array(treat_all_nb),
        "treat_none_nb": np.zeros_like(thresholds),
    }


def expected_cost_curve(y_true, y_prob, cost_ratios=None):
    """
    Compute expected cost across a range of FN cost ratios.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes.
    y_prob : array-like
        Predicted probabilities.
    cost_ratios : array-like, optional
        Normalized FN cost ratios. Default: 0.01 to 0.99 in steps of 0.01.

    Returns
    -------
    dict
        - cost_ratios: array of FN cost ratios
        - model_ec: expected cost of the model at each ratio
        - treat_all_ec: expected cost of treating all
        - treat_none_ec: expected cost of treating none
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if cost_ratios is None:
        cost_ratios = np.arange(0.01, 1.0, 0.01)

    prevalence = np.mean(y_true)
    model_ec = []
    treat_all_ec = []
    treat_none_ec = []

    for cfn in cost_ratios:
        cfp = 1 - cfn
        ec, _ = _expected_cost(y_true, y_prob, cfn)
        model_ec.append(ec)
        # Treat all: FN=0, FP = all non-events
        treat_all_ec.append(cfp * (1 - prevalence))
        # Treat none: FN = all events, FP=0
        treat_none_ec.append(cfn * prevalence)

    return {
        "cost_ratios": cost_ratios,
        "model_ec": np.array(model_ec),
        "treat_all_ec": np.array(treat_all_ec),
        "treat_none_ec": np.array(treat_none_ec),
    }
