"""
Calibration performance measures for binary prediction models.

Calibration focuses on the extent to which predicted probabilities correspond
to observed event proportions. Three levels of calibration are assessed:

- Mean calibration: O:E ratio, calibration intercept
- Weak calibration: calibration slope
- Moderate calibration: ECI, ICI, ECE (via calibration plots)

All calibration measures are semi-proper with clear focus on statistical
performance.

Reference: Van Calster et al. (2025), The Lancet Digital Health.
"""

import numpy as np
import warnings
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression


def _oe_ratio(y_true, y_prob):
    """
    Observed over Expected (O:E) ratio.

    O:E = observed prevalence / mean predicted probability.
    Values >1 indicate the model underestimates risk on average.
    Values <1 indicate the model overestimates risk on average.
    """
    observed = np.mean(y_true)
    expected = np.mean(y_prob)
    if expected == 0:
        return np.nan
    return observed / expected


def _calibration_intercept_slope(y_true, y_prob):
    """
    Calibration intercept and slope from logistic recalibration.

    Fits: logit(P(Y=1)) = alpha + beta * logit(y_prob)

    - Intercept (alpha): 0 indicates perfect mean calibration
    - Slope (beta): 1 indicates adequate spread of probabilities
      - slope < 1: probabilities too spread (too extreme)
      - slope > 1: probabilities too close to prevalence
    """
    # Clip probabilities to avoid infinite logit values
    y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
    lp = logit(y_prob_clipped)

    # Fit logistic regression: outcome ~ logit(predicted_prob)
    # Use very large C to effectively disable regularization
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y_true)

    intercept = lr.intercept_[0]
    slope = lr.coef_[0][0]
    return intercept, slope


def _loess_calibration(y_true, y_prob, frac=0.75):
    """
    Compute LOESS-smoothed calibration curve.

    Returns sorted predicted probabilities and their corresponding
    smoothed observed proportions.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    result = lowess(y_true, y_prob, frac=frac, return_sorted=True)
    return result[:, 0], result[:, 1]


def _eci(y_true, y_prob, frac=0.75):
    """
    Estimated Calibration Index (ECI).

    Based on a smoothed calibration curve, ECI is the average squared
    difference between the smoothed calibration curve and the diagonal.
    """
    p_sorted, p_smooth = _loess_calibration(y_true, y_prob, frac=frac)
    return np.mean((p_smooth - p_sorted) ** 2)


def _ici(y_true, y_prob, frac=0.75):
    """
    Integrated Calibration Index (ICI).

    The weighted average absolute difference between the smoothed
    calibration curve and the diagonal (perfect calibration).
    """
    p_sorted, p_smooth = _loess_calibration(y_true, y_prob, frac=frac)
    return np.mean(np.abs(p_smooth - p_sorted))


def _ece(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE).

    Groups predictions into bins and computes the weighted average of
    the absolute difference between the bin's average prediction and
    the observed proportion in each bin.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_bin = np.sum(mask)
        if n_bin == 0:
            continue

        avg_pred = np.mean(y_prob[mask])
        avg_true = np.mean(y_true[mask])
        ece += (n_bin / n) * np.abs(avg_true - avg_pred)

    return ece


def calibration_metrics(y_true, y_prob, n_bins=10, loess_frac=0.75):
    """
    Compute all calibration performance measures.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities of the event.
    n_bins : int, default=10
        Number of bins for ECE calculation.
    loess_frac : float, default=0.75
        Fraction of data used in LOESS smoothing for ECI/ICI.

    Returns
    -------
    dict
        Dictionary with keys:
        - oe_ratio: Observed/Expected ratio (ideal = 1.0)
        - calibration_intercept: Logistic recalibration intercept (ideal = 0)
        - calibration_slope: Logistic recalibration slope (ideal = 1.0)
        - eci: Estimated Calibration Index (ideal = 0)
        - ici: Integrated Calibration Index (ideal = 0)
        - ece: Expected Calibration Error (ideal = 0)

    Notes
    -----
    All calibration measures are semi-proper. The calibration plot (not a
    single number) is the recommended approach for assessing calibration.
    ECI, ICI, and ECE have known issues with statistical consistency and
    conceal the direction of miscalibration.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    intercept, slope = _calibration_intercept_slope(y_true, y_prob)

    return {
        "oe_ratio": _oe_ratio(y_true, y_prob),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "eci": _eci(y_true, y_prob, frac=loess_frac),
        "ici": _ici(y_true, y_prob, frac=loess_frac),
        "ece": _ece(y_true, y_prob, n_bins=n_bins),
    }


def logistic_recalibration(y_true, y_prob):
    """
    Perform logistic recalibration (Platt scaling).

    Fits logit(P(Y=1)) = alpha + beta * logit(y_prob) and returns
    recalibrated probabilities.

    This is a rank-preserving transformation, so discrimination measures
    are unchanged after recalibration.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Original predicted probabilities.

    Returns
    -------
    array
        Recalibrated predicted probabilities.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
    lp = logit(y_prob_clipped)

    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y_true)

    recal_lp = lr.intercept_[0] + lr.coef_[0][0] * lp
    return expit(recal_lp)


def calibration_curve_data(y_true, y_prob, n_groups=10, loess_frac=0.75):
    """
    Compute data for calibration plot (grouped and smoothed).

    Parameters
    ----------
    y_true : array-like
        Binary outcomes.
    y_prob : array-like
        Predicted probabilities.
    n_groups : int, default=10
        Number of equal-sized groups for grouped calibration.
    loess_frac : float, default=0.75
        LOESS smoothing fraction.

    Returns
    -------
    dict
        - grouped_pred: mean predicted probability per group
        - grouped_obs: observed proportion per group
        - smooth_pred: sorted predicted probabilities (LOESS)
        - smooth_obs: smoothed observed proportions (LOESS)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Grouped calibration (equal-size groups)
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    groups = np.array_split(np.arange(len(y_true)), n_groups)
    grouped_pred = [np.mean(y_prob_sorted[g]) for g in groups]
    grouped_obs = [np.mean(y_true_sorted[g]) for g in groups]

    # LOESS smoothed calibration
    smooth_pred, smooth_obs = _loess_calibration(y_true, y_prob, frac=loess_frac)

    return {
        "grouped_pred": np.array(grouped_pred),
        "grouped_obs": np.array(grouped_obs),
        "smooth_pred": smooth_pred,
        "smooth_obs": smooth_obs,
    }
