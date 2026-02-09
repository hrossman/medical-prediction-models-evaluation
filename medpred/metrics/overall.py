"""
Overall performance measures for binary prediction models.

Overall performance combines discrimination and calibration by quantifying
how closely the probability estimates approach the actual outcomes of 0/1.

Measures implemented:
- Loglikelihood (strictly proper, clear focus)
- Logloss / cross-entropy (strictly proper, clear focus)
- Brier score (strictly proper, clear focus)
- Scaled Brier / IPA (asymptotically strictly proper, clear focus)
- McFadden R-squared (asymptotically strictly proper, clear focus)
- Cox-Snell R-squared (asymptotically strictly proper, clear focus)
- Nagelkerke R-squared (asymptotically strictly proper, clear focus)
- Discrimination slope (improper, clear focus)
- Mean absolute prediction error (improper, clear focus)

Reference: Van Calster et al. (2025), The Lancet Digital Health.
"""

import numpy as np
from sklearn.metrics import log_loss, brier_score_loss


def _loglikelihood(y_true, y_prob):
    """
    Loglikelihood of the model.

    Strictly proper measure. Higher (less negative) values indicate
    better performance.
    """
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    ll = np.sum(
        y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped)
    )
    return ll


def _logloss(y_true, y_prob):
    """
    Logloss (cross-entropy, negative loglikelihood).

    Strictly proper measure. Lower values indicate better performance.
    This is the negative loglikelihood (total, not averaged per sample).
    """
    return -_loglikelihood(y_true, y_prob)


def _brier_score(y_true, y_prob):
    """
    Brier score: mean squared difference between outcomes and predictions.

    Strictly proper. Range [0, 1]. Lower is better.
    """
    return np.mean((y_true - y_prob) ** 2)


def _null_model_brier(y_true):
    """Brier score for a null model that predicts prevalence for everyone."""
    prevalence = np.mean(y_true)
    return prevalence * (1 - prevalence)


def _scaled_brier(y_true, y_prob):
    """
    Scaled Brier score (Brier skill score / Index of Prediction Accuracy).

    Scaled Brier = 1 - Brier / Brier_null

    Range (-inf, 1]. 0 = no better than prevalence, 1 = perfect.
    Asymptotically strictly proper.
    """
    brier = _brier_score(y_true, y_prob)
    brier_null = _null_model_brier(y_true)
    if brier_null == 0:
        return np.nan
    return 1 - brier / brier_null


def _null_loglikelihood(y_true):
    """Loglikelihood for a null model predicting prevalence."""
    prevalence = np.mean(y_true)
    prevalence = np.clip(prevalence, 1e-15, 1 - 1e-15)
    n = len(y_true)
    n1 = np.sum(y_true)
    n0 = n - n1
    return n1 * np.log(prevalence) + n0 * np.log(1 - prevalence)


def _mcfadden_r2(y_true, y_prob):
    """
    McFadden's R-squared: 1 - LL_model / LL_null.

    Asymptotically strictly proper. Range [0, 1). Higher is better.
    """
    ll_model = _loglikelihood(y_true, y_prob)
    ll_null = _null_loglikelihood(y_true)
    if ll_null == 0:
        return np.nan
    return 1 - ll_model / ll_null


def _cox_snell_r2(y_true, y_prob):
    """
    Cox-Snell R-squared.

    R2_CS = 1 - (L_null / L_model)^(2/n)

    Asymptotically strictly proper. Cannot reach 1.
    """
    n = len(y_true)
    ll_model = _loglikelihood(y_true, y_prob)
    ll_null = _null_loglikelihood(y_true)
    return 1 - np.exp((2 / n) * (ll_null - ll_model))


def _nagelkerke_r2(y_true, y_prob):
    """
    Nagelkerke's R-squared: Cox-Snell R2 scaled to have max of 1.

    R2_N = R2_CS / R2_CS_max

    Asymptotically strictly proper. Range [0, 1].
    """
    n = len(y_true)
    ll_null = _null_loglikelihood(y_true)
    r2_cs = _cox_snell_r2(y_true, y_prob)
    r2_cs_max = 1 - np.exp((2 / n) * ll_null)
    if r2_cs_max == 0:
        return np.nan
    return r2_cs / r2_cs_max


def _discrimination_slope(y_true, y_prob):
    """
    Discrimination slope (coefficient of discrimination).

    Mean predicted probability for events minus mean for non-events.
    Improper measure. Higher is better.
    """
    events = y_prob[y_true == 1]
    non_events = y_prob[y_true == 0]
    if len(events) == 0 or len(non_events) == 0:
        return np.nan
    return np.mean(events) - np.mean(non_events)


def _mean_absolute_prediction_error(y_true, y_prob):
    """
    Mean absolute prediction error (MAPE).

    Average absolute difference between observed outcomes and predicted
    probabilities. Improper measure. Lower is better.
    """
    return np.mean(np.abs(y_true - y_prob))


def overall_metrics(y_true, y_prob):
    """
    Compute all overall performance measures.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities of the event.

    Returns
    -------
    dict
        Dictionary with keys:
        - loglikelihood: Model loglikelihood (strictly proper)
        - logloss: Negative loglikelihood / cross-entropy (strictly proper)
        - brier_score: Brier score (strictly proper)
        - scaled_brier: Scaled Brier / IPA (asympt. strictly proper)
        - mcfadden_r2: McFadden R-squared (asympt. strictly proper)
        - cox_snell_r2: Cox-Snell R-squared (asympt. strictly proper)
        - nagelkerke_r2: Nagelkerke R-squared (asympt. strictly proper)
        - discrimination_slope: Discrimination slope (improper)
        - mape: Mean absolute prediction error (improper)

    Notes
    -----
    The paper recommends evaluating discrimination and calibration separately
    rather than using overall measures. These are more relevant for model
    selection tasks.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    return {
        "loglikelihood": _loglikelihood(y_true, y_prob),
        "logloss": _logloss(y_true, y_prob),
        "brier_score": _brier_score(y_true, y_prob),
        "scaled_brier": _scaled_brier(y_true, y_prob),
        "mcfadden_r2": _mcfadden_r2(y_true, y_prob),
        "cox_snell_r2": _cox_snell_r2(y_true, y_prob),
        "nagelkerke_r2": _nagelkerke_r2(y_true, y_prob),
        "discrimination_slope": _discrimination_slope(y_true, y_prob),
        "mape": _mean_absolute_prediction_error(y_true, y_prob),
    }
