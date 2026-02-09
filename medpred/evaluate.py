"""
Comprehensive model evaluation pipeline.

Implements the recommended evaluation framework from Van Calster et al. (2025):
- AUROC (discrimination)
- Calibration plot (calibration)
- Net benefit with decision curve (clinical utility)
- Risk distribution plots

Plus all 32 performance measures across 5 domains.
"""

import numpy as np
import pandas as pd

from medpred.metrics.discrimination import discrimination_metrics
from medpred.metrics.calibration import calibration_metrics, logistic_recalibration
from medpred.metrics.overall import overall_metrics
from medpred.metrics.classification import classification_metrics
from medpred.metrics.clinical_utility import clinical_utility_metrics
from medpred.utils.bootstrap import bootstrap_ci


def evaluate_model(y_true, y_prob, threshold=0.1, cost_fn_ratio=0.9,
                   min_sensitivity=0.8):
    """
    Run complete model evaluation across all 5 performance domains.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities of the event.
    threshold : float, default=0.1
        Decision threshold for classification and clinical utility.
    cost_fn_ratio : float, default=0.9
        Normalized cost of false negative for expected cost.
    min_sensitivity : float, default=0.8
        Minimum sensitivity for partial AUROC.

    Returns
    -------
    dict
        Nested dictionary with results for each domain:
        - discrimination: AUROC, AUPRC, pAUROC
        - calibration: O:E ratio, intercept, slope, ECI, ICI, ECE
        - overall: Brier, logloss, R-squared measures, etc.
        - classification: accuracy, F1, MCC, sensitivity, etc.
        - clinical_utility: net benefit, standardized NB, expected cost
        - meta: prevalence, sample size, threshold used
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    results = {
        "discrimination": discrimination_metrics(y_true, y_prob, min_sensitivity),
        "calibration": calibration_metrics(y_true, y_prob),
        "overall": overall_metrics(y_true, y_prob),
        "classification": classification_metrics(y_true, y_prob, threshold),
        "clinical_utility": clinical_utility_metrics(y_true, y_prob, threshold, cost_fn_ratio),
        "meta": {
            "n": len(y_true),
            "n_events": int(np.sum(y_true)),
            "n_non_events": int(np.sum(y_true == 0)),
            "prevalence": float(np.mean(y_true)),
            "threshold": threshold,
            "cost_fn_ratio": cost_fn_ratio,
        },
    }

    return results


def evaluate_with_ci(y_true, y_prob, threshold=0.1, cost_fn_ratio=0.9,
                     min_sensitivity=0.8, n_bootstrap=1000, ci_level=0.95,
                     random_state=42):
    """
    Run complete evaluation with bootstrap confidence intervals.

    Uses the percentile bootstrap method with the specified number of
    bootstrap samples, as described in the paper.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities of the event.
    threshold : float, default=0.1
        Decision threshold.
    cost_fn_ratio : float, default=0.9
        Normalized cost of false negative.
    min_sensitivity : float, default=0.8
        Minimum sensitivity for pAUROC.
    n_bootstrap : int, default=1000
        Number of bootstrap samples.
    ci_level : float, default=0.95
        Confidence level for intervals.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    dict
        Same structure as evaluate_model, but each metric value is replaced
        with a dict containing 'point', 'lower', 'upper' keys.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    results = {
        "discrimination": bootstrap_ci(
            y_true, y_prob, discrimination_metrics,
            n_bootstrap=n_bootstrap, ci_level=ci_level,
            random_state=random_state, min_sensitivity=min_sensitivity
        ),
        "calibration": bootstrap_ci(
            y_true, y_prob, calibration_metrics,
            n_bootstrap=n_bootstrap, ci_level=ci_level,
            random_state=random_state
        ),
        "overall": bootstrap_ci(
            y_true, y_prob, overall_metrics,
            n_bootstrap=n_bootstrap, ci_level=ci_level,
            random_state=random_state
        ),
        "classification": bootstrap_ci(
            y_true, y_prob, classification_metrics,
            n_bootstrap=n_bootstrap, ci_level=ci_level,
            random_state=random_state, threshold=threshold
        ),
        "clinical_utility": bootstrap_ci(
            y_true, y_prob, clinical_utility_metrics,
            n_bootstrap=n_bootstrap, ci_level=ci_level,
            random_state=random_state, threshold=threshold,
            cost_fn_ratio=cost_fn_ratio
        ),
        "meta": {
            "n": len(y_true),
            "n_events": int(np.sum(y_true)),
            "n_non_events": int(np.sum(y_true == 0)),
            "prevalence": float(np.mean(y_true)),
            "threshold": threshold,
            "cost_fn_ratio": cost_fn_ratio,
            "n_bootstrap": n_bootstrap,
            "ci_level": ci_level,
        },
    }

    return results


def results_to_dataframe(results, with_ci=False):
    """
    Convert evaluation results to a pandas DataFrame.

    Parameters
    ----------
    results : dict
        Output from evaluate_model or evaluate_with_ci.
    with_ci : bool
        Whether results include confidence intervals.

    Returns
    -------
    pandas.DataFrame
        Table with domain, measure, value (and CI bounds if applicable).
    """
    rows = []
    properness = _get_properness_map()
    focus = _get_focus_map()
    recommendation = _get_recommendation_map()

    for domain in ["discrimination", "calibration", "overall",
                   "classification", "clinical_utility"]:
        if domain not in results:
            continue
        domain_results = results[domain]

        for measure, value in domain_results.items():
            if measure in ("threshold", "tp", "tn", "fp", "fn",
                          "treat_none_nb", "prevalence",
                          "expected_cost_threshold"):
                continue

            row = {
                "domain": domain,
                "measure": measure,
                "properness": properness.get(measure, ""),
                "focus": focus.get(measure, ""),
                "recommendation": recommendation.get(measure, ""),
            }

            if with_ci and isinstance(value, dict):
                row["value"] = value.get("point", np.nan)
                row["ci_lower"] = value.get("lower", np.nan)
                row["ci_upper"] = value.get("upper", np.nan)
            else:
                row["value"] = value

            rows.append(row)

    return pd.DataFrame(rows)


def _get_properness_map():
    """Map measure names to properness status from Table 1."""
    return {
        "auroc": "semi-proper",
        "auprc": "semi-proper",
        "pauroc": "semi-proper",
        "oe_ratio": "semi-proper",
        "calibration_intercept": "semi-proper",
        "calibration_slope": "semi-proper",
        "eci": "semi-proper",
        "ici": "semi-proper",
        "ece": "semi-proper",
        "loglikelihood": "strictly proper",
        "logloss": "strictly proper",
        "brier_score": "strictly proper",
        "scaled_brier": "asympt. strictly proper",
        "mcfadden_r2": "asympt. strictly proper",
        "cox_snell_r2": "asympt. strictly proper",
        "nagelkerke_r2": "asympt. strictly proper",
        "discrimination_slope": "improper",
        "mape": "improper",
        "accuracy": "improper",
        "balanced_accuracy": "improper",
        "youden_index": "improper",
        "diagnostic_odds_ratio": "improper",
        "kappa": "improper",
        "f1_score": "improper",
        "mcc": "improper",
        "sensitivity": "improper",
        "specificity": "improper",
        "ppv": "improper",
        "npv": "improper",
        "net_benefit": "semi-proper",
        "standardized_net_benefit": "semi-proper",
        "expected_cost": "semi-proper",
    }


def _get_focus_map():
    """Map measure names to focus status (clear or not) from Table 1."""
    return {
        "auroc": "clear",
        "auprc": "unclear",
        "pauroc": "unclear",
        "oe_ratio": "clear",
        "calibration_intercept": "clear",
        "calibration_slope": "clear",
        "eci": "clear",
        "ici": "clear",
        "ece": "clear",
        "loglikelihood": "clear",
        "logloss": "clear",
        "brier_score": "clear",
        "scaled_brier": "clear",
        "mcfadden_r2": "clear",
        "cox_snell_r2": "clear",
        "nagelkerke_r2": "clear",
        "discrimination_slope": "clear",
        "mape": "clear",
        "accuracy": "clear",
        "balanced_accuracy": "clear",
        "youden_index": "clear",
        "diagnostic_odds_ratio": "clear",
        "kappa": "clear",
        "f1_score": "unclear",
        "mcc": "clear",
        "sensitivity": "clear",
        "specificity": "clear",
        "ppv": "clear",
        "npv": "clear",
        "net_benefit": "clear",
        "standardized_net_benefit": "clear",
        "expected_cost": "clear",
    }


def _get_recommendation_map():
    """Map measure names to paper recommendations from Table 2."""
    return {
        "auroc": "Recommended",
        "auprc": "Inadvisable",
        "pauroc": "Inadvisable",
        "oe_ratio": "Not essential",
        "calibration_intercept": "Not essential",
        "calibration_slope": "Not essential",
        "eci": "Not essential",
        "ici": "Not essential",
        "ece": "Not essential",
        "loglikelihood": "Not essential",
        "logloss": "Not essential",
        "brier_score": "Not essential",
        "scaled_brier": "Not essential",
        "mcfadden_r2": "Not essential",
        "cox_snell_r2": "Not essential",
        "nagelkerke_r2": "Not essential",
        "discrimination_slope": "Inadvisable",
        "mape": "Inadvisable",
        "accuracy": "Inadvisable",
        "balanced_accuracy": "Inadvisable",
        "youden_index": "Inadvisable",
        "diagnostic_odds_ratio": "Inadvisable",
        "kappa": "Inadvisable",
        "f1_score": "Inadvisable",
        "mcc": "Inadvisable",
        "sensitivity": "Descriptive (with specificity)",
        "specificity": "Descriptive (with sensitivity)",
        "ppv": "Descriptive (with NPV)",
        "npv": "Descriptive (with PPV)",
        "net_benefit": "Recommended",
        "standardized_net_benefit": "Recommended",
        "expected_cost": "Recommended",
    }
