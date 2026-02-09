"""
Bootstrap confidence interval computation.

Implements the percentile bootstrap method as used in the paper
(1000 bootstrap samples for 95% CIs).
"""

import numpy as np


def bootstrap_ci(y_true, y_prob, metric_fn, n_bootstrap=1000,
                 ci_level=0.95, random_state=42, **metric_kwargs):
    """
    Compute bootstrap confidence intervals for any metric function.

    Uses the percentile bootstrap method as described in the paper.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities.
    metric_fn : callable
        Function that takes (y_true, y_prob, **kwargs) and returns a dict.
    n_bootstrap : int, default=1000
        Number of bootstrap samples.
    ci_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI).
    random_state : int, default=42
        Random seed for reproducibility.
    **metric_kwargs
        Additional keyword arguments passed to metric_fn.

    Returns
    -------
    dict
        For each metric key, returns a dict with:
        - point: point estimate on original data
        - lower: lower bound of CI
        - upper: upper bound of CI
        - bootstrap_values: array of bootstrap estimates
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    rng = np.random.default_rng(random_state)

    # Point estimate
    point_estimates = metric_fn(y_true, y_prob, **metric_kwargs)

    # Bootstrap
    bootstrap_results = {key: [] for key in point_estimates}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_true_boot = y_true[idx]
        y_prob_boot = y_prob[idx]

        # Skip degenerate samples
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            boot_metrics = metric_fn(y_true_boot, y_prob_boot, **metric_kwargs)
            for key in bootstrap_results:
                bootstrap_results[key].append(boot_metrics[key])
        except Exception:
            continue

    # Compute CIs
    alpha = (1 - ci_level) / 2
    results = {}

    for key in point_estimates:
        boot_vals = np.array(bootstrap_results[key])
        boot_vals = boot_vals[np.isfinite(boot_vals)]

        if len(boot_vals) < 10:
            results[key] = {
                "point": point_estimates[key],
                "lower": np.nan,
                "upper": np.nan,
                "bootstrap_values": boot_vals,
            }
        else:
            results[key] = {
                "point": point_estimates[key],
                "lower": np.percentile(boot_vals, 100 * alpha),
                "upper": np.percentile(boot_vals, 100 * (1 - alpha)),
                "bootstrap_values": boot_vals,
            }

    return results
