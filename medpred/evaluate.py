"""Compatibility imports for the public evaluation API."""

from medpred.core import EvaluationReport, evaluate, evaluate_all


def evaluate_model(*args, **kwargs):
    """Deprecated alias for :func:`medpred.evaluate_all`."""
    return evaluate_all(*args, **kwargs)


def evaluate_with_ci(*args, **kwargs):
    """Deprecated alias returning the core report with bootstrap intervals."""
    if "n_bootstrap" not in kwargs:
        kwargs["n_bootstrap"] = 1000
    return evaluate(*args, **kwargs)


def results_to_dataframe(results, *_, **__):
    """Return a tidy table from either an EvaluationReport or a result dict."""
    if isinstance(results, EvaluationReport):
        return results.to_frame()

    report = EvaluationReport(
        core={
            "discrimination": {"auroc": results["discrimination"]["auroc"]},
            "calibration": {
                "oe_ratio": results["calibration"]["oe_ratio"],
                "calibration_intercept": results["calibration"]["calibration_intercept"],
                "calibration_slope": results["calibration"]["calibration_slope"],
            },
            "overall": {"brier_score": results["overall"]["brier_score"]},
            "clinical_utility": {
                "net_benefit": results["clinical_utility"]["net_benefit"],
                "standardized_net_benefit": results["clinical_utility"][
                    "standardized_net_benefit"
                ],
            },
            "classification_descriptive": {
                "sensitivity": results["classification"]["sensitivity"],
                "specificity": results["classification"]["specificity"],
                "ppv": results["classification"]["ppv"],
                "npv": results["classification"]["npv"],
            },
        },
        meta=results.get("meta", {}),
        extended=results,
    )
    return report.to_frame()


__all__ = [
    "EvaluationReport",
    "evaluate",
    "evaluate_all",
    "evaluate_model",
    "evaluate_with_ci",
    "results_to_dataframe",
]
