"""Opinionated evaluation API for binary medical prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from medpred.metrics.calibration import calibration_metrics
from medpred.metrics.classification import classification_metrics
from medpred.metrics.clinical_utility import clinical_utility_metrics
from medpred.metrics.discrimination import discrimination_metrics
from medpred.metrics.overall import overall_metrics
from medpred.utils.bootstrap import bootstrap_ci
from medpred.visualization.plots import plot_core_evaluation


CORE_MEASURES = {
    "discrimination": ("auroc",),
    "calibration": ("oe_ratio", "calibration_intercept", "calibration_slope"),
    "overall": ("brier_score",),
    "clinical_utility": ("net_benefit", "standardized_net_benefit"),
    "classification_descriptive": ("sensitivity", "specificity", "ppv", "npv"),
}


MEASURE_GUIDANCE = {
    "auroc": ("recommended", "semi-proper", "clear statistical focus"),
    "oe_ratio": ("supporting", "semi-proper", "partial calibration summary"),
    "calibration_intercept": ("supporting", "semi-proper", "partial calibration summary"),
    "calibration_slope": ("supporting", "semi-proper", "partial calibration summary"),
    "brier_score": ("supporting", "strictly proper", "overall statistical score"),
    "net_benefit": ("recommended", "semi-proper", "decision-analytical focus"),
    "standardized_net_benefit": ("recommended", "semi-proper", "decision-analytical focus"),
    "sensitivity": ("descriptive", "improper alone", "report with specificity"),
    "specificity": ("descriptive", "improper alone", "report with sensitivity"),
    "ppv": ("descriptive", "improper alone", "report with NPV"),
    "npv": ("descriptive", "improper alone", "report with PPV"),
}


@dataclass(frozen=True)
class EvaluationReport:
    """Container returned by :func:`evaluate`."""

    core: dict[str, dict[str, float]]
    meta: dict[str, Any]
    extended: dict[str, dict[str, float]] | None = None
    ci: dict[str, dict[str, dict[str, float]]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like dictionary."""
        meta = {key: value for key, value in self.meta.items() if not key.startswith("_")}
        result: dict[str, Any] = {"core": self.core, "meta": meta}
        if self.extended is not None:
            result["extended"] = self.extended
        if self.ci is not None:
            result["ci"] = self.ci
        return result

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy table with paper guidance attached."""
        rows = []
        ci_lookup = self.ci or {}

        for domain, measures in self.core.items():
            for measure, value in measures.items():
                recommendation, properness, focus = MEASURE_GUIDANCE.get(
                    measure, ("", "", "")
                )
                row = {
                    "domain": domain,
                    "measure": measure,
                    "value": value,
                    "recommendation": recommendation,
                    "properness": properness,
                    "focus": focus,
                }
                ci_value = ci_lookup.get(domain, {}).get(measure)
                if ci_value:
                    row["ci_lower"] = ci_value.get("lower")
                    row["ci_upper"] = ci_value.get("upper")
                rows.append(row)

        return pd.DataFrame(rows)

    def plot(
        self,
        threshold_range: tuple[float, float] | None = None,
        save_path: str | None = None,
    ):
        """Plot the recommended visual evaluation panel."""
        return plot_report(self, threshold_range=threshold_range, save_path=save_path)


def evaluate(
    y_true,
    y_prob,
    *,
    threshold: float = 0.1,
    n_bootstrap: int | None = None,
    ci_level: float = 0.95,
    random_state: int | None = 42,
    include_extended: bool = False,
) -> EvaluationReport:
    """Evaluate a binary medical prediction model.

    The default report follows the paper's core recommendation: AUROC,
    calibration summaries supporting a calibration plot, Brier score as a
    compact proper overall score, net benefit, and descriptive paired
    threshold measures. Use :func:`evaluate_all` for the full metric inventory.
    """
    y_true, y_prob = _validate_inputs(y_true, y_prob)
    _validate_threshold(threshold)

    all_results = evaluate_all(y_true, y_prob, threshold=threshold)
    core = _select_core(all_results)
    ci = None

    if n_bootstrap is not None:
        ci = _core_ci(
            y_true,
            y_prob,
            threshold=threshold,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            random_state=random_state,
        )

    return EvaluationReport(
        core=core,
        meta={**all_results["meta"], "_y_true": y_true, "_y_prob": y_prob},
        extended=all_results if include_extended else None,
        ci=ci,
    )


def evaluate_all(
    y_true,
    y_prob,
    *,
    threshold: float = 0.1,
    cost_fn_ratio: float | None = None,
    min_sensitivity: float = 0.8,
) -> dict[str, dict[str, float]]:
    """Compute the full metric inventory discussed by the paper."""
    y_true, y_prob = _validate_inputs(y_true, y_prob)
    _validate_threshold(threshold)
    if cost_fn_ratio is None:
        cost_fn_ratio = 1 - threshold

    results = {
        "discrimination": discrimination_metrics(y_true, y_prob, min_sensitivity),
        "calibration": calibration_metrics(y_true, y_prob),
        "overall": overall_metrics(y_true, y_prob),
        "classification": classification_metrics(y_true, y_prob, threshold),
        "clinical_utility": clinical_utility_metrics(
            y_true, y_prob, threshold, cost_fn_ratio
        ),
        "meta": _metadata(y_true, threshold, cost_fn_ratio),
    }
    return results


def plot_report(
    report: EvaluationReport,
    *,
    threshold_range: tuple[float, float] | None = None,
    save_path: str | None = None,
):
    """Plot the recommended panel for an :class:`EvaluationReport`."""
    y_true = report.meta.get("_y_true")
    y_prob = report.meta.get("_y_prob")
    if y_true is None or y_prob is None:
        raise ValueError("Report does not contain plotting arrays.")
    if threshold_range is None:
        threshold_range = (0.01, 0.5)
    return plot_core_evaluation(
        y_true,
        y_prob,
        auroc=report.core["discrimination"]["auroc"],
        threshold_range=threshold_range,
        save_path=save_path,
    )


def _select_core(results: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        "discrimination": {"auroc": results["discrimination"]["auroc"]},
        "calibration": {
            key: results["calibration"][key] for key in CORE_MEASURES["calibration"]
        },
        "overall": {"brier_score": results["overall"]["brier_score"]},
        "clinical_utility": {
            key: results["clinical_utility"][key]
            for key in CORE_MEASURES["clinical_utility"]
        },
        "classification_descriptive": {
            key: results["classification"][key]
            for key in CORE_MEASURES["classification_descriptive"]
        },
    }


def _core_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    n_bootstrap: int,
    ci_level: float,
    random_state: int | None,
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        "discrimination": _keep_ci(
            bootstrap_ci(
                y_true,
                y_prob,
                discrimination_metrics,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                random_state=random_state,
            ),
            CORE_MEASURES["discrimination"],
        ),
        "calibration": _keep_ci(
            bootstrap_ci(
                y_true,
                y_prob,
                calibration_metrics,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                random_state=random_state,
            ),
            CORE_MEASURES["calibration"],
        ),
        "overall": _keep_ci(
            bootstrap_ci(
                y_true,
                y_prob,
                overall_metrics,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                random_state=random_state,
            ),
            CORE_MEASURES["overall"],
        ),
        "clinical_utility": _keep_ci(
            bootstrap_ci(
                y_true,
                y_prob,
                clinical_utility_metrics,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                random_state=random_state,
                threshold=threshold,
                cost_fn_ratio=1 - threshold,
            ),
            CORE_MEASURES["clinical_utility"],
        ),
        "classification_descriptive": _keep_ci(
            bootstrap_ci(
                y_true,
                y_prob,
                classification_metrics,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                random_state=random_state,
                threshold=threshold,
            ),
            CORE_MEASURES["classification_descriptive"],
        ),
    }


def _keep_ci(ci: dict[str, dict[str, float]], names: tuple[str, ...]):
    return {name: ci[name] for name in names}


def _metadata(y_true: np.ndarray, threshold: float, cost_fn_ratio: float) -> dict[str, Any]:
    return {
        "n": int(len(y_true)),
        "n_events": int(np.sum(y_true)),
        "n_non_events": int(np.sum(y_true == 0)),
        "prevalence": float(np.mean(y_true)),
        "threshold": float(threshold),
        "false_negative_cost": float(cost_fn_ratio),
        "false_positive_cost": float(1 - cost_fn_ratio),
    }


def _validate_inputs(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional.")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if len(y_true) == 0:
        raise ValueError("y_true and y_prob cannot be empty.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")
    if len(np.unique(y_true)) != 2:
        raise ValueError("y_true must contain at least one event and one non-event.")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob must contain finite values.")
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("y_prob must contain probabilities in [0, 1].")

    return y_true, y_prob


def _validate_threshold(threshold: float) -> None:
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")
