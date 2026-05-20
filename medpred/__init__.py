"""Medical prediction model evaluation for binary risk models."""

from medpred.core import EvaluationReport, evaluate, evaluate_all, plot_report
from medpred.metrics.calibration import logistic_recalibration

__all__ = [
    "EvaluationReport",
    "evaluate",
    "evaluate_all",
    "logistic_recalibration",
    "plot_report",
]

__version__ = "0.2.0"
