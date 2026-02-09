"""
Visualization functions for medical prediction model evaluation.

Recommended plots per Van Calster et al. (2025):
1. Calibration plot (smoothed) - RECOMMENDED
2. Decision curve (net benefit) - RECOMMENDED
3. Risk distribution by outcome - RECOMMENDED
4. ROC curve - acceptable but limited added value over AUROC
5. PR curve - acceptable but inadvisable per paper

Additional:
6. Expected cost curve
7. Classification plot (threshold on x-axis, measures on y-axis)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from medpred.metrics.discrimination import roc_curve_data, pr_curve_data
from medpred.metrics.calibration import calibration_curve_data
from medpred.metrics.clinical_utility import net_benefit_curve, expected_cost_curve
from medpred.metrics.classification import classification_metrics


def plot_roc_curve(y_true, y_prob, auroc=None, ax=None, save_path=None):
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes.
    y_prob : array-like
        Predicted probabilities.
    auroc : float, optional
        Pre-computed AUROC value to display.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    save_path : str, optional
        Path to save the figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.get_figure()

    data = roc_curve_data(y_true, y_prob)

    ax.plot(data["fpr"], data["tpr"], color="#2563eb", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect("equal")

    if auroc is not None:
        ax.set_title(f"ROC Curve (AUROC = {auroc:.3f})")
    else:
        ax.set_title("ROC Curve")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_pr_curve(y_true, y_prob, auprc=None, ax=None, save_path=None):
    """
    Plot Precision-Recall curve.

    Note: The paper considers AUPRC inadvisable for clinical validation
    because it mixes statistical and decision-analytical performance.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.get_figure()

    data = pr_curve_data(y_true, y_prob)
    prevalence = np.mean(np.asarray(y_true))

    ax.plot(data["recall"], data["precision"], color="#2563eb", linewidth=2)
    ax.axhline(y=prevalence, linestyle="--", color="gray", linewidth=1, label=f"Prevalence = {prevalence:.2f}")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="lower left")

    if auprc is not None:
        ax.set_title(f"Precision-Recall Curve (AUPRC = {auprc:.3f})")
    else:
        ax.set_title("Precision-Recall Curve")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_calibration(y_true, y_prob, n_groups=10, loess_frac=0.75,
                     ax=None, save_path=None):
    """
    Plot calibration diagram with grouped and LOESS-smoothed curves.

    RECOMMENDED by the paper. This is the most insightful approach
    to assess calibration, particularly with smoothing.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.get_figure()

    data = calibration_curve_data(y_true, y_prob, n_groups, loess_frac)

    # Diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Ideal")

    # LOESS smoothed curve
    ax.plot(data["smooth_pred"], data["smooth_obs"],
            color="#2563eb", linewidth=2, label="Flexible calibration (LOESS)")

    # Grouped observations
    ax.scatter(data["grouped_pred"], data["grouped_obs"],
               color="#dc2626", s=60, zorder=5, label=f"Grouped ({n_groups} groups)")

    # Rug plot of predictions
    events = np.asarray(y_prob)[np.asarray(y_true) == 1]
    non_events = np.asarray(y_prob)[np.asarray(y_true) == 0]
    ax.plot(events, np.full_like(events, -0.02), "|", color="#dc2626",
            alpha=0.3, markersize=4)
    ax.plot(non_events, np.full_like(non_events, -0.04), "|", color="#2563eb",
            alpha=0.3, markersize=4)

    ax.set_xlabel("Estimated Probability")
    ax.set_ylabel("Observed Proportion")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.06, 1.02])
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Calibration Plot")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_risk_distribution(y_true, y_prob, ax=None, save_path=None):
    """
    Plot distribution of predicted probabilities by outcome category.

    RECOMMENDED by the paper. Provides valuable insights into a model's
    behavior by showing how risk estimates are distributed for events
    and non-events.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.get_figure()

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    events = y_prob[y_true == 1]
    non_events = y_prob[y_true == 0]

    parts = ax.violinplot([non_events, events], positions=[0, 1],
                          showmeans=True, showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(["#93c5fd", "#fca5a5"][i])
        pc.set_edgecolor(["#2563eb", "#dc2626"][i])
        pc.set_alpha(0.7)

    # Overlay strip plot (subsample for readability)
    for i, (data, color) in enumerate(
        [(non_events, "#2563eb"), (events, "#dc2626")]
    ):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(data))
        sample_idx = np.random.default_rng(42).choice(
            len(data), size=min(200, len(data)), replace=False
        )
        ax.scatter(
            i + jitter[sample_idx], data[sample_idx],
            color=color, alpha=0.3, s=8, zorder=3
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-event", "Event"])
    ax.set_ylabel("Estimated Probability")
    ax.set_title("Risk Distribution by Outcome")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_decision_curve(y_true, y_prob, threshold_range=(0.01, 0.99),
                        ax=None, save_path=None):
    """
    Plot decision curve showing net benefit across thresholds.

    RECOMMENDED by the paper. Shows whether the model has better
    clinical utility than the reference strategies (treat all, treat none)
    across a range of decision thresholds.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    thresholds = np.arange(threshold_range[0], threshold_range[1], 0.01)
    data = net_benefit_curve(y_true, y_prob, thresholds)

    ax.plot(data["thresholds"], data["model_nb"],
            color="#2563eb", linewidth=2, label="Model")
    ax.plot(data["thresholds"], data["treat_all_nb"],
            color="#dc2626", linewidth=1.5, linestyle="--", label="Treat All")
    ax.axhline(y=0, color="#059669", linewidth=1.5, linestyle=":", label="Treat None")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Net Benefit")
    ax.set_xlim([threshold_range[0], threshold_range[1]])
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Decision Curve Analysis")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_expected_cost_curve(y_true, y_prob, ax=None, save_path=None):
    """
    Plot expected cost curve across normalized FN cost ratios.

    Shows expected cost for the model vs treat-all and treat-none
    strategies across a range of misclassification cost assumptions.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    data = expected_cost_curve(y_true, y_prob)

    ax.plot(data["cost_ratios"], data["model_ec"],
            color="#2563eb", linewidth=2, label="Model")
    ax.plot(data["cost_ratios"], data["treat_all_ec"],
            color="#dc2626", linewidth=1.5, linestyle="--", label="Treat All")
    ax.plot(data["cost_ratios"], data["treat_none_ec"],
            color="#059669", linewidth=1.5, linestyle=":", label="Treat None")

    ax.set_xlabel("Normalized Cost of False Negative")
    ax.set_ylabel("Expected Cost")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Expected Cost Curve")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_classification_at_thresholds(y_true, y_prob, thresholds=None,
                                       ax=None, save_path=None):
    """
    Classification plot showing sensitivity, specificity, PPV, NPV by threshold.

    Per the paper, this can be presented descriptively with either
    sensitivity+specificity or PPV+NPV pairs.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    sens_list, spec_list, ppv_list, npv_list = [], [], [], []

    for t in thresholds:
        cm = classification_metrics(y_true, y_prob, threshold=t)
        sens_list.append(cm["sensitivity"])
        spec_list.append(cm["specificity"])
        ppv_list.append(cm["ppv"])
        npv_list.append(cm["npv"])

    ax.plot(thresholds, sens_list, color="#2563eb", linewidth=1.5, label="Sensitivity")
    ax.plot(thresholds, spec_list, color="#dc2626", linewidth=1.5, label="Specificity")
    ax.plot(thresholds, ppv_list, color="#059669", linewidth=1.5, linestyle="--", label="PPV")
    ax.plot(thresholds, npv_list, color="#d97706", linewidth=1.5, linestyle="--", label="NPV")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Measure Value")
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="center right", fontsize=9)
    ax.set_title("Classification Measures by Threshold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_full_evaluation(y_true, y_prob, auroc=None, auprc=None,
                         threshold_range=(0.01, 0.99), save_path=None):
    """
    Generate the complete recommended set of evaluation plots.

    Creates a 2x3 figure with:
    1. ROC curve
    2. Calibration plot (RECOMMENDED)
    3. Decision curve (RECOMMENDED)
    4. Risk distribution (RECOMMENDED)
    5. Expected cost curve
    6. Classification measures by threshold

    Parameters
    ----------
    y_true : array-like
        Binary outcomes.
    y_prob : array-like
        Predicted probabilities.
    auroc : float, optional
        Pre-computed AUROC.
    auprc : float, optional
        Pre-computed AUPRC.
    threshold_range : tuple, default=(0.01, 0.99)
        Range for decision curve thresholds.
    save_path : str, optional
        Path to save the combined figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Medical Prediction Model Evaluation", fontsize=14, fontweight="bold")

    plot_roc_curve(y_true, y_prob, auroc=auroc, ax=axes[0, 0])
    plot_calibration(y_true, y_prob, ax=axes[0, 1])
    plot_decision_curve(y_true, y_prob, threshold_range=threshold_range, ax=axes[0, 2])
    plot_risk_distribution(y_true, y_prob, ax=axes[1, 0])
    plot_expected_cost_curve(y_true, y_prob, ax=axes[1, 1])
    plot_classification_at_thresholds(y_true, y_prob, ax=axes[1, 2])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
