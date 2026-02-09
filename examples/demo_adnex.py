#!/usr/bin/env python3
"""
Demo: Evaluating the ADNEX model using the Lancet framework.

Reproduces the case study from Van Calster et al. (2025):
External validation of the ADNEX model for predicting malignancy
in women with an ovarian tumour (894 patients, 49% prevalence).

Decision threshold: 0.1 (10% - accepting up to 9 FP per TP)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from medpred.utils.data import load_case_study_data
from medpred.evaluate import evaluate_model, evaluate_with_ci, results_to_dataframe
from medpred.metrics.calibration import logistic_recalibration
from medpred.visualization.plots import plot_full_evaluation


def main():
    print("=" * 70)
    print("ADNEX Model Evaluation - Lancet Digital Health Framework")
    print("Van Calster et al. (2025)")
    print("=" * 70)

    # Load data
    y_true, y_prob = load_case_study_data()
    print(f"\nDataset: {len(y_true)} patients")
    print(f"Events (malignant): {np.sum(y_true)} ({100*np.mean(y_true):.1f}%)")
    print(f"Non-events (benign): {np.sum(y_true == 0)} ({100*np.mean(y_true == 0):.1f}%)")

    # --- Evaluation before recalibration ---
    print("\n" + "=" * 70)
    print("BEFORE RECALIBRATION")
    print("=" * 70)

    results = evaluate_model(y_true, y_prob, threshold=0.1)

    # Print recommended measures
    print("\n--- RECOMMENDED MEASURES (Table 2) ---")
    print(f"  AUROC:                    {results['discrimination']['auroc']:.3f}")
    print(f"  Net Benefit (t=0.1):      {results['clinical_utility']['net_benefit']:.3f}")
    print(f"  Standardized NB (t=0.1):  {results['clinical_utility']['standardized_net_benefit']:.3f}")

    print("\n--- DISCRIMINATION ---")
    for k, v in results["discrimination"].items():
        print(f"  {k:25s}: {v:.3f}")

    print("\n--- CALIBRATION ---")
    for k, v in results["calibration"].items():
        print(f"  {k:25s}: {v:.3f}")

    print("\n--- OVERALL PERFORMANCE ---")
    for k, v in results["overall"].items():
        print(f"  {k:25s}: {v:.3f}")

    print(f"\n--- CLASSIFICATION (threshold = {results['classification']['threshold']}) ---")
    cm = results["classification"]
    print(f"  Confusion matrix: TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}")
    for k in ["accuracy", "balanced_accuracy", "youden_index", "kappa",
              "f1_score", "mcc", "sensitivity", "specificity", "ppv", "npv"]:
        print(f"  {k:25s}: {cm[k]:.3f}")

    print("\n--- CLINICAL UTILITY ---")
    cu = results["clinical_utility"]
    print(f"  Net benefit (t=0.1):      {cu['net_benefit']:.3f}")
    print(f"  Standardized NB:          {cu['standardized_net_benefit']:.3f}")
    print(f"  Expected cost:            {cu['expected_cost']:.3f}")
    print(f"  EC optimal threshold:     {cu['expected_cost_threshold']:.3f}")
    print(f"  Treat All NB:             {cu['treat_all_nb']:.3f}")
    print(f"  Treat None NB:            {cu['treat_none_nb']:.3f}")

    # --- Recalibrated model ---
    print("\n" + "=" * 70)
    print("AFTER LOGISTIC RECALIBRATION (Platt scaling)")
    print("=" * 70)

    y_prob_recal = logistic_recalibration(y_true, y_prob)
    results_recal = evaluate_model(y_true, y_prob_recal, threshold=0.1)

    print(f"\n  AUROC:                    {results_recal['discrimination']['auroc']:.3f} (unchanged - rank-preserving)")
    print(f"  O:E ratio:                {results_recal['calibration']['oe_ratio']:.3f}")
    print(f"  Cal. intercept:           {results_recal['calibration']['calibration_intercept']:.3f}")
    print(f"  Cal. slope:               {results_recal['calibration']['calibration_slope']:.3f}")
    print(f"  Brier score:              {results_recal['overall']['brier_score']:.3f}")
    print(f"  Net benefit (t=0.1):      {results_recal['clinical_utility']['net_benefit']:.3f}")

    # --- Full results table ---
    print("\n" + "=" * 70)
    print("COMPLETE RESULTS TABLE (matching paper Table 1)")
    print("=" * 70)

    df = results_to_dataframe(results)
    print(df.to_string(index=False))

    # --- Generate plots ---
    print("\n\nGenerating evaluation plots...")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    fig = plot_full_evaluation(
        y_true, y_prob,
        auroc=results["discrimination"]["auroc"],
        save_path=os.path.join(output_dir, "adnex_evaluation.png")
    )
    print(f"Plots saved to {output_dir}/adnex_evaluation.png")

    # --- Bootstrap CIs (reduced samples for demo speed) ---
    print("\n" + "=" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVALS (100 samples for demo)")
    print("=" * 70)

    results_ci = evaluate_with_ci(y_true, y_prob, threshold=0.1,
                                   n_bootstrap=100, random_state=42)

    for domain in ["discrimination", "calibration", "clinical_utility"]:
        print(f"\n--- {domain.upper()} ---")
        for measure, val in results_ci[domain].items():
            if isinstance(val, dict) and "point" in val:
                print(f"  {measure:25s}: {val['point']:.3f} "
                      f"(95% CI: {val['lower']:.3f} to {val['upper']:.3f})")

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
