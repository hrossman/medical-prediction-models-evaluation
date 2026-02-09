"""
Medical Prediction Model Evaluation Toolkit

Based on: Van Calster et al. (2025) "Evaluation of performance measures in
predictive artificial intelligence models to support medical decisions:
overview and guidance." The Lancet Digital Health.

Implements 32 performance measures across 5 domains:
  - Discrimination (AUROC, AUPRC, pAUROC)
  - Calibration (O:E ratio, intercept, slope, ECI, ICI, ECE)
  - Overall Performance (Brier, logloss, R-squared variants, etc.)
  - Classification (accuracy, F1, MCC, sensitivity, specificity, etc.)
  - Clinical Utility (net benefit, standardized net benefit, expected cost)
"""

from medpred.evaluate import evaluate_model, evaluate_with_ci

__version__ = "0.1.0"
