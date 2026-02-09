# Medical Prediction Model Evaluation Toolkit

## Project Overview
Python toolkit implementing the comprehensive framework for evaluating medical
prediction models from Van Calster et al. (2025), "Evaluation of performance
measures in predictive AI models to support medical decisions", The Lancet
Digital Health.

## Architecture
```
medpred/
  __init__.py           # Package entry point
  evaluate.py           # Main evaluation pipeline (evaluate_model, evaluate_with_ci)
  metrics/
    discrimination.py   # AUROC, AUPRC, pAUROC
    calibration.py      # O:E ratio, intercept, slope, ECI, ICI, ECE, recalibration
    overall.py          # Brier, logloss, R-squared variants
    classification.py   # Accuracy, F1, MCC, sensitivity, specificity, etc.
    clinical_utility.py # Net benefit, standardized NB, expected cost
  visualization/
    plots.py            # All plotting functions (calibration, decision curve, etc.)
  utils/
    bootstrap.py        # Bootstrap confidence intervals
    data.py             # Data loading utilities
data/
  data_case_study.txt   # ADNEX case study (894 patients)
examples/
  demo_adnex.py         # Complete demo reproducing the paper's case study
```

## Key Dependencies
numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn

## Running
```bash
pip install -r requirements.txt
python examples/demo_adnex.py
```

## Paper's Core Recommendations
1. Report: AUROC, calibration plot, net benefit (decision curve), risk distributions
2. Avoid: F1 score, AUPRC, classification accuracy at clinical thresholds
3. Understand: properness (can the measure be fooled?) and focus (statistical vs decision-analytical)
