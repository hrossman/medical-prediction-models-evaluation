# Evaluate Medical Prediction Model

## Description
Evaluate a binary prediction model using the comprehensive framework from
Van Calster et al. (2025), The Lancet Digital Health. Computes 32 performance
measures across 5 domains and generates recommended visualizations.

## When to Use
Use this skill when you need to:
- Evaluate a medical prediction model's performance
- Compute discrimination, calibration, and clinical utility metrics
- Generate calibration plots, decision curves, and risk distribution plots
- Compare model performance before and after recalibration
- Assess whether a prediction model is suitable for clinical use

## Instructions

### Quick Evaluation (Point Estimates)
```python
import numpy as np
from medpred.evaluate import evaluate_model, results_to_dataframe

# y_true: binary outcomes (0 or 1)
# y_prob: predicted probabilities
results = evaluate_model(y_true, y_prob, threshold=0.1)
df = results_to_dataframe(results)
print(df.to_string(index=False))
```

### Evaluation with Bootstrap Confidence Intervals
```python
from medpred.evaluate import evaluate_with_ci, results_to_dataframe

results = evaluate_with_ci(y_true, y_prob, threshold=0.1, n_bootstrap=1000)
df = results_to_dataframe(results, with_ci=True)
print(df.to_string(index=False))
```

### Generate Recommended Plots
```python
from medpred.visualization import plot_full_evaluation

fig = plot_full_evaluation(y_true, y_prob, save_path="evaluation_plots.png")
```

### Individual Plots
```python
from medpred.visualization import (
    plot_calibration,        # RECOMMENDED - most insightful for calibration
    plot_decision_curve,     # RECOMMENDED - clinical utility assessment
    plot_risk_distribution,  # RECOMMENDED - model behavior insight
    plot_roc_curve,          # Acceptable but limited added value
)

plot_calibration(y_true, y_prob, save_path="calibration.png")
plot_decision_curve(y_true, y_prob, save_path="decision_curve.png")
plot_risk_distribution(y_true, y_prob, save_path="risk_distribution.png")
```

### Logistic Recalibration
```python
from medpred.metrics.calibration import logistic_recalibration

y_prob_recal = logistic_recalibration(y_true, y_prob)
results_recal = evaluate_model(y_true, y_prob_recal, threshold=0.1)
```

### Key Parameters
- `threshold`: Decision threshold (default 0.1 = accepting up to 9 FP per TP)
- `cost_fn_ratio`: Normalized FN cost for expected cost (default 0.9)
- `n_bootstrap`: Number of bootstrap samples for CIs (default 1000)
- `min_sensitivity`: Minimum sensitivity for partial AUROC (default 0.8)

## Paper Recommendations (Table 2)

**Essential to report:**
1. AUROC (discrimination)
2. Calibration plot with smoothing (calibration)
3. Net benefit with decision curve analysis (clinical utility)
4. Risk distribution by outcome category

**Inadvisable measures:**
- F1 score (only measure that is both improper AND lacks clear focus)
- AUPRC, pAUROC (mix statistical and decision-analytical performance)
- Classification accuracy, balanced accuracy, Youden index, kappa, MCC
- Discrimination slope, MAPE

**Key concepts:**
- Proper measures cannot be fooled: correct model always wins in expectation
- Clinical utility is the most important domain for clinical practice
- Class imbalance should not be conflated with misclassification costs
