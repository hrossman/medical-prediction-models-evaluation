# Medical Prediction Model Evaluation Toolkit

A Python toolkit implementing the comprehensive evaluation framework from **Van Calster et al. (2025)**, *"Evaluation of performance measures in predictive AI models to support medical decisions"*, The Lancet Digital Health.

Computes **32 performance measures** across **5 domains**, with visualizations, logistic recalibration, and bootstrap confidence intervals.

## Quick start

```bash
uv sync
```

```python
from medpred.evaluate import evaluate_model
from medpred.utils.data import load_case_study_data

y_true, y_prob = load_case_study_data()
results = evaluate_model(y_true, y_prob, threshold=0.1)

print(results["discrimination"]["auroc"])         # 0.928
print(results["clinical_utility"]["net_benefit"])  # net benefit at t=0.1
```

## Performance domains

| Domain | Measures | Question answered |
|---|---|---|
| **Discrimination** | AUROC, AUPRC, pAUROC | Can the model rank events above non-events? |
| **Calibration** | O:E ratio, intercept, slope, ECI, ICI, ECE | Do predicted probabilities match observed rates? |
| **Overall** | Brier, logloss, R-squared variants | Combined discrimination + calibration quality |
| **Classification** | Accuracy, F1, MCC, sensitivity, specificity, PPV, NPV, ... | Performance at a fixed decision threshold |
| **Clinical utility** | Net benefit, standardized NB, expected cost | Is the model useful for clinical decisions? |

## Paper recommendations

**Report these:**
- **AUROC** — best single discrimination measure
- **Calibration plot** — visual assessment (not a single number)
- **Net benefit / decision curve** — clinical utility across thresholds
- **Risk distributions** — predictions stratified by outcome

**Avoid these:**
- **F1 score** — the only measure that is both *improper* and has *unclear focus*
- **AUPRC** — unclear decision-analytical focus
- **Accuracy** at clinical thresholds — improper (can be optimized by a non-ideal model)

## API

### Full evaluation

```python
from medpred.evaluate import evaluate_model, evaluate_with_ci, results_to_dataframe

# All 32 metrics in one call
results = evaluate_model(y_true, y_prob, threshold=0.1)

# With bootstrap confidence intervals
results_ci = evaluate_with_ci(y_true, y_prob, threshold=0.1, n_bootstrap=1000)

# As a pandas DataFrame (with properness, focus, and recommendation annotations)
df = results_to_dataframe(results)
```

### Individual metric modules

```python
from medpred.metrics.discrimination import discrimination_metrics
from medpred.metrics.calibration import calibration_metrics, logistic_recalibration
from medpred.metrics.overall import overall_metrics
from medpred.metrics.classification import classification_metrics
from medpred.metrics.clinical_utility import clinical_utility_metrics

disc = discrimination_metrics(y_true, y_prob)           # auroc, auprc, pauroc
cal  = calibration_metrics(y_true, y_prob)               # oe_ratio, intercept, slope, eci, ici, ece
ovr  = overall_metrics(y_true, y_prob)                   # brier, logloss, r-squared variants
clf  = classification_metrics(y_true, y_prob, threshold=0.1)  # accuracy, f1, mcc, sens, spec, ...
cu   = clinical_utility_metrics(y_true, y_prob, threshold=0.1) # net_benefit, expected_cost, ...

# Logistic recalibration (Platt scaling) — rank-preserving
y_prob_recal = logistic_recalibration(y_true, y_prob)
```

### Visualizations

```python
from medpred.visualization.plots import (
    plot_roc_curve,
    plot_pr_curve,
    plot_calibration,            # recommended
    plot_risk_distribution,      # recommended
    plot_decision_curve,         # recommended
    plot_expected_cost_curve,
    plot_classification_at_thresholds,
    plot_full_evaluation,        # 2x3 panel with all plots
)

# Individual plots (all return fig, ax)
plot_calibration(y_true, y_prob)
plot_decision_curve(y_true, y_prob)
plot_risk_distribution(y_true, y_prob)

# Complete 2x3 evaluation panel
plot_full_evaluation(y_true, y_prob, auroc=disc["auroc"])
```

### Curve data for custom plots

```python
from medpred.metrics.discrimination import roc_curve_data, pr_curve_data
from medpred.metrics.calibration import calibration_curve_data
from medpred.metrics.clinical_utility import net_benefit_curve, expected_cost_curve

roc = roc_curve_data(y_true, y_prob)          # fpr, tpr, thresholds
pr  = pr_curve_data(y_true, y_prob)           # precision, recall, thresholds
cal = calibration_curve_data(y_true, y_prob)  # grouped + LOESS-smoothed
nb  = net_benefit_curve(y_true, y_prob)       # model_nb, treat_all_nb across thresholds
ec  = expected_cost_curve(y_true, y_prob)     # model_ec, treat_all_ec, treat_none_ec
```

### Bootstrap confidence intervals

```python
from medpred.utils.bootstrap import bootstrap_ci

disc_ci = bootstrap_ci(y_true, y_prob, discrimination_metrics, n_bootstrap=1000)
# disc_ci["auroc"] = {"point": 0.928, "lower": 0.910, "upper": 0.944, "bootstrap_values": [...]}
```

## Project structure

```
medpred/
  __init__.py              # Package entry — exports evaluate_model, evaluate_with_ci
  evaluate.py              # Main evaluation pipeline + results_to_dataframe
  metrics/
    discrimination.py      # AUROC, AUPRC, pAUROC
    calibration.py         # O:E ratio, intercept, slope, ECI, ICI, ECE, recalibration
    overall.py             # Brier, logloss, R-squared variants
    classification.py      # Accuracy, F1, MCC, sensitivity, specificity, etc.
    clinical_utility.py    # Net benefit, standardized NB, expected cost
  visualization/
    plots.py               # All plotting functions (7 individual + 1 panel)
  utils/
    bootstrap.py           # Percentile bootstrap confidence intervals
    data.py                # Data loading utilities
data/
  data_case_study.txt      # ADNEX case study (894 patients, 49% prevalence)
examples/
  demo_adnex.py            # Script reproducing the paper's case study
  showcase.ipynb           # Jupyter notebook showcasing all features
```

## Case study

The included dataset is from external validation of the **ADNEX model** for predicting malignancy in women with an ovarian tumour:

- **894 patients**, 49% malignant (high prevalence)
- **Decision threshold: 0.1** (10%) — accepting up to 9 false positives per true positive, appropriate for cancer screening
- AUROC ~0.93, demonstrating good discrimination but room for calibration improvement

## Dependencies

numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn

## Reference

Van Calster B, Steyerberg EW, Wynants L, van Smeden M. (2025). *Evaluation of performance measures in predictive artificial intelligence models to support medical decisions: overview and guidance.* The Lancet Digital Health.
