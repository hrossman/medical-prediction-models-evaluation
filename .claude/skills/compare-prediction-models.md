# Compare Medical Prediction Models

## Description
Compare two or more binary prediction models using the framework from
Van Calster et al. (2025). Assesses relative performance across discrimination,
calibration, and clinical utility domains.

## When to Use
Use this skill when you need to:
- Compare a model before and after recalibration
- Compare competing prediction models on the same validation dataset
- Evaluate whether model updates improve performance
- Assess incremental value of adding predictors

## Instructions

### Compare Two Models
```python
import numpy as np
from medpred.evaluate import evaluate_model, results_to_dataframe
from medpred.metrics.calibration import logistic_recalibration

# Model A (original)
results_a = evaluate_model(y_true, y_prob_a, threshold=0.1)

# Model B (e.g., recalibrated or competing model)
results_b = evaluate_model(y_true, y_prob_b, threshold=0.1)

# Compare key recommended measures
print("Comparison of Key Measures:")
print(f"{'Measure':<30} {'Model A':>10} {'Model B':>10}")
print("-" * 52)
print(f"{'AUROC':<30} {results_a['discrimination']['auroc']:>10.3f} {results_b['discrimination']['auroc']:>10.3f}")
print(f"{'Brier Score':<30} {results_a['overall']['brier_score']:>10.3f} {results_b['overall']['brier_score']:>10.3f}")
print(f"{'Net Benefit':<30} {results_a['clinical_utility']['net_benefit']:>10.3f} {results_b['clinical_utility']['net_benefit']:>10.3f}")
print(f"{'O:E Ratio':<30} {results_a['calibration']['oe_ratio']:>10.3f} {results_b['calibration']['oe_ratio']:>10.3f}")
```

### Compare with Decision Curves
```python
from medpred.visualization.plots import plot_decision_curve
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
# Plot both models on same decision curve
from medpred.metrics.clinical_utility import net_benefit_curve

thresholds = np.arange(0.01, 0.99, 0.01)
nb_a = net_benefit_curve(y_true, y_prob_a, thresholds)
nb_b = net_benefit_curve(y_true, y_prob_b, thresholds)

ax.plot(thresholds, nb_a['model_nb'], label='Model A', linewidth=2)
ax.plot(thresholds, nb_b['model_nb'], label='Model B', linewidth=2)
ax.plot(thresholds, nb_a['treat_all_nb'], '--', color='gray', label='Treat All')
ax.axhline(y=0, color='black', linestyle=':', label='Treat None')
ax.legend()
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Net Benefit')
ax.set_title('Decision Curve Comparison')
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
```

## Key Points for Model Comparison
- After rank-preserving recalibration, discrimination measures (AUROC) stay unchanged
- Strictly proper measures should improve after recalibration
- Improper measures may worsen after recalibration (this illustrates why properness matters)
- Compare models on the SAME external validation dataset to avoid misleading conclusions
- Clinical utility (net benefit) is the most relevant domain for comparing clinical value
