# medpred-eval

Concise Python toolkit for validating binary medical prediction models.

The default API follows Van Calster et al. (2025), *The Lancet Digital Health*:
report AUROC, inspect calibration, assess clinical utility with net benefit,
and show risk distributions. The full metric inventory is still available, but
the main workflow keeps inadvisable threshold scores out of the primary report.

## Install

```bash
uv add medpred-eval
```

For local development:

```bash
uv sync --dev
uv run pytest
```

## Quick Use

```python
import numpy as np
import medpred

y_true = np.array([0, 0, 1, 1])
y_prob = np.array([0.05, 0.20, 0.70, 0.90])

report = medpred.evaluate(y_true, y_prob, threshold=0.10)

print(report.to_frame())
report.plot(save_path="evaluation.png")
```

## Core API

```python
report = medpred.evaluate(
    y_true,
    y_prob,
    threshold=0.10,
    n_bootstrap=None,  # set to 1000 for percentile bootstrap intervals
)
```

The returned `EvaluationReport` has:

- `report.core`: recommended and supporting core measures
- `report.meta`: sample size, event count, prevalence, threshold, cost ratio
- `report.to_frame()`: tidy table with recommendation/properness guidance
- `report.plot()`: 2x2 panel with ROC, calibration, decision curve, and risk distribution

For the complete paper inventory:

```python
full = medpred.evaluate_all(y_true, y_prob, threshold=0.10)
```

## Walkthrough

The ADNEX case-study data is included only as an example asset:

```bash
uv run python examples/adnex_walkthrough.py
```

This prints the core report, saves `adnex_core_evaluation.png`, and demonstrates
logistic recalibration.

For a fuller explanation with saved example figures, function inventory, tips,
and the main paper guidance, see [docs/walkthrough.md](docs/walkthrough.md).

## What The Default Report Includes

| Domain | Measures |
| --- | --- |
| Discrimination | AUROC |
| Calibration | O:E ratio, calibration intercept, calibration slope |
| Overall | Brier score |
| Clinical utility | Net benefit, standardized net benefit |
| Descriptive threshold measures | Sensitivity, specificity, PPV, NPV |

The default report deliberately excludes F1, accuracy, Youden index, MCC, AUPRC,
and pAUROC because they are not recommended as primary validation evidence for
clinical decision support in this framework.

## Reference

Van Calster B, Collins GS, Vickers AJ, et al. Evaluation of performance measures
in predictive artificial intelligence models to support medical decisions:
overview and guidance. *The Lancet Digital Health*. 2025.
