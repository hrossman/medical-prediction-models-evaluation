# Medical Prediction Evaluation

## Purpose

Use this skill when evaluating, interpreting, plotting, or comparing binary
medical prediction models with the `medpred-eval` package.

The package is intentionally opinionated: lead with the core validation report,
then reach for the full metric inventory only when the user explicitly needs it.

## Core Workflow

```python
import medpred

report = medpred.evaluate(y_true, y_prob, threshold=0.10)
print(report.to_frame().round(3))
report.plot(save_path="evaluation.png")
```

The core report contains:

- AUROC
- O:E ratio, calibration intercept, calibration slope
- Brier score
- Net benefit and standardized net benefit
- Sensitivity/specificity and PPV/NPV as paired descriptive threshold measures

## Full Inventory

```python
full = medpred.evaluate_all(y_true, y_prob, threshold=0.10)
```

Use this when a user asks for all paper metrics, compatibility with older code,
or a detailed audit. Do not treat the full inventory as the default report.

## Bootstrap Intervals

```python
report = medpred.evaluate(
    y_true,
    y_prob,
    threshold=0.10,
    n_bootstrap=1000,
)
```

Bootstrap intervals are appropriate for statistical performance measures.
Avoid over-emphasizing uncertainty intervals for clinical utility measures;
the paper notes that this remains debated in decision analysis.

## Recalibration

```python
recalibrated = medpred.logistic_recalibration(y_true, y_prob)
recalibrated_report = medpred.evaluate(y_true, recalibrated, threshold=0.10)
```

Logistic recalibration is rank preserving. AUROC should stay unchanged, while
calibration and strictly proper scores may improve.

## Interpretation Guardrails

- AUROC is useful for discrimination, but it does not establish clinical utility.
- Calibration plots are more informative than calibration summary metrics alone.
- Decision thresholds should be clinically justified, not optimized by Youden
  index or another statistical threshold rule.
- Net benefit should be compared against treat-all and treat-none strategies
  over a clinically plausible threshold range.
- F1 is inadvisable for this use case because it is improper and lacks clear
  statistical or decision-analytical focus.
- AUPRC and pAUROC are available in the full inventory but should not be framed
  as preferred medical-decision metrics.

## Example Data

The ADNEX case study is example-only:

```bash
uv run python examples/adnex_walkthrough.py
```
