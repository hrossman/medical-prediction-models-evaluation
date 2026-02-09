# Interpret Medical Prediction Model Performance

## Description
Guide for interpreting performance measures from the evaluation framework.
Based on Van Calster et al. (2025), The Lancet Digital Health.

## When to Use
Use this skill when you need to:
- Interpret evaluation results from the medpred toolkit
- Understand what performance values mean clinically
- Decide whether a model is suitable for clinical use
- Explain results to non-technical stakeholders

## Interpretation Guide

### Discrimination (AUROC)
- **AUROC = 0.5**: No discrimination (random guessing)
- **AUROC = 0.7-0.8**: Acceptable discrimination
- **AUROC = 0.8-0.9**: Good discrimination
- **AUROC > 0.9**: Excellent discrimination
- AUROC alone cannot determine if a model is useful for clinical practice
- Good discrimination is necessary but not sufficient

### Calibration
- **O:E ratio = 1.0**: Perfect mean calibration
  - O:E > 1: Model underestimates risk (more events observed than expected)
  - O:E < 1: Model overestimates risk (fewer events observed than expected)
- **Calibration intercept = 0**: Perfect calibration-in-the-large
  - Intercept > 0: Underestimation on average
  - Intercept < 0: Overestimation on average
- **Calibration slope = 1.0**: Adequate spread of probabilities
  - Slope < 1: Probabilities too extreme (overfitting signal in internal validation)
  - Slope > 1: Probabilities too close to prevalence (underfitting)
- **Calibration plot**: Most informative tool. Look for deviation from diagonal.

### Clinical Utility (Net Benefit)
- Compare model's net benefit to reference strategies:
  - **Model > Treat All AND Model > Treat None**: Model adds clinical value
  - **Model < Treat All or Model < Treat None**: Model is harmful at that threshold
- The decision threshold encodes misclassification cost ratio:
  - t = 0.1: False negative is 9x worse than false positive
  - t = 0.2: False negative is 4x worse than false positive
  - t = 0.5: False negative and false positive are equally bad

### What the Properness Property Means
```
Proper measure: The correct model ALWAYS has the best expected value.
                You can trust the ranking of models.

Improper measure: An incorrect model might score BETTER than the correct one.
                  Rankings cannot be trusted at clinically relevant thresholds.
```

### Decision Framework
```
1. Check discrimination (AUROC):
   - Is the model able to distinguish events from non-events?

2. Check calibration (calibration plot):
   - Are predicted probabilities accurate?
   - If not, consider recalibration.

3. Check clinical utility (decision curve):
   - Does the model improve decisions over default strategies?
   - At what range of thresholds is the model useful?

4. Risk distribution:
   - Are predicted probabilities well-separated between outcomes?
   - This provides insight into the model's behavior.
```

### Common Pitfalls to Avoid
1. Do NOT optimize threshold using Youden index (detached from clinical relevance)
2. Do NOT use F1 score for clinical validation (improper + unclear focus)
3. Do NOT conflate class imbalance with misclassification costs
4. Do NOT compare models validated on different datasets
5. Do NOT cite good discrimination alone as evidence of clinical utility
