# Repository Instructions

## Project

`medpred-eval` is a concise Python library for evaluating binary medical
prediction models. The import package is `medpred`.

The default user-facing workflow follows Van Calster et al. (2025):

1. Report AUROC for discrimination.
2. Inspect calibration with a smoothed calibration plot.
3. Evaluate clinical utility with net benefit and decision curve analysis.
4. Show predicted-risk distributions by outcome.

## Development

Use `uv`.

```bash
uv sync --dev
uv run pytest
uv run python examples/adnex_walkthrough.py
```

Keep the library API small. The preferred public calls are:

```python
import medpred

report = medpred.evaluate(y_true, y_prob, threshold=0.10)
full = medpred.evaluate_all(y_true, y_prob, threshold=0.10)
```

## Design Rules

- Keep the core report focused on recommended measures and paired descriptive
  threshold measures.
- Keep the complete metric inventory available through `evaluate_all`.
- Do not promote F1, accuracy, Youden index, MCC, AUPRC, or pAUROC as primary
  clinical-validation evidence.
- Treat the ADNEX data as example material, not a runtime package dataset.
- Prefer clear numpy/pandas/scikit-learn implementations over clever abstractions.
- Add focused tests for public API behavior before changing metrics.

## Agent Skill

The canonical skill lives at:

```text
skills/medical-prediction-evaluation/SKILL.md
```

Claude and Codex skill entries should be symlinks to that file, not separate
copies.
