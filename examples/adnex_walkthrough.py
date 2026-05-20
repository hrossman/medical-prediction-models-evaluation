"""Walkthrough: evaluate ADNEX ovarian tumour malignancy predictions."""

from pathlib import Path

import pandas as pd

import medpred


DATA = Path(__file__).parent / "data" / "adnex_case_study.txt"


def load_adnex():
    """Return outcomes and predicted probabilities from the example data."""
    df = pd.read_csv(DATA, sep=r"\s+")
    return df["Outcome1"].to_numpy(dtype=int), df["pmalwo"].to_numpy(dtype=float)


def main():
    y_true, y_prob = load_adnex()

    report = medpred.evaluate(y_true, y_prob, threshold=0.10)

    print("Core validation report")
    print(report.to_frame().round(3).to_string(index=False))

    print("\nWhy threshold=0.10?")
    print("It encodes a false-negative cost about 9x the false-positive cost.")

    report.plot(save_path="adnex_core_evaluation.png")
    print("\nSaved adnex_core_evaluation.png")

    recalibrated = medpred.logistic_recalibration(y_true, y_prob)
    recalibrated_report = medpred.evaluate(y_true, recalibrated, threshold=0.10)

    print("\nAfter logistic recalibration")
    print(
        recalibrated_report.to_frame()
        .loc[:, ["domain", "measure", "value"]]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
