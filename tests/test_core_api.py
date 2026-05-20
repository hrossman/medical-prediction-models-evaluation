import numpy as np

import medpred


def test_evaluate_returns_core_report():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.02, 0.10, 0.35, 0.45, 0.80, 0.95])

    report = medpred.evaluate(y_true, y_prob, threshold=0.2)

    assert report.meta["n"] == 6
    assert report.core["discrimination"]["auroc"] == 1.0
    assert set(report.core) == {
        "discrimination",
        "calibration",
        "overall",
        "clinical_utility",
        "classification_descriptive",
    }
    assert "f1_score" not in report.to_frame()["measure"].to_list()


def test_evaluate_all_keeps_full_inventory():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.02, 0.10, 0.35, 0.45, 0.80, 0.95])

    results = medpred.evaluate_all(y_true, y_prob, threshold=0.2)

    assert "f1_score" in results["classification"]
    assert "auprc" in results["discrimination"]
    assert results["meta"]["false_negative_cost"] == 0.8


def test_report_to_dict_does_not_expose_arrays():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.8])

    payload = medpred.evaluate(y_true, y_prob).to_dict()

    assert "_y_true" not in payload["meta"]
    assert "_y_prob" not in payload["meta"]
