"""Data loading utilities."""

import os
import numpy as np
import pandas as pd


def load_case_study_data(filepath=None):
    """
    Load the ADNEX case study data.

    The data contains predicted probabilities and binary outcomes
    for 894 women from the TransIOTA external validation study.

    Parameters
    ----------
    filepath : str, optional
        Path to the data file. If None, looks in the default data/ directory.

    Returns
    -------
    tuple of (y_true, y_prob)
        y_true: numpy array of binary outcomes (0=benign, 1=malignant)
        y_prob: numpy array of predicted probabilities of malignancy
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        filepath = os.path.join(base_dir, "data", "data_case_study.txt")

    df = pd.read_csv(filepath, sep=r"\s+")

    y_true = df["Outcome1"].values.astype(int)
    y_prob = df["pmalwo"].values.astype(float)

    return y_true, y_prob
