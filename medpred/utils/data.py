"""Small data helpers for user-supplied validation tables."""

from __future__ import annotations

import pandas as pd


def load_binary_table(
    path: str,
    *,
    outcome: str,
    probability: str,
    sep: str | None = None,
):
    """Load outcome and predicted-probability columns from a table.

    Parameters
    ----------
    path:
        CSV, TSV, or whitespace-delimited table.
    outcome:
        Name of the binary outcome column.
    probability:
        Name of the predicted event probability column.
    sep:
        Optional separator passed to :func:`pandas.read_csv`. If omitted,
        pandas uses its Python engine to infer common delimited text formats.
    """
    if sep is None:
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        df = pd.read_csv(path, sep=sep)
    return df[outcome].astype(int).to_numpy(), df[probability].astype(float).to_numpy()
