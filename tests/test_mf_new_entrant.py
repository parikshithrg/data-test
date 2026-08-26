"""dtest.signals.mf_new_entrant - a direct pass-through of an already
event-shaped panel, verified it correctly converts NaN/True/False cells
to a pure boolean signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtest.signals.mf_new_entrant import mf_new_entrant_signal


def test_true_cells_pass_through():
    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.DataFrame({"A": [True, np.nan, np.nan], "B": [np.nan, np.nan, True]}, index=dates)
    sig = mf_new_entrant_signal(data)
    assert sig.loc[dates[0], "A"] == True
    assert sig.loc[dates[2], "B"] == True
    assert sig.loc[dates[0], "B"] == False


def test_nan_cells_become_false_not_dropped():
    dates = pd.date_range("2024-01-01", periods=2)
    data = pd.DataFrame({"A": [np.nan, np.nan]}, index=dates)
    sig = mf_new_entrant_signal(data)
    assert sig.shape == (2, 1)
    assert sig["A"].sum() == 0


def test_output_is_pure_boolean_dtype():
    dates = pd.date_range("2024-01-01", periods=2)
    data = pd.DataFrame({"A": [True, np.nan]}, index=dates)
    sig = mf_new_entrant_signal(data)
    assert sig.dtypes.unique().tolist() == [np.dtype(bool)]
