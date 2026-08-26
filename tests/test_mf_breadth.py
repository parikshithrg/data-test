"""dtest.signals.mf_breadth - cross-sectional top-quantile ranking by
ownership breadth, verified against small synthetic event panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtest.signals.mf_breadth import mf_breadth_signal


def test_fires_only_on_top_quantile_by_breadth():
    dates = pd.date_range("2024-01-01", periods=1)
    data = pd.DataFrame({"A": [10], "B": [8], "C": [5], "D": [2], "E": [1]}, index=dates)
    sig = mf_breadth_signal(data, top_quantile=0.2, min_comparable=5)
    assert sig.loc[dates[0], "A"] == True   # broadest ownership
    assert sig.loc[dates[0], "E"] == False  # narrowest


def test_ties_broken_deterministically_by_symbol_name():
    dates = pd.date_range("2024-01-01", periods=1)
    data = pd.DataFrame({"Z": [5], "A": [5], "M": [5], "B": [1], "C": [1]}, index=dates)
    sig1 = mf_breadth_signal(data, top_quantile=0.2, min_comparable=5)
    sig2 = mf_breadth_signal(data, top_quantile=0.2, min_comparable=5)
    assert sig1.equals(sig2)
    # top 20% of 5 = 1 name; among the 3-way tie at value 5, "A" sorts first
    assert sig1.loc[dates[0], "A"] == True
    assert sig1.loc[dates[0], "M"] == False


def test_skips_rows_with_too_few_comparable_stocks():
    dates = pd.date_range("2024-01-01", periods=1)
    data = pd.DataFrame({"A": [10], "B": [1]}, index=dates)
    sig = mf_breadth_signal(data, top_quantile=0.5, min_comparable=5)
    assert sig.loc[dates[0]].sum() == 0


def test_all_nan_row_produces_no_firings():
    dates = pd.date_range("2024-01-01", periods=2)
    data = pd.DataFrame({"A": [10, np.nan], "B": [5, np.nan]}, index=dates)
    sig = mf_breadth_signal(data, min_comparable=1)
    assert sig.loc[dates[1]].sum() == 0


def test_output_is_pure_boolean_no_nan():
    dates = pd.date_range("2024-01-01", periods=1)
    data = pd.DataFrame({"A": [10], "B": [1]}, index=dates)
    sig = mf_breadth_signal(data, min_comparable=1)
    assert sig.dtypes.unique().tolist() == [np.dtype(bool)]
