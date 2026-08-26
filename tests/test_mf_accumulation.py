"""dtest.signals.mf_accumulation - cross-sectional top-percentile
threshold, verified against small synthetic event panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtest.signals.mf_accumulation import mf_accumulation_signal


def test_fires_only_on_stocks_at_or_above_the_cross_sectional_percentile():
    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.DataFrame(
        {"A": [0.50, np.nan, np.nan], "B": [0.10, np.nan, np.nan],
         "C": [0.05, np.nan, np.nan], "D": [0.01, np.nan, np.nan],
         "E": [0.20, np.nan, np.nan]},
        index=dates,
    )
    sig = mf_accumulation_signal(data, top_percentile=80.0, min_comparable=5)
    day0 = sig.loc[dates[0]]
    assert day0["A"] == True  # highest value, clears the 80th percentile
    assert day0["D"] == False  # lowest value
    assert sig.loc[dates[1]].sum() == 0  # all-NaN row -> no firings


def test_skips_rows_with_too_few_comparable_stocks():
    dates = pd.date_range("2024-01-01", periods=1)
    data = pd.DataFrame({"A": [0.90], "B": [0.10]}, index=dates)
    sig = mf_accumulation_signal(data, top_percentile=50.0, min_comparable=5)
    assert sig.loc[dates[0]].sum() == 0


def test_recomputes_threshold_fresh_each_disclosed_month():
    dates = pd.date_range("2024-01-01", periods=2)
    data = pd.DataFrame(
        {"A": [0.50, 0.01], "B": [0.10, 0.02], "C": [0.05, 0.03],
         "D": [0.01, 0.04], "E": [0.20, 0.90]},
        index=dates,
    )
    sig = mf_accumulation_signal(data, top_percentile=80.0, min_comparable=5)
    assert sig.loc[dates[0], "A"] == True
    assert sig.loc[dates[0], "E"] == False
    assert sig.loc[dates[1], "E"] == True   # E is now the top performer
    assert sig.loc[dates[1], "A"] == False


def test_output_is_pure_boolean_no_nan():
    dates = pd.date_range("2024-01-01", periods=2)
    data = pd.DataFrame({"A": [0.5, np.nan], "B": [0.1, np.nan]}, index=dates)
    sig = mf_accumulation_signal(data, min_comparable=1)
    assert sig.dtypes.unique().tolist() == [np.dtype(bool)]
