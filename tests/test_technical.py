"""ATR/true-range tests on a hand-computable fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtest.features.technical import atr, true_range


def test_true_range_hand_computed():
    idx = pd.bdate_range("2020-01-06", periods=4)
    high = pd.DataFrame({"A": [10.0, 12.0, 11.0, 15.0]}, index=idx)
    low = pd.DataFrame({"A": [9.0, 10.5, 9.5, 11.0]}, index=idx)
    close = pd.DataFrame({"A": [9.5, 11.0, 10.0, 14.0]}, index=idx)

    tr = true_range(high, low, close)
    # day0: no prior close -> NaN
    assert np.isnan(tr["A"].iloc[0])
    # day1: hl=1.5, hc=|12-9.5|=2.5, lc=|10.5-9.5|=1.0 -> max=2.5
    assert tr["A"].iloc[1] == pytest.approx(2.5)
    # day2: hl=1.5, hc=|11-11|=0, lc=|9.5-11|=1.5 -> max=1.5
    assert tr["A"].iloc[2] == pytest.approx(1.5)
    # day3: hl=4.0, hc=|15-10|=5.0, lc=|11-10|=1.0 -> max=5.0
    assert tr["A"].iloc[3] == pytest.approx(5.0)


def test_true_range_skips_a_gap_using_last_traded_close():
    idx = pd.bdate_range("2020-01-06", periods=4)
    high = pd.DataFrame({"A": [10.0, np.nan, 11.0, 15.0]}, index=idx)
    low = pd.DataFrame({"A": [9.0, np.nan, 9.5, 11.0]}, index=idx)
    close = pd.DataFrame({"A": [9.5, np.nan, 10.0, 14.0]}, index=idx)

    tr = true_range(high, low, close)
    # day2 compares against day0's close (9.5), not the untraded day1.
    assert tr["A"].iloc[2] == pytest.approx(max(1.5, abs(11 - 9.5), abs(9.5 - 9.5)))


def test_atr_is_nan_until_window_fills():
    idx = pd.bdate_range("2020-01-06", periods=20)
    rng = np.random.default_rng(0)
    close = pd.Series(100 + rng.normal(size=20).cumsum())
    high = close + 1.0
    low = close - 1.0
    close_df = pd.DataFrame({"A": close.to_numpy()}, index=idx)
    a = atr(pd.DataFrame({"A": high.to_numpy()}, index=idx),
           pd.DataFrame({"A": low.to_numpy()}, index=idx), close_df, window=14)
    # TR itself is NaN at index 0 (no prior close), so the 14th valid TR - and
    # therefore the first non-NaN ATR - lands at index 14, not 13.
    assert a["A"].iloc[:14].isna().all()
    assert a["A"].iloc[14:].notna().all()


def test_atr_deterministic():
    idx = pd.bdate_range("2020-01-06", periods=30)
    rng = np.random.default_rng(1)
    close = 100 + rng.normal(size=30).cumsum()
    high, low = close + 1.5, close - 1.5
    frames = [pd.DataFrame({"A": v}, index=idx) for v in (high, low, close)]
    a1 = atr(*frames, window=14)
    a2 = atr(*frames, window=14)
    pd.testing.assert_frame_equal(a1, a2)
