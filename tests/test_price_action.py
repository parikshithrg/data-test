"""Close-location-within-bar, hand-computable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtest.features.price_action import close_location


def test_close_location_hand_computed():
    idx = pd.bdate_range("2020-01-06", periods=3)
    high = pd.DataFrame({"A": [110.0, 105.0, 100.0]}, index=idx)
    low = pd.DataFrame({"A": [100.0, 95.0, 90.0]}, index=idx)
    close = pd.DataFrame({"A": [109.0, 95.0, 95.0]}, index=idx)
    loc = close_location(high, low, close)
    assert loc["A"].iloc[0] == pytest.approx(0.9)   # closed near the high
    assert loc["A"].iloc[1] == pytest.approx(0.0)   # closed at the low
    assert loc["A"].iloc[2] == pytest.approx(0.5)   # closed mid-range


def test_close_location_zero_range_is_nan_not_inf():
    idx = pd.bdate_range("2020-01-06", periods=1)
    high = pd.DataFrame({"A": [100.0]}, index=idx)
    low = pd.DataFrame({"A": [100.0]}, index=idx)
    close = pd.DataFrame({"A": [100.0]}, index=idx)
    loc = close_location(high, low, close)
    assert np.isnan(loc["A"].iloc[0])


def test_close_location_bounded_zero_to_one_for_normal_bars():
    idx = pd.bdate_range("2020-01-06", periods=50)
    rng = np.random.default_rng(0)
    low = pd.Series(100 + rng.normal(0, 1, 50).cumsum())
    high = low + rng.uniform(1, 5, 50)
    close = low + rng.uniform(0, 1, 50) * (high - low)
    df_high = pd.DataFrame({"A": high.values}, index=idx)
    df_low = pd.DataFrame({"A": low.values}, index=idx)
    df_close = pd.DataFrame({"A": close.values}, index=idx)
    loc = close_location(df_high, df_low, df_close)
    assert (loc["A"] >= 0).all() and (loc["A"] <= 1).all()


def test_close_location_deterministic():
    idx = pd.bdate_range("2020-01-06", periods=20)
    rng = np.random.default_rng(0)
    low = pd.DataFrame({"A": 100 + rng.normal(0, 1, 20).cumsum()}, index=idx)
    high = low + 2.0
    close = low + 1.0
    l1 = close_location(high, low, close)
    l2 = close_location(high, low, close)
    pd.testing.assert_frame_equal(l1, l2)
