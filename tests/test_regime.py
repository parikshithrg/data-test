"""Causal trailing-return regime feature, hand-computable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtest.features.regime import trailing_return


def test_trailing_return_hand_computed():
    idx = pd.bdate_range("2020-01-06", periods=5)
    price = pd.Series([100.0, 105.0, 110.0, 90.0, 95.0], index=idx)
    r = trailing_return(price, lookback=3)
    assert np.isnan(r.iloc[2])   # only 2 prior sessions exist, needs 3
    assert r.iloc[3] == pytest.approx(90.0 / 100.0 - 1.0)
    assert r.iloc[4] == pytest.approx(95.0 / 105.0 - 1.0)


def test_trailing_return_warmup_is_nan_not_zero():
    idx = pd.bdate_range("2020-01-06", periods=10)
    price = pd.Series(np.linspace(100, 110, 10), index=idx)
    r = trailing_return(price, lookback=63)
    assert r.isna().all()   # fewer than 63 sessions exist anywhere in this fixture


def test_trailing_return_sign_matches_direction():
    idx = pd.bdate_range("2020-01-06", periods=10)
    rising = pd.Series(np.linspace(100, 150, 10), index=idx)
    falling = pd.Series(np.linspace(150, 100, 10), index=idx)
    r_up = trailing_return(rising, lookback=5)
    r_down = trailing_return(falling, lookback=5)
    assert (r_up.dropna() > 0).all()
    assert (r_down.dropna() < 0).all()


def test_trailing_return_deterministic():
    idx = pd.bdate_range("2020-01-06", periods=100)
    rng = np.random.default_rng(0)
    price = pd.Series(100 + rng.normal(0, 1, 100).cumsum(), index=idx)
    r1 = trailing_return(price)
    r2 = trailing_return(price)
    pd.testing.assert_series_equal(r1, r2)


def test_trailing_return_skip_zero_is_unchanged_default_behaviour():
    idx = pd.bdate_range("2020-01-06", periods=5)
    price = pd.Series([100.0, 105.0, 110.0, 90.0, 95.0], index=idx)
    assert trailing_return(price, lookback=3, skip=0).equals(trailing_return(price, lookback=3))


def test_trailing_return_skip_excludes_the_most_recent_window():
    idx = pd.bdate_range("2020-01-06", periods=6)
    price = pd.Series([100.0, 110.0, 120.0, 130.0, 10.0, 200.0], index=idx)
    # skip=2, lookback=2: at idx[5], compares price 2 sessions ago (idx[3]=130)
    # against price 4 sessions ago (idx[1]=110) - the wild idx[4] value (10.0)
    # must NOT enter the calculation at all.
    r = trailing_return(price, lookback=2, skip=2)
    assert r.iloc[5] == pytest.approx(130.0 / 110.0 - 1.0)


def test_trailing_return_works_on_a_dataframe_not_just_a_series():
    idx = pd.bdate_range("2020-01-06", periods=5)
    prices = pd.DataFrame({
        "A": [100.0, 105.0, 110.0, 90.0, 95.0],
        "B": [50.0, 50.0, 55.0, 55.0, 60.5],
    }, index=idx)
    r = trailing_return(prices, lookback=3)
    assert r.loc[idx[3], "A"] == pytest.approx(90.0 / 100.0 - 1.0)
    assert r.loc[idx[4], "B"] == pytest.approx(60.5 / 50.0 - 1.0)
