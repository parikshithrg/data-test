"""Cross-asset stress composite, hand-computable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtest.features.stress import causal_percentile, cross_asset_stress_composite


def test_causal_percentile_hand_computed():
    idx = pd.bdate_range("2020-01-06", periods=5)
    s = pd.Series([10.0, 20.0, 30.0, 5.0, 25.0], index=idx)
    r = causal_percentile(s, window=3)
    assert np.isnan(r.iloc[1])   # only 2 obs exist, needs 3
    # window [10,20,30] at idx[2]: 30 is the max -> all 3 <= 30 -> 100%
    assert r.iloc[2] == pytest.approx(100.0)
    # window [20,30,5] at idx[3]: value 5 is <= only itself -> 1/3
    assert r.iloc[3] == pytest.approx(100.0 / 3.0)
    # window [30,5,25] at idx[4]: value 25, <=25 are {5,25} -> 2/3
    assert r.iloc[4] == pytest.approx(200.0 / 3.0)


def test_causal_percentile_warmup_is_nan_not_zero():
    idx = pd.bdate_range("2020-01-06", periods=10)
    s = pd.Series(np.linspace(1, 10, 10), index=idx)
    r = causal_percentile(s, window=252)
    assert r.isna().all()


def test_causal_percentile_deterministic():
    idx = pd.bdate_range("2020-01-06", periods=50)
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 50), index=idx)
    r1 = causal_percentile(s, window=10)
    r2 = causal_percentile(s, window=10)
    pd.testing.assert_series_equal(r1, r2)


def _synthetic_inputs(n=400, seed=0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    india_vix = pd.Series(15 + rng.normal(0, 2, n).cumsum() * 0.05 + 15, index=idx).abs() + 5
    us_vix = pd.Series(15 + rng.normal(0, 2, n).cumsum() * 0.05 + 15, index=idx).abs() + 5
    breadth = pd.Series(rng.uniform(20, 80, n), index=idx)
    usdinr = pd.Series(80 + rng.normal(0, 0.2, n).cumsum(), index=idx)
    dxy = pd.Series(100 + rng.normal(0, 0.3, n).cumsum(), index=idx)
    gold = pd.Series(1800 + rng.normal(0, 5, n).cumsum(), index=idx)
    return india_vix, us_vix, breadth, usdinr, dxy, gold


def test_composite_requires_all_six_dimensions_present():
    india_vix, us_vix, breadth, usdinr, dxy, gold = _synthetic_inputs()
    dims = cross_asset_stress_composite(india_vix, us_vix, breadth, usdinr, dxy, gold, window=252)
    # first 251 sessions cannot have a full 252-window percentile for ANY
    # dimension yet, so composite must be NaN there, not partially averaged
    assert dims["composite"].iloc[:251].isna().all()
    assert dims["composite"].iloc[300:].notna().any()


def test_composite_is_nan_if_even_one_dimension_is_nan_after_warmup():
    india_vix, us_vix, breadth, usdinr, dxy, gold = _synthetic_inputs()
    gold_with_gap = gold.copy()
    gold_with_gap.iloc[320] = np.nan
    dims = cross_asset_stress_composite(india_vix, us_vix, breadth, usdinr, dxy, gold_with_gap, window=252)
    # the NaN at 320 propagates through gold's own rolling window (raw=True
    # rolling.apply with a NaN in the window yields NaN), so composite must
    # be NaN for every date whose 252-window still contains that gap
    assert dims["composite"].iloc[320].__class__ is not None  # sanity: index exists
    assert pd.isna(dims.loc[dims.index[320], "composite"])


def test_breadth_is_inverted_low_breadth_means_high_stress():
    idx = pd.bdate_range("2020-01-01", periods=300)
    flat = pd.Series(15.0, index=idx)
    dxy = pd.Series(100.0, index=idx)
    gold = pd.Series(1800.0, index=idx)
    usdinr = pd.Series(80.0, index=idx)
    low_breadth = pd.Series(10.0, index=idx)   # always the minimum in its own window
    high_breadth = pd.Series(90.0, index=idx)  # always the maximum in its own window
    # inject one differing observation near the end so the rolling window has
    # genuine dispersion to rank against
    low_breadth = low_breadth.copy()
    low_breadth.iloc[-1] = 50.0
    high_breadth = high_breadth.copy()
    high_breadth.iloc[-1] = 50.0

    dims_low = cross_asset_stress_composite(flat, flat, low_breadth, usdinr, dxy, gold, window=252)
    dims_high = cross_asset_stress_composite(flat, flat, high_breadth, usdinr, dxy, gold, window=252)
    # low_breadth's own last value (50) is the MAX of its own window (mostly
    # 10s) -> breadth itself ranks high -> stress must rank LOW (inverted)
    assert dims_low["breadth_stress"].iloc[-1] < dims_high["breadth_stress"].iloc[-1]


def test_usdinr_and_gold_use_pct_change_not_level():
    idx = pd.bdate_range("2020-01-01", periods=300)
    flat = pd.Series(15.0, index=idx)
    breadth = pd.Series(50.0, index=idx)
    # USDINR sitting at a HIGH LEVEL the whole time but perfectly flat
    # (zero % change) must NOT register as stressed - only a recent MOVE
    # should. Flat level -> pct_change is 0 throughout -> mid-percentile.
    flat_high_usdinr = pd.Series(120.0, index=idx)
    dxy = pd.Series(100.0, index=idx)
    gold = pd.Series(1800.0, index=idx)
    dims = cross_asset_stress_composite(flat, flat, breadth, flat_high_usdinr, dxy, gold, window=252)
    # a flat series has zero pct_change everywhere -> its own rolling window
    # is constant -> every value ties for the max -> percentile is 100, but
    # the point of this test is that a HIGHER LEVEL alone (120 vs e.g. 80)
    # would NOT itself be read as more or less stressed - a genuinely
    # different level fixture should produce the SAME stress reading if both
    # are equally flat (zero recent change).
    flat_low_usdinr = pd.Series(80.0, index=idx)
    dims2 = cross_asset_stress_composite(flat, flat, breadth, flat_low_usdinr, dxy, gold, window=252)
    pd.testing.assert_series_equal(dims["usdinr_stress"], dims2["usdinr_stress"])
