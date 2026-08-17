"""Wide-range conviction bar signal, fixtures numerically verified against
the real function output before being locked in here."""

from __future__ import annotations

import pandas as pd

from dtest.signals.price_action import price_action_signal

IDX = pd.bdate_range("2020-01-06", periods=25)
BASE_H, BASE_L, BASE_C, BASE_V = [102.0] * 24, [98.0] * 24, [100.0] * 24, [1000.0] * 24
KW = dict(range_zscore_window=10, range_z_threshold=1.5,
         volume_zscore_window=10, volume_z_threshold=1.0,
         close_location_high=0.8, close_location_low=0.2)


def _run(high, low, close, vol):
    return price_action_signal(
        pd.DataFrame({"A": high}, index=IDX), pd.DataFrame({"A": low}, index=IDX),
        pd.DataFrame({"A": close}, index=IDX), pd.DataFrame({"A": vol}, index=IDX), **KW,
    )


def test_fires_long_on_wide_range_high_close_high_volume():
    h = BASE_H + [110.0]
    l = BASE_L + [100.0]
    c = BASE_C + [109.0]
    v = BASE_V + [3000.0]
    long_sig, short_sig = _run(h, l, c, v)
    assert long_sig["A"].iloc[-1]
    assert not short_sig["A"].any()
    assert not long_sig["A"].iloc[:-1].any()   # never fires during the calm baseline


def test_fires_short_on_wide_range_low_close_high_volume():
    h = BASE_H + [110.0]
    l = BASE_L + [100.0]
    c = BASE_C + [101.0]
    v = BASE_V + [3000.0]
    long_sig, short_sig = _run(h, l, c, v)
    assert short_sig["A"].iloc[-1]
    assert not long_sig["A"].any()


def test_does_not_fire_on_a_mid_range_close():
    """Wide range and heavy volume alone are not enough - an indecisive
    close in the middle of the bar is exactly the "volatile but tells you
    nothing" case this signal exists to exclude."""
    h = BASE_H + [110.0]
    l = BASE_L + [100.0]
    c = BASE_C + [105.0]
    v = BASE_V + [3000.0]
    long_sig, short_sig = _run(h, l, c, v)
    assert not long_sig["A"].any()
    assert not short_sig["A"].any()


def test_does_not_fire_without_a_wide_range():
    """An extreme close on heavy volume inside an ORDINARY range is not
    a conviction bar - the range itself has to be the outlier too."""
    h = BASE_H + [102.0]
    l = BASE_L + [98.0]
    c = BASE_C + [101.6]
    v = BASE_V + [3000.0]
    long_sig, _ = _run(h, l, c, v)
    assert not long_sig["A"].any()


def test_does_not_fire_without_volume_confirmation():
    h = BASE_H + [110.0]
    l = BASE_L + [100.0]
    c = BASE_C + [109.0]
    v = BASE_V + [1000.0]
    long_sig, _ = _run(h, l, c, v)
    assert not long_sig["A"].any()


def test_respects_window_warmup():
    idx = pd.bdate_range("2020-01-06", periods=5)
    high = pd.DataFrame({"A": [110.0] * 5}, index=idx)
    low = pd.DataFrame({"A": [100.0] * 5}, index=idx)
    close = pd.DataFrame({"A": [109.0] * 5}, index=idx)
    vol = pd.DataFrame({"A": [3000.0] * 5}, index=idx)
    long_sig, short_sig = price_action_signal(high, low, close, vol, **KW)
    assert not long_sig["A"].any()   # fewer than the 10-session zscore window
    assert not short_sig["A"].any()


def test_deterministic():
    h = BASE_H + [110.0]
    l = BASE_L + [100.0]
    c = BASE_C + [109.0]
    v = BASE_V + [3000.0]
    l1, s1 = _run(h, l, c, v)
    l2, s2 = _run(h, l, c, v)
    pd.testing.assert_frame_equal(l1, l2)
    pd.testing.assert_frame_equal(s1, s2)
