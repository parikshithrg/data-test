"""rolling_zscore and mean_reversion_signal tests, hand-computable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtest.features.technical import rolling_zscore
from dtest.signals.delivery_breakout import delivery_breakout_signal
from dtest.signals.mean_reversion import mean_reversion_signal
from dtest.signals.oi_momentum import oi_momentum_signal
from dtest.signals.participant_tilt import participant_tilt_signal
from dtest.signals.vol_squeeze_breakout import vol_squeeze_breakout_signal


def test_rolling_zscore_hand_computed():
    idx = pd.bdate_range("2020-01-06", periods=6)
    vals = [10.0, 12.0, 11.0, 13.0, 9.0, 8.0]
    x = pd.DataFrame({"A": vals}, index=idx)
    z = rolling_zscore(x, window=5, min_periods=5)
    window_vals = np.array(vals[:5])
    expected = (vals[4] - window_vals.mean()) / window_vals.std(ddof=1)
    assert z["A"].iloc[4] == pytest.approx(expected)
    assert np.isnan(z["A"].iloc[3])   # not enough history yet


def test_rolling_zscore_uses_todays_own_value_not_lookahead():
    """A z-score at date T legitimately includes T's own close - a real trader
    deciding at today's close CAN see today's price. This is not look-ahead;
    verified by construction (window includes the current row)."""
    idx = pd.bdate_range("2020-01-06", periods=10)
    x = pd.DataFrame({"A": np.arange(10, dtype=float)}, index=idx)
    z = rolling_zscore(x, window=5, min_periods=5)
    # A later value must not change an EARLIER computed z-score.
    x2 = x.copy()
    x2.iloc[9] = 999.0
    z2 = rolling_zscore(x2, window=5, min_periods=5)
    pd.testing.assert_series_equal(z["A"].iloc[:9], z2["A"].iloc[:9])


def test_rolling_zscore_zero_std_is_nan_not_inf():
    idx = pd.bdate_range("2020-01-06", periods=6)
    x = pd.DataFrame({"A": [10.0] * 6}, index=idx)   # frozen price
    z = rolling_zscore(x, window=5, min_periods=5)
    assert z["A"].iloc[4:].isna().all()


def test_mean_reversion_signal_fires_on_a_genuine_dip():
    idx = pd.bdate_range("2020-01-06", periods=55)
    # Flat at 100 for 50 days, then a sharp one-day drop to 70.
    vals = [100.0] * 50 + [70.0] + [70.0] * 4
    close = pd.DataFrame({"A": vals}, index=idx)
    sig = mean_reversion_signal(close, window=50, z_threshold=1.5)
    assert sig["A"].iloc[50]     # the drop day itself should fire


def test_mean_reversion_signal_does_not_fire_on_flat_prices():
    idx = pd.bdate_range("2020-01-06", periods=60)
    close = pd.DataFrame({"A": [100.0] * 60}, index=idx)
    sig = mean_reversion_signal(close, window=50, z_threshold=1.5)
    assert not sig["A"].any()


def test_mean_reversion_signal_does_not_fire_on_a_rally():
    idx = pd.bdate_range("2020-01-06", periods=60)
    vals = [100.0] * 50 + [130.0] * 10   # rallies, does not dip
    close = pd.DataFrame({"A": vals}, index=idx)
    sig = mean_reversion_signal(close, window=50, z_threshold=1.5)
    assert not sig["A"].any()


def test_mean_reversion_signal_respects_window_warmup():
    idx = pd.bdate_range("2020-01-06", periods=30)
    close = pd.DataFrame({"A": np.linspace(100, 50, 30)}, index=idx)
    sig = mean_reversion_signal(close, window=50, z_threshold=1.5)
    assert not sig["A"].any()   # never reaches 50 bars of history in this fixture


def test_mean_reversion_signal_deterministic():
    idx = pd.bdate_range("2020-01-06", periods=100)
    rng = np.random.default_rng(0)
    close = pd.DataFrame({"A": 100 + rng.normal(0, 1, 100).cumsum()}, index=idx)
    s1 = mean_reversion_signal(close)
    s2 = mean_reversion_signal(close)
    pd.testing.assert_frame_equal(s1, s2)


def test_delivery_breakout_fires_on_confirmed_breakout():
    idx = pd.bdate_range("2020-01-06", periods=12)
    close = pd.DataFrame({"A": [100.0] * 5 + [110.0] * 7}, index=idx)
    # Delivery spikes on the breakout day itself (index 5), flat before/after.
    deliv = pd.DataFrame({"A": [50.0] * 5 + [90.0] + [50.0] * 6}, index=idx)
    sig = delivery_breakout_signal(close, deliv, breakout_window=5,
                                   zscore_window=5, z_threshold=1.0)
    assert sig["A"].iloc[5]


def test_delivery_breakout_does_not_fire_without_delivery_confirmation():
    idx = pd.bdate_range("2020-01-06", periods=12)
    close = pd.DataFrame({"A": [100.0] * 5 + [110.0] * 7}, index=idx)
    deliv = pd.DataFrame({"A": [50.0] * 12}, index=idx)   # never elevated
    sig = delivery_breakout_signal(close, deliv, breakout_window=5,
                                   zscore_window=5, z_threshold=1.0)
    assert not sig["A"].any()


def test_delivery_breakout_does_not_fire_without_a_price_breakout():
    idx = pd.bdate_range("2020-01-06", periods=12)
    close = pd.DataFrame({"A": [100.0] * 12}, index=idx)   # flat, never breaks out
    deliv = pd.DataFrame({"A": [50.0] * 5 + [90.0] + [50.0] * 6}, index=idx)
    sig = delivery_breakout_signal(close, deliv, breakout_window=5,
                                   zscore_window=5, z_threshold=1.0)
    assert not sig["A"].any()


def test_delivery_breakout_respects_breakout_window_warmup():
    idx = pd.bdate_range("2020-01-06", periods=4)   # fewer than breakout_window=5
    close = pd.DataFrame({"A": [100.0, 100.0, 100.0, 130.0]}, index=idx)
    deliv = pd.DataFrame({"A": [50.0, 50.0, 50.0, 90.0]}, index=idx)
    sig = delivery_breakout_signal(close, deliv, breakout_window=5,
                                   zscore_window=5, z_threshold=1.0)
    assert not sig["A"].any()


def test_delivery_breakout_deterministic():
    idx = pd.bdate_range("2020-01-06", periods=100)
    rng = np.random.default_rng(0)
    close = pd.DataFrame({"A": 100 + rng.normal(0, 1, 100).cumsum()}, index=idx)
    deliv = pd.DataFrame({"A": rng.uniform(20, 80, 100)}, index=idx)
    s1 = delivery_breakout_signal(close, deliv)
    s2 = delivery_breakout_signal(close, deliv)
    pd.testing.assert_frame_equal(s1, s2)


def _oi_fixture(dte_on_breakout_day: float):
    idx = pd.bdate_range("2020-01-06", periods=12)
    close = pd.DataFrame({"A": [100.0] * 5 + [110.0] * 7}, index=idx)
    oi_chg = pd.DataFrame({"A": [10.0] * 5 + [90.0] + [10.0] * 6}, index=idx)
    dte = pd.DataFrame({"A": [dte_on_breakout_day] * 12}, index=idx)
    return close, oi_chg, dte


def test_oi_momentum_fires_inside_the_safe_expiry_band():
    close, oi_chg, dte = _oi_fixture(15)   # comfortably inside [5, 25]
    sig = oi_momentum_signal(close, oi_chg, dte, breakout_window=5,
                             zscore_window=5, z_threshold=1.0)
    assert sig["A"].iloc[5]


def test_oi_momentum_does_not_fire_too_close_to_expiry():
    close, oi_chg, dte = _oi_fixture(2)   # pre-expiry rolldown zone
    sig = oi_momentum_signal(close, oi_chg, dte, breakout_window=5,
                             zscore_window=5, z_threshold=1.0)
    assert not sig["A"].any()


def test_oi_momentum_does_not_fire_too_soon_after_rollover():
    close, oi_chg, dte = _oi_fixture(28)   # post-rollover ramp zone
    sig = oi_momentum_signal(close, oi_chg, dte, breakout_window=5,
                             zscore_window=5, z_threshold=1.0)
    assert not sig["A"].any()


def test_oi_momentum_does_not_fire_without_oi_confirmation():
    idx = pd.bdate_range("2020-01-06", periods=12)
    close = pd.DataFrame({"A": [100.0] * 5 + [110.0] * 7}, index=idx)
    oi_chg = pd.DataFrame({"A": [10.0] * 12}, index=idx)   # never elevated
    dte = pd.DataFrame({"A": [15.0] * 12}, index=idx)
    sig = oi_momentum_signal(close, oi_chg, dte, breakout_window=5,
                             zscore_window=5, z_threshold=1.0)
    assert not sig["A"].any()


def test_oi_momentum_does_not_fire_without_a_price_breakout():
    idx = pd.bdate_range("2020-01-06", periods=12)
    close = pd.DataFrame({"A": [100.0] * 12}, index=idx)   # flat
    oi_chg = pd.DataFrame({"A": [10.0] * 5 + [90.0] + [10.0] * 6}, index=idx)
    dte = pd.DataFrame({"A": [15.0] * 12}, index=idx)
    sig = oi_momentum_signal(close, oi_chg, dte, breakout_window=5,
                             zscore_window=5, z_threshold=1.0)
    assert not sig["A"].any()


def test_oi_momentum_deterministic():
    idx = pd.bdate_range("2020-01-06", periods=100)
    rng = np.random.default_rng(0)
    close = pd.DataFrame({"A": 100 + rng.normal(0, 1, 100).cumsum()}, index=idx)
    oi_chg = pd.DataFrame({"A": rng.uniform(-20, 20, 100)}, index=idx)
    dte = pd.DataFrame({"A": rng.integers(1, 30, 100).astype(float)}, index=idx)
    s1 = oi_momentum_signal(close, oi_chg, dte)
    s2 = oi_momentum_signal(close, oi_chg, dte)
    pd.testing.assert_frame_equal(s1, s2)


def _dip_fixture():
    """Same shape as test_mean_reversion_signal_fires_on_a_genuine_dip:
    flat at 100 for 50 days, then a sharp one-day drop to 70 - fires the
    base mean_reversion_signal at index 50."""
    idx = pd.bdate_range("2020-01-06", periods=55)
    vals = [100.0] * 50 + [70.0] + [70.0] * 4
    return pd.DataFrame({"A": vals}, index=idx)


def test_participant_tilt_fires_when_fii_flow_is_accumulating():
    close = _dip_fixture()
    # Monotonically rising: today's value is always above its own trailing
    # window mean, so the gate is open every day including the dip day.
    fii_net = pd.Series(1000.0 + 10.0 * np.arange(55), index=close.index)
    sig = participant_tilt_signal(close, fii_net, zscore_window=20, z_threshold=0.0)
    assert sig["A"].iloc[50]


def test_participant_tilt_does_not_fire_when_fii_flow_is_distributing():
    close = _dip_fixture()
    # Monotonically falling: today's value is always below its own trailing
    # window mean, so the gate is closed even on the dip day.
    fii_net = pd.Series(5000.0 - 10.0 * np.arange(55), index=close.index)
    sig = participant_tilt_signal(close, fii_net, zscore_window=20, z_threshold=0.0)
    assert not sig["A"].any()


def test_participant_tilt_does_not_fire_without_a_base_dip():
    idx = pd.bdate_range("2020-01-06", periods=55)
    close = pd.DataFrame({"A": [100.0] * 55}, index=idx)   # flat, never dips
    fii_net = pd.Series(1000.0 + 10.0 * np.arange(55), index=idx)   # gate open
    sig = participant_tilt_signal(close, fii_net, zscore_window=20, z_threshold=0.0)
    assert not sig["A"].any()


def test_participant_tilt_deterministic():
    idx = pd.bdate_range("2020-01-06", periods=100)
    rng = np.random.default_rng(0)
    close = pd.DataFrame({"A": 100 + rng.normal(0, 1, 100).cumsum()}, index=idx)
    fii_net = pd.Series(rng.normal(0, 1000, 100).cumsum(), index=idx)
    s1 = participant_tilt_signal(close, fii_net)
    s2 = participant_tilt_signal(close, fii_net)
    pd.testing.assert_frame_equal(s1, s2)


def _squeeze_fixture():
    """20 bars: a normal-range baseline (0-9), a genuine squeeze - high-low
    collapses to near-nothing (10-14), then a sharp true-range expansion
    on bar 15 that ALSO breaks the prior 5-day closing high. The short/long
    ATR ratio (windows 3/10) crosses up through 1.0 exactly on bar 15 -
    verified numerically before being locked in here, not hand-guessed."""
    idx = pd.bdate_range("2020-01-06", periods=20)
    close = [100.0] * 10 + [100.0] * 5 + [110.0] * 5
    high = [102.0] * 10 + [100.1] * 5 + [111.0] * 5
    low = [98.0] * 10 + [99.9] * 5 + [105.0] * 5
    high[15], low[15] = 112.0, 103.0
    return (pd.DataFrame({"A": high}, index=idx),
            pd.DataFrame({"A": low}, index=idx),
            pd.DataFrame({"A": close}, index=idx))


def test_vol_squeeze_breakout_fires_on_contraction_then_expansion():
    high, low, close = _squeeze_fixture()
    sig = vol_squeeze_breakout_signal(high, low, close, short_atr_window=3,
                                      long_atr_window=10, ratio_threshold=1.0,
                                      breakout_window=5)
    assert sig["A"].iloc[15]


def test_vol_squeeze_breakout_does_not_fire_without_a_price_breakout():
    idx = pd.bdate_range("2020-01-06", periods=20)
    close = pd.DataFrame({"A": [100.0] * 20}, index=idx)   # never breaks out
    # Same squeeze-then-expand shape in range, but close stays flat.
    high = pd.DataFrame({"A": [102.0] * 10 + [100.1] * 5 + [112.0] + [102.0] * 4}, index=idx)
    low = pd.DataFrame({"A": [98.0] * 10 + [99.9] * 5 + [103.0] + [98.0] * 4}, index=idx)
    sig = vol_squeeze_breakout_signal(high, low, close, short_atr_window=3,
                                      long_atr_window=10, ratio_threshold=1.0,
                                      breakout_window=5)
    assert not sig["A"].any()


def test_vol_squeeze_breakout_does_not_fire_without_a_prior_squeeze():
    """Volatility is already elevated well before the breakout day (never
    contracted first), so the ratio is already above threshold going in -
    no fresh cross on the breakout bar itself, even though price genuinely
    breaks out. An already-active range breaking out is not this signal's
    story."""
    idx = pd.bdate_range("2020-01-06", periods=20)
    close = pd.DataFrame({"A": [100.0] * 15 + [110.0] * 5}, index=idx)
    high = pd.DataFrame({"A": [101.0] * 10 + [105.0] * 6 + [111.0] * 4}, index=idx)
    low = pd.DataFrame({"A": [99.0] * 10 + [95.0] * 6 + [104.0] * 4}, index=idx)
    sig = vol_squeeze_breakout_signal(high, low, close, short_atr_window=3,
                                      long_atr_window=10, ratio_threshold=1.0,
                                      breakout_window=5)
    assert not sig["A"].any()


def test_vol_squeeze_breakout_respects_long_atr_window_warmup():
    high, low, close = _squeeze_fixture()   # only 20 bars
    sig = vol_squeeze_breakout_signal(high, low, close)   # default long window=50
    assert not sig["A"].any()   # never reaches 50 bars of history in this fixture


def test_vol_squeeze_breakout_zero_atr_is_not_a_free_pass():
    """A stock frozen at one price for the whole long_atr_window has
    long_atr == 0 - must come out as no-signal (NaN ratio), not a
    trivially-cleared +inf ratio."""
    idx = pd.bdate_range("2020-01-06", periods=15)
    close = pd.DataFrame({"A": [100.0] * 15}, index=idx)   # frozen, zero range
    high = pd.DataFrame({"A": [100.0] * 15}, index=idx)
    low = pd.DataFrame({"A": [100.0] * 15}, index=idx)
    sig = vol_squeeze_breakout_signal(high, low, close, short_atr_window=3,
                                      long_atr_window=10, breakout_window=5)
    assert not sig["A"].any()


def test_vol_squeeze_breakout_deterministic():
    idx = pd.bdate_range("2020-01-06", periods=100)
    rng = np.random.default_rng(0)
    close_vals = 100 + rng.normal(0, 1, 100).cumsum()
    spread = rng.uniform(0.5, 3.0, 100)
    close = pd.DataFrame({"A": close_vals}, index=idx)
    high = pd.DataFrame({"A": close_vals + spread}, index=idx)
    low = pd.DataFrame({"A": close_vals - spread}, index=idx)
    s1 = vol_squeeze_breakout_signal(high, low, close)
    s2 = vol_squeeze_breakout_signal(high, low, close)
    pd.testing.assert_frame_equal(s1, s2)
