"""rolling_zscore and mean_reversion_signal tests, hand-computable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtest.features.technical import rolling_zscore
from dtest.signals.mean_reversion import mean_reversion_signal


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
