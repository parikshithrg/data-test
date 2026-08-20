"""Value (P/E-vs-own-history) signal, hand-computable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtest.signals.value import value_signal


def test_fires_on_a_genuine_downward_cross_only():
    idx = pd.bdate_range("2020-01-06", periods=300)
    # Constant EPS (so P/E moves 1:1 with price) - small real noise pre-drop
    # (a fully flat price gives zero variance, which rolling_zscore
    # correctly reads as NaN, same convention vol_squeeze_breakout uses -
    # not a realistic "known prior state" to cross FROM), then a real,
    # sustained drop that should push the z-score below -1 exactly once.
    eps = pd.Series(10.0, index=idx)
    rng = np.random.default_rng(0)
    price = pd.Series(100.0 + rng.normal(0, 1.0, 300), index=idx)
    price.iloc[280:] = 70.0   # a real, sustained drop late in the window
    close = pd.DataFrame({"SYM": price})
    eps_panel = pd.DataFrame({"SYM": eps})

    signal = value_signal(close, eps_panel, zscore_window=252, z_threshold=1.0)
    fires = signal["SYM"][signal["SYM"]]
    assert len(fires) >= 1
    # Exactly one cross - the day the drop happens, not every day the
    # price stays low afterward (a plateau must not re-fire).
    first_fire = fires.index[0]
    assert not signal["SYM"].iloc[signal.index.get_loc(first_fire) + 1]


def test_negative_or_zero_eps_never_fires():
    idx = pd.bdate_range("2020-01-06", periods=300)
    price = pd.Series(np.linspace(100.0, 50.0, 300), index=idx)   # a real decline
    eps = pd.Series(-5.0, index=idx)   # negative TTM EPS throughout
    close = pd.DataFrame({"SYM": price})
    eps_panel = pd.DataFrame({"SYM": eps})

    signal = value_signal(close, eps_panel, zscore_window=252, z_threshold=1.0)
    assert not signal["SYM"].any()


def test_zero_eps_masked_not_divided():
    idx = pd.bdate_range("2020-01-06", periods=50)
    price = pd.Series(100.0, index=idx)
    eps = pd.Series(0.0, index=idx)
    close = pd.DataFrame({"SYM": price})
    eps_panel = pd.DataFrame({"SYM": eps})

    signal = value_signal(close, eps_panel, zscore_window=20, z_threshold=1.0)
    assert not signal["SYM"].any()


def test_flat_price_and_eps_never_fires():
    idx = pd.bdate_range("2020-01-06", periods=300)
    close = pd.DataFrame({"SYM": pd.Series(100.0, index=idx)})
    eps_panel = pd.DataFrame({"SYM": pd.Series(10.0, index=idx)})
    signal = value_signal(close, eps_panel, zscore_window=252, z_threshold=1.0)
    assert not signal["SYM"].any()
