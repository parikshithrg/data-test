"""12-1 month cross-sectional momentum signal, hand-computable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtest.signals.momentum import momentum_signal


def _panel(n_days=40, n_syms=6):
    idx = pd.bdate_range("2020-01-06", periods=n_days)
    rng = np.random.default_rng(0)
    data = {}
    for i in range(n_syms):
        drift = (i - n_syms / 2) * 0.01   # spread out trailing returns cleanly
        data[f"S{i}"] = 100.0 * np.exp(np.cumsum(drift + rng.normal(0, 0.001, n_days)))
    return pd.DataFrame(data, index=idx)


def test_momentum_signal_fires_only_on_rebalance_dates():
    close = _panel()
    membership = pd.DataFrame(True, index=close.index, columns=close.columns)
    rebalance_dates = [close.index[10], close.index[25]]
    sig = momentum_signal(close, membership, rebalance_dates, lookback_days=5, skip_days=1)
    fired_dates = set(sig.index[sig.any(axis=1)])
    assert fired_dates <= set(rebalance_dates)


def test_momentum_signal_picks_the_top_quantile_by_trailing_return():
    close = _panel()
    membership = pd.DataFrame(True, index=close.index, columns=close.columns)
    d = close.index[30]
    sig = momentum_signal(close, membership, [d], lookback_days=5, skip_days=1, top_quantile=1 / 6)
    picked = sig.columns[sig.loc[d]]
    assert len(picked) == 1
    assert picked[0] == "S5"   # S5 has the largest positive drift, i.e. the highest trailing return


def test_momentum_signal_respects_universe_eligibility():
    close = _panel()
    membership = pd.DataFrame(True, index=close.index, columns=close.columns)
    d = close.index[30]
    membership.loc[d, "S5"] = False   # exclude the actual top performer
    sig = momentum_signal(close, membership, [d], lookback_days=5, skip_days=1, top_quantile=1 / 6)
    picked = sig.columns[sig.loc[d]]
    assert "S5" not in picked
    assert picked[0] == "S4"   # the next-best ELIGIBLE name


def test_momentum_signal_ceils_the_top_quantile_count():
    close = _panel(n_syms=7)
    membership = pd.DataFrame(True, index=close.index, columns=close.columns)
    d = close.index[30]
    # top_quantile=0.2 of 7 eligible names -> ceil(1.4) = 2
    sig = momentum_signal(close, membership, [d], lookback_days=5, skip_days=1, top_quantile=0.2)
    assert int(sig.loc[d].sum()) == 2


def test_momentum_signal_drops_symbols_without_enough_history():
    close = _panel()
    membership = pd.DataFrame(True, index=close.index, columns=close.columns)
    d = close.index[5]   # too early for lookback=5 + skip=1 (needs 6 sessions of history)
    sig = momentum_signal(close, membership, [d], lookback_days=5, skip_days=1)
    assert not sig.loc[d].any()


def test_momentum_signal_deterministic():
    close = _panel()
    membership = pd.DataFrame(True, index=close.index, columns=close.columns)
    d = close.index[30]
    sig1 = momentum_signal(close, membership, [d], lookback_days=5, skip_days=1)
    sig2 = momentum_signal(close, membership, [d], lookback_days=5, skip_days=1)
    pd.testing.assert_frame_equal(sig1, sig2)
