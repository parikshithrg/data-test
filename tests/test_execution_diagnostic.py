"""Tests for the same-bar-close diagnostic. Fewer than the production
simulator's suite (this is explicitly diagnostic-only), but the one property
that MUST hold - entry on the signal bar itself, first stop/target check on
the FOLLOWING bar, not the entry bar's own already-past range - is pinned."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from dtest import load_config
from dtest.engine.simulate import EXIT_STOP, EXIT_TARGET, EXIT_TIME, ExitRule
from dtest.research.execution_diagnostic import simulate_same_bar_close


@pytest.fixture
def cfg():
    c = load_config()
    return replace(c, execution=replace(c.execution, max_participation_pct=100.0))


def _ohlcv(rows, n, symbols):
    idx = pd.bdate_range("2020-01-06", periods=n)
    o = pd.DataFrame(100.0, index=idx, columns=symbols)
    h = pd.DataFrame(101.0, index=idx, columns=symbols)
    l = pd.DataFrame(99.0, index=idx, columns=symbols)
    c = pd.DataFrame(100.0, index=idx, columns=symbols)
    v = pd.DataFrame(1_000_000.0, index=idx, columns=symbols)
    for sym, entries in rows.items():
        for day_i, oo, hh, ll, cc in entries:
            o.at[idx[day_i], sym] = oo
            h.at[idx[day_i], sym] = hh
            l.at[idx[day_i], sym] = ll
            c.at[idx[day_i], sym] = cc
    return idx, o, h, l, c, v


def _signal(idx, symbols, hits):
    sig = pd.DataFrame(False, index=idx, columns=symbols)
    for i, s in hits:
        sig.at[idx[i], s] = True
    return sig


def test_entry_fills_at_the_signal_bars_own_close(cfg):
    idx, o, h, l, c, v = _ohlcv({"A": [(2, 100.0, 101.0, 99.0, 95.0)]}, 10, ["A"])
    sig = _signal(idx, ["A"], [(2, "A")])
    rule = ExitRule(max_hold_days=5)
    trades = simulate_same_bar_close(sig, "long", rule, open_=o, high=h, low=l,
                                     close=c, volume=v, atr_panel=None,
                                     target_value_per_trade=10_000, cfg=cfg)
    t = trades[0]
    assert t.entry_date == idx[2]      # SAME bar as the signal, not idx[3]
    assert t.entry_price == 95.0       # that bar's own CLOSE


def test_entry_bars_own_range_cannot_trigger_the_stop(cfg):
    """The headline correctness property: entering at bar T's close means bar
    T's high/low already happened BEFORE the entry. A stop level that bar T's
    own low would have "touched" must NOT fire on bar T itself."""
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({
        # Entry bar (day 2) has a very wide range that would trip a tight stop
        # if (wrongly) checked - but the entry only happens at ITS close.
        "A": [(2, 100.0, 105.0, 80.0, 100.0)],
    }, 10, symbols)
    atr_panel = pd.DataFrame(2.0, index=idx, columns=symbols)   # stop = 100 - 2*2 = 96
    sig = _signal(idx, symbols, [(2, "A")])
    rule = ExitRule(max_hold_days=5, atr_stop_multiple=2.0, risk_reward=2.5)
    trades = simulate_same_bar_close(sig, "long", rule, open_=o, high=h, low=l,
                                     close=c, volume=v, atr_panel=atr_panel,
                                     target_value_per_trade=10_000, cfg=cfg)
    t = trades[0]
    assert t.stop_price == pytest.approx(96.0)
    assert t.entry_date == idx[2]
    # entry bar's low (80) is WAY below the stop, but must not have triggered it -
    # the position wasn't held during that already-past range.
    assert t.exit_reason != EXIT_STOP


def test_stop_triggers_on_the_bar_after_entry(cfg):
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({
        "A": [(2, 100.0, 101.0, 99.0, 100.0),      # entry bar, tame range
             (3, 100.0, 101.0, 90.0, 95.0)],       # NEXT bar: stop-triggering low
    }, 10, symbols)
    atr_panel = pd.DataFrame(2.0, index=idx, columns=symbols)
    sig = _signal(idx, symbols, [(2, "A")])
    rule = ExitRule(max_hold_days=5, atr_stop_multiple=2.0, risk_reward=2.5)
    trades = simulate_same_bar_close(sig, "long", rule, open_=o, high=h, low=l,
                                     close=c, volume=v, atr_panel=atr_panel,
                                     target_value_per_trade=10_000, cfg=cfg)
    t = trades[0]
    assert t.exit_reason == EXIT_STOP
    assert t.exit_date == idx[3]
    assert t.exit_price == pytest.approx(96.0)


def test_target_triggers_on_the_bar_after_entry(cfg):
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({
        "A": [(2, 100.0, 101.0, 99.0, 100.0),
             (3, 100.0, 112.0, 99.0, 105.0)],
    }, 10, symbols)
    atr_panel = pd.DataFrame(2.0, index=idx, columns=symbols)
    sig = _signal(idx, symbols, [(2, "A")])
    rule = ExitRule(max_hold_days=6, atr_stop_multiple=2.0, risk_reward=2.5)
    trades = simulate_same_bar_close(sig, "long", rule, open_=o, high=h, low=l,
                                     close=c, volume=v, atr_panel=atr_panel,
                                     target_value_per_trade=10_000, cfg=cfg)
    t = trades[0]
    assert t.exit_reason == EXIT_TARGET
    assert t.exit_date == idx[3]
    assert t.exit_price == pytest.approx(110.0)


def test_pure_time_exit_still_fills_at_next_open(cfg):
    """Only ENTRY timing changes in this diagnostic - a time exit still needs a
    fresh decision and fills at the next available open, same as production."""
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({"A": [(6, 103.0, 104.0, 102.0, 103.5)]}, 10, symbols)
    sig = _signal(idx, symbols, [(2, "A")])
    rule = ExitRule(max_hold_days=3)
    trades = simulate_same_bar_close(sig, "long", rule, open_=o, high=h, low=l,
                                     close=c, volume=v, atr_panel=None,
                                     target_value_per_trade=10_000, cfg=cfg)
    t = trades[0]
    assert t.entry_date == idx[2]
    assert t.exit_reason == EXIT_TIME
    assert t.exit_date == idx[6]     # entry(2) + 3 exposed sessions (3,4,5) -> exit open of 6
    assert t.exit_price == pytest.approx(103.0)


def test_one_position_per_symbol_still_enforced(cfg):
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({}, 15, symbols)
    sig = _signal(idx, symbols, [(2, "A"), (4, "A")])   # 2nd falls within 1st's hold
    rule = ExitRule(max_hold_days=5)
    trades = simulate_same_bar_close(sig, "long", rule, open_=o, high=h, low=l,
                                     close=c, volume=v, atr_panel=None,
                                     target_value_per_trade=10_000, cfg=cfg)
    assert len(trades) == 1


def test_deterministic_repeated_runs(cfg):
    import numpy as np
    symbols = [f"S{i}" for i in range(5)]
    idx, o, h, l, c, v = _ohlcv({}, 60, symbols)
    rng = np.random.default_rng(0)
    sig = pd.DataFrame(rng.random((60, 5)) > 0.9, index=idx, columns=symbols)
    atr_panel = pd.DataFrame(2.0, index=idx, columns=symbols)
    rule = ExitRule(max_hold_days=7, atr_stop_multiple=2.0, risk_reward=2.5)
    t1 = simulate_same_bar_close(sig, "long", rule, open_=o, high=h, low=l, close=c,
                                 volume=v, atr_panel=atr_panel,
                                 target_value_per_trade=10_000, cfg=cfg)
    t2 = simulate_same_bar_close(sig, "long", rule, open_=o, high=h, low=l, close=c,
                                 volume=v, atr_panel=atr_panel,
                                 target_value_per_trade=10_000, cfg=cfg)
    assert [vars(x) for x in t1] == [vars(x) for x in t2]
