"""Simulator tests. Each pins one mechanic the docstring in simulate.py claims."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dtest import load_config
from dtest.engine.simulate import (
    EXIT_NO_FILL, EXIT_STOP, EXIT_TARGET, EXIT_TIME, EXIT_UNRESOLVED,
    ExitRule, simulate_trades, trades_to_frame,
)


@pytest.fixture
def cfg():
    c = load_config()
    return replace(c, execution=replace(c.execution, max_participation_pct=100.0))


def _ohlcv(rows: dict[str, list[float]], n: int, symbols: list[str]):
    """Build flat OHLCV panels from a dict of per-symbol override rows.

    Base price is 100 flat for every symbol/day; `rows["A"]` etc. supply a list
    of (day_index, open, high, low, close) override tuples applied on top.
    """
    idx = pd.bdate_range("2020-01-06", periods=n)
    o = pd.DataFrame(100.0, index=idx, columns=symbols)
    h = pd.DataFrame(101.0, index=idx, columns=symbols)
    l = pd.DataFrame(99.0, index=idx, columns=symbols)
    c = pd.DataFrame(100.0, index=idx, columns=symbols)
    v = pd.DataFrame(100_000.0, index=idx, columns=symbols)
    for sym, entries in rows.items():
        for day_i, oo, hh, ll, cc in entries:
            o.at[idx[day_i], sym] = oo
            h.at[idx[day_i], sym] = hh
            l.at[idx[day_i], sym] = ll
            c.at[idx[day_i], sym] = cc
    return idx, o, h, l, c, v


def _signal(idx, symbols, hits: list[tuple[int, str]]):
    sig = pd.DataFrame(False, index=idx, columns=symbols)
    for i, s in hits:
        sig.at[idx[i], s] = True
    return sig


def test_entry_fills_at_next_open(cfg):
    idx, o, h, l, c, v = _ohlcv({}, 10, ["A"])
    sig = _signal(idx, ["A"], [(2, "A")])
    rule = ExitRule(max_hold_days=3)
    trades = simulate_trades(sig, "long", rule, open_=o, high=h, low=l, close=c,
                             volume=v, atr_panel=None, target_value_per_trade=10_000, cfg=cfg)
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_date == idx[3]
    assert t.entry_price == 100.0


def test_stop_hit_intrabar_exits_at_stop_level(cfg):
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({"A": [(4, 100.0, 101.0, 90.0, 95.0)]}, 10, symbols)
    sig = _signal(idx, symbols, [(2, "A")])
    atr_panel = pd.DataFrame(2.0, index=idx, columns=symbols)   # ATR=2 -> stop 2*2=4 below entry
    rule = ExitRule(max_hold_days=5, atr_stop_multiple=2.0, risk_reward=2.5)
    trades = simulate_trades(sig, "long", rule, open_=o, high=h, low=l, close=c,
                             volume=v, atr_panel=atr_panel, target_value_per_trade=10_000, cfg=cfg)
    t = trades[0]
    assert t.entry_price == 100.0
    assert t.stop_price == pytest.approx(96.0)          # 100 - 2*2
    assert t.target_price == pytest.approx(110.0)        # 100 + 2.5*4
    assert t.exit_reason == EXIT_STOP
    assert t.exit_date == idx[4]
    assert t.exit_price == pytest.approx(96.0)
    assert t.gross_pnl_pct == pytest.approx(-4.0)


def test_target_hit_intrabar_exits_at_target_level(cfg):
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({"A": [(5, 100.0, 112.0, 99.0, 105.0)]}, 10, symbols)
    sig = _signal(idx, symbols, [(2, "A")])
    atr_panel = pd.DataFrame(2.0, index=idx, columns=symbols)
    rule = ExitRule(max_hold_days=6, atr_stop_multiple=2.0, risk_reward=2.5)
    trades = simulate_trades(sig, "long", rule, open_=o, high=h, low=l, close=c,
                             volume=v, atr_panel=atr_panel, target_value_per_trade=10_000, cfg=cfg)
    t = trades[0]
    assert t.exit_reason == EXIT_TARGET
    assert t.exit_date == idx[5]
    assert t.exit_price == pytest.approx(110.0)
    assert t.gross_pnl_pct == pytest.approx(10.0)


def test_stop_wins_a_same_bar_tie(cfg):
    """Both stop and target touched on the same bar: stop must win, always."""
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({"A": [(4, 100.0, 115.0, 90.0, 100.0)]}, 10, symbols)
    sig = _signal(idx, symbols, [(2, "A")])
    atr_panel = pd.DataFrame(2.0, index=idx, columns=symbols)
    rule = ExitRule(max_hold_days=5, atr_stop_multiple=2.0, risk_reward=2.5)
    trades = simulate_trades(sig, "long", rule, open_=o, high=h, low=l, close=c,
                             volume=v, atr_panel=atr_panel, target_value_per_trade=10_000, cfg=cfg)
    assert trades[0].exit_reason == EXIT_STOP


def test_pure_time_exit_after_max_hold_days_fills_at_next_open(cfg):
    """max_hold_days=3: entry day + 2 more sessions checked, exit at the 4th
    session's open if nothing else fired."""
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({"A": [(6, 103.0, 104.0, 102.0, 103.5)]}, 10, symbols)
    sig = _signal(idx, symbols, [(2, "A")])
    rule = ExitRule(max_hold_days=3)     # no stop/target - pure time exit
    trades = simulate_trades(sig, "long", rule, open_=o, high=h, low=l, close=c,
                             volume=v, atr_panel=None, target_value_per_trade=10_000, cfg=cfg)
    t = trades[0]
    assert t.entry_date == idx[3]                     # signal at 2 -> fill at 3
    assert t.exit_reason == EXIT_TIME
    # held sessions 3,4,5 (3 sessions, inclusive of entry day) -> exit at open of 6
    assert t.exit_date == idx[6]
    assert t.exit_price == pytest.approx(103.0)
    assert t.held_days == 3


def test_one_position_per_symbol_blocks_a_signal_during_an_open_hold(cfg):
    """A second signal while the first trade is still open must be skipped -
    this pins the bug found before testing: a flag set-and-cleared within one
    resolution step is a no-op, since the whole trade resolves synchronously."""
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({}, 15, symbols)
    # First signal at day 2 (fills day 3, pure time exit max_hold=5 -> exits day 8).
    # Second signal at day 5 falls WITHIN that hold and must be ignored.
    sig = _signal(idx, symbols, [(2, "A"), (5, "A")])
    rule = ExitRule(max_hold_days=5)
    trades = simulate_trades(sig, "long", rule, open_=o, high=h, low=l, close=c,
                             volume=v, atr_panel=None, target_value_per_trade=10_000, cfg=cfg)
    assert len(trades) == 1, "the day-5 signal must not spawn a second, overlapping trade"


def test_signal_after_exit_date_is_allowed_to_open_a_new_trade(cfg):
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({}, 20, symbols)
    rule = ExitRule(max_hold_days=3)
    # First trade: signal 2 -> entry 3 -> exit (time) at day 6.
    # Second signal AT day 6 (the exit date itself) must be allowed, since the
    # position is confirmed closed by that day's close either way.
    sig = _signal(idx, symbols, [(2, "A"), (6, "A")])
    trades = simulate_trades(sig, "long", rule, open_=o, high=h, low=l, close=c,
                             volume=v, atr_panel=None, target_value_per_trade=10_000, cfg=cfg)
    assert len(trades) == 2
    assert trades[0].exit_date == idx[6]
    assert trades[1].entry_date == idx[7]     # signal 6 -> fill 7


def test_signal_on_last_bar_records_a_no_fill_trade_not_a_crash(cfg):
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({}, 5, symbols)
    sig = _signal(idx, symbols, [(4, "A")])   # last bar - no next session to fill at
    rule = ExitRule(max_hold_days=3)
    trades = simulate_trades(sig, "long", rule, open_=o, high=h, low=l, close=c,
                             volume=v, atr_panel=None, target_value_per_trade=10_000, cfg=cfg)
    assert len(trades) == 1
    assert trades[0].exit_reason == EXIT_NO_FILL
    assert trades[0].entry_price is None


def test_unresolved_trade_when_data_runs_out_before_max_hold(cfg):
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({}, 6, symbols)
    sig = _signal(idx, symbols, [(2, "A")])   # fills at 3; max_hold=10 runs off the end
    rule = ExitRule(max_hold_days=10)
    trades = simulate_trades(sig, "long", rule, open_=o, high=h, low=l, close=c,
                             volume=v, atr_panel=None, target_value_per_trade=10_000, cfg=cfg)
    assert trades[0].exit_reason == EXIT_UNRESOLVED
    assert trades[0].exit_price is None
    assert trades[0].net_pnl_pct is None


def test_short_direction_flips_stop_and_target_sides(cfg):
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({"A": [(4, 100.0, 108.0, 95.0, 100.0)]}, 10, symbols)
    sig = _signal(idx, symbols, [(2, "A")])
    atr_panel = pd.DataFrame(2.0, index=idx, columns=symbols)
    rule = ExitRule(max_hold_days=5, atr_stop_multiple=2.0, risk_reward=2.5)
    trades = simulate_trades(sig, "short", rule, open_=o, high=h, low=l, close=c,
                             volume=v, atr_panel=atr_panel, target_value_per_trade=10_000, cfg=cfg)
    t = trades[0]
    # short: stop is ABOVE entry, target is BELOW entry
    assert t.stop_price == pytest.approx(104.0)
    assert t.target_price == pytest.approx(90.0)
    assert t.exit_reason == EXIT_STOP     # high=108 touches the 104 stop
    assert t.gross_pnl_pct == pytest.approx(-4.0)   # short loses when price rises


def test_costs_reduce_net_relative_to_gross(cfg):
    symbols = ["A"]
    idx, o, h, l, c, v = _ohlcv({"A": [(5, 103.0, 104.0, 102.0, 103.5)]}, 10, symbols)
    sig = _signal(idx, symbols, [(2, "A")])
    rule = ExitRule(max_hold_days=3)
    trades = simulate_trades(sig, "long", rule, open_=o, high=h, low=l, close=c,
                             volume=v, atr_panel=None, target_value_per_trade=10_000, cfg=cfg)
    t = trades[0]
    assert t.net_pnl_pct < t.gross_pnl_pct
    assert t.cost_pct > 0
    assert t.net_pnl_pct == pytest.approx(t.gross_pnl_pct - t.cost_pct, abs=1e-9)


def test_invalid_exit_rule_stop_without_target_raises():
    with pytest.raises(ValueError):
        ExitRule(max_hold_days=5, atr_stop_multiple=2.0, risk_reward=None)
    with pytest.raises(ValueError):
        ExitRule(max_hold_days=5, atr_stop_multiple=None, risk_reward=2.5)


def test_deterministic_repeated_runs(cfg):
    symbols = [f"S{i}" for i in range(5)]
    idx, o, h, l, c, v = _ohlcv({}, 60, symbols)
    rng = np.random.default_rng(0)
    sig = pd.DataFrame(rng.random((60, 5)) > 0.9, index=idx, columns=symbols)
    atr_panel = pd.DataFrame(2.0, index=idx, columns=symbols)
    rule = ExitRule(max_hold_days=7, atr_stop_multiple=2.0, risk_reward=2.5)
    t1 = trades_to_frame(simulate_trades(sig, "long", rule, open_=o, high=h, low=l,
                                          close=c, volume=v, atr_panel=atr_panel,
                                          target_value_per_trade=10_000, cfg=cfg))
    t2 = trades_to_frame(simulate_trades(sig, "long", rule, open_=o, high=h, low=l,
                                          close=c, volume=v, atr_panel=atr_panel,
                                          target_value_per_trade=10_000, cfg=cfg))
    pd.testing.assert_frame_equal(t1, t2)


def test_trades_to_frame_empty_list_has_stable_columns():
    df = trades_to_frame([])
    assert list(df.columns) == list(
        __import__("dtest.engine.simulate", fromlist=["Trade"]).Trade.__dataclass_fields__.keys()
    )
    assert len(df) == 0
