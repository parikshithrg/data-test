"""Two-leg pairs simulator: honest T+1 fills, real costs on both legs,
rollover-forced exits. Constructs PairTrade objects directly (a plain
dataclass) rather than going through pair_trade_events, to isolate the
execution/costing layer under test from the signal-timing layer."""

from __future__ import annotations

import pandas as pd
import pytest

from dtest import load_config
from dtest.engine.pairs_simulate import EXIT_ROLLOVER, simulate_pair_trades
from dtest.signals.pairs_reversion import PairTrade


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _panels():
    idx = pd.bdate_range("2020-01-06", periods=15)
    cash_open = pd.DataFrame({
        "LONGCO": [100.0 + i for i in range(15)],
        "SHORTCO": [200.0 - i for i in range(15)],
    }, index=idx)
    fut_open = pd.DataFrame({
        "SHORTCO": [200.5 - i for i in range(15)],
    }, index=idx)
    fut_mark = pd.DataFrame({
        "SHORTCO": [200.3 - i for i in range(15)],
    }, index=idx)
    fut_rollover = pd.DataFrame({
        "SHORTCO": [False] * 15,
    }, index=idx)
    return idx, cash_open, fut_open, fut_mark, fut_rollover


def _trade(idx, entry_i, exit_i, exit_reason="reverted"):
    return PairTrade(
        symbol_a="LONGCO", symbol_b="SHORTCO", entry_date=idx[entry_i], exit_date=idx[exit_i],
        long_symbol="LONGCO", short_symbol="SHORTCO", entry_z=2.1, exit_z=0.3,
        exit_reason=exit_reason, held_days=exit_i - entry_i,
    )


def test_fills_at_next_open_on_both_legs_not_the_signal_bar(cfg):
    idx, cash_open, fut_open, fut_mark, fut_rollover = _panels()
    trades = simulate_pair_trades(
        [_trade(idx, 2, 6)], cash_open=cash_open, fut_open=fut_open, fut_mark=fut_mark,
        fut_rollover=fut_rollover, calendar=idx, cfg=cfg,
    )
    t = trades[0]
    assert t.entry_fill_date == idx[3]   # T+1 after the signal's own entry_date (idx[2])
    assert t.exit_fill_date == idx[7]    # T+1 after the signal's own exit_date (idx[6])
    assert t.long_entry_price == pytest.approx(cash_open.at[idx[3], "LONGCO"])
    assert t.short_entry_price == pytest.approx(fut_open.at[idx[3], "SHORTCO"])


def test_net_pct_reflects_real_costs_on_both_legs(cfg):
    idx, cash_open, fut_open, fut_mark, fut_rollover = _panels()
    trades = simulate_pair_trades(
        [_trade(idx, 2, 6)], cash_open=cash_open, fut_open=fut_open, fut_mark=fut_mark,
        fut_rollover=fut_rollover, calendar=idx, cfg=cfg,
    )
    t = trades[0]
    long_gross = 100.0 * (t.long_exit_price / t.long_entry_price - 1.0)
    short_gross = 100.0 * -(t.short_exit_price / t.short_entry_price - 1.0)
    assert t.long_net_pct < long_gross     # costs strictly reduce a long's net
    assert t.short_net_pct < short_gross   # costs strictly reduce a short's net too
    assert t.net_pnl_pct == pytest.approx((t.long_net_pct + t.short_net_pct) / 2.0)
    assert t.long_cost_pct > 0 and t.short_cost_pct > 0


def test_rollover_forces_an_early_close_at_the_mark_not_the_new_contracts_open(cfg):
    idx, cash_open, fut_open, fut_mark, fut_rollover = _panels()
    fut_rollover = fut_rollover.copy()
    fut_rollover.loc[idx[5], "SHORTCO"] = True   # rolls on idx[5], inside the hold

    trades = simulate_pair_trades(
        [_trade(idx, 2, 10)],   # signal exit far past the roll
        cash_open=cash_open, fut_open=fut_open, fut_mark=fut_mark,
        fut_rollover=fut_rollover, calendar=idx, cfg=cfg,
    )
    t = trades[0]
    assert t.exit_reason == EXIT_ROLLOVER
    assert t.exit_fill_date == idx[4]   # the LAST day before the roll (idx[5])
    assert t.short_exit_price == pytest.approx(fut_mark.at[idx[4], "SHORTCO"])
    assert t.short_exit_price != pytest.approx(fut_open.at[idx[5], "SHORTCO"])


def test_no_rollover_uses_the_signal_exit_date_normally(cfg):
    idx, cash_open, fut_open, fut_mark, fut_rollover = _panels()
    trades = simulate_pair_trades(
        [_trade(idx, 2, 6)], cash_open=cash_open, fut_open=fut_open, fut_mark=fut_mark,
        fut_rollover=fut_rollover, calendar=idx, cfg=cfg,
    )
    t = trades[0]
    assert t.exit_reason == "reverted"   # unchanged from the signal's own reason
    assert t.exit_fill_date == idx[7]


def test_no_fill_when_entry_date_is_the_last_calendar_day(cfg):
    idx, cash_open, fut_open, fut_mark, fut_rollover = _panels()
    trades = simulate_pair_trades(
        [_trade(idx, len(idx) - 1, len(idx) - 1)],
        cash_open=cash_open, fut_open=fut_open, fut_mark=fut_mark,
        fut_rollover=fut_rollover, calendar=idx, cfg=cfg,
    )
    assert trades[0].exit_reason == "no_fill"
    assert trades[0].net_pnl_pct is None


def test_missing_price_data_is_a_no_fill_not_a_crash(cfg):
    idx, cash_open, fut_open, fut_mark, fut_rollover = _panels()
    cash_open = cash_open.copy()
    cash_open.loc[idx[3], "LONGCO"] = float("nan")
    trades = simulate_pair_trades(
        [_trade(idx, 2, 6)], cash_open=cash_open, fut_open=fut_open, fut_mark=fut_mark,
        fut_rollover=fut_rollover, calendar=idx, cfg=cfg,
    )
    assert trades[0].exit_reason == "no_fill"
    assert trades[0].net_pnl_pct is None
