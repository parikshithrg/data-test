"""Two-leg pairs simulator: honest T+1 fills, real costs on both legs,
rollforward-at-entry contract selection, rollover-forced exits. Constructs
PairTrade objects directly (a plain dataclass) rather than going through
pair_trade_events, to isolate the execution/costing layer under test from
the signal-timing layer."""

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
    """LONGCO/SHORTCO cash panel, plus a single SHORTCO futures contract
    (rank 1) that lives comfortably past the whole 15-day window - the
    "nothing forces this trade out early" baseline every non-rollover test
    starts from."""
    idx = pd.bdate_range("2020-01-06", periods=15)
    cash_open = pd.DataFrame({
        "LONGCO": [100.0 + i for i in range(15)],
        "SHORTCO": [200.0 - i for i in range(15)],
    }, index=idx)
    expiry = idx[-1] + pd.Timedelta(days=30)
    fut_contracts = pd.DataFrame({
        "date": idx, "symbol": "SHORTCO", "expiry_date": expiry,
        "price": [200.3 - i for i in range(15)],
        "open_price": [200.5 - i for i in range(15)],
        "rank": 1, "days_to_expiry": [(expiry - d).days for d in idx],
    })
    return idx, cash_open, fut_contracts


def _trade(idx, entry_i, exit_i, exit_reason="reverted"):
    return PairTrade(
        symbol_a="LONGCO", symbol_b="SHORTCO", entry_date=idx[entry_i], exit_date=idx[exit_i],
        long_symbol="LONGCO", short_symbol="SHORTCO", entry_z=2.1, exit_z=0.3,
        exit_reason=exit_reason, held_days=exit_i - entry_i,
    )


def test_fills_at_next_open_on_both_legs_not_the_signal_bar(cfg):
    idx, cash_open, fut_contracts = _panels()
    trades = simulate_pair_trades(
        [_trade(idx, 2, 6)], cash_open=cash_open, fut_contracts=fut_contracts,
        calendar=idx, cfg=cfg,
    )
    t = trades[0]
    assert t.entry_fill_date == idx[3]   # T+1 after the signal's own entry_date (idx[2])
    assert t.exit_fill_date == idx[7]    # T+1 after the signal's own exit_date (idx[6])
    assert t.long_entry_price == pytest.approx(cash_open.at[idx[3], "LONGCO"])
    assert t.short_entry_price == pytest.approx(200.5 - 3)
    assert t.rolled_forward_at_entry is False   # plenty of runway, front month used as-is


def test_net_pct_reflects_real_costs_on_both_legs(cfg):
    idx, cash_open, fut_contracts = _panels()
    trades = simulate_pair_trades(
        [_trade(idx, 2, 6)], cash_open=cash_open, fut_contracts=fut_contracts,
        calendar=idx, cfg=cfg,
    )
    t = trades[0]
    long_gross = 100.0 * (t.long_exit_price / t.long_entry_price - 1.0)
    short_gross = 100.0 * -(t.short_exit_price / t.short_entry_price - 1.0)
    assert t.long_net_pct < long_gross     # costs strictly reduce a long's net
    assert t.short_net_pct < short_gross   # costs strictly reduce a short's net too
    assert t.net_pnl_pct == pytest.approx((t.long_net_pct + t.short_net_pct) / 2.0)
    assert t.long_cost_pct > 0 and t.short_cost_pct > 0


def test_rollforward_at_entry_picks_next_contract_when_front_nearly_expired(cfg):
    """The fix: a front month with only 1 day to expiry at entry (idx[3],
    expiry idx[4]) must NOT be used - the next contract (listed alongside
    it, rank 2, expiring well past the window) should be picked instead,
    and the whole trade should price off THAT contract's own levels."""
    idx, cash_open, _ = _panels()
    front_expiry = idx[4]
    front = pd.DataFrame({
        "date": idx[:5], "symbol": "SHORTCO", "expiry_date": front_expiry,
        "price": [200.3 - i for i in range(5)], "open_price": [200.5 - i for i in range(5)],
        "rank": 1, "days_to_expiry": [(front_expiry - idx[i]).days for i in range(5)],
    })
    far_expiry = idx[-1] + pd.Timedelta(days=30)
    nxt = pd.DataFrame({
        "date": idx, "symbol": "SHORTCO", "expiry_date": far_expiry,
        "price": [300.0 + i for i in range(15)], "open_price": [300.5 + i for i in range(15)],
        "rank": 2, "days_to_expiry": [(far_expiry - d).days for d in idx],
    })
    fut_contracts = pd.concat([front, nxt], ignore_index=True)

    trades = simulate_pair_trades(
        [_trade(idx, 2, 6)], cash_open=cash_open, fut_contracts=fut_contracts,
        calendar=idx, cfg=cfg, min_dte_for_entry_days=10,
    )
    t = trades[0]
    assert t.rolled_forward_at_entry is True
    assert t.short_contract_expiry == far_expiry
    assert t.short_entry_price == pytest.approx(300.5 + 3)   # the NEXT contract's own open
    assert t.exit_reason == "reverted"   # the far contract easily covers this hold


def test_no_alternative_contract_falls_back_to_front_despite_low_dte(cfg):
    """If no next contract is listed yet, rollforward-at-entry cannot pick
    one - the trade still opens in the (nearly-expired) front month, and
    still gets rollover-forced out. A stated limitation, not a crash."""
    idx, cash_open, _ = _panels()
    expiry = idx[4]
    fut_contracts = pd.DataFrame({
        "date": idx[:5], "symbol": "SHORTCO", "expiry_date": expiry,
        "price": [200.3 - i for i in range(5)], "open_price": [200.5 - i for i in range(5)],
        "rank": 1, "days_to_expiry": [(expiry - idx[i]).days for i in range(5)],
    })
    trades = simulate_pair_trades(
        [_trade(idx, 2, 6)], cash_open=cash_open, fut_contracts=fut_contracts,
        calendar=idx, cfg=cfg,   # default min_dte=10; dte at entry is 1, no rank-2 exists
    )
    t = trades[0]
    assert t.rolled_forward_at_entry is False
    assert t.short_contract_expiry == expiry
    assert t.exit_reason == EXIT_ROLLOVER


def test_rollover_forces_an_early_close_at_the_mark_not_the_new_contracts_open(cfg):
    idx, cash_open, fut_contracts = _panels()
    fut_contracts = fut_contracts[fut_contracts["date"] <= idx[4]].copy()
    fut_contracts["expiry_date"] = idx[4]
    fut_contracts["days_to_expiry"] = (idx[4] - fut_contracts["date"]).dt.days

    trades = simulate_pair_trades(
        [_trade(idx, 2, 10)],   # signal exit far past the contract's own expiry
        cash_open=cash_open, fut_contracts=fut_contracts, calendar=idx, cfg=cfg,
        min_dte_for_entry_days=0,   # isolate the forced-exit mechanic from rollforward-at-entry
    )
    t = trades[0]
    assert t.exit_reason == EXIT_ROLLOVER
    assert t.exit_fill_date == idx[4]   # the LAST day this contract still traded
    assert t.short_exit_price == pytest.approx(200.3 - 4)   # mark, not open
    assert t.short_exit_price != pytest.approx(200.5 - 4)   # i.e. not the open-price fallback


def test_no_rollover_uses_the_signal_exit_date_normally(cfg):
    idx, cash_open, fut_contracts = _panels()
    trades = simulate_pair_trades(
        [_trade(idx, 2, 6)], cash_open=cash_open, fut_contracts=fut_contracts,
        calendar=idx, cfg=cfg,
    )
    t = trades[0]
    assert t.exit_reason == "reverted"   # unchanged from the signal's own reason
    assert t.exit_fill_date == idx[7]


def test_no_fill_when_entry_date_is_the_last_calendar_day(cfg):
    idx, cash_open, fut_contracts = _panels()
    trades = simulate_pair_trades(
        [_trade(idx, len(idx) - 1, len(idx) - 1)],
        cash_open=cash_open, fut_contracts=fut_contracts, calendar=idx, cfg=cfg,
    )
    assert trades[0].exit_reason == "no_fill"
    assert trades[0].net_pnl_pct is None


def test_missing_price_data_is_a_no_fill_not_a_crash(cfg):
    idx, cash_open, fut_contracts = _panels()
    cash_open = cash_open.copy()
    cash_open.loc[idx[3], "LONGCO"] = float("nan")
    trades = simulate_pair_trades(
        [_trade(idx, 2, 6)], cash_open=cash_open, fut_contracts=fut_contracts,
        calendar=idx, cfg=cfg,
    )
    assert trades[0].exit_reason == "no_fill"
    assert trades[0].net_pnl_pct is None
