"""Portfolio tests. The central property this module exists to catch: a
sizing-independent edge can still be a bad ACCOUNT (correlated drawdown), and
this is the only layer that can see that."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dtest import load_config
from dtest.engine.portfolio import (
    TRADING_DAYS_PER_YEAR, benchmark_equity_curve, portfolio_metrics, run_portfolio,
)
from dtest.engine.simulate import EXIT_STOP, EXIT_TARGET, EXIT_TIME, EXIT_UNRESOLVED, ExitRule


@pytest.fixture
def cfg():
    c = load_config()
    c = replace(c, execution=replace(c.execution, max_participation_pct=100.0))
    c = replace(c, portfolio=replace(c.portfolio, initial_capital=100_000.0,
                                    max_positions=2, max_sector_weight_pct=100.0))
    return c


def _panels(n=30, symbols=("A", "B", "C")):
    idx = pd.bdate_range("2020-01-06", periods=n)
    o = pd.DataFrame(100.0, index=idx, columns=symbols)
    h = pd.DataFrame(101.0, index=idx, columns=symbols)
    l = pd.DataFrame(99.0, index=idx, columns=symbols)
    c = pd.DataFrame(100.0, index=idx, columns=symbols)
    v = pd.DataFrame(1_000_000.0, index=idx, columns=symbols)
    return idx, o, h, l, c, v


def _signal(idx, symbols, hits):
    sig = pd.DataFrame(False, index=idx, columns=symbols)
    for i, s in hits:
        sig.at[idx[i], s] = True
    return sig


def test_equal_weight_sizing_splits_equity_across_max_positions(cfg):
    idx, o, h, l, c, v = _panels(20, ("A", "B"))
    sig = _signal(idx, ("A", "B"), [(2, "A"), (2, "B")])
    rule = ExitRule(max_hold_days=10)
    res = run_portfolio(sig, "long", rule, {}, open_=o, high=h, low=l, close=c,
                        volume=v, atr_panel=None, cfg=cfg)
    # capital 100,000 / max_positions 2 = 50,000 target each, price 100 -> 500 shares
    a_trade = res.trades[res.trades["symbol"] == "A"].iloc[0]
    b_trade = res.trades[res.trades["symbol"] == "B"].iloc[0]
    assert a_trade["shares"] == 500
    assert b_trade["shares"] == 500


def test_slot_exhaustion_skips_a_third_signal(cfg):
    """max_positions=2: a third simultaneous signal must be SKIPPED, not
    silently funded by shrinking everyone's size."""
    idx, o, h, l, c, v = _panels(20, ("A", "B", "C"))
    sig = _signal(idx, ("A", "B", "C"), [(2, "A"), (2, "B"), (2, "C")])
    rule = ExitRule(max_hold_days=10)
    res = run_portfolio(sig, "long", rule, {}, open_=o, high=h, low=l, close=c,
                        volume=v, atr_panel=None, cfg=cfg)
    assert res.skipped_no_slot == 1
    assert len(res.trades) == 2


def test_sector_cap_blocks_overconcentration(cfg):
    cfg2 = replace(cfg, portfolio=replace(cfg.portfolio, max_positions=3,
                                          max_sector_weight_pct=40.0))
    idx, o, h, l, c, v = _panels(20, ("A", "B", "C"))
    # A and B are the same sector; a 40% cap with 3 equal-weight slots (each
    # ~33%) means a SECOND same-sector position would push that sector over 40%.
    sig = _signal(idx, ("A", "B", "C"), [(2, "A"), (2, "B"), (2, "C")])
    rule = ExitRule(max_hold_days=10)
    res = run_portfolio(sig, "long", rule, {"A": "Bank", "B": "Bank", "C": "IT"},
                        open_=o, high=h, low=l, close=c, volume=v, atr_panel=None, cfg=cfg2)
    assert res.skipped_sector_cap >= 1
    # only one of A/B got in, plus C
    booked_symbols = set(res.trades["symbol"])
    assert "C" in booked_symbols
    assert len({"A", "B"} & booked_symbols) == 1


def test_correlated_crash_produces_real_drawdown_a_trade_level_view_would_miss(cfg):
    """The central motivating case: two positions that crash TOGETHER produce
    real portfolio drawdown, even though each trade's own percentage loss is
    modest and 'sizing-independent' metrics would never see the correlation."""
    idx, o, h, l, c, v = _panels(30, ("A", "B"))
    # Both crash 20% on the same day (day 10), after entering day 3.
    for sym in ("A", "B"):
        o.at[idx[10], sym] = 100.0
        h.at[idx[10], sym] = 100.0
        l.at[idx[10], sym] = 79.0
        c.at[idx[10], sym] = 80.0
    sig = _signal(idx, ("A", "B"), [(2, "A"), (2, "B")])
    rule = ExitRule(max_hold_days=15)   # pure time exit, no stop cushioning
    res = run_portfolio(sig, "long", rule, {}, open_=o, high=h, low=l, close=c,
                        volume=v, atr_panel=None, cfg=cfg)
    m = res.metrics()
    assert m["max_drawdown_pct"] < -15.0, (
        "a simultaneous ~20% crash in both open positions should show up as "
        "real portfolio drawdown"
    )


def test_time_exit_frees_a_slot_for_a_later_signal(cfg):
    idx, o, h, l, c, v = _panels(20, ("A", "B", "C"))
    rule = ExitRule(max_hold_days=3)
    # A and B fill first, filling both slots; C's signal only fires AFTER A's
    # time-exit frees a slot.
    sig = _signal(idx, ("A", "B", "C"), [(2, "A"), (2, "B"), (8, "C")])
    res = run_portfolio(sig, "long", rule, {}, open_=o, high=h, low=l, close=c,
                        volume=v, atr_panel=None, cfg=cfg)
    assert res.skipped_no_slot == 0
    assert set(res.trades["symbol"]) == {"A", "B", "C"}


def test_stop_hit_reduces_cash_correctly(cfg):
    idx, o, h, l, c, v = _panels(20, ("A",))
    l.at[idx[5], "A"] = 90.0
    atr_panel = pd.DataFrame(2.0, index=idx, columns=["A"])
    sig = _signal(idx, ("A",), [(2, "A")])
    rule = ExitRule(max_hold_days=10, atr_stop_multiple=2.0, risk_reward=2.5)
    res = run_portfolio(sig, "long", rule, {}, open_=o, high=h, low=l, close=c,
                        volume=v, atr_panel=atr_panel, cfg=cfg)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == EXIT_STOP
    assert t["net_pnl_pct"] < 0
    final_equity = res.equity_curve.iloc[-1]["equity"]
    assert final_equity < cfg.portfolio.initial_capital   # net loss after a stop + costs


def test_unresolved_position_still_appears_in_equity_curve_via_mtm(cfg):
    idx, o, h, l, c, v = _panels(10, ("A",))
    sig = _signal(idx, ("A",), [(2, "A")])
    rule = ExitRule(max_hold_days=20)   # runs off the end of available data
    res = run_portfolio(sig, "long", rule, {}, open_=o, high=h, low=l, close=c,
                        volume=v, atr_panel=None, cfg=cfg)
    assert res.trades.iloc[0]["exit_reason"] == EXIT_UNRESOLVED
    # equity curve must still be complete through the last available date
    assert len(res.equity_curve) == 10
    assert res.equity_curve.iloc[-1]["n_open"] == 1


def test_portfolio_metrics_hand_computed():
    idx = pd.bdate_range("2020-01-06", periods=TRADING_DAYS_PER_YEAR + 1)
    # Exactly doubles over one trading year -> CAGR should read ~100%.
    equity = pd.Series(np.linspace(100_000, 200_000, len(idx)), index=idx)
    m = portfolio_metrics(equity)
    assert m["cagr_pct"] == pytest.approx(100.0, abs=1.0)
    assert m["n_days"] == len(idx)


def test_portfolio_metrics_flat_curve_has_zero_drawdown_and_nan_sharpe():
    idx = pd.bdate_range("2020-01-06", periods=50)
    equity = pd.Series(100_000.0, index=idx)
    m = portfolio_metrics(equity)
    assert m["max_drawdown_pct"] == pytest.approx(0.0)
    assert np.isnan(m["sharpe"])   # zero variance -> undefined, not zero


def test_portfolio_metrics_drawdown_hand_computed():
    idx = pd.bdate_range("2020-01-06", periods=5)
    equity = pd.Series([100.0, 120.0, 90.0, 95.0, 130.0], index=idx)
    m = portfolio_metrics(equity)
    assert m["max_drawdown_pct"] == pytest.approx((90.0 / 120.0 - 1.0) * 100.0)


def test_rejects_short_direction_with_clear_reason(cfg):
    idx, o, h, l, c, v = _panels(10, ("A",))
    sig = _signal(idx, ("A",), [(2, "A")])
    rule = ExitRule(max_hold_days=5)
    with pytest.raises(ValueError, match="cash-equity delivery cannot short"):
        run_portfolio(sig, "short", rule, {}, open_=o, high=h, low=l, close=c,
                      volume=v, atr_panel=None, cfg=cfg)


def test_benchmark_equity_curve_matches_index_return():
    idx = pd.bdate_range("2020-01-06", periods=5)
    bench = pd.Series([1000.0, 1010.0, 1050.0, 1030.0, 1100.0], index=idx)
    curve = benchmark_equity_curve(bench, 50_000.0, idx)
    assert curve.iloc[0] == pytest.approx(50_000.0)
    assert curve.iloc[-1] == pytest.approx(50_000.0 * 1100.0 / 1000.0)


def test_deterministic_repeated_runs(cfg):
    idx, o, h, l, c, v = _panels(60, ("A", "B", "C"))
    rng = np.random.default_rng(0)
    sig = pd.DataFrame(rng.random((60, 3)) > 0.85, index=idx, columns=["A", "B", "C"])
    rule = ExitRule(max_hold_days=7)
    r1 = run_portfolio(sig, "long", rule, {}, open_=o, high=h, low=l, close=c,
                       volume=v, atr_panel=None, cfg=cfg)
    r2 = run_portfolio(sig, "long", rule, {}, open_=o, high=h, low=l, close=c,
                       volume=v, atr_panel=None, cfg=cfg)
    pd.testing.assert_frame_equal(r1.equity_curve, r2.equity_curve)
    pd.testing.assert_frame_equal(r1.trades, r2.trades)
