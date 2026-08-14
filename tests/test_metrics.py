"""Metrics tests on hand-built trade frames - no simulator dependency needed
for these, since the metrics operate on the trade frame's own columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtest.engine.simulate import EXIT_NO_FILL, EXIT_STOP, EXIT_TARGET, EXIT_TIME, EXIT_UNRESOLVED
from dtest.evaluate.metrics import benchmark_excess, capital_day_edge, non_overlapping_tstat, summary_stats


def _trade_row(**kw):
    base = dict(symbol="A", signal_date=pd.Timestamp("2020-01-01"),
               entry_date=pd.Timestamp("2020-01-02"), entry_price=100.0,
               stop_price=95.0, target_price=110.0,
               exit_date=pd.Timestamp("2020-01-05"), exit_price=105.0,
               exit_reason=EXIT_TARGET, held_days=3, shares=10,
               gross_pnl_pct=5.0, net_pnl_pct=4.7, cost_pct=0.3)
    base.update(kw)
    return base


def test_summary_stats_hit_rate_excludes_time_exits():
    rows = [
        _trade_row(exit_reason=EXIT_TARGET, net_pnl_pct=5.0),
        _trade_row(exit_reason=EXIT_TARGET, net_pnl_pct=5.0),
        _trade_row(exit_reason=EXIT_STOP, net_pnl_pct=-4.0),
        _trade_row(exit_reason=EXIT_TIME, net_pnl_pct=1.0),
        _trade_row(exit_reason=EXIT_TIME, net_pnl_pct=1.0),
        _trade_row(exit_reason=EXIT_TIME, net_pnl_pct=1.0),
    ]
    df = pd.DataFrame(rows)
    s = summary_stats(df)
    # hit rate = target / (target + stop) = 2 / 3, TIME exits excluded entirely
    assert s.hit_rate_pct == pytest.approx(200.0 / 3.0)
    assert s.n_resolved == 6
    assert s.expiry_rate_pct == pytest.approx(50.0)   # 3 of 6 were time exits


def test_summary_stats_win_rate_includes_time_exits():
    rows = [
        _trade_row(exit_reason=EXIT_TARGET, net_pnl_pct=5.0),
        _trade_row(exit_reason=EXIT_STOP, net_pnl_pct=-4.0),
        _trade_row(exit_reason=EXIT_TIME, net_pnl_pct=-0.5),
    ]
    df = pd.DataFrame(rows)
    s = summary_stats(df)
    # win_rate is over ALL resolved trades, unlike hit_rate
    assert s.win_rate_pct == pytest.approx(100.0 / 3.0)


def test_summary_stats_no_stop_target_gives_none_hit_rate():
    rows = [_trade_row(exit_reason=EXIT_TIME, stop_price=None, target_price=None,
                       net_pnl_pct=1.0) for _ in range(3)]
    df = pd.DataFrame(rows)
    s = summary_stats(df)
    assert s.hit_rate_pct is None


def test_summary_stats_counts_no_fill_and_unresolved_separately():
    rows = [
        _trade_row(exit_reason=EXIT_NO_FILL, entry_price=None, net_pnl_pct=None,
                  gross_pnl_pct=None, cost_pct=None, exit_date=None, exit_price=None),
        _trade_row(exit_reason=EXIT_UNRESOLVED, exit_date=None, exit_price=None,
                  net_pnl_pct=None, gross_pnl_pct=None, cost_pct=None),
        _trade_row(exit_reason=EXIT_TARGET, net_pnl_pct=5.0),
    ]
    df = pd.DataFrame(rows)
    s = summary_stats(df)
    assert s.n_signals == 3
    assert s.n_no_fill == 1
    assert s.n_unresolved == 1
    assert s.n_resolved == 1     # only the filled-and-exited trade


def test_summary_stats_empty_frame_does_not_crash():
    df = pd.DataFrame(columns=["exit_reason", "net_pnl_pct", "gross_pnl_pct",
                               "held_days", "cost_pct"])
    s = summary_stats(df)
    assert s.n_signals == 0
    assert s.n_resolved == 0
    assert np.isnan(s.mean_net_pct)


def test_benchmark_excess_hand_computed():
    df = pd.DataFrame([_trade_row(
        entry_date=pd.Timestamp("2020-01-02"), exit_date=pd.Timestamp("2020-01-06"),
        net_pnl_pct=5.0,
    )])
    bench = pd.Series(
        {pd.Timestamp("2020-01-02"): 1000.0, pd.Timestamp("2020-01-06"): 1030.0},
    )
    excess = benchmark_excess(df, bench)
    # benchmark returned +3.0% over the same window; trade made 5.0% net -> excess +2.0
    assert excess.iloc[0] == pytest.approx(2.0)


def test_non_overlapping_tstat_reduces_n_to_bucket_count():
    """10 trades all in the SAME week must collapse to n_buckets=1, not n=10 -
    a single bucket can't produce a t-stat (no variance across buckets), which
    is exactly the point: ten same-week trades are not ten independent draws."""
    rows = [_trade_row(entry_date=pd.Timestamp("2020-01-06") + pd.Timedelta(days=i % 4),
                       net_pnl_pct=float(i)) for i in range(10)]
    df = pd.DataFrame(rows)
    result = non_overlapping_tstat(df, freq="W")
    assert result["n_trades"] == 10
    assert result["n_buckets"] == 1
    assert np.isnan(result["t_stat"])   # can't compute variance from one bucket


def test_non_overlapping_tstat_across_distinct_weeks():
    # Small variation around 2.0% - a perfectly constant series triggers scipy's
    # catastrophic-cancellation warning for a (correctly) near-infinite t-stat.
    values = [1.8, 2.1, 1.9, 2.2, 2.0]
    rows = [_trade_row(entry_date=pd.Timestamp("2020-01-06") + pd.Timedelta(weeks=i),
                       net_pnl_pct=v) for i, v in enumerate(values)]
    df = pd.DataFrame(rows)
    result = non_overlapping_tstat(df, freq="W")
    assert result["n_buckets"] == 5
    assert result["bucket_mean_of_means"] == pytest.approx(np.mean(values))
    assert result["t_stat"] > 0    # clearly positive mean, should show up as such


def test_capital_day_edge_is_aggregate_not_per_trade_mean():
    """Two trades: a fast +1% (1 day) and a slow +10% (10 days). Per-trade mean
    of (pnl/days) is dominated by the fast trade; the aggregate measure is not."""
    rows = [
        _trade_row(net_pnl_pct=1.0, held_days=1),
        _trade_row(net_pnl_pct=10.0, held_days=10),
    ]
    df = pd.DataFrame(rows)
    naive_mean_of_ratios = np.mean([1.0 / 1, 10.0 / 10])   # = 1.0
    edge = capital_day_edge(df)
    # aggregate: mean(pnl)/mean(days) = 5.5 / 5.5 = 1.0 here by coincidence of
    # symmetric construction - use an asymmetric case to show the divergence.
    rows2 = [
        _trade_row(net_pnl_pct=1.0, held_days=1),
        _trade_row(net_pnl_pct=1.0, held_days=1),
        _trade_row(net_pnl_pct=10.0, held_days=10),
    ]
    df2 = pd.DataFrame(rows2)
    naive2 = np.mean([1.0, 1.0, 1.0])       # each trade's pnl/days = 1.0 -> naive says 1.0
    agg2 = capital_day_edge(df2)             # mean(pnl)=4.0, mean(days)=4.0 -> also 1.0... adjust
    # Use a case where the two measures genuinely diverge: many fast small wins,
    # one slow big win.
    rows3 = [_trade_row(net_pnl_pct=0.5, held_days=1) for _ in range(9)] + \
            [_trade_row(net_pnl_pct=9.0, held_days=9)]
    df3 = pd.DataFrame(rows3)
    naive3 = np.mean([0.5 / 1] * 9 + [9.0 / 9])              # = 0.5
    agg3 = capital_day_edge(df3)                              # mean_pnl=1.35, mean_days=1.8 -> 0.75
    assert agg3 != pytest.approx(naive3)
    assert agg3 == pytest.approx(df3["net_pnl_pct"].mean() / df3["held_days"].mean())
