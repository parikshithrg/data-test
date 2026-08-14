"""Sizing-independent trade metrics, benchmark-relative, non-overlapping t-stats.

SIZING-INDEPENDENT FIRST. Everything in this module operates on per-trade
PERCENTAGES - mean, median, win rate. The predecessor project's rule, earned
the hard way: summing per-trade percentages across overlapping, unsized trades
produced a phantom 14.5M-rupee "return" nobody could have earned. A
sizing-DEPENDENT question (does this survive at Rs 50,000 with 5 concurrent
slots, real drawdown) is a separate, later check - `engine/portfolio.py`.

HIT RATE VS WIN RATE - both computed, never confused. `hit_rate` is
target-hits / (target-hits + stop-hits), TIME exits excluded from the
denominator - the natural reading for a target/stop system, and the one
insensitive to how generous the time exit is. `win_rate` is
(net_pnl_pct > 0).mean() over every RESOLVED trade including time exits - the
convention used everywhere else a "win rate" is reported. Old headline claims
in the predecessor project silently switched between one convention and the
other from one session to the next; this module always returns both, labelled.

NON-OVERLAPPING T-STAT. Two trades entered in the same week largely share the
same market draw for their holding period, so treating every trade as an
independent observation inflates significance. The predecessor project
measured this directly on a signal's forward-return series: an all-dates t-stat
ran 3-5x the non-overlapping one. The same logic applies here, adapted for a
trade list rather than a daily series: trades are bucketed by entry week, and
the t-stat is computed on BUCKET MEANS, so the effective sample size is the
number of independent weeks touched, not the raw trade count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from dtest.engine.simulate import EXIT_NO_FILL, EXIT_STOP, EXIT_TARGET, EXIT_TIME, EXIT_UNRESOLVED


@dataclass(frozen=True)
class SummaryStats:
    n_signals: int              # every row in the trade frame, including no-fill
    n_resolved: int             # entry filled AND exit resolved
    n_unresolved: int
    n_no_fill: int
    mean_net_pct: float
    median_net_pct: float
    std_net_pct: float
    mean_gross_pct: float
    win_rate_pct: float         # net_pnl_pct > 0, over RESOLVED trades
    hit_rate_pct: float | None  # target / (target + stop), None if no stop/target used
    expiry_rate_pct: float      # share of resolved trades that were TIME exits
    mean_held_days: float
    mean_cost_pct: float

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def summary_stats(trades: pd.DataFrame) -> SummaryStats:
    """Sizing-independent summary of a trade frame from `simulate_trades`."""
    n_signals = len(trades)
    no_fill = trades[trades["exit_reason"] == EXIT_NO_FILL]
    unresolved = trades[trades["exit_reason"] == EXIT_UNRESOLVED]
    resolved = trades[trades["net_pnl_pct"].notna()]

    n_resolved = len(resolved)
    has_stop_target = resolved["exit_reason"].isin([EXIT_STOP, EXIT_TARGET]).any()
    n_stop = int((resolved["exit_reason"] == EXIT_STOP).sum())
    n_target = int((resolved["exit_reason"] == EXIT_TARGET).sum())
    n_time = int((resolved["exit_reason"] == EXIT_TIME).sum())

    return SummaryStats(
        n_signals=n_signals,
        n_resolved=n_resolved,
        n_unresolved=len(unresolved),
        n_no_fill=len(no_fill),
        mean_net_pct=float(resolved["net_pnl_pct"].mean()) if n_resolved else float("nan"),
        median_net_pct=float(resolved["net_pnl_pct"].median()) if n_resolved else float("nan"),
        std_net_pct=float(resolved["net_pnl_pct"].std(ddof=1)) if n_resolved > 1 else float("nan"),
        mean_gross_pct=float(resolved["gross_pnl_pct"].mean()) if n_resolved else float("nan"),
        win_rate_pct=100.0 * float((resolved["net_pnl_pct"] > 0).mean()) if n_resolved else float("nan"),
        hit_rate_pct=(100.0 * n_target / (n_target + n_stop)) if has_stop_target and (n_target + n_stop) else None,
        expiry_rate_pct=100.0 * n_time / n_resolved if n_resolved else float("nan"),
        mean_held_days=float(resolved["held_days"].mean()) if n_resolved else float("nan"),
        mean_cost_pct=float(resolved["cost_pct"].mean()) if n_resolved else float("nan"),
    )


def benchmark_excess(trades: pd.DataFrame, benchmark_close: pd.Series) -> pd.Series:
    """Each resolved trade's net return minus the benchmark's return over the
    SAME holding window (entry_date -> exit_date).

    This is the operationalisation of "benchmark-relative, always": a trade
    that made 2% while the index made 5% over the same days is not a win, and
    this makes that comparison a column instead of a claim.

    Uses the benchmark's own close-to-close return, not a T+1-execution model -
    a passive buy-and-hold has no signal/fill distinction to correct for.
    """
    resolved = trades[trades["net_pnl_pct"].notna()]
    if resolved.empty:
        return pd.Series(dtype=float, name="excess_pct")

    entry_px = benchmark_close.reindex(resolved["entry_date"]).to_numpy()
    exit_px = benchmark_close.reindex(resolved["exit_date"]).to_numpy()
    bench_ret_pct = (exit_px / entry_px - 1.0) * 100.0
    excess = resolved["net_pnl_pct"].to_numpy() - bench_ret_pct
    return pd.Series(excess, index=resolved.index, name="excess_pct")


def non_overlapping_tstat(trades: pd.DataFrame, freq: str = "W") -> dict:
    """One-sample t-test of net_pnl_pct against zero, on entry-week bucket means.

    Returns a dict rather than a bare float so the DEGRADED sample size (number
    of buckets, not number of trades) is always visible next to the statistic -
    reporting only the t-value invites quoting an inflated all-trades number by
    mistake.
    """
    resolved = trades[trades["net_pnl_pct"].notna()].copy()
    if resolved.empty:
        return {"t_stat": float("nan"), "p_value": float("nan"), "n_buckets": 0, "n_trades": 0}

    resolved["bucket"] = pd.to_datetime(resolved["entry_date"]).dt.to_period(freq)
    bucket_means = resolved.groupby("bucket")["net_pnl_pct"].mean()
    n = len(bucket_means)
    if n < 2:
        return {"t_stat": float("nan"), "p_value": float("nan"),
               "n_buckets": n, "n_trades": len(resolved)}

    t_stat, p_value = stats.ttest_1samp(bucket_means.to_numpy(), 0.0)
    return {
        "t_stat": float(t_stat), "p_value": float(p_value),
        "n_buckets": n, "n_trades": len(resolved),
        "bucket_mean_of_means": float(bucket_means.mean()),
    }


def capital_day_edge(trades: pd.DataFrame) -> float:
    """Total net P&L over total capital-DAYS, i.e. mean(net_pnl_pct)/mean(held_days).

    NOT mean(net_pnl_pct / held_days) per trade - the predecessor project
    measured that this is dominated by whichever trades resolve fastest (a stop
    usually fires in a day or two; a target can take weeks), and it read
    NEGATIVE in a window where mean per-trade P&L was positive. This is the
    corrected, aggregate version.
    """
    resolved = trades[trades["net_pnl_pct"].notna() & (trades["held_days"] > 0)]
    if resolved.empty:
        return float("nan")
    return float(resolved["net_pnl_pct"].mean() / resolved["held_days"].mean())
