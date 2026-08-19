"""DIAGNOSTIC ONLY - digs into delivery_breakout's holding-period result from
`build_holding_period_analysis.py` (mean 12m excess vs NIFTY50 +7.86%, but
only 41.9% of individual trades beat NIFTY at 12m - the same
mean-driven-by-outliers shape this project has been burned by before).

    python scripts/dig_delivery_breakout_holding_period.py

Three checks, all reusing REAL project machinery rather than inventing new
statistics:
1. 30-seed placebo band (`dtest.evaluate.placebo._placebo_signals` - the
   exact same same-dates/same-counts/blind-draw function every other
   hypothesis in this project was tested against), run through the identical
   time-based-hold entry computation `build_holding_period_analysis.py`
   already uses, at all four checkpoints (1/3/6/12m).
2. A bucketed t-stat on excess_return_12m_pct - bucketed by ENTRY MONTH, not
   entry week. Stated explicitly: even a monthly bucket UNDER-corrects for a
   252-trading-day hold, since entries even several months apart still share
   most of their forward window - this is a conservative approximation, not
   a claim of the true independent-observation count.
3. Tail/skew diagnostics at 12m (mean vs median, top/bottom 5% contribution)
   to make the mean-vs-majority divergence concrete rather than inferred
   from the 41.9% figure alone.

NOT logged to hypothesis_log.csv - this is a diagnostic on an exploratory
read, not a new hypothesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dtest import load_config, set_seeds
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.data.delivery import load_delivery_long
from dtest.evaluate.placebo import _placebo_signals
from dtest.signals.delivery_breakout import delivery_breakout_signal
from dtest.universe import build_universe

from build_holding_period_analysis import (  # noqa: E402
    N_MONTHS, TRADING_DAYS_PER_MONTH, _entries_from_signal, _load_price_panels,
)

BASE_SEED = 42
N_SEEDS = 30
RUNS = Path(__file__).resolve().parent.parent / "runs" / "holding_period_analysis"


def _mean_stats(df: pd.DataFrame) -> dict:
    return {f"mean_return_{m}m_pct": df[f"return_{m}m_pct"].mean() for m in (1, 3, 6, 12)}


def main() -> int:
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()

    print("loading full price panels ...")
    panels, stocks = _load_price_panels(cfg)
    close, open_, high, low, volume, turnover = (
        panels["close"], panels["open"], panels["high"], panels["low"],
        panels["volume"], panels["turnover"],
    )
    stock_close, stock_open = close[stocks], open_[stocks]
    stock_turnover = turnover[stocks]
    dates = stock_close.index
    symbols = np.array(stock_close.columns)
    close_arr = stock_close.to_numpy(dtype=float)
    open_arr = stock_open.to_numpy(dtype=float)

    print("building point-in-time universe (full history) ...")
    uni = build_universe(stock_close, stock_turnover, cfg)
    eligible = uni.membership

    print("loading NIFTY50 ...")
    nifty_df = pd.read_csv(cfg.paths.price_dir / "NIFTY50_DAILY.csv",
                            parse_dates=["date"]).set_index("date").sort_index()
    nifty_df = nifty_df.loc[nifty_df.index <= pd.Timestamp(cfg.as_of)]
    nifty_close = nifty_df["close"].reindex(dates).ffill().to_numpy(dtype=float)

    print("loading delivery data ...")
    deliv_aligned = to_panel(load_delivery_long(cfg.paths.fno_db), "delivery_pct").reindex(
        index=stock_close.index, columns=stock_close.columns)

    raw_signal = delivery_breakout_signal(stock_close, deliv_aligned)
    real_signal = raw_signal.fillna(False).astype(bool) & eligible
    n_real = int(real_signal.to_numpy().sum())
    print(f"real delivery_breakout signal: {n_real} eligible firings")

    real_df = _entries_from_signal(real_signal, open_arr, close_arr, dates, symbols, nifty_close)
    real_stats = _mean_stats(real_df)
    print(f"real mean returns: {real_stats}")

    # ---- 1. Placebo band, same dates/counts, blind draw --------------------
    print(f"\nrunning {N_SEEDS} placebo seeds (same dates/counts as the real signal) ...")
    placebo_rows = []
    for i in range(N_SEEDS):
        seed = BASE_SEED + i
        placebo_sig = _placebo_signals(real_signal, eligible, stock_close, seed)
        placebo_df = _entries_from_signal(placebo_sig, open_arr, close_arr, dates, symbols, nifty_close)
        row = {"seed": seed, "n_entries": len(placebo_df)}
        row.update(_mean_stats(placebo_df))
        placebo_rows.append(row)
    placebo_summary = pd.DataFrame(placebo_rows)
    placebo_summary.to_csv(RUNS / "delivery_breakout_placebo_seeds.csv", index=False)

    print("\n=== PLACEBO COMPARISON (raw return_Xm_pct, real vs 30 blind-draw seeds) ===")
    for m in (1, 3, 6, 12):
        col = f"mean_return_{m}m_pct"
        real_v = real_stats[col]
        band = placebo_summary[col]
        pct_rank = float((band < real_v).mean() * 100.0)
        print(f"  {m:>2}m: real={real_v:7.3f}%  placebo min={band.min():7.3f}%  "
              f"mean={band.mean():7.3f}%  max={band.max():7.3f}%  "
              f"beats_best={real_v > band.max()}  percentile={pct_rank:.1f}")

    # ---- 2. Bucketed t-stat on excess_return_12m_pct, by entry MONTH -------
    print("\n=== BUCKETED T-STAT on excess_return_12m_pct (entry-MONTH buckets - "
          "conservative approximation, see module docstring) ===")
    resolved_12m = real_df.dropna(subset=["excess_return_12m_pct"]).copy()
    resolved_12m["bucket"] = pd.to_datetime(resolved_12m["entry_date"]).dt.to_period("M")
    bucket_means = resolved_12m.groupby("bucket")["excess_return_12m_pct"].mean()
    t_stat, p_value = stats.ttest_1samp(bucket_means.to_numpy(), 0.0)
    print(f"  n_trades={len(resolved_12m)}  n_month_buckets={len(bucket_means)}  "
          f"bucket_mean_of_means={bucket_means.mean():.3f}%  t={t_stat:.3f}  p={p_value:.4f}")

    # ---- 3. Tail / skew diagnostics at 12m ----------------------------------
    print("\n=== TAIL / SKEW DIAGNOSTICS at 12m (excess_return_12m_pct) ===")
    ex = resolved_12m["excess_return_12m_pct"]
    print(f"  mean={ex.mean():.3f}%  median={ex.median():.3f}%  std={ex.std():.3f}%  "
          f"pct_positive={100 * (ex > 0).mean():.1f}%")
    sorted_ex = ex.sort_values()
    n = len(sorted_ex)
    top5 = sorted_ex.iloc[int(n * 0.95):]
    bottom5 = sorted_ex.iloc[:int(n * 0.05)]
    rest = sorted_ex.iloc[int(n * 0.05):int(n * 0.95)]
    print(f"  top 5% ({len(top5)} trades) mean excess: {top5.mean():.2f}%  "
          f"(sum contribution to total: {100 * top5.sum() / ex.sum():.1f}%)")
    print(f"  bottom 5% ({len(bottom5)} trades) mean excess: {bottom5.mean():.2f}%")
    print(f"  middle 90% ({len(rest)} trades) mean excess: {rest.mean():.2f}%")

    real_df.to_csv(RUNS / "delivery_breakout_entries_full.csv", index=False)
    print(f"\nWrote {RUNS / 'delivery_breakout_placebo_seeds.csv'} and "
          f"{RUNS / 'delivery_breakout_entries_full.csv'}")
    print("\nNOT logged to hypothesis_log.csv - diagnostic on an exploratory read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
