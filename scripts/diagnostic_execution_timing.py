"""DIAGNOSTIC: how much of mean_reversion's rejection is execution timing?

    python scripts/diagnostic_execution_timing.py --split primary --window train

Runs the PRODUCTION T+1-open simulator and the same-bar-CLOSE diagnostic
(`dtest.research.execution_diagnostic`) against the IDENTICAL signal panel,
universe, window, exit rule, and cost model - entry timing is the only thing
that differs. This isolates one candidate explanation for why this project's
mean_reversion result (rejected, worse than random) differs from the
predecessor's claim (its only validated survivor).

This is NOT a hypothesis test and writes NOTHING to the hypothesis log - it
answers "how sensitive is this result to one specific assumption", not "should
this ship". `dtest.research.execution_diagnostic` is explicitly diagnostic-only
for exactly this reason (see its module docstring).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config, set_seeds
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.engine.simulate import ExitRule, simulate_trades, trades_to_frame
from dtest.evaluate.metrics import benchmark_excess, non_overlapping_tstat, summary_stats
from dtest.evaluate.placebo import run_placebos
from dtest.features.technical import atr
from dtest.research.execution_diagnostic import simulate_same_bar_close
from dtest.signals.mean_reversion import mean_reversion_signal
from dtest.universe import build_universe

TARGET_VALUE_PER_TRADE = 10_000.0


def _load_panels(cfg, years):
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=years)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    if years is not None:
        long_df = long_df[long_df["date"].dt.year.isin(years)]
    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in ("open", "high", "low", "close",
                                                 "volume", "turnover", "prev_close")}
    return panels, stocks


def _report(label: str, trades: pd.DataFrame, bench: pd.Series | None) -> dict:
    stats = summary_stats(trades)
    tstat = non_overlapping_tstat(trades)
    excess = None
    if bench is not None:
        e = benchmark_excess(trades, bench)
        excess = float(e.mean()) if len(e) else float("nan")
    print(f"\n=== {label} ===")
    print(f"  n_resolved       {stats.n_resolved}")
    print(f"  mean_net_pct     {stats.mean_net_pct:.4f}")
    print(f"  median_net_pct   {stats.median_net_pct:.4f}")
    print(f"  win_rate_pct     {stats.win_rate_pct:.2f}")
    print(f"  hit_rate_pct     {stats.hit_rate_pct}")
    print(f"  expiry_rate_pct  {stats.expiry_rate_pct:.2f}")
    print(f"  mean_held_days   {stats.mean_held_days:.2f}")
    print(f"  t_stat           {tstat['t_stat']:.3f} (n_buckets={tstat['n_buckets']})")
    if excess is not None:
        print(f"  mean excess vs benchmark: {excess:.4f}pp")
    return {"label": label, "stats": stats, "tstat": tstat, "excess": excess}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=None)
    ap.add_argument("--split", default=None, choices=["primary", "delivery"])
    ap.add_argument("--window", default=None, choices=["train", "val", "test"])
    ap.add_argument("--seeds", type=int, default=15)
    args = ap.parse_args()
    if not args.years and not (args.split and args.window):
        ap.error("pass either --years or --split + --window")

    pd.set_option("display.width", 150)
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()

    panels, stocks = _load_panels(cfg, args.years)
    close, open_, high, low, volume, turnover, prev_close = (
        panels["close"], panels["open"], panels["high"], panels["low"],
        panels["volume"], panels["turnover"], panels["prev_close"],
    )
    print(f"panels: {close.shape[0]} sessions x {close.shape[1]} symbols")

    if args.split:
        split = cfg.split(args.split)
        start, end = split.window(args.window)
        if args.window in ("val", "test"):
            after_start = close.index[close.index >= pd.Timestamp(start)]
            start = after_start[split.embargo_days]
        end_pos = close.index.searchsorted(pd.Timestamp(end), side="right")
        buffer_pos = min(end_pos + split.embargo_days, len(close.index))
        keep = close.index[:buffer_pos]
        close, open_, high, low = close.loc[keep], open_.loc[keep], high.loc[keep], low.loc[keep]
        volume, turnover, prev_close = volume.loc[keep], turnover.loc[keep], prev_close.loc[keep]
        mask = (keep >= pd.Timestamp(start)) & (keep <= pd.Timestamp(end))
        print(f"  window {args.split}/{args.window}: {start.date() if hasattr(start,'date') else start} "
             f".. {end}, panels truncated to {keep[0].date()}..{keep[-1].date()}")
    else:
        warm = close.index[0] + pd.DateOffset(years=1)
        mask = close.index >= warm

    stock_close, stock_open = close[stocks], open_[stocks]
    stock_high, stock_low = high[stocks], low[stocks]
    stock_volume, stock_turnover = volume[stocks], turnover[stocks]

    print("building point-in-time universe ...")
    uni = build_universe(stock_close, stock_turnover, cfg)

    raw_signal = mean_reversion_signal(stock_close)
    a = atr(stock_high, stock_low, stock_close, window=14)
    eligible = uni.membership & mask[:, None]
    signal = raw_signal & eligible
    n_sig = int(signal.to_numpy().sum())
    print(f"signal: {n_sig} firings after universe/window intersection")
    if n_sig == 0:
        print("zero signals - nothing to compare")
        return 0

    rule = ExitRule(max_hold_days=7, atr_stop_multiple=2.0, risk_reward=2.5)

    bench = None
    bench_path = cfg.paths.price_dir / "NIFTY50_DAILY.csv"
    if bench_path.exists():
        bdf = pd.read_csv(bench_path, parse_dates=["date"]).set_index("date").sort_index()
        bench = bdf["close"].reindex(close.index).ffill()

    print("\nrunning production T+1-open simulator ...")
    t1_trades = trades_to_frame(simulate_trades(
        signal, "long", rule, open_=stock_open, high=stock_high, low=stock_low,
        close=stock_close, volume=stock_volume, atr_panel=a,
        target_value_per_trade=TARGET_VALUE_PER_TRADE, cfg=cfg,
    ))
    t1 = _report("PRODUCTION: T+1 open fill", t1_trades, bench)

    print("\nrunning DIAGNOSTIC same-bar-close simulator ...")
    sb_trades = trades_to_frame(simulate_same_bar_close(
        signal, "long", rule, open_=stock_open, high=stock_high, low=stock_low,
        close=stock_close, volume=stock_volume, atr_panel=a,
        target_value_per_trade=TARGET_VALUE_PER_TRADE, cfg=cfg,
    ))
    sb = _report("DIAGNOSTIC: same-bar close fill (predecessor's convention)", sb_trades, bench)

    print("\n=== SIDE BY SIDE ===")
    print(f"{'metric':22s} {'T+1 open (real)':>18s} {'same-bar close (diag)':>22s} {'gap':>10s}")
    for label, key in (("mean_net_pct", "mean_net_pct"), ("win_rate_pct", "win_rate_pct"),
                      ("expiry_rate_pct", "expiry_rate_pct"), ("mean_held_days", "mean_held_days")):
        v1 = getattr(t1["stats"], key)
        v2 = getattr(sb["stats"], key)
        print(f"{label:22s} {v1:18.4f} {v2:22.4f} {v2 - v1:10.4f}")
    print(f"{'t_stat':22s} {t1['tstat']['t_stat']:18.3f} {sb['tstat']['t_stat']:22.3f}")
    if t1["excess"] is not None:
        print(f"{'excess_vs_bench':22s} {t1['excess']:18.4f} {sb['excess']:22.4f} "
             f"{sb['excess'] - t1['excess']:10.4f}")

    print(f"\nplacebo check on same-bar-close result ({args.seeds} seeds) ...")
    placebos = run_placebos(
        signal, eligible, "long", rule, open_=stock_open, high=stock_high, low=stock_low,
        close=stock_close, volume=stock_volume, atr_panel=a,
        target_value_per_trade=TARGET_VALUE_PER_TRADE, cfg=cfg, n_seeds=args.seeds,
    )
    cmp = placebos.compare(sb["stats"], "mean_net_pct")
    print(f"  same-bar-close real mean:  {cmp['real']:.4f}")
    print(f"  placebo band:              [{cmp['placebo_min']:.4f}, {cmp['placebo_max']:.4f}]")
    print(f"  beats best placebo:        {cmp['beats_best_placebo']}")

    out = Path(cfg.paths.runs) / "diagnostic_execution_timing"
    out.mkdir(parents=True, exist_ok=True)
    t1_trades.to_csv(out / "t1_open_trades.csv", index=False)
    sb_trades.to_csv(out / "same_bar_close_trades.csv", index=False)
    print(f"\nWrote {out}")

    print("\n[DIAGNOSTIC ONLY - not logged to hypothesis_log. Same-bar-close entry "
         "is not a real execution model for this account and is never used to "
         "decide whether a hypothesis ships.]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
