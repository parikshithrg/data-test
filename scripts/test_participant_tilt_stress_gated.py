"""participant_tilt gated to the CALM tercile of the cross-asset stress
composite - promoted from `diagnostic_stress_gate_tercile.py`'s screen
(2026-08-19), the only variant of 36 tested across both stress-gate screens
(median-split and tercile-split, 6 signals x up to 3 gate directions each)
to cross this project's own t>2.0 bar (t=2.202, 93.3rd percentile vs. 30
placebo seeds) - though it still did NOT beat the single best placebo seed,
stated plainly, not glossed over. Promoted anyway on explicit request, as a
genuine long shot worth the real test, not because it already looked like a
sure thing.

    python scripts/test_participant_tilt_stress_gated.py --window train

KNOWN CAVEAT ON THIS SPECIFIC CANDIDATE, inherited from participant_tilt's
own prior finding (2026-08-17): its ungated failure is dominated by one
COVID-week cluster (2020-03-16, ~60% of total negative P&L by itself) - a
composite that flags COVID week as extreme stress and a calm-only gate that
excludes it are mechanically primed to look better for exactly that reason,
not necessarily because "calm-only participant tilt" is a real edge. This
test does not resolve that ambiguity - only a real, disciplined test/val
result can.

STRUCTURAL DIFFERENCE FROM `test_participant_tilt.py`: the stress composite
needs its own 252-session trailing warm-up (and India VIX's own history only
starts 2009-03-02), which extends before any split's train window starts -
so panels are NOT truncated to window+embargo here, unlike every other
test_*.py script. The composite/breadth are computed over the FULL available
history first, and only the SIGNAL is masked to the train window - a
deliberate, stated deviation, not an oversight. Same runs/hypothesis_log.csv
outcome either way since resolution still only ever uses real, already-
elapsed data.

Runs on `delivery`/train only, matching participant_tilt's own established
split (FII flow data has no signal before 2018-01-01) and the screen this
candidate was found on - no primary-split version was ever tested, none is
implied here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config, set_seeds
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.data.participant_flow import load_fii_net_index_flow
from dtest.engine.portfolio import run_portfolio
from dtest.engine.simulate import ExitRule, simulate_trades, trades_to_frame
from dtest.evaluate.hypothesis_log import HypothesisEntry, append_entry
from dtest.evaluate.metrics import benchmark_excess, non_overlapping_tstat, summary_stats
from dtest.evaluate.placebo import run_placebos
from dtest.features.stress import causal_percentile, cross_asset_stress_composite
from dtest.features.technical import atr
from dtest.signals.participant_tilt import participant_tilt_signal
from dtest.universe import build_universe

TARGET_VALUE_PER_TRADE = 10_000.0
LOWER_TERCILE = 33.333
STORY = (
    "A mean-reversion dip bought while FII net index-futures positioning "
    "sits above its own recent trend is disproportionately a normal "
    "pullback inside a market institutions are still net accumulating "
    "into. This gates participant_tilt further, to only the CALM tercile "
    "of a 6-dimension cross-asset systemic-stress composite (India VIX, "
    "US VIX, market breadth-inverted, USDINR 20d change, DXY, gold 20d "
    "return - the same construction Local Terminal's own Black Swan Radar "
    "dashboard already built, run here through this project's causal "
    "discipline for the first time) - the screening pass this candidate "
    "was chosen from found the calm tercile the only variant, of 36 tested "
    "across two separate threshold conventions, to cross this project's "
    "own t>2.0 significance bar. KNOWN CAVEAT, not resolved by this test: "
    "participant_tilt's own prior failure is dominated by one COVID-week "
    "cluster a stress-aware calm-only gate is mechanically primed to "
    "exclude - this test cannot on its own distinguish that from a real, "
    "general edge."
)


def _load_price_panels(cfg):
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in ("open", "high", "low", "close",
                                                 "volume", "turnover")}
    return panels, stocks


def _benchmark_series(cfg, calendar):
    path = cfg.paths.price_dir / "NIFTY50_DAILY.csv"
    if not path.exists():
        print(f"  WARNING: benchmark file not found at {path}")
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df.loc[df.index <= pd.Timestamp(cfg.as_of)]
    return df["close"].reindex(calendar).ffill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="delivery", choices=["delivery"])
    ap.add_argument("--window", default=None, choices=["train", "val", "test"], required=True)
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()

    pd.set_option("display.width", 150)
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()

    print(f"=== participant_tilt, CALM-TERCILE stress-gated ({args.split}/{args.window}) ===")

    panels, stocks = _load_price_panels(cfg)
    close, open_, high, low, volume, turnover = (
        panels["close"], panels["open"], panels["high"], panels["low"],
        panels["volume"], panels["turnover"],
    )
    stock_close, stock_open = close[stocks], open_[stocks]
    stock_high, stock_low = high[stocks], low[stocks]
    stock_volume, stock_turnover = volume[stocks], turnover[stocks]
    dates = stock_close.index
    print(f"price panels: {close.shape[0]} sessions x {close.shape[1]} symbols "
          f"({dates[0].date()} .. {dates[-1].date()})")

    print("loading FII net index-futures flow ...")
    fii_net = load_fii_net_index_flow(cfg.paths.fno_db)
    fii_aligned = fii_net.reindex(dates)
    coverage = fii_aligned.notna().mean()
    print(f"  FII flow coverage: {coverage:.1%} of sessions")

    print("building point-in-time universe (full history) ...")
    uni = build_universe(stock_close, stock_turnover, cfg)
    eligible_full = uni.membership

    print("loading macro stress series and building the composite ...")
    india_vix = pd.read_csv(cfg.paths.price_dir / "INDIAVIX_DAILY.csv",
                             parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    us_vix = pd.read_csv(cfg.paths.macro_dir / "VIX_DAILY.csv",
                          parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    usdinr = pd.read_csv(cfg.paths.macro_dir / "USDINR_DAILY.csv",
                          parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    dxy = pd.read_csv(cfg.paths.macro_dir / "DXY_DAILY.csv",
                       parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    gold = pd.read_csv(cfg.paths.macro_dir / "GOLD_DAILY.csv",
                        parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    sma200 = stock_close.rolling(200, min_periods=200).mean()
    above = (stock_close > sma200) & eligible_full
    denom = eligible_full.sum(axis=1).astype(float).replace(0.0, np.nan)
    breadth_pct = (above.sum(axis=1).astype(float) / denom * 100.0)

    dims = cross_asset_stress_composite(india_vix, us_vix, breadth_pct, usdinr, dxy, gold)
    composite_pctile = causal_percentile(dims["composite"])
    calm_mask = (composite_pctile <= LOWER_TERCILE).fillna(False)
    print(f"  {int(calm_mask.sum())} calm-tercile sessions "
          f"({composite_pctile.first_valid_index()} .. {composite_pctile.last_valid_index()})")

    split = cfg.split(args.split)
    start, end = split.window(args.window)
    if args.window in ("val", "test"):
        after_start = dates[dates >= pd.Timestamp(start)]
        if len(after_start) <= split.embargo_days:
            print(f"not enough sessions after {start} for a {split.embargo_days}-day embargo")
            return 1
        start = after_start[split.embargo_days]
        print(f"  embargo applied: {args.window} signals start {start.date()}")
    window_mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if not window_mask.any():
        print(f"no data for {args.split}/{args.window} ({start}..{end})")
        return 1

    raw_signal = participant_tilt_signal(stock_close, fii_aligned)
    a = atr(stock_high, stock_low, stock_close, window=14)

    eligible = eligible_full & window_mask[:, None]
    signal = raw_signal & eligible & calm_mask.to_numpy()[:, None]
    n_sig = int(signal.to_numpy().sum())
    print(f"signal: {n_sig} firings after universe/window/calm-tercile intersection "
          f"(before gating: {int((raw_signal & eligible).to_numpy().sum())})")
    if n_sig == 0:
        print("zero signals in this window - nothing to simulate")
        return 0

    rule = ExitRule(max_hold_days=7, atr_stop_multiple=2.0, risk_reward=2.5)
    trades = trades_to_frame(simulate_trades(
        signal, "long", rule, open_=stock_open, high=stock_high, low=stock_low,
        close=stock_close, volume=stock_volume, atr_panel=a,
        target_value_per_trade=TARGET_VALUE_PER_TRADE, cfg=cfg,
    ))
    stats = summary_stats(trades)
    print("\n=== TRADE-LEVEL (sizing-independent) ===")
    for k, v in stats.as_dict().items():
        print(f"  {k:16s} {v}")

    tstat = non_overlapping_tstat(trades)
    print(f"\n  non-overlapping t-stat: t={tstat['t_stat']:.3f} "
          f"(n_buckets={tstat['n_buckets']}, n_trades={tstat['n_trades']})")

    bench_series = _benchmark_series(cfg, dates)
    if bench_series is not None:
        excess = benchmark_excess(trades, bench_series)
        if len(excess):
            print(f"  mean excess vs NIFTY50 over same holding window: "
                  f"{excess.mean():.3f}pp (n={len(excess)})")

    n_seeds = args.seeds if args.seeds is not None else cfg.placebo_seeds
    print(f"\nrunning {n_seeds} placebo seeds ...")
    placebos = run_placebos(
        signal, eligible, "long", rule,
        open_=stock_open, high=stock_high, low=stock_low, close=stock_close,
        volume=stock_volume, atr_panel=a, target_value_per_trade=TARGET_VALUE_PER_TRADE,
        cfg=cfg, n_seeds=n_seeds,
    )
    cmp = placebos.compare(stats, "mean_net_pct")
    print("=== PLACEBO COMPARISON (mean_net_pct) ===")
    for k, v in cmp.items():
        print(f"  {k:24s} {v}")

    industry_ref = pd.read_csv(cfg.paths.industry_map)
    sector_map = dict(zip(industry_ref["symbol"].astype(str).str.strip(),
                           industry_ref["industry"].astype(str).str.strip()))
    port = run_portfolio(signal, "long", rule, sector_map, open_=stock_open,
                          high=stock_high, low=stock_low, close=stock_close,
                          volume=stock_volume, atr_panel=a, cfg=cfg)
    pm = port.metrics()
    print("\n=== PORTFOLIO-LEVEL (Rs %.0f, %d slots) ===" % (
        cfg.portfolio.initial_capital, cfg.portfolio.max_positions))
    print(f"  CAGR {pm['cagr_pct']:.2f}%  Sharpe {pm['sharpe']:.3f}  "
          f"max DD {pm['max_drawdown_pct']:.2f}%  n_days {pm['n_days']}")
    print(f"  skipped: no_slot={port.skipped_no_slot} sector_cap={port.skipped_sector_cap} "
          f"no_cash={port.skipped_no_cash} no_fill={port.skipped_no_fill}")

    if bench_series is not None:
        from dtest.engine.portfolio import benchmark_equity_curve, portfolio_metrics
        bench_curve = benchmark_equity_curve(
            bench_series.reindex(port.equity_curve["date"]),
            cfg.portfolio.initial_capital, port.equity_curve["date"],
        )
        bm = portfolio_metrics(bench_curve)
        print(f"  NIFTY50 buy&hold, same capital/dates: CAGR {bm['cagr_pct']:.2f}%  "
              f"Sharpe {bm['sharpe']:.3f}  max DD {bm['max_drawdown_pct']:.2f}%")

    out_dir = Path(cfg.paths.runs) / f"participant_tilt_stress_gated_{args.split}_{args.window}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "trades.csv", index=False)
    placebos.per_seed.to_csv(out_dir / "placebo_seeds.csv", index=False)
    port.equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)
    port.trades.to_csv(out_dir / "portfolio_trades.csv", index=False)
    print(f"\nWrote {out_dir}")

    decision = "accepted" if cmp["beats_best_placebo"] and pm["sharpe"] > 0 else "rejected"
    entry = HypothesisEntry(
        title="participant_tilt, CALM-TERCILE stress-gated", story=STORY,
        split=args.split, window=args.window, metric="mean_net_pct",
        real_value=stats.mean_net_pct, placebo_max=cmp["placebo_max"],
        placebo_mean=cmp["placebo_mean"], beats_best_placebo=cmp["beats_best_placebo"],
        t_stat=tstat["t_stat"], n_buckets=tstat["n_buckets"], n_trades=tstat["n_trades"],
        decision=decision,
        notes=f"portfolio Sharpe {pm['sharpe']:.3f}, CAGR {pm['cagr_pct']:.2f}%, "
              f"max DD {pm['max_drawdown_pct']:.2f}%, FII flow coverage {coverage:.1%}, "
              f"promoted from diagnostic_stress_gate_tercile.py's screen",
    )
    log_path = Path(cfg.paths.runs) / "hypothesis_log.csv"
    append_entry(log_path, entry)
    print(f"\nlogged to {log_path}: decision={decision}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
