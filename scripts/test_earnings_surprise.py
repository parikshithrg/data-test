"""Phase 6, signal 1: earnings surprise / PEAD, long-only, tested under
this project's rules - the first fundamentals-based hypothesis in this
project.

    python scripts/test_earnings_surprise.py --split primary --window train

Same harness as `test_momentum.py` (bhav store -> point-in-time universe ->
signal -> trade simulator -> sizing-independent metrics -> benchmark
excess -> placebo noise floor -> portfolio account simulation ->
hypothesis log) - PEAD needs no bespoke lifecycle: `ExitRule(
atr_stop_multiple=None)` is already a pure calendar hold, the same
"enter, hold N sessions" shape momentum uses, just triggered by a
fundamentals EVENT instead of a monthly rebalance. See
`dtest/signals/earnings_surprise.py` for the full story and
`dtest/features/fundamentals.py` for the SUE (standardized unexpected
earnings) construction and its point-in-time discipline.

HOLD_DAYS = 60 (~1 quarter) - the classic PEAD holding window from the
literature (drift persists roughly through the next earnings cycle), a
pre-stated, literature-grounded choice, not tuned against this project's
own data.

FUNDAMENTALS DATA SCOPE: `data/fundamentals/*.csv`, fetched 2026-08-20 for
all 926 symbols ever in this project's own point-in-time eligible universe
(`scripts/fetch_financial_results.py`) - 706/926 (76.2%) have any data,
633 have enough standalone history for at least one real SUE reading. Real,
partial coverage, same shape every other external data source in this
project already has (delivery 7.7%, OI 3.0%, FII flow 29-35%) - not a gap
specific to this signal.
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
from dtest.data.financial_results import load_financials
from dtest.engine.portfolio import run_portfolio
from dtest.engine.simulate import ExitRule, simulate_trades, trades_to_frame
from dtest.evaluate.hypothesis_log import HypothesisEntry, append_entry
from dtest.evaluate.metrics import benchmark_excess, non_overlapping_tstat, summary_stats
from dtest.evaluate.placebo import run_placebos
from dtest.features.fundamentals import point_in_time_series, sue_zscore, to_daily_panel
from dtest.signals.earnings_surprise import SUE_THRESHOLD, earnings_surprise_signal
from dtest.universe import build_universe

TARGET_VALUE_PER_TRADE = 10_000.0
HOLD_DAYS = 60

STORY = (
    "A quarter's EPS print is not instantly and fully absorbed into the "
    "price the moment it is disclosed - institutions and other slower "
    "participants take time to update position sizes on genuinely new "
    "information, so the market tends to keep drifting in the direction "
    "of a real surprise for weeks afterward rather than jumping straight "
    "to the new fair value (Bernard & Thomas 1989's post-earnings-"
    "announcement-drift finding, replicated across many markets since). "
    "No analyst-consensus data exists in this project, so 'surprise' is "
    "the standard SUE proxy: this quarter's standalone EPS minus the SAME "
    "quarter last year, z-scored against the stock's own trailing "
    "8-quarter history of such surprises - a naive seasonal random walk "
    "'expectation', not consensus, stated plainly as a real limitation of "
    "the proxy. Unlike every Phase 1-5 signal in this project (all react "
    "to a price/flow/technical dislocation and failed the same way, "
    "entering into the tail of an unfinished move), this reacts to a "
    "discrete, dated, FUNDAMENTAL disclosure event - a structurally "
    "different information source, the first fundamentals-based "
    "hypothesis tested here."
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


def _build_sue_per_symbol(cfg, symbols: list[str]) -> dict[str, pd.Series]:
    fund_dir = Path(cfg.paths.fundamentals_dir)
    out = {}
    n_with_data = 0
    for symbol in symbols:
        df = load_financials(symbol, fund_dir)
        if df.empty:
            continue
        eps = point_in_time_series(df, "eps_basic")
        sue = sue_zscore(eps)
        if sue.notna().any():
            out[symbol] = sue
            n_with_data += 1
    print(f"  SUE series built for {n_with_data}/{len(symbols)} symbols")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="primary", choices=["primary", "delivery"])
    ap.add_argument("--window", default=None, choices=["train", "val", "test"], required=True)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=SUE_THRESHOLD)
    args = ap.parse_args()

    pd.set_option("display.width", 150)
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()

    print(f"=== earnings surprise / PEAD ({args.split}/{args.window}) ===")

    panels, stocks = _load_price_panels(cfg)
    close, open_, high, low, volume, turnover = (
        panels["close"], panels["open"], panels["high"], panels["low"],
        panels["volume"], panels["turnover"],
    )
    print(f"price panels: {close.shape[0]} sessions x {close.shape[1]} symbols "
         f"({close.index[0].date()} .. {close.index[-1].date()})")

    split = cfg.split(args.split)
    start, end = split.window(args.window)
    if args.window in ("val", "test"):
        after_start = close.index[close.index >= pd.Timestamp(start)]
        if len(after_start) <= split.embargo_days:
            print(f"not enough sessions after {start} for a {split.embargo_days}-day embargo")
            return 1
        start = after_start[split.embargo_days]
        print(f"  embargo applied: {args.window} signals start {start.date()}")
    mask = (close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))
    if not mask.any():
        print(f"no data for {args.split}/{args.window} ({start}..{end})")
        return 1

    end_pos = close.index.searchsorted(pd.Timestamp(end), side="right")
    buffer_pos = min(end_pos + split.embargo_days, len(close.index))
    keep = close.index[:buffer_pos]
    close, open_, high, low = close.loc[keep], open_.loc[keep], high.loc[keep], low.loc[keep]
    volume, turnover = volume.loc[keep], turnover.loc[keep]
    mask = mask[:buffer_pos]
    print(f"  panels truncated to {keep[0].date()}..{keep[-1].date()} "
         f"({len(keep)} sessions: window + {split.embargo_days}-session resolution buffer)")

    stock_close, stock_open = close[stocks], open_[stocks]
    stock_high, stock_low = high[stocks], low[stocks]
    stock_volume, stock_turnover = volume[stocks], turnover[stocks]

    print("building point-in-time universe ...")
    uni = build_universe(stock_close, stock_turnover, cfg)
    eligible = uni.membership & mask[:, None]

    print("building point-in-time SUE series per symbol ...")
    sue_per_symbol = _build_sue_per_symbol(cfg, stocks)

    raw_signal = earnings_surprise_signal(sue_per_symbol, close.index, threshold=args.threshold)
    raw_signal = raw_signal.reindex(columns=stocks, fill_value=False)

    signal = raw_signal & eligible
    n_sig = int(signal.to_numpy().sum())
    print(f"signal: {n_sig} firings after universe/window intersection "
         f"(before: {int(raw_signal.to_numpy().sum())})")
    if n_sig == 0:
        print("zero signals in this window - nothing to simulate")
        return 0

    rule = ExitRule(max_hold_days=HOLD_DAYS, atr_stop_multiple=None, risk_reward=None)
    trades = trades_to_frame(simulate_trades(
        signal, "long", rule, open_=stock_open, high=stock_high, low=stock_low,
        close=stock_close, volume=stock_volume, atr_panel=None,
        target_value_per_trade=TARGET_VALUE_PER_TRADE, cfg=cfg,
    ))
    stats = summary_stats(trades)
    print("\n=== TRADE-LEVEL (sizing-independent) ===")
    for k, v in stats.as_dict().items():
        print(f"  {k:16s} {v}")

    tstat = non_overlapping_tstat(trades)
    print(f"\n  non-overlapping t-stat: t={tstat['t_stat']:.3f} "
         f"(n_buckets={tstat['n_buckets']}, n_trades={tstat['n_trades']})")

    bench_series = _benchmark_series(cfg, close.index)
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
        volume=stock_volume, atr_panel=None, target_value_per_trade=TARGET_VALUE_PER_TRADE,
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
                         volume=stock_volume, atr_panel=None, cfg=cfg)
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

    out_dir = Path(cfg.paths.runs) / f"earnings_surprise_{args.split}_{args.window}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "trades.csv", index=False)
    placebos.per_seed.to_csv(out_dir / "placebo_seeds.csv", index=False)
    port.equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)
    port.trades.to_csv(out_dir / "portfolio_trades.csv", index=False)
    print(f"\nWrote {out_dir}")

    decision = "accepted" if cmp["beats_best_placebo"] and pm["sharpe"] > 0 else "rejected"
    entry = HypothesisEntry(
        title="earnings_surprise (SUE/PEAD, long-only, honest execution)", story=STORY,
        split=args.split, window=args.window, metric="mean_net_pct",
        real_value=stats.mean_net_pct, placebo_max=cmp["placebo_max"],
        placebo_mean=cmp["placebo_mean"], beats_best_placebo=cmp["beats_best_placebo"],
        t_stat=tstat["t_stat"], n_buckets=tstat["n_buckets"], n_trades=tstat["n_trades"],
        decision=decision,
        notes=f"portfolio Sharpe {pm['sharpe']:.3f}, CAGR {pm['cagr_pct']:.2f}%, "
             f"max DD {pm['max_drawdown_pct']:.2f}%, sue_threshold={args.threshold}, "
             f"hold_days={HOLD_DAYS}",
    )
    log_path = Path(cfg.paths.runs) / "hypothesis_log.csv"
    append_entry(log_path, entry)
    print(f"\nlogged to {log_path}: decision={decision}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
