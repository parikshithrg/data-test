"""MF ownership breadth: at each real Axis+SBI disclosure, go long the
stocks held by the broadest number of distinct schemes, tested under this
project's rules.

    python scripts/test_mf_breadth.py --split delivery --window train

Same harness as `test_momentum.py` (bhav store -> point-in-time universe
-> signal -> trade simulator -> sizing-independent metrics -> benchmark
excess -> placebo noise floor -> portfolio account simulation ->
hypothesis log) - breadth needs no bespoke lifecycle, same reasoning
momentum's own docstring gives: `ExitRule` with `atr_stop_multiple=None`
is already a pure calendar hold, which is what "enter at each disclosure,
hold ~1 month until the next one" is. See `dtest/signals/mf_breadth.py`
for the full story - a structurally different, persistent-state bet
(institutional consensus/stability) rather than the short-horizon EVENT
shape every other MF-holdings signal (mf_accumulation, mf_new_entrant)
and every price-derived signal in this project shares.

ONLY `--split delivery` IS MEANINGFUL, same reasoning as
`test_mf_accumulation.py` - the holdings data starts 2021-09 (Axis) /
2023-01 (SBI).

REAL, STATED SCOPE LIMITATION, printed with every run: breadth here means
how many of Axis's and SBI's OWN schemes hold a stock (2 of ~50 AMCs),
not real industry-wide ownership breadth.
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
from dtest.engine.portfolio import run_portfolio
from dtest.engine.simulate import ExitRule, simulate_trades, trades_to_frame
from dtest.evaluate.hypothesis_log import HypothesisEntry, append_entry
from dtest.evaluate.metrics import benchmark_excess, non_overlapping_tstat, summary_stats
from dtest.evaluate.placebo import run_placebos
from dtest.features.mf_holdings import (
    aggregate_monthly_quantity,
    breadth_panel,
    build_isin_symbol_map,
    to_event_panel,
)
from dtest.signals.mf_breadth import TOP_QUANTILE, mf_breadth_signal
from dtest.universe import build_universe

TARGET_VALUE_PER_TRADE = 10_000.0
HOLD_DAYS = 21  # ~1 month - re-rank and re-enter at the next real disclosure
STORY = (
    "A stock held by only one or two schemes reflects one manager's "
    "idiosyncratic call; a stock many distinct schemes independently "
    "choose to hold has survived several separate underwriting decisions "
    "- closer to a genuine quality/stability read than a single "
    "manager's conviction. Structurally different from mf_accumulation/ "
    "mf_new_entrant (both single-day events): this is a persistent "
    "cross-sectional state, re-ranked at each disclosure and held until "
    "the next one, the same shape momentum_signal already uses for its "
    "own long-horizon claim. REAL SCOPE LIMITATION: breadth here means "
    "how many of Axis's and SBI's OWN schemes hold a stock (2 of ~50 "
    "AMCs), not real industry-wide ownership breadth."
)


def _load_price_panels(cfg):
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in ("open", "high", "low", "close",
                                                 "volume", "turnover")}
    return panels, stocks, long_df


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
    ap.add_argument("--split", default="delivery", choices=["primary", "delivery"])
    ap.add_argument("--window", default=None, choices=["train", "val", "test"], required=True)
    ap.add_argument("--top-quantile", type=float, default=TOP_QUANTILE)
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()

    pd.set_option("display.width", 150)
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()

    print(f"=== mf_breadth ({args.split}/{args.window}) ===")
    print("REAL SCOPE LIMITATION: breadth is Axis+SBI's own schemes only "
          "(2 of ~50 AMCs), not real industry-wide ownership breadth.")

    panels, stocks, long_df = _load_price_panels(cfg)
    close, open_, high, low, volume, turnover = (
        panels["close"], panels["open"], panels["high"], panels["low"],
        panels["volume"], panels["turnover"],
    )
    print(f"price panels: {close.shape[0]} sessions x {close.shape[1]} symbols "
         f"({close.index[0].date()} .. {close.index[-1].date()})")

    isin_map = build_isin_symbol_map(long_df)
    holdings_path = Path(cfg.paths.amc_portfolios_dir) / "equity_holdings.csv"
    holdings = pd.read_csv(holdings_path, parse_dates=["period_end"])
    monthly = aggregate_monthly_quantity(holdings, isin_map)
    breadth = breadth_panel(monthly)
    print(f"  {len(monthly)} (period_end, symbol) aggregate rows, "
         f"{monthly['period_end'].nunique()} distinct disclosed months, "
         f"breadth range {int(breadth.min().min())}-{int(breadth.max().max())} schemes")

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

    event_panel = to_event_panel(breadth, stock_close.index)
    event_panel = event_panel.reindex(columns=stock_close.columns)
    n_events = int(event_panel.notna().to_numpy().sum())
    print(f"  {n_events} (date,symbol) cells carry a real disclosed breadth "
         f"reading within this window's price calendar")

    print("building point-in-time universe ...")
    uni = build_universe(stock_close, stock_turnover, cfg)

    raw_signal = mf_breadth_signal(event_panel, top_quantile=args.top_quantile)

    eligible = uni.membership & mask[:, None]
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

    out_dir = Path(cfg.paths.runs) / f"mf_breadth_{args.split}_{args.window}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "trades.csv", index=False)
    placebos.per_seed.to_csv(out_dir / "placebo_seeds.csv", index=False)
    port.equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)
    port.trades.to_csv(out_dir / "portfolio_trades.csv", index=False)
    print(f"\nWrote {out_dir}")

    decision = "accepted" if cmp["beats_best_placebo"] and pm["sharpe"] > 0 else "rejected"
    entry = HypothesisEntry(
        title="mf_breadth (Axis+SBI holdings, honest execution)", story=STORY,
        split=args.split, window=args.window, metric="mean_net_pct",
        real_value=stats.mean_net_pct, placebo_max=cmp["placebo_max"],
        placebo_mean=cmp["placebo_mean"], beats_best_placebo=cmp["beats_best_placebo"],
        t_stat=tstat["t_stat"], n_buckets=tstat["n_buckets"], n_trades=tstat["n_trades"],
        decision=decision,
        notes=f"portfolio Sharpe {pm['sharpe']:.3f}, CAGR {pm['cagr_pct']:.2f}%, "
             f"max DD {pm['max_drawdown_pct']:.2f}%, top_quantile={args.top_quantile}, "
             f"2-AMC scope (Axis+SBI only)",
    )
    log_path = Path(cfg.paths.runs) / "hypothesis_log.csv"
    append_entry(log_path, entry)
    print(f"\nlogged to {log_path}: decision={decision}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
