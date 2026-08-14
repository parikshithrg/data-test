"""Phase 1: re-test mean_reversion under this project's rules.

    python scripts/test_mean_reversion.py --years 2004 2014          # smoke test
    python scripts/test_mean_reversion.py --split primary --window train

Wires every layer built so far into one run: bhav store -> point-in-time
universe -> signal -> ATR -> trade simulator -> sizing-independent metrics ->
benchmark excess -> placebo noise floor -> portfolio account simulation ->
hypothesis log. This is the first genuine end-to-end exercise of the whole
harness, not just of one module in isolation.

`--years` runs against whatever has been ingested so far and is EXPLICITLY a
smoke test (labelled as such in the output and never written to the
hypothesis log) - the bhavcopy ingest was still filling in when this was
first run. `--split`/`--window` runs against the real, pre-committed config
split and is the only mode that logs a result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config, set_seeds
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.corporate_actions import daily_returns, detect_actions
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.engine.portfolio import run_portfolio
from dtest.engine.simulate import ExitRule, simulate_trades, trades_to_frame
from dtest.evaluate.hypothesis_log import HypothesisEntry, append_entry
from dtest.evaluate.metrics import benchmark_excess, non_overlapping_tstat, summary_stats
from dtest.evaluate.placebo import run_placebos
from dtest.features.technical import atr
from dtest.signals.mean_reversion import mean_reversion_signal
from dtest.universe import build_universe

TARGET_VALUE_PER_TRADE = 10_000.0
STORY = (
    "A name pushed 1.5+ sample std devs below its own 50-day average in a "
    "short window is disproportionately a forced or panicked seller (margin "
    "calls, index-fund rebalancing, tax-loss selling, an overreacted headline) "
    "rather than someone with new information about permanently impaired "
    "value; the imbalance is expected to partially correct once the selling "
    "pressure itself is exhausted. This is the predecessor project's only "
    "strategy that survived a genuine held-out walk-forward validation - this "
    "run re-tests it under honest execution (T+1 open fills, real costs, a "
    "point-in-time universe) rather than trusting the earlier result."
)


def _load_panels(cfg, years):
    """Every symbol in the parsed bhav store IS a stock, by construction - the
    store keeps `series == "EQ"` only (`bhav_store.KEEP_SERIES`), and NSE's
    cash bhavcopy never carries an index LEVEL as a row at all. So, unlike the
    predecessor's CSV-based approach, no separate stock/index classification
    step is needed here, and NONE is applied against `nifty500.csv` - doing so
    would silently reintroduce survivorship bias, since that file is TODAY's
    membership snapshot and would exclude every delisted/renamed name (HDFC,
    MINDTREE, LTI, ...) this rebuild exists to keep. `nifty500.csv` is used
    ELSEWHERE in this script only for its industry labels (sector-cap
    bookkeeping), never to restrict which symbols are tradeable.
    """
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=years)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    if years is not None:
        long_df = long_df[long_df["date"].dt.year.isin(years)]

    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in ("open", "high", "low", "close",
                                                 "volume", "turnover", "prev_close")}
    return panels, stocks, long_df


def _benchmark_series(cfg, calendar) -> pd.Series | None:
    """NIFTY50 close, reindexed onto the trading calendar.

    NSE's cash bhavcopy lists TRADEABLE instruments only - the index LEVEL
    itself never appears as a row in it, so the benchmark cannot come from the
    same bhav_store the stock panel does. Read from the existing, already-
    audited `NIFTY50_DAILY.csv` instead. This is a deliberately narrow
    exception to "everything comes from the freshly-ingested store": an index
    LEVEL is a single published number with no survivorship question of its
    own (unlike a universe of STOCK picks, which is exactly what the
    bhavcopy rebuild exists to fix) - reusing it as a comparison yardstick
    does not reintroduce the bias this project was built to remove.
    """
    path = cfg.paths.price_dir / "NIFTY50_DAILY.csv"
    if not path.exists():
        print(f"  WARNING: benchmark file not found at {path} - "
             "benchmark comparison will be skipped")
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df.loc[df.index <= pd.Timestamp(cfg.as_of)]
    return df["close"].reindex(calendar).ffill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="smoke-test mode: run on whatever years are ingested")
    ap.add_argument("--split", default=None, choices=["primary", "delivery"])
    ap.add_argument("--window", default=None, choices=["train", "val", "test"])
    ap.add_argument("--seeds", type=int, default=None, help="override placebo_seeds")
    args = ap.parse_args()

    if not args.years and not (args.split and args.window):
        ap.error("pass either --years (smoke test) or --split + --window (real run)")

    pd.set_option("display.width", 150)
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()

    smoke = args.years is not None
    print(("=== SMOKE TEST" if smoke else "=== REAL RUN") +
         f" mean_reversion re-test {'(years ' + str(args.years) + ')' if smoke else f'({args.split}/{args.window})'} ===")

    panels, stocks, long_df = _load_panels(cfg, args.years)
    close, open_, high, low, volume, turnover, prev_close = (
        panels["close"], panels["open"], panels["high"], panels["low"],
        panels["volume"], panels["turnover"], panels["prev_close"],
    )
    print(f"panels: {close.shape[0]} sessions x {close.shape[1]} symbols "
         f"({close.index[0].date()} .. {close.index[-1].date()})")

    if args.split:
        split = cfg.split(args.split)
        start, end = split.window(args.window)
        if args.window in ("val", "test"):
            # Embargo: push the SIGNAL start forward by embargo_days TRADING
            # sessions past the config boundary, so a position opened near the
            # end of the prior window has fully resolved (or been forcibly
            # time-exited, at most `max_hold_days` sessions later) before this
            # window's own signals begin. Without this, a trade counted as a
            # "train result" could depend on val-period price action, and
            # train/val stop being independently measurable - exactly the
            # leak `splits.embargo_days` in config.toml commits to closing.
            # embargo_days=60 comfortably covers this signal's max_hold_days=7.
            after_start = close.index[close.index >= pd.Timestamp(start)]
            if len(after_start) <= split.embargo_days:
                print(f"not enough sessions after {start} to apply a "
                     f"{split.embargo_days}-day embargo")
                return 1
            start = after_start[split.embargo_days]
            print(f"  embargo applied: {args.window} signals start {start.date()} "
                 f"({split.embargo_days} trading sessions after the raw split boundary)")
        mask = (close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))
        if not mask.any():
            print(f"no data for {args.split}/{args.window} ({start}..{end}) in the "
                 "currently ingested range - re-run once the ingest reaches it")
            return 1

        # Truncate every panel to the window's end PLUS a resolution buffer
        # (embargo_days trading sessions - comfortably covers max_hold_days=7),
        # never to the full multi-decade calendar. A signal near the window's
        # own end still needs a few forward bars to resolve (that is not
        # look-ahead - see the embargo comment above), but running the account
        # simulation for years past the last possible entry is not neutral:
        # the equity curve sits FLAT once trading stops, which understates
        # volatility, distorts CAGR (computed over years of inactivity), and
        # makes the NIFTY50 comparison apples-to-oranges (a real 22-year
        # buy-and-hold against an account that only traded for 13 of them).
        # Caught by inspecting the actual equity curve after the first run of
        # this script, not assumed correct from a clean exit code.
        end_pos = close.index.searchsorted(pd.Timestamp(end), side="right")
        buffer_pos = min(end_pos + split.embargo_days, len(close.index))
        keep = close.index[:buffer_pos]
        close, open_, high, low = close.loc[keep], open_.loc[keep], high.loc[keep], low.loc[keep]
        volume, turnover, prev_close = volume.loc[keep], turnover.loc[keep], prev_close.loc[keep]
        mask = mask[:buffer_pos]
        print(f"  panels truncated to {keep[0].date()}..{keep[-1].date()} "
             f"({len(keep)} sessions: window + {split.embargo_days}-session resolution buffer)")
    else:
        # smoke test: use everything except the first year (universe warm-up
        # needs min_history_days of runway before it can select anything).
        warm = close.index[0] + pd.DateOffset(years=1)
        mask = close.index >= warm

    # ---- point-in-time universe --------------------------------------------
    stock_close = close[stocks]
    stock_open = open_[stocks]
    stock_high = high[stocks]
    stock_low = low[stocks]
    stock_volume = volume[stocks]
    stock_turnover = turnover[stocks]
    stock_prev_close = prev_close[stocks]

    print("building point-in-time universe ...")
    uni = build_universe(stock_close, stock_turnover, cfg)
    print(f"  universe log: median size {uni.log['n_selected'].median():.0f}, "
         f"{uni.log['n_entries'].sum()} entries, {uni.log['n_exits'].sum()} exits "
         f"over {len(uni.rebalance_dates)} rebalances")

    # ---- signal + ATR (whole-panel, then intersect with universe + window) --
    action_report = detect_actions(stock_close, stock_prev_close)
    print(f"corporate actions: {action_report.n_actions} credible, "
         f"{len(action_report.rejected_dates)} dates rejected as mass events")

    raw_signal = mean_reversion_signal(stock_close)
    a = atr(stock_high, stock_low, stock_close, window=14)

    eligible = uni.membership & mask[:, None]
    signal = raw_signal & eligible
    print(f"signal: {int(signal.to_numpy().sum())} raw firings "
         f"(before universe/window intersection: {int(raw_signal.to_numpy().sum())})")

    if int(signal.to_numpy().sum()) == 0:
        print("zero signals in this window - nothing to simulate")
        return 0

    # ---- trade-level (sizing-independent) simulation -----------------------
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

    bench_series = _benchmark_series(cfg, close.index)
    if bench_series is not None:
        excess = benchmark_excess(trades, bench_series)
        if len(excess):
            print(f"  mean excess vs NIFTY50 over same holding window: "
                 f"{excess.mean():.3f}pp (n={len(excess)})")

    # ---- placebo noise floor ------------------------------------------------
    n_seeds = args.seeds if args.seeds is not None else (5 if smoke else cfg.placebo_seeds)
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

    # ---- portfolio-level (sizing-dependent) account simulation --------------
    # nifty500.csv used ONLY for its industry LABELS here, never to restrict
    # which symbols are tradeable (see _load_panels docstring). A symbol
    # missing from this label set - any delisted/renamed name not in TODAY's
    # snapshot - falls back to its own singleton "Unknown:<symbol>" sector in
    # portfolio.py, so the sector cap degrades gracefully rather than lumping
    # unmapped names together and letting them exempt each other from it.
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

    out_dir = Path(cfg.paths.runs) / ("mean_reversion_smoke" if smoke else
                                      f"mean_reversion_{args.split}_{args.window}")
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "trades.csv", index=False)
    placebos.per_seed.to_csv(out_dir / "placebo_seeds.csv", index=False)
    port.equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)
    port.trades.to_csv(out_dir / "portfolio_trades.csv", index=False)
    print(f"\nWrote {out_dir}")

    if not smoke:
        decision = "accepted" if cmp["beats_best_placebo"] and pm["sharpe"] > 0 else "rejected"
        entry = HypothesisEntry(
            title="mean_reversion re-test (honest execution)", story=STORY,
            split=args.split, window=args.window, metric="mean_net_pct",
            real_value=stats.mean_net_pct, placebo_max=cmp["placebo_max"],
            placebo_mean=cmp["placebo_mean"], beats_best_placebo=cmp["beats_best_placebo"],
            t_stat=tstat["t_stat"], n_buckets=tstat["n_buckets"], n_trades=tstat["n_trades"],
            decision=decision,
            notes=f"portfolio Sharpe {pm['sharpe']:.3f}, CAGR {pm['cagr_pct']:.2f}%, "
                 f"max DD {pm['max_drawdown_pct']:.2f}%",
        )
        log_path = Path(cfg.paths.runs) / "hypothesis_log.csv"
        append_entry(log_path, entry)
        print(f"\nlogged to {log_path}: decision={decision}")
    else:
        print("\n[SMOKE TEST - not logged to hypothesis_log; data ingest was "
             "still in progress when this ran]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
