"""Phase 3, real hypothesis test: pairs-reversion, honestly filled and
costed - the follow-up `diagnostic_pairs_reversion.py`'s own docstring
named as the bar this premise had to clear before it could be logged.

    python scripts/test_pairs_reversion.py --split primary --window train

Same pair formation as the screening pass (`features.pairs.select_pairs`,
same-sector correlation screen, monthly rebalance) and the same signal
timing (`signals.pairs_reversion.pair_trade_events`, unchanged) - only the
FILL and COST layer is new: `engine/pairs_simulate.py`'s honest T+1-open
fills on both legs, real cash-equity costs on the long leg, real futures
costs on the short leg, a rollforward-at-entry contract choice (skip an
almost-expired front month for the next contract), and a rollover-forced
exit where the CHOSEN contract would otherwise be held across expiry.

Reports BOTH correlation-selected pairs (the actual `select_pairs`
hypothesis) and a random-same-sector comparison (this test's placebo,
same role it played in the screening pass) - the screening pass found
random pairing scored HIGHER gross, so this run is the honest answer to
whether either survives real costs, not just whether the gross number was
positive.

RE-RUN 2026-08-18 with the rollforward-at-entry fix (`engine/
pairs_simulate.py::MIN_DTE_FOR_ENTRY_DAYS`) after diagnosing the first
honest run's 37% rollover-forced-exit rate: it was not the strategy's own
20-day hold cap firing (only 1/197 trades ever reached it), it was
entering into contracts already close to expiry. See that module's
docstring for the full diagnosis and the fix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config, set_seeds
from dtest.data.bhav_store import build_store, load_long, to_panel as to_cash_panel
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.data.fno_price import load_stock_futures_contracts
from dtest.engine.pairs_simulate import MIN_DTE_FOR_ENTRY_DAYS, simulate_pair_trades, trades_to_frame
from dtest.evaluate.hypothesis_log import HypothesisEntry, append_entry
from dtest.evaluate.metrics import non_overlapping_tstat
from dtest.features.pairs import select_pairs
from dtest.signals.pairs_reversion import pair_trade_events
from dtest.universe import build_universe

FORMATION_WINDOW = 252
MIN_CORR = 0.8
MAX_PAIRS_PER_SECTOR = 3
ZSCORE_WINDOW = 20
Z_ENTRY = 2.0
Z_EXIT = 0.5
MAX_HOLD_DAYS = 20

STORY = (
    "Two stocks from the same sector, historically correlated, whose "
    "log-price spread has drifted unusually wide are disproportionately "
    "showing a temporary, idiosyncratic dislocation (one name overreacting "
    "to news that is really about the sector, an index-fund flow, an "
    "earnings surprise bleeding into its peer's price) rather than a "
    "genuine re-rating of their relative value - the position only needs "
    "the RELATIVE mispricing to close, a narrower claim than a directional "
    "bet on the sector. Market-neutral by construction, the one design in "
    "this project that does not need the sector or the market to move a "
    "particular direction to work."
)


def _to_panel_generic(long_df, field):
    return to_cash_panel(long_df, field)


def _random_same_sector_pairs(sector_map, eligible_symbols, n_target, rng):
    by_sector: dict[str, list[str]] = {}
    for sym in eligible_symbols:
        sector = sector_map.get(sym)
        if sector is not None:
            by_sector.setdefault(sector, []).append(sym)
    candidates = []
    for symbols in by_sector.values():
        if len(symbols) < 2:
            continue
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                candidates.append((symbols[i], symbols[j]))
    if not candidates:
        return []
    idx = rng.choice(len(candidates), size=min(n_target, len(candidates)), replace=False)
    return [candidates[i] for i in idx]


def _run_variant(label, pair_lists, close, cash_open, fut_contracts, cfg):
    all_events = []
    for sym_a, sym_b, rdate, window_end in pair_lists:
        events = pair_trade_events(close[sym_a], close[sym_b], rdate, window_end, sym_a, sym_b,
                                   zscore_window=ZSCORE_WINDOW, z_entry=Z_ENTRY,
                                   z_exit=Z_EXIT, max_hold_days=MAX_HOLD_DAYS)
        all_events.extend(events)

    if not all_events:
        print(f"\n{label}: 0 signal events")
        return None

    trades = simulate_pair_trades(
        all_events, cash_open=cash_open, fut_contracts=fut_contracts,
        calendar=close.index, cfg=cfg,
    )
    df = trades_to_frame(trades)
    resolved = df[df["net_pnl_pct"].notna()]
    n_events = len(df)
    n_resolved = len(resolved)
    if n_resolved == 0:
        print(f"\n{label}: {n_events} events, 0 resolved (all no_fill)")
        return None

    mean_pnl = resolved["net_pnl_pct"].mean()
    win_rate = (resolved["net_pnl_pct"] > 0).mean() * 100
    mean_cost = (resolved["long_cost_pct"] + resolved["short_cost_pct"]).mean()
    reverted_pct = (resolved["exit_reason"] == "reverted").mean() * 100
    rollover_pct = (resolved["exit_reason"] == "rollover").mean() * 100
    rollforward_pct = resolved["rolled_forward_at_entry"].mean() * 100

    tstat_input = resolved.rename(columns={"signal_entry_date": "entry_date"})
    tstat = non_overlapping_tstat(tstat_input)

    print(f"\n{label}: n_events={n_events} n_resolved={n_resolved} "
         f"mean_net_pnl%={mean_pnl:.4f} win_rate%={win_rate:.2f} "
         f"mean_cost%={mean_cost:.4f} reverted%={reverted_pct:.1f} rollover%={rollover_pct:.1f} "
         f"rolled_forward_at_entry%={rollforward_pct:.1f}")
    print(f"  non-overlapping t-stat: t={tstat['t_stat']:.3f} (n_buckets={tstat['n_buckets']})")

    return {"label": label, "df": df, "n_resolved": n_resolved, "mean_pnl": mean_pnl,
            "win_rate": win_rate, "t_stat": tstat["t_stat"], "n_buckets": tstat["n_buckets"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="primary", choices=["primary", "delivery"])
    ap.add_argument("--window", default=None, choices=["train", "val", "test"], required=True)
    args = ap.parse_args()

    pd.set_option("display.width", 150)
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()
    rng = np.random.default_rng(42)

    print(f"=== pairs-reversion, HONEST fills+costs ({args.split}/{args.window}) ===")

    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    close = _to_panel_generic(long_df, "close")[stocks]
    cash_open = _to_panel_generic(long_df, "open")[stocks]
    turnover = _to_panel_generic(long_df, "turnover")[stocks]
    print(f"cash panel: {close.shape[0]} sessions x {close.shape[1]} symbols")

    print("loading stock-futures contracts (all live, not just front-month) ...")
    fut_contracts = load_stock_futures_contracts(cfg.paths.fno_db)
    print(f"  futures contracts: {fut_contracts['symbol'].nunique()} symbols, "
         f"{len(fut_contracts)} (date, symbol, expiry) rows")

    split = cfg.split(args.split)
    start, end = split.window(args.window)
    if args.window in ("val", "test"):
        after_start = close.index[close.index >= pd.Timestamp(start)]
        start = after_start[split.embargo_days]
        print(f"  embargo applied: {args.window} signals start {start.date()}")

    print("building point-in-time universe ...")
    uni = build_universe(close, turnover, cfg)

    industry_ref = pd.read_csv(cfg.paths.industry_map)
    sector_map = dict(zip(industry_ref["symbol"].astype(str).str.strip(),
                          industry_ref["industry"].astype(str).str.strip()))
    fut_symbols = set(fut_contracts["symbol"].unique())

    rebalances = [d for d in uni.rebalance_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    print(f"{len(rebalances)} rebalance dates in window")

    real_pairs_by_date, placebo_pairs_by_date = [], []
    for k, rdate in enumerate(rebalances):
        window_end = rebalances[k + 1] if k + 1 < len(rebalances) else pd.Timestamp(end)
        eligible = list(uni.membership.loc[rdate][uni.membership.loc[rdate]].index)
        # Only consider symbols with real futures coverage - a pair whose
        # short leg has no futures data at all cannot be honestly filled.
        eligible = [s for s in eligible if s in fut_symbols]
        if len(eligible) < 4:
            continue

        pairs = select_pairs(close, sector_map, eligible, as_of=rdate,
                             formation_window=FORMATION_WINDOW, min_corr=MIN_CORR,
                             max_pairs_per_sector=MAX_PAIRS_PER_SECTOR)
        placebo_pairs = _random_same_sector_pairs(sector_map, eligible, len(pairs), rng)

        for a, b in pairs:
            real_pairs_by_date.append((a, b, rdate, window_end))
        for a, b in placebo_pairs:
            placebo_pairs_by_date.append((a, b, rdate, window_end))

        if (k + 1) % 30 == 0:
            print(f"  ... {k + 1}/{len(rebalances)} rebalances formed "
                 f"({len(real_pairs_by_date)} real, {len(placebo_pairs_by_date)} placebo pairs so far)")

    real = _run_variant("CORRELATION-SELECTED (real hypothesis)", real_pairs_by_date,
                        close, cash_open, fut_contracts, cfg)
    placebo = _run_variant("RANDOM SAME-SECTOR (placebo)", placebo_pairs_by_date,
                           close, cash_open, fut_contracts, cfg)

    out_dir = Path(cfg.paths.runs) / "pairs_reversion_honest"
    out_dir.mkdir(parents=True, exist_ok=True)
    if real is not None:
        real["df"].to_csv(out_dir / f"real_trades_{args.split}_{args.window}.csv", index=False)
    if placebo is not None:
        placebo["df"].to_csv(out_dir / f"placebo_trades_{args.split}_{args.window}.csv", index=False)
    print(f"\nWrote {out_dir}")

    if real is None:
        print("\nNo real trades resolved - cannot log a decision")
        return 0

    beats_placebo = placebo is None or real["mean_pnl"] > placebo["mean_pnl"]
    decision = "accepted" if (beats_placebo and real["mean_pnl"] > 0 and real["t_stat"] > 2.0) else "rejected"
    entry = HypothesisEntry(
        title="pairs_reversion (honest fills + costs)", story=STORY,
        split=args.split, window=args.window, metric="net_pnl_pct",
        real_value=real["mean_pnl"],
        placebo_max=placebo["mean_pnl"] if placebo else float("nan"),
        placebo_mean=placebo["mean_pnl"] if placebo else float("nan"),
        beats_best_placebo=beats_placebo,
        t_stat=real["t_stat"], n_buckets=real["n_buckets"], n_trades=real["n_resolved"],
        decision=decision,
        notes=(f"rollforward-at-entry fix applied (min_dte={MIN_DTE_FOR_ENTRY_DAYS}d); "
              f"placebo(random same-sector): n={placebo['n_resolved']} "
              f"mean={placebo['mean_pnl']:.4f}% t={placebo['t_stat']:.3f}"
              if placebo else
              f"rollforward-at-entry fix applied (min_dte={MIN_DTE_FOR_ENTRY_DAYS}d); "
              "no placebo trades resolved"),
    )
    log_path = Path(cfg.paths.runs) / "hypothesis_log.csv"
    append_entry(log_path, entry)
    print(f"\nlogged to {log_path}: decision={decision}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
