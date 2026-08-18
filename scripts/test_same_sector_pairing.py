"""Phase 3 follow-up: "plain same-sector pairing" as its OWN hypothesis,
not merely the placebo `select_pairs` (correlation-screened pairs) was
compared against.

    python scripts/test_same_sector_pairing.py --split primary --window train

WHY THIS EXISTS. The 2026-08-17 honest pairs re-test found the
correlation screen didn't earn its complexity - a same-sized RANDOM
same-sector draw scored higher, both gross and (after the 2026-08-18
rollover fix) net, and cleared t=2.339 for the first time in this
project's history. But that number was only ever measured as a
comparator, sized to match however many correlated pairs `select_pairs`
happened to find that month - never scoped, tested, or placebo'd as its
own candidate. This script does that properly.

STORY (shared by both real variants below): two stocks from the same
sector are linked enough by common exposure - sector-wide fund flows, an
index/ETF rebalancing pressure that hits the whole sector, one company's
earnings surprise bleeding sentiment into its peer, shared regulatory or
input-cost shocks - that a wide relative-price dislocation between them
is disproportionately temporary. Unlike `pairs_reversion`'s original
premise, this does NOT require the two names to have been historically
correlated first - shared sector membership alone is the claim.

TWO SELECTION RULES FOR THE SAME STORY, both tested (user's explicit
choice, 2026-08-18): `random_same_sector_pairs` (uniform random draw
within each sector, capped like `select_pairs`) and
`liquidity_ranked_same_sector_pairs` (deterministic - each sector's most
liquid names, no RNG). Both are compared against ONE placebo,
`random_pairs_any_sector` (random pairs from the WHOLE eligible universe,
ignoring sector) - the placebo that isolates whether "same-sector"
specifically is doing anything, as opposed to any-two-stocks mean
reversion.

Same fill/cost/rollforward machinery as `test_pairs_reversion.py`,
unchanged: T+1-open on both legs, real cash-equity and futures costs,
the 2026-08-18 rollforward-at-entry contract selection.
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
from dtest.engine.pairs_simulate import simulate_pair_trades, trades_to_frame
from dtest.evaluate.hypothesis_log import HypothesisEntry, append_entry
from dtest.evaluate.metrics import non_overlapping_tstat
from dtest.features.pairs import (
    liquidity_ranked_same_sector_pairs,
    random_pairs_any_sector,
    random_same_sector_pairs,
)
from dtest.signals.pairs_reversion import pair_trade_events
from dtest.universe import build_universe

ZSCORE_WINDOW = 20
Z_ENTRY = 2.0
Z_EXIT = 0.5
MAX_HOLD_DAYS = 20
MAX_PAIRS_PER_SECTOR = 3   # matches select_pairs' own default cap

STORY = (
    "Two stocks from the same sector are linked enough by common exposure "
    "(sector-wide fund flows, an index/ETF rebalance that hits the whole "
    "sector, one company's earnings surprise bleeding sentiment into its "
    "peer, shared regulatory or input-cost shocks) that a wide relative-"
    "price dislocation between them is disproportionately temporary - "
    "unlike the original pairs_reversion premise, this does NOT require "
    "the two names to already be historically correlated; shared sector "
    "membership alone is the claim being tested."
)


def _to_panel_generic(long_df, field):
    return to_cash_panel(long_df, field)


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

    tstat_input = resolved.rename(columns={"signal_entry_date": "entry_date"})
    tstat = non_overlapping_tstat(tstat_input)

    print(f"\n{label}: n_events={n_events} n_resolved={n_resolved} "
         f"mean_net_pnl%={mean_pnl:.4f} win_rate%={win_rate:.2f} "
         f"mean_cost%={mean_cost:.4f} reverted%={reverted_pct:.1f} rollover%={rollover_pct:.1f}")
    print(f"  non-overlapping t-stat: t={tstat['t_stat']:.3f} (n_buckets={tstat['n_buckets']})")

    return {"label": label, "df": df, "n_resolved": n_resolved, "mean_pnl": mean_pnl,
            "win_rate": win_rate, "t_stat": tstat["t_stat"], "n_buckets": tstat["n_buckets"]}


def _log_decision(cfg, args, title, real, placebo):
    beats_placebo = placebo is None or real["mean_pnl"] > placebo["mean_pnl"]
    decision = "accepted" if (beats_placebo and real["mean_pnl"] > 0 and real["t_stat"] > 2.0) else "rejected"
    entry = HypothesisEntry(
        title=title, story=STORY, split=args.split, window=args.window, metric="net_pnl_pct",
        real_value=real["mean_pnl"],
        placebo_max=placebo["mean_pnl"] if placebo else float("nan"),
        placebo_mean=placebo["mean_pnl"] if placebo else float("nan"),
        beats_best_placebo=beats_placebo,
        t_stat=real["t_stat"], n_buckets=real["n_buckets"], n_trades=real["n_resolved"],
        decision=decision,
        notes=(f"placebo(random, any sector): n={placebo['n_resolved']} "
              f"mean={placebo['mean_pnl']:.4f}% t={placebo['t_stat']:.3f}"
              if placebo else "no placebo trades resolved"),
    )
    log_path = Path(cfg.paths.runs) / "hypothesis_log.csv"
    append_entry(log_path, entry)
    print(f"logged '{title}': decision={decision}")
    return decision


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

    print(f"=== same-sector pairing, own hypothesis ({args.split}/{args.window}) ===")

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
    print(f"  futures contracts: {fut_contracts['symbol'].nunique()} symbols")

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

    random_by_date, liquidity_by_date = [], []
    placebo_for_random_by_date, placebo_for_liquidity_by_date = [], []

    for k, rdate in enumerate(rebalances):
        window_end = rebalances[k + 1] if k + 1 < len(rebalances) else pd.Timestamp(end)
        eligible = list(uni.membership.loc[rdate][uni.membership.loc[rdate]].index)
        eligible = [s for s in eligible if s in fut_symbols]
        if len(eligible) < 4:
            continue

        random_pairs = random_same_sector_pairs(sector_map, eligible, MAX_PAIRS_PER_SECTOR, rng)
        liquidity_pairs = liquidity_ranked_same_sector_pairs(
            sector_map, eligible, turnover, as_of=rdate, max_pairs_per_sector=MAX_PAIRS_PER_SECTOR)
        placebo_for_random = random_pairs_any_sector(eligible, len(random_pairs), rng)
        placebo_for_liquidity = random_pairs_any_sector(eligible, len(liquidity_pairs), rng)

        for a, b in random_pairs:
            random_by_date.append((a, b, rdate, window_end))
        for a, b in liquidity_pairs:
            liquidity_by_date.append((a, b, rdate, window_end))
        for a, b in placebo_for_random:
            placebo_for_random_by_date.append((a, b, rdate, window_end))
        for a, b in placebo_for_liquidity:
            placebo_for_liquidity_by_date.append((a, b, rdate, window_end))

        if (k + 1) % 30 == 0:
            print(f"  ... {k + 1}/{len(rebalances)} rebalances formed "
                 f"({len(random_by_date)} random, {len(liquidity_by_date)} liquidity pairs so far)")

    random_res = _run_variant("SAME-SECTOR, random draw (real hypothesis 1)", random_by_date,
                              close, cash_open, fut_contracts, cfg)
    placebo_random_res = _run_variant("PLACEBO for (1): random, any sector", placebo_for_random_by_date,
                                      close, cash_open, fut_contracts, cfg)
    liquidity_res = _run_variant("SAME-SECTOR, liquidity-ranked (real hypothesis 2)", liquidity_by_date,
                                 close, cash_open, fut_contracts, cfg)
    placebo_liquidity_res = _run_variant("PLACEBO for (2): random, any sector", placebo_for_liquidity_by_date,
                                         close, cash_open, fut_contracts, cfg)

    out_dir = Path(cfg.paths.runs) / "same_sector_pairing"
    out_dir.mkdir(parents=True, exist_ok=True)
    for res, name in ((random_res, "random"), (placebo_random_res, "placebo_for_random"),
                      (liquidity_res, "liquidity"), (placebo_liquidity_res, "placebo_for_liquidity")):
        if res is not None:
            res["df"].to_csv(out_dir / f"{name}_{args.split}_{args.window}.csv", index=False)
    print(f"\nWrote {out_dir}")

    if random_res is not None:
        _log_decision(cfg, args, "same_sector_pairing (random draw)", random_res, placebo_random_res)
    else:
        print("\nrandom-draw variant: no trades resolved - cannot log a decision")

    if liquidity_res is not None:
        _log_decision(cfg, args, "same_sector_pairing (liquidity-ranked)", liquidity_res, placebo_liquidity_res)
    else:
        print("\nliquidity-ranked variant: no trades resolved - cannot log a decision")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
