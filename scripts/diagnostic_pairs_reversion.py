"""Phase 3 screening pass: does the pairs-reversion phenomenon exist at all
on this data, before any of the expensive infrastructure (continuous
futures price, margin cost model, a real two-leg simulator) gets built.

    python scripts/diagnostic_pairs_reversion.py --split primary --window train

APPROXIMATE ON PURPOSE, STATED UP FRONT. This uses same-day CLOSE prices for
both legs of both entry and exit (not the honest T+1-open convention
`engine/simulate.py` enforces for every other signal in this project) and
NO COST MODEL at all - no statutory charges, no futures margin, no
short-borrow. Same precedent as `execution_diagnostic.py`'s deliberate
same-bar-close comparison: a legitimate thing to want to know (does the
underlying phenomenon exist), not a legitimate thing to trade on. This
script never writes to hypothesis_log.csv for that reason - only a
follow-up with real costs and honest fills on top of a continuous futures
price series (Phase 1 of the "Long, Short, Neutral" architecture) could
earn that.

P&L CONVENTION: each leg is treated as equal notional; a resolved trade's
P&L is the AVERAGE of the long leg's and the short leg's own close-to-close
return (long leg return, and the NEGATIVE of the short leg's return, since
a falling short leg is the profitable direction) - the return on capital
committed per leg, not netted against real margin, which would differ from
this in practice and isn't modelled anywhere in either project yet.

THE COMPARISON THAT MATTERS: correlation-selected pairs (`features.pairs.
select_pairs`) against a SAME-SECTOR RANDOM-PAIR placebo formed the exact
same way except correlation is ignored. This isolates the one design choice
actually under test - does screening for correlated pairs add anything
over just betting that any two same-sector stocks mean-revert against each
other - rather than asking whether same-sector pairing itself is enough.
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


def _load_panels(cfg):
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in ("close", "turnover")}
    return panels, stocks


def _trade_pnl(close, trade) -> float | None:
    try:
        entry_long = close.at[trade.entry_date, trade.long_symbol]
        exit_long = close.at[trade.exit_date, trade.long_symbol]
        entry_short = close.at[trade.entry_date, trade.short_symbol]
        exit_short = close.at[trade.exit_date, trade.short_symbol]
    except KeyError:
        return None
    if any(pd.isna(x) or x <= 0 for x in (entry_long, exit_long, entry_short, exit_short)):
        return None
    long_ret = exit_long / entry_long - 1.0
    short_ret = -(exit_short / entry_short - 1.0)
    return 100.0 * (long_ret + short_ret) / 2.0


def _random_same_sector_pairs(sector_map, eligible_symbols, n_target, rng):
    by_sector: dict[str, list[str]] = {}
    for sym in eligible_symbols:
        sector = sector_map.get(sym)
        if sector is not None:
            by_sector.setdefault(sector, []).append(sym)
    candidates = []
    for sector, symbols in by_sector.items():
        if len(symbols) < 2:
            continue
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                candidates.append((symbols[i], symbols[j]))
    if not candidates:
        return []
    idx = rng.choice(len(candidates), size=min(n_target, len(candidates)), replace=False)
    return [candidates[i] for i in idx]


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

    print(f"=== pairs-reversion screening: mean_reversion-of-spread ({args.split}/{args.window}) ===")

    panels, stocks = _load_panels(cfg)
    close, turnover = panels["close"][stocks], panels["turnover"][stocks]
    print(f"price panel: {close.shape[0]} sessions x {close.shape[1]} symbols "
         f"({close.index[0].date()} .. {close.index[-1].date()})")

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

    rebalances = [d for d in uni.rebalance_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    print(f"{len(rebalances)} rebalance dates in window")

    real_trades, placebo_trades = [], []
    for k, rdate in enumerate(rebalances):
        window_end = rebalances[k + 1] if k + 1 < len(rebalances) else pd.Timestamp(end)
        eligible = list(uni.membership.loc[rdate][uni.membership.loc[rdate]].index)
        if len(eligible) < 4:
            continue

        pairs = select_pairs(close, sector_map, eligible, as_of=rdate,
                             formation_window=FORMATION_WINDOW, min_corr=MIN_CORR,
                             max_pairs_per_sector=MAX_PAIRS_PER_SECTOR)
        placebo_pairs = _random_same_sector_pairs(sector_map, eligible, len(pairs), rng)

        for sym_a, sym_b in pairs:
            trades = pair_trade_events(close[sym_a], close[sym_b], rdate, window_end, sym_a, sym_b,
                                       zscore_window=ZSCORE_WINDOW, z_entry=Z_ENTRY,
                                       z_exit=Z_EXIT, max_hold_days=MAX_HOLD_DAYS)
            for t in trades:
                pnl = _trade_pnl(close, t)
                if pnl is not None:
                    real_trades.append({"pair": f"{sym_a}/{sym_b}", "entry": t.entry_date,
                                        "exit_reason": t.exit_reason, "pnl_pct": pnl})

        for sym_a, sym_b in placebo_pairs:
            trades = pair_trade_events(close[sym_a], close[sym_b], rdate, window_end, sym_a, sym_b,
                                       zscore_window=ZSCORE_WINDOW, z_entry=Z_ENTRY,
                                       z_exit=Z_EXIT, max_hold_days=MAX_HOLD_DAYS)
            for t in trades:
                pnl = _trade_pnl(close, t)
                if pnl is not None:
                    placebo_trades.append({"pair": f"{sym_a}/{sym_b}", "entry": t.entry_date,
                                           "exit_reason": t.exit_reason, "pnl_pct": pnl})

        if (k + 1) % 20 == 0:
            print(f"  ... {k + 1}/{len(rebalances)} rebalances processed "
                 f"({len(real_trades)} real, {len(placebo_trades)} placebo trades so far)")

    real_df = pd.DataFrame(real_trades)
    placebo_df = pd.DataFrame(placebo_trades)

    def _summary(df, label):
        if df.empty:
            print(f"\n{label}: 0 trades")
            return
        n = len(df)
        mean_pnl = df["pnl_pct"].mean()
        win_rate = (df["pnl_pct"] > 0).mean() * 100
        reverted_pct = (df["exit_reason"] == "reverted").mean() * 100
        print(f"\n{label}: n={n}  mean_pnl%={mean_pnl:.4f}  win_rate%={win_rate:.2f}  "
             f"reverted%={reverted_pct:.1f}")

    _summary(real_df, "CORRELATION-SELECTED pairs")
    _summary(placebo_df, "RANDOM same-sector pairs (placebo)")

    out_dir = Path(cfg.paths.runs) / "diagnostic_pairs_reversion"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not real_df.empty:
        real_df.to_csv(out_dir / f"real_trades_{args.split}_{args.window}.csv", index=False)
    if not placebo_df.empty:
        placebo_df.to_csv(out_dir / f"placebo_trades_{args.split}_{args.window}.csv", index=False)
    print(f"\nWrote {out_dir}")
    print("\nNOT logged to hypothesis_log.csv - approximate P&L, no costs, "
         "no honest fills. See this file's own docstring for exactly what "
         "would need building before this could become a real hypothesis test.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
