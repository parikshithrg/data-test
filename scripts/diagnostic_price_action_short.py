"""Price-action SHORT side: a cheap screening pass, because no honest
short simulator exists yet (`engine/simulate.py` is long-only; the
two-leg `engine/pairs_simulate.py` built for pairs trading doesn't fit a
standalone single-symbol short either). Same precedent as
`diagnostic_pairs_reversion.py`: approximate, no-cost, close-to-close
P&L, explicitly NOT wired into the honest engine, NOT logged as a
hypothesis - answers "does this side show any life" before the real
single-leg short simulator (T+1 futures fills, real futures costs,
rollover-forced exits) gets built.

    python scripts/diagnostic_price_action_short.py --split primary --window train

P&L CONVENTION: short entry at the signal bar's own NEXT close (a rough
stand-in for a T+1 fill - not honest, stated plainly), held `HOLD_DAYS`
sessions, exit at that later close. Return = -(exit/entry - 1), the
profitable direction for a short. Compared against a placebo: the same
count of random (symbol, date) draws from the same eligible pool on the
same dates, same holding period - isolates whether THIS signal's short
side has any edge over blind selection, the same question every other
placebo comparison in this project asks.
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
from dtest.signals.price_action import price_action_signal
from dtest.universe import build_universe

HOLD_DAYS = 7


def _load_panels(cfg):
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in ("high", "low", "close", "volume", "turnover")}
    return panels, stocks


def _approx_short_pnl(close: pd.DataFrame, entry_date, symbol, hold_days: int) -> float | None:
    calendar = close.index
    try:
        pos = calendar.get_loc(entry_date)
    except KeyError:
        return None
    entry_pos = pos + 1
    exit_pos = min(entry_pos + hold_days, len(calendar) - 1)
    if entry_pos >= len(calendar) or exit_pos <= entry_pos:
        return None
    entry_px = close.iat[entry_pos, close.columns.get_loc(symbol)]
    exit_px = close.iat[exit_pos, close.columns.get_loc(symbol)]
    if pd.isna(entry_px) or pd.isna(exit_px) or entry_px <= 0:
        return None
    return 100.0 * -(exit_px / entry_px - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="primary", choices=["primary", "delivery"])
    ap.add_argument("--window", default=None, choices=["train", "val", "test"], required=True)
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()

    pd.set_option("display.width", 150)
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()
    rng = np.random.default_rng(42)

    print(f"=== price_action SHORT screening ({args.split}/{args.window}) ===")

    panels, stocks = _load_panels(cfg)
    high, low, close, volume, turnover = (
        panels["high"][stocks], panels["low"][stocks], panels["close"][stocks],
        panels["volume"][stocks], panels["turnover"][stocks],
    )

    split = cfg.split(args.split)
    start, end = split.window(args.window)
    if args.window in ("val", "test"):
        after_start = close.index[close.index >= pd.Timestamp(start)]
        start = after_start[split.embargo_days]
        print(f"  embargo applied: {args.window} signals start {start.date()}")
    mask = (close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))

    print("building point-in-time universe ...")
    uni = build_universe(close, turnover, cfg)
    eligible = uni.membership & mask[:, None]

    _long_raw, short_raw = price_action_signal(high, low, close, volume)
    short_signal = short_raw & eligible
    n_sig = int(short_signal.to_numpy().sum())
    print(f"signal: {n_sig} firings after universe/window intersection "
         f"(before: {int(short_raw.to_numpy().sum())})")
    if n_sig == 0:
        print("zero signals in this window")
        return 0

    rows, cols = np.where(short_signal.to_numpy())
    real_pnls = []
    for r, c in zip(rows, cols):
        pnl = _approx_short_pnl(close, close.index[r], short_signal.columns[c], HOLD_DAYS)
        if pnl is not None:
            real_pnls.append(pnl)

    real_arr = np.array(real_pnls)
    print(f"\nREAL short signals: n={len(real_arr)}  mean_pnl%={real_arr.mean():.4f}  "
         f"win_rate%={(real_arr > 0).mean() * 100:.2f}")

    eligible_np = eligible.to_numpy()
    placebo_means = []
    for seed in range(args.seeds):
        seed_rng = np.random.default_rng(seed)
        placebo_pnls = []
        for r, c in zip(rows, cols):
            d = close.index[r]
            elig_row = eligible_np[r]
            elig_syms = np.where(elig_row)[0]
            if len(elig_syms) == 0:
                continue
            pick = short_signal.columns[seed_rng.choice(elig_syms)]
            pnl = _approx_short_pnl(close, d, pick, HOLD_DAYS)
            if pnl is not None:
                placebo_pnls.append(pnl)
        if placebo_pnls:
            placebo_means.append(float(np.mean(placebo_pnls)))

    placebo_arr = np.array(placebo_means)
    print(f"PLACEBO (n={args.seeds} seeds, same dates/count, random eligible symbol): "
         f"mean_of_means%={placebo_arr.mean():.4f}  max%={placebo_arr.max():.4f}  "
         f"min%={placebo_arr.min():.4f}")
    beats_best = real_arr.mean() > placebo_arr.max()
    print(f"real beats every placebo seed: {beats_best}")

    out_dir = Path(cfg.paths.runs) / "diagnostic_price_action_short"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pnl_pct": real_arr}).to_csv(
        out_dir / f"real_{args.split}_{args.window}.csv", index=False)
    pd.DataFrame({"placebo_mean_pct": placebo_arr}).to_csv(
        out_dir / f"placebo_{args.split}_{args.window}.csv", index=False)
    print(f"\nWrote {out_dir}")
    print("\nNOT logged to hypothesis_log.csv - approximate P&L, no costs, "
         "no honest fills, no real short instrument. See this file's own "
         "docstring for what would need building before this could become "
         "a real hypothesis test.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
