"""DIAGNOSTIC ONLY - same per-symbol best/worst breakdown as
`analyze_stock_performance.py`, restricted to POST-2021 entries only, using
the real delivery-split trades that already exist (delivery/train
2019-06-27..2023-06-30, plus momentum's own delivery/val 2023-07..2025-03) -
NOT primary split's held-out test window (2022-2026), which stays untouched
per this project's standing discipline. Every signal that only ever ran on
primary/train has no real post-2021 data at all and is correctly absent
here, not backfilled with a new backtest.

    python scripts/analyze_stock_performance_post2021.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config

RUNS = Path(__file__).resolve().parent.parent / "runs"
MIN_TRADES = 5
TOP_N = 5
CUTOFF = pd.Timestamp("2021-01-01")

# strategy name -> (trades file(s) relative to runs/, kind, date_col)
# momentum combines train+val since both are real, already-run delivery-split
# data and val alone (233 trades) would be too thin on its own.
STRATEGIES: dict[str, tuple[list[str], str, str]] = {
    "mean_reversion": (["mean_reversion_delivery_train/trades.csv"], "single", "entry_date"),
    "delivery_breakout": (["delivery_breakout_delivery_train/trades.csv"], "single", "entry_date"),
    "oi_momentum": (["oi_momentum_delivery_train/trades.csv"], "single", "entry_date"),
    "participant_tilt": (["participant_tilt_delivery_train/trades.csv"], "single", "entry_date"),
    "participant_tilt_stress_gated": (
        ["participant_tilt_stress_gated_delivery_train/trades.csv"], "single", "entry_date"),
    "vol_squeeze_breakout": (["vol_squeeze_breakout_delivery_train/trades.csv"], "single", "entry_date"),
    "price_action_long": (["price_action_long_delivery_train/trades.csv"], "single", "entry_date"),
    "momentum": (["momentum_delivery_train/trades.csv", "momentum_delivery_val/trades.csv"],
                 "single", "entry_date"),
    "pairs_reversion": (["pairs_reversion_honest/real_trades_delivery_train.csv"],
                        "pairs", "entry_fill_date"),
    "same_sector_pairing": (["same_sector_pairing/random_delivery_train.csv"],
                            "pairs", "entry_fill_date"),
}


def _load_post2021(rel_paths: list[str], date_col: str) -> pd.DataFrame:
    frames = []
    for rel_path in rel_paths:
        df = pd.read_csv(RUNS / rel_path, parse_dates=[date_col])
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined[combined[date_col] >= CUTOFF]


def _per_symbol_single(df: pd.DataFrame) -> pd.DataFrame:
    resolved = df[df["net_pnl_pct"].notna()]
    g = resolved.groupby("symbol")["net_pnl_pct"]
    out = pd.DataFrame({
        "n_trades": g.size(), "mean_net_pct": g.mean(), "total_net_pct": g.sum(),
        "win_rate_pct": resolved.groupby("symbol")["net_pnl_pct"].apply(lambda s: (s > 0).mean() * 100),
    })
    return out.reset_index()


def _per_symbol_pairs(df: pd.DataFrame) -> pd.DataFrame:
    resolved = df[df["net_pnl_pct"].notna()]
    long_leg = resolved[["long_symbol", "long_net_pct"]].rename(
        columns={"long_symbol": "symbol", "long_net_pct": "net_pnl_pct"})
    short_leg = resolved[["short_symbol", "short_net_pct"]].rename(
        columns={"short_symbol": "symbol", "short_net_pct": "net_pnl_pct"})
    legs = pd.concat([long_leg, short_leg], ignore_index=True)
    g = legs.groupby("symbol")["net_pnl_pct"]
    out = pd.DataFrame({
        "n_trades": g.size(), "mean_net_pct": g.mean(), "total_net_pct": g.sum(),
        "win_rate_pct": legs.groupby("symbol")["net_pnl_pct"].apply(lambda s: (s > 0).mean() * 100),
    })
    return out.reset_index()


def main() -> int:
    cfg = load_config()
    sector = pd.read_csv(cfg.paths.industry_map)
    sector_map = dict(zip(sector["symbol"].astype(str).str.strip(),
                           sector["industry"].astype(str).str.strip()))

    all_best, all_worst = [], []
    for strat_name, (rel_paths, kind, date_col) in STRATEGIES.items():
        df = _load_post2021(rel_paths, date_col)
        per_symbol = _per_symbol_single(df) if kind == "single" else _per_symbol_pairs(df)
        per_symbol = per_symbol[per_symbol["n_trades"] >= MIN_TRADES].copy()
        per_symbol["sector"] = per_symbol["symbol"].map(sector_map).fillna("(unmapped)")
        per_symbol = per_symbol.sort_values("mean_net_pct", ascending=False)

        best = per_symbol.head(TOP_N).copy()
        worst = per_symbol.tail(TOP_N).sort_values("mean_net_pct").copy()
        best["strategy"], worst["strategy"] = strat_name, strat_name
        all_best.append(best)
        all_worst.append(worst)

        print(f"\n{'=' * 70}\n{strat_name}  (post-2021, n_symbols={len(per_symbol)}, "
              f"min_trades={MIN_TRADES})\n{'=' * 70}")
        print(f"BEST {TOP_N}:")
        print(best[["symbol", "sector", "n_trades", "mean_net_pct", "win_rate_pct"]].to_string(index=False)
              if len(best) else "  (fewer than MIN_TRADES symbols qualify)")
        print(f"\nWORST {TOP_N}:")
        print(worst[["symbol", "sector", "n_trades", "mean_net_pct", "win_rate_pct"]].to_string(index=False)
              if len(worst) else "  (fewer than MIN_TRADES symbols qualify)")

    out_dir = RUNS / "stock_performance_post2021"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_all = pd.concat(all_best, ignore_index=True)
    worst_all = pd.concat(all_worst, ignore_index=True)
    best_all.to_csv(out_dir / "best_by_strategy.csv", index=False)
    worst_all.to_csv(out_dir / "worst_by_strategy.csv", index=False)

    worst_counts = worst_all["symbol"].value_counts()
    best_counts = best_all["symbol"].value_counts()
    print(f"\n\n{'=' * 70}\nCROSS-STRATEGY REPEAT NAMES (post-2021)\n{'=' * 70}")
    print(f"Symbols in >1 strategy's WORST-{TOP_N} list:")
    print(worst_counts[worst_counts > 1].to_string() if (worst_counts > 1).any() else "  (none)")
    print(f"\nSymbols in >1 strategy's BEST-{TOP_N} list:")
    print(best_counts[best_counts > 1].to_string() if (best_counts > 1).any() else "  (none)")

    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
