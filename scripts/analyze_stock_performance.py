"""DIAGNOSTIC ONLY - per-symbol performance breakdown for each of this
project's 10 distinct strategies, using their real, already-saved trades
(no new backtest). Ranks best/worst stocks by mean net_pnl_pct per symbol
(min 5 trades to appear, so a single lucky/unlucky fill can't dominate the
list), and cross-references against sector (industry_map.csv) and against
every OTHER strategy's own best/worst list, to separate a genuinely
stock-specific effect from a signal-specific one.

    python scripts/analyze_stock_performance.py

Pairs strategies (same_sector_pairing, pairs_reversion) attribute each trade
to BOTH legs separately (long_symbol with its own long_net_pct, short_symbol
with its own short_net_pct) rather than to one "trade" per pair - a stock
that keeps showing up on the losing side of a short leg is a different
finding than one that's a bad long pick, and collapsing them would hide
that.

NOT logged to hypothesis_log.csv - descriptive breakdown of results already
in the log, not a new hypothesis.
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

# strategy name -> (trades file relative to runs/, kind)
# kind "single" = one symbol column; "pairs" = long_symbol/short_symbol
STRATEGIES: dict[str, tuple[str, str]] = {
    "mean_reversion": ("mean_reversion_primary_train/trades.csv", "single"),
    "delivery_breakout": ("delivery_breakout_delivery_train/trades.csv", "single"),
    "oi_momentum": ("oi_momentum_primary_train/trades.csv", "single"),
    "participant_tilt": ("participant_tilt_delivery_train/trades.csv", "single"),
    "participant_tilt_stress_gated": (
        "participant_tilt_stress_gated_delivery_train/trades.csv", "single"),
    "vol_squeeze_breakout": ("vol_squeeze_breakout_primary_train/trades.csv", "single"),
    "price_action_long": ("price_action_long_primary_train/trades.csv", "single"),
    "momentum": ("momentum_primary_train/trades.csv", "single"),
    "pairs_reversion": ("pairs_reversion_honest/real_trades_primary_train.csv", "pairs"),
    "same_sector_pairing": ("same_sector_pairing/random_primary_train.csv", "pairs"),
}


def _per_symbol_single(df: pd.DataFrame) -> pd.DataFrame:
    resolved = df[df["net_pnl_pct"].notna()]
    g = resolved.groupby("symbol")["net_pnl_pct"]
    out = pd.DataFrame({
        "n_trades": g.size(),
        "mean_net_pct": g.mean(),
        "total_net_pct": g.sum(),
        "win_rate_pct": resolved.groupby("symbol")["net_pnl_pct"].apply(lambda s: (s > 0).mean() * 100),
    })
    return out.reset_index()


def _per_symbol_pairs(df: pd.DataFrame) -> pd.DataFrame:
    resolved = df[df["net_pnl_pct"].notna()]
    long_leg = resolved[["long_symbol", "long_net_pct"]].rename(
        columns={"long_symbol": "symbol", "long_net_pct": "net_pnl_pct"})
    long_leg["leg"] = "long"
    short_leg = resolved[["short_symbol", "short_net_pct"]].rename(
        columns={"short_symbol": "symbol", "short_net_pct": "net_pnl_pct"})
    short_leg["leg"] = "short"
    legs = pd.concat([long_leg, short_leg], ignore_index=True)
    g = legs.groupby("symbol")["net_pnl_pct"]
    out = pd.DataFrame({
        "n_trades": g.size(),
        "mean_net_pct": g.mean(),
        "total_net_pct": g.sum(),
        "win_rate_pct": legs.groupby("symbol")["net_pnl_pct"].apply(lambda s: (s > 0).mean() * 100),
    })
    return out.reset_index()


def main() -> int:
    cfg = load_config()
    sector = pd.read_csv(cfg.paths.industry_map)
    sector_map = dict(zip(sector["symbol"].astype(str).str.strip(),
                           sector["industry"].astype(str).str.strip()))

    all_best = []
    all_worst = []
    per_strategy_tables = {}

    for strat_name, (rel_path, kind) in STRATEGIES.items():
        df = pd.read_csv(RUNS / rel_path)
        per_symbol = _per_symbol_single(df) if kind == "single" else _per_symbol_pairs(df)
        per_symbol = per_symbol[per_symbol["n_trades"] >= MIN_TRADES].copy()
        per_symbol["sector"] = per_symbol["symbol"].map(sector_map).fillna("(unmapped)")
        per_symbol = per_symbol.sort_values("mean_net_pct", ascending=False)
        per_strategy_tables[strat_name] = per_symbol

        best = per_symbol.head(TOP_N).copy()
        worst = per_symbol.tail(TOP_N).sort_values("mean_net_pct").copy()
        best["strategy"] = strat_name
        worst["strategy"] = strat_name
        all_best.append(best)
        all_worst.append(worst)

        print(f"\n{'=' * 70}\n{strat_name}  ({kind}, n_symbols={len(per_symbol)}, "
              f"min_trades={MIN_TRADES})\n{'=' * 70}")
        print(f"BEST {TOP_N}:")
        print(best[["symbol", "sector", "n_trades", "mean_net_pct", "win_rate_pct"]]
              .to_string(index=False))
        print(f"\nWORST {TOP_N}:")
        print(worst[["symbol", "sector", "n_trades", "mean_net_pct", "win_rate_pct"]]
              .to_string(index=False))

    out_dir = RUNS / "stock_performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_best, ignore_index=True).to_csv(out_dir / "best_by_strategy.csv", index=False)
    pd.concat(all_worst, ignore_index=True).to_csv(out_dir / "worst_by_strategy.csv", index=False)
    for name, tbl in per_strategy_tables.items():
        tbl.to_csv(out_dir / f"{name}_all_symbols.csv", index=False)

    # cross-strategy: which symbols appear in >1 strategy's worst/best list
    worst_all = pd.concat(all_worst, ignore_index=True)
    best_all = pd.concat(all_best, ignore_index=True)
    worst_counts = worst_all["symbol"].value_counts()
    best_counts = best_all["symbol"].value_counts()
    repeat_worst = worst_counts[worst_counts > 1]
    repeat_best = best_counts[best_counts > 1]

    print(f"\n\n{'=' * 70}\nCROSS-STRATEGY REPEAT OFFENDERS / OVERPERFORMERS\n{'=' * 70}")
    print(f"Symbols in >1 strategy's WORST-{TOP_N} list:")
    print(repeat_worst.to_string() if len(repeat_worst) else "  (none)")
    print(f"\nSymbols in >1 strategy's BEST-{TOP_N} list:")
    print(repeat_best.to_string() if len(repeat_best) else "  (none)")

    # sector concentration in worst lists
    print(f"\n\nSector distribution, WORST-{TOP_N} lists pooled across all strategies:")
    print(worst_all["sector"].value_counts().to_string())
    print(f"\nSector distribution, BEST-{TOP_N} lists pooled across all strategies:")
    print(best_all["sector"].value_counts().to_string())

    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
