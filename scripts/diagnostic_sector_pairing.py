"""Sector-restricted same_sector_pairing: does slicing the ALREADY-RUN
same_sector_pairing trades by sector reveal a real per-sector effect the
all-sector-pooled aggregate hid - on real, already-saved trades, no new
backtest.

    python scripts/diagnostic_sector_pairing.py

THE PREMISE UNDER TEST, stated before running anything. `same_sector_pairing`
(both random-draw and liquidity-ranked variants, 2026-08-18) was accepted on
primary/train (t=3.49-3.70) but rejected on primary/val (t=0.77-0.95) and
flipped sign on delivery/train - closed as a train-window-specific effect,
pooled across all 13 sectors. The 2026-08-20 study of "which hypotheses
performed well" found the real per-symbol data (`runs/stock_performance/
same_sector_pairing_all_symbols.csv`) shows the highest-volume, most
consistent-looking names cluster in a few large liquid sectors (Banking,
Auto, Metals, Financial Services - 40-70 trades each, +1.7% to +2.6% mean).
This asks directly: is that visual clustering a real, sector-specific effect
that survives train->val, or just noise inside the already-rejected pooled
aggregate?

WHY NO NEW BACKTEST. Every trade `test_same_sector_pairing.py` already
produced (`runs/same_sector_pairing/{random,liquidity}_{primary}_{train,val}.csv`)
carries `symbol_a`/`symbol_b` - enough to map each trade to its own sector
via the SAME `industry_map.csv` sector_map every other script in this
project already builds, and re-cut by sector with no re-simulation. This is
the same "diagnose on already-saved trades" precedent as the 2026-08-15
entry-timing exit-skew diagnostic and the 2026-08-19 per-symbol best/worst
breakdown - genuinely new evidence, zero new simulation risk.

DECISION RULE, stated before looking at any sector's own numbers: a sector
is worth naming as a real candidate only if it clears train AND stays the
same sign with a comparable magnitude on val, for the SAME selection rule -
matching this project's own train-decides/val-confirms discipline, just
applied to a per-sector slice instead of the whole pool. A sector that
looks great on train alone is exactly the kind of number this project has
already been burned by twice (same_sector_pairing itself, market_gate's
driver-smoothing) and is NOT sufficient on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config

MIN_TRADES = 20


def _sector_breakdown(df: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
    df = df[df["net_pnl_pct"].notna()].copy()
    df["sector"] = df["symbol_a"].map(sector_map)
    df = df[df["sector"].notna()]
    df["entry_date"] = pd.to_datetime(df["signal_entry_date"])
    df["bucket"] = df["entry_date"].dt.to_period("W")

    rows = []
    for sector, g in df.groupby("sector"):
        n = len(g)
        if n < MIN_TRADES:
            continue
        bucket_means = g.groupby("bucket")["net_pnl_pct"].mean()
        if len(bucket_means) >= 2:
            t_stat, _ = sstats.ttest_1samp(bucket_means.to_numpy(), 0.0)
        else:
            t_stat = float("nan")
        rows.append({
            "sector": sector, "n_trades": n,
            "mean_net_pct": float(g["net_pnl_pct"].mean()),
            "win_rate_pct": float((g["net_pnl_pct"] > 0).mean() * 100),
            "t_stat": float(t_stat), "n_buckets": len(bucket_means),
        })
    out = pd.DataFrame(rows).sort_values("mean_net_pct", ascending=False)
    return out


def main() -> int:
    pd.set_option("display.width", 160)
    cfg = load_config()

    industry_ref = pd.read_csv(cfg.paths.industry_map)
    sector_map = dict(zip(industry_ref["symbol"].astype(str).str.strip(),
                          industry_ref["industry"].astype(str).str.strip()))

    base = Path(cfg.paths.runs) / "same_sector_pairing"
    all_rows = []
    for variant in ("random", "liquidity"):
        for window in ("train", "val"):
            path = base / f"{variant}_primary_{window}.csv"
            if not path.exists():
                print(f"  missing {path}, skipping")
                continue
            df = pd.read_csv(path)
            breakdown = _sector_breakdown(df, sector_map)
            breakdown.insert(0, "window", window)
            breakdown.insert(0, "variant", variant)
            all_rows.append(breakdown)
            print(f"\n=== {variant} / primary / {window} (sectors with >={MIN_TRADES} trades) ===")
            print(breakdown.to_string(index=False))

    if not all_rows:
        print("no data found")
        return 1

    combined = pd.concat(all_rows, ignore_index=True)

    print("\n=== TRAIN -> VAL CONSISTENCY CHECK (same sector, same variant, same sign both windows) ===")
    for variant in ("random", "liquidity"):
        train = combined[(combined["variant"] == variant) & (combined["window"] == "train")]
        val = combined[(combined["variant"] == variant) & (combined["window"] == "val")]
        merged = train.merge(val, on="sector", suffixes=("_train", "_val"))
        if merged.empty:
            print(f"  {variant}: no sector present in both windows with >={MIN_TRADES} trades")
            continue
        merged["same_sign"] = (merged["mean_net_pct_train"] > 0) == (merged["mean_net_pct_val"] > 0)
        cols = ["sector", "n_trades_train", "mean_net_pct_train", "t_stat_train",
               "n_trades_val", "mean_net_pct_val", "t_stat_val", "same_sign"]
        print(f"\n  {variant}:")
        print(merged[cols].sort_values("mean_net_pct_train", ascending=False).to_string(index=False))

    out_dir = Path(cfg.paths.runs) / "diagnostic_sector_pairing"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "sector_breakdown.csv", index=False)
    print(f"\nWrote {out_dir}")
    print("\nNOT logged to hypothesis_log.csv - descriptive slice of already-run "
         "trades, not a new hypothesis test.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
