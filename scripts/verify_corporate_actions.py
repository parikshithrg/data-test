"""Verify, on real data, that NSE restates prev_close on corporate actions.

The corporate_actions module rests entirely on that convention, so it is
measured rather than assumed. Run this on whatever years the ingest has
completed so far.

    python scripts/verify_corporate_actions.py --years 2004 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config, set_seeds
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.corporate_actions import (
    detect_actions, daily_returns, previous_traded_close, verify,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    pd.set_option("display.width", 150)

    cfg = load_config()
    set_seeds()
    cache = Path(cfg.paths.artifacts) / "bhav"

    print("Building parsed store ...")
    summary = build_store(cache, years=args.years)
    print(summary.to_string(index=False))

    long_df = load_long(cache, columns=["date", "symbol", "close", "prev_close"])
    long_df = long_df[long_df["date"].dt.year.isin(args.years)]
    close = to_panel(long_df, "close")
    prev = to_panel(long_df, "prev_close")
    print(f"\npanel: {close.shape[0]} sessions x {close.shape[1]} symbols")

    print("\n=== EVIDENCE ===")
    evidence = verify(close, prev)
    print(evidence.to_string(index=False))

    rep = detect_actions(close, prev)

    if not rep.rejected_dates.empty:
        print(f"\n=== DATES REJECTED AS MASS EVENTS: {len(rep.rejected_dates)} ===")
        print("(a real action hits a few symbols; a whole-market flag is a data artifact)")
        r = rep.rejected_dates.copy()
        r["date"] = r["date"].dt.date
        print(r.head(10).to_string(index=False))

    print(f"\n=== CREDIBLE ACTIONS: {rep.n_actions} ===")
    if rep.n_actions:
        labelled = rep.label()
        print("\nby likely type:")
        print(labelled["likely"].value_counts().to_string())

        print("\n=== RAW vs CORRECTED RETURN on action days ===")
        adj = daily_returns(close, prev, rep)
        prior = previous_traded_close(close)
        raw = close / prior - 1.0
        rows = []
        for _, a in labelled.head(12).iterrows():
            d, s = a["date"], a["symbol"]
            rows.append({
                "symbol": s, "date": d.date(), "factor": round(a["factor"], 4),
                "likely": a["likely"],
                "raw_return_%": round(float(raw.at[d, s]) * 100, 2),
                "corrected_%": round(float(adj.at[d, s]) * 100, 2),
            })
        print(pd.DataFrame(rows).to_string(index=False))

    out = Path(cfg.paths.runs) / "corporate_actions"
    out.mkdir(parents=True, exist_ok=True)
    rep.label().to_csv(out / "actions.csv", index=False)
    rep.rejected_dates.to_csv(out / "rejected_dates.csv", index=False)
    evidence.to_csv(out / "evidence.csv", index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
