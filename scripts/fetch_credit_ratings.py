"""One-time acquisition: fetch and parse NSE's `corporate-credit-rating`
endpoint into a consolidated table of multi-agency credit rating actions.
NOT a live call inside the deterministic harness - same category as
every other `fetch_*.py` script in this project.

    python scripts/fetch_credit_ratings.py
    python scripts/fetch_credit_ratings.py --start 01-01-2024 --end 31-12-2024   # subset, for testing

MONTHLY WINDOWED, DATE-ONLY QUERIES - never combined with `symbol`.
Confirmed live, 2026-08-24: `symbol`+date-range together reliably 502s on
this endpoint; a date-only range works every time. `symbol` resolution
therefore happens client-side, via `dtest.data.credit_ratings`'s
name-match against this project's own `index_reconstitution.py` events -
loaded once, reused for every record.

REAL FLOOR IS ~APRIL 2023 - starts there by default (see
`dtest/data/credit_ratings.py`'s own module docstring for the live sweep
that confirmed this; earlier months returned 0 or a handful of records in
every year checked back to 2010).

`AppID`-DEDUPED - a real API-side duplication artifact was found live (one
April-2023 window returned 158,705 rows collapsing to 18 distinct
`AppID`s) - every window is deduped on `AppID` before being kept, and the
final table is deduped again across windows in case of overlap.

RATE LIMITING: same courtesy every other NSE fetch script in this project
extends.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config
from dtest.data.credit_ratings import SCHEMA, build_name_to_symbol_lookup, parse_credit_rating_record

BASE_URL = "https://www.nseindia.com/api/corporate-credit-rating"
DELAY_SECONDS = 0.4
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DEFAULT_START = pd.Timestamp("2023-04-01")


def _month_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows = []
    cur = pd.Timestamp(start.year, start.month, 1)
    while cur <= end:
        window_end = min(cur + pd.offsets.MonthEnd(0), end)
        windows.append((max(cur, start), window_end))
        cur = cur + pd.offsets.MonthBegin(1)
    return windows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="DD-MM-YYYY, default 01-04-2023 (the real floor)")
    ap.add_argument("--end", default=None, help="DD-MM-YYYY, default today")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(cfg.paths.credit_ratings_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(pd.to_datetime(args.start, dayfirst=True)) if args.start else DEFAULT_START
    end = pd.Timestamp(pd.to_datetime(args.end, dayfirst=True)) if args.end else pd.Timestamp.today()

    print("loading index-reconstitution events for name->symbol resolution ...")
    recon_path = Path(cfg.paths.index_reconstitution_dir) / "events.csv"
    recon_events = pd.read_csv(recon_path, usecols=["company_name", "symbol"])
    name_to_symbol = build_name_to_symbol_lookup(recon_events)
    print(f"  {len(name_to_symbol)} distinct company names resolvable")

    session = requests.Session()
    session.headers.update(HEADERS)
    print("warming up session (nseindia.com home page for cookies) ...")
    session.get("https://www.nseindia.com", timeout=10)

    windows = _month_windows(start, end)
    all_records: dict[str, object] = {}   # keyed by app_id, dedupes across AND within windows
    t0 = time.time()
    for i, (win_start, win_end) in enumerate(windows):
        params = {
            "index": "equities",
            "from_date": win_start.strftime("%d-%m-%Y"),
            "to_date": win_end.strftime("%d-%m-%Y"),
        }
        try:
            r = session.get(BASE_URL, params=params, timeout=30)
            raw = r.json() if r.status_code == 200 else []
        except Exception as e:
            print(f"  [{i + 1}/{len(windows)}] {win_start.date()}..{win_end.date()} ERROR: {e}")
            raw = []

        n_new = 0
        for rec in raw:
            parsed = parse_credit_rating_record(rec, name_to_symbol)
            if parsed is not None and parsed.app_id not in all_records:
                all_records[parsed.app_id] = parsed
                n_new += 1
        print(f"  [{i + 1}/{len(windows)}] {win_start.date()}..{win_end.date()}: "
             f"{len(raw)} raw, {n_new} new distinct AppIDs (running total {len(all_records)})")
        time.sleep(DELAY_SECONDS)

    if not all_records:
        df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in SCHEMA.items()})
    else:
        df = pd.DataFrame([{
            "app_id": r.app_id, "symbol": r.symbol, "company_name": r.company_name, "isin": r.isin,
            "agency": r.agency, "rating": r.rating, "rating_earlier": r.rating_earlier,
            "action": r.action, "outlook": r.outlook, "date_of_rating": r.date_of_rating,
            "filing_date": r.filing_date,
        } for r in all_records.values()])
        df = df.sort_values(["filing_date", "company_name"]).reset_index(drop=True)

    out_path = out_dir / "rating_actions.csv"
    df.to_csv(out_path, index=False)
    n_resolved = df["symbol"].notna().sum()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. {len(df)} distinct rating actions, "
         f"{n_resolved} ({100 * n_resolved / max(len(df), 1):.1f}%) resolved to a real symbol.")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
