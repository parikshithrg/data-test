"""One-time acquisition: fetch and parse NSE's `corporate-announcements`
endpoint into a consolidated table of material corporate announcements
(M&A, contract wins, dividend/bonus/buyback, and other categories in
`dtest.data.corporate_announcements.RELEVANT_CATEGORIES`). NOT a live call
inside the deterministic harness - same category as every other
`fetch_*.py` script in this project.

    python scripts/fetch_corporate_announcements.py
    python scripts/fetch_corporate_announcements.py --start 01-01-2024 --end 31-12-2024   # subset

MONTHLY WINDOWED, INDUSTRY-WIDE (no `symbol` filter) - confirmed live,
2026-08-24, that BOTH `symbol` and date-range filters genuinely work on
this endpoint (unlike `shareholding.py`'s master endpoint or
`credit_ratings.py`'s feed, where at least one was unreliable), so a
per-symbol loop is unnecessary - one date-windowed call per month covers
every symbol at once, the same efficient shape
`fetch_index_reconstitution.py`/`fetch_credit_ratings.py` already use.

REAL FLOOR IS ~2004 for a sampled large-cap (RELIANCE) - matches this
project's own `primary` split's train_start on structural grounds, not
by coincidence checked here.

`seq_id`-DEDUPED - this feed's own real unique record identifier, same
role `AppID` plays in `credit_ratings.py`. Deduped across AND within
windows in case of any month-boundary overlap.

RATE LIMITING: same courtesy every other NSE fetch script in this project
extends. Peak months are genuinely large (May 2024 alone: 22,088 raw
records, 2,528 kept after the category filter) - expect this fetch to run
longer than the smaller sources built earlier this session.
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
from dtest.data.corporate_announcements import SCHEMA, parse_announcement_record

BASE_URL = "https://www.nseindia.com/api/corporate-announcements"
DELAY_SECONDS = 0.4
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DEFAULT_START = pd.Timestamp("2004-01-01")


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
    ap.add_argument("--start", default=None, help="DD-MM-YYYY, default 01-01-2004 (the real floor)")
    ap.add_argument("--end", default=None, help="DD-MM-YYYY, default today")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(cfg.paths.corporate_announcements_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(pd.to_datetime(args.start, dayfirst=True)) if args.start else DEFAULT_START
    end = pd.Timestamp(pd.to_datetime(args.end, dayfirst=True)) if args.end else pd.Timestamp.today()

    session = requests.Session()
    session.headers.update(HEADERS)
    print("warming up session (nseindia.com home page for cookies) ...")
    session.get("https://www.nseindia.com", timeout=10)

    windows = _month_windows(start, end)
    all_records: dict[str, object] = {}
    t0 = time.time()
    for i, (win_start, win_end) in enumerate(windows):
        params = {
            "index": "equities",
            "from_date": win_start.strftime("%d-%m-%Y"),
            "to_date": win_end.strftime("%d-%m-%Y"),
        }
        try:
            r = session.get(BASE_URL, params=params, timeout=40)
            raw = r.json() if r.status_code == 200 else []
        except Exception as e:
            print(f"  [{i + 1}/{len(windows)}] {win_start.date()}..{win_end.date()} ERROR: {e}")
            raw = []

        n_new = 0
        for rec in raw:
            parsed = parse_announcement_record(rec)
            if parsed is not None and parsed.seq_id not in all_records:
                all_records[parsed.seq_id] = parsed
                n_new += 1
        elapsed = time.time() - t0
        print(f"  [{i + 1}/{len(windows)}] {win_start.date()}..{win_end.date()}: "
             f"{len(raw)} raw, {n_new} new kept ({elapsed:.0f}s elapsed, running total {len(all_records)})")
        time.sleep(DELAY_SECONDS)

    if not all_records:
        df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in SCHEMA.items()})
    else:
        df = pd.DataFrame([{
            "symbol": r.symbol, "filing_date": r.filing_date, "category": r.category,
            "company_name": r.company_name, "isin": r.isin, "industry": r.industry,
            "description": r.description, "seq_id": r.seq_id,
        } for r in all_records.values()])
        df = df.sort_values(["filing_date", "symbol"]).reset_index(drop=True)

    out_path = out_dir / "announcements.csv"
    df.to_csv(out_path, index=False)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. {len(df)} announcements kept.")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
