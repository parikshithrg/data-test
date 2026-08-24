"""One-time acquisition: fetch and extract full text of earnings-call /
analyst-meet transcripts from NSE's `corporate-announcements` endpoint (see
`dtest.data.earnings_transcripts` module docstring for the real source and
filter logic). NOT a live call inside the deterministic harness.

    python scripts/fetch_earnings_transcripts.py
    python scripts/fetch_earnings_transcripts.py --start 01-01-2022 --end 31-12-2022   # subset

TWO STAGES, deliberately separate:
  1. Metadata sweep - monthly-windowed JSON calls, 2004-2026 (~272 windows,
     same shape as `fetch_corporate_announcements.py`), to find every real
     transcript filing (confirmed live, 2026-08-24: 19,382 across full
     history, ~90% concentrated 2022-2026 - a real regulatory-adoption
     curve, see the module docstring).
  2. PDF fetch + text extraction, ONE HTTP GET + PyMuPDF parse per filing.
     Benchmarked live at ~0.46s/doc sequential - at 19,382 docs that is
     2.5-4h serial, so this stage uses a small thread pool (PDF
     download+parse is I/O-bound, not CPU-bound, and nsearchives.nseindia.com
     is a separate archive host from the www.nseindia.com JSON API stage 1
     already rate-limits against).

RESUMABLE: stage 2 skips any `seq_id` whose text file already exists on
disk, so an interrupted run costs nothing to resume - same convention as
`ingest_bhavcopy.py`.

STORAGE: one `.txt` file per filing under `<earnings_transcripts_dir>/text/`
(named by a filesystem-safe `seq_id`), not one giant CSV column - the real
corpus is ~1GB of text (confirmed via live benchmark: ~55KB average per
transcript), an order of magnitude larger than any other single-CSV source
this project has built. `transcripts.csv` carries only metadata + a
`text_path` pointer.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config
from dtest.data.earnings_transcripts import SCHEMA, extract_transcript_text, parse_transcript_filing

BASE_URL = "https://www.nseindia.com/api/corporate-announcements"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DEFAULT_START = pd.Timestamp("2004-01-01")
DELAY_SECONDS = 0.3
PDF_WORKERS = 8
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _month_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows = []
    cur = pd.Timestamp(start.year, start.month, 1)
    while cur <= end:
        window_end = min(cur + pd.offsets.MonthEnd(0), end)
        windows.append((max(cur, start), window_end))
        cur = cur + pd.offsets.MonthBegin(1)
    return windows


def _safe_filename(seq_id: str) -> str:
    return _UNSAFE_CHARS.sub("_", seq_id)[:150] + ".txt"


def _sweep_metadata(session: requests.Session, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    windows = _month_windows(start, end)
    filings: dict[str, object] = {}
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
            parsed = parse_transcript_filing(rec)
            if parsed is not None and parsed.seq_id not in filings:
                filings[parsed.seq_id] = parsed
                n_new += 1
        elapsed = time.time() - t0
        print(f"  [{i + 1}/{len(windows)}] {win_start.date()}..{win_end.date()}: "
              f"{len(raw)} raw, {n_new} new transcripts ({elapsed:.0f}s elapsed, running total {len(filings)})")
        time.sleep(DELAY_SECONDS)
    return filings


def _fetch_one(session: requests.Session, seq_id: str, filing, text_dir: Path) -> tuple[str, int, str] | None:
    out_path = text_dir / _safe_filename(seq_id)
    if out_path.exists():
        existing_text = out_path.read_text(encoding="utf-8", errors="ignore")
        return seq_id, len(existing_text), out_path.name
    try:
        r = session.get(filing.source_url, timeout=30)
        if r.status_code != 200 or not r.content:
            return None
        text = extract_transcript_text(r.content)
        if not text.strip():
            return None
        out_path.write_text(text, encoding="utf-8")
        return seq_id, len(text), out_path.name
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="DD-MM-YYYY, default 01-01-2004")
    ap.add_argument("--end", default=None, help="DD-MM-YYYY, default today")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(cfg.paths.earnings_transcripts_dir)
    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(pd.to_datetime(args.start, dayfirst=True)) if args.start else DEFAULT_START
    end = pd.Timestamp(pd.to_datetime(args.end, dayfirst=True)) if args.end else pd.Timestamp.today()

    session = requests.Session()
    session.headers.update(HEADERS)
    print("warming up session (nseindia.com home page for cookies) ...")
    session.get("https://www.nseindia.com", timeout=10)

    print("stage 1: metadata sweep ...")
    filings = _sweep_metadata(session, start, end)
    print(f"stage 1 done: {len(filings)} candidate transcript filings")

    print(f"stage 2: PDF fetch + extract, {PDF_WORKERS} workers ...")
    t0 = time.time()
    rows = []
    n_done = 0
    n_failed = 0
    dl_session = requests.Session()
    dl_session.headers.update(HEADERS)
    with ThreadPoolExecutor(max_workers=PDF_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, dl_session, seq_id, filing, text_dir): (seq_id, filing)
                   for seq_id, filing in filings.items()}
        for fut in as_completed(futures):
            seq_id, filing = futures[fut]
            result = fut.result()
            n_done += 1
            if result is None:
                n_failed += 1
            else:
                _, char_count, fname = result
                rows.append({
                    "symbol": filing.symbol, "filing_date": filing.filing_date,
                    "company_name": filing.company_name, "isin": filing.isin,
                    "industry": filing.industry, "seq_id": filing.seq_id,
                    "source_url": filing.source_url, "text_path": f"text/{fname}",
                    "char_count": char_count,
                })
            if n_done % 250 == 0 or n_done == len(filings):
                elapsed = time.time() - t0
                print(f"  {n_done}/{len(filings)} processed ({n_failed} failed, {elapsed:.0f}s elapsed)")

    if not rows:
        df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in SCHEMA.items()})
    else:
        df = pd.DataFrame(rows)
        df = df.sort_values(["filing_date", "symbol"]).reset_index(drop=True)

    out_path = out_dir / "transcripts.csv"
    df.to_csv(out_path, index=False)
    elapsed = time.time() - t0
    print(f"\nStage 2 done in {elapsed:.0f}s. {len(df)} transcripts extracted, {n_failed} failed.")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
