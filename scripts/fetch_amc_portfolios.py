"""One-time acquisition: fetch SBI Mutual Fund's real monthly portfolio
disclosure files (Tier 1 item 4 of the dataset priority queue, SBI only -
see `dtest/data/amc_portfolios.py`'s own module docstring for the full
scoping story and why this AMC, unlike the ones checked 2026-08-24, needed
no browser automation). NOT a live call inside the deterministic harness -
same category as every other `fetch_*.py` script in this project.

    python scripts/fetch_amc_portfolios.py

Downloads every real workbook (68 as of 2026-08-26) into
`cfg.paths.amc_portfolios_dir / "sbi"`, skipping any file already on disk
byte-identically named (re-running is cheap and idempotent - a filing's
`sfvrsn` version token in its URL changes if SBI ever republishes the same
month, which would naturally produce a new local filename rather than
silently overwrite). Writes a filing-level manifest CSV alongside the raw
files - RAW FILES ONLY, no holdings-table parsing (see the module docstring
for why that is real, separate follow-on work).
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config
from dtest.data.amc_portfolios import (
    AMC_NAME,
    SBI_PORTFOLIO_SHEETS_URL,
    SCHEMA,
    parse_portfolio_sheets_response,
)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY_SECONDS = 0.3
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"|?*]')


def _local_filename(url: str) -> str:
    """The URL's own basename, `sfvrsn` query stripped (a cache-busting
    version token, not part of the real filename) and %-decoded."""
    path = urlparse(url).path
    return _UNSAFE_FILENAME_RE.sub("_", unquote(Path(path).name))


def main() -> int:
    cfg = load_config()
    out_dir = Path(cfg.paths.amc_portfolios_dir) / "sbi"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"fetching filing list from {SBI_PORTFOLIO_SHEETS_URL} ...")
    r = session.post(
        SBI_PORTFOLIO_SHEETS_URL,
        headers={"Content-Type": "application/json;charset=utf-8"},
        data=json.dumps({"FundId": 0, "PSYear": "", "PSMonth": "", "PSFrequency": "Monthly"}),
        timeout=30,
    )
    r.raise_for_status()
    filings = parse_portfolio_sheets_response(r.text, amc_name=AMC_NAME)
    print(f"  found {len(filings)} distinct filings")
    if not filings:
        print("  FAILED - zero filings parsed, the endpoint's response shape may have changed")
        return 1

    rows = []
    n_downloaded, n_skipped, n_failed = 0, 0, 0
    for f in filings:
        local_name = _local_filename(f.url)
        local_path = out_dir / local_name
        if not local_path.exists():
            try:
                resp = session.get(f.url, timeout=60)
                resp.raise_for_status()
                local_path.write_bytes(resp.content)
                n_downloaded += 1
                print(f"  downloaded {local_name} ({len(resp.content)} bytes)")
            except Exception as e:
                print(f"  FAILED {f.title}: {e}")
                n_failed += 1
                time.sleep(DELAY_SECONDS)
                continue
            time.sleep(DELAY_SECONDS)
        else:
            n_skipped += 1

        rows.append({
            "amc_name": f.amc_name, "title": f.title, "period_end": f.period_end,
            "filing_date": f.filing_date, "url": f.url, "local_filename": local_name,
        })

    df = pd.DataFrame(rows).astype({k: v for k, v in SCHEMA.items() if k in rows[0]} if rows else SCHEMA)
    df = df.sort_values("period_end", na_position="last").reset_index(drop=True)
    manifest_path = Path(cfg.paths.amc_portfolios_dir) / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    n_unparsed = int(df["period_end"].isna().sum())
    print(f"\n{n_downloaded} downloaded, {n_skipped} already on disk, {n_failed} failed, "
          f"{n_unparsed}/{len(df)} filings had an unparseable title/date")
    print(f"Wrote manifest ({len(df)} rows) to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
