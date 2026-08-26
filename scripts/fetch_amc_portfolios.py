"""One-time acquisition: fetch real monthly portfolio disclosure files for
SBI Mutual Fund and Axis Mutual Fund (Tier 1 item 4 of the dataset priority
queue - see `dtest/data/amc_portfolios.py`'s own module docstring for the
full scoping story, including why ICICI Prudential and HDFC are NOT covered
here despite being scoped). NOT a live call inside the deterministic
harness - same category as every other `fetch_*.py` script in this project.

    python scripts/fetch_amc_portfolios.py
    python scripts/fetch_amc_portfolios.py --amc sbi    # one AMC only
    python scripts/fetch_amc_portfolios.py --amc axis

Downloads every real workbook into `cfg.paths.amc_portfolios_dir /
"sbi"|"axis"`, skipping any file already on disk under the same generated
local filename (re-running is cheap and idempotent). Writes ONE combined
filing-level manifest CSV across both AMCs - RAW FILES ONLY, no
holdings-table parsing (see the module docstring for why that is real,
separate follow-on work).

AXIS IS A MUCH LARGER FETCH THAN SBI, STATED PLAINLY: ~3,690 real files
(32 consolidated + ~3,660 per-scheme, real average ~75KB/file measured
live before committing to the full fetch) versus SBI's 68 - expect
several minutes and roughly 250-300MB on disk, not a quick job.
"""

from __future__ import annotations

import argparse
import html as html_module
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
    AXIS_AMC_NAME,
    AXIS_DOCUMENTS_BODY,
    AXIS_DOCUMENTS_URL,
    AXIS_TOKEN_URL,
    SBI_AMC_NAME,
    SBI_PORTFOLIO_SHEETS_URL,
    SCHEMA,
    parse_axis_documents_response,
    parse_portfolio_sheets_response,
)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY_SECONDS = 0.2
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"|?*\\/]')


def _sbi_filings(session: requests.Session) -> list:
    print(f"fetching SBI filing list from {SBI_PORTFOLIO_SHEETS_URL} ...")
    r = session.post(
        SBI_PORTFOLIO_SHEETS_URL,
        headers={"Content-Type": "application/json;charset=utf-8"},
        data=json.dumps({"FundId": 0, "PSYear": "", "PSMonth": "", "PSFrequency": "Monthly"}),
        timeout=30,
    )
    r.raise_for_status()
    filings = parse_portfolio_sheets_response(r.text, amc_name=SBI_AMC_NAME)
    print(f"  found {len(filings)} distinct filings")
    return filings


def _axis_filings(session: requests.Session) -> list:
    print(f"fetching Axis token from {AXIS_TOKEN_URL} ...")
    tok_resp = session.post(AXIS_TOKEN_URL, headers={"Content-Type": "application/json"},
                             json={}, timeout=30)
    tok_resp.raise_for_status()
    token = tok_resp.json()["data"]["token"]
    print(f"fetching Axis filing list from {AXIS_DOCUMENTS_URL} ...")
    r = session.post(AXIS_DOCUMENTS_URL,
                      headers={"Content-Type": "application/json", "Authorization": token},
                      json=AXIS_DOCUMENTS_BODY, timeout=60)
    r.raise_for_status()
    filings = parse_axis_documents_response(r.json(), amc_name=AXIS_AMC_NAME)
    print(f"  found {len(filings)} distinct Monthly Portfolio filings")
    return filings


def _sbi_local_filename(filing) -> str:
    """SBI's own URL basename is already unique per filing (see module
    docstring) - just strip the `sfvrsn` cache-busting token and %-decode."""
    path = urlparse(filing.url).path
    return _UNSAFE_FILENAME_RE.sub("_", unquote(Path(path).name))


def _axis_local_filename(filing) -> str:
    """Axis's own URL basenames are NOT guaranteed unique across schemes
    (they're CMS-generated slugs under numeric content-tree folders, not
    guaranteed collision-free once flattened into one directory) - build a
    filename from the parsed (scheme, period_end) instead, which IS unique
    by construction (one filing per scheme per month, or one per month for
    the consolidated series)."""
    ext = Path(urlparse(filing.url).path).suffix or ".xlsx"
    date_str = filing.period_end.strftime("%Y-%m-%d") if filing.period_end is not None else "unknown-date"
    scheme_part = filing.scheme_name if filing.scheme_name else "consolidated"
    name = f"{date_str}_{scheme_part}{ext}"
    return _UNSAFE_FILENAME_RE.sub("_", name)


AMCS = {
    "sbi": (_sbi_filings, _sbi_local_filename),
    "axis": (_axis_filings, _axis_local_filename),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amc", choices=sorted(AMCS), default=None,
                     help="fetch only this AMC (default: all)")
    args = ap.parse_args()
    amcs_to_run = [args.amc] if args.amc else sorted(AMCS)

    cfg = load_config()
    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows = []
    for amc_key in amcs_to_run:
        list_fn, filename_fn = AMCS[amc_key]
        out_dir = Path(cfg.paths.amc_portfolios_dir) / amc_key
        out_dir.mkdir(parents=True, exist_ok=True)

        filings = list_fn(session)
        if not filings:
            print(f"  FAILED for {amc_key} - zero filings parsed, the endpoint's response shape may have changed")
            continue

        n_downloaded, n_skipped, n_failed, n_unparsed_skipped = 0, 0, 0, 0
        for i, f in enumerate(filings):
            if f.period_end is None:
                n_unparsed_skipped += 1
                continue
            local_name = filename_fn(f)
            local_path = out_dir / local_name
            if not local_path.exists():
                try:
                    resp = session.get(f.url, timeout=60)
                    resp.raise_for_status()
                    local_path.write_bytes(resp.content)
                    n_downloaded += 1
                    if n_downloaded % 100 == 0:
                        print(f"  [{amc_key}] {n_downloaded} downloaded so far ({i+1}/{len(filings)} filings processed)")
                except Exception as e:
                    print(f"  FAILED [{amc_key}] {f.title}: {e}")
                    n_failed += 1
                    time.sleep(DELAY_SECONDS)
                    continue
                time.sleep(DELAY_SECONDS)
            else:
                n_skipped += 1

            all_rows.append({
                "amc_name": f.amc_name, "title": f.title, "scheme_name": f.scheme_name,
                "period_end": f.period_end, "filing_date": f.filing_date,
                "url": f.url, "local_filename": f"{amc_key}/{local_name}",
            })

        print(f"[{amc_key}] {n_downloaded} downloaded, {n_skipped} already on disk, "
              f"{n_failed} failed, {n_unparsed_skipped} skipped (unparseable title/date)")

    if not all_rows:
        print("No filings recorded across any AMC - nothing to write.")
        return 1

    df = pd.DataFrame(all_rows)
    df["title"] = df["title"].map(lambda s: html_module.unescape(s) if isinstance(s, str) else s)
    df = df.astype({k: v for k, v in SCHEMA.items() if k in df.columns})
    df = df.sort_values(["amc_name", "period_end"], na_position="last").reset_index(drop=True)
    manifest_path = Path(cfg.paths.amc_portfolios_dir) / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"\nWrote combined manifest ({len(df)} rows, {df['amc_name'].nunique()} AMCs) to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
