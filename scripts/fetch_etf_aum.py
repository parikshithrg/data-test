"""One-time acquisition: fetch and parse AMFI's `average-aum-schemewise`
endpoint into a consolidated table of quarterly scheme-level ETF/Index-Fund
average AUM, across every AMC. NOT a live call inside the deterministic
harness - same category as every other `fetch_*.py` script in this project.

    python scripts/fetch_etf_aum.py
    python scripts/fetch_etf_aum.py --max-fy-id 3   # subset, for testing

ONE INDUSTRY-WIDE CALL PER QUARTER, `MF_ID=0` (all AMCs). Iterates `fyId`
1..21 (confirmed live, 2026-08-24: fyId=1 is the CURRENT financial year and
increases going backward one Indian financial year - April-March - at a
time; fyId=22+ returns HTTP 400, so 21 is the real, confirmed floor) and
`periodId` 1..4 for each, skipping any combination the API itself rejects
(a partial financial year, e.g. the current one, legitimately has fewer
than 4 valid periods). The true calendar quarter for each successful call
comes from the response's own `selectedPeriod` text field, never from
fyId/periodId arithmetic - see `dtest/data/etf_aum.py`'s own module
docstring for why that arithmetic is not reliable on its own.

RATE LIMITING: same courtesy every other fetch script in this project
extends, applied here too even though amfiindia.com is a different host.
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
from dtest.data.etf_aum import SCHEMA, parse_average_aum_response

BASE_URL = "https://www.amfiindia.com/api/average-aum-schemewise"
DELAY_SECONDS = 0.3
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
MAX_FY_ID_CEILING = 30  # a safety bound; the real floor (confirmed 21) is discovered live


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-fy-id", type=int, default=MAX_FY_ID_CEILING,
                    help="stop probing once this many consecutive fyId values fail (default: full history)")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(cfg.paths.etf_aum_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    all_records = []
    seen_quarters: set[str] = set()
    t0 = time.time()
    n_calls, n_ok = 0, 0
    fy_id = 1
    consecutive_fy_failures = 0
    while fy_id <= args.max_fy_id and consecutive_fy_failures < 2:
        fy_had_any_success = False
        for period_id in range(1, 5):
            n_calls += 1
            try:
                r = session.get(BASE_URL, params={
                    "strType": "Categorywise", "fyId": fy_id, "periodId": period_id, "MF_ID": 0,
                }, timeout=30)
            except Exception as e:
                print(f"  fyId={fy_id} periodId={period_id} ERROR: {e}")
                time.sleep(DELAY_SECONDS)
                continue
            if r.status_code != 200:
                time.sleep(DELAY_SECONDS)
                continue
            try:
                payload = r.json()
            except ValueError:
                time.sleep(DELAY_SECONDS)
                continue
            recs = parse_average_aum_response(payload)
            quarter_label = payload.get("selectedPeriod")
            if quarter_label in seen_quarters:
                # fyId/periodId can overlap at the edges (e.g. a not-yet-
                # complete current FY) - dedupe by the real quarter label
                time.sleep(DELAY_SECONDS)
                continue
            seen_quarters.add(quarter_label)
            all_records.extend(recs)
            fy_had_any_success = True
            n_ok += 1
            print(f"  fyId={fy_id} periodId={period_id} ({quarter_label}): {len(recs)} ETF/index-fund rows")
            time.sleep(DELAY_SECONDS)
        consecutive_fy_failures = 0 if fy_had_any_success else consecutive_fy_failures + 1
        fy_id += 1

    elapsed = time.time() - t0
    print(f"\n{n_ok}/{n_calls} calls succeeded, {len(seen_quarters)} distinct quarters, "
         f"{elapsed:.0f}s elapsed")

    if not all_records:
        df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in SCHEMA.items()})
    else:
        df = pd.DataFrame([{
            "amfi_code": r.amfi_code, "scheme_name": r.scheme_name, "amc_name": r.amc_name,
            "category": r.category, "period_end": r.period_end, "filing_date": r.filing_date,
            "average_aum_lakhs": r.average_aum_lakhs,
        } for r in all_records])
        df = df.sort_values(["period_end", "amc_name", "scheme_name"]).reset_index(drop=True)

    out_path = out_dir / "quarterly_aum.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
