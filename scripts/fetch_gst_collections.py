"""ONE-TIME DATA ACQUISITION, not part of the deterministic backtest
harness - same category as `fetch_index_reconstitution.py`. Downloads
GSTN's own "9 Years of GST" statistical report PDF and parses the monthly
collection table `dtest/data/gst_collections.py` documents, saving it to
`cfg.paths.gst_collections_dir`.

    python scripts/fetch_gst_collections.py

Run once. Re-run only if GSTN republishes a newer edition of the report -
the URL discovered live, 2026-08-25, has no version/date in it, so a
future run may return updated content at the same address.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config
from dtest.data.gst_collections import SCHEMA, parse_9years_report_pdf

REPORT_URL = "https://tutorial.gst.gov.in/offlineutilities/gst_statistics/9YearsReport.pdf"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def main() -> int:
    cfg = load_config()
    out_dir = Path(cfg.paths.gst_collections_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"fetching {REPORT_URL} ...")
    r = requests.get(REPORT_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    print(f"  {len(r.content)} bytes")

    records = parse_9years_report_pdf(r.content)
    rows = [{"date": rec.date, **rec.values} for rec in records]
    df = pd.DataFrame(rows)
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[list(SCHEMA)].sort_values("date", kind="stable").reset_index(drop=True)

    # Real cross-check, not a blocking assertion: each fiscal year's monthly
    # `total_rs_crore` values should sum to that FY's own printed grand
    # total on the source PDF's page 16 summary table (hand-transcribed
    # here from that page, confirmed live 2026-08-25 - see the module
    # docstring's own docstring for the page reference).
    fy_totals = {
        "2017-18": 740648, "2018-19": 1177369, "2019-20": 1222116,
        "2020-21": 1136801, "2021-22": 1488227, "2022-23": 1807680,
        "2023-24": 2018249, "2024-25": 2208861, "2025-26": 2331819,
    }
    df["fy"] = df["date"].apply(
        lambda d: f"{d.year}-{str(d.year + 1)[-2:]}" if d.month >= 4 else f"{d.year - 1}-{str(d.year)[-2:]}"
    )
    for fy, expected in fy_totals.items():
        actual = df.loc[df["fy"] == fy, "total_rs_crore"].sum()
        if actual and abs(actual - expected) > 5:  # crore-level rounding only
            print(f"  WARNING: FY{fy} monthly sum {actual:.0f} != printed FY total {expected} (rs crore)")
    df = df.drop(columns=["fy"])

    out_path = out_dir / "GST_COLLECTIONS_MONTHLY.csv"
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path}: {df['date'].min().date()} .. {df['date'].max().date()} ({len(df)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
