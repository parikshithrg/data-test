"""ONE-TIME DATA ACQUISITION, not part of the deterministic backtest
harness - same category as `fetch_macro_stress_series.py`. Fetches the 2
real FRED series `dtest/data/gsec_yields.py` documents (India 10Y G-Sec
yield, India 3M interbank rate), saves them as `{TENOR}_MONTHLY.csv` in
`cfg.paths.gsec_dir` - real, direct CSV downloads, no HTML/JS involved.

    python scripts/fetch_gsec_yields.py

Run once. Re-run only to refresh with more recent data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config
from dtest.data.gsec_yields import FRED_CSV_URL, FRED_SERIES

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def main() -> int:
    cfg = load_config()
    out_dir = Path(cfg.paths.gsec_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    for tenor, series_id in FRED_SERIES.items():
        print(f"fetching {tenor} ({series_id}) ...")
        r = session.get(FRED_CSV_URL.format(series_id=series_id), timeout=20)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            print(f"  FAILED - empty response for {series_id}")
            continue
        df = pd.DataFrame(
            [line.split(",") for line in lines[1:]],
            columns=["date", "yield_pct"],
        )
        df["date"] = pd.to_datetime(df["date"])
        df["yield_pct"] = pd.to_numeric(df["yield_pct"], errors="coerce")
        df = df.dropna(subset=["yield_pct"])
        out_path = out_dir / f"{tenor}_MONTHLY.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path}: {df['date'].iloc[0].date()} .. {df['date'].iloc[-1].date()} ({len(df)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
