"""One-time PILOT probe: how much real SEBI PIT (insider-trading) coverage
does NSE's `corporates-pit` endpoint have across a real sample of this
project's own universe.

    python scripts/probe_insider_trading_coverage.py

Same method as `probe_shareholding_coverage.py` (40-symbol sample from
`industry_map.csv`, seed=42) - see `dtest/data/insider_trading.py`'s
module docstring for the live probes (RELIANCE/TCS/ZEEL/M&M, 2026-08-24)
this pilot extends to a broader, unbiased sample before committing to the
full ~900-symbol fetch.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config
from dtest.data.insider_trading import parse_pit_record

N_SAMPLE = 40
SEED = 42
FROM_DATE = "01-01-2015"
DELAY_SECONDS = 0.3
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _fetch_symbol(session: requests.Session, symbol: str, to_date: str) -> list:
    url = ("https://www.nseindia.com/api/corporates-pit"
          f"?index=equities&symbol={quote(symbol, safe='')}&from_date={FROM_DATE}&to_date={to_date}")
    try:
        r = session.get(url, timeout=20)
        raw = r.json().get("data", [])
    except Exception as e:
        print(f"    {symbol} ERROR: {e}")
        return []
    return [rec for r in raw if (rec := parse_pit_record(r)) is not None]


def main() -> int:
    cfg = load_config()
    industry_ref = pd.read_csv(cfg.paths.industry_map)
    all_symbols = sorted(industry_ref["symbol"].astype(str).str.strip().unique())

    rng = np.random.default_rng(SEED)
    sample = sorted(rng.choice(all_symbols, size=min(N_SAMPLE, len(all_symbols)), replace=False))
    print(f"sampling {len(sample)}/{len(all_symbols)} symbols from industry_map.csv (seed={SEED})")

    session = requests.Session()
    session.headers.update(HEADERS)
    print("warming up session (nseindia.com home page for cookies) ...")
    session.get("https://www.nseindia.com", timeout=10)

    to_date = date.today().strftime("%d-%m-%Y")
    t0 = time.time()
    rows = []
    for i, sym in enumerate(sample):
        recs = _fetch_symbol(session, sym, to_date)
        dates = sorted(r.filing_date for r in recs)
        rows.append({
            "symbol": sym, "n_disclosures": len(recs),
            "earliest": dates[0] if dates else None, "latest": dates[-1] if dates else None,
        })
        time.sleep(DELAY_SECONDS)
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(sample)} symbols probed ({time.time() - t0:.0f}s elapsed)")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    n_with_data = int((df["n_disclosures"] > 0).sum())
    print(f"\n=== COVERAGE, {len(sample)}-symbol sample ===")
    print(f"  symbols with ANY disclosures: {n_with_data}/{len(sample)} ({100 * n_with_data / len(sample):.1f}%)")
    covered = df[df["n_disclosures"] > 0]
    if len(covered):
        print(f"  mean disclosures/covered symbol: {covered['n_disclosures'].mean():.1f}")
        print(f"  median disclosures/covered symbol: {covered['n_disclosures'].median():.1f}")
    print(f"\n  wall time: {elapsed:.1f}s for {len(sample)} symbols ({elapsed / len(sample):.2f}s/symbol)")

    out_dir = Path(cfg.paths.runs) / "probe_insider_trading_coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sample_coverage.csv", index=False)
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
