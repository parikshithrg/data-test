"""One-time PILOT probe (not a fetch/cache pipeline, not part of the
deterministic harness): how much real quarterly-results coverage does
NSE's `corporates-financial-results` endpoint actually have across a real
sample of this project's own universe, and how long does it take.

    python scripts/probe_financial_results_coverage.py

WHY THIS EXISTS. 2026-08-20 spot checks on 4 symbols (RELIANCE, GESHIP,
PUNJLLOYD, RUSHIL, BINDALAGRO) confirmed the endpoint is real and gives
genuine filing-date metadata plus parseable P&L line items (HTML for older
filings, XBRL for newer ones) - but 4 hand-picked symbols is not evidence
about the ~200-500 name universe this project actually needs. User asked
to scope this properly (coverage %, build time, balance-sheet availability)
before picking any specific fundamentals hypothesis or building a real
fetch/cache module.

BALANCE SHEET: already checked separately (2026-08-20) - the quarterly
filing HTML only carries P&L + segment reporting, no balance sheet (no
debt/cash/total assets). That is a real, structural limitation of
QUARTERLY disclosures under SEBI LODR, not a fetch bug - a balance-sheet-
dependent factor (P/B, debt/equity) would need ANNUAL filings specifically,
a separate, unconfirmed source. Not investigated further in this probe.

METHOD: sample symbols from `industry_map.csv` (this project's existing
sector-mapping reference, not a point-in-time universe - fine for a
coverage PILOT, not something a real hypothesis test would use for
eligibility). For each symbol, query the metadata endpoint across THREE
windows (2004-2010, 2010-2018, 2018-2026) since a single wide-range query
was observed truncating to ~120-130 records (2026-08-20 finding) - windowed
queries are needed to see true full history, same lesson the fno_oi/
fno_price stitchers already learned about NSE's own bhavcopy archives.
Small delay between requests - a real, if unofficial, endpoint, not
something to hammer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config

N_SAMPLE = 40
SEED = 42
WINDOWS = [("01-01-2004", "31-12-2010"), ("01-01-2010", "31-12-2018"), ("01-01-2018", "20-08-2026")]
DELAY_SECONDS = 0.4

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _fetch_symbol(session: requests.Session, symbol: str) -> list[dict]:
    all_records = []
    for frm, to in WINDOWS:
        url = (
            "https://www.nseindia.com/api/corporates-financial-results"
            f"?index=equities&period=Quarterly&fo_sec=false&symbol={symbol}"
            f"&from_date={frm}&to_date={to}"
        )
        try:
            r = session.get(url, timeout=15)
            data = r.json()
            if isinstance(data, list):
                all_records.extend(data)
        except Exception as e:
            print(f"    {symbol} [{frm}..{to}] ERROR: {e}")
        time.sleep(DELAY_SECONDS)
    return all_records


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

    t0 = time.time()
    rows = []
    for i, sym in enumerate(sample):
        records = _fetch_symbol(session, sym)
        # de-dup on seqNumber - windows can overlap at their boundary dates
        seen = {}
        for rec in records:
            seen[rec.get("seqNumber")] = rec
        records = list(seen.values())
        to_dates = sorted(r["toDate"] for r in records if r.get("toDate"))
        rows.append({
            "symbol": sym,
            "n_filings": len(records),
            "earliest_period_end": to_dates[0] if to_dates else None,
            "latest_period_end": to_dates[-1] if to_dates else None,
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  ... {i + 1}/{len(sample)} symbols probed ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)

    n_with_data = int((df["n_filings"] > 0).sum())
    print(f"\n=== COVERAGE, {len(sample)}-symbol sample ===")
    print(f"  symbols with ANY filings found: {n_with_data}/{len(sample)} "
         f"({100 * n_with_data / len(sample):.1f}%)")
    print(f"  mean filings per covered symbol: {df.loc[df['n_filings'] > 0, 'n_filings'].mean():.1f}")
    print(f"  median filings per covered symbol: {df.loc[df['n_filings'] > 0, 'n_filings'].median():.1f}")
    covered = df[df["n_filings"] > 0]
    if len(covered):
        earliest_years = pd.to_datetime(covered["earliest_period_end"], format="%d-%b-%Y", errors="coerce").dt.year
        print("\n  earliest period-end year distribution (covered symbols):")
        print(earliest_years.value_counts().sort_index().to_string())

    print(f"\n  wall time: {elapsed:.1f}s for {len(sample)} symbols x {len(WINDOWS)} windows "
         f"({elapsed / len(sample):.2f}s/symbol)")
    full_universe_estimate_min = (elapsed / len(sample)) * 300 / 60
    print(f"  extrapolated to a ~300-symbol universe: ~{full_universe_estimate_min:.1f} min "
         f"(metadata only, not the per-filing detail pages)")

    out_dir = Path(cfg.paths.runs) / "probe_financial_results_coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sample_coverage.csv", index=False)
    print(f"\nWrote {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
