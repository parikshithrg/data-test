"""One-time PILOT probe (not a fetch/cache pipeline): how much real
shareholding-pattern coverage does NSE's `corporate-share-holdings-master`
endpoint have across a real sample of this project's own universe, and how
far back does it actually go.

    python scripts/probe_shareholding_coverage.py

WHY THIS EXISTS. 2026-08-24 ad-hoc spot checks (RELIANCE, TCS, ITC,
TATASTEEL, INFY, HDFCBANK, WIPRO, SBIN, ONGC, SUNPHARMA, MARUTI, ZEEL,
SUZLON, JPPOWER) confirmed the endpoint is real, gives clean per-category
percentages via a linked XBRL detail doc (promoter/MF/insurance/FII-FPI-I/
FII-FPI-II/DII-total/public, plus real promoter-pledge % on names that have
it, e.g. JPPOWER 72.99%) - but EVERY one of those 14 hand-picked large/well-
known names showed the exact same earliest period-end, 30-SEP-2021, with
`from_date`/`to_date` params returning [] for anything before that. This
pilot exists to confirm that ~2021 floor is a real, structural property of
NSE's own XBRL-based shareholding disclosure system (which the ad-hoc
checks suggest, most likely a SEBI mandate that took effect around then),
not an artifact of only checking large, well-covered names - same
discipline as the 40-symbol financial-results pilot before it.

METHOD: sample symbols from `industry_map.csv` (this project's existing
sector-mapping reference - fine for a coverage PILOT, not a point-in-time
universe). For each: fetch the master JSON with NO date filter (the only
query shape confirmed to return real history per symbol - date-filtered
queries were tested live and return [] for anything before ~2021 even
when combined with a valid symbol). Record count, earliest/latest period,
and time whether one XBRL detail fetch+parse (for the OLDEST available
filing, i.e. the deepest history point) works at all - if `EncumberedShares
HeldAsPercentageOfTotalNumberOfShares` and the category `_ContextI` facts
aren't present on a genuinely old filing, the schema may have changed
mid-window and would need handling before a real fetch/cache module gets
built.
"""

from __future__ import annotations

import re
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
DELAY_SECONDS = 0.4
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

_CATEGORY_CONTEXTS = [
    "ShareholdingOfPromoterAndPromoterGroup_ContextI", "MutualFundsOrUTI_ContextI",
    "InsuranceCompanies_ContextI", "InstitutionsForeign_ContextI", "InstitutionsDomestic_ContextI",
    "PublicShareholding_ContextI",
]


def _fetch_master(session: requests.Session, symbol: str) -> list[dict]:
    url = f"https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={symbol}"
    try:
        r = session.get(url, timeout=15)
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"    {symbol} ERROR: {e}")
        return []


def _check_xbrl_schema(session: requests.Session, xbrl_url: str) -> dict:
    try:
        r = session.get(xbrl_url, timeout=20)
        txt = r.text
    except Exception as e:
        return {"xbrl_ok": False, "error": str(e)}

    facts = dict(re.findall(
        r'<in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares[^>]*contextRef="([^"]+)"[^>]*>([^<]*)'
        r'</in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares>', txt))
    found = {c: facts.get(c) for c in _CATEGORY_CONTEXTS}
    return {"xbrl_ok": all(v is not None for v in found.values()), "categories_found": found}


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
    oldest_xbrl_checked = False
    for i, sym in enumerate(sample):
        records = _fetch_master(session, sym)
        dates = [r.get("date") for r in records if r.get("date")]
        rows.append({
            "symbol": sym, "n_filings": len(records),
            "latest_period_end": dates[0] if dates else None,
            "earliest_period_end": dates[-1] if dates else None,
        })
        # on the very first symbol with real data, sanity-check the OLDEST
        # available filing's XBRL still has the expected category schema
        if not oldest_xbrl_checked and records:
            oldest = records[-1]
            xbrl_url = oldest.get("xbrl")
            if xbrl_url:
                check = _check_xbrl_schema(session, xbrl_url)
                print(f"  schema check on {sym}'s oldest filing ({oldest.get('date')}): "
                     f"xbrl_ok={check['xbrl_ok']}")
                if not check["xbrl_ok"]:
                    print(f"    categories_found: {check.get('categories_found')}")
                oldest_xbrl_checked = True
        time.sleep(DELAY_SECONDS)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  ... {i + 1}/{len(sample)} symbols probed ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)

    n_with_data = int((df["n_filings"] > 0).sum())
    print(f"\n=== COVERAGE, {len(sample)}-symbol sample ===")
    print(f"  symbols with ANY filings found: {n_with_data}/{len(sample)} "
         f"({100 * n_with_data / len(sample):.1f}%)")
    covered = df[df["n_filings"] > 0]
    if len(covered):
        print(f"  mean filings per covered symbol: {covered['n_filings'].mean():.1f}")
        print(f"  median filings per covered symbol: {covered['n_filings'].median():.1f}")
        earliest_dates = pd.to_datetime(covered["earliest_period_end"], format="%d-%b-%Y", errors="coerce")
        print("\n  earliest period-end distribution (covered symbols):")
        print(earliest_dates.dt.to_period("Q").value_counts().sort_index().to_string())

    print(f"\n  wall time: {elapsed:.1f}s for {len(sample)} symbols (metadata only, 1 call/symbol)")

    out_dir = Path(cfg.paths.runs) / "probe_shareholding_coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sample_coverage.csv", index=False)
    print(f"\nWrote {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
