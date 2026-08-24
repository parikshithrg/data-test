"""One-time acquisition: fetch and cache per-symbol shareholding-pattern
data from NSE's own `corporate-share-holdings-master` endpoint, for every
symbol that was EVER part of this project's own point-in-time eligible
universe (not survivorship-biased - reuses `build_universe`, same as
`fetch_financial_results.py`). NOT a live call inside the deterministic
harness - same category as that script.

    python scripts/fetch_shareholding.py
    python scripts/fetch_shareholding.py --symbols RELIANCE,ZEEL,JPPOWER,NMDC   # subset, for testing
    python scripts/fetch_shareholding.py --resume                              # skip symbols already cached

MASTER THEN XBRL, two passes per symbol, same shape as
`fetch_financial_results.py`'s METADATA-then-DETAIL split. Pass 1: one call
per symbol to the master endpoint with NO date filter (the only query shape
confirmed live to return real multi-quarter history - see
`dtest/data/shareholding.py`'s module docstring) gives `promoter_pct`/
`public_pct` directly plus every filing's own XBRL detail-document URL.
Pass 2: one call per FILING to fetch+parse that XBRL for the finer MF/
insurance/FII/DII/pledge breakdown. Much lighter than
`fetch_financial_results.py` per symbol (median ~20 filings here vs ~95
there, per the 2026-08-24 pilot, `runs/probe_shareholding_coverage/`), but
still filing-count-bound, not symbol-count-bound, since pass 2 is the slow
step. Cached to disk immediately per symbol so a killed/interrupted run
loses at most one symbol's progress - `--resume` skips any symbol whose
cache file already exists.

RATE LIMITING: same courtesy every other NSE fetch script in this project
extends.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.data.shareholding import SCHEMA, parse_master_record, parse_xbrl_categories
from dtest.universe import build_universe

DELAY_SECONDS = 0.3
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_XBRL_FIELDS = ("mf_pct", "insurance_pct", "fii_pct", "dii_pct", "promoter_pledge_pct")


def _ever_eligible_universe(cfg) -> list[str]:
    print("computing the full point-in-time eligible universe (ever-eligible union) ...")
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    close = to_panel(long_df, "close")[stocks]
    turnover = to_panel(long_df, "turnover")[stocks]
    uni = build_universe(close, turnover, cfg)
    ever = sorted(uni.membership.columns[uni.membership.any(axis=0)])
    print(f"  {len(ever)} symbols were ever eligible across {close.index[0].date()}..{close.index[-1].date()}")
    return ever


def _fetch_master(session: requests.Session, symbol: str) -> list[dict]:
    # A bare "&" in the symbol (M&M, J&KBANK, ...) truncates the query
    # string and silently returns 0 records - confirmed live 2026-08-24
    # (M&M/GVT&D returned real data once encoded, MCX still returned 0 even
    # encoded so that one gap is real, not a query bug). quote() with
    # safe="" also covers the rarer case of a literal "&" mid-symbol vs one
    # from string formatting - both are the same character to urlencode.
    url = ("https://www.nseindia.com/api/corporate-share-holdings-master"
          f"?index=equities&symbol={quote(symbol, safe='')}")
    try:
        r = session.get(url, timeout=15)
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"    {symbol} master fetch ERROR: {e}")
        return []


def _fetch_xbrl(session: requests.Session, xbrl_url: str | None, master_promoter_pct: float) -> dict:
    # NSE serves a literal ".../xbrl/-" placeholder URL on some backfilled
    # records (confirmed live, 2026-08-24, on an NMDC record whose
    # broadcastDate lagged its period_end by 6 years) - a real, expected
    # "no detail doc for this record" case, not worth a wasted request.
    if xbrl_url is None or xbrl_url.rstrip("/").endswith("/-"):
        return {k: float("nan") for k in _XBRL_FIELDS}
    try:
        r = session.get(xbrl_url, timeout=20)
        r.raise_for_status()
        return parse_xbrl_categories(r.text, master_promoter_pct)
    except Exception as e:
        print(f"      xbrl fetch failed ({xbrl_url}): {e}")
        return {k: float("nan") for k in _XBRL_FIELDS}


def _fetch_symbol(session: requests.Session, symbol: str) -> pd.DataFrame:
    raw_records = _fetch_master(session, symbol)
    metas = [m for r in raw_records if (m := parse_master_record(symbol, r)) is not None]
    metas.sort(key=lambda m: m.filing_date)
    print(f"  {symbol}: {len(metas)} filings with usable metadata")

    rows = []
    for i, m in enumerate(metas):
        xbrl_fields = _fetch_xbrl(session, m.xbrl_url, m.promoter_pct)
        rows.append({
            "symbol": m.symbol, "filing_date": m.filing_date, "period_end": m.period_end,
            "revised": m.revised, "promoter_pct": m.promoter_pct, "public_pct": m.public_pct,
            **xbrl_fields,
        })
        time.sleep(DELAY_SECONDS)
        if (i + 1) % 20 == 0:
            print(f"    ... {i + 1}/{len(metas)} XBRL detail docs fetched")

    if not rows:
        return pd.DataFrame({k: pd.Series(dtype=v) for k, v in SCHEMA.items()})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None, help="comma-separated subset, for testing")
    ap.add_argument("--resume", action="store_true", help="skip symbols already cached")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(cfg.paths.shareholding_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = _ever_eligible_universe(cfg)

    if args.resume:
        before = len(symbols)
        symbols = [s for s in symbols if not (out_dir / f"{s}.csv").exists()]
        print(f"--resume: {before - len(symbols)} already cached, {len(symbols)} remaining")

    session = requests.Session()
    session.headers.update(HEADERS)
    print("warming up session (nseindia.com home page for cookies) ...")
    session.get("https://www.nseindia.com", timeout=10)

    t0 = time.time()
    for i, symbol in enumerate(symbols):
        df = _fetch_symbol(session, symbol)
        df.to_csv(out_dir / f"{symbol}.csv", index=False)
        elapsed = time.time() - t0
        print(f"[{i + 1}/{len(symbols)}] {symbol}: wrote {len(df)} rows "
             f"({elapsed:.0f}s elapsed, {elapsed / (i + 1):.1f}s/symbol avg)")

    print(f"\nDone. {len(symbols)} symbols fetched to {out_dir} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
