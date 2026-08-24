"""One-time acquisition: fetch and cache per-symbol SEBI PIT (insider-
trading) disclosures from NSE's own `corporates-pit` endpoint, for every
symbol that was EVER part of this project's own point-in-time eligible
universe (reuses `build_universe`, same as `fetch_shareholding.py`/
`fetch_financial_results.py`). NOT a live call inside the deterministic
harness - same category as those scripts.

    python scripts/fetch_insider_trading.py
    python scripts/fetch_insider_trading.py --symbols RELIANCE,TCS   # subset, for testing
    python scripts/fetch_insider_trading.py --resume                # skip symbols already cached

ONE CALL PER SYMBOL, not two-pass like shareholding/financial_results -
`dtest/data/insider_trading.py`'s module docstring explains why (a flat
JSON schema with no linked detail document worth fetching). ALSO ONE WIDE
DATE RANGE PER CALL, not windowed like financial_results.py needed -
confirmed live 2026-08-24 that a single `01-01-2015..as_of` query returns
the exact same record count as the sum of 12 separate yearly queries for
the same symbol (2,397 both ways) - no silent truncation on a wide range,
unlike financial_results.py's metadata endpoint which DID truncate near
~120-130 records on an unwindowed query.

RATE LIMITING: same courtesy every other NSE fetch script in this project
extends.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.data.insider_trading import SCHEMA, parse_pit_record
from dtest.universe import build_universe

# SEBI PIT Regulations, 2015 took effect mid-2015 - a real regulatory
# floor confirmed live (a 2010-2012 query returns 0 rows for RELIANCE, a
# handful trickle in from 2013-2015), not worth querying further back.
FROM_DATE = "01-01-2015"
DELAY_SECONDS = 0.3
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


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


def _fetch_symbol(session: requests.Session, symbol: str, to_date: str) -> pd.DataFrame:
    url = ("https://www.nseindia.com/api/corporates-pit"
          f"?index=equities&symbol={quote(symbol, safe='')}&from_date={FROM_DATE}&to_date={to_date}")
    try:
        r = session.get(url, timeout=20)
        raw_records = r.json().get("data", [])
    except Exception as e:
        print(f"    {symbol} ERROR: {e}")
        raw_records = []

    recs = [rec for r in raw_records if (rec := parse_pit_record(r)) is not None]
    recs.sort(key=lambda rec: rec.filing_date)
    print(f"  {symbol}: {len(recs)} disclosures")

    if not recs:
        return pd.DataFrame({k: pd.Series(dtype=v) for k, v in SCHEMA.items()})
    return pd.DataFrame([{
        "symbol": rec.symbol, "filing_date": rec.filing_date, "intim_date": rec.intim_date,
        "acq_from_date": rec.acq_from_date, "acq_to_date": rec.acq_to_date,
        "person_category": rec.person_category, "acq_mode": rec.acq_mode,
        "transaction_type": rec.transaction_type, "sec_type": rec.sec_type,
        "security_acquired_disposed": rec.security_acquired_disposed,
        "buy_quantity": rec.buy_quantity, "sell_quantity": rec.sell_quantity,
        "buy_value": rec.buy_value, "sell_value": rec.sell_value,
        "shares_before_no": rec.shares_before_no, "shares_before_pct": rec.shares_before_pct,
        "shares_after_no": rec.shares_after_no, "shares_after_pct": rec.shares_after_pct,
    } for rec in recs])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None, help="comma-separated subset, for testing")
    ap.add_argument("--resume", action="store_true", help="skip symbols already cached")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(cfg.paths.insider_trading_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    to_date = date.today().strftime("%d-%m-%Y")

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
        df = _fetch_symbol(session, symbol, to_date)
        df.to_csv(out_dir / f"{symbol}.csv", index=False)
        elapsed = time.time() - t0
        print(f"[{i + 1}/{len(symbols)}] {symbol}: wrote {len(df)} rows "
             f"({elapsed:.0f}s elapsed, {elapsed / (i + 1):.1f}s/symbol avg)")
        time.sleep(DELAY_SECONDS)

    print(f"\nDone. {len(symbols)} symbols fetched to {out_dir} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
