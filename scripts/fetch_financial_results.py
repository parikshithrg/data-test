"""One-time acquisition: fetch and cache per-symbol quarterly financial
results from NSE's own `corporates-financial-results` endpoint, for every
symbol that was EVER part of this project's own point-in-time eligible
universe (not survivorship-biased, not just today's index snapshot -
reuses `build_universe` the same way every other real hypothesis test in
this project does). NOT a live call inside the deterministic harness -
same category as `fetch_macro_stress_series.py`.

    python scripts/fetch_financial_results.py
    python scripts/fetch_financial_results.py --symbols RELIANCE,TCS   # subset, for testing
    python scripts/fetch_financial_results.py --resume                # skip symbols already cached

WHY "ever in the point-in-time universe," not the full bhav-store symbol
list. The full bhav store carries thousands of names (indices, thinly-
traded delisted stocks, etc.) most of which were never economically
relevant to any hypothesis this project tests - fetching fundamentals for
all of them would be mostly wasted NSE calls. The point-in-time eligible
universe (`universe.py::build_universe`, top-200-by-turnover-banded-to-250,
monthly) is the actual real target set: bounded, non-survivorship-biased
(a name that delisted in 2015 but was genuinely traded/eligible before
then is still included), and the same set every hypothesis test in this
project already restricts its own signals to.

METADATA THEN DETAIL, two passes, both resumable independently.
Pass 1 fetches each symbol's filing metadata (dates, format, detail link) -
fast (~9 min for the full universe, confirmed by the 2026-08-20 40-symbol
pilot). Pass 2 fetches and parses each filing's own DETAIL page (the actual
P&L numbers) - one HTTP call per filing, the slow step (median ~95 filings
per covered symbol). Both are cached to disk immediately per symbol so a
killed/interrupted run loses at most one symbol's progress, not the whole
fetch - `--resume` skips any symbol whose cache file already exists.

RATE LIMITING: a real, if unofficial, endpoint - throttled with a small
delay between requests throughout, same courtesy `_news_data.py` (Local
Terminal) and `probe_financial_results_coverage.py` already extend to NSE.
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
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.data.financial_results import SCHEMA, parse_metadata_record, parse_old_html, parse_xbrl
from dtest.universe import build_universe

METADATA_WINDOWS = [("01-01-2004", "31-12-2010"), ("01-01-2010", "31-12-2018"), ("01-01-2018", "20-08-2026")]
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


def _fetch_metadata(session: requests.Session, symbol: str) -> list[dict]:
    seen: dict[str, dict] = {}
    for frm, to in METADATA_WINDOWS:
        url = (
            "https://www.nseindia.com/api/corporates-financial-results"
            f"?index=equities&period=Quarterly&fo_sec=false&symbol={symbol}"
            f"&from_date={frm}&to_date={to}"
        )
        try:
            r = session.get(url, timeout=15)
            data = r.json()
            if isinstance(data, list):
                for rec in data:
                    seen[rec.get("seqNumber")] = rec
        except Exception as e:
            print(f"    {symbol} metadata [{frm}..{to}] ERROR: {e}")
        time.sleep(DELAY_SECONDS)
    return list(seen.values())


def _fetch_detail(session: requests.Session, meta) -> dict:
    if meta.detail_url is None:
        return {k: float("nan") for k in ("revenue", "total_income", "total_expenses",
                                          "profit_before_tax", "net_profit", "eps_basic",
                                          "eps_diluted", "paidup_equity_capital", "reserves")}
    try:
        r = session.get(meta.detail_url, timeout=20)
        r.raise_for_status()
        if meta.source_format == "html_old":
            return parse_old_html(r.text)
        return parse_xbrl(r.text)
    except Exception as e:
        print(f"      detail fetch failed ({meta.symbol}, {meta.seq_number}): {e}")
        return {k: float("nan") for k in ("revenue", "total_income", "total_expenses",
                                          "profit_before_tax", "net_profit", "eps_basic",
                                          "eps_diluted", "paidup_equity_capital", "reserves")}


def _fetch_symbol(session: requests.Session, symbol: str) -> pd.DataFrame:
    raw_records = _fetch_metadata(session, symbol)
    metas = [m for r in raw_records if (m := parse_metadata_record(symbol, r)) is not None]
    metas.sort(key=lambda m: m.filing_date)
    print(f"  {symbol}: {len(metas)} filings with usable metadata")

    rows = []
    for i, m in enumerate(metas):
        fields = _fetch_detail(session, m)
        rows.append({
            "symbol": m.symbol, "filing_date": m.filing_date, "period_end": m.period_end,
            "period_start": m.period_start, "consolidated": m.consolidated,
            "source_format": m.source_format, "seq_number": m.seq_number, **fields,
        })
        time.sleep(DELAY_SECONDS)
        if (i + 1) % 20 == 0:
            print(f"    ... {i + 1}/{len(metas)} detail pages fetched")

    if not rows:
        return pd.DataFrame({k: pd.Series(dtype=v) for k, v in SCHEMA.items()})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None, help="comma-separated subset, for testing")
    ap.add_argument("--resume", action="store_true", help="skip symbols already cached")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(cfg.paths.fundamentals_dir)
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
