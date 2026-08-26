"""ONE-TIME DATA ACQUISITION, not part of the deterministic backtest harness -
same category as `ingest_bhavcopy.py`. Originally fetched only the 4 series
the cross-asset stress-regime gate needed (US VIX, USDINR, DXY, gold - India
VIX and breadth already exist/are derivable) - that hypothesis has since been
tested and REJECTED in full (`dtest.features.stress`, commit `71222274`,
"all rejected, closing the line"), so those 4 series are now historical
inputs to a closed line of research, kept because they're real and other
diagnostics may still read them.

EXTENDED 2026-08-26 (dataset priority queue Tier 6 item 16, "global
cross-asset") with 8 more series, purely as raw material for whatever the
analysis phase builds next - not wired into any signal or hypothesis yet,
per this project's collect-first-analyze-later sequencing. Deliberately
avoids FRED entirely (every FRED endpoint refused connections all session
during the item-13 macro work, `dtest/data/rbi_rates_credit.py`'s own
docstring) - `^TNX` (yfinance, CBOE 10Y treasury yield index) substitutes
for the "FRED rates" ask in the tier list, same instrument class, no new
transport problem to solve. All 8 candidates were live-probed via
`yf.download` before being added here - none were dead ends this time,
unlike several sources earlier in this project's history.

Saves every series as static CSVs in the SAME `date,open,high,low,close,volume`
shape `NIFTY50_DAILY.csv`/`INDIAVIX_DAILY.csv` already use, so every later
script reads them the same way - but into `cfg.paths.macro_dir`, NOT
`price_dir`. `price_dir` points at market_gate's own data folder, which
config.toml explicitly documents as read-only ("nothing in this project
writes to these paths") - this script is new source data THIS project
acquired itself, so it gets its own path, never mixed into that read-only
directory.

    python scripts/fetch_macro_stress_series.py

Run ONCE. Re-run only to refresh with more recent data - the resulting files
are then the source of truth, read-only, same as every other file this
project reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config

TICKERS = {
    # Original 4 - built for the now-closed cross-asset stress-gate test.
    "VIX": "^VIX",
    "USDINR": "USDINR=X",
    "DXY": "DX-Y.NYB",
    "GOLD": "GC=F",
    # Tier 6 item 16 - global cross-asset, added 2026-08-26.
    "CRUDE_WTI": "CL=F",       # WTI crude front-month futures
    "EM_FX": "CEW",            # WisdomTree Emerging Currency Strategy Fund - EM FX basket proxy
    "US10Y": "^TNX",           # CBOE 10Y treasury yield index - rates proxy (FRED substitute)
    "SP500": "^GSPC",          # US equity
    "STOXX50": "^STOXX50E",    # Europe equity
    "NIKKEI": "^N225",         # Japan equity
    "HANGSENG": "^HSI",        # China/HK equity
    "EM_EQUITY": "EEM",        # broad EM equity (iShares MSCI EM ETF)
}


def main() -> int:
    cfg = load_config()
    out_dir = cfg.paths.macro_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ticker in TICKERS.items():
        print(f"fetching {name} ({ticker}) ...")
        df = yf.download(ticker, period="max", progress=False)
        if df.empty:
            print(f"  FAILED - empty response for {ticker}")
            continue
        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        out_path = out_dir / f"{name}_DAILY.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path}: {df['date'].iloc[0]} .. {df['date'].iloc[-1]} ({len(df)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
