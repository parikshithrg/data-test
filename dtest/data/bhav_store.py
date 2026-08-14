"""Parse the raw bhavcopy cache into a queryable store, partitioned by year.

Kept separate from `bhavcopy.py` (which only fetches) so that changing a parsing
decision costs a re-parse and never a re-download. The raw zips stay on disk
exactly as the exchange served them and remain the evidence for what was
published.

SERIES FILTERING, and why it is not just 'EQ'. NSE's `series` column is the
settlement segment:
  EQ  normal rolling settlement - what this project trades
  BE  trade-to-trade: compulsory delivery, no intraday netting, usually applied
      to names under surveillance. Real, tradeable, but a different instrument
      with different costs and much wider spreads.
  SM/ST  SME platform - lot-based, thin, not comparable
  GB/GS  sovereign gold bonds and government securities
  N1..N9, W1..W4  warrants, partly-paid, rights entitlements
Only EQ is kept. Including BE would quietly mix a surveillance segment into a
liquidity ranking, and the SME series would inject names that cannot be bought
in single shares at all.

MEMORY. ~5,500 sessions x ~1,800 EQ names is roughly 10M rows. Held long and
partitioned by year; wide panels are built only for the universe subset, because
a full (date x every symbol ever listed) panel is ~22M cells per field and there
are six fields.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from dtest.data.bhavcopy import COLUMNS, load_fetch_log, parse_day

log = logging.getLogger(__name__)

KEEP_SERIES = ("EQ",)

# Written as one parquet per calendar year.
_SCHEMA = {
    "date": "datetime64[ns]", "symbol": "string", "series": "string",
    "open": "float64", "high": "float64", "low": "float64", "close": "float64",
    "last": "float64", "prev_close": "float64", "volume": "float64",
    "turnover": "float64", "trades": "float64", "isin": "string",
}


def parsed_dir(cache_root: Path) -> Path:
    return Path(cache_root) / "parsed"


def build_store(
    cache_root: Path,
    *,
    years: list[int] | None = None,
    keep_series: tuple[str, ...] = KEEP_SERIES,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Parse every cached day into artifacts/bhav/parsed/<year>.parquet.

    Returns a per-year summary. Existing year files are skipped unless
    `rebuild=True`, so this is cheap to re-run as the ingest fills in.
    """
    cache_root = Path(cache_root)
    out_dir = parsed_dir(cache_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_df = load_fetch_log(cache_root)
    if log_df.empty:
        raise FileNotFoundError(f"no fetch log under {cache_root}; run the ingest first")
    ok = log_df[log_df["status"] == "ok"].copy()
    ok["year"] = ok["date"].dt.year

    target_years = sorted(set(ok["year"])) if years is None else sorted(years)
    summary = []

    for year in target_years:
        path = out_dir / f"{year}.parquet"
        if path.exists() and not rebuild:
            existing = pd.read_parquet(path, columns=["date", "symbol"])
            summary.append({"year": year, "days": existing["date"].nunique(),
                            "rows": len(existing), "symbols": existing["symbol"].nunique(),
                            "status": "cached"})
            continue

        days = sorted(ok.loc[ok["year"] == year, "date"].dt.date)
        frames = []
        for d in days:
            df = parse_day(cache_root, d)
            if df is None or df.empty:
                continue
            df = df[df["series"].isin(keep_series)]
            if not df.empty:
                frames.append(df)
        if not frames:
            log.warning("year %d: no parseable days", year)
            continue

        year_df = pd.concat(frames, ignore_index=True)
        # Deterministic row order. Without this the parquet bytes - and therefore
        # every downstream hash - depend on filesystem iteration order.
        year_df = year_df.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
        for col, dtype in _SCHEMA.items():
            year_df[col] = year_df[col].astype(dtype)
        year_df.to_parquet(path, index=False)

        summary.append({"year": year, "days": year_df["date"].nunique(),
                        "rows": len(year_df), "symbols": year_df["symbol"].nunique(),
                        "status": "built"})
        log.info("year %d: %d days, %d rows, %d symbols",
                 year, year_df["date"].nunique(), len(year_df), year_df["symbol"].nunique())

    return pd.DataFrame(summary)


def load_long(
    cache_root: Path,
    start: dt.date | None = None,
    end: dt.date | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the parsed store as one long frame, optionally date-bounded.

    Both bounds are applied here rather than downstream. A lower bound alone is
    not a time machine - that exact omission in the predecessor project let an
    ATR helper return today's value for a historical date.
    """
    out_dir = parsed_dir(cache_root)
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parsed parquet under {out_dir}; run build_store first")

    if start or end:
        lo = start.year if start else -1
        hi = end.year if end else 9999
        files = [f for f in files if lo <= int(f.stem) <= hi]

    cols = columns if columns is None else sorted(set(columns) | {"date", "symbol"})
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def to_panel(long_df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Pivot one field to a (date x symbol) panel.

    Duplicate (date, symbol) pairs would silently make `pivot` raise or
    aggregate; they are checked for explicitly so a data problem surfaces as a
    data problem rather than as a pivot error.
    """
    dupes = long_df.duplicated(subset=["date", "symbol"]).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicate (date, symbol) rows - refusing to pivot")
    return (
        long_df.pivot(index="date", columns="symbol", values=field)
        .sort_index()
        .sort_index(axis=1)
    )
