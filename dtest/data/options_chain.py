"""Per-stock options chain (strike-level OI/volume/settle), read straight
from `fno_bhavcopy_full` - the same read-only source `fno_oi.py`/
`fno_price.py` already use for stock futures, filtered to
`asset_class='STOCK' AND contract_type='OPT'` instead of `'FUT'`.

WHY THIS WASN'T THOUGHT TO EXIST - a real correction, worth recording so
it isn't "rediscovered" the same way twice. [[project-data-test-status]]'s
2026-08-17 entry recorded "no per-stock implied vol exists anywhere in
this environment" - TRUE at the time for what was checked
(`options_derived_daily`, a curated table that only covers NIFTY/
BANKNIFTY - confirmed again live, 2026-08-24, still index-only, 0 stock
rows) - but nobody had checked the RAW `fno_bhavcopy_full` table itself
for stock-level options rows. It has 460 real stock symbols under
`asset_class='STOCK' AND contract_type='OPT'`, real strikes/CE-PE/OI/
volume/settle, 2008-2026. No live fetch needed - this data was already
sitting in the existing read-only database the whole time.

THE `instrument` COLUMN IS A REAL TRAP - DO NOT FILTER ON IT, a landmine
found live, 2026-08-24, before building anything. `fno_bhavcopy_full` was
populated by (at least) two ingestion batches using DIFFERENT category
codes for the exact same thing: an OLDER batch (rows through
2024-05-31) coded stock options as `instrument='OPTSTK'`; a NEWER,
CURRENT batch (2024-06-01 onward, real data through 2026-08-07 confirmed
live) codes the identical rows as `instrument='STO'` instead (stock
futures similarly split `FUTSTK` vs `STF`, index options `OPTIDX` vs
`IDO`, index futures `FUTIDX` vs `IDF`). A query filtering
`instrument='OPTSTK'` silently returns only pre-2024-06 data and looks
complete (no error, plausible row counts) - this is exactly how the
2026-08-17 "no per-stock options" finding could have been drawn from a
narrower check without anyone noticing the gap. `asset_class`+
`contract_type` (`'STOCK'`+`'OPT'`) is the reliable filter across BOTH
eras - the same columns `fno_oi.py`/`fno_price.py` already used, which is
exactly why those two modules never hit this trap themselves.

NO IMPLIED VOLATILITY COLUMN EXISTS on these raw rows (the `iv` column
this project already knows about lives only on `options_chain_daily`,
which is empty - a scaffolded-but-never-populated table, checked directly,
0 rows). Computing IV from `settle` via a Black-Scholes inversion is real,
standard, and buildable, but is FEATURE ENGINEERING, not data collection -
deliberately left to `dtest/features/` for the analysis phase, matching
this module's own scope: raw strike-level OI/volume/settle only, the same
"data extraction only, no derived computation" boundary
`fno_oi.py`/`fno_price.py` already keep.

NO SEPARATE FETCH SCRIPT - unlike every other Tier 1/2 source added this
session, this data requires no live network call at all: it already lives
in the local, read-only `fno.db` this project has depended on since its
first commit. `load_options_chain` below is a query function, called at
analysis time, exactly like `fno_oi.py::load_front_month_oi` and
`fno_price.py::load_front_month_price` already are - there is nothing to
cache or re-fetch.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = {
    "trade_date": "datetime64[ns]", "symbol": "string", "expiry_date": "datetime64[ns]",
    "strike": "float64", "option_type": "string", "open": "float64", "high": "float64",
    "low": "float64", "close": "float64", "settle": "float64", "contracts": "float64",
    "open_interest": "float64", "chg_in_oi": "float64",
}


def load_options_chain(
    fno_db: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Every raw per-stock options row in range, long format, one row per
    (trade_date, symbol, expiry_date, strike, option_type).

    Both date bounds are applied in the SQL - a lower bound alone is not a
    time machine, and filtering before loading into pandas keeps this
    cheap against a 161M-row table (same reasoning as every other loader
    in this module's own family: `fno_oi.py`, `fno_price.py`,
    `bhav_store.load_long`). Opened read-only (`mode=ro`) - this project
    never writes to a database it does not own. `symbols`, if given,
    pushes the filter into SQL too rather than loading the full chain and
    filtering in pandas, since a single day's stock options chain can run
    to tens of thousands of strike/expiry/type combinations across all 460
    symbols.
    """
    query = (
        "SELECT trade_date, symbol, expiry_date, strike, option_type, "
        "open, high, low, close, settle, contracts, open_interest, chg_in_oi "
        "FROM fno_bhavcopy_full WHERE asset_class='STOCK' AND contract_type='OPT'"
    )
    params: list[str] = []
    if start is not None:
        query += " AND trade_date >= ?"
        params.append(pd.Timestamp(start).strftime("%Y-%m-%d"))
    if end is not None:
        query += " AND trade_date <= ?"
        params.append(pd.Timestamp(end).strftime("%Y-%m-%d"))
    if symbols:
        placeholders = ",".join("?" * len(symbols))
        query += f" AND symbol IN ({placeholders})"
        params.extend(symbols)

    uri = Path(fno_db).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        df = pd.read_sql_query(query, con, params=params)
    finally:
        con.close()

    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in SCHEMA.items()})

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    for col, dtype in SCHEMA.items():
        if col not in ("trade_date", "expiry_date"):
            df[col] = df[col].astype(dtype)
    return df.sort_values(
        ["trade_date", "symbol", "expiry_date", "strike", "option_type"], kind="stable"
    ).reset_index(drop=True)
