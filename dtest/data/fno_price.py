"""Front-month stock-futures settlement price, reconstructed the same way
`fno_oi.py` reconstructs open interest - one row per (date, symbol), the
live contract with the soonest `expiry_date >= date`. The Phase 1
prerequisite named in the "Long, Short, Neutral" architecture proposal
(2026-08-17): marking a short futures position to market needs a
continuous price read, which neither project has ever built before this.

WHY THIS IS NOT A "STITCHED CONTINUOUS SERIES" IN THE CHARTING SENSE. The
classic technique for a continuous futures chart applies an additive or
ratio adjustment to every price before a roll, so the series shows no jump
at the boundary. That adjustment is a real fiction, and it would be the
WRONG fiction here: `fno_oi.py` could safely carry `chg_in_oi` across a
rollover because that field is the EXCHANGE's own contract-native
day-over-day number - the OI jump at a naive stitch was a pure artifact
of diffing levels across two different instruments, not real information.
Futures PRICE has no equivalent safe field: the front-month and next-month
contracts trade at genuinely different levels (a real cost-of-carry/
dividend basis), so a price difference across a roll is real economic
content, not a stitching artifact - erasing it with a back-adjustment
would erase a fact, not fix a bug.

THE CONSEQUENCE FOR ANY CALLER, stated so it cannot be missed: a position
must never be marked across a rollover boundary as if it were one
uninterrupted contract. `is_rollover` (identical definition to
`fno_oi.py`'s) exists precisely so a caller's simulator can treat it as a
hard exit boundary - close the position at the last front-month price
before the roll, same as any other forced exit - never compute a return
spanning two different instruments.

SETTLE, NOT CLOSE, as the primary MARK (`price`). NSE's own `settle` price
is what actually determines a real account's daily mark-to-market margin -
the authoritative EOD number, not the last traded price. Falls back to
`close` only where `settle` is null or non-positive (thin-contract days),
matching `bhavcopy.py`'s own "authoritative field first, approximate only
where it is missing" convention.

`open_price` IS ALSO EXPOSED, alongside `price`, for one specific reason:
`config.execution.fill_at` requires every position in this project to fill
at the NEXT session's OPEN, never the signal bar's own close - a rule that
applies exactly as much to a futures leg as to a cash-equity one. Without
a futures OPEN, a short leg could only ever be honestly filled at the mark
(`price`), which is a settlement/close-time number and would quietly
reintroduce the same look-ahead this project's whole execution model
exists to remove. `price` (the mark) and `open_price` (the fill) serve
different callers - a rollover check needs the mark; an entry/exit fill
needs the open - and are kept as two columns rather than one so a caller
cannot accidentally use one for the other's job.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

_SCHEMA = {
    "date": "datetime64[ns]", "symbol": "string",
    "expiry_date": "datetime64[ns]", "price": "float64", "open_price": "float64",
    "is_rollover": "bool", "days_to_expiry": "int64",
}


def load_front_month_price(
    fno_db: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Front-month stock-futures mark, long format, one row per (date,
    symbol). Both bounds applied in SQL, same reasoning as
    `fno_oi.load_front_month_oi` - a lower bound alone is not a time
    machine, and filtering before loading into pandas keeps this cheap
    against a 161M-row table. Opened read-only (`mode=ro`).
    """
    query = (
        "SELECT trade_date AS date, symbol, expiry_date, open, close, settle "
        "FROM fno_bhavcopy_full WHERE asset_class='STOCK' AND contract_type='FUT'"
    )
    params: list[str] = []
    if start is not None:
        query += " AND trade_date >= ?"
        params.append(pd.Timestamp(start).strftime("%Y-%m-%d"))
    if end is not None:
        query += " AND trade_date <= ?"
        params.append(pd.Timestamp(end).strftime("%Y-%m-%d"))

    uri = Path(fno_db).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        df = pd.read_sql_query(query, con, params=params)
    finally:
        con.close()

    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _SCHEMA.items()})

    df["date"] = pd.to_datetime(df["date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["price"] = df["settle"].where(df["settle"] > 0, df["close"])
    # No fallback for open_price - a missing/zero open on a contract that
    # otherwise traded is a real data gap, not something a substitute mark
    # should paper over silently for a FILL price specifically.
    df["open_price"] = df["open"].where(df["open"] > 0)

    live = df[df["expiry_date"] >= df["date"]]
    idx = live.groupby(["symbol", "date"])["expiry_date"].idxmin()
    front = live.loc[idx].sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)

    expiry_diff = front.groupby("symbol")["expiry_date"].diff()
    front["is_rollover"] = expiry_diff.notna() & (expiry_diff != pd.Timedelta(0))
    front["days_to_expiry"] = (front["expiry_date"] - front["date"]).dt.days

    front = front[["date", "symbol", "expiry_date", "price", "open_price",
                   "is_rollover", "days_to_expiry"]]
    for col, dtype in _SCHEMA.items():
        front[col] = front[col].astype(dtype)
    return front.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
