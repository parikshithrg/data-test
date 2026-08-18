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

ADDED 2026-08-18: `load_stock_futures_contracts` exposes every live
contract (not collapsed to front month) so a caller can pick the NEXT
contract instead of the front one - see that function's own docstring for
why (a front month entered with little runway left gets rollover-forced
almost immediately, independent of the signal's own exit logic).
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


def _query_live_contracts(
    fno_db: Path,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    """Shared SQL + price/open_price derivation for both `load_front_month_price`
    and `load_stock_futures_contracts` - every stock-futures contract still
    unexpired on the date it traded (`expiry_date >= date`), NOT yet collapsed
    to front month. Both bounds applied in SQL, same reasoning as
    `fno_oi.load_front_month_oi` - a lower bound alone is not a time machine,
    and filtering before loading into pandas keeps this cheap against a
    161M-row table. Opened read-only (`mode=ro`).
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
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["price"] = df["settle"].where(df["settle"] > 0, df["close"])
    # No fallback for open_price - a missing/zero open on a contract that
    # otherwise traded is a real data gap, not something a substitute mark
    # should paper over silently for a FILL price specifically.
    df["open_price"] = df["open"].where(df["open"] > 0)

    return df[df["expiry_date"] >= df["date"]]


def load_front_month_price(
    fno_db: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Front-month stock-futures mark, long format, one row per (date,
    symbol) - the soonest-unexpired contract each day, collapsed via
    `idxmin`. See `load_stock_futures_contracts` for the un-collapsed
    version (needed to pick a LATER contract, not just the front one).
    """
    live = _query_live_contracts(fno_db, start, end)
    if live.empty:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _SCHEMA.items()})

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


_CONTRACTS_SCHEMA = {
    "date": "datetime64[ns]", "symbol": "string", "expiry_date": "datetime64[ns]",
    "price": "float64", "open_price": "float64", "rank": "int64", "days_to_expiry": "int64",
}


def load_stock_futures_contracts(
    fno_db: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Every live stock-futures contract, long format, one row per (date,
    symbol, expiry_date) - the raw material `load_front_month_price`
    collapses away via `idxmin`. `rank` orders same-day contracts by
    expiry ascending (1 = front month, 2 = next month, ...).

    Built for `engine/pairs_simulate.py`'s rollforward-at-entry rule: a
    short leg opened into an almost-expired front month gets rollover-forced
    out almost immediately regardless of the signal's own exit logic
    (confirmed 2026-08-18: of 73 rollover-forced trades in the honest
    pairs re-test, 48 had 10 days to expiry at entry, and days-to-expiry
    at entry for rollover-forced trades ran a median 8 days vs 21 for
    trades that reverted normally) - picking the NEXT contract instead
    when the front month is nearly expired needs to see that next
    contract's own price and expiry, which `load_front_month_price` never
    exposes.
    """
    live = _query_live_contracts(fno_db, start, end)
    if live.empty:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _CONTRACTS_SCHEMA.items()})

    live = live.sort_values(["symbol", "date", "expiry_date"], kind="stable")
    live["rank"] = live.groupby(["symbol", "date"]).cumcount() + 1
    live["days_to_expiry"] = (live["expiry_date"] - live["date"]).dt.days

    live = live[["date", "symbol", "expiry_date", "price", "open_price",
                "rank", "days_to_expiry"]]
    for col, dtype in _CONTRACTS_SCHEMA.items():
        live[col] = live[col].astype(dtype)
    return live.sort_values(["date", "symbol", "rank"], kind="stable").reset_index(drop=True)
