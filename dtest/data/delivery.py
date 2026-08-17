"""Per-stock delivery data from the exchange's own daily settlement figures.

Read-only from market_gate/fno.db's `cash_delivery_daily` table - built and
maintained by that project's own ingest pipeline (`nse_extra_sources.py`'s
"delivery" source, see [[project-market-gate-status]]), never written to
here. Only EQ-series rows are kept: same series filter and same reasoning
as `bhav_store.py` (BE is trade-to-trade, a different instrument with
different costs; SM/GB/warrants are not comparable, see that module's own
docstring for the full breakdown).

Delivery data begins 2019-06-27 - this is why `config.toml` carries a
separate `[splits.delivery]` block with far less history than `primary`,
and any hypothesis built on this module must be reported against THAT
split, never primary/train. The predecessor project measured this same
floor and found `participation` (its delivery+volume composite term)
silently degrading to volume-only before this date - a hypothesis here
that lets its signal fire before 2019-06-27 is making the identical
mistake.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

KEEP_SERIES = ("EQ",)
EARLIEST_DATE = pd.Timestamp("2019-06-27")

_SCHEMA = {
    "date": "datetime64[ns]", "symbol": "string",
    "total_traded_qty": "float64", "delivery_qty": "float64",
    "delivery_pct": "float64", "turnover": "float64",
}


def load_delivery_long(
    fno_db: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """EQ-series delivery figures from `cash_delivery_daily`, long format.

    Both bounds are applied here in the SQL, same reasoning as
    `bhav_store.load_long`: a lower bound alone is not a time machine, and
    filtering in the query (not after loading into pandas) keeps this cheap
    against a 4.1M-row table.

    Opened read-only (`mode=ro`) - this project never writes to a database
    it does not own.
    """
    query = (
        "SELECT trade_date AS date, symbol, total_traded_qty, delivery_qty, "
        "delivery_pct, turnover FROM cash_delivery_daily WHERE series = 'EQ'"
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

    df["date"] = pd.to_datetime(df["date"])
    for col, dtype in _SCHEMA.items():
        df[col] = df[col].astype(dtype)
    # Deterministic row order - same reasoning as bhav_store.build_store:
    # without this, row order (and any hash of it) depends on SQLite's own
    # unspecified scan order.
    return df.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
