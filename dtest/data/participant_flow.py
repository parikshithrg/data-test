"""FII net positioning in NIFTY INDEX futures - a market-wide (not
per-symbol) daily state variable, from `fno.db`'s `participant_oi_daily`.

Read-only, same `long - short` net convention market_gate's own
`participant_oi.py::load_participant_oi` uses for its "Participant-wise
OI" dashboard section (that module is a DISPLAY layer only, never fed into
a strategy signal in either project before this).

Only `future_index_long`/`future_index_short` for `participant='FII'` -
NSE's most closely watched flow category, and the one the story in
`dtest.signals.participant_tilt` is actually about. INDEX futures, not
stock futures: NSE's participant-OI report is published at the aggregate/
index level, there is no per-stock breakdown - this is why the resulting
signal must be a market-wide GATE applied uniformly to every symbol on a
given day, not a per-symbol confirmation like `delivery_breakout`/
`oi_momentum`.

Data begins 2018-01-01 - inside the `delivery` split's own train_start
(2019-06-27), so reporting on that split needs no new window and no
extension of the pre-committed split boundaries (see the 08-15
window-vs-execution diagnostic entry in [[project-data-test-status]] for
why a post-hoc window choice is exactly what this project's splits exist
to prevent).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

EARLIEST_DATE = pd.Timestamp("2018-01-01")


def load_fii_net_index_flow(
    fno_db: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """Daily FII net INDEX-futures OI (`future_index_long - future_index_short`),
    indexed by date. Opened read-only - this project never writes to a
    database it does not own.
    """
    query = (
        "SELECT trade_date AS date, future_index_long, future_index_short "
        "FROM participant_oi_daily WHERE participant = 'FII'"
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
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    dupes = df["date"].duplicated().sum()
    if dupes:
        raise ValueError(f"{dupes} duplicate FII trade_date rows - refusing to build a series")

    net = pd.Series(
        (df["future_index_long"] - df["future_index_short"]).astype("float64").to_numpy(),
        index=pd.DatetimeIndex(df["date"], name="date"),
        name="fii_net_index_oi",
    )
    return net.sort_index()
