"""Per-stock futures open interest, reconstructed as a continuous
front-month series from raw per-contract rows.

Read-only from market_gate/fno.db's `fno_bhavcopy_full` table
(`asset_class='STOCK' AND contract_type='FUT'`), filtered the same way
market_gate's own live gate does (`stock_movers.py::_latest_oi_change_pct`)
- confirmed by reading that function before building this, since it is the
only place either project has ever picked a front-month stock-futures
contract. That function only reads ONE day at a time for a live gate;
NEITHER project has a continuous HISTORICAL per-stock OI series before this
module. (The pre-built `futures_continuous_daily`/`futures_front_month`/
`oi_study_features` tables in the same database do NOT cover this - checked
directly, they hold only 2-5 INDEX underlyings for market_gate's own
`futures_term_structure` gate signal, never stocks.)

FRONT MONTH, defined identically to market_gate: for each (symbol, date),
the live contract with the SOONEST `expiry_date >= date` - the near-month
convention every NSE F&O desk uses. Selection happens independently per
day, so the source contract naturally switches once the previous front
month's own expiry passes; a contract's row for `trade_date == expiry_date`
is its last trading day and is still correctly the front month that day.

THE ROLLOVER TRAP, and why raw `open_interest` LEVELS must not be diffed
across a switch. A stitched LEVEL series jumps at every rollover - the new
front-month contract typically already carries most of the market's open
interest (traders roll ahead of expiry), so a naive `level(t) - level(t-1)`
across the switch reads as a spurious multi-hundred-percent OI spike with
nothing to do with that day's real activity - the exact class of trap
`corporate_actions.py` solved for price splits/bonuses, here for futures
contracts instead. THE FIX: never diff levels across a switch. `chg_in_oi`
in `fno_bhavcopy_full` is already the EXCHANGE's own per-CONTRACT
day-over-day change - that same contract was already trading (as the
next-month contract) the day before, just not yet selected as front month,
so its own `chg_in_oi` on the switch day is a real, contract-continuous
number, not a stitching artifact. This module therefore carries
`chg_in_oi` straight through from the source rows and derives `oi_chg_pct`
from it (matching market_gate's own
`chg_in_oi / (open_interest - chg_in_oi) * 100` derivation exactly, so the
two projects' figures are comparable) - it never computes its own diff of
the stitched `open_interest` level. `is_rollover` is still recorded
(source contract's `expiry_date` changed from the prior row for that
symbol) so any FUTURE level-based feature (an OI-rank-over-252-days, say -
not built here) knows where it must not compare across the boundary; the
momentum signal built on this module is deliberately change-based, not
level-based, specifically to sidestep the trap rather than work around it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

_SCHEMA = {
    "date": "datetime64[ns]", "symbol": "string",
    "expiry_date": "datetime64[ns]", "open_interest": "float64",
    "chg_in_oi": "float64", "oi_chg_pct": "float64", "is_rollover": "bool",
    "days_to_expiry": "int64",
}


def load_front_month_oi(
    fno_db: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Front-month stock-futures OI, long format, one row per (date, symbol).

    Both bounds are applied in the SQL, same reasoning as
    `bhav_store.load_long` / `delivery.load_delivery_long`: a lower bound
    alone is not a time machine, and filtering before loading into pandas
    keeps this cheap against a 161M-row table. Opened read-only (`mode=ro`)
    - this project never writes to a database it does not own.
    """
    query = (
        "SELECT trade_date AS date, symbol, expiry_date, open_interest, chg_in_oi "
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

    live = df[df["expiry_date"] >= df["date"]]
    idx = live.groupby(["symbol", "date"])["expiry_date"].idxmin()
    front = live.loc[idx].sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)

    prev_oi = front["open_interest"] - front["chg_in_oi"]
    prev_oi_safe = prev_oi.where(prev_oi != 0)   # avoid a divide-by-zero -> inf
    front["oi_chg_pct"] = front["chg_in_oi"] / prev_oi_safe * 100.0

    expiry_diff = front.groupby("symbol")["expiry_date"].diff()
    front["is_rollover"] = expiry_diff.notna() & (expiry_diff != pd.Timedelta(0))
    # Calendar days, not trading days - a coarse but adequate measure for the
    # expiry-cycle filter this feeds (see dtest.signals.oi_momentum): OI
    # mechanically ramps for days after a rollover (traders building back into
    # the new front month) and mechanically decays in the final days before
    # expiry (traders rolling OUT) - both real, both nothing to do with
    # sentiment. A signal reading raw chg_in_oi without knowing where in this
    # cycle a given day sits cannot tell mechanical rolldown from genuine
    # conviction.
    front["days_to_expiry"] = (front["expiry_date"] - front["date"]).dt.days

    front = front[["date", "symbol", "expiry_date", "open_interest", "chg_in_oi",
                   "oi_chg_pct", "is_rollover", "days_to_expiry"]]
    for col, dtype in _SCHEMA.items():
        front[col] = front[col].astype(dtype)
    return front.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
