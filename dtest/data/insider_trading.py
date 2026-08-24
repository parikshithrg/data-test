"""Per-symbol insider-trading disclosures (SEBI PIT Regulations, 2015),
sourced from NSE's own `corporates-pit` endpoint. Confirmed live,
2026-08-24, before building anything: unlike `shareholding.py`'s master
endpoint, this one genuinely RESPECTS `from_date`/`to_date` combined with
`symbol` (spot-checked: a 2010-2012 window returns 0 records, a narrow
Jan-Feb 2017 window returns exactly the in-range subset, RELIANCE alone
returns 2,397 records 2015-2026) - the only source in this project so far
where the date-filtered query actually works as documented.

WHY 2015 IS A REAL REGULATORY FLOOR, not a data-source limitation like
shareholding's ~2021 floor. SEBI's Prohibition of Insider Trading
Regulations, 2015 (replacing the older 1992 regime) took effect mid-2015;
RELIANCE's own earliest record is 2015-11-19, and a query for 2010-2012
returns genuinely zero rows while 2013-2015 returns a small handful (3) -
consistent with a real regulatory transition, not a broken/unindexed query
(confirmed by the date-filter actually working, unlike shareholding).

FLAT JSON SCHEMA, NO XBRL PARSING NEEDED - the biggest structural
difference from `shareholding.py`. Every field this module needs
(quantities, values, before/after holding %, person category, transaction
type) is already present as plain JSON keys on the metadata response
itself; the linked `xbrl` document (present on newer filings, `None` on
the oldest ones, e.g. 2015) was inspected and found to add nothing this
module doesn't already have from the flat fields, so it is NOT fetched -
a deliberate, one-pass design, unlike financial_results.py/shareholding.py's
two-pass metadata-then-detail split. Confirmed schema-stable across the
full 2015-2026 span sampled (oldest and newest RELIANCE records share
identical keys), so no multi-era alias handling is needed here either.

WHY `date` (NSE's own broadcast timestamp) IS THE ONLY CAUSAL FIELD - same
point-in-time convention as every other source in this project. A record
carries THREE dates: `acqfromDt`/`acqtoDt` (when the actual trade
happened), `intimDt` (when the insider/company informed the exchange), and
`date` (when NSE itself published the disclosure, always the latest of the
three, with a time component). Only `date` is knowable to a market
participant at that moment - `acqfromDt`/`acqtoDt`/`intimDt` describe a
transaction that had already happened privately by the time it was public.

NUMERIC FIELDS ARRIVE AS STRINGS, best-effort comma-stripped before
float-casting (no comma observed in any sampled value, but not assumed
absent at full scale) - `-` (real, means "not applicable for this
transaction type", e.g. `sellValue` on a pure-buy row) parses to NaN, same
"missing data blocks, never guesses" convention as every other source here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

SCHEMA = {
    "symbol": "string", "filing_date": "datetime64[ns]",
    "intim_date": "datetime64[ns]", "acq_from_date": "datetime64[ns]", "acq_to_date": "datetime64[ns]",
    "person_category": "string", "acq_mode": "string", "transaction_type": "string", "sec_type": "string",
    "security_acquired_disposed": "float64", "buy_quantity": "float64", "sell_quantity": "float64",
    "buy_value": "float64", "sell_value": "float64",
    "shares_before_no": "float64", "shares_before_pct": "float64",
    "shares_after_no": "float64", "shares_after_pct": "float64",
}


@dataclass(frozen=True)
class InsiderRecord:
    symbol: str
    filing_date: pd.Timestamp
    intim_date: pd.Timestamp | None
    acq_from_date: pd.Timestamp | None
    acq_to_date: pd.Timestamp | None
    person_category: str
    acq_mode: str
    transaction_type: str
    sec_type: str
    security_acquired_disposed: float
    buy_quantity: float
    sell_quantity: float
    buy_value: float
    sell_value: float
    shares_before_no: float
    shares_before_pct: float
    shares_after_no: float
    shares_after_pct: float


def _num(v) -> float:
    if v in (None, "", "-"):
        return float("nan")
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return float("nan")


def _date(v: str | None, fmt: str) -> pd.Timestamp | None:
    if not v or v == "-":
        return None
    try:
        return pd.Timestamp(datetime.strptime(v, fmt))
    except ValueError:
        return None


def parse_pit_record(rec: dict) -> InsiderRecord | None:
    """One row of the `corporates-pit` JSON `data` array."""
    symbol = rec.get("symbol")
    filing_date = _date(rec.get("date"), "%d-%b-%Y %H:%M")
    if not symbol or filing_date is None:
        return None

    return InsiderRecord(
        symbol=symbol, filing_date=filing_date,
        intim_date=_date(rec.get("intimDt"), "%d-%b-%Y"),
        acq_from_date=_date(rec.get("acqfromDt"), "%d-%b-%Y"),
        acq_to_date=_date(rec.get("acqtoDt"), "%d-%b-%Y"),
        person_category=rec.get("personCategory") or "",
        acq_mode=rec.get("acqMode") or "",
        transaction_type=rec.get("tdpTransactionType") or "",
        sec_type=rec.get("secType") or "",
        security_acquired_disposed=_num(rec.get("secAcq")),
        buy_quantity=_num(rec.get("buyQuantity")), sell_quantity=_num(rec.get("sellquantity")),
        buy_value=_num(rec.get("buyValue")), sell_value=_num(rec.get("sellValue")),
        shares_before_no=_num(rec.get("befAcqSharesNo")), shares_before_pct=_num(rec.get("befAcqSharesPer")),
        shares_after_no=_num(rec.get("afterAcqSharesNo")), shares_after_pct=_num(rec.get("afterAcqSharesPer")),
    )
