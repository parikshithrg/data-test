"""Credit rating actions (CRISIL/ICRA/CARE/India Ratings and others),
sourced from NSE's own `corporate-credit-rating` endpoint. Confirmed live,
2026-08-24: a single feed covers MULTIPLE agencies (found real records
from CARE Ratings, CRISIL Ratings, India Ratings and others in one query -
no need to scrape each agency's own site separately, unlike the AMC-
portfolio dead end the same session).

REAL FLOOR IS ~APRIL 2023, NOT DEEPER - confirmed live by a monthly sweep
back to 2010 (every month from 2010 through Q1 2023 returns 0 or a
handful of records; real, substantial monthly volumes of 500-5,000+ start
around April-September 2023). Plausibly a genuine XBRL-structured-
disclosure mandate rollout (same shape as `shareholding.py`'s own 2021
floor), not independently confirmed against a specific SEBI circular -
stated as the most likely explanation, not a verified fact.

THE `Symbol` FIELD ON THIS FEED IS NOT RELIABLE - a real, structural
finding, not a parsing bug. Sampled across multiple real quarters: the
overwhelming majority of records carry a placeholder in `Symbol`
("NOTLISTED", "Not Listed", "NOTAPPLICABLE", "000000", "111111" - all
confirmed real values seen live, not guessed) even for companies that DO
have real NSE-listed equity (e.g. "TVS Holdings Limited" - a real,
NSE-listed company under symbol TVSHLTD - showed `Symbol="NOTAPPLICABLE"`
on this feed). This is a genuine data-quality gap in NSE's own feed: most
credit ratings here are for DEBT instruments (bonds/NCDs), and even where
the issuer has separately-listed equity, this feed does not reliably link
the two. Only ~1-in-several-thousand records carry a directly-usable
ticker-shaped `Symbol` value.

SYMBOL RESOLUTION VIA NAME-MATCH, BEST-EFFORT, NOT GUARANTEED - the
practical fix for the gap above. `build_name_to_symbol_lookup` reuses
this project's OWN already-collected `index_reconstitution.py` events
(18,391 real `company_name`/`symbol` pairs, 2010-2026) as a reference
table - zero additional fetch cost, and a real, if partial, resource: a
40-symbol-window spot check matched ~57% of distinct companies in a real
quarter. Matching is exact on a NORMALIZED name (uppercase, "Limited"/
"Ltd."/"Private"/"Pvt" suffixes stripped, non-alphanumerics removed) -
never fuzzy, same "missing data blocks, never guesses" convention as
every other source in this project. A company never represented in any
index over 2010-2026 (small/niche names) will not resolve here - `symbol`
stays null for those rows, not fabricated.

WHY `AppID` IS THE REAL DEDUP KEY, confirmed live 2026-08-24: at least one
real query (April 2023) returned 158,705 rows collapsing to only 18
distinct `AppID` values - a genuine API-side duplication artifact (not
investigated further; root cause unclear, but the fix is unambiguous).
Every fetch in this module deduplicates on `AppID` before use.

WHY `symbol`+`from_date`/`to_date` COMBINED TRIGGERS A SERVER ERROR,
confirmed live: the identical date-only query succeeds every time; adding
a `symbol` param to a date-ranged query reliably 502s. `fetch_credit_
ratings.py` therefore NEVER combines them - it fetches broad date windows
only and resolves `symbol` client-side via the name-match above.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

SCHEMA = {
    "app_id": "string", "symbol": "string", "company_name": "string", "isin": "string",
    "agency": "string", "rating": "string", "rating_earlier": "string",
    "action": "string", "outlook": "string", "date_of_rating": "datetime64[ns]",
    "filing_date": "datetime64[ns]",
}

_SUFFIX_RE = re.compile(r'\b(LIMITED|LTD\.?|PRIVATE|PVT\.?)\b')
_NONALNUM_RE = re.compile(r'[^A-Z0-9]')
# A real NSE ticker always has at least one letter - real placeholders seen
# live include bare numeric filler ("000000", "222333", "111111") that
# would otherwise look like a plausible-if-odd symbol, caught by requiring
# a letter. Text placeholders ("NOTLISTED", "NOTAPPLICABLE") pass that
# same shape check (all letters, no spaces) so they need an explicit name
# check too - both real values seen live, not guessed.
_LOOKS_LIKE_TICKER_RE = re.compile(r'^(?=.*[A-Z])[A-Z0-9&\-\.]{1,20}$')
_TEXT_PLACEHOLDERS = {"NOTLISTED", "NOTAPPLICABLE", "NA", "NOT", "APPLICABLE"}


def normalize_company_name(name: str | None) -> str:
    """Uppercase, strip common suffixes and all non-alphanumerics, so
    "TVS Holdings Limited" and "TVS Holdings Ltd." collapse to the same
    key. Exact matching only - this project never fuzzy-matches an
    identity."""
    name = (name or "").upper()
    name = _SUFFIX_RE.sub("", name)
    return _NONALNUM_RE.sub("", name)


def build_name_to_symbol_lookup(index_reconstitution_events: pd.DataFrame) -> dict[str, str]:
    """A best-effort company-name -> symbol table built from this
    project's own `index_reconstitution.py` events (real company_name/
    symbol pairs, 2010-2026) - see module docstring for why this feed's
    own `Symbol` field can't be used directly."""
    lookup: dict[str, str] = {}
    for _, row in index_reconstitution_events.drop_duplicates(subset=["company_name"]).iterrows():
        key = normalize_company_name(row["company_name"])
        if key:
            lookup[key] = row["symbol"]
    return lookup


@dataclass(frozen=True)
class CreditRatingRecord:
    app_id: str
    symbol: str | None
    company_name: str
    isin: str | None
    agency: str
    rating: str | None
    rating_earlier: str | None
    action: str
    outlook: str | None
    date_of_rating: pd.Timestamp | None
    filing_date: pd.Timestamp


def _date(v: str | None, fmt: str) -> pd.Timestamp | None:
    if not v or v == "-":
        return None
    try:
        return pd.Timestamp(datetime.strptime(v, fmt))
    except ValueError:
        return None


def parse_credit_rating_record(rec: dict, name_to_symbol: dict[str, str]) -> CreditRatingRecord | None:
    """One record from `corporate-credit-rating`'s JSON array."""
    app_id = rec.get("AppID")
    company_name = rec.get("CompanyName")
    filing_date = _date(rec.get("BroadcastDateTime"), "%d-%b-%Y %H:%M:%S")
    if not app_id or not company_name or filing_date is None:
        return None

    raw_symbol = (rec.get("Symbol") or "").strip().upper()
    is_ticker_shaped = bool(_LOOKS_LIKE_TICKER_RE.match(raw_symbol)) and raw_symbol not in _TEXT_PLACEHOLDERS
    symbol = raw_symbol if is_ticker_shaped else None
    if symbol is None:
        symbol = name_to_symbol.get(normalize_company_name(company_name))

    return CreditRatingRecord(
        app_id=str(app_id), symbol=symbol, company_name=company_name,
        isin=rec.get("ISIN") or None, agency=rec.get("NameOfCRAgency") or "",
        rating=rec.get("CreditRating") or None, rating_earlier=rec.get("CreditRatingEarlier") or None,
        action=rec.get("RatingAction") or "", outlook=rec.get("Outlook") or None,
        date_of_rating=_date(rec.get("DateofCR"), "%d-%m-%Y"), filing_date=filing_date,
    )
