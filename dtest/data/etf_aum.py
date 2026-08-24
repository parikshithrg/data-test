"""Quarterly scheme-level ETF/Index-Fund average AUM, sourced from AMFI's
own `average-aum-schemewise` endpoint (Categorywise, all AMCs combined).
Confirmed live, 2026-08-24, before building anything: this is the
INDUSTRY'S OWN real quarterly disclosure, not a scrape of a rendered page -
`amfiindia.com` is a JS-heavy Next.js site (same category of problem that
ruled out AMC monthly portfolio disclosures the same session), but this
specific data lives behind a clean JSON API found by inspecting the
network log while driving the real "Average AUM" page, not guessed.

REAL DEPTH: fyId=1..21 all return real data (fyId=22+ return HTTP 400) -
`fyId=1` is the CURRENT financial year (confirmed live: selecting "April
2026 - March 2027" on the real page produced `fyId=1`) and INCREASES
going backward, one per Indian financial year (April-March) - so fyId=21
is FY2006-07, giving ~20 years of real quarterly scheme-level AUM, deeper
than every other source this project sourced itself except
`financial_results.py`. `periodId` is NOT a fixed quarter-of-year code -
it is the reverse-chronological POSITION within that FY's available
period list (1 = most recently completed quarter in that FY, up to 4 for
a complete year) - confirmed live: the current partial FY2026-27 only
offered ONE period option and it was `periodId=1` for Apr-Jun 2026, not a
fixed "Q1" code. Consequently this module derives the true calendar
quarter from the response's OWN `selectedPeriod` text field (e.g.
"April - June 2026"), never from fyId/periodId arithmetic - the only
robust way, confirmed necessary by that discovery.

PRE-OCT-2010 IS MONTHLY, NOT QUARTERLY, AND ONLY PARTIALLY COVERED HERE -
a real scope limitation, stated plainly. AMFI's own disclosure cadence was
monthly through September 2010 (confirmed by the real "Average AUM" page's
own text, and by real single-month `selectedPeriod` labels like "March
2010" actually appearing at fyId>=18 in the live fetch). Since
`fetch_etf_aum.py` still only walks `periodId` 1-4 per `fyId` (matching
the quarterly-era shape), a monthly-disclosure fiscal year's 4 fetched
periods are only 4 of its real ~12 monthly disclosures, not full monthly
coverage - the ~2010-2026 quarterly-era data (the bulk of this dataset) is
complete, the ~2006-2010 monthly tail is a partial sample of it, not
investigated further this session.

UNITS ARE RS LAKHS, confirmed live, not assumed from convention -
cross-checked the fetched industry-wide grand total (~831.4 million,
i.e. ~83.1 lakh crore in Rupees) and one large AMC's own total (Aditya
Birla Sun Life: ~42.77 million lakhs = ~Rs 4.28 lakh crore) against
real, independently known approximate industry/AMC AUM figures - both
line up only if the raw numbers are Rs Lakhs, not Rs Crores or absolute
Rupees.

WHY POINT-IN-TIME CORRECTNESS NEEDS A STATED ASSUMPTION, not a confirmed
fact. Unlike every other NSE-sourced module in this project, this
endpoint's response carries NO real disclosure/broadcast timestamp - only
the quarter label itself. `filing_date` here is therefore an ASSUMED
value: quarter-end + 15 calendar days, based on AMFI's well-documented
practice of disclosing quarterly AAUM within roughly two weeks of
quarter-end (the same "Average AUM" page's own text confirms a real
disclosure-lag convention exists, without stating the exact day count) -
NOT independently verified against a specific dated circular. Flagged
explicitly so a future caller does not treat this date with the same
confidence as `insider_trading.py`'s real broadcast timestamps; a
14-15-day uncertainty band is immaterial for a quarterly-frequency
signal, but would matter for anything higher-frequency built on this data.

WHY ONLY ETF/INDEX-FUND CATEGORIES, not the full industry-wide response
this same endpoint also carries (SEBI-rationalized categories for every
scheme type - equity, debt, hybrid, solution-oriented, funds-of-funds).
This module scopes to the passive-fund categories the priority-queue item
("ETF flow / AUM data, passive-flow proxy") actually asked for -
`_is_passive_category` matches on substring "etf" or "index fund"
case-insensitively, which is necessary (not just convenient) because the
category LABEL itself has drifted across the ~20-year window (pre-2017
rationalization: "Other ETFs", "GOLD ETFs"; post-2017: "Exchange Traded
Funds (ETFs) - Equity ETF", "Exchange Traded Funds (ETFs) - Gold ETF",
"Other Scheme - Index Funds", "Other Scheme - Other  ETFs" - all
confirmed present as real category strings in the live response, not
assumed). The full industry-wide response remains available from the
same endpoint if a future hypothesis needs it - not fetched here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

SCHEMA = {
    "amfi_code": "Int64", "scheme_name": "string", "amc_name": "string",
    "category": "string", "period_end": "datetime64[ns]", "filing_date": "datetime64[ns]",
    "average_aum_lakhs": "float64",
}

_PASSIVE_CATEGORY_RE = re.compile(r'etf|index fund', re.I)
_ASSUMED_DISCLOSURE_LAG_DAYS = 15
# Two real label shapes, confirmed live 2026-08-24: a quarterly range
# ("April - June 2026", the norm since the quarter ended Dec 31, 2010) and
# a single month ("March 2010", real AMFI practice up to September 2010 -
# confirmed by the "Average AUM" page's own stated note, and by real
# single-month labels actually appearing at fyId>=18 in the live fetch).
# The range form is tried first since it's the common case.
_QUARTER_LABEL_RE = re.compile(r'([A-Za-z]+)\s*-\s*([A-Za-z]+)\s+(\d{4})')
_MONTH_LABEL_RE = re.compile(r'^([A-Za-z]+)\s+(\d{4})$')


@dataclass(frozen=True)
class AumRecord:
    amfi_code: int
    scheme_name: str
    amc_name: str
    category: str
    period_end: pd.Timestamp
    filing_date: pd.Timestamp
    average_aum_lakhs: float


def _period_end_from_label(selected_period: str) -> pd.Timestamp | None:
    """"April - June 2026" -> 2026-06-30; "March 2010" -> 2010-03-31. Real
    month-end handled via pandas' own MonthEnd offset, not a hand-rolled
    days-in-month table."""
    label = selected_period.strip()
    m = _QUARTER_LABEL_RE.search(label)
    if m:
        _, end_month_name, year = m.groups()
        month_label = f"{end_month_name} 1, {year}"
    else:
        m = _MONTH_LABEL_RE.match(label)
        if not m:
            return None
        month_name, year = m.groups()
        month_label = f"{month_name} 1, {year}"
    try:
        month_start = pd.Timestamp(month_label)
    except ValueError:
        return None
    return month_start + pd.offsets.MonthEnd(0)


def parse_average_aum_response(payload: dict) -> list[AumRecord]:
    """One `average-aum-schemewise?strType=Categorywise&...` response,
    filtered to ETF/index-fund categories only (see module docstring)."""
    selected_period = payload.get("selectedPeriod")
    if not selected_period:
        return []
    period_end = _period_end_from_label(selected_period)
    if period_end is None:
        return []
    filing_date = period_end + pd.Timedelta(days=_ASSUMED_DISCLOSURE_LAG_DAYS)

    records: list[AumRecord] = []
    for group in payload.get("data", []):
        category = group.get("SchemeCat_Desc")
        if not category or category == "Total" or not _PASSIVE_CATEGORY_RE.search(category):
            continue
        amc_name = group.get("Mfname") or ""
        for scheme in group.get("schemes", []):
            amfi_code = scheme.get("AMFI_Code")
            scheme_name = scheme.get("SchemeNAVName")
            aum = (scheme.get("AverageAumForTheMonth") or {}).get(
                "ExcludingFundOfFundsDomesticButIncludingFundOfFundsOverseas")
            if amfi_code is None or scheme_name is None or aum is None:
                continue
            records.append(AumRecord(
                amfi_code=int(amfi_code), scheme_name=scheme_name, amc_name=amc_name,
                category=category, period_end=period_end, filing_date=filing_date,
                average_aum_lakhs=float(aum),
            ))
    return records
