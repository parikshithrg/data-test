"""Corporate announcement bulletins (M&A, contract wins, dividend/bonus/
buyback, and other economically material disclosures), sourced from NSE's
own `corporate-announcements` endpoint. Confirmed live, 2026-08-24: the
best-behaved source found this session - `symbol` AND `from_date`/
`to_date` BOTH genuinely filter (unlike `shareholding.py`'s master
endpoint or `credit_ratings.py`'s feed, where at least one of those was
unreliable), real history for a sampled large-cap (RELIANCE) back to 2004,
consistent with this project's own `primary` split's train_start.

WHY A CATEGORY FILTER IS NECESSARY, not optional - a real scale finding.
The unfiltered feed is enormous: a single peak month (May 2024, results/
dividend season) returned 22,099 records industry-wide, dominated by
administrative noise ("Loss of Share Certificates", "Certificate under
SEBI (Depositories and Participants) Regulations", "Trading Window",
generic "Updates") that NSE's own `desc` field already labels distinctly
from the economically material categories. `RELEVANT_CATEGORIES` is an
INCLUDE-list of real category strings confirmed present in a live sample
(not a keyword regex) spanning M&A (Acquisition, Amalgamation/Merger,
Scheme of Arrangement, Open Offer, Memorandum of Understanding/
Agreements, Diversification/Disinvestment), contract wins (Bagging/
Receiving of orders/contracts), capital actions (Dividend, Dividend
Updates, Date of payment of dividend, Bonus, Buyback, Public Announcement
- Buyback of Shares, Stock split, Rights Issue, Qualified Institutional
Placement, Allotment/Issue of Securities, Record Date, Book Closure), and
distress signals (Corporate Insolvency Resolution Process, Defaults on
Payment of Interest/Principal, Pendency of Litigation(s)/dispute(s),
Change in Management). Credit-rating categories ("Credit Rating",
"Credit Rating- Revision") are DELIBERATELY EXCLUDED - already covered by
`credit_ratings.py`, a genuinely richer multi-agency source for that
specific category, not duplicated here.

STATED LIMITATION: `RELEVANT_CATEGORIES` was built from a live sample
(several months across 2024), not an exhaustive taxonomy audit across 20+
years - NSE's own category labels have almost certainly drifted some
over that span (the same lesson `index_reconstitution.py`'s numbering-
convention drift and `shareholding.py`'s taxonomy-era drift both taught
this project already). A category present in an older filing under
slightly different wording would be silently dropped by this filter,
not fabricated - stated here so a future session knows to re-audit the
category list against the real fetched output before assuming completeness.

WHY `an_dt` (NOT `sort_date`) IS THE CAUSAL FIELD - `an_dt` is NSE's own
"announcement date/time" (the real broadcast moment); `sort_date` is
derived from it (same value, different string format) and used here only
for chronological sorting convenience. Neither should be confused with
any date mentioned INSIDE `attchmntText` (e.g. a future record date for a
dividend) - that text describes a FUTURE event, not something knowable
before `an_dt`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

SCHEMA = {
    "symbol": "string", "filing_date": "datetime64[ns]", "category": "string",
    "company_name": "string", "isin": "string", "industry": "string",
    "description": "string", "seq_id": "string",
}

# Confirmed present in a real live sample - BUT the label taxonomy itself
# drifted across at least two real eras, found live 2026-08-24 only AFTER
# the first full-scale fetch showed a suspicious 2013 floor despite raw
# industry-wide data existing back to 2008: a 2010 sample used "Interim
# Dividend" (not "Dividend"), "Buy back" with a space (not "Buyback"),
# "Allotment of Equity Shares"/"Allotment of shares" (not "Allotment of
# Securities"), "Change in Board of Directors" (not "Change in
# Management") - the modern labels below were confirmed already stable by
# a May-2017 sample, so the "old era" is roughly pre-2013. Same category
# of drift `shareholding.py`'s 3 XBRL eras and `index_reconstitution.py`'s
# numbering conventions already taught this project - fixed the same way,
# an alias list, not a keyword regex (regex risks false-positive noise
# categories this feed has plenty of). "Demerger" was also missed on the
# first pass (present in both eras) and added here.
RELEVANT_CATEGORIES = {
    # M&A
    "Acquisition", "Amalgamation/Merger", "Scheme of Arrangement", "Demerger",
    "Memorandum of Understanding/Agreements", "Open Offer", "Diversification/Disinvestment",
    # Contract wins
    "Bagging/Receiving of orders/contracts",
    # Capital actions (modern + old-era alias)
    "Dividend", "Interim Dividend", "Dividend Updates", "Date of payment of dividend",
    "Bonus", "Buyback", "Buy back", "Public Announcement - Buyback of Shares",
    "Stock split", "Stock Split/Others", "Rights Issue",
    "Qualified Institutional Placement", "Allotment of Securities",
    "Allotment of Equity Shares", "Allotment of shares", "Issue of Securities",
    "Record Date", "Record Date/Others", "Book Closure",
    # Distress / governance (modern + old-era alias)
    "Corporate Insolvency Resolution Process", "Defaults on Payment of Interest/Principal",
    "Pendency of Litigation(s)/dispute(s) or the outcome impacting the Company",
    "Change in Management", "Change in Board of Directors",
    "Commencement of commercial production/operations",
}


@dataclass(frozen=True)
class AnnouncementRecord:
    symbol: str
    filing_date: pd.Timestamp
    category: str
    company_name: str
    isin: str | None
    industry: str | None
    description: str | None
    seq_id: str


def parse_announcement_record(rec: dict) -> AnnouncementRecord | None:
    """One record from `corporate-announcements`'s JSON array. Returns
    None for records outside `RELEVANT_CATEGORIES` or missing required
    fields - never guesses a category or a missing symbol.

    `seq_id` IS NULL ON EVERY PRE-2013 RECORD - a real finding, not a rare
    edge case: confirmed live, 2026-08-24, that this silently dropped
    EVERY SINGLE record from 2004-2012 on the first full-scale fetch (a
    suspiciously clean cutover at the Dec-2012/Jan-2013 boundary was the
    tell - real economic events like interim dividends and buybacks were
    being disclosed and category-matched correctly that whole period, only
    the `seq_id` requirement silently discarded them). Fixed with a
    synthetic key (`symbol|an_dt`) when the source `seq_id` is missing -
    sufficiently unique in practice (the same company issuing two
    different disclosures at the exact same broadcast second is not a
    real scenario this feed's granularity would produce), and still
    genuinely deterministic/reproducible, unlike a random or row-position
    key would be.
    """
    category = rec.get("desc")
    if category not in RELEVANT_CATEGORIES:
        return None

    symbol = rec.get("symbol")
    an_dt = rec.get("an_dt")
    if not symbol or not an_dt:
        return None
    try:
        filing_date = pd.Timestamp(datetime.strptime(an_dt, "%d-%b-%Y %H:%M:%S"))
    except ValueError:
        return None

    seq_id = rec.get("seq_id")
    key = str(seq_id) if seq_id else f"{symbol}|{an_dt}"

    return AnnouncementRecord(
        symbol=symbol, filing_date=filing_date, category=category,
        company_name=rec.get("sm_name") or "", isin=rec.get("sm_isin") or None,
        industry=rec.get("smIndustry") or None, description=rec.get("attchmntText") or None,
        seq_id=key,
    )
