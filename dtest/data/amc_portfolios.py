"""AMC (mutual fund house) monthly portfolio disclosures - Tier 1 item 4 of
the dataset priority queue. SBI Mutual Fund ONLY this pass (the largest AMC
by AUM, confirmed live 2026-08-26 via AMFI's own `average-aum-schemewise`
endpoint - 15% of industry AAUM for Jan-Mar 2026, ahead of ICICI Prudential
and HDFC). This is real stock-level holdings data (fund X held Y shares of
stock Z on date D) - a genuinely different mechanism from every other
source in this project, which are all transformations of price/OI/delivery/
flow activity.

WHY THIS WAS PREVIOUSLY CALLED A DEAD END (`credit_ratings.py`'s own
docstring: "unlike the AMC-portfolio dead end the same session") AND ISN'T
ONE FOR SBI: the 2026-08-24 scoping checked Axis MF and HDFC and found
JS-single-page-app sites where the real file list loads via an on-click
AJAX call with no visible network payload in the raw page load - concluding
real browser-automation tooling (Playwright/Selenium-class) would be needed
project-wide. SBI's own equivalent page (`sbimf.com/portfolios`) IS a
similar JS-rendered React/jQuery page, but its file list is fetched via one
plain POST fired unconditionally on page load (not gated behind a click),
found by reading `sbimf.com/Content/Service/Portfolios.js` directly (the
site's own un-minified source) rather than reverse-engineering the network
log alone - the exact call is:

    POST https://www.sbimf.com/ajaxcall/CMS/GetSchemePortfolioSheets
    Content-Type: application/json;charset=utf-8
    body: {"FundId": 0, "PSYear": "", "PSMonth": "", "PSFrequency": "Monthly"}

Confirmed live, twice: once from inside a real browser session (to prove
the request shape is right), once via plain `requests` with no cookies/
session/referer at all (to prove no auth or WAF gate exists) - both return
the identical HTML fragment. THE ONE REAL GOTCHA, cost real trial-and-error
to find: `PSFrequency` must be the literal string `"Monthly"`, not empty -
the page's own `$(document).ready` sets that dropdown's displayed text to
"Monthly" as a default BEFORE the first `BindPortfolioSheets()` call reads
it, so an empty/omitted value (the natural first guess) silently returns
"No Records Found" rather than an error, and looks identical to a hard
block. No browser/Playwright is needed anywhere in this pipeline - a
meaningfully different, cheaper outcome than the 2026-08-24 scoping found
for the two AMCs it checked.

REAL COVERAGE, NOT YET FULLY EXPLAINED: 68 distinct files, 2026-07-31 back
to 2013-11-30, but with a genuine unexplained gap - the `PSFrequency=
"Monthly"` feed has nothing between 2016-05-31 and 2023-01-31. Not a
taxonomy-drift artifact of this parser (checked: `PSFrequency="Fortnightly"`
returns a completely different, per-SCHEME set of static links, not
per-date files, so it isn't just "the 2016-2023 monthly files are mislabeled
Fortnightly"). Left as a stated, real limitation - not investigated further
this pass. Two real naming-convention eras, both handled by
`parse_filing_period`: 2023+ uses one consolidated "All Schemes Monthly
Portfolio - as on {DDth Month YYYY}" file per month; 2013-2016 splits into
several files per month ("Equity and Debt ... Scheme Portfolios",
"SDFS (FMPs) ... Scheme Portfolios") with looser date phrasing ("As on 31
December 2013", no ordinal suffix on some).

RAW FILES ONLY THIS PASS, NOT PARSED INTO A HOLDINGS TABLE - the actual
point of this item (per-stock, per-scheme quantity/market-value/%-AUM) is
real, substantial follow-on work, scoped but not built: a real July-2026
workbook was inspected and has ~120 sheets (one per scheme, indexed by an
"Index" sheet mapping scheme code -> short code -> name), each sheet's
holdings table starts several rows down (after a scheme-name/as-on-date
header block), covers multiple instrument-type sections (equity, debt,
money-market, derivatives - each with different column semantics, e.g.
YTM%/YTC% apply to debt rows only) under a header row whose "Name of the
Instrument / Issuer" label visually spans what is actually TWO data
columns (an internal numeric code, then the real name) - a real parsing
job comparable in scope to `index_reconstitution.py`'s PDF work, not a
quick follow-up. `scripts/fetch_amc_portfolios.py` stores every raw
workbook plus a filing-level manifest (title, parsed period, url,
local path) - the manifest IS structured/queryable even before any
holdings parser exists.

FILING DATE IS ASSUMED, NOT CONFIRMED, same caveat class as
`etf_aum.py`'s own 15-day assumption: this feed carries only the "as on"
(portfolio) date, not a real broadcast/disclosure timestamp. SEBI mutual
fund regulations require monthly portfolio disclosure within 10 days of
month-end - `filing_date = period_end + 10 days` is that regulatory
convention, not an independently verified per-file timestamp.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

SCHEMA = {
    "amc_name": "string", "title": "string", "period_end": "datetime64[ns]",
    "filing_date": "datetime64[ns]", "url": "string", "local_filename": "string",
}

AMC_NAME = "SBI Mutual Fund"
SBI_PORTFOLIO_SHEETS_URL = "https://www.sbimf.com/ajaxcall/CMS/GetSchemePortfolioSheets"
_ASSUMED_DISCLOSURE_LAG_DAYS = 10  # SEBI's regulatory filing deadline, not a confirmed broadcast time

_MONTH_ALIASES = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "sept": "September", "oct": "October",
    "nov": "November", "dec": "December",
}
# "as on 31st July 2026" / "As on 31 December 2013" / "as on 30th Sep 2023" -
# ordinal suffix and month abbreviation are both optional, confirmed necessary
# by real title text spanning both eras (see module docstring).
_PERIOD_RE = re.compile(
    r"as\s+on\s+(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})", re.I)

_ANCHOR_RE = re.compile(r'<a href="([^"]+)"[^>]*>([^<]*)</a>')


@dataclass(frozen=True)
class PortfolioFiling:
    amc_name: str
    title: str
    period_end: pd.Timestamp | None
    filing_date: pd.Timestamp | None
    url: str


def parse_filing_period(title: str) -> pd.Timestamp | None:
    """"All Schemes Monthly Portfolio - as on 31st July 2026" -> 2026-07-31.
    Returns None if the title doesn't match the expected "as on DATE" shape -
    the caller's job to decide whether that's acceptable (real titles are
    dropped, never guessed at)."""
    m = _PERIOD_RE.search(title)
    if not m:
        return None
    day, month_raw, year = m.groups()
    month = _MONTH_ALIASES.get(month_raw.strip().lower(), month_raw.strip().title())
    try:
        return pd.Timestamp(f"{day} {month} {year}")
    except ValueError:
        return None


def parse_portfolio_sheets_response(html: str, amc_name: str = AMC_NAME) -> list[PortfolioFiling]:
    """One `GetSchemePortfolioSheets` HTML-fragment response. Each real row
    renders TWO anchors sharing the same href (the title link and the
    "Download" button) - deduped by href, first-seen title kept."""
    seen: dict[str, str] = {}
    for href, text in _ANCHOR_RE.findall(html):
        title = text.strip()
        if not title or title.lower() == "download":
            continue
        if href not in seen:
            seen[href] = title

    out = []
    for url, title in seen.items():
        period_end = parse_filing_period(title)
        filing_date = (period_end + pd.Timedelta(days=_ASSUMED_DISCLOSURE_LAG_DAYS)
                       if period_end is not None else None)
        out.append(PortfolioFiling(amc_name=amc_name, title=title, period_end=period_end,
                                    filing_date=filing_date, url=url))
    return out
