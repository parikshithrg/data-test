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

AXIS MUTUAL FUND ADDED 2026-08-26 - a real correction to the 2026-08-24
"needs browser automation" finding, and the best AMC of the 4 checked so
far (SBI, ICICI Prudential, HDFC, Axis). The 2026-08-24 finding was about
the PAGE UI (true - `axismf.com/statutory-disclosures` does need a click
to reveal the "Portfolios" section before its data API ever fires), not
the underlying DATA, which turned out fully open once the real calls were
found (via a hooked `window.fetch`, since this site's JS is a minified
Next.js bundle, not readable source like SBI's):

    POST https://www.axismf.com/cms/token
    body: {} (empty)
    -> {"data": {"token": "Bearer <opaque>", "updatedAt": "2025-12-09T..."}}

    POST https://www.axismf.com/cms/get-scheme-documents
    Authorization: <the token above>
    body: {"sdType": "yearMonthSchemeDocs", "sdID": "sdMonthSchemePortfolio"}
    -> {"data": {"documentList": [{"docuementURL": ..., "documentName": ...,
        "documentPostedDate": ...}, ...]}}   # "docuementURL" - the site's
                                              # own real typo, not ours

CONFIRMED THE TOKEN IS STATIC, NOT SESSION-BOUND - the single most
valuable finding here: the token's own `updatedAt` field read
"2025-12-09", ~9 months before this was checked, proving it is a
long-lived public/anonymous credential, not issued fresh per browser
session. A plain `requests.post()` with an empty body and no prior page
visit returns the byte-identical token, which then authorizes
`get-scheme-documents` with zero cookies/session/referer - the entire
pipeline needs no browser at all, unlike ICICI, and no bot-wall blocks
it, unlike HDFC.

ONE CALL RETURNS ALL 4,636 DOCUMENTS the feed has ever carried, but a real
correction to an early over-broad read of that number: the RAW list's own
2012-10 floor is NOT the depth of usable "Monthly Portfolio" data - most
of pre-2021 is a completely different, non-standardized naming era
("Portfolio - Axis Liquid Fund for 9 November 2018", bare fragments like
"18-Nov" or "March") that predates Axis's own move to a consistent
"Monthly Portfolio..." title, and is correctly excluded by
`AXIS_MONTHLY_DOCUMENT_RE` rather than mis-parsed. **Real usable depth for
the standardized monthly title convention is 2021-09 to 2026-07** (~5
years) - narrower than the raw feed's headline range, still real filings,
not a bug in the filter. Confirmed live: 3,672 filings actually downloaded
(2026-08-26), 3,658 new + 14 already fetched during scoping, 19 genuine
404s on Axis's own CMS (dead links in their own metadata, not a bug here -
all from the same messy 2022 batch that also had non-standard titles).
TWO document families within that window: 32 "consolidated all-schemes"
files (2022-06 onward, same shape as SBI's headline file) and ~3,600
"Monthly Portfolio" PER-SCHEME files (one workbook per scheme per month) -
still deeper and more complete than SBI's 68 consolidated-only files with
their own 2016-2023 gap. The rest of the 4,636 raw documents are
pre-2021 legacy-format filings plus weekly/adhoc/fortnightly debt-fund
disclosures, both explicitly excluded by `AXIS_MONTHLY_DOCUMENT_RE`.

SIX REAL NAMING SHAPES IN `documentName`, ALL HANDLED - found by iterating
"still failing" samples against real data three times, not assumed from
the first handful checked: (1) consolidated, space-separated numeric date
- "Monthly Portfolio-31 01 26"; (2) consolidated, hyphen-separated numeric
date - "Monthly Portfolio 31-10-2025"; (3) consolidated, TEXTUAL date, no
scheme name - "Monthly Portfolio-30 June 2024"; (4) per-scheme, textual
date with the scheme name in between - "Monthly Portfolio - Axis Nifty
Smallcap 50 Index Fund - 31 January  2026" (real double space before the
year) or with only a 2-digit year - "...- 30 September 25"; (5) "as on"
phrasing in several date sub-formats - "as on Feb 29, 2024" (Month Day,
Year), "as on 31 July 2023" (Day Month Year), "as on 30.09.2023"
(DD.MM.YYYY, Indian day-first order matching every other numeric date in
this module); (6) the plural "Monthly Portfolios" (not just singular)
appears on real filings too, e.g. "Monthly Portfolios - Axis Floater Fund
-31 Dec 2023" (also note the missing space before the day here - a real
quirk, not hypothetical). Scheme names also carry raw HTML entities
(`&amp;` for "&") needing `html.unescape`. `parse_axis_filing_name` tries
shapes in order from most-specific/unambiguous (no scheme name possible)
to least. A real bug caught while widening coverage, not shipped: naively
stripping the comma in "Aug 31,2023" without replacing it with a space
merged the day and year into "312023" - fixed by replacing with a space,
not deleting.

FINAL COVERAGE, confirmed against the real live feed, not assumed from
the fixture set: 3,691/3,694 filtered "Monthly Portfolio" titles parse
(99.92%). The 3 that don't are genuinely malformed source strings, left
null rather than guessed at: a literal "September 022" typo (missing a
digit), an "FOF= August 2022" stray "=" where a "-" belongs, and one
"Monthly Portfolio-April 22" (a consolidated month-only shape seen
exactly once - not worth a dedicated pattern for one file).

13 REAL (SCHEME, PERIOD_END) COLLISIONS among the 3,691 parsed filings -
e.g. "Monthly Portfolio - Axis Overnight Fund - November 2022" and
"Monthly Portfolio-Axis Overnight Fund-November2022" are two DIFFERENT
real URLs that both parse to the identical scheme+month (almost
certainly the same underlying file re-listed under a differently
formatted title, not two genuine holdings snapshots for one month).
`scripts/fetch_amc_portfolios.py` names local files by (scheme,
period_end) rather than the URL's own CMS-generated slug (which is not
guaranteed collision-free once ~3,600 files land in one flat directory),
so these 13 groups naturally keep only the first file seen and skip the
second as "already on disk" - the correct behavior if they are genuinely
duplicate content, not verified byte-for-byte.

`documentPostedDate` IS NOT THE FILING'S OWN "AS ON" DATE - a real trap
avoided, not assumed safe: it looks like a per-filing date but is
actually just the MONTH BUCKET the CMS filed the upload under (e.g. a
document titled "...31 March 2026" carries `documentPostedDate:
"2026-03-01"`, always the 1st of a month) - close to right for files
posted the same month as their "as on" date, but not guaranteed, and
useless for the numeric-only consolidated titles which need their real
date parsed from the name regardless. `period_end` is therefore ALWAYS
derived from `documentName` via `parse_axis_filing_name`, never from
`documentPostedDate`.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import pandas as pd

SCHEMA = {
    "amc_name": "string", "title": "string", "scheme_name": "string",
    "period_end": "datetime64[ns]", "filing_date": "datetime64[ns]",
    "url": "string", "local_filename": "string",
}

SBI_AMC_NAME = "SBI Mutual Fund"
SBI_PORTFOLIO_SHEETS_URL = "https://www.sbimf.com/ajaxcall/CMS/GetSchemePortfolioSheets"

AXIS_AMC_NAME = "Axis Mutual Fund"
AXIS_TOKEN_URL = "https://www.axismf.com/cms/token"
AXIS_DOCUMENTS_URL = "https://www.axismf.com/cms/get-scheme-documents"
AXIS_DOCUMENTS_BODY = {"sdType": "yearMonthSchemeDocs", "sdID": "sdMonthSchemePortfolio"}
# Excludes Weekly/Fortnightly/Adhoc debt-fund disclosures - a different SEBI
# cadence, not the monthly holdings snapshot this item wants (see docstring).
AXIS_MONTHLY_DOCUMENT_RE = re.compile(r"^monthly\s+portfolio", re.I)

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
    scheme_name: str | None = None  # None = filing covers the whole AMC (SBI's shape,
                                     # Axis's 32 consolidated files), not one scheme


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


def parse_portfolio_sheets_response(html_text: str, amc_name: str = SBI_AMC_NAME) -> list[PortfolioFiling]:
    """One `GetSchemePortfolioSheets` HTML-fragment response. Each real row
    renders TWO anchors sharing the same href (the title link and the
    "Download" button) - deduped by href, first-seen title kept."""
    seen: dict[str, str] = {}
    for href, text in _ANCHOR_RE.findall(html_text):
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


# "Monthly Portfolio-31 01 26" (space-separated numeric, no scheme name)
_AXIS_NUMERIC_SPACED_RE = re.compile(
    r"^monthly\s+portfolios?\s*-?\s*(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})\s*$", re.I)
# "Monthly Portfolio 31-10-2025" (hyphen-separated numeric, no scheme name)
_AXIS_NUMERIC_HYPHEN_RE = re.compile(
    r"^monthly\s+portfolios?\s*-?\s*(\d{1,2})-(\d{1,2})-(\d{2,4})\s*$", re.I)
# "Monthly Portfolio(s) - Axis Nifty Smallcap 50 Index Fund - 31 January  2026"
# (textual date, scheme name in between - note \s+ tolerates the real
# double-space quirk seen before some years, the year is sometimes only 2
# digits - "...- 30 September 25", confirmed real on ~200 2023-era filings -
# the plural "Portfolios" also appears, the leading hyphen after "Portfolio"
# is sometimes missing entirely ("Monthly Portfolio Axis ESG Equity Fund -
# 31 July 2023"), month/year are sometimes run together with no space
# ("...-November2022"), and the day sometimes carries an ordinal suffix
# ("...-31st October 2022") - all confirmed real, not hypothetical)
_AXIS_PER_SCHEME_RE = re.compile(
    r"^monthly\s+portfolios?\s*-?\s*(.+?)\s*-\s*(\d{1,2})(?:st|nd|rd|th)?\s*([A-Za-z]+)\s*(\d{2,4})\s*$", re.I)
# "Monthly Portfolio - Axis Overnight Fund - November 2022" (per-scheme, a
# 2022-era batch with no DAY at all, only Month YYYY - confirmed real, not
# hypothetical). No day to parse, so period_end falls back to the calendar
# month-end - the same convention `etf_aum.py`'s own month-only label
# handling already uses for the identical "no day given" situation.
_AXIS_PER_SCHEME_MONTH_ONLY_RE = re.compile(
    r"^monthly\s+portfolios?\s*-?\s*(.+?)\s*-\s*([A-Za-z]+)\.?\s*(\d{4})\s*$", re.I)
# "Monthly Portfolio-30 June 2024" (textual date, NO scheme name - a third
# consolidated shape alongside the two numeric ones above, confirmed real)
_AXIS_CONSOLIDATED_TEXTUAL_RE = re.compile(
    r"^monthly\s+portfolios?\s*-\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{2,4})\s*$", re.I)
# A handful of 2023-era consolidated filings use "as on" phrasing instead of
# the bare-numeric shapes above, in several sub-formats confirmed real, not
# hypothetical: "as on Feb 29, 2024" (Month Day, Year), "as on 31 July 2023"
# (Day Month Year), "as on 30.09.2023" (DD.MM.YYYY) - `dayfirst=True` handles
# the numeric one correctly (Indian date convention, matches every other
# numeric date in this module).
_AXIS_AS_ON_TEXTUAL_RE = re.compile(
    r"as\s+on\s+([A-Za-z]+\s+\d{1,2},?\s*\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})", re.I)
_AXIS_AS_ON_NUMERIC_RE = re.compile(r"as\s+on\s+(\d{1,2}\.\d{1,2}\.\d{4})", re.I)
# "Monthly Portfolio-31.01.2023" (consolidated, dotted numeric, no "as on"
# phrase and no scheme name - confirmed real, a bare variant of the "as on"
# dotted shape above)
_AXIS_CONSOLIDATED_DOTTED_RE = re.compile(
    r"^monthly\s+portfolios?\s*-?\s*(\d{1,2})\.(\d{1,2})\.(\d{4})\s*$", re.I)


def _axis_year(raw: str) -> int:
    year = int(raw)
    return year + 2000 if year < 100 else year


def parse_axis_filing_name(document_name: str) -> tuple[str | None, pd.Timestamp | None]:
    """Returns (scheme_name, period_end) - scheme_name is None for a
    consolidated (all-schemes) filing. Tries the two unambiguous numeric-date
    shapes first (see module docstring) before falling back to the
    textual-date-with-scheme-name shape. Returns (None, None) if nothing
    matches - real titles are dropped, never guessed at."""
    name = document_name.strip()

    m = _AXIS_NUMERIC_SPACED_RE.match(name)
    if m:
        day, month, year = m.groups()
        try:
            return None, pd.Timestamp(year=_axis_year(year), month=int(month), day=int(day))
        except ValueError:
            return None, None

    m = _AXIS_NUMERIC_HYPHEN_RE.match(name)
    if m:
        day, month, year = m.groups()
        try:
            return None, pd.Timestamp(year=_axis_year(year), month=int(month), day=int(day))
        except ValueError:
            return None, None

    m = _AXIS_CONSOLIDATED_DOTTED_RE.match(name)
    if m:
        day, month, year = m.groups()
        try:
            return None, pd.Timestamp(year=int(year), month=int(month), day=int(day))
        except ValueError:
            return None, None

    m = _AXIS_CONSOLIDATED_TEXTUAL_RE.match(name)
    if m:
        day, month_raw, year = m.groups()
        month = _MONTH_ALIASES.get(month_raw.strip().lower(), month_raw.strip().title())
        try:
            return None, pd.Timestamp(f"{day} {month} {_axis_year(year)}")
        except ValueError:
            return None, None

    m = _AXIS_PER_SCHEME_RE.match(name)
    if m:
        scheme_name, day, month_raw, year = m.groups()
        month = _MONTH_ALIASES.get(month_raw.strip().lower(), month_raw.strip().title())
        try:
            period_end = pd.Timestamp(f"{day} {month} {_axis_year(year)}")
        except ValueError:
            return None, None
        return html.unescape(scheme_name).strip(), period_end

    m = _AXIS_AS_ON_NUMERIC_RE.search(name)
    if m:
        day, month, year = m.group(1).split(".")
        try:
            return None, pd.Timestamp(year=int(year), month=int(month), day=int(day))
        except ValueError:
            return None, None

    m = _AXIS_AS_ON_TEXTUAL_RE.search(name)
    if m:
        try:
            return None, pd.Timestamp(m.group(1).replace(",", " "))
        except ValueError:
            return None, None

    m = _AXIS_PER_SCHEME_MONTH_ONLY_RE.match(name)
    if m:
        scheme_name, month_raw, year = m.groups()
        month = _MONTH_ALIASES.get(month_raw.strip().lower(), month_raw.strip().title())
        try:
            period_end = pd.Timestamp(f"1 {month} {_axis_year(year)}") + pd.offsets.MonthEnd(0)
        except ValueError:
            return None, None
        return html.unescape(scheme_name).strip(), period_end

    return None, None


def parse_axis_documents_response(payload: dict, amc_name: str = AXIS_AMC_NAME) -> list[PortfolioFiling]:
    """One `get-scheme-documents` JSON response, filtered to real "Monthly
    Portfolio" filings (`AXIS_MONTHLY_DOCUMENT_RE`) - excludes the
    Weekly/Fortnightly/Adhoc debt-fund disclosures the same feed also
    carries (see module docstring)."""
    out = []
    for doc in payload.get("data", {}).get("documentList", []):
        name = (doc.get("documentName") or "").strip()
        url = doc.get("docuementURL")  # the site's own real typo, not ours
        if not name or not url or not AXIS_MONTHLY_DOCUMENT_RE.match(name):
            continue
        scheme_name, period_end = parse_axis_filing_name(name)
        filing_date = (period_end + pd.Timedelta(days=_ASSUMED_DISCLOSURE_LAG_DAYS)
                       if period_end is not None else None)
        out.append(PortfolioFiling(amc_name=amc_name, title=name, period_end=period_end,
                                    filing_date=filing_date, url=url, scheme_name=scheme_name))
    return out
