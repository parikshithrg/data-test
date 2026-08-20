"""Per-symbol quarterly financial results (P&L only - see below), sourced
from NSE's own `corporates-financial-results` endpoint and its two linked
per-filing detail formats. Confirmed live, 2026-08-20, before building
anything: real filing/broadcast TIMESTAMPS (the actual disclosure moment,
not just the period it covers - the one thing yfinance's fundamentals
completely lack, and the reason yfinance was ruled out for this project),
real P&L line items, coverage back to ~2007 on 87.5% of a 40-symbol pilot
sample (`runs/probe_financial_results_coverage/`), median 95 filings per
covered symbol.

WHY POINT-IN-TIME CORRECTNESS is the whole reason this module exists
rather than just calling yfinance. A quarter's results are NOT knowable
the instant the quarter ends - there is a real reporting lag (weeks to a
couple months under SEBI LODR) before the actual disclosure. This module's
`filing_date` field (from NSE's own `filingDate`/`broadCastDate`) is the
ONLY date a causal feature is allowed to key off - never `period_end`.
Same date-level (not intraday) causality convention every other signal in
this project already uses: a filing disclosed on date D is treated as
knowable as of D's own close (conservative even for filings disclosed
during market hours, since T+1-open fill still respects the information
being public before that fill), usable for a signal firing D+1 onward via
the engine's own T+1-open fill - no new mechanic needed.

TWO SOURCE FORMATS, one normalized schema. Filings roughly pre-~2017 link
to a static HTML "detail" page (`resultDetailedDataLink`); newer filings
link to a structured XBRL XML document (`xbrl`) instead. Both are parsed
here into the SAME flat schema (`FIELDS` below) - callers never need to
know which source format a given row came from.

OLD HTML FORMAT - PARSING IS INHERENTLY BEST-EFFORT, not a clean table.
The page nests one giant `<tr>` per section with every (label, amount)
pair as sibling `<td>`s rather than one row per line item (confirmed by
inspecting the real DOM, not assumed) - `_extract_after` flattens every
cell in the target table to one ordered list and pulls the value cell
immediately following a known label substring. Label wording has surely
drifted across 15+ years of filings (pre-2016 Clause-41 era vs. later
Ind-AS format) - a field that doesn't match ANY known label returns NaN
rather than raising, same "missing data blocks, never guesses" convention
every other source in this project uses. Do not expect 100% field
completeness on old-format filings; expect it to be reasonably high on
material fields (revenue, net profit, EPS) since NSE's report template is
fairly stable on those specific lines even as the surrounding form varies.

NEW XBRL FORMAT - real, standard, machine-parseable tags
(`in-bse-fin:RevenueFromOperations`, `in-bse-fin:ProfitLossForPeriod`,
`in-bse-fin:BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations`,
etc. - confirmed against a real filing's actual tag list, not guessed from
XBRL spec docs). `contextRef="OneD"` is used throughout - confirmed via the
filing's own `<xbrli:context>` definitions that "OneD" spans exactly the
filing's own single-quarter window (matches `fromDate`..`toDate` from the
metadata endpoint); "FourD" is the four-quarter/cumulative context, never
used here since this module reports single-quarter figures only, matching
the old-format parser's own "Net sales/income... for the period" (not
cumulative) convention.

NO BALANCE SHEET. Confirmed live, 2026-08-20: quarterly result filings
under SEBI LODR carry P&L + segment reporting only - no debt, cash, total
assets. A "Paid-up equity share capital" field IS present in both formats
(kept here as `paidup_equity_capital`), and a "Reserve excluding
Revaluation Reserves" field exists in the OLD format's own label set but
was empty ("-") on the one filing inspected - kept as `reserves`,
best-effort, expect frequent NaN. Book-value-per-share (needed for a
P/B-based value factor) is therefore NOT reliably available from this
source; a value hypothesis built on this data should lean on trailing EPS
(P/E) rather than assume reserves/book-value coverage without checking it
first on the real fetched data.

CONSOLIDATED VS STANDALONE: both are logged separately as they appear
(the `consolidated` bool field, straight from NSE's own flag) - never
merged or silently preferred one over the other. A caller building a
feature must pick one explicitly (this project's convention: state the
choice, don't default silently) - not decided here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SCHEMA = {
    "symbol": "string", "filing_date": "datetime64[ns]", "period_end": "datetime64[ns]",
    "period_start": "datetime64[ns]", "consolidated": "bool", "source_format": "string",
    "seq_number": "string", "revenue": "float64", "total_income": "float64",
    "total_expenses": "float64", "profit_before_tax": "float64", "net_profit": "float64",
    "eps_basic": "float64", "eps_diluted": "float64", "paidup_equity_capital": "float64",
    "reserves": "float64",
}

# Old-format HTML spans (at least) TWO genuinely different real templates
# across 15+ years - confirmed live, 2026-08-20, not assumed: a pre-~2016
# "Clause 41"-era template ("Net Sales/Income from Operation", "Basic EPS
# after Extraordinary items", "Net Profit (+) / Loss (-) for the period")
# and a later Ind-AS-old template ("Net sales/income from operations",
# "Basic EPS for continued and discontinued operations", "Net Profit /
# (Loss) for the period") - different wording, different casing, even
# singular-vs-plural on the same line. Each field maps to an ORDERED list
# of RULES, most specific/preferred first; a rule is a tuple of substrings
# that must ALL appear in the SAME cell (handles phrasing that varies in
# extra words, e.g. "before tax" appears with different prefixes across
# eras). Matching is case-INSENSITIVE and whitespace-normalized - the two
# eras don't even agree on capitalization for the same line.
# A rule starting with "=" requires the CELL, once stripped, to equal the
# phrase exactly (after the leading "=" is dropped) rather than merely
# contain it - needed for "Total Income"/"Total Expenses" specifically,
# which are real substrings of OTHER, unrelated lines nearby ("Total
# income from operations (net)", "...10% of the total expenses relating
# to..." inside the Other Expenses line's own parenthetical note) -
# confirmed as a real false-positive by testing against a real filing, not
# a hypothetical edge case.
_HTML_LABELS: dict[str, list[tuple[str, ...]]] = {
    "revenue": [("sales/income from operation",)],
    "total_income": [("=total income",)],
    "total_expenses": [("=total expenses",), ("=total expenditure",)],
    "profit_before_tax": [("activities before tax",)],
    "net_profit": [("net profit", "for the period")],
    "eps_basic": [("basic eps", "continued and discontinued"),
                 ("basic eps", "after extraordinary")],
    "eps_diluted": [("diluted eps", "continued and discontinued"),
                    ("diluted eps", "after extraordinary")],
    "paidup_equity_capital": [("paid-up equity share capital",), ("paid up equity share capital",)],
    "reserves": [("reserve", "excluding revaluation")],
}

# New-format XBRL: field name -> exact `in-bse-fin:` tag, `contextRef="OneD"`
# only (the single-quarter context, confirmed against real context defs).
_XBRL_TAGS: dict[str, str] = {
    "revenue": "RevenueFromOperations",
    "total_income": "Income",
    "total_expenses": "Expenses",
    "profit_before_tax": "ProfitBeforeTax",
    "net_profit": "ProfitLossForPeriod",
    "eps_basic": "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
    "eps_diluted": "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
    "paidup_equity_capital": "PaidUpValueOfEquityShareCapital",
}


def _to_float(text: str) -> float:
    text = text.strip()
    if text in ("", "-", "--", "NA", "N.A."):
        return float("nan")
    text = text.replace(",", "")
    neg = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    try:
        val = float(text)
    except ValueError:
        return float("nan")
    return -val if neg else val


def _flatten_table_cells(table) -> list[str]:
    cells = []
    for row in table.find_all("tr"):
        for c in row.find_all(["td", "th"]):
            text = re.sub(r"\s+", " ", c.get_text(strip=True))
            if text:
                cells.append(text)
    return cells


def _extract_after(cells_lower: list[str], cells_orig: list[str], rules: list[tuple[str, ...]]) -> float:
    for rule in rules:
        exact = len(rule) == 1 and rule[0].startswith("=")
        parts = (rule[0][1:],) if exact else rule
        for i, cell in enumerate(cells_lower[:-1]):
            matched = cell.strip() == parts[0] if exact else all(part in cell for part in parts)
            if matched:
                return _to_float(cells_orig[i + 1])
    return float("nan")


def parse_old_html(html_text: str) -> dict:
    """Best-effort field extraction from an old-format detail page. Returns
    NaN for any field whose label isn't found - never raises on a
    format/wording mismatch, since 15+ years of filings span at least two
    genuinely different real templates (see module docstring)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, "html.parser")
    tables = soup.find_all("table")
    # The P&L table is whichever one ACTUALLY YIELDS the revenue figure -
    # not merely whichever table contains the label text somewhere.
    # Real pages nest tables inside tables, and an outer wrapper table's
    # flattened cells can contain the label merged into one giant blob
    # (no adjacent label/value cell pair to extract) while a genuinely
    # split-out inner table, found later in document order, has the real
    # clean (label, value) pairs - confirmed by inspecting a real 2008-era
    # filing where the first candidate table matched the label but
    # extraction failed, and a later table succeeded. Trying extraction on
    # each candidate and keeping the first one that actually produces a
    # number is robust to this, not just "found the label somewhere."
    target_cells: list[str] = []
    for t in tables:
        cells = _flatten_table_cells(t)
        if not any("sales/income from operation" in c.lower() for c in cells):
            continue
        cells_lower_try = [c.lower() for c in cells]
        if not pd.isna(_extract_after(cells_lower_try, cells, _HTML_LABELS["revenue"])):
            target_cells = cells
            break
    if not target_cells:
        return {k: float("nan") for k in _HTML_LABELS}
    cells_lower = [c.lower() for c in target_cells]
    return {field: _extract_after(cells_lower, target_cells, rules)
            for field, rules in _HTML_LABELS.items()}


def parse_xbrl(xml_text: str) -> dict:
    """Field extraction from a new-format XBRL filing via its real
    `in-bse-fin:` tags, `contextRef="OneD"` (single-quarter) only."""
    out: dict[str, float] = {}
    for field, tag in _XBRL_TAGS.items():
        pattern = (
            rf'<in-bse-fin:{tag}\b[^>]*contextRef="OneD"[^>]*>([^<]*)</in-bse-fin:{tag}>'
        )
        m = re.search(pattern, xml_text)
        out[field] = _to_float(m.group(1)) if m else float("nan")
    out["reserves"] = float("nan")   # no reserves/book-value tag found in this taxonomy
    return out


@dataclass(frozen=True)
class FilingMeta:
    symbol: str
    filing_date: pd.Timestamp
    period_end: pd.Timestamp
    period_start: pd.Timestamp
    consolidated: bool
    source_format: str
    seq_number: str
    detail_url: str | None


def parse_metadata_record(symbol: str, rec: dict) -> FilingMeta | None:
    """Normalize one raw record from the `corporates-financial-results`
    metadata endpoint. Returns None for a record with no usable detail
    link (nothing to parse) rather than a half-populated row."""
    filing_date = pd.to_datetime(rec.get("filingDate"), format="%d-%b-%Y %H:%M", errors="coerce")
    if pd.isna(filing_date):
        filing_date = pd.to_datetime(rec.get("broadCastDate"), errors="coerce")
    period_end = pd.to_datetime(rec.get("toDate"), format="%d-%b-%Y", errors="coerce")
    period_start = pd.to_datetime(rec.get("fromDate"), format="%d-%b-%Y", errors="coerce")
    if pd.isna(filing_date) or pd.isna(period_end):
        return None

    detail_url = rec.get("resultDetailedDataLink") or rec.get("xbrl")
    source_format = "html_old" if rec.get("resultDetailedDataLink") else "xbrl_new"
    consolidated = str(rec.get("consolidated", "")).strip().lower().startswith("consol")

    return FilingMeta(
        symbol=symbol, filing_date=filing_date, period_end=period_end,
        period_start=period_start, consolidated=consolidated,
        source_format=source_format, seq_number=str(rec.get("seqNumber", "")),
        detail_url=detail_url,
    )


def load_financials(symbol: str, fundamentals_dir: Path) -> pd.DataFrame:
    """Read one symbol's cached quarterly-results file, sorted by
    `filing_date` (the point-in-time key - never `period_end`). Returns an
    empty, correctly-typed frame if the symbol was never fetched or has no
    cached filings, same "missing coverage is normal, not an error"
    convention every other data source in this project uses."""
    path = Path(fundamentals_dir) / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame({k: pd.Series(dtype=v) for k, v in SCHEMA.items()})
    df = pd.read_csv(path, parse_dates=["filing_date", "period_end", "period_start"])
    return df.sort_values("filing_date", kind="stable").reset_index(drop=True)


def as_of_latest(df: pd.DataFrame, as_of: pd.Timestamp, *, consolidated: bool | None = None) -> pd.Series | None:
    """The most recent filing whose `filing_date <= as_of` - the single
    point-in-time-correct lookup every fundamentals-based feature in this
    project must go through, never a direct `period_end`-indexed read
    (which would leak the filing's own reporting lag). `consolidated=None`
    takes whichever type filed most recently as of that date (mixed);
    pass True/False to require one specific type only - the caller's
    explicit choice, not defaulted here (see module docstring)."""
    if df.empty:
        return None
    eligible = df[df["filing_date"] <= as_of]
    if consolidated is not None:
        eligible = eligible[eligible["consolidated"] == consolidated]
    if eligible.empty:
        return None
    return eligible.iloc[-1]
