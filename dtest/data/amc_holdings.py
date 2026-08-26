"""Per-stock EQUITY holdings extracted from real monthly portfolio
workbooks (already fetched by `scripts/fetch_amc_portfolios.py` into
`data/amc_portfolios/{axis,sbi}/`) - Tier 1 item 4's actual point ("actual
mutual fund buy/sell/hold by stock, monthly"), not just the filing-list
manifest `dtest/data/amc_portfolios.py` already builds.

ONE SHARED PARSER FOR BOTH AXIS AND SBI, NOT TWO - built against Axis
first, then verified (and corrected) against a real SBI workbook, which
turned out to share the SAME section vocabulary, the SAME "instrument
name is actually two columns" quirk, and the SAME lettered sub-header
convention, despite being a completely different AMC's template. This is
evidently a shared SEBI-mandated disclosure layout, not a coincidence
worth building two parsers around. The one thing that genuinely differs
per AMC is how a sheet states ITS OWN scheme identity (see
`parse_scheme_sheet`'s own docstring) - deliberately left to the caller
rather than guessed per-AMC inside this function.

COVERS ONLY SBI's MODERN TEMPLATE (2023-2026), NOT ITS FULL FILE RANGE -
a real gap found only by running the full build and checking the output's
actual date range, not by the earlier per-file sample checks. SBI's
2013-2016 files use a genuinely different, narrower 6-column holdings
table (`NAME OF THE INSTRUMENT, ISIN, QUANTITY, MARKET VALUE, RATING,
REMARKS` - no separate Industry/%-to-NAV split) that this module's
7-field equity-row shape does not match, so every row from that era is
silently skipped by the `len(vals) < 7` check rather than mis-parsed -
see `scripts/build_amc_equity_holdings.py`'s own module docstring for the
full finding. A legacy-template branch for 2013-2016 is real, unscoped
follow-on work, not started this session.

SCOPED TO EQUITY HOLDINGS ONLY, A DELIBERATE, STATED NARROWING - each
workbook's scheme sheet covers MULTIPLE asset classes (equity, debt,
money-market, government securities, derivatives, fixed deposits, cash)
under separate sections, each with a DIFFERENT real column shape (debt/
money-market rows carry a Rating instead of an Industry, and populate
YTM/YTC columns equity rows leave blank - confirmed on a real
`Axis Liquid Fund` workbook) - real, substantial extra work, and not what
this item's own framing asks for ("buy/sell/hold BY STOCK"). Only rows
under the "Equity & Equity related" section are extracted; every other
asset class is deliberately skipped, not silently mis-parsed as equity.

REAL LAYOUT, inspected across a 2022 Axis workbook, a 2026 Axis workbook,
and a 2026 SBI workbook (three different templates) before writing/fixing
parser code - stable across all three, not assumed from one file: an
identity block (AMC-specific shape, see `parse_scheme_sheet`'s own
docstring); Axis's own identity row also carries "Monthly Portfolio
Statement as on <Month DD, YYYY>" - a second, independent source for the
filing's own "as on" date (`sheet_as_on_date`), kept alongside but never
substituted for the filename-derived `period_end` a caller already has
from `amc_portfolios.py`'s manifest (SBI's own "PORTFOLIO STATEMENT AS
ON :" row pairs a label with a real `datetime` value rather than one
combined string - not parsed by this module, `sheet_as_on_date` is simply
None for SBI, a stated, accepted gap since the manifest's `period_end`
remains authoritative regardless); one header row (label text ignored -
column POSITION is what's parsed, since the label itself has drifted,
e.g. a stray literal "null" 9th column seen in one real 2022 Axis file);
then a stream of SECTION-HEADER rows and SUB-SECTION rows.

SECTION-HEADER STATE MACHINE IS DELIBERATELY A WHITELIST OF EXITS, NOT A
WHITELIST OF STAYS - a real correction made after checking SBI, not
assumed safe from Axis alone: `_EQUITY_SECTION_RE` turns equity-tracking
ON; `_NON_EQUITY_SECTION_RE` (a small, real, SHARED vocabulary - "Debt
Instruments", "Money Market Instruments", "Government Securities",
"Derivatives", "TREPS", "Reverse Repo", "Cash & ...", "(Net) (Current)
Assets" - confirmed identical wording across Axis and SBI) turns it OFF;
any OTHER single-cell label (a sub-header) leaves the state UNCHANGED.
This matters because a real SBI workbook has equity sub-headers an
Axis-only design never anticipated - "Equity Shares" and "Foreign
Securities and/or overseas ETF" - which a naive "equity-match-or-exit"
rule would have wrongly treated as exiting the equity section, silently
dropping every foreign/ETF equity holding. The unchanged-by-default rule
handles any such unanticipated sub-header on either AMC without needing
its exact wording hardcoded.

REAL EQUITY DATA ROW SHAPE: the first 7 populated cells are always
[instrument_code, instrument_name, isin, industry, quantity,
market_value_lakhs, pct_to_nav] - confirmed across every real file
inspected, including a genuine template variant found via `xlrd` (legacy
`.xls` files, see below): ESG-branded schemes append 3 MORE trailing
columns (ESG Score, Core ESG Score, a BRSR disclosure URL) after the
standard 7 - the parser takes only the first 7 and ignores anything past
that, rather than requiring an exact column count. The header's own "Name
of the Instrument" label visually spans what is actually TWO data columns
(an internal Axis instrument code, then the real name) - the same quirk
`amc_portfolios.py`'s own module docstring notes for SBI's workbooks,
confirmed here independently on a COMPLETELY DIFFERENT AMC's template,
not a coincidence specific to one vendor - this is evidently a standard
SEBI-mandated disclosure layout, not an AMC-specific quirk. A row with
fewer than 7 populated cells (e.g. a genuine unlisted-equity row missing
its ISIN) is dropped, not guessed at.

TWO REAL FILE FORMATS BOTH NAMED `.xls`, NOT INTERCHANGEABLE - found by
`openpyxl` raising on ~47% of a random 60-file sample, not assumed:
Axis's own CMS serves genuine legacy BIFF-format `.xls` files (needs
`xlrd`) for some months/schemes and modern XLSX-zip content saved under a
`.xls` extension (needs `openpyxl`, since `xlrd` 2.x dropped .xlsx
support entirely) for others - the real format must be detected from the
file's own magic bytes (`PK\x03\x04` = zip/xlsx, `\xd0\xcf\x11\xe0...` =
OLE2/legacy xls), never assumed from the extension. `xlrd` also returns
empty cells as `''` rather than `None` (openpyxl's convention) - this
module's `_row_values` filters both identically, so `parse_scheme_sheet`
itself is agnostic to which library produced its input rows; only the
caller (`scripts/build_amc_equity_holdings.py`) needs to pick the right
reader per file. Confirmed the same two-format split exists for SBI's
own `.xls`/`.xlsx` files too, not just Axis's.

"SUB TOTAL"/"TOTAL" AGGREGATION ROWS APPEAR INSIDE EVERY SECTION,
EXPLICITLY SKIPPED - matched by the first cell's own text, not by
position, since they can appear after either sub-bucket ((a)/(b)) of a
section.

MANY SCHEMES HAVE ZERO EQUITY HOLDINGS BY DESIGN, NOT A PARSING FAILURE -
confirmed on a real `Axis Liquid Fund` workbook, which has no "Equity &
Equity related" section at all (Derivatives, Debt Instruments, Money
Market Instruments only) - a debt/liquid scheme simply holds no equity,
and this parser correctly returns an empty holdings list for it rather
than erroring or guessing.

A REAL DOUBLE-COUNTING BUG, CAUGHT BY AUDITING THE FULL-SCALE OUTPUT, NOT
ASSUMED CORRECT FROM A CLEAN SAMPLE RUN: `amc_portfolios.py`'s own module
docstring documents 13 real (scheme, period_end) filename collisions
(two different real filing-metadata entries, usually differing only in
title CASING, resolving to the same local filename on a case-insensitive
Windows filesystem). `scripts/build_amc_equity_holdings.py` MUST
deduplicate the manifest on a case-insensitive `local_filename` before
parsing, or it silently parses that one physical file twice under two
scheme_name labels - caught live by auditing summed `pct_to_nav` per
(scheme, month): several came out near 2.0 instead of the normal
0.85-1.0 before this fix was in place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

SCHEMA = {
    "scheme_code": "string", "scheme_name": "string", "sheet_as_on_date": "datetime64[ns]",
    "instrument_code": "string", "instrument_name": "string", "isin": "string",
    "industry": "string", "quantity": "float64", "market_value_lakhs": "float64",
    "pct_to_nav": "float64",
}

_EQUITY_SECTION_RE = re.compile(r"^equity\s*(&|and)\s*equity\s*related", re.I)
# Known TOP-LEVEL non-equity sections that end an equity block - a real,
# shared SEBI-mandated vocabulary confirmed identical across two completely
# different AMC templates (Axis and SBI). Deliberately NOT a blacklist of
# "everything else" - equity sections carry their own sub-headers that must
# NOT be mistaken for a section change (see _default-unchanged behavior in
# `parse_scheme_sheet`, and the module docstring's SBI findings: "Equity
# Shares" and "Foreign Securities and/or overseas ETF" are real equity
# SUB-headers, not new sections, on a real SBI workbook).
_NON_EQUITY_SECTION_RE = re.compile(
    r"^(debt instruments|money market instruments|government securities|derivatives"
    r"|treps|reverse repo|cash\s*&|other current assets|net assets|net current assets)", re.I)
_TOTAL_ROW_RE = re.compile(r"^(sub\s*|grand\s*)?total$", re.I)
_ISIN_RE = re.compile(r"^IN[A-Z0-9]{10}$")
_AS_ON_RE = re.compile(r"as\s+on\s+([A-Za-z]+\s+\d{1,2},?\s*\d{4})", re.I)


@dataclass(frozen=True)
class EquityHolding:
    scheme_code: str | None
    scheme_name: str | None
    sheet_as_on_date: pd.Timestamp | None
    instrument_code: str
    instrument_name: str
    isin: str
    industry: str
    quantity: float
    market_value_lakhs: float
    pct_to_nav: float


def _row_values(row: tuple) -> list:
    return [c for c in row if c is not None and str(c).strip() != ""]


def parse_scheme_sheet(
    rows: Iterable[tuple], scheme_code: str | None = None, scheme_name: str | None = None,
) -> list[EquityHolding]:
    """`rows` is an iterable of row-value tuples, e.g.
    `worksheet.iter_rows(values_only=True)` for a single scheme's sheet
    (works identically whether that sheet is the only one in a per-scheme
    Axis workbook, one of many in a consolidated all-schemes workbook, or
    one of the ~120 scheme sheets inside a single SBI workbook). Extracts
    ONLY equity holdings - see module docstring for why every other asset
    class is deliberately skipped.

    `scheme_code`/`scheme_name` SHOULD be supplied by the caller from a
    source authoritative for THAT AMC's real layout (Axis: the filing's
    own manifest entry; SBI: that workbook's own "Index" sheet, which maps
    sheet name -> scheme code -> full name) rather than left to this
    function's own best-effort in-sheet detection, which only recognizes
    Axis's row-0 shape (`[scheme_code, scheme_name]`) - SBI's own identity
    rows use a completely different, two-row layout (AMC name + code on
    one row, "SCHEME NAME :" + name on another) that this function does
    NOT attempt to parse, by design, since the Index-sheet source is
    already authoritative and simpler to use directly."""
    detected_code = detected_name = None
    sheet_as_on_date = None
    in_equity = False
    out: list[EquityHolding] = []

    for row in rows:
        vals = _row_values(row)
        if not vals:
            continue

        if (scheme_code is None and detected_code is None
                and len(vals) >= 2 and not isinstance(vals[0], (int, float))):
            detected_code, detected_name = str(vals[0]), str(vals[1])
            continue

        if sheet_as_on_date is None:
            m = _AS_ON_RE.search(" ".join(str(v) for v in vals))
            if m:
                try:
                    sheet_as_on_date = pd.Timestamp(m.group(1).replace(",", " "))
                except ValueError:
                    pass
                continue

        first = str(vals[0]).strip()

        if len(vals) == 1:
            if _EQUITY_SECTION_RE.match(first):
                in_equity = True
            elif _NON_EQUITY_SECTION_RE.match(first):
                in_equity = False
            # else: an unrecognized single-cell label is a SUB-header
            # ("(a) Listed...", "Equity Shares", "Foreign Securities
            # and/or overseas ETF" - all confirmed real) - section state
            # is deliberately left unchanged, the safer default (see
            # module docstring for why a whitelist of sub-header text
            # would need to keep growing per AMC, while this doesn't).
            continue

        if _TOTAL_ROW_RE.match(first):
            continue

        if not in_equity or len(vals) < 7:
            continue

        # Take only the first 7 fields - some templates (ESG-branded
        # schemes, confirmed real on a live file) append extra trailing
        # columns (ESG Score, Core ESG Score, a BRSR disclosure URL) after
        # the standard 7; those are ignored, not parsed.
        code, name, isin, industry, qty, mval, pct = vals[:7]
        if not _ISIN_RE.match(str(isin).strip()):
            continue
        try:
            out.append(EquityHolding(
                scheme_code=scheme_code if scheme_code is not None else detected_code,
                scheme_name=scheme_name if scheme_name is not None else detected_name,
                sheet_as_on_date=sheet_as_on_date,
                instrument_code=str(code), instrument_name=str(name), isin=str(isin).strip(),
                industry=str(industry), quantity=float(qty), market_value_lakhs=float(mval),
                pct_to_nav=float(pct),
            ))
        except (TypeError, ValueError):
            continue

    return out
