"""Per-stock EQUITY holdings extracted from Axis Mutual Fund's real monthly
portfolio workbooks (already fetched by `scripts/fetch_amc_portfolios.py`
into `data/amc_portfolios/axis/`) - Tier 1 item 4's actual point ("actual
mutual fund buy/sell/hold by stock, monthly"), not just the filing-list
manifest `dtest/data/amc_portfolios.py` already builds.

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

REAL LAYOUT, inspected across a 2022 and a 2026 real workbook (different
schemes, ~4 years apart) before writing any parser code - stable across
that whole window, not assumed from one file: row 0 = [scheme_code,
scheme_name]; a later row reads "Monthly Portfolio Statement as on
<Month DD, YYYY>" - a second, independent source for the filing's own
"as on" date (`sheet_as_on_date`), kept alongside but never substituted
for the filename-derived `period_end` a caller already has from
`amc_portfolios.py`'s manifest; one header row (label text ignored -
column POSITION is what's parsed, since the label itself has drifted,
e.g. a stray literal "null" 9th column seen in one real 2022 file); then
a stream of SECTION-HEADER rows (a single populated cell, e.g. "Equity &
Equity related", "Debt Instruments", "Money Market Instruments",
"Derivatives") and SUB-SECTION rows ("(a) Listed / awaiting listing on
Stock Exchanges") that do NOT change section state, and DATA rows.

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
module's `_row_values` filters both identically, so `parse_axis_scheme_
sheet` itself is agnostic to which library produced its input rows; only
the caller (`scripts/build_amc_equity_holdings.py`) needs to pick the
right reader per file.

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
_TOTAL_ROW_RE = re.compile(r"^(sub\s*|grand\s*)?total$", re.I)
_SUBSECTION_RE = re.compile(r"^\(?[a-z]\)?\s", re.I)  # "(a) Listed...", "b) Unlisted..."
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


def parse_axis_scheme_sheet(rows: Iterable[tuple]) -> list[EquityHolding]:
    """`rows` is an iterable of row-value tuples, e.g.
    `worksheet.iter_rows(values_only=True)` for a single scheme's sheet
    (works identically whether that sheet is the only one in a per-scheme
    workbook, or one of many in a consolidated all-schemes workbook).
    Extracts ONLY equity holdings - see module docstring for why every
    other asset class is deliberately skipped."""
    scheme_code = scheme_name = None
    sheet_as_on_date = None
    in_equity = False
    out: list[EquityHolding] = []

    for row in rows:
        vals = _row_values(row)
        if not vals:
            continue

        if scheme_code is None and len(vals) >= 2 and not isinstance(vals[0], (int, float)):
            scheme_code, scheme_name = str(vals[0]), str(vals[1])
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
            if not _SUBSECTION_RE.match(first):
                in_equity = bool(_EQUITY_SECTION_RE.match(first))
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
                scheme_code=scheme_code, scheme_name=scheme_name,
                sheet_as_on_date=sheet_as_on_date,
                instrument_code=str(code), instrument_name=str(name), isin=str(isin).strip(),
                industry=str(industry), quantity=float(qty), market_value_lakhs=float(mval),
                pct_to_nav=float(pct),
            ))
        except (TypeError, ValueError):
            continue

    return out
