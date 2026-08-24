"""NIFTY index reconstitution events (additions/exclusions), sourced from
NSE Indices Limited's own press-release archive at niftyindices.com.
Confirmed live, 2026-08-24, before building anything: a single static page
(`niftyindices.com/press-release`) carries the FULL press-release list,
1998-2026, 1,473 entries, real dated announcements with direct PDF links -
no JS-rendering needed, unlike the AMC-portfolio source this project
scoped and set aside the same day for exactly that reason.

REAL COVERAGE FLOOR IS ~2010, NOT 1998 - stated precisely, not glossed
over. Every press-release PDF is genuine, machine-readable text (confirmed
even on a 2000-vintage sample - not a scanned image), but the TABLE FORMAT
changed materially over time: releases from ~2010 onward use a stable
`Sr. No. / Company Name / Symbol` table (spot-checked 2010, 2012, 2015,
2026 - all parse cleanly with this module); releases before that use
company-NAME-only tables with no ticker column at all (spot-checked 2000,
2003, 2005, 2007, 2008, 2009 - none have a Symbol column), AND the row
layout within those older tables is itself inconsistent even WITHIN one
document (some rows one cell per line, others multiple cells crammed onto
one line) - real, verified inconsistency, not a hypothetical. This module
deliberately does NOT attempt to parse or guess symbols for that pre-2010
era: `parse_press_release_pdf` only emits an event when the source PDF's
own table explicitly declared a `Symbol` column, the same "missing data
blocks, never guesses" convention every other source in this project
uses. A pre-2010 hypothesis on this data needs a separate company-name-to-
symbol resolution step this module does not provide.

WHY TWO DATES, BOTH KEPT. `announcement_date` (from the press-release
list's own posting date) is the ONLY causal field - the date the change
became public knowledge, same point-in-time convention as every other
source here. `effective_date` (parsed from the PDF body, e.g. "effective
from September 30, 2026") is real but NOT causal on its own - it is
FUTURE relative to announcement (typically 2-6 weeks out) and describes
when the actual index-membership change (and the associated passive-fund
rebalancing flow) happens. Both matter for this hypothesis category
specifically: a signal must key off `announcement_date` for legality, but
the economic mechanism this data exists to test (a predictable flow
around reconstitution) centers on `effective_date`.

TEXT-EXTRACTION ORDERING: a real PDF layout quirk, found by inspecting a
2010-era release, not assumed. Naive `page.get_text()` reading order
dumped every table in a page to the END, after all the section headers/
prose - correct content, wrong order, which would silently mismatch a
table to the wrong index-name header if read naively. Fixed by extracting
`page.get_text("blocks")` and sorting by vertical position - confirmed
this recovers true visual reading order on the affected sample, and does
not change anything on samples that were already fine.

ROW-BOUNDARY DETECTION: rather than trust ad-hoc "stop markers" between a
table's own rows and the surrounding prose/notes/boilerplate that follows
it (a real, ultimately open-ended list across 16+ years of documents),
this module determines each table's real column count ONCE, from its own
header line (by counting recognized secondary-column keywords: `symbol`,
`industry name`, `impact cost`, `market capital`), then reads EXACTLY that
many cells per data row - a serial-number line starts a row, and the row
ends after that fixed cell count regardless of what odd text might follow
in the source document. This is what let a single, uniform parser handle
every sampled document without a hand-tuned stop-marker list per era.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import fitz
import pandas as pd

SCHEMA = {
    "symbol": "string", "company_name": "string", "index_name": "string",
    "action": "string", "announcement_date": "datetime64[ns]", "effective_date": "string",
    "source_url": "string",
}

_HEADER_RE = re.compile(r'^\s*(?:\(\d+\)|\d+[\.\)]|[a-z]\))\s+([A-Za-z][A-Za-z0-9 &/\-\.\']*?)\s*:?\s*$')
_HEADER_NUM_ONLY_RE = re.compile(r'^\s*(?:\(\d+\)|\d+[\.\)]|[a-z]\))\s*$')
_FOLLOWING_RE = re.compile(r'\bbeing (excluded|included)\b', re.I)
_COMPANY_NAME_ANCHOR_RE = re.compile(r'company\s*name', re.I)
_SERIALNUM_RE = re.compile(r'^\d+\.?$')
_EFFECTIVE_RE = re.compile(r'effective from ([A-Za-z]+\s+\d{1,2},?\s*\d{4})', re.I)
_SECONDARY_COL_KEYWORDS = ("symbol", "industry name", "impact cost", "market capital")
# A real NSE ticker: all-caps letters/digits plus & - . (M&M, GVT&D, L&TFH,
# BAJAJ-AUTO, 3MINDIA), at least one letter (never purely numeric - real
# tickers never are, and a bare number here is almost always a stray
# serial-number line, not a symbol).
_SYMBOL_RE = re.compile(r'^(?=.*[A-Z])[A-Z0-9&\-\.]{1,20}$')
# Real index names are short proper nouns ("Nifty Free Float Midcap 100");
# `Note:` explanations use the exact same numbered-list shape ("1. BSE Ltd.
# has been included in Nifty 50 index as the 6-month average free-float
# market capitalization ...") and were confirmed live, 2026-08-24, to
# corrupt `current_index` for every table that followed in the same
# document (18,464-row full-scale fetch: 788 rows had `index_name` set to
# note prose, not a real index name, all traced to this). Length + a
# handful of prose-only connector words reliably tell the two apart -
# every real index name observed across the whole fetch is under 50 chars
# and none contain these connectors.
_NOTE_PROSE_MARKERS = (" has been ", " have been ", " due to ", " on account of ",
                      " as it ", " which ", " because ", " ranks ", "average free-float")


def _looks_like_index_name(text: str) -> bool:
    if len(text) > 50:
        return False
    low = f" {text.lower()} "
    return not any(marker in low for marker in _NOTE_PROSE_MARKERS)


@dataclass(frozen=True)
class ReconstitutionEvent:
    index_name: str | None
    action: str
    company_name: str
    symbol: str
    effective_date: str | None


def _ordered_text(pdf_bytes: bytes) -> str:
    """Page text in true visual reading order (see module docstring's
    TEXT-EXTRACTION ORDERING note - naive get_text() misorders some real
    documents)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    lines = []
    for page in doc:
        blocks = page.get_text("blocks")
        for b in sorted(blocks, key=lambda b: (round(b[1], 1), b[0])):
            lines.append(b[4])
    return "\n".join(lines)


def parse_press_release_pdf(pdf_bytes: bytes) -> list[ReconstitutionEvent]:
    """Symbol-level add/exclude events from one press-release PDF's raw
    bytes. Only emits events for tables with a confirmed `Symbol` column -
    see the module docstring's "REAL COVERAGE FLOOR" section for why.
    """
    text = _ordered_text(pdf_bytes)
    lines = [re.sub(r'\s+', ' ', l.strip()) for l in text.split('\n')]
    n = len(lines)
    eff_match = _EFFECTIVE_RE.search(re.sub(r'\s+', ' ', text))
    default_effective_date = eff_match.group(1) if eff_match else None

    events: list[ReconstitutionEvent] = []
    current_index: str | None = None
    i = 0
    while i < n:
        line = lines[i]
        if not line:
            i += 1
            continue

        m = _HEADER_RE.match(line)
        if m and not _FOLLOWING_RE.search(line) and _looks_like_index_name(m.group(1)):
            current_index = m.group(1).strip()
            i += 1
            continue
        if _HEADER_NUM_ONLY_RE.match(line) and i + 1 < n and lines[i + 1] \
                and not _FOLLOWING_RE.search(lines[i + 1]) and not _HEADER_RE.match(lines[i + 1]) \
                and _looks_like_index_name(lines[i + 1]):
            current_index = lines[i + 1].strip().rstrip(':')
            i += 2
            continue

        fm = _FOLLOWING_RE.search(line)
        if not fm:
            i += 1
            continue
        action = fm.group(1).lower()

        # locate the "Company Name" anchor within a reasonable window
        j = i + 1
        found_anchor = None
        while j < n and j - i < 15:
            if _COMPANY_NAME_ANCHOR_RE.search(lines[j]):
                found_anchor = j
                break
            if _HEADER_RE.match(lines[j]) or _FOLLOWING_RE.search(lines[j]):
                break
            j += 1
        if found_anchor is None:
            i += 1
            continue

        # scan header cells up to the first serial-number line, counting
        # recognized secondary columns (this determines cells-per-row)
        k = found_anchor + 1
        secondary_cols: set[str] = set()
        symbol_col_index = None
        col_count_seen = 0
        while k < n and not _SERIALNUM_RE.match(lines[k]):
            if lines[k]:
                low = lines[k].lower()
                for kw in _SECONDARY_COL_KEYWORDS:
                    if kw in low and kw not in secondary_cols:
                        secondary_cols.add(kw)
                        if kw == "symbol":
                            symbol_col_index = col_count_seen
                        col_count_seen += 1
            if _HEADER_RE.match(lines[k]) or _FOLLOWING_RE.search(lines[k]):
                break
            k += 1
            if k - found_anchor > 10:
                break
        if k >= n or not _SERIALNUM_RE.match(lines[k]):
            i = found_anchor + 1
            continue

        ncols = 1 + len(secondary_cols)  # company name + secondary columns
        symbol_cell_index = (1 + symbol_col_index) if symbol_col_index is not None else None

        k += 1  # move past the first serial-number line already matched
        rows: list[list[str]] = []
        if symbol_cell_index == 1 and ncols == 2:
            # Symbol-pattern-driven, not fixed-width: a company's official
            # name may wrap across multiple physical lines (e.g. "Indian
            # Railway Catering And Tourism" / "Corporation Ltd." / "IRCTC")
            # - confirmed live, 2026-08-24, on the full-scale fetch, where
            # naive fixed-2-cells-per-row consumption misparsed every such
            # wrapped name (85/18,048 rows, several cascading into
            # misaligned neighbouring rows too). Consume name lines until
            # one matches a real ticker shape - company names are always
            # Title Case in these documents, so they never accidentally
            # match the all-caps `_SYMBOL_RE`.
            while k < n:
                name_parts: list[str] = []
                found_symbol = False
                while k < n:
                    if lines[k] and _SYMBOL_RE.match(lines[k]):
                        found_symbol = True
                        break
                    if _HEADER_RE.match(lines[k]) or _FOLLOWING_RE.search(lines[k]) \
                            or _HEADER_NUM_ONLY_RE.match(lines[k]):
                        break  # ran into the next section without ever finding a symbol
                    if lines[k]:
                        name_parts.append(lines[k])
                    k += 1
                    if len(name_parts) > 5:
                        break  # a real company name never wraps this many lines
                if not found_symbol or not name_parts:
                    break  # genuine parse failure for this table - stop, emit nothing more
                rows.append([" ".join(name_parts), lines[k]])
                k += 1
                p = k
                while p < n and not lines[p]:
                    p += 1
                if p < n and _SERIALNUM_RE.match(lines[p]):
                    k = p + 1
                    continue
                k = p
                break
        else:
            # fixed-width fallback for column layouts this module never
            # emits events for anyway (no confirmed Symbol column)
            while True:
                cells: list[str] = []
                while len(cells) < ncols and k < n:
                    if lines[k]:
                        cells.append(lines[k])
                    k += 1
                if len(cells) < ncols:
                    break
                rows.append(cells)
                p = k
                while p < n and not lines[p]:
                    p += 1
                if p < n and _SERIALNUM_RE.match(lines[p]):
                    k = p + 1
                    continue
                k = p
                break

        if symbol_cell_index is not None:
            for cells in rows:
                symbol = cells[symbol_cell_index] if symbol_cell_index < len(cells) else None
                if symbol:
                    events.append(ReconstitutionEvent(
                        index_name=current_index, action=action, company_name=cells[0],
                        symbol=symbol, effective_date=default_effective_date,
                    ))
        i = k

    return events
