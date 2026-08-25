"""Monthly GST tax collection (Tier 6 item 15 of the dataset priority
queue), sourced from GSTN's own retrospective PDF report "9 Years of GST:
A Statistical Report on the Completion of Nine Years of GST" (2017-2026),
published on `gst.gov.in`'s own GST Statistics download page and linked
from the GST Council's own site (`gstcouncil.gov.in`) - confirmed live,
2026-08-25.

WHY THIS SOURCE, NOT `Gross_Net_Tax_collection.xlsx` (the other GST-
collection file on the same download page) - that file only covers
Apr 2024 onward (27 months). This PDF's own "Payment" section (pages
16-25) has a genuine, complete monthly table for EVERY month since GST's
launch, July 2017 through March 2026 (104 months across 9 fiscal-year
tables) - real depth this project has not matched with any other single
PDF source. Confirmed text-native (not a scanned image) - `fitz`'s plain
`page.get_text()` already returns each table's cells in correct row-major
reading order (no block-position sorting needed, unlike
`index_reconstitution.py`'s press releases - checked directly against
the real extracted text before writing this parser, not assumed).

`www.gst.gov.in` ITSELF IS WAF-PROTECTED (an F5/BIG-IP-style "Request
Rejected" page on a bare `requests.get`, confirmed live) - the download
page had to be driven through a real browser session to find the actual
file URL. The file itself, once found, lives on a DIFFERENT host
(`tutorial.gst.gov.in`) with no such protection - a plain `requests.get`
against `tutorial.gst.gov.in/offlineutilities/gst_statistics/9YearsReport
.pdf` works with no browser needed, same as this project's other real
XLSX/PDF downloads.

TABLE SHAPE, per fiscal-year page: a `Month`/`Months` header row (9 months
for FY2017-18, GST's launch year, 12 for every year since) followed by 9
value rows in a FIXED order: CGST, SGST, IGST (total/domestic/imports),
Comp Cess (total/domestic/imports), Total - real, verified across all 9
FY pages, not assumed. Three real formatting quirks handled, found by
running the parser and checking the actual record count (84, not the
expected 105) rather than trusting a clean-looking first pass: (1) the
column header reads all-caps "MONTH" for FY2017-18/2018-19 but title-case
"Month"/"Months" every year after - the two earliest fiscal years were
silently dropped entirely (21 of 105 months) until this was matched
case-insensitively; (2) month labels use a dash in most years ("Apr-24")
but an apostrophe in FY2018-19 ("Sep'18") - both accepted; (3) the IGST/
Cess TOTAL rows carry a trailing `*` footnote marker in most years
("IGST *", "Comp Cess *") - stripped, not treated as a different row.

Cess became largely inapplicable from the September 2025 return period
(a real GST Council rate-rationalization decision stated in the source
document's own footnote, not a parsing artifact) - Cess values from
Oct-2025 onward are genuinely near-zero (e.g. 0.03-0.07 for the smallest
sub-line), not missing data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz
import pandas as pd

SCHEMA = {
    "date": "datetime64[ns]",
    "cgst_rs_crore": "float64", "sgst_rs_crore": "float64",
    "igst_total_rs_crore": "float64", "igst_domestic_rs_crore": "float64",
    "igst_imports_rs_crore": "float64",
    "cess_total_rs_crore": "float64", "cess_domestic_rs_crore": "float64",
    "cess_imports_rs_crore": "float64",
    "total_rs_crore": "float64",
}

_FY_HEADER_RE = re.compile(r'^Payments\s*[—–-]\s*FY\s*(\d{4})-(\d{2})$')
_MONTH_RE = re.compile(r"^([A-Za-z]{3})[-'](\d{2})$")
_MONTH_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
# Row labels, in the fixed order this source always uses, each mapped to
# its SCHEMA column - trailing "*" footnote markers stripped before match.
_ROW_LABELS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'^CGST$'), "cgst_rs_crore"),
    (re.compile(r'^SGST$'), "sgst_rs_crore"),
    (re.compile(r'^IGST$'), "igst_total_rs_crore"),
    (re.compile(r'^IGST\s*[-–]\s*Domestic$'), "igst_domestic_rs_crore"),
    (re.compile(r'^IGST\s*[-–]\s*Imports$'), "igst_imports_rs_crore"),
    (re.compile(r'^Comp\s*Cess$'), "cess_total_rs_crore"),
    (re.compile(r'^Cess\s*[-–]\s*Domestic$'), "cess_domestic_rs_crore"),
    (re.compile(r'^Cess\s*[-–]\s*Imports$'), "cess_imports_rs_crore"),
    (re.compile(r'^Total$'), "total_rs_crore"),
]


@dataclass(frozen=True)
class MonthlyCollection:
    date: pd.Timestamp
    values: dict[str, float]


def _num(s: str) -> float | None:
    s = s.strip().replace(",", "")
    if not s or s in ("-", "—"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_fy_page(lines: list[str]) -> list[MonthlyCollection]:
    """One fiscal-year "Payments — FY XXXX-XX" page's lines -> one
    `MonthlyCollection` per real calendar month found on it."""
    try:
        month_hdr_idx = next(i for i, l in enumerate(lines) if l.upper() in ("MONTH", "MONTHS"))
    except StopIteration:
        return []

    months: list[pd.Timestamp] = []
    i = month_hdr_idx + 1
    while i < len(lines):
        m = _MONTH_RE.match(lines[i])
        if not m:
            break
        mon, yy = m.group(1), int(m.group(2))
        months.append(pd.Timestamp(year=2000 + yy, month=_MONTH_NUM[mon], day=1))
        i += 1
    if not months or lines[i] != "TOTAL":
        return []
    i += 1  # skip the header's own "TOTAL" column label
    n = len(months)

    per_month: list[dict[str, float]] = [{} for _ in months]
    while i < len(lines):
        label = re.sub(r'\s*\*\s*$', '', lines[i]).strip()
        matched_col = None
        for pattern, col in _ROW_LABELS:
            if pattern.match(label):
                matched_col = col
                break
        if matched_col is None:
            i += 1
            continue
        i += 1
        values = [_num(lines[i + k]) for k in range(n)]
        # the (n+1)-th cell is that row's own FY TOTAL column - a real
        # cross-check, not persisted here (see fetch_gst_collections.py).
        i += n + 1
        for k, v in enumerate(values):
            if v is not None:
                per_month[k][matched_col] = v
        if matched_col == "total_rs_crore":
            break  # Total is always the last of the 9 rows on this source

    return [MonthlyCollection(date=d, values=v) for d, v in zip(months, per_month) if v]


def parse_9years_report_pdf(pdf_bytes: bytes) -> list[MonthlyCollection]:
    """Every monthly collection row across all fiscal-year pages in the
    real "9 Years of GST" PDF's own byte content."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out: list[MonthlyCollection] = []
    for page in doc:
        text = page.get_text()
        lines = [l.strip() for l in text.split("\n")]
        header_idx = next(
            (i for i, l in enumerate(lines) if _FY_HEADER_RE.match(l)), None,
        )
        if header_idx is None:
            continue
        out.extend(_parse_fy_page(lines[header_idx:]))
    return out


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in SCHEMA.items()})


def load_gst_collections(gst_collections_dir: Path) -> pd.DataFrame:
    """Monthly GST collection by head (CGST/SGST/IGST/Cess, domestic vs
    imports, and the grand Total), July 2017 - March 2026."""
    path = Path(gst_collections_dir) / "GST_COLLECTIONS_MONTHLY.csv"
    if not path.exists():
        return _empty()
    df = pd.read_csv(path, parse_dates=["date"])
    for col, dtype in SCHEMA.items():
        if dtype != "datetime64[ns]":
            df[col] = df[col].astype(dtype)
    return df.sort_values("date", kind="stable").reset_index(drop=True)
