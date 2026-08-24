"""Earnings call / analyst-meet transcript full text, sourced from NSE's own
`corporate-announcements` endpoint (the same feed as `corporate_announcements.py`)
- confirmed live, 2026-08-24, that NSE has NO dedicated transcript-only
endpoint or category. Real transcripts are filed under an inconsistent mix of
`desc` values across eras: the modern label is "Analysts/Institutional
Investor Meet/Con. Call Updates" (from ~2017 on), but that same bucket also
carries schedule intimations, presentations, and audio-recording links for
the SAME event - and the pre-2017 era files transcripts under the generic,
otherwise-noise "Updates" bucket with no distinguishing category at all.

THE REAL, ROBUST FILTER IS THE `attchmntText` BLURB ITSELF, not `desc` -
confirmed live: every genuine transcript filing's auto-generated blurb
contains the word "transcript" ("...has informed the Exchange about
Transcript", "...regarding Transcript of the Earnings Call", "...has
submitted...a copy of transcript of earnings call..."). A small, real false-
positive class exists too - AGM/shareholder-meeting transcripts (~0.7% of
raw "transcript" matches in a live sample) are NOT earnings calls and are
explicitly excluded via `EXCLUDE_RE`.

SCALE, confirmed live via a full metadata-only sweep 2004-2026 (no PDFs
downloaded): 19,382 real transcript filings. Heavily back-loaded - 2004-2015
combined is only 272 filings; 2022-2026 alone is 17,391 (~90% of the total) -
tracking SEBI's own tightening of investor-meet/transcript disclosure
requirements over that period, a real regulatory-adoption curve, not a
data-quality artifact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

SCHEMA = {
    "symbol": "string", "filing_date": "datetime64[ns]", "company_name": "string",
    "isin": "string", "industry": "string", "seq_id": "string",
    "source_url": "string", "text_path": "string", "char_count": "int64",
}

# Matches the auto-generated NSE blurb, not the (unreliable, era-drifted)
# `desc` category field. See module docstring for why.
TRANSCRIPT_RE = re.compile(r"transcript", re.I)
EXCLUDE_RE = re.compile(
    r"annual general meeting|\bAGM\b|shareholders meeting|court|scheme of arrangement", re.I
)


@dataclass(frozen=True)
class TranscriptFiling:
    symbol: str
    filing_date: pd.Timestamp
    company_name: str
    isin: str | None
    industry: str | None
    seq_id: str
    source_url: str


def is_transcript_announcement(attchmnt_text: str | None) -> bool:
    if not attchmnt_text:
        return False
    return bool(TRANSCRIPT_RE.search(attchmnt_text)) and not EXCLUDE_RE.search(attchmnt_text)


def parse_transcript_filing(rec: dict) -> TranscriptFiling | None:
    """One record from `corporate-announcements`'s JSON array. Returns None
    unless `attchmntText` reads as a genuine earnings-call transcript (see
    `is_transcript_announcement`) and the record carries a real PDF URL -
    never guesses either.

    `seq_id` fallback mirrors `corporate_announcements.py`'s own fix: null
    on pre-2013-era records, so a synthetic `symbol|an_dt` key is used
    instead of dropping the filing.
    """
    if not is_transcript_announcement(rec.get("attchmntText")):
        return None

    symbol = rec.get("symbol")
    an_dt = rec.get("an_dt")
    pdf_url = rec.get("attchmntFile")
    # "-" is NSE's own placeholder for "no attachment on file" - confirmed
    # live, 2026-08-24, on every one of a real 2004-2009 sample (65
    # candidate transcripts, genuinely zero recoverable): the feed still
    # matches on `attchmntText`/`desc` but never carried a real file link
    # for this era. Rejected here explicitly, not left to fail downstream
    # as an ambiguous fetch error.
    if not symbol or not an_dt or not pdf_url or pdf_url == "-":
        return None
    try:
        filing_date = pd.Timestamp(pd.to_datetime(an_dt, format="%d-%b-%Y %H:%M:%S"))
    except ValueError:
        return None

    seq_id = rec.get("seq_id")
    key = str(seq_id) if seq_id else f"{symbol}|{an_dt}"

    return TranscriptFiling(
        symbol=symbol, filing_date=filing_date, company_name=rec.get("sm_name") or "",
        isin=rec.get("sm_isin") or None, industry=rec.get("smIndustry") or None,
        seq_id=key, source_url=pdf_url,
    )


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Concatenated page text, in real reading order. PyMuPDF's default
    `get_text()` is reliable for these transcripts (prose paragraphs, not
    the multi-column tables `index_reconstitution.py` needed block-sorting
    for)."""
    import fitz

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def extract_transcript_text(content: bytes) -> str:
    """`extract_pdf_text`, but transparent to NSE's older zip-wrapped
    filings. Found live, 2026-08-24, on the full-scale fetch's first pass:
    every pre-~2018 transcript attachment is a `.zip` (one PDF inside, no
    other files seen in a live sample) rather than a raw PDF - the first
    full run silently failed on ALL of them (a PDF-only parser choking on
    zip bytes), wiping out the entire 2004-2017 span even though stage 1's
    metadata sweep had found the real filings. Picks the first `.pdf`
    member inside the zip; raises (caller treats as a failed fetch, same
    as any other bad PDF) if none is found - never guesses at a non-PDF
    member."""
    import zipfile
    import io

    if content[:2] == b"PK":  # zip local-file-header magic bytes
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            pdf_names = [n for n in zf.namelist() if n.lower().endswith(".pdf")]
            if not pdf_names:
                raise ValueError("zip attachment has no PDF member")
            content = zf.read(pdf_names[0])
    return extract_pdf_text(content)
