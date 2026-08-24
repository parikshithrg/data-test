"""Earnings transcript filtering/parsing, verified against real observed
NSE `corporate-announcements` blurb shapes (see
`dtest/data/earnings_transcripts.py`'s own module docstring for the live
probe, 2026-08-24)."""

from __future__ import annotations

import io
import zipfile

import fitz
import pytest

from dtest.data.earnings_transcripts import (
    extract_pdf_text,
    extract_transcript_text,
    is_transcript_announcement,
    parse_transcript_filing,
)


def _make_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    return doc.tobytes()


def _rec(**overrides) -> dict:
    rec = {
        "an_dt": "31-May-2024 23:10:49", "desc": "Analysts/Institutional Investor Meet/Con. Call Updates",
        "symbol": "NAZARA", "sm_name": "Nazara Technologies Limited",
        "sm_isin": "INE00Q001011", "smIndustry": "Media & Entertainment",
        "attchmntText": "Nazara Technologies Limited has informed the Exchange about Transcript",
        "attchmntFile": "https://nsearchives.nseindia.com/corporate/NAZARA_transcript.pdf",
        "seq_id": "105123456",
    }
    rec.update(overrides)
    return rec


def test_parse_basic_fields():
    f = parse_transcript_filing(_rec())
    assert f.symbol == "NAZARA"
    assert f.company_name == "Nazara Technologies Limited"
    assert f.isin == "INE00Q001011"
    assert f.seq_id == "105123456"
    assert f.source_url.endswith(".pdf")


def test_real_transcript_phrasings_all_pass():
    # Real blurb shapes found live across eras - modern (desc-labeled) and
    # old-era (filed under the generic "Updates" desc, no distinguishing
    # category at all - the whole reason the filter is on attchmntText).
    phrasings = [
        "Company X has informed the Exchange about Transcript",
        "Company X has informed the Exchange regarding Analysts/Institutional Investor Meet/Con. Call Updates-Transcript of Earnings Call",
        "Company X has submitted to the Exchange a copy of transcript of earnings call of Company X for the quarter ended December 31, 2012.",
        "Company X has submitted to the Exchange a copy of the conference call transcripts in respect of Company X dated January 11, 2013.",
    ]
    for text in phrasings:
        assert is_transcript_announcement(text), f"{text!r} should be recognised"


def test_agm_transcript_excluded():
    # Real false-positive class found live: AGM/shareholder-meeting
    # transcripts are not earnings calls.
    excluded = [
        "Route Mobile Limited has informed the Exchange regarding 'Transcript of 17th Annual General Meeting of Route Mobile Limited'",
        "Elecon Engineering Company Limited has informed the Exchange regarding 'Transcript of the 61st Annual General Meeting of the Company'",
    ]
    for text in excluded:
        assert not is_transcript_announcement(text), f"{text!r} should be excluded (AGM, not earnings call)"


def test_non_transcript_announcement_dropped():
    for text in ("Company X has informed the Exchange about Investor Presentation",
                 "Company X has informed the Exchange about Link of Recording",
                 "Company X has informed the Exchange about Schedule of meet"):
        assert not is_transcript_announcement(text)
        assert parse_transcript_filing(_rec(attchmntText=text)) is None


def test_missing_required_fields_returns_none():
    assert parse_transcript_filing(_rec(symbol=None)) is None
    assert parse_transcript_filing(_rec(an_dt=None)) is None
    assert parse_transcript_filing(_rec(attchmntFile=None)) is None


def test_placeholder_attachment_url_returns_none():
    # Real 2004-2009 shape found live: NSE's feed still matches on
    # attchmntText but has "-" (its own placeholder) as attchmntFile -
    # no real file ever existed for this era, not a broken link.
    assert parse_transcript_filing(_rec(attchmntFile="-")) is None


def test_null_seq_id_gets_synthetic_key():
    f = parse_transcript_filing(_rec(seq_id=None))
    assert f is not None
    assert f.seq_id == "NAZARA|31-May-2024 23:10:49"


def test_zip_wrapped_pdf_extracted():
    # Real shape found live, 2026-08-24: every pre-~2018 transcript
    # attachment is a zip with one PDF inside, not a raw PDF - the first
    # full-scale fetch silently failed on ALL of them until this was fixed.
    pdf_bytes = _make_pdf_bytes("Transcript of the Earnings Call")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("transcript.pdf", pdf_bytes)
    text = extract_transcript_text(buf.getvalue())
    assert "Transcript of the Earnings Call" in text


def test_raw_pdf_still_works_unwrapped():
    pdf_bytes = _make_pdf_bytes("Q4 results discussion")
    text = extract_transcript_text(pdf_bytes)
    assert "Q4 results discussion" in text
    assert text == extract_pdf_text(pdf_bytes)


def test_zip_with_no_pdf_member_raises():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("notes.txt", "not a pdf")
    with pytest.raises(ValueError):
        extract_transcript_text(buf.getvalue())
