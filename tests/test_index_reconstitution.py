"""NIFTY index reconstitution PDF parsing, verified against REAL NSE
Indices press releases saved as fixtures (2010, 2012, 2000 - see
`dtest/data/index_reconstitution.py`'s own module docstring for the live
probes these came from, 2026-08-24). Synthetic PDFs were tried first and
rejected: `fitz`'s own text-insertion API does not reproduce the same
per-cell block layout real Word-exported PDFs have, so a synthetic
fixture doesn't exercise the same code path as production input - real
saved bytes are the only fixture that does."""

from __future__ import annotations

from pathlib import Path

from dtest.data.index_reconstitution import parse_press_release_pdf

FIXTURES = Path(__file__).parent / "fixtures" / "index_reconstitution"


def test_two_index_sections_same_company_both_directions():
    # Real 2010 release: Jubilant Life Sciences demerger triggers a swap
    # in TWO different indices, same company excluded from both.
    pdf_bytes = (FIXTURES / "2010_two_index_sections.pdf").read_bytes()
    events = parse_press_release_pdf(pdf_bytes)
    by_index: dict[str | None, list] = {}
    for e in events:
        by_index.setdefault(e.index_name, []).append(e)

    assert set(by_index) == {"S&P CNX 500 Index", "CNX Midcap Index"}
    assert len(events) == 4

    sp500 = {(e.action, e.symbol) for e in by_index["S&P CNX 500 Index"]}
    assert sp500 == {("excluded", "JUBILANT"), ("included", "SUNTECK")}

    midcap = {(e.action, e.symbol) for e in by_index["CNX Midcap Index"]}
    assert midcap == {("excluded", "JUBILANT"), ("included", "IFCI")}

    assert all(e.effective_date == "November 25, 2010" for e in events)


def test_no_numbered_header_falls_back_to_none_index_name():
    # Real 2012 release: only ONE index changes, and its name is only
    # mentioned in prose ("...changes in CNX Infrastructure Index...highlighted"
    # sentence), never as a standalone numbered header line - index_name
    # correctly comes back None rather than a wrong guess.
    pdf_bytes = (FIXTURES / "2012_no_header_line_index_name.pdf").read_bytes()
    events = parse_press_release_pdf(pdf_bytes)
    assert len(events) == 4
    assert all(e.index_name is None for e in events)
    assert {(e.action, e.symbol) for e in events} == {
        ("excluded", "ABB"), ("excluded", "SUZLON"),
        ("included", "CESC"), ("included", "PTC"),
    }
    assert all(e.effective_date == "February 01, 2013" for e in events)


def test_pre_symbol_era_table_yields_no_events():
    # Real 2000 release: tables have Company Name + Industry Name (or
    # Impact Cost + Market Cap) but NO Symbol column at all - this era
    # must yield zero events, never a guessed/merged symbol.
    pdf_bytes = (FIXTURES / "2000_no_symbol_column.pdf").read_bytes()
    events = parse_press_release_pdf(pdf_bytes)
    assert events == []
