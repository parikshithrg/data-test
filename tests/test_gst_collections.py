"""GST monthly collection PDF parsing, verified against REAL single pages
extracted from GSTN's own "9 Years of GST" report (see
`dtest/data/gst_collections.py`'s own module docstring) - one fixture per
real header-casing variant found live (all-caps "MONTH" for the two
earliest fiscal years, title-case "Month"/"Months" for every year since).
Synthetic PDFs weren't used, same reasoning as
`test_index_reconstitution.py`'s own fixture choice: a real saved page
exercises the real per-line text-extraction order, a hand-built one
wouldn't prove anything about the actual source document.

`load_gst_collections` is tested separately against a small synthetic
CSV, same pattern as `test_gsec_yields.py` - that half has no PDF
involved, so a synthetic fixture is fine there."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dtest.data.gst_collections import parse_9years_report_pdf, load_gst_collections

FIXTURES = Path(__file__).parent / "fixtures" / "gst_collections"


def test_all_caps_month_header_fy2017_18():
    # The real bug this fixture exists to catch: FY2017-18/2018-19 use
    # "MONTH" (all caps) while every later year uses "Month"/"Months" -
    # an exact-match check silently dropped both years (21/105 months)
    # until this was fixed to match case-insensitively.
    pdf_bytes = (FIXTURES / "fy2017_18_month_allcaps.pdf").read_bytes()
    records = parse_9years_report_pdf(pdf_bytes)
    dates = sorted(r.date for r in records)
    assert dates[0] == pd.Timestamp("2017-07-01")
    assert dates[-1] == pd.Timestamp("2018-03-01")
    assert len(records) == 9  # GST launched mid fiscal year - only 9 months


def test_title_case_month_header_fy2019_20():
    pdf_bytes = (FIXTURES / "fy2019_20_month_titlecase.pdf").read_bytes()
    records = parse_9years_report_pdf(pdf_bytes)
    assert len(records) == 12
    apr = next(r for r in records if r.date == pd.Timestamp("2019-04-01"))
    assert apr.values["cgst_rs_crore"] == 21163.0
    assert apr.values["total_rs_crore"] == 113865.0


def test_months_plural_header_and_trailing_asterisk_stripped():
    pdf_bytes = (FIXTURES / "fy2025_26_months_plural.pdf").read_bytes()
    records = parse_9years_report_pdf(pdf_bytes)
    assert len(records) == 12
    # "IGST *" and "Comp Cess *" must resolve to the plain columns, not
    # be silently skipped as an unrecognized row label.
    apr = next(r for r in records if r.date == pd.Timestamp("2025-04-01"))
    assert apr.values["igst_total_rs_crore"] == 115259.0
    assert apr.values["cess_total_rs_crore"] == 13451.0


def test_igst_domestic_and_imports_distinguished_from_igst_total():
    pdf_bytes = (FIXTURES / "fy2019_20_month_titlecase.pdf").read_bytes()
    records = parse_9years_report_pdf(pdf_bytes)
    apr = next(r for r in records if r.date == pd.Timestamp("2019-04-01"))
    assert apr.values["igst_domestic_rs_crore"] == 31444.0
    assert apr.values["igst_imports_rs_crore"] == 23289.0
    assert apr.values["igst_total_rs_crore"] == 54733.0


@pytest.fixture
def gst_dir(tmp_path):
    pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01"],
        "cgst_rs_crore": [30000.0, 31000.0], "sgst_rs_crore": [38000.0, 39000.0],
        "igst_total_rs_crore": [85000.0, 86000.0],
        "igst_domestic_rs_crore": [45000.0, 46000.0], "igst_imports_rs_crore": [40000.0, 40000.0],
        "cess_total_rs_crore": [12000.0, 12000.0],
        "cess_domestic_rs_crore": [11000.0, 11000.0], "cess_imports_rs_crore": [1000.0, 1000.0],
        "total_rs_crore": [165000.0, 168000.0],
    }).to_csv(tmp_path / "GST_COLLECTIONS_MONTHLY.csv", index=False)
    return tmp_path


def test_load_gst_collections(gst_dir):
    df = load_gst_collections(gst_dir)
    assert len(df) == 2
    assert df.iloc[0]["total_rs_crore"] == 165000.0


def test_load_gst_collections_missing_file_returns_empty_with_correct_schema(tmp_path):
    df = load_gst_collections(tmp_path)
    assert df.empty
    assert list(df.columns) == [
        "date", "cgst_rs_crore", "sgst_rs_crore", "igst_total_rs_crore",
        "igst_domestic_rs_crore", "igst_imports_rs_crore", "cess_total_rs_crore",
        "cess_domestic_rs_crore", "cess_imports_rs_crore", "total_rs_crore",
    ]
