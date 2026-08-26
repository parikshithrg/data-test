"""SBI Mutual Fund and Axis Mutual Fund monthly portfolio-filing list
parsing, verified against the real observed response shapes and title
conventions across both AMCs (see `dtest/data/amc_portfolios.py`'s own
module docstring for the live probes, 2026-08-26)."""

from __future__ import annotations

from dtest.data.amc_portfolios import (
    parse_axis_filing_name,
    parse_axis_documents_response,
    parse_filing_period,
    parse_portfolio_sheets_response,
)


def _row(href: str, title: str) -> str:
    return (
        f'<tr><td><a href="{href}" target="_blank">{title}</a></td>'
        f'<td><img src="/x.jpg" /></td><td>2  MB</td>'
        f'<td class="text-center"><a href="{href}" class="primary-button" download="true">Download</a></td></tr>'
    )


def test_parse_filing_period_modern_era_with_ordinal_suffix():
    assert str(parse_filing_period(
        "All Schemes Monthly Portfolio - as on 31st July 2026").date()) == "2026-07-31"


def test_parse_filing_period_older_era_no_ordinal_suffix():
    assert str(parse_filing_period(
        "Equity and Debt Scheme Portfolios - As on 30 November 2013").date()) == "2013-11-30"


def test_parse_filing_period_abbreviated_month():
    assert str(parse_filing_period(
        "All Schemes Monthly Portfolio - as on 30th Sep 2023").date()) == "2023-09-30"


def test_parse_filing_period_unparseable_title_returns_none():
    assert parse_filing_period("Something with no date pattern at all") is None


def test_parse_portfolio_sheets_response_dedupes_title_and_download_anchors():
    html = _row(
        "https://www.sbimf.com/docs/all-schemes-monthly-portfolio---as-on-31st-july-2026.xlsx?sfvrsn=abc_2",
        "All Schemes Monthly Portfolio - as on 31st July 2026",
    )
    filings = parse_portfolio_sheets_response(html)
    assert len(filings) == 1
    f = filings[0]
    assert f.amc_name == "SBI Mutual Fund"
    assert f.title == "All Schemes Monthly Portfolio - as on 31st July 2026"
    assert str(f.period_end.date()) == "2026-07-31"
    assert str(f.filing_date.date()) == "2026-08-10"  # +10 assumed SEBI disclosure lag


def test_parse_portfolio_sheets_response_multiple_rows_and_no_records_found():
    html = _row("https://x/a.xlsx?sfvrsn=1", "All Schemes Monthly Portfolio - as on 31st July 2026") + \
        _row("https://x/b.xlsx?sfvrsn=2", "All Schemes Monthly Portfolio - as on 30th June 2026")
    filings = parse_portfolio_sheets_response(html)
    assert len(filings) == 2
    assert parse_portfolio_sheets_response(
        '\n    <td>No Records Found</td>\n    <td></td>\n    <td></td>\n    <td></td>\n'
    ) == []


def test_parse_portfolio_sheets_response_keeps_row_with_unparseable_title():
    html = _row("https://x/c.xlsx?sfvrsn=3", "Some Odd Title With No Date")
    filings = parse_portfolio_sheets_response(html)
    assert len(filings) == 1
    assert filings[0].period_end is None
    assert filings[0].filing_date is None


def test_parse_axis_filing_name_numeric_spaced_consolidated():
    scheme, period = parse_axis_filing_name("Monthly Portfolio-31 01 26")
    assert scheme is None
    assert str(period.date()) == "2026-01-31"


def test_parse_axis_filing_name_numeric_hyphen_consolidated():
    scheme, period = parse_axis_filing_name("Monthly Portfolio 31-10-2025")
    assert scheme is None
    assert str(period.date()) == "2025-10-31"


def test_parse_axis_filing_name_per_scheme_with_double_space_quirk():
    scheme, period = parse_axis_filing_name(
        "Monthly Portfolio - Axis Nifty Smallcap 50 Index Fund - 31 January  2026")
    assert scheme == "Axis Nifty Smallcap 50 Index Fund"
    assert str(period.date()) == "2026-01-31"


def test_parse_axis_filing_name_unescapes_html_entities_in_scheme_name():
    scheme, period = parse_axis_filing_name(
        "Monthly Portfolio - Axis Large &amp; Mid cap Fund - 31 January  2026")
    assert scheme == "Axis Large & Mid cap Fund"
    assert str(period.date()) == "2026-01-31"


def test_parse_axis_filing_name_unparseable_returns_none_none():
    assert parse_axis_filing_name("Weekly Debt Portfolios and Quants - 27Feb26") == (None, None)


def test_parse_axis_filing_name_per_scheme_two_digit_year():
    scheme, period = parse_axis_filing_name(
        "Monthly Portfolio - Axis NIFTY Next 50 Index Fund - 30 September 25")
    assert scheme == "Axis NIFTY Next 50 Index Fund"
    assert str(period.date()) == "2025-09-30"


def test_parse_axis_filing_name_as_on_month_day_comma_year():
    scheme, period = parse_axis_filing_name("Monthly Portfolio as on Feb 29, 2024")
    assert scheme is None
    assert str(period.date()) == "2024-02-29"


def test_parse_axis_filing_name_as_on_day_month_year_no_comma():
    scheme, period = parse_axis_filing_name("Monthly Portfolio as on 31 July 2023")
    assert scheme is None
    assert str(period.date()) == "2023-07-31"


def test_parse_axis_filing_name_as_on_dotted_numeric_dayfirst():
    scheme, period = parse_axis_filing_name("Monthly Portfolio as on 30.09.2023")
    assert scheme is None
    assert str(period.date()) == "2023-09-30"


def test_parse_axis_filing_name_consolidated_textual_date():
    scheme, period = parse_axis_filing_name("Monthly Portfolio-30 June 2024")
    assert scheme is None
    assert str(period.date()) == "2024-06-30"


def test_parse_axis_filing_name_plural_portfolios_no_space_before_day():
    scheme, period = parse_axis_filing_name("Monthly Portfolios - Axis Floater Fund -31 Dec 2023")
    assert scheme == "Axis Floater Fund"
    assert str(period.date()) == "2023-12-31"


def test_parse_axis_filing_name_as_on_month_day_comma_no_space_before_year():
    scheme, period = parse_axis_filing_name("Monthly Portfolio as on Aug 31,2023")
    assert scheme is None
    assert str(period.date()) == "2023-08-31"


def test_parse_axis_filing_name_per_scheme_month_only_falls_back_to_month_end():
    scheme, period = parse_axis_filing_name("Monthly Portfolio - Axis Overnight Fund - November 2022")
    assert scheme == "Axis Overnight Fund"
    assert str(period.date()) == "2022-11-30"


def test_parse_axis_filing_name_per_scheme_ordinal_suffix_no_spaces():
    scheme, period = parse_axis_filing_name("Monthly Portfolio-Axis Silver Fund of Fund-31st October 2022")
    assert scheme == "Axis Silver Fund of Fund"
    assert str(period.date()) == "2022-10-31"


def test_parse_axis_filing_name_consolidated_bare_dotted_numeric():
    scheme, period = parse_axis_filing_name("Monthly Portfolio-31.01.2023")
    assert scheme is None
    assert str(period.date()) == "2023-01-31"


def test_parse_axis_documents_response_excludes_non_monthly_and_keeps_monthly():
    payload = {"data": {"documentList": [
        {"docuementURL": "https://x/a.xlsx", "documentName": "Monthly Portfolio-31 01 26",
         "documentPostedDate": "2026-01-01"},
        {"docuementURL": "https://x/b.xlsx",
         "documentName": "Monthly Portfolio - Axis Value Fund - 31 January 2026",
         "documentPostedDate": "2026-01-01"},
        {"docuementURL": "https://x/c.xlsx", "documentName": "Weekly Debt Portfolios and Quants - 27Feb26",
         "documentPostedDate": "2026-02-01"},
        {"docuementURL": "https://x/d.xlsx", "documentName": "Adhoc Portfolio - 15Aug26",
         "documentPostedDate": "2026-08-01"},
    ]}}
    filings = parse_axis_documents_response(payload)
    assert len(filings) == 2
    assert filings[0].amc_name == "Axis Mutual Fund"
    assert filings[0].scheme_name is None
    assert str(filings[0].period_end.date()) == "2026-01-31"
    assert filings[1].scheme_name == "Axis Value Fund"
    assert str(filings[1].filing_date.date()) == "2026-02-10"  # +10 assumed lag


def test_parse_axis_documents_response_ignores_documentposteddate_for_period():
    # documentPostedDate is always the 1st of a bucket month (see module
    # docstring) - period_end must come from the real title, not this field.
    payload = {"data": {"documentList": [
        {"docuementURL": "https://x/a.xlsx", "documentName": "Monthly Portfolio-31 03 26",
         "documentPostedDate": "2026-02-01"},
    ]}}
    filings = parse_axis_documents_response(payload)
    assert str(filings[0].period_end.date()) == "2026-03-31"
