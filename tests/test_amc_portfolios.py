"""SBI Mutual Fund monthly portfolio-filing list parsing, verified against
the real observed `GetSchemePortfolioSheets` response shape and title
conventions across both naming eras (see `dtest/data/amc_portfolios.py`'s
own module docstring for the live probe, 2026-08-26)."""

from __future__ import annotations

from dtest.data.amc_portfolios import parse_filing_period, parse_portfolio_sheets_response


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
