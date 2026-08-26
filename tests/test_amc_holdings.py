"""Axis Mutual Fund per-scheme equity-holdings parsing, verified against
real observed workbook row shapes from both a 2022 and a 2026 file (see
`dtest/data/amc_holdings.py`'s own module docstring for the live
inspection)."""

from __future__ import annotations

from dtest.data.amc_holdings import parse_axis_scheme_sheet


def _real_equity_fund_rows():
    """Mirrors the real row sequence of a 2026 equity-fund workbook
    (Axis Momentum Fund), trimmed to a few holdings."""
    return [
        ("AXISMIF", "Axis Momentum Fund"),
        (None,),
        ("\n\n\n  ", "Monthly Portfolio Statement as on July 31, 2026"),
        ("Name of the Instrument", "ISIN", "Industry", "Quantity",
         "Market/Fair Value\n (Rs. in Lakhs)", "% to Net\n Assets", "YTM~", "YTC^"),
        ("Equity & Equity related",),
        ("(a) Listed / awaiting listing on Stock Exchanges",),
        ("PNGJ01", "P N Gadgil Jewellers Limited", "INE953R01016", "Consumer Durables",
         710033, 4759.3512, 0.0504),
        ("SBAI02", "State Bank of India", "INE062A01020", "Banks", 344076, 3535.0368, 0.0374),
        ("Sub Total", 8294.388, 0.0878),
        ("Total", 8294.388, 0.0878),
        ("Cash & Other Receivables",),
        ("Net Assets",),
    ]


def _real_liquid_fund_rows():
    """Mirrors the real row sequence of Axis Liquid Fund - no equity
    section at all, only Derivatives/Debt/Money Market."""
    return [
        ("AXISLFA", "Axis Liquid Fund"),
        ("\n\n\n  ", "Monthly Portfolio Statement as on July 31, 2026"),
        ("Name of the Instrument", "ISIN", "Rating", "Quantity",
         "Market/Fair Value\n (Rs. in Lakhs)", "% to Net\n Assets", "YTM~", "YTC^"),
        ("Derivatives",),
        ("Interest Rate Swaps",),
        ("IRS2836354", "Interest Rate Swaps Pay Fix Receive Floating -BARC", -0.15, "$0.00%"),
        ("Total", -0.15, "$0.00%"),
        ("Debt Instruments",),
        ("(a) Listed / awaiting listing on Stock Exchange",),
        ("NBAR719", "7.50% National Bank For Agriculture", "INE261F08EA6", "CRISIL AAA",
         113500, 113534.958, 0.0201, 0.06665),
        ("Sub Total", 469575.9406, 0.0832),
    ]


def test_extracts_equity_rows_only_from_equity_section():
    holdings = parse_axis_scheme_sheet(_real_equity_fund_rows())
    assert len(holdings) == 2
    assert holdings[0].instrument_name == "P N Gadgil Jewellers Limited"
    assert holdings[0].isin == "INE953R01016"
    assert holdings[0].industry == "Consumer Durables"
    assert holdings[0].quantity == 710033
    assert holdings[0].market_value_lakhs == 4759.3512
    assert holdings[0].pct_to_nav == 0.0504


def test_scheme_identity_and_as_on_date_attached_to_every_row():
    holdings = parse_axis_scheme_sheet(_real_equity_fund_rows())
    for h in holdings:
        assert h.scheme_code == "AXISMIF"
        assert h.scheme_name == "Axis Momentum Fund"
        assert str(h.sheet_as_on_date.date()) == "2026-07-31"


def test_sub_total_and_total_rows_excluded():
    holdings = parse_axis_scheme_sheet(_real_equity_fund_rows())
    names = [h.instrument_name for h in holdings]
    assert "Sub Total" not in names
    assert "Total" not in names


def test_debt_only_scheme_yields_zero_equity_holdings_not_an_error():
    holdings = parse_axis_scheme_sheet(_real_liquid_fund_rows())
    assert holdings == []


def test_subsection_header_does_not_reset_equity_state():
    rows = [
        ("SCHM01", "Some Scheme"),
        ("Equity & Equity related",),
        ("(a) Listed / awaiting listing on Stock Exchanges",),
        ("ABCD01", "Some Company Ltd", "INE123456789", "Banks", 1000, 500.0, 0.01),
        ("(b) Unlisted",),
        ("EFGH01", "Another Company Ltd", "INE987654321", "IT - Software", 2000, 800.0, 0.02),
    ]
    holdings = parse_axis_scheme_sheet(rows)
    assert len(holdings) == 2
    assert holdings[1].instrument_name == "Another Company Ltd"


def test_row_missing_isin_is_dropped_not_guessed():
    rows = [
        ("SCHM01", "Some Scheme"),
        ("Equity & Equity related",),
        ("ABCD01", "Some Company Ltd", None, "Banks", 1000, 500.0, 0.01),
    ]
    assert parse_axis_scheme_sheet(rows) == []


def test_esg_extended_row_with_trailing_esg_columns_still_parses_first_7():
    # Real xlrd-read row shape: leading '' cells instead of None, plus 3
    # extra trailing columns (ESG Score, Core ESG Score, BRSR URL) - see
    # a real Axis ESG Integration Strategy Fund workbook.
    rows = [
        ("AXISESG", "Axis ESG Integration Strategy Fund", "", "", "", "", "", "", "", "", "", ""),
        ("", "", "", "", "", "", "", "", "", "", "", ""),
        ("\n\n\n  ", "Monthly Portfolio Statement as on July 31, 2025", "", "", "", "", "", "", "", "", "", ""),
        ("", "Equity & Equity related", "", "", "", "", "", "", "", "", "", ""),
        ("", "(a) Listed / awaiting listing on Stock Exchanges", "", "", "", "", "", "", "", "", "", ""),
        ("IBCL05", "ICICI Bank Limited", "INE090A01021", "Banks", 630000.0, 9332.82, 0.0759,
         "", "", 75.8, 100.0, "https://nsearchives.nseindia.com/corporate/x.pdf"),
    ]
    holdings = parse_axis_scheme_sheet(rows)
    assert len(holdings) == 1
    h = holdings[0]
    assert h.instrument_name == "ICICI Bank Limited"
    assert h.industry == "Banks"
    assert h.quantity == 630000.0
    assert h.pct_to_nav == 0.0759
    assert h.scheme_name == "Axis ESG Integration Strategy Fund"


def test_blank_rows_and_none_only_rows_ignored():
    rows = [
        ("SCHM01", "Some Scheme"),
        (None, None, None),
        ("Equity & Equity related",),
        (None,),
        ("ABCD01", "Some Company Ltd", "INE123456789", "Banks", 1000, 500.0, 0.01),
    ]
    holdings = parse_axis_scheme_sheet(rows)
    assert len(holdings) == 1
