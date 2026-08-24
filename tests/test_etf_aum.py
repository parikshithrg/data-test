"""AMFI quarterly scheme-level ETF/Index-Fund AUM parsing, verified
against real observed response shapes (see `dtest/data/etf_aum.py`'s own
module docstring for the live probe, 2026-08-24)."""

from __future__ import annotations

from dtest.data.etf_aum import parse_average_aum_response


def _payload(selected_period="April - June 2026", data=None):
    return {"selectedPeriod": selected_period, "data": data or []}


def _group(category, amc_name="Test Mutual Fund", schemes=None):
    return {
        "Mfname": amc_name, "SchemeCat_Desc": category,
        "schemes": schemes or [], "totalAUM": {},
    }


def _scheme(name, code, aum):
    return {
        "SchemeNAVName": name, "AMFI_Code": code,
        "AverageAumForTheMonth": {
            "ExcludingFundOfFundsDomesticButIncludingFundOfFundsOverseas": aum,
            "FundOfFundsDomestic": 0,
        },
    }


def test_period_end_and_filing_date_derived_from_label():
    payload = _payload("April - June 2026", data=[
        _group("Other ETFs", schemes=[_scheme("IIFL NIFTY ETF-GROWTH", 115912, 1181.58)]),
    ])
    recs = parse_average_aum_response(payload)
    assert len(recs) == 1
    r = recs[0]
    assert str(r.period_end.date()) == "2026-06-30"
    assert str(r.filing_date.date()) == "2026-07-15"  # +15 assumed disclosure lag
    assert r.average_aum_lakhs == 1181.58
    assert r.amfi_code == 115912


def test_single_month_label_era_parses_too():
    # Real pre-Oct-2010 shape: AMFI disclosed AAUM monthly, not quarterly,
    # until the quarter ended Dec 31, 2010 - the response's own label is a
    # bare "Month YYYY", not a "Month - Month YYYY" range.
    payload = _payload("March 2010", data=[
        _group("Other ETFs", schemes=[_scheme("Some Old ETF", 44444, 12.0)]),
    ])
    recs = parse_average_aum_response(payload)
    assert len(recs) == 1
    assert str(recs[0].period_end.date()) == "2010-03-31"


def test_non_passive_categories_are_dropped():
    payload = _payload(data=[
        _group("Income", schemes=[_scheme("Some Debt Fund", 111111, 500.0)]),
        _group("Equity Scheme - Large Cap Fund", schemes=[_scheme("Some Equity Fund", 222222, 700.0)]),
        _group("Other ETFs", schemes=[_scheme("Some ETF", 333333, 900.0)]),
    ])
    recs = parse_average_aum_response(payload)
    assert len(recs) == 1
    assert recs[0].amfi_code == 333333


def test_total_row_is_dropped_not_double_counted():
    payload = _payload(data=[
        _group("Other ETFs", schemes=[_scheme("Some ETF", 333333, 900.0)]),
        _group("Total", schemes=[_scheme("(grand total placeholder)", 999999, 999999.0)]),
    ])
    recs = parse_average_aum_response(payload)
    assert len(recs) == 1
    assert all(r.amfi_code != 999999 for r in recs)


def test_pre_and_post_2017_category_label_eras_both_match():
    # Real drift confirmed live: SEBI's Oct 2017 rationalization changed
    # AMFI's own category label wording for the same underlying scheme type.
    for label in ("GOLD ETFs", "Exchange Traded Funds (ETFs) - Gold ETF",
                  "Other Scheme - Gold ETF", "Other Scheme - Index Funds",
                  "Other Scheme - Other  ETFs"):
        payload = _payload(data=[_group(label, schemes=[_scheme("X", 1, 10.0)])])
        recs = parse_average_aum_response(payload)
        assert len(recs) == 1, f"category label not matched: {label!r}"


def test_missing_selected_period_yields_no_records():
    assert parse_average_aum_response({"data": [_group("Other ETFs")]}) == []


def test_scheme_missing_amfi_code_is_skipped_not_guessed():
    payload = _payload(data=[_group("Other ETFs", schemes=[
        {"SchemeNAVName": "Broken Row", "AMFI_Code": None,
         "AverageAumForTheMonth": {"ExcludingFundOfFundsDomesticButIncludingFundOfFundsOverseas": 5.0}},
        _scheme("Good Row", 42, 5.0),
    ])])
    recs = parse_average_aum_response(payload)
    assert len(recs) == 1
    assert recs[0].scheme_name == "Good Row"
