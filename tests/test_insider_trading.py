"""Insider-trading (SEBI PIT) record parsing, hand-verified against a real
RELIANCE record (see `dtest/data/insider_trading.py`'s own module
docstring for the live probe, 2026-08-24)."""

from __future__ import annotations

import math

from dtest.data.insider_trading import parse_pit_record


def _rec(**overrides) -> dict:
    rec = {
        "symbol": "RELIANCE", "date": "19-Nov-2015 20:27", "intimDt": "17-Nov-2015",
        "acqfromDt": "13-Oct-2015", "acqtoDt": "13-Oct-2015",
        "personCategory": "Other", "acqMode": "ESOP", "tdpTransactionType": "Buy",
        "secType": "Equity Shares", "secAcq": "300", "buyQuantity": "300", "sellquantity": "0",
        "buyValue": "192600", "sellValue": "0",
        "befAcqSharesNo": "12000", "befAcqSharesPer": "0", "afterAcqSharesNo": "12300",
        "afterAcqSharesPer": "0",
    }
    rec.update(overrides)
    return rec


def test_parse_pit_record_basic_fields():
    r = parse_pit_record(_rec())
    assert r.symbol == "RELIANCE"
    assert str(r.filing_date) == "2015-11-19 20:27:00"
    assert str(r.acq_to_date.date()) == "2015-10-13"
    assert r.buy_quantity == 300.0
    assert r.buy_value == 192600.0
    assert r.sell_quantity == 0.0
    assert r.transaction_type == "Buy"


def test_parse_pit_record_dash_and_missing_numeric_is_nan():
    r = parse_pit_record(_rec(sellValue="-", buyValue=None))
    assert math.isnan(r.sell_value)
    assert math.isnan(r.buy_value)


def test_parse_pit_record_comma_thousands_separator():
    r = parse_pit_record(_rec(buyValue="1,92,600"))
    assert r.buy_value == 192600.0


def test_parse_pit_record_missing_symbol_or_date_returns_none():
    assert parse_pit_record(_rec(symbol=None)) is None
    assert parse_pit_record(_rec(date=None)) is None
    assert parse_pit_record(_rec(date="")) is None


def test_parse_pit_record_missing_intim_date_is_none_not_error():
    r = parse_pit_record(_rec(intimDt="-"))
    assert r.intim_date is None
