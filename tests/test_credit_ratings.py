"""Credit rating record parsing, verified against real observed NSE
corporate-credit-rating shapes (see `dtest/data/credit_ratings.py`'s own
module docstring for the live probe, 2026-08-24)."""

from __future__ import annotations

import pandas as pd

from dtest.data.credit_ratings import (
    build_name_to_symbol_lookup,
    normalize_company_name,
    parse_credit_rating_record,
)


def _rec(**overrides) -> dict:
    rec = {
        "AppID": "14538", "Symbol": "Notlisted", "CompanyName": "Punjab & Sind Bank",
        "ISIN": "INE608A08041", "NameOfCRAgency": "CARE Ratings Limited",
        "CreditRating": "AA", "CreditRatingEarlier": "AA", "RatingAction": "Reaffirm",
        "Outlook": "Stable", "DateofCR": "21-08-2026",
        "BroadcastDateTime": "22-AUG-2026 18:29:52",
    }
    rec.update(overrides)
    return rec


def test_normalize_company_name_strips_suffixes_and_punctuation():
    assert normalize_company_name("TVS Holdings Limited") == normalize_company_name("TVS Holdings Ltd.")
    assert normalize_company_name("TVS Holdings Limited") == "TVSHOLDINGS"


def test_parse_basic_fields():
    r = parse_credit_rating_record(_rec(), {})
    assert r.app_id == "14538"
    assert r.company_name == "Punjab & Sind Bank"
    assert r.agency == "CARE Ratings Limited"
    assert r.action == "Reaffirm"
    assert str(r.filing_date) == "2026-08-22 18:29:52"
    assert str(r.date_of_rating.date()) == "2026-08-21"


def test_placeholder_symbol_variants_all_treated_as_missing():
    # Real placeholders seen live: text ("Notlisted", "NOTAPPLICABLE") AND
    # bare numeric filler ("000000", "222333") that superficially looks
    # like a plausible-if-odd ticker but is never real (a real NSE symbol
    # always has at least one letter).
    for placeholder in ("Notlisted", "NOTLISTED", "NOTAPPLICABLE", "-", "", "000000", "222333"):
        r = parse_credit_rating_record(_rec(Symbol=placeholder), {})
        assert r.symbol is None, f"{placeholder!r} should resolve to no symbol"


def test_real_ticker_shaped_symbol_is_kept_directly():
    r = parse_credit_rating_record(_rec(Symbol="BLS", CompanyName="BLS International Services Ltd."), {})
    assert r.symbol == "BLS"


def test_placeholder_symbol_falls_back_to_name_lookup():
    lookup = {"TVSHOLDINGS": "TVSHLTD"}
    r = parse_credit_rating_record(
        _rec(Symbol="NOTAPPLICABLE", CompanyName="TVS Holdings Limited"), lookup)
    assert r.symbol == "TVSHLTD"


def test_name_not_in_lookup_stays_null_not_guessed():
    r = parse_credit_rating_record(
        _rec(Symbol="NOTLISTED", CompanyName="Some Totally Unknown Debt Issuer Ltd."), {})
    assert r.symbol is None


def test_missing_app_id_or_company_or_date_returns_none():
    assert parse_credit_rating_record(_rec(AppID=None), {}) is None
    assert parse_credit_rating_record(_rec(CompanyName=None), {}) is None
    assert parse_credit_rating_record(_rec(BroadcastDateTime=None), {}) is None


def test_build_name_to_symbol_lookup_from_recon_events():
    events = pd.DataFrame({
        "company_name": ["TVS Holdings Ltd.", "BLS International Services Ltd.", "TVS Holdings Ltd."],
        "symbol": ["TVSHLTD", "BLS", "TVSHLTD"],
    })
    lookup = build_name_to_symbol_lookup(events)
    assert lookup[normalize_company_name("TVS Holdings Limited")] == "TVSHLTD"
    assert lookup[normalize_company_name("BLS International Services Limited")] == "BLS"
