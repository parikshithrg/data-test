"""Corporate announcement parsing, verified against a real observed NSE
corporate-announcements record shape (see
`dtest/data/corporate_announcements.py`'s own module docstring for the
live probe, 2026-08-24)."""

from __future__ import annotations

from dtest.data.corporate_announcements import parse_announcement_record


def _rec(**overrides) -> dict:
    rec = {
        "an_dt": "31-Dec-2024 18:27:09", "desc": "Acquisition",
        "symbol": "RELIANCE", "sm_name": "Reliance Industries Limited",
        "sm_isin": "INE002A01018", "smIndustry": "Refineries",
        "attchmntText": "Reliance Industries Limited has informed the Exchange about conversion...",
        "seq_id": "106101211",
    }
    rec.update(overrides)
    return rec


def test_parse_basic_fields():
    r = parse_announcement_record(_rec())
    assert r.symbol == "RELIANCE"
    assert r.category == "Acquisition"
    assert r.company_name == "Reliance Industries Limited"
    assert r.isin == "INE002A01018"
    assert str(r.filing_date) == "2024-12-31 18:27:09"
    assert r.seq_id == "106101211"


def test_irrelevant_category_dropped():
    for noise_category in ("Loss of Share Certificates", "Trading Window",
                           "Updates", "Press Release", "Credit Rating"):
        r = parse_announcement_record(_rec(desc=noise_category))
        assert r is None, f"{noise_category!r} should be filtered out"


def test_relevant_categories_all_pass():
    for cat in ("Acquisition", "Amalgamation/Merger", "Dividend", "Bonus", "Buyback",
               "Bagging/Receiving of orders/contracts", "Stock split", "Rights Issue"):
        r = parse_announcement_record(_rec(desc=cat))
        assert r is not None, f"{cat!r} should be kept"
        assert r.category == cat


def test_old_era_category_aliases_also_pass():
    # Real pre-2013 label drift, found live 2026-08-24 on a 2010 sample:
    # different wording for the same disclosure type as the modern labels.
    for cat in ("Interim Dividend", "Buy back", "Allotment of Equity Shares",
               "Allotment of shares", "Change in Board of Directors"):
        r = parse_announcement_record(_rec(desc=cat))
        assert r is not None, f"old-era alias {cat!r} should be kept"


def test_credit_rating_category_excluded_deliberately():
    # Deliberately excluded here - dtest.data.credit_ratings is the richer,
    # multi-agency source for this specific category.
    assert parse_announcement_record(_rec(desc="Credit Rating")) is None
    assert parse_announcement_record(_rec(desc="Credit Rating- Revision")) is None


def test_missing_symbol_or_date_returns_none():
    assert parse_announcement_record(_rec(symbol=None)) is None
    assert parse_announcement_record(_rec(an_dt=None)) is None


def test_null_seq_id_gets_synthetic_key_not_dropped():
    # Real pre-2013 shape, found live 2026-08-24: seq_id is null on every
    # record from that era - dropping them silently zeroed out 9 real
    # years of data on the first full-scale fetch. Must fall back to a
    # synthetic (symbol, an_dt) key instead of discarding the record.
    r = parse_announcement_record(_rec(seq_id=None, desc="Interim Dividend"))
    assert r is not None
    assert r.seq_id == "RELIANCE|31-Dec-2024 18:27:09"
