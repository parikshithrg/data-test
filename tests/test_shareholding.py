"""Shareholding-pattern parsing: master-JSON records and the two real XBRL
taxonomy eras, hand-verified against real filing shapes (see
`dtest/data/shareholding.py`'s own module docstring for the live probes
these fixtures mirror - RELIANCE/NMDC/JPPOWER/ZEEL, 2026-08-24)."""

from __future__ import annotations

import math

from dtest.data.shareholding import parse_master_record, parse_xbrl_categories


def _master_rec(**overrides) -> dict:
    rec = {
        "date": "30-JUN-2025", "broadcastDate": "18-JUL-2025 16:08:01",
        "pr_and_prgrp": "60.79", "public_val": "39.21", "revisedData": "N",
        "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/SHP_x.xml",
        "recordId": "12345",
    }
    rec.update(overrides)
    return rec


def test_parse_master_record_basic_fields():
    m = parse_master_record("NMDC", _master_rec())
    assert m.symbol == "NMDC"
    assert m.promoter_pct == 60.79
    assert m.public_pct == 39.21
    assert str(m.filing_date.date()) == "2025-07-18"
    assert str(m.period_end.date()) == "2025-06-30"
    assert m.revised is False


def test_parse_master_record_revised_flag():
    m = parse_master_record("NMDC", _master_rec(revisedData="Y"))
    assert m.revised is True


def test_parse_master_record_missing_dates_returns_none():
    assert parse_master_record("NMDC", _master_rec(date=None)) is None
    assert parse_master_record("NMDC", _master_rec(broadcastDate=None)) is None


def _xbrl_new_era(mf="7.16", insurance="7.15", fii="13.04", dii="14.39",
                  promoter="60.79", pledge="0") -> str:
    # New era: `_ContextI`-suffixed contexts, fraction-scale example values
    # match a real NMDC filing (2025-09-30) at percent scale (7.16 not 0.0716)
    # unless a test overrides to check the fraction-scale calibration path.
    def fact(tag, ctx, val):
        return f'<in-bse-shp:{tag} contextRef="{ctx}" decimals="INF" unitRef="pure">{val}</in-bse-shp:{tag}>'
    return "".join([
        fact("ShareholdingAsAPercentageOfTotalNumberOfShares", "MutualFundsOrUTI_ContextI", mf),
        fact("ShareholdingAsAPercentageOfTotalNumberOfShares", "InsuranceCompanies_ContextI", insurance),
        fact("ShareholdingAsAPercentageOfTotalNumberOfShares", "InstitutionsForeign_ContextI", fii),
        fact("ShareholdingAsAPercentageOfTotalNumberOfShares", "InstitutionsDomestic_ContextI", dii),
        fact("ShareholdingAsAPercentageOfTotalNumberOfShares", "ShareholdingOfPromoterAndPromoterGroup_ContextI", promoter),
        fact("EncumberedSharesHeldAsPercentageOfTotalNumberOfShares", "ShareholdingOfPromoterAndPromoterGroup_ContextI", pledge),
    ])


def test_parse_xbrl_categories_percent_scale_passthrough():
    # promoter fact (60.79) matches master_promoter_pct (60.79) directly ->
    # scale=1, values pass through unchanged. Real NMDC 2025-09-30 shape.
    out = parse_xbrl_categories(_xbrl_new_era(), master_promoter_pct=60.79)
    assert out["mf_pct"] == 7.16
    assert out["fii_pct"] == 13.04
    assert out["dii_pct"] == 14.39
    assert out["promoter_pledge_pct"] == 0.0


def test_parse_xbrl_categories_fraction_scale_normalized():
    # Same real quantities as above but the filing reports everything as a
    # 0-1 fraction (promoter "0.6079" instead of "60.79") - the real NMDC
    # 2025-06-30-vs-2025-09-30 inconsistency this module exists to fix.
    xml = _xbrl_new_era(mf="0.0716", insurance="0.0715", fii="0.1304", dii="0.1439",
                        promoter="0.6079", pledge="0.0125")
    out = parse_xbrl_categories(xml, master_promoter_pct=60.79)
    assert math.isclose(out["mf_pct"], 7.16, abs_tol=1e-6)
    assert math.isclose(out["fii_pct"], 13.04, abs_tol=1e-6)
    assert math.isclose(out["promoter_pledge_pct"], 1.25, abs_tol=1e-6)


def test_parse_xbrl_categories_old_era_bare_context_ids():
    # Old era: no underscore, bare "I" suffix, "Institutions"+"Foreign"/
    # "Domestic" split differs by sub-era - this fixture uses the middle
    # era's InstitutionsDomesticI/InstitutionsForeignI naming (real NMDC
    # 2022-09-30 shape).
    def fact(tag, ctx, val):
        return f'<in-bse-shp:{tag} contextRef="{ctx}" decimals="INF" unitRef="pure">{val}</in-bse-shp:{tag}>'
    xml = "".join([
        fact("ShareholdingAsAPercentageOfTotalNumberOfShares", "MutualFundsOrUtiI", "4.58"),
        fact("ShareholdingAsAPercentageOfTotalNumberOfShares", "InsuranceCompaniesI", "15.02"),
        fact("ShareholdingAsAPercentageOfTotalNumberOfShares", "InstitutionsForeignI", "6.20"),
        fact("ShareholdingAsAPercentageOfTotalNumberOfShares", "InstitutionsDomesticI", "27.50"),
        fact("ShareholdingAsAPercentageOfTotalNumberOfShares", "ShareholdingOfPromoterAndPromoterGroupI", "60.79"),
        fact("PledgedOrEncumberedSharesHeldAsPercentageOfTotalNumberOfShares",
             "ShareholdingOfPromoterAndPromoterGroupI", "0"),
    ])
    out = parse_xbrl_categories(xml, master_promoter_pct=60.79)
    assert out["mf_pct"] == 4.58
    assert out["fii_pct"] == 6.20
    assert out["dii_pct"] == 27.50
    assert out["promoter_pledge_pct"] == 0.0


def test_parse_xbrl_categories_missing_category_is_nan_not_guessed():
    # Oldest era: no MutualFundsOrUTI/InsuranceCompanies context at all on
    # this filing (real NMDC 2015/2021 shape) - must be NaN, never 0 or a
    # fabricated fallback.
    xml = (
        '<in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares '
        'contextRef="ShareholdingOfPromoterAndPromoterGroupI" decimals="INF" unitRef="pure">'
        '60.79</in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares>'
    )
    out = parse_xbrl_categories(xml, master_promoter_pct=60.79)
    assert math.isnan(out["mf_pct"])
    assert math.isnan(out["fii_pct"])
    assert math.isnan(out["promoter_pledge_pct"])


def test_parse_xbrl_categories_pledge_heavy_name_real_shape():
    # JPPOWER 2022-03-31 real shape: pledge alone 94.63% is itself
    # implausible as a coincidence to hand-verify against - locks in the
    # exact real observed value.
    xml = _xbrl_new_era(promoter="24.00", pledge="94.63")
    out = parse_xbrl_categories(xml, master_promoter_pct=24.00)
    assert out["promoter_pledge_pct"] == 94.63
