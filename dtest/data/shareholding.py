"""Per-symbol quarterly shareholding-pattern data, sourced from NSE's own
`corporate-share-holdings-master` endpoint and its linked per-filing XBRL
detail document. Confirmed live, 2026-08-24, before building anything: a
40-symbol pilot (`runs/probe_shareholding_coverage/`) found 97.5% coverage,
median 20 filings/covered symbol, and a real, clean, DIMENSIONAL XBRL
schema (`in-bse-shp:` namespace) giving promoter/MF/insurance/FII/DII/
public percentages plus real promoter-PLEDGE percentages on names that have
any (spot-checked against JPPOWER's 72.99% and ZEEL's 5.38% - both
plausible, well-known pledge-heavy names, not made up).

COVERAGE IS SHORT AND SYMBOL-DEPENDENT - stated plainly, not glossed over.
Unlike `financial_results.py` (real coverage back to ~2007), this source's
real floor is recent: 29/39 covered pilot symbols start at 2021-Q3 exactly,
a handful (NMDC, RECLTD: 2015-Q4; UPL: 2017-Q2) go back further - no fixed
"NSE didn't have this before X" cutoff, it genuinely varies by company (most
likely a phased XBRL-taxonomy rollout, not confirmed further). A hypothesis
built on this data has at most ~5 years of history on most names - CHECK
per-signal sample size/statistical power before trusting any result, the
same "thin evidence" caution `same_sector_pairing`'s Oil & Gas subset and
the `delivery` split's shorter window already needed elsewhere in this
project.

WHY POINT-IN-TIME CORRECTNESS. Same convention as `financial_results.py`:
`filing_date` (this module uses NSE's own `broadcastDate`, the real
disclosure timestamp, not `submissionDate` which can lag broadcast by
company-side processing) is the ONLY date a causal feature may key off -
never `period_end`. A filing disclosed on date D is knowable as of D's own
close, usable D+1 onward via the engine's T+1-open fill.

TWO-TIER FETCH, mirroring `financial_results.py`'s METADATA-then-DETAIL
split. The master JSON (one call per symbol, `?symbol=X`, deliberately NO
date filter - `from_date`/`to_date` were tested live combined with a
symbol and returned [] even for in-range real periods, so the only query
shape confirmed to return genuine multi-quarter history is the
no-date-filter one) gives `promoter_pct`/`public_pct` directly AND every
filing's own linked `xbrl` detail-document URL. The XBRL fetch (one call
per FILING, not per symbol - the slow step, matches financial_results.py's
own "detail page" pattern) resolves the finer promoter/MF/insurance/FII/
DII/pledge breakdown the coarse master JSON does not carry.

XBRL SCHEMA - dimensional, not flat named tags like financial_results.py's
XBRL. Every category (Promoter, MutualFundsOrUTI, InsuranceCompanies, ...)
is its own `<xbrli:context>` block, distinguished by a
`CategoryOfShareholdersAxis` dimension member; the context's OWN id string
(e.g. `MutualFundsOrUTI_ContextI`) is a readable label - confirmed by
inspecting a real filing's context list, not assumed from spec docs - so
this module keys off context id substrings rather than decoding the
dimension member value. `in-bse-shp:ShareholdingAsAPercentageOfTotalNumber
OfShares` (contextRef=<category>_ContextI) is the per-category % of the
whole company; `in-bse-shp:EncumberedSharesHeldAsPercentageOfTotalNumberOf
Shares` (contextRef=`ShareholdingOfPromoterAndPromoterGroup_ContextI`) is
the promoter-pledge %, confirmed present (0%) even on filings with no
pledge, and real/nonzero on filings that do have one.

TWO REAL TAXONOMY ERAS, confirmed live 2026-08-24 (NMDC's own filing
history spans both) - same category of problem financial_results.py
already solved for its own two HTML eras, handled the same way here (an
ORDERED per-field alias list, most-specific/newest first, first context
that actually exists on THIS filing wins). Filings roughly pre-mid-2025 use
BARE `I`-suffixed context ids with no underscore (`MutualFundsOrUtiI`,
`InsuranceCompaniesI`, `InstitutionsForeignPortfolioInvestorI`,
`InstitutionsI`, `ShareholdingOfPromoterAndPromoterGroupI`); newer filings
use the richer `_ContextI`-suffixed ids documented above, with FII split
into explicit Category-I/II sub-contexts. The old era's `dii_pct` proxy
(`InstitutionsI`) and `fii_pct` proxy (`InstitutionsForeignPortfolioInvestorI`)
are the closest single old-era context to the new era's combined totals,
not a guaranteed exact match to what the new era would have reported for
the same quarter - best-effort, same "missing/approximate data blocks,
never silently guesses a number" convention as financial_results.py's own
HTML-era handling.

UNIT-SCALE INCONSISTENCY - a real, filer-side data-quality landmine, found
by inspecting two consecutive real NMDC filings, NOT an assumption: both
tag `unitRef="pure"` and `decimals="INF"` identically, yet one filing
reports MF holding as `7.07` (already a percentage) and the very next
quarter's filing reports the same field as `0.0716` (a 0-1 fraction) - the
XBRL's own unit metadata cannot distinguish the two, confirmed by reading
both filings' raw unit definitions side by side. FIXED by a per-filing,
data-driven calibration rather than a magnitude-guessing heuristic: this
module ALSO reads the promoter category's own
`ShareholdingAsAPercentageOfTotalNumberOfShares` fact (same tag, same
context family) and compares it against the master JSON's own
`promoter_pct` (always correct, 0-100 scale, straight from NSE - never
derived from XBRL) for the SAME filing - the two describe the identical
quantity, so their ratio reveals the filing's own scale unambiguously
(confirmed: the two NMDC filings above showed XBRL promoter values of
`60.79` and `0.6079` against the identical master `promoter_pct=60.79`).
The resulting per-filing scale factor is applied uniformly to every
category and the pledge field parsed from that filing, on the assumption
that scale is a whole-document convention set by whichever filing software
produced it, not a per-field choice - not verified field-by-field within a
single filing, stated as the assumption it is.

FII IS REPORTED AS ONE COMBINED FIGURE HERE, not split Category-I/II -
`InstitutionsForeign_ContextI` is the FPI Category I + II total (confirmed:
on a real filing, `InstitutionsForeignPortfolioInvestorCategoryOne_ContextI`
+ `..CategoryTwo_ContextI` sums to `InstitutionsForeign_ContextI`, e.g.
RELIANCE 16.52 + 0.54 = 17.06 vs the reported 17.20 - small residual is
other-foreign-institution sub-categories, immaterial for this module's
purpose). Likewise `dii_pct` uses `InstitutionsDomestic_ContextI` (the
combined MF+insurance+banks+AIF+PF+SWF+NBFC+other-financial total), NOT a
manual sum of its own sub-fields - the combined context is the real,
disclosed figure, a manual sum risks drifting from it if NSE adds a new
domestic sub-category later.

BROADCAST-VS-PERIOD LAG CAN BE YEARS, and that is NOT a look-ahead bug -
worth stating precisely since it looks alarming at a glance. One real NMDC
record: `period_end=2015-12-31` but `broadcastDate` is 2022-01-06, a
6-year gap - a genuine backfilled/restated disclosure, not a parsing
error (confirmed by checking neighbouring records' broadcast dates are
normal quarterly cadence). This module's `filing_date` is ALWAYS the real
broadcast timestamp (never `period_end`), so any caller following this
module's own point-in-time rule treats that record as knowable on
2022-01-06, exactly correct - but it also means the naive "earliest
period_end per symbol" read the 2026-08-24 coverage pilot used is
misleading as a "how far back does real history go" measure; a handful of
pilot symbols (NMDC, RECLTD: 2015-Q4; UPL: 2017-Q2) that looked like they
had multi-year-deeper coverage than the ~2021-Q3 floor most symbols show
are largely/entirely this backfill artifact, not genuine older disclosure
history - re-check by filing_date, not period_end, before trusting any
symbol's apparent depth of history.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

SCHEMA = {
    "symbol": "string", "filing_date": "datetime64[ns]", "period_end": "datetime64[ns]",
    "revised": "bool", "promoter_pct": "float64", "public_pct": "float64",
    "mf_pct": "float64", "insurance_pct": "float64", "fii_pct": "float64",
    "dii_pct": "float64", "promoter_pledge_pct": "float64",
}

# Ordered, most-specific/newest-era context id first. The first context that
# actually appears on a given filing wins - see the two-taxonomy-era note above.
_CATEGORY_FIELD_ALIASES: dict[str, list[str]] = {
    "mf_pct": ["MutualFundsOrUTI_ContextI", "MutualFundsOrUtiI"],
    "insurance_pct": ["InsuranceCompanies_ContextI", "InsuranceCompaniesI"],
    # THREE real eras for the fii/dii combined-total context id, not two -
    # confirmed on NMDC's own filing history: a bare "InstitutionsI" era
    # (~2021Q3-2022Q1, no Domestic/Foreign split in the id), a middle
    # "InstitutionsDomesticI"/"InstitutionsForeignI" era (~2022Q2-2025Q1),
    # then the newest "..._ContextI" era (~2025Q2 on). MF/Insurance never
    # needed a third alias since those two categories have no Domestic/
    # Foreign qualifier to drift - this split only affects fii_pct/dii_pct.
    "fii_pct": ["InstitutionsForeign_ContextI", "InstitutionsForeignI", "InstitutionsForeignPortfolioInvestorI"],
    "dii_pct": ["InstitutionsDomestic_ContextI", "InstitutionsDomesticI", "InstitutionsI"],
}
_PROMOTER_CONTEXTS = ["ShareholdingOfPromoterAndPromoterGroup_ContextI", "ShareholdingOfPromoterAndPromoterGroupI"]

_PCT_FACT_RE = re.compile(
    r'<in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares[^>]*contextRef="([^"]+)"[^>]*>'
    r'([^<]*)</in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares>')
# `EncumberedSharesHeldAsPercentageOfTotalNumberOfShares` (new era) and
# `PledgedOrEncumberedSharesHeldAsPercentageOfTotalNumberOfShares` (old era)
# are each ONE combined pledge+NDU+other-encumbrance total, not pledge alone
# - the new era's more specific `EncumberedShareUnderPledgedAsPercentage...`
# (pledge only, excludes NDU) has no old-era equivalent, so the combined
# total is used for both eras to keep the field comparable across the full
# history - confirmed via a real pledge+NDU filing (JPPOWER: pledge 72.99% +
# NDU 6.21% = the combined total 79.20%, all three facts present and
# additive on that filing).
_ENCUMBERED_FACT_RE = re.compile(
    r'<in-bse-shp:(?:EncumberedSharesHeldAsPercentageOfTotalNumberOfShares'
    r'|PledgedOrEncumberedSharesHeldAsPercentageOfTotalNumberOfShares)'
    r'[^>]*contextRef="([^"]+)"[^>]*>([^<]*)</in-bse-shp:(?:EncumberedSharesHeldAsPercentageOfTotalNumberOfShares'
    r'|PledgedOrEncumberedSharesHeldAsPercentageOfTotalNumberOfShares)>')

# A scale factor is only trusted as "fraction-scale" (needs x100) when the
# XBRL promoter value sits far closer to master_promoter_pct/100 than to
# master_promoter_pct itself - guards against a coincidentally-small
# promoter stake making the ratio ambiguous.
_FRACTION_SCALE_RATIO_TOLERANCE = 0.15


@dataclass(frozen=True)
class MasterRecord:
    symbol: str
    filing_date: pd.Timestamp
    period_end: pd.Timestamp
    revised: bool
    promoter_pct: float
    public_pct: float
    xbrl_url: str | None
    record_id: str


def parse_master_record(symbol: str, rec: dict) -> MasterRecord | None:
    """One row of the `corporate-share-holdings-master` JSON response."""
    period_end = rec.get("date")
    broadcast = rec.get("broadcastDate")
    if not period_end or not broadcast:
        return None
    try:
        filing_dt = datetime.strptime(broadcast, "%d-%b-%Y %H:%M:%S")
        period_dt = datetime.strptime(period_end, "%d-%b-%Y")
    except ValueError:
        return None

    def _pct(key: str) -> float:
        v = rec.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    return MasterRecord(
        symbol=symbol, filing_date=pd.Timestamp(filing_dt), period_end=pd.Timestamp(period_dt),
        revised=(rec.get("revisedData") == "Y"),
        promoter_pct=_pct("pr_and_prgrp"), public_pct=_pct("public_val"),
        xbrl_url=rec.get("xbrl"), record_id=str(rec.get("recordId", "")),
    )


def _first_present(facts: dict[str, str], aliases: list[str]) -> float:
    for context in aliases:
        v = facts.get(context)
        if v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                continue
    return float("nan")


def parse_xbrl_categories(xml_text: str, master_promoter_pct: float) -> dict[str, float]:
    """Category-level percentages from one filing's XBRL detail document,
    normalized to a consistent 0-100 percent scale.

    `master_promoter_pct` (from the SAME filing's master-JSON record,
    always correct/0-100) is required to calibrate this filing's own
    percent-vs-fraction scale - see the module docstring's "UNIT-SCALE
    INCONSISTENCY" section for why this can't be inferred from the XBRL's
    own unit metadata alone.

    Returns NaN for any field whose context isn't present on this specific
    filing under EITHER known taxonomy era (see "TWO REAL TAXONOMY ERAS"
    above) - same "missing data blocks, never guesses" convention every
    other source in this project uses.
    """
    pct_facts = dict(_PCT_FACT_RE.findall(xml_text))
    encumbered_facts = dict(_ENCUMBERED_FACT_RE.findall(xml_text))

    xbrl_promoter = _first_present(pct_facts, _PROMOTER_CONTEXTS)
    scale = 1.0
    if pd.notna(xbrl_promoter) and master_promoter_pct > 0:
        frac_candidate = xbrl_promoter * 100
        if abs(frac_candidate - master_promoter_pct) / master_promoter_pct < _FRACTION_SCALE_RATIO_TOLERANCE:
            scale = 100.0

    out: dict[str, float] = {}
    for field, aliases in _CATEGORY_FIELD_ALIASES.items():
        v = _first_present(pct_facts, aliases)
        out[field] = v * scale if pd.notna(v) else v

    pledge = _first_present(encumbered_facts, _PROMOTER_CONTEXTS)
    out["promoter_pledge_pct"] = pledge * scale if pd.notna(pledge) else pledge
    return out
