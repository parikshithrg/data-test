"""Point-in-time pair formation and spread construction, hand-computable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtest.features.pairs import (
    liquidity_ranked_same_sector_pairs,
    log_spread,
    random_pairs_any_sector,
    random_same_sector_pairs,
    select_pairs,
)


def _panel():
    idx = pd.bdate_range("2020-01-06", periods=12)
    # BANK_A/BANK_B: identical shape, scaled - perfectly correlated.
    bank_a = [100.0 + i for i in range(12)]
    bank_b = [200.0 + 2 * i for i in range(12)]
    # IT_A/IT_B: identical shape (downtrend), scaled - perfectly correlated,
    # but a DIFFERENT pattern from the banks - never compared to them anyway
    # since sector grouping alone should prevent that, not correlation.
    it_a = [50.0 - i for i in range(12)]
    it_b = [75.0 - 1.5 * i for i in range(12)]
    # PHARMA_A/PHARMA_B: no real relationship - alternating vs monotonic.
    pharma_a = [30.0, 32.0, 29.0, 33.0, 28.0, 34.0, 27.0, 35.0, 26.0, 36.0, 25.0, 37.0]
    pharma_b = [60.0 + i for i in range(12)]
    return pd.DataFrame({
        "BANK_A": bank_a, "BANK_B": bank_b,
        "IT_A": it_a, "IT_B": it_b,
        "PHARMA_A": pharma_a, "PHARMA_B": pharma_b,
    }, index=idx)


SECTOR_MAP = {
    "BANK_A": "Banking", "BANK_B": "Banking",
    "IT_A": "IT", "IT_B": "IT",
    "PHARMA_A": "Pharma", "PHARMA_B": "Pharma",
}


def test_select_pairs_finds_same_sector_high_correlation_pairs():
    close = _panel()
    pairs = select_pairs(close, SECTOR_MAP, list(SECTOR_MAP.keys()),
                         as_of=close.index[-1], formation_window=10, min_corr=0.9)
    assert ("BANK_A", "BANK_B") in pairs
    assert ("IT_A", "IT_B") in pairs


def test_select_pairs_never_crosses_sectors():
    close = _panel()
    pairs = select_pairs(close, SECTOR_MAP, list(SECTOR_MAP.keys()),
                         as_of=close.index[-1], formation_window=10, min_corr=0.0)
    for a, b in pairs:
        assert SECTOR_MAP[a] == SECTOR_MAP[b]


def test_select_pairs_respects_min_corr():
    close = _panel()
    pairs = select_pairs(close, SECTOR_MAP, list(SECTOR_MAP.keys()),
                         as_of=close.index[-1], formation_window=10, min_corr=0.9)
    assert ("PHARMA_A", "PHARMA_B") not in pairs   # alternating vs monotonic - not correlated


def test_select_pairs_requires_full_formation_window():
    close = _panel()
    close = close.copy()
    close.loc[close.index[3], "BANK_A"] = np.nan   # a gap inside the window
    pairs = select_pairs(close, SECTOR_MAP, list(SECTOR_MAP.keys()),
                         as_of=close.index[-1], formation_window=10, min_corr=0.9)
    assert ("BANK_A", "BANK_B") not in pairs


def test_select_pairs_excludes_ineligible_symbols():
    close = _panel()
    eligible = [s for s in SECTOR_MAP if s != "BANK_B"]   # BANK_B not eligible
    pairs = select_pairs(close, SECTOR_MAP, eligible,
                         as_of=close.index[-1], formation_window=10, min_corr=0.9)
    assert ("BANK_A", "BANK_B") not in pairs
    assert not any("BANK_B" in p for p in pairs)


def test_select_pairs_caps_max_pairs_per_sector():
    idx = pd.bdate_range("2020-01-06", periods=12)
    # 4 identically-shaped symbols in one sector -> 6 possible pairs, all
    # perfectly correlated - cap should keep only the top max_pairs_per_sector.
    close = pd.DataFrame({
        f"S{i}": [100.0 + (i + 1) * j for j in range(12)] for i in range(4)
    }, index=idx)
    sector_map = {f"S{i}": "Sector" for i in range(4)}
    pairs = select_pairs(close, sector_map, list(sector_map.keys()),
                         as_of=close.index[-1], formation_window=10, min_corr=0.9,
                         max_pairs_per_sector=2)
    assert len(pairs) <= 2


def test_select_pairs_too_little_history_returns_empty():
    close = _panel()   # only 12 rows
    pairs = select_pairs(close, SECTOR_MAP, list(SECTOR_MAP.keys()),
                         as_of=close.index[-1], formation_window=252, min_corr=0.5)
    assert pairs == []


def test_random_same_sector_pairs_never_crosses_sectors():
    rng = np.random.default_rng(0)
    pairs = random_same_sector_pairs(SECTOR_MAP, list(SECTOR_MAP.keys()),
                                     max_pairs_per_sector=5, rng=rng)
    for a, b in pairs:
        assert SECTOR_MAP[a] == SECTOR_MAP[b]
    assert len(pairs) == 3   # exactly one pair per 2-symbol sector


def test_random_same_sector_pairs_caps_per_sector():
    idx = pd.bdate_range("2020-01-06", periods=5)
    sector_map = {f"S{i}": "Sector" for i in range(4)}   # C(4,2) = 6 possible pairs
    rng = np.random.default_rng(0)
    pairs = random_same_sector_pairs(sector_map, list(sector_map.keys()),
                                     max_pairs_per_sector=2, rng=rng)
    assert len(pairs) == 2


def test_random_same_sector_pairs_deterministic_given_same_rng_state():
    pairs1 = random_same_sector_pairs(SECTOR_MAP, list(SECTOR_MAP.keys()),
                                      max_pairs_per_sector=5, rng=np.random.default_rng(7))
    pairs2 = random_same_sector_pairs(SECTOR_MAP, list(SECTOR_MAP.keys()),
                                      max_pairs_per_sector=5, rng=np.random.default_rng(7))
    assert pairs1 == pairs2


def test_liquidity_ranked_same_sector_pairs_picks_the_liquid_names():
    idx = pd.bdate_range("2020-01-06", periods=70)
    sector_map = {f"S{i}": "Sector" for i in range(4)}
    # S3 is far more liquid than the other three. Top-3-by-liquidity is
    # {S3, S1, S0} (k=3, the smallest k with C(k,2) >= max_pairs_per_sector
    # of 2); keeping the 2 highest-combined-turnover pairs among those
    # three keeps both pairs that include S3 and drops (S1, S0).
    turnover = pd.DataFrame({
        "S0": [10.0] * 70, "S1": [11.0] * 70, "S2": [9.0] * 70, "S3": [1000.0] * 70,
    }, index=idx)
    pairs = liquidity_ranked_same_sector_pairs(
        sector_map, list(sector_map.keys()), turnover, as_of=idx[-1], max_pairs_per_sector=2)
    assert len(pairs) == 2
    assert all("S3" in p for p in pairs)   # the most liquid name is in every kept pair
    assert not any("S2" in p for p in pairs)   # the least liquid name never gets in


def test_liquidity_ranked_same_sector_pairs_never_crosses_sectors():
    idx = pd.bdate_range("2020-01-06", periods=70)
    turnover = pd.DataFrame({s: [10.0 + i] * 70 for i, s in enumerate(SECTOR_MAP)}, index=idx)
    pairs = liquidity_ranked_same_sector_pairs(
        SECTOR_MAP, list(SECTOR_MAP.keys()), turnover, as_of=idx[-1], max_pairs_per_sector=5)
    for a, b in pairs:
        assert SECTOR_MAP[a] == SECTOR_MAP[b]


def test_liquidity_ranked_same_sector_pairs_deterministic():
    idx = pd.bdate_range("2020-01-06", periods=70)
    turnover = pd.DataFrame({s: [10.0 + i] * 70 for i, s in enumerate(SECTOR_MAP)}, index=idx)
    p1 = liquidity_ranked_same_sector_pairs(SECTOR_MAP, list(SECTOR_MAP.keys()), turnover,
                                            as_of=idx[-1], max_pairs_per_sector=5)
    p2 = liquidity_ranked_same_sector_pairs(SECTOR_MAP, list(SECTOR_MAP.keys()), turnover,
                                            as_of=idx[-1], max_pairs_per_sector=5)
    assert p1 == p2


def test_random_pairs_any_sector_can_cross_sectors():
    rng = np.random.default_rng(0)
    # Draw enough pairs that, if sector were respected, some would have to
    # be cross-sector anyway (6 symbols -> 15 possible pairs total, only 3
    # possible if sector were enforced) - confirms the function truly
    # ignores SECTOR_MAP (it never even receives it as an argument).
    pairs = random_pairs_any_sector(list(SECTOR_MAP.keys()), n_target=10, rng=rng)
    assert any(SECTOR_MAP[a] != SECTOR_MAP[b] for a, b in pairs)


def test_random_pairs_any_sector_no_duplicates_and_respects_target():
    rng = np.random.default_rng(0)
    pairs = random_pairs_any_sector(list(SECTOR_MAP.keys()), n_target=5, rng=rng)
    assert len(pairs) == 5
    assert len(set(pairs)) == 5   # no duplicate pair drawn twice


def test_random_pairs_any_sector_caps_at_max_possible():
    rng = np.random.default_rng(0)
    pairs = random_pairs_any_sector(["A", "B", "C"], n_target=100, rng=rng)
    assert len(pairs) == 3   # C(3,2) = 3, however large n_target asks for


def test_log_spread_hand_computed():
    idx = pd.bdate_range("2020-01-06", periods=3)
    a = pd.Series([100.0, 110.0, 121.0], index=idx)
    b = pd.Series([50.0, 50.0, 55.0], index=idx)
    spread = log_spread(a, b)
    assert spread.iloc[0] == pytest.approx(np.log(100.0) - np.log(50.0))
    assert spread.iloc[2] == pytest.approx(np.log(121.0) - np.log(55.0))
