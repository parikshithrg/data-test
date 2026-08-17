"""Point-in-time pair formation and spread construction, hand-computable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtest.features.pairs import log_spread, select_pairs


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


def test_log_spread_hand_computed():
    idx = pd.bdate_range("2020-01-06", periods=3)
    a = pd.Series([100.0, 110.0, 121.0], index=idx)
    b = pd.Series([50.0, 50.0, 55.0], index=idx)
    spread = log_spread(a, b)
    assert spread.iloc[0] == pytest.approx(np.log(100.0) - np.log(50.0))
    assert spread.iloc[2] == pytest.approx(np.log(121.0) - np.log(55.0))
