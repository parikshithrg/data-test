"""Point-in-time pair formation and spread construction for pairs trading.

WHY THIS IS A DIFFERENT SHAPE FROM EVERY OTHER SIGNAL IN THIS PROJECT.
`signals/*.py` are all `close: pd.DataFrame -> (date x symbol) bool panel`,
matching the single-leg long-only simulator. A pair trade is two
simultaneous legs (long one symbol, short the other) picked from a
CANDIDATE SET that itself has to be chosen ahead of time and refreshed
periodically - there is no single boolean panel that represents "should
symbol X fire today" independent of which OTHER symbol it is paired with.
This module produces the pair candidate list and the spread; the entry/exit
decision on top of it is `signals/pairs_reversion.py`'s job, kept separate
on purpose, same division of responsibility `signals/` already uses versus
`features/`.

PAIR SELECTION METHOD - the "ratio method" (log-price spread, correlation
screen), not the classic Gatev/Goetzmann/Rouwenhorst minimum-distance
method. Both are standard; ratio is simpler to reason about and to keep
point-in-time correct with the tools already in this codebase (the spread
is just `log(A) - log(B)`, z-scored with the ALREADY-BUILT
`technical.rolling_zscore` - no new statistical machinery, no freshly
fitted hedge-ratio regression that could itself overfit a short formation
window). If the premise survives a first screening pass, the
minimum-distance method is a legitimate later comparison, not assumed
superior here.

SAME-SECTOR ONLY, deliberately. Two statistically correlated stocks from
unrelated sectors are disproportionately a coincidence of the sample
window, not a real economic linkage that should persist - the same
reasoning `market_gate`'s own sector-neutralisation finding relied on
(sector was carrying real signal there). Reuses `nifty500.csv`'s industry
column via the same `sector_map` dict every other script in this project
already builds from `cfg.paths.industry_map`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_pairs(
    close: pd.DataFrame,
    sector_map: dict[str, str],
    eligible_symbols: list[str],
    as_of: pd.Timestamp,
    formation_window: int = 252,
    min_corr: float = 0.8,
    max_pairs_per_sector: int = 5,
) -> list[tuple[str, str]]:
    """Same-sector pairs among `eligible_symbols`, ranked by trailing daily
    log-return correlation over the `formation_window` sessions ending on
    or before `as_of` (uses `close.loc[:as_of]` - never a session after
    `as_of`, so calling this at a rebalance date is point-in-time correct
    by construction). Keeps the top `max_pairs_per_sector` pairs per
    sector with correlation >= `min_corr`; a symbol with no sector_map
    entry (a delisted/renamed name not in today's snapshot) is dropped
    rather than guessed into a sector.

    Requires a FULL `formation_window` of return history with no gaps for
    both legs - a pair with any missing data in the window is skipped
    rather than scored on a partial, potentially misleading sample.
    """
    window = close.loc[:as_of].tail(formation_window + 1)
    if len(window) < formation_window + 1:
        return []
    log_ret = np.log(window).diff().iloc[1:]

    by_sector: dict[str, list[str]] = {}
    for sym in eligible_symbols:
        sector = sector_map.get(sym)
        if sector is None or sym not in log_ret.columns:
            continue
        by_sector.setdefault(sector, []).append(sym)

    pairs: list[tuple[str, str, float]] = []
    for sector, symbols in by_sector.items():
        if len(symbols) < 2:
            continue
        sub = log_ret[symbols].dropna(axis=1, how="any")
        cols = list(sub.columns)
        if len(cols) < 2:
            continue
        corr = sub.corr()
        sector_pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = corr.iloc[i, j]
                if pd.notna(c) and c >= min_corr:
                    sector_pairs.append((cols[i], cols[j], float(c)))
        sector_pairs.sort(key=lambda p: p[2], reverse=True)
        pairs.extend(sector_pairs[:max_pairs_per_sector])

    pairs.sort(key=lambda p: p[2], reverse=True)
    return [(a, b) for a, b, _ in pairs]


def random_same_sector_pairs(
    sector_map: dict[str, str],
    eligible_symbols: list[str],
    max_pairs_per_sector: int,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """Same-sector pairs among `eligible_symbols`, picked uniformly at
    random within each sector - NO correlation screen. Up to
    `max_pairs_per_sector` per sector, the same cap `select_pairs` uses,
    so trade COUNT stays comparable across selection rules rather than one
    variant flooding the sample with a big sector's C(n,2) combinations.

    Added 2026-08-18 as its own hypothesis (not a placebo): the 2026-08-17
    pairs re-test found the correlation screen didn't earn its complexity -
    a same-sized RANDOM same-sector draw scored higher, both gross and
    (after the rollover fix) net. That result was only ever measured as a
    placebo for `select_pairs`, sized to match however many correlated
    pairs existed that month - this function is the same idea promoted to
    a real, independently-scoped selection rule with its own cap and its
    own placebo (`random_pairs_any_sector`, below).
    """
    by_sector: dict[str, list[str]] = {}
    for sym in eligible_symbols:
        sector = sector_map.get(sym)
        if sector is not None:
            by_sector.setdefault(sector, []).append(sym)

    pairs: list[tuple[str, str]] = []
    for symbols in by_sector.values():
        if len(symbols) < 2:
            continue
        candidates = [(symbols[i], symbols[j])
                     for i in range(len(symbols)) for j in range(i + 1, len(symbols))]
        n = min(max_pairs_per_sector, len(candidates))
        idx = rng.choice(len(candidates), size=n, replace=False)
        pairs.extend(candidates[i] for i in idx)
    return pairs


def liquidity_ranked_same_sector_pairs(
    sector_map: dict[str, str],
    eligible_symbols: list[str],
    turnover: pd.DataFrame,
    as_of: pd.Timestamp,
    max_pairs_per_sector: int,
    lookback_days: int = 63,
) -> list[tuple[str, str]]:
    """Same-sector pairs formed from each sector's most liquid names - a
    DETERMINISTIC alternative to `random_same_sector_pairs` (no RNG, so a
    re-run always picks the identical pairs), ranked by trailing mean
    turnover over `lookback_days` (matches `config.toml`'s own
    `universe.lookback_days` liquidity window - the same window the
    universe rebalance itself already uses to judge liquidity, reused
    rather than a freshly invented one).

    Takes the smallest k such that C(k,2) >= max_pairs_per_sector
    most-liquid names in the sector, then keeps the
    `max_pairs_per_sector` highest-combined-turnover pairs among THOSE k -
    liquid names are picked as a GROUP first, not by directly optimising
    over all C(n,2) combinations for the single best-turnover pair, so
    this stays "trade the sector's liquid names" and not a second
    correlation-shaped selection rule wearing a liquidity label.
    """
    window = turnover.loc[:as_of].tail(lookback_days)
    mean_turnover = window.mean()

    by_sector: dict[str, list[str]] = {}
    for sym in eligible_symbols:
        sector = sector_map.get(sym)
        if sector is not None and sym in mean_turnover.index and pd.notna(mean_turnover[sym]):
            by_sector.setdefault(sector, []).append(sym)

    pairs: list[tuple[str, str]] = []
    for symbols in by_sector.values():
        if len(symbols) < 2:
            continue
        ranked = sorted(symbols, key=lambda s: mean_turnover[s], reverse=True)
        k = 2
        while k * (k - 1) // 2 < max_pairs_per_sector and k < len(ranked):
            k += 1
        top = ranked[:k]
        candidates = [(top[i], top[j]) for i in range(len(top)) for j in range(i + 1, len(top))]
        candidates.sort(key=lambda p: mean_turnover[p[0]] + mean_turnover[p[1]], reverse=True)
        pairs.extend(candidates[:max_pairs_per_sector])
    return pairs


def random_pairs_any_sector(
    eligible_symbols: list[str],
    n_target: int,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """The placebo for both same-sector selection rules above: pairs drawn
    uniformly at random from the WHOLE eligible universe, ignoring sector
    entirely. If a same-sector rule cannot beat this, sector membership
    isn't the thing doing the work - generic pair mean-reversion would be,
    and the "same-sector" story would not be supported."""
    n = len(eligible_symbols)
    if n < 2 or n_target <= 0:
        return []
    max_pairs = n * (n - 1) // 2
    n_target = min(n_target, max_pairs)
    chosen: set[tuple[str, str]] = set()
    # Rejection sampling - fine at this scale (eligible universe tops out
    # at a few hundred names, n_target at most a few hundred pairs/month).
    while len(chosen) < n_target:
        i, j = rng.choice(n, size=2, replace=False)
        a, b = eligible_symbols[i], eligible_symbols[j]
        pair = (a, b) if a < b else (b, a)
        chosen.add(pair)
    return list(chosen)


def log_spread(price_a: pd.Series, price_b: pd.Series) -> pd.Series:
    """log(A) - log(B) - the ratio-method spread. Positive means A has
    risen relative to B since whatever reference point the caller's own
    z-score window uses; this function itself carries no window - it is
    the raw input `rolling_zscore` is applied to downstream.
    """
    return np.log(price_a) - np.log(price_b)
