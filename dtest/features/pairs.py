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


def log_spread(price_a: pd.Series, price_b: pd.Series) -> pd.Series:
    """log(A) - log(B) - the ratio-method spread. Positive means A has
    risen relative to B since whatever reference point the caller's own
    z-score window uses; this function itself carries no window - it is
    the raw input `rolling_zscore` is applied to downstream.
    """
    return np.log(price_a) - np.log(price_b)
