"""Pairs mean-reversion: within a formed pair, buy the leg that has drifted
cheap relative to the other and sell the leg that has drifted rich, on the
story that the two are economically linked (same sector, `features/
pairs.py`'s correlation screen) and a wide spread is disproportionately
temporary - one leg overreacting to news that is really about the sector,
not that specific company - rather than a genuine re-rating of their
relative value.

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): a
trader who sells the rich leg outright is betting the whole sector is
overvalued, and one who buys the cheap leg outright is betting the whole
sector is undervalued - both are directional bets on market-wide sentiment
that this trade does NOT need to be right about. This position only needs
the RELATIVE mispricing between two similar businesses to close, which is a
narrower, more mechanical claim (index-fund flows, a sector ETF's own
rebalancing, one stock's earnings surprise bleeding into its peer's price)
than "the sector's fair value has changed."

DELIBERATELY NOT WIRED INTO `engine/simulate.py`. That simulator is
single-leg, long-only, T+1-open-fill, real-cost-model - none of which this
screening pass has (see the module docstring in `scripts/
diagnostic_pairs_reversion.py` for exactly what is simplified and why). This
module answers "does the spread-reversion phenomenon exist at all" as a
CHEAP first read, same sequencing the exit-geometry and entry-delay
diagnostics used before their own more expensive follow-ups - a genuine
honest two-leg simulator (real futures short costs, T+1 fills on both legs)
is deliberately NOT built until this screening pass earns it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dtest.features.pairs import log_spread
from dtest.features.technical import rolling_zscore

Z_ENTRY = 2.0
Z_EXIT = 0.5
ZSCORE_WINDOW = 20
MAX_HOLD_DAYS = 20


@dataclass
class PairTrade:
    symbol_a: str
    symbol_b: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    long_symbol: str
    short_symbol: str
    entry_z: float
    exit_z: float | None
    exit_reason: str   # "reverted" | "time" | "window_end"
    held_days: int


def pair_trade_events(
    price_a: pd.Series,
    price_b: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbol_a: str,
    symbol_b: str,
    zscore_window: int = ZSCORE_WINDOW,
    z_entry: float = Z_ENTRY,
    z_exit: float = Z_EXIT,
    max_hold_days: int = MAX_HOLD_DAYS,
) -> list[PairTrade]:
    """Walk the z-scored log-spread of (A, B) between `start` and `end`
    (inclusive), opening a trade whenever |z| >= z_entry and no trade is
    currently open for this pair, closing it at whichever comes first:
    |z| <= z_exit ("reverted"), `max_hold_days` sessions held ("time"), or
    `end` itself ("window_end" - the pair's own validity period, e.g. until
    the next universe rebalance, ran out first). One position at a time per
    pair, same `busy_until` convention `engine/simulate.py` uses for
    single-leg trades - a live position blocks a new entry until it
    resolves, rather than pyramiding into the same pair.

    z is computed on `price_a`/`price_b`'s FULL history up to each date
    (not just the [start, end] slice), so the rolling window has real
    lookback at `start` rather than an artificial cold start.
    """
    spread = log_spread(price_a, price_b)
    z = rolling_zscore(spread.to_frame("s"), zscore_window)["s"]

    window_idx = z.index[(z.index >= start) & (z.index <= end)]
    trades: list[PairTrade] = []
    busy_until: pd.Timestamp | None = None

    for i, d in enumerate(window_idx):
        if busy_until is not None and d <= busy_until:
            continue
        zt = z.get(d)
        if pd.isna(zt) or abs(zt) < z_entry:
            continue

        # A is rich relative to B (positive z) -> long B, short A; and
        # the mirror when z is negative.
        if zt > 0:
            long_sym, short_sym = symbol_b, symbol_a
        else:
            long_sym, short_sym = symbol_a, symbol_b

        entry_date = d
        remaining = window_idx[i + 1:]
        exit_date, exit_z, exit_reason = end, None, "window_end"
        held = 0
        capped = remaining[:max_hold_days]
        for j, fd in enumerate(capped, start=1):
            fz = z.get(fd)
            if pd.notna(fz) and abs(fz) <= z_exit:
                exit_date, exit_z, exit_reason, held = fd, float(fz), "reverted", j
                break
        else:
            # Never reverted within the days actually available. Distinguish
            # WHY the walk stopped: `max_hold_days` reached with more window
            # still ahead ("time") versus the pair's own validity window
            # (e.g. until the next universe rebalance) running out first
            # ("window_end") - both fall into this branch, but they mean
            # different things to a caller deciding whether to re-enter.
            if len(capped):
                exit_date = capped[-1]
                exit_z = z.get(exit_date)
                exit_reason = "time" if len(capped) == max_hold_days else "window_end"
                held = len(capped)

        trades.append(PairTrade(
            symbol_a=symbol_a, symbol_b=symbol_b, entry_date=entry_date, exit_date=exit_date,
            long_symbol=long_sym, short_symbol=short_sym, entry_z=float(zt),
            exit_z=float(exit_z) if exit_z is not None and pd.notna(exit_z) else None,
            exit_reason=exit_reason, held_days=held,
        ))
        busy_until = exit_date

    return trades
