"""Volatility squeeze breakout: buy a name whose own volatility has just
started expanding out of an unusually quiet stretch, confirmed by a genuine
price breakout - not volatility expanding for its own sake.

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): a
price breakout that shows up while a stock's own recent range is already
wide is easy to shrug off - there is no quiet baseline to break out FROM,
so it reads as just another day for a name that always moves a lot. A
breakout that follows a genuine contraction (the stock's own short-term
range has compressed well below its own longer-run normal) is different:
participants have been sitting on their hands, and the move that finally
breaks the range is disproportionately the first real re-pricing after
that quiet, not noise inside an already-active range. This is the one
dimension none of the four Phase 1-3 signals tested - price, delivery, OI,
and FII flow all describe WHO is trading; this describes HOW MUCH the
stock itself has been moving, a structurally different axis. See
[[project-data-test-status]] for the Phase 1-3 score (0 for 4) this is
tested against.

PARAMETERS. Volatility: `short_atr` (fast, `short_atr_window`-bar Wilder
ATR) divided by `long_atr` (`long_atr_window`-bar Wilder ATR, the stock's
own longer-run baseline) - a RATIO, not a percentile or z-score, per an
explicit choice to trigger on a CROSS through parity (short-term vol
re-crossing above long-term vol) rather than an arbitrary percentile
threshold. The cross must be a genuine cross (ratio at or below
`ratio_threshold` on the PRIOR bar, strictly above it today) - a ratio
merely SITTING above threshold for many consecutive bars would fire every
one of those bars, which is "already expanded", not "just started
expanding". Breakout: `close` exceeds the highest close of the PRIOR
`breakout_window` sessions (today excluded), the same construction
`delivery_breakout.py` and `oi_momentum.py` use, so a later comparison
across signals is not confounded by a different breakout convention.

NO EXTERNAL DATA SOURCE. Unlike delivery/OI/participant-flow, ATR is
computed from the price panel alone (`dtest.features.technical.atr`), so
this signal can run on the FULL `primary` split (2004-2026), not just the
shorter `delivery` split - the first Phase 4 signal with that property.

ZERO-ATR GUARD. A stock frozen at one price for an entire `long_atr_window`
stretch (halted, illiquid) has `long_atr == 0`; dividing by it would give
+inf, which trivially clears any ratio threshold - a corrupted signal, not
a real one. Matches `rolling_zscore`'s own `std > 0` guard: those cells
become NaN, not an unbounded ratio.
"""

from __future__ import annotations

import pandas as pd

from dtest.features.technical import atr

SHORT_ATR_WINDOW = 10
LONG_ATR_WINDOW = 50
RATIO_THRESHOLD = 1.0
BREAKOUT_WINDOW = 20


def vol_squeeze_breakout_signal(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    short_atr_window: int = SHORT_ATR_WINDOW,
    long_atr_window: int = LONG_ATR_WINDOW,
    ratio_threshold: float = RATIO_THRESHOLD,
    breakout_window: int = BREAKOUT_WINDOW,
) -> pd.DataFrame:
    """True where the short/long ATR ratio crosses UP through
    `ratio_threshold` (contraction ending) on the SAME bar `close` breaks
    its own prior `breakout_window`-day high (direction confirmed).
    """
    short_atr = atr(high, low, close, window=short_atr_window)
    long_atr = atr(high, low, close, window=long_atr_window)
    ratio = (short_atr / long_atr).where(long_atr > 0)

    prior_ratio = ratio.shift(1)
    crossed_up = (prior_ratio <= ratio_threshold) & (ratio > ratio_threshold)

    prior_high = close.shift(1).rolling(breakout_window, min_periods=breakout_window).max()
    breakout = close > prior_high

    return crossed_up & breakout & ratio.notna()
