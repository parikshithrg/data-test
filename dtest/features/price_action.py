"""Close-location-within-bar - where today's close sits inside today's own
high-low range, as a fraction from 0 (closed at the low) to 1 (closed at
the high). A standard price-action primitive (the same construction the
Accumulation/Distribution line uses), not something invented for this
project - kept here rather than folded into `signals/price_action.py`
because it is a pure per-bar transform, same division of responsibility
`technical.py` already uses for `true_range`/`rolling_zscore`.
"""

from __future__ import annotations

import pandas as pd


def close_location(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """(close - low) / (high - low), per bar. NaN where high == low (a
    frozen/halted bar has no range to locate the close within) rather than
    a divide-by-zero-driven +/-inf - same "no real signal, don't fake one"
    guard `rolling_zscore`'s own `std > 0` check uses.
    """
    rng = high - low
    return ((close - low) / rng).where(rng > 0)
