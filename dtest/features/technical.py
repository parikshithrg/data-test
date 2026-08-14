"""Technical price transforms. Point-in-time by construction (rolling windows
only ever look backward), but callers are still responsible for not consulting
a value computed AT bar T when deciding something that must be known before T's
close - see `engine/simulate.py`'s use of ATR as of the entry SIGNAL bar, not
the fill bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """True range: the widest of today's range and yesterday's close.

    Uses the previous TRADED close (`ffill().shift(1)`), matching the same
    convention as `corporate_actions.previous_traded_close` - a gap after a
    non-trading bar must not compare against a stale, non-existent "yesterday".
    """
    prior_close = close.ffill().shift(1)
    hl = high - low
    hc = (high - prior_close).abs()
    lc = (low - prior_close).abs()
    return pd.DataFrame(
        np.maximum(np.maximum(hl.to_numpy(), hc.to_numpy()), lc.to_numpy()),
        index=high.index, columns=high.columns,
    )


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
       window: int = 14) -> pd.DataFrame:
    """Average True Range via Wilder's smoothing (EMA with alpha = 1/window).

    Wilder's original method, not a plain rolling mean - it is the convention
    every stop-multiple in Indian retail trading literature assumes, and it
    weights recent volatility more than a flat window average would.
    """
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
