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


def rolling_zscore(x: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """(x - trailing mean) / trailing std, using only bars up to and including
    today - a real decision made at today's close can see today's own value, so
    this is NOT look-ahead. `ddof=1` (sample std) throughout this project, for
    consistency with `SummaryStats.std_net_pct` and everywhere else a std is
    reported.

    A trailing std of exactly 0 (a symbol frozen at one price for the whole
    window - possible for a thin or halted name) would divide to +/-inf, not a
    real signal; those cells are set to NaN rather than an unbounded z-score
    that would trivially "pass" any threshold.
    """
    mp = min_periods if min_periods is not None else window
    mean = x.rolling(window, min_periods=mp).mean()
    std = x.rolling(window, min_periods=mp).std(ddof=1)
    z = (x - mean) / std
    return z.where(std > 0)
