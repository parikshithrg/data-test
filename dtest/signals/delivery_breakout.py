"""Delivery-confirmed breakout: buy a name closing above its own trailing
N-day high on unusually high delivery, not just volume.

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): a
price breakout on ordinary or low delivery is disproportionately intraday
speculation squaring off the same day - momentum with no one actually
taking the shares home, prone to reverse once that flow unwinds overnight.
A breakout where `delivery_pct` sits meaningfully above ITS OWN recent
normal is disproportionately real buyers converting the move into settled
positions, which the market then has to keep absorbing rather than give
back. This is the one piece of data `market_gate`'s live composite ever
blended into a multi-term score (its `participation` term merged delivery
with volume) but never tested on its own, under honest execution, over
real history - see [[project-market-gate-status]] and the Phase 3 plan in
[[project-data-test-status]].

PARAMETERS. Breakout: `close` exceeds the highest close of the PRIOR
`breakout_window` sessions (today excluded - a breakout is relative to
what came before it, not to itself). Confirmation: that day's
`delivery_pct` sits at least `z_threshold` sample-std deviations above its
own trailing `zscore_window`-day mean - the SAME z-score construction
`mean_reversion.py` uses on price, just applied to delivery instead, so a
later comparison between the two signals is not confounded by a different
statistical convention.

REQUIRES THE DELIVERY SPLIT. `delivery_pct` is NaN before 2019-06-27 (see
`dtest.data.delivery.EARLIEST_DATE`) - `rolling_zscore` already refuses to
produce a z-score against an all-NaN window, so this signal fires zero
trades on `primary`/train by construction, not by an extra guard elsewhere.
Report results only on `[splits.delivery]`, and treat the evidence bar as
higher there per that split's own config comment (a third of the history).
"""

from __future__ import annotations

import pandas as pd

from dtest.features.technical import rolling_zscore

BREAKOUT_WINDOW = 20
ZSCORE_WINDOW = 20
Z_THRESHOLD = 1.0


def delivery_breakout_signal(
    close: pd.DataFrame,
    delivery_pct: pd.DataFrame,
    breakout_window: int = BREAKOUT_WINDOW,
    zscore_window: int = ZSCORE_WINDOW,
    z_threshold: float = Z_THRESHOLD,
) -> pd.DataFrame:
    """True where `close` breaks its own prior `breakout_window`-day high AND
    that day's `delivery_pct` is at least `z_threshold` std above its own
    trailing `zscore_window`-day mean.

    `delivery_pct` must already be reindexed onto `close`'s index/columns -
    callers own that alignment (see `scripts/test_delivery_breakout.py`),
    the same division of responsibility every signal in this project uses.
    """
    prior_high = close.shift(1).rolling(breakout_window, min_periods=breakout_window).max()
    breakout = close > prior_high

    z = rolling_zscore(delivery_pct, zscore_window)
    confirmed = z > z_threshold

    return breakout & confirmed & z.notna()
