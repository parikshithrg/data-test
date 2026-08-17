"""Wide-range conviction bar: a session whose own range moved unusually far
relative to its own recent normal, closing at the extreme of that move
(not drifting back to the middle), on unusually heavy participation.
Symmetric - the same construction fires long (closed near the high) or
short (closed near the low); direction is which extreme the close lands
near, not two different rules.

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): an
ordinary session has a "normal" range, and most wide-range sessions are
still indecisive - price whips both directions and settles back near the
middle, telling you nothing except that the day was volatile. A session
that is BOTH unusually wide AND closes pinned at one extreme is a
different claim: participants pushed the price one way and never let it
come back, all session, on volume well above what that stock normally
trades. Someone dismissing this as "just a volatile day" is ignoring that
the combination - wide range, an extreme close (not a mid-range one), AND
unusual volume, together - is disproportionately genuine new information
or real institutional participation entering that name, not noise.

DELIBERATELY DIFFERENT SHAPE FROM `vol_squeeze_breakout.py`, the project's
other volatility-based signal. That one requires a CONTRACTION before the
expansion (a quiet stretch, then a break) - this one requires nothing
about what came before today; today's own bar has to be the outlier,
standing alone. Two different claims about when a wide move means
something, deliberately not tested as one signal.

CLOSE LOCATION, not a fresh construction - `features.price_action.
close_location` is the same (close-low)/(high-low) fraction the
Accumulation/Distribution line has used for decades. PARAMETERS are not
freshly tuned to this signal: `range_z_threshold=1.5` matches
`mean_reversion`'s own z-threshold, `volume_z_threshold=1.0` and
`window=20` match `delivery_breakout`'s confirmation convention -
reusing the project's own existing thresholds rather than fitting new
ones on a signal that has never been tested, avoiding exactly the
free-parameter-tuned-to-nothing trap named repeatedly elsewhere in this
project. `close_location_high/low=0.8/0.2` is the plainest possible
"closed in the top/bottom fifth of the bar, not the middle" cut.
"""

from __future__ import annotations

import pandas as pd

from dtest.features.price_action import close_location
from dtest.features.technical import rolling_zscore, true_range

RANGE_ZSCORE_WINDOW = 20
RANGE_Z_THRESHOLD = 1.5
VOLUME_ZSCORE_WINDOW = 20
VOLUME_Z_THRESHOLD = 1.0
CLOSE_LOCATION_HIGH = 0.8
CLOSE_LOCATION_LOW = 0.2


def price_action_signal(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    range_zscore_window: int = RANGE_ZSCORE_WINDOW,
    range_z_threshold: float = RANGE_Z_THRESHOLD,
    volume_zscore_window: int = VOLUME_ZSCORE_WINDOW,
    volume_z_threshold: float = VOLUME_Z_THRESHOLD,
    close_location_high: float = CLOSE_LOCATION_HIGH,
    close_location_low: float = CLOSE_LOCATION_LOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (long_signal, short_signal) - two boolean (date x symbol)
    panels, matching every other signal's shape, but as a PAIR since this
    is the first signal in the project that is genuinely two-directional
    rather than long-only. A caller feeds `long_signal` through the
    existing long-only cash-equity path and `short_signal` through the
    (separate) futures short path - this function makes no assumption
    about which engine handles either.
    """
    tr = true_range(high, low, close)
    range_z = rolling_zscore(tr, range_zscore_window)
    volume_z = rolling_zscore(volume, volume_zscore_window)
    loc = close_location(high, low, close)

    wide_and_loud = (range_z > range_z_threshold) & (volume_z > volume_z_threshold)
    valid = range_z.notna() & volume_z.notna() & loc.notna()

    long_signal = wide_and_loud & (loc >= close_location_high) & valid
    short_signal = wide_and_loud & (loc <= close_location_low) & valid
    return long_signal, short_signal
