"""OI-confirmed momentum: buy a name closing above its own trailing N-day
high while stock-futures open interest is building unusually fast - real
new positioning, not short-covering or a contract-lifecycle artifact.

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): a
price breakout on FALLING or merely average open interest is
disproportionately short-covering or spot-only buying with no fresh
leveraged conviction behind it. A breakout accompanied by open interest
rising meaningfully faster than its own recent normal means participants
are opening NEW leveraged positions into the move, which is more costly to
reverse than an unlevered spot rally and disproportionately reflects real
conviction rather than noise.

THE EXPIRY-CYCLE CONFOUND, measured before trusting raw OI change as a
feature at all (see `dtest.data.fno_oi` module docstring for the full
mechanism). Stock-futures OI mechanically RAMPS for roughly the first two
weeks after a rollover (traders rebuilding the new front month) and
mechanically DECAYS in the final week before expiry (traders rolling OUT)
- both real, both entirely about contract lifecycle, neither about that
day's market sentiment. Confirmed directly on RELIANCE Jan-Mar 2024:
`oi_chg_pct` swings from +6.8% the day AFTER a rollover to -68.5% on the
expiry day itself, a 75-point range driven by nothing but calendar
position. A z-score of raw `oi_chg_pct` with no expiry-cycle control would
mistake this mechanical pattern for a burst of conviction (right after
every rollover) or its opposite (right before every expiry), firing on a
predictable calendar effect rather than on anything informative.

THE GUARD: `days_to_expiry` (from `fno_oi.load_front_month_oi`) must fall
inside `[min_days_to_expiry, max_days_to_expiry]` - the "stable middle" of
the ~30-day contract cycle, away from both the post-rollover ramp and the
pre-expiry decay. This does not prove the remaining OI changes are
sentiment rather than something else; it only removes one specific,
already-measured mechanical confound before the signal is tested, rather
than leaving it in and mistaking calendar structure for edge.

PARAMETERS. Breakout: `close` exceeds the highest close of the PRIOR
`breakout_window` sessions (today excluded), identical construction to
`delivery_breakout.py`. Confirmation: `oi_chg_pct` at least `z_threshold`
sample-std deviations above its own trailing `zscore_window`-day mean -
same z-score convention as every other signal in this project.
"""

from __future__ import annotations

import pandas as pd

from dtest.features.technical import rolling_zscore

BREAKOUT_WINDOW = 20
ZSCORE_WINDOW = 20
Z_THRESHOLD = 1.0
MIN_DAYS_TO_EXPIRY = 5
MAX_DAYS_TO_EXPIRY = 25


def oi_momentum_signal(
    close: pd.DataFrame,
    oi_chg_pct: pd.DataFrame,
    days_to_expiry: pd.DataFrame,
    breakout_window: int = BREAKOUT_WINDOW,
    zscore_window: int = ZSCORE_WINDOW,
    z_threshold: float = Z_THRESHOLD,
    min_days_to_expiry: int = MIN_DAYS_TO_EXPIRY,
    max_days_to_expiry: int = MAX_DAYS_TO_EXPIRY,
) -> pd.DataFrame:
    """True where `close` breaks its own prior `breakout_window`-day high AND
    `oi_chg_pct` is at least `z_threshold` std above its own trailing
    `zscore_window`-day mean AND `days_to_expiry` sits inside the safe band.

    `oi_chg_pct` and `days_to_expiry` must already be reindexed onto
    `close`'s index/columns - callers own that alignment, the same division
    of responsibility every signal in this project uses.
    """
    prior_high = close.shift(1).rolling(breakout_window, min_periods=breakout_window).max()
    breakout = close > prior_high

    z = rolling_zscore(oi_chg_pct, zscore_window)
    confirmed = z > z_threshold

    safe_cycle = (days_to_expiry >= min_days_to_expiry) & (days_to_expiry <= max_days_to_expiry)

    return breakout & confirmed & z.notna() & safe_cycle.fillna(False)
