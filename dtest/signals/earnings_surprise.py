"""Earnings surprise / PEAD (post-earnings-announcement drift): buy a name
right after it discloses standalone quarterly EPS meaningfully above its
own trailing surprise history.

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): a
quarter's EPS print is not instantly and fully absorbed into the price the
moment it is disclosed - institutions and other slower participants take
time to update position sizes on genuinely new information, so the market
tends to keep drifting in the direction of a real surprise for weeks
afterward rather than jumping straight to the new fair value (Bernard &
Thomas 1989's original PEAD finding, replicated across many markets
since). The trader fading a real earnings beat on no more information than
"it already popped" is betting the market fully priced the news in one
session, a strong claim with nothing behind it. UNLIKE every Phase 1-4
signal in this project (all react to a DAYS-scale PRICE/flow dislocation
and bet on what happens next over the same short horizon - and all failed
the same way, entering into the tail of an unfinished move, see
[[project-data-test-status]]'s synthesis), this reacts to a discrete,
dated, fundamental EVENT with its own real disclosure timestamp - a
structurally different information source, not another variant of the
same price-reactive shape.

NO CONSENSUS ESTIMATES EXIST anywhere in this project (no analyst-estimate
data source was found or built) - "surprise" here is the standard
SUE (standardized unexpected earnings) proxy used exactly for this
situation: this quarter's standalone EPS minus the SAME quarter last year
("expected" = a naive seasonal random walk, not analyst consensus - stated
plainly, a real limitation of that proxy, not disguised as something
stronger), z-scored against the stock's own trailing 8-quarter history of
such surprises (`dtest.features.fundamentals.sue_zscore` - reused, not
reinvented).

FIRES ONCE PER FILING, AS A SHARP EVENT - not "every day the surprise
stays elevated." A filing is a one-time disclosure, not a continuously
updating bar; re-firing on every day its ffill'd value stays above
threshold would (a) not match how any real PEAD strategy is actually
implemented (enter once, hold a fixed window) and (b) walk straight back
into this project's own already-diagnosed entry-timing problem by
"chasing" a surprise that is, days later, no longer fresh news. The signal
is keyed off `filing_date` directly (mapped to the same trading day if it
is one, else rolled forward to the next trading day - the same
"knowable-by-close, fillable at T+1-open" convention `features/
fundamentals.py::to_daily_panel` documents), not off any ffill'd plateau.

CALENDAR HOLD, NOT ATR-STOP. Same reasoning `momentum.py` already
documented for its own construction: PEAD is explicitly about a multi-week
drift, not a volatility-stop-managed short-term trade - `ExitRule(
atr_stop_multiple=None)` (a pure calendar hold) is the right exit
mechanic, decided here, applied by the caller's own `ExitRule`
construction (this module only builds the entry panel).
"""

from __future__ import annotations

import pandas as pd

SUE_THRESHOLD = 1.0


def earnings_surprise_signal(
    sue_per_symbol: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
    threshold: float = SUE_THRESHOLD,
) -> pd.DataFrame:
    """True on exactly one trading day per qualifying filing (SUE >
    `threshold`): `filing_date` itself if it is a trading day, else the
    next trading day after it (`searchsorted` after normalizing away the
    filing's own intraday time, so a post-close disclosure timestamp
    doesn't spuriously roll to the following session).
    """
    symbols = list(sue_per_symbol.keys())
    signal = pd.DataFrame(False, index=calendar, columns=symbols)
    for symbol, s in sue_per_symbol.items():
        fires = s[s > threshold]
        for filing_date in fires.index:
            day = pd.Timestamp(filing_date).normalize()
            pos = calendar.searchsorted(day)
            if pos >= len(calendar):
                continue
            signal.loc[calendar[pos], symbol] = True
    return signal
