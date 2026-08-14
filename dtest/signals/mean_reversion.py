"""Mean reversion: buy a name that has fallen unusually far below its own
recent average, on the story that a short-term dislocation from temporary
selling pressure partially reverses absent a fundamental change - the
predecessor project's only strategy that survived a genuine walk-forward
validation (46.0% win rate against a ~36-40% breakeven, 73.3% of symbols
profitable, positive net P&L on a held-out window) rather than being tuned
into passing.

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): the
seller pushing a name 1.5+ standard deviations below its own 50-day average in
a short window is disproportionately a forced or panicked seller - margin
calls, index-fund rebalancing flows, tax-loss selling, a single bad headline
overreacted to - rather than someone with new information about permanently
impaired value. That imbalance is expected to partially correct once the
selling pressure itself is exhausted.

THIS IS ALSO THE PREDECESSOR'S WEAKEST LINK, RESTATED HONESTLY. Best per-trade
expectancy of its three systems, WORST portfolio Sharpe (0.119, max drawdown
-46.8%) - concurrent positions (~8, correlated) crash together, because "the
market is falling" and "individually oversold" are not independent events.
`engine/portfolio.py` exists specifically to see that failure mode; a
sizing-independent re-test of this signal alone would repeat the predecessor's
mistake of trusting the metric that cannot see it.

PARAMETERS, and why no trend filter accompanies them. The predecessor measured,
not assumed, that requiring price above the 20/50-day average is MATHEMATICALLY
INCOMPATIBLE with this entry condition: an entry needs price >= 1.5 sigma BELOW
the 50-day average, so it can structurally never also be above it - 0 of 82
sampled entries survived that combination, and the real run produced zero
trades. That result is not re-tested here; it is a closed question and this
signal is deliberately built without a trend gate.

SMA vs EMA for the trailing mean was also measured and found immaterial (26.5%
vs 26.8% train hit rate) - SMA is used here via `rolling_zscore`'s plain
rolling mean, matching the predecessor's kept default.
"""

from __future__ import annotations

import pandas as pd

from dtest.features.technical import rolling_zscore

WINDOW = 50
Z_THRESHOLD = 1.5


def mean_reversion_signal(
    close: pd.DataFrame,
    window: int = WINDOW,
    z_threshold: float = Z_THRESHOLD,
) -> pd.DataFrame:
    """True where `close` sits more than `z_threshold` sample-std deviations
    below its own trailing `window`-day mean, as of that bar's own close.

    Uses `close` alone (no volume/delivery confirmation) - deliberately the
    simplest version of the signal, so any later addition (a participation
    filter, say) can be measured as a delta against this exact baseline rather
    than being entangled with it from the start.
    """
    z = rolling_zscore(close, window)
    return (z < -z_threshold) & z.notna()
