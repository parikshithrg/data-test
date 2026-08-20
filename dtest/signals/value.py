"""Value: buy a name whose trailing P/E has dropped meaningfully below its
own recent normal - "cheap vs. own history," not sector- or market-
relative (no consensus, sector-multiple, or reserves/book-value data
reliably exists in this project - see `dtest/data/financial_results.py`'s
own module docstring on why P/B is out of scope here).

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): a
stock trading at a P/E well below what the market has recently been
willing to pay for its OWN trailing-twelve-month earnings is
disproportionately underpriced relative to its own recent valuation
regime - the classic value premise, here operationalized against the
stock's own history rather than a peer or market multiple. The trader
buying it back UP to "fair" on no more information than "the multiple
looks cheap" is betting the market's recent re-rating was wrong with
nothing specific behind that view beyond the ratio itself.

A REAL, STATED RISK OF OVERLAP WITH mean_reversion (already rejected,
2026-08-14/19) - flagged before running, not discovered after: P/E =
price / trailing EPS, and EPS only updates quarterly while price updates
daily, so most day-to-day P/E movement is mechanically PRICE movement, not
an earnings change. A "cheap P/E" reading can very often just mean "the
price fell recently" - the same phenomenon `mean_reversion.py` already
tests directly on price alone, and which failed decisively (0/2 across
both splits, entry-timing mechanism confirmed). This signal is only doing
something DIFFERENT from mean_reversion to the extent EPS itself, not just
price, is moving the ratio - a real possibility this test cannot separate
from the price-only story on its own; worth reading the result with that
overlap in mind, not as an independent confirmation either way.

CONSTRUCTION: `trailing_eps_ttm` (`features.fundamentals.trailing_ttm` on
the standalone EPS series, 4 quarters) reindexed to a daily panel via
`to_daily_panel`. `pe = close / eps_ttm`, masked to `eps_ttm > 0` (a
negative or zero TTM EPS makes P/E meaningless/sign-flipped - screened
out, not inverted or clipped). `z = rolling_zscore(pe, window=PE_ZSCORE_
WINDOW)` (252 sessions, ~1 trading year - the stock's own recent
valuation regime). Fires on a genuine DOWNWARD cross through
`-Z_THRESHOLD` (z at or above threshold on the PRIOR bar, strictly below
today) - the same "cross, not sit-and-refire" convention `vol_squeeze_
breakout.py` already established, so a name that stays cheap for months
does not generate a new signal every single day.
"""

from __future__ import annotations

import pandas as pd

from dtest.features.technical import rolling_zscore

PE_ZSCORE_WINDOW = 252
Z_THRESHOLD = 1.0


def value_signal(
    close: pd.DataFrame,
    eps_ttm_panel: pd.DataFrame,
    zscore_window: int = PE_ZSCORE_WINDOW,
    z_threshold: float = Z_THRESHOLD,
) -> pd.DataFrame:
    """True where trailing P/E crosses DOWN through `-z_threshold`
    std deviations of its own trailing `zscore_window`-day history.

    `eps_ttm_panel` must already be reindexed onto `close`'s index/columns
    (daily, forward-filled from quarterly filings) - callers own that
    alignment, the same division of responsibility every signal in this
    project uses for an external data panel.
    """
    pe = (close / eps_ttm_panel).where(eps_ttm_panel > 0)
    z = rolling_zscore(pe, window=zscore_window)

    prior_z = z.shift(1)
    crossed_down = (prior_z >= -z_threshold) & (z < -z_threshold)

    return crossed_down & z.notna()
