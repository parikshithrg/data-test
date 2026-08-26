"""MF accumulation: buy a stock on the day its combined Axis+SBI mutual
fund holdings are newly disclosed to have grown, month-over-month, more
than most other stocks' holdings did that same month.

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): a
fund manager adding to an EXISTING position (not opening a new one - see
`dtest.features.mf_holdings`'s own docstring for why new entrants are a
separate, unscoped hypothesis) has already done the underlying research
and is choosing to size up with real capital, not merely holding. The
market has not yet reacted to this specific disclosure at the moment it
becomes public (`period_end + 10 days`, the same assumed SEBI filing lag
this project already uses elsewhere) - anyone trading on the stale
month-old holding itself is late; this signal is a bet that conviction
buying which has JUST become visible still has real information content
for the following days, not that it is already priced in.

REAL, STATED SCOPE LIMITATION - this is Axis+SBI's own combined activity
(2 of ~50 AMCs), not aggregate mutual-fund-industry flow. See
`dtest.features.mf_holdings`'s own module docstring for the full caveat;
repeated here because every result this signal produces inherits it.

CROSS-SECTIONAL, NOT AN ABSOLUTE THRESHOLD - `top_percentile` is computed
FRESH each disclosed month from that month's own distribution of valid
%% changes (a genuinely causal computation: only that month's already-
disclosed data, no lookahead), not a fixed number tuned to look good on
one period. A month with fewer than `min_comparable` stocks with a valid
change is skipped entirely (no percentile is meaningful on too small a
cross-section) rather than firing on whatever happens to clear a
degenerate threshold.
"""

from __future__ import annotations

import pandas as pd

TOP_PERCENTILE = 90.0
MIN_COMPARABLE = 5


def mf_accumulation_signal(
    event_pct_change: pd.DataFrame,
    top_percentile: float = TOP_PERCENTILE,
    min_comparable: int = MIN_COMPARABLE,
) -> pd.DataFrame:
    """True on the single trading day a stock's disclosed month-over-month
    MF-quantity change lands at or above the `top_percentile`th percentile
    of that SAME disclosed month's own cross-section of valid changes.

    `event_pct_change` must already be aligned onto the daily calendar via
    `dtest.features.mf_holdings.to_event_panel` - callers own that
    alignment, the same division of responsibility every signal in this
    project uses. Rows with fewer than `min_comparable` non-NaN values
    (either a genuinely quiet calendar day, or a real disclosed month with
    too thin a cross-section) produce all-False, not a degenerate
    percentile."""
    out = pd.DataFrame(False, index=event_pct_change.index, columns=event_pct_change.columns)
    quantile = top_percentile / 100.0

    # Only real disclosed-month rows carry any data - every other trading
    # day is all-NaN by construction (`to_event_panel` never ffills), so
    # skip straight to the sparse rows worth ranking rather than iterating
    # the full multi-year daily calendar.
    non_empty = event_pct_change.index[event_pct_change.notna().any(axis=1)]
    for date in non_empty:
        row = event_pct_change.loc[date]
        valid = row.dropna()
        if len(valid) < min_comparable:
            continue
        cutoff = valid.quantile(quantile)
        out.loc[date] = row >= cutoff

    return out.fillna(False)
