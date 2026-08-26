"""MF ownership breadth: at each real Axis+SBI disclosure, go long the
stocks held by the BROADEST number of distinct schemes - a stability/
quality proxy, not a momentum-style entry trigger.

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): a
stock held by only one or two schemes reflects one manager's (or one
fund house's) idiosyncratic call - it could be right, but the market has
no corroborating signal that independent research processes agree. A
stock many distinct schemes independently choose to hold has survived
several separate underwriting decisions, which is closer to a genuine
quality/stability read than a single manager's conviction. This is a
STRUCTURALLY DIFFERENT bet from every other signal built on this data:
`mf_accumulation` and `mf_new_entrant` are both single-day EVENTS (a
disclosure just landed, trade the next few days); this is a persistent
cross-sectional STATE, re-ranked at each disclosure and held until the
next one - the same shape `momentum_signal` already uses for its own
long-horizon claim, not the short-horizon event shape delivery_breakout/
oi_momentum/mf_accumulation all share.

REAL, STATED SCOPE LIMITATION, same as every MF-holdings construction in
this project: Axis+SBI's own combined activity (2 of ~50 AMCs) - breadth
here means "how many of Axis's and SBI's OWN schemes hold this stock",
not "how many of the ~50 AMC industry's schemes do". A real ceiling
follows directly: Axis alone runs ~100 schemes, SBI ~120-130, so a
maximum breadth reading here is nowhere near what true industry-wide
breadth would show - this is a narrower, noisier version of the real
construction, by the same explicit scope decision as every other
MF-holdings signal.

NO NEW SIMULATOR NEEDED, SAME PRECEDENT AS MOMENTUM - `engine/
simulate.py`'s `ExitRule` with `atr_stop_multiple=None` is already a pure
calendar hold, exactly what "enter at each disclosure, hold until the
next one" is. See `scripts/test_mf_breadth.py` for the exact hold-days
convention used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOP_QUANTILE = 0.2  # top quintile of the eligible pool, same convention as momentum
MIN_COMPARABLE = 5


def _rank_desc(values: pd.Series) -> pd.Index:
    """Deterministic rank order: value descending, symbol name as
    tie-break - same pattern `momentum_signal`'s own `_rank_desc` uses,
    for the identical reason (an unstable sort would let exact-tie rank
    assignment vary run to run - breadth ties are common, since it's an
    integer count)."""
    tmp = pd.DataFrame({"value": values.to_numpy(), "symbol": values.index})
    tmp = tmp.sort_values(["value", "symbol"], ascending=[False, True], kind="stable")
    return pd.Index(tmp["symbol"].to_numpy())


def mf_breadth_signal(
    event_breadth_panel: pd.DataFrame,
    top_quantile: float = TOP_QUANTILE,
    min_comparable: int = MIN_COMPARABLE,
) -> pd.DataFrame:
    """`event_breadth_panel` must already be `dtest.features.mf_holdings.
    breadth_panel`'s output run through `to_event_panel` - callers own
    that alignment, the same division of responsibility every signal in
    this project uses. True on the single trading day a disclosed month's
    breadth reading lands in the top `top_quantile` of that SAME month's
    own cross-section of valid readings (a fresh, non-fitted causal rank
    each time, not a fixed threshold). A disclosed month with fewer than
    `min_comparable` stocks with a valid reading produces all-False for
    that day, not a degenerate rank on too small a pool."""
    out = pd.DataFrame(False, index=event_breadth_panel.index, columns=event_breadth_panel.columns)

    non_empty = event_breadth_panel.index[event_breadth_panel.notna().any(axis=1)]
    for date in non_empty:
        row = event_breadth_panel.loc[date]
        pool = row.dropna()
        if len(pool) < min_comparable:
            continue
        ranked = _rank_desc(pool)
        n_top = max(1, int(np.ceil(len(ranked) * top_quantile)))
        out.loc[date, ranked[:n_top]] = True

    return out
