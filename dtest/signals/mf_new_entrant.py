"""MF new-entrant: buy a stock on the day it is newly disclosed that some
Axis or SBI scheme opened a BRAND-NEW position in it - not merely grew an
existing one.

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): a
fund manager growing a position they already hold has a spread of
possible motives - rebalancing, a small conviction top-up, riding a name
that has already worked. A manager opening a position from ZERO has just
made a discrete, deliberate decision to underwrite a name for the first
time - the underlying research call is freshest and least ambiguous right
at that moment. This is a DIFFERENT, more discrete construction than
`mf_accumulation` (rejected 2026-08-26, t=-2.945): that signal explicitly
excluded new positions to isolate "grew an existing holding" as its own
story; this one isolates the complementary event, "opened a position that
didn't exist last month" (see `dtest.features.mf_holdings.new_entrant_
flag`'s own docstring for the exact scope, including why a re-buy after a
genuine sell-out counts again).

REAL, STATED SCOPE LIMITATION, same as every MF-holdings construction in
this project: Axis+SBI's own combined activity (2 of ~50 AMCs), not
mutual-fund-industry-wide flow.

NO THRESHOLD TO TUNE, UNLIKE mf_accumulation - `new_entrant_flag` is
already a discrete yes/no event (a position exists or it doesn't), so
this signal is a direct pass-through of that event panel once mapped onto
the daily calendar via `dtest.features.mf_holdings.to_event_panel` - no
cross-sectional percentile threshold to pick, because there is no
continuous quantity to rank.
"""

from __future__ import annotations

import pandas as pd


def mf_new_entrant_signal(event_new_entrant_panel: pd.DataFrame) -> pd.DataFrame:
    """`event_new_entrant_panel` must already be `new_entrant_flag`'s
    output run through `dtest.features.mf_holdings.to_event_panel` -
    callers own that alignment, the same division of responsibility every
    signal in this project uses. True on exactly the trading day a
    disclosed month's new-entrant event becomes known, False everywhere
    else (including every NaN cell, i.e. every day with no real disclosed
    event)."""
    return event_new_entrant_panel.fillna(False).astype(bool)
