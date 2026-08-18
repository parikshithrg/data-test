"""12-1 month cross-sectional momentum: at each monthly rebalance, go long
the top quintile of the point-in-time eligible universe ranked by trailing
return, skipping the most recent month.

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): a
trader fading a stock's 12-month run is betting the market has been wrong
about that name for a year and is about to correct - a strong claim,
against a genuine trend, made on no more information than "it went up a
lot." Momentum's premise is the opposite and narrower: institutions
build/unwind large positions over months, not days, and information about
a genuine improvement (or deterioration) diffuses into the price slowly
rather than all at once - so a stock that has outperformed for a year is
more likely still mid-diffusion than already fully priced. This is a
structurally different bet from every signal tried so far in this
project: Phase 1-4 (price, delivery, OI, FII flow, volatility) all react
to a dislocation over DAYS and bet on what happens next over the same
short horizon - and every one of them failed the same way, entering right
into the tail of a move that had not finished (see [[project-data-test-
status]]'s 2026-08-18 synthesis entry). Momentum is the first hypothesis
in this project that is long-horizon by construction, not short-horizon
with a different trigger.

WHY 12-1, NOT A STRAIGHT 12-MONTH LOOKBACK. Skipping the most recent
~month specifically excludes the short-term reversal window this
project's own entry-timing diagnostic found to be real and adverse -
buying immediately after a move walks into its own tail. Folding that
window into the momentum lookback would let one effect this project has
already found to be negative (short-term chasing) contaminate a
different, longer-horizon claim being tested here. The classic
Jegadeesh-Titman construction skips it for the same reason, independently
arrived at in the finance literature decades before this diagnostic.

WHY THIS NEEDS NO NEW SIMULATOR, UNLIKE PAIRS. `engine/simulate.py`'s
`ExitRule` already supports a pure calendar hold - `atr_stop_multiple=None`
disables stop/target entirely, so a signal that fires once a month and
holds for one rebalance cycle fits the EXISTING single-leg, long-only
machinery directly. No bespoke trade lifecycle needed, unlike pairs
trading's genuinely different two-leg construction.

LONG-ONLY, STATED SCOPE. A real short leg (shorting the bottom quintile,
the classic winners-minus-losers construction) would need the single-leg
futures short simulator this project has never built (see the 2026-08-18
price-action SHORT finding) - deliberately out of scope for this first
test, which reuses only already-tested infrastructure. Worth revisiting
if the long side shows something real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtest.features.regime import trailing_return

LOOKBACK_DAYS = 252   # ~12 months of trading sessions
SKIP_DAYS = 21        # ~1 month
TOP_QUANTILE = 0.2    # top quintile of the eligible pool


def _rank_desc(values: pd.Series) -> pd.Index:
    """Deterministic rank order: value descending, symbol name as
    tie-break - same pattern `universe.py::_rank_by_turnover` already
    uses, for the identical reason: an unstable sort would let rank
    assignment for exact ties vary run to run."""
    tmp = pd.DataFrame({"value": values.to_numpy(), "symbol": values.index})
    tmp = tmp.sort_values(["value", "symbol"], ascending=[False, True], kind="stable")
    return pd.Index(tmp["symbol"].to_numpy())


def momentum_signal(
    close: pd.DataFrame,
    universe_membership: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    lookback_days: int = LOOKBACK_DAYS,
    skip_days: int = SKIP_DAYS,
    top_quantile: float = TOP_QUANTILE,
) -> pd.DataFrame:
    """Boolean (date x symbol) entry panel: True ONLY on a rebalance date,
    for symbols in the top `top_quantile` of that date's point-in-time
    eligible pool (`universe_membership.loc[d]`, same convention every
    other script in this project already uses to read universe state at a
    rebalance date) ranked by `trailing_return(skip=skip_days,
    lookback=lookback_days)`. A symbol needs a full, gap-free
    `skip_days + lookback_days` history to be scored - `dropna()` removes
    anything without one rather than ranking on a partial window.

    Point-in-time by construction: `trailing_return` only ever looks
    backward from each date via `shift`, and `universe_membership` is
    already the caller's own point-in-time universe.
    """
    ret = trailing_return(close, lookback=lookback_days, skip=skip_days)
    signal = pd.DataFrame(False, index=close.index, columns=close.columns)

    for d in rebalance_dates:
        if d not in ret.index or d not in universe_membership.index:
            continue
        elig_cols = universe_membership.columns.intersection(ret.columns)
        elig_syms = elig_cols[universe_membership.loc[d, elig_cols].to_numpy()]
        pool = ret.loc[d, elig_syms].dropna()
        if pool.empty:
            continue

        ranked = _rank_desc(pool)
        n_top = max(1, int(np.ceil(len(ranked) * top_quantile)))
        signal.loc[d, ranked[:n_top]] = True

    return signal
