"""Causal market-regime read - NIFTY50's own trailing trend, computed using
only data through and including the labelled date, so gating a same-day
entry decision on it can never leak.

WHY NOT regime_hmm's smoothed Bull/Choppy/Bear. market_gate's own
Markov-switching classifier uses Kim smoothing, which incorporates FUTURE
observations when assigning a historical date's regime label - correct for
describing the past, wrong for gating a decision that has to be made before
the future is known. market_gate's own `research/regime_periods.py`
deliberately built a separate causal label for exactly this reason (see
[[project-market-gate-status]], 2026-08-13) rather than reuse regime_hmm.
This module is that same causal-label idea, ported into this project's own
discipline - a raw FEATURE (continuous trailing return, NaN before enough
history exists), not a pre-thresholded gate. Thresholding and NaN-handling
are the caller's job, same division of responsibility `technical.py`'s
`atr`/`rolling_zscore` already use versus the `signals/` layer.

LOOKBACK = 63 sessions (~3 months) - not freshly tuned here, it is the same
window market_gate's own causal regime study used. Re-deriving it from
scratch on this project's own data is explicitly OUT OF SCOPE for a Phase 0
screening pass; if the premise survives, tuning the window becomes a
legitimate later question with its own train/val discipline.
"""

from __future__ import annotations

import pandas as pd

LOOKBACK = 63


def trailing_return(price: pd.Series, lookback: int = LOOKBACK, skip: int = 0) -> pd.Series:
    """(price `skip` sessions ago / price `skip + lookback` sessions ago) - 1,
    as of today's own close. NaN wherever no prior value exists to compare
    against - not silently coerced to a value that would read as bull or
    bear either way. `price` may be a Series (one symbol, this module's
    original regime-gate use) or a DataFrame (one column per symbol,
    `signals/momentum.py`'s cross-sectional use) - `shift`/division are
    the same operation either way.

    `skip` (default 0, so every existing caller is unaffected) exists for
    `signals/momentum.py`'s 12-1 month construction: skipping the most
    recent `skip` sessions excludes exactly the short-term reversal window
    this project's own entry-timing diagnostic (2026-08-17) found to be a
    real, adverse effect - folding it into a momentum lookback would
    confound two effects already found to point in opposite directions.
    """
    return price.shift(skip) / price.shift(skip + lookback) - 1.0
