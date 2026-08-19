"""Cross-asset systemic-stress composite - a causal feature, same division of
responsibility `regime.py` and `technical.py` already use (a raw, continuous
read; thresholding is the caller's job).

SIX DIMENSIONS, matching the exact composite already built for Local
Terminal's own Black Swan Radar dashboard (2026-08-18) - not a new
construction invented here, the same one, now run through this project's
causal/point-in-time discipline instead of a single live snapshot:
    - India VIX (level)
    - US VIX (level)
    - market breadth (level, INVERTED - low breadth = high stress)
    - USDINR (20-session % change - a weakening rupee is the stress signal,
      not the raw level, which can sit anywhere for a long stretch without
      acute movement)
    - DXY (level - dollar strength pressures EM currencies/flows)
    - gold (20-session % return - a flight-to-safety rally, not the level)

Every dimension is percentile-ranked against its own trailing `window`
(default 252 sessions, ~1 year) CAUSALLY - only data through and including
that day, exactly as `regime.trailing_return` computes its own trailing
window. HIGH = MORE STRESSED throughout, so the six dimensions can be
averaged directly without a sign-flip table scattered through calling code.

UNKNOWN BLOCKS BOTH GATES, NEVER A FREE PASS - same convention every other
regime/gate feature in this project uses (`regime.py`'s bull/bear split,
`diagnostic_regime_gate.py`'s NaN handling). The composite requires ALL SIX
dimensions to have a valid percentile that day; if even one dimension's own
252-session warm-up or its own data's start date hasn't been reached yet,
the composite is NaN for that day - not silently averaged over 5 of 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 252
CHANGE_WINDOW = 20


def causal_percentile(series: pd.Series, window: int = WINDOW) -> pd.Series:
    """Percentile rank (0-100) of each value against the trailing `window`
    sessions INCLUDING itself - NaN until `window` full observations exist."""
    return series.rolling(window, min_periods=window).apply(
        lambda w: (w <= w[-1]).mean() * 100.0, raw=True)


def cross_asset_stress_composite(
    india_vix: pd.Series,
    us_vix: pd.Series,
    breadth_pct: pd.Series,
    usdinr: pd.Series,
    dxy: pd.Series,
    gold: pd.Series,
    window: int = WINDOW,
) -> pd.DataFrame:
    """Six causal percentile-ranked dimensions (HIGH = more stressed) plus
    their equal-weighted average `composite`. All inputs must already share
    the same DatetimeIndex (the caller's job, same division of
    responsibility every other signal module in this project uses)."""
    india_vix_pct = causal_percentile(india_vix, window)
    us_vix_pct = causal_percentile(us_vix, window)
    breadth_stress_pct = 100.0 - causal_percentile(breadth_pct, window)
    usdinr_chg = usdinr.pct_change(CHANGE_WINDOW) * 100.0
    usdinr_pct = causal_percentile(usdinr_chg, window)
    dxy_pct = causal_percentile(dxy, window)
    gold_chg = gold.pct_change(CHANGE_WINDOW) * 100.0
    gold_pct = causal_percentile(gold_chg, window)

    dims = pd.DataFrame({
        "india_vix_stress": india_vix_pct,
        "us_vix_stress": us_vix_pct,
        "breadth_stress": breadth_stress_pct,
        "usdinr_stress": usdinr_pct,
        "dxy_stress": dxy_pct,
        "gold_stress": gold_pct,
    })
    all_present = dims.notna().all(axis=1)
    composite = dims.mean(axis=1).where(all_present, np.nan)
    dims["composite"] = composite
    return dims
