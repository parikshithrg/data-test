"""Participant-flow regime tilt: gate mean_reversion entries to days when
FII index-futures positioning is itself in an accumulation regime, not a
distribution one.

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): a
mean-reversion dip bought while FII net index-futures positioning sits
ABOVE its own recent trend is disproportionately a normal pullback inside
a market institutions are still net accumulating into - the flow that
matters most for index-level support is still constructive. The identical
dip bought while FII positioning sits BELOW its own recent trend
(distribution) is disproportionately the early stage of a real
breakdown - buying it fights the flow that actually moves the market,
not just the individual name's own price action.

WHY THIS GATES mean_reversion SPECIFICALLY, rather than being a fresh
per-symbol signal. NSE's participant-OI report is published at the INDEX
level only - there is no per-stock FII position. So this cannot select
WHICH stock to buy (that is still `mean_reversion_signal`'s job); it can
only decide WHETHER any dip should be bought AT ALL on a given day,
uniformly across every symbol - a market-wide gate, structurally the same
shape as market_gate's own `allowed_regimes` regime gate (`core/engine.py`)
blocking NEW ENTRIES only, never forcing an exit.

REGIME, not EVENT - deliberately a different statistical shape than
`delivery_breakout`/`oi_momentum`'s spike-style z>1.0 confirmation. Those
ask "is today unusual". This asks "which side of normal is today on" - a
persistent state, not a one-day event - so the default threshold is 0.0
(simply above/below `fii_net_index_oi`'s own trailing mean), not the same
z_threshold=1.0 event bar used elsewhere in this project.
"""

from __future__ import annotations

import pandas as pd

from dtest.features.technical import rolling_zscore
from dtest.signals.mean_reversion import mean_reversion_signal

ZSCORE_WINDOW = 20
Z_THRESHOLD = 0.0


def participant_tilt_signal(
    close: pd.DataFrame,
    fii_net_index_oi: pd.Series,
    zscore_window: int = ZSCORE_WINDOW,
    z_threshold: float = Z_THRESHOLD,
) -> pd.DataFrame:
    """`mean_reversion_signal(close)`, gated to dates where FII net
    index-futures OI sits at least `z_threshold` std above its own trailing
    `zscore_window`-day mean - an accumulation regime, broadcast identically
    across every symbol (this data has no per-symbol breakdown).

    `fii_net_index_oi` must already be reindexed onto `close`'s date index,
    same length and order - callers own that alignment, the same division
    of responsibility every signal in this project uses.
    """
    base = mean_reversion_signal(close)

    z = rolling_zscore(fii_net_index_oi.to_frame(name="flow"), zscore_window)["flow"]
    accumulating = (z > z_threshold) & z.notna()

    gate = accumulating.to_numpy()[:, None]
    return pd.DataFrame(base.to_numpy() & gate, index=base.index, columns=base.columns)
