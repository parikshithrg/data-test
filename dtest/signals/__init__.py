"""Signals. Each is a pure function: point-in-time panels -> (date x symbol)
boolean panel. A signal never reads the universe membership panel and never
applies a trend/eligibility filter of its own - see `dtest/universe.py`'s
docstring on why ranking, eligibility, and signal logic are kept in separate
layers. The caller (a research script) intersects a signal with the universe
mask before handing it to the simulator.

Every signal here is expected to carry an economic STORY in its own docstring,
matching the mandatory field in `evaluate.hypothesis_log.HypothesisEntry` - a
function with no story is a coefficient in a search, not a hypothesis.

HIDDEN 2026-08-17 - no signal is currently endorsed. All six hypotheses
tried so far (mean_reversion, delivery_breakout, oi_momentum,
participant_tilt, vol_squeeze_breakout, and its delay=2 variant) are
REJECTED - see runs/hypothesis_log.csv and HIDDEN_STRATEGIES.md at the
Dashboard root for the full record. Nothing is deleted (each module and
its tests stay intact, importable directly by module path, e.g.
`from dtest.signals.mean_reversion import mean_reversion_signal`) - this
package just re-exports nothing, so `from dtest.signals import <name>`
finds no "current" signal to accidentally pick up.
"""

__all__: list[str] = []
