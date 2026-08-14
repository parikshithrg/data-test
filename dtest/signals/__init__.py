"""Signals. Each is a pure function: point-in-time panels -> (date x symbol)
boolean panel. A signal never reads the universe membership panel and never
applies a trend/eligibility filter of its own - see `dtest/universe.py`'s
docstring on why ranking, eligibility, and signal logic are kept in separate
layers. The caller (a research script) intersects a signal with the universe
mask before handing it to the simulator.

Every signal here is expected to carry an economic STORY in its own docstring,
matching the mandatory field in `evaluate.hypothesis_log.HypothesisEntry` - a
function with no story is a coefficient in a search, not a hypothesis.
"""

from dtest.signals.mean_reversion import mean_reversion_signal

__all__ = ["mean_reversion_signal"]
