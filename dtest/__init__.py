"""Data test - a deterministic research harness for Indian cash-equity strategies.

Design rules, in priority order. Each exists because it was learned the hard way
in the predecessor project (market_gate); the reasoning is kept next to the code
that enforces it rather than in a chat log.

1. **Deterministic.** Same inputs, byte-identical outputs, provable via run
   manifests. Nothing calls `date.today()`.
2. **Point-in-time.** No datum reaches a decision before it existed. The universe
   is a recomputable rule, not today's index membership.
3. **Executable.** Signals at bar T's close fill at bar T+1's OPEN, with costs
   and a participation cap. A price you could not have traded is not a result.
4. **Benchmark-relative.** Every headline is excess over NIFTY50, on capital,
   after costs.
5. **Counted.** Every hypothesis tested is logged, so the significance bar can
   rise with the size of the search.
"""

from dtest.config import Config, load_config
from dtest.determinism import SEED, RunManifest, set_seeds

__all__ = ["Config", "load_config", "RunManifest", "set_seeds", "SEED"]
__version__ = "0.1.0"
