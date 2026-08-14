"""Evaluation: sizing-independent metrics, benchmark comparison, noise floor.

This layer is what decides whether a hypothesis ships, and it exists to make
that decision hard to fake. Three rules from the project's own methodology,
enforced here rather than left to discipline:

  benchmark-relative  - every headline is excess over NIFTY50, never absolute
  counted             - every hypothesis tried is logged, win or lose
  noise floor          - a result must clear a placebo band, not just be positive
"""

from dtest.evaluate.metrics import benchmark_excess, non_overlapping_tstat, summary_stats
from dtest.evaluate.placebo import PlaceboResult, run_placebos

__all__ = [
    "summary_stats", "benchmark_excess", "non_overlapping_tstat",
    "run_placebos", "PlaceboResult",
]
