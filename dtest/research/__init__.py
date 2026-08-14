"""One-off diagnostics and experiments. NOT production code.

Nothing in this package is imported by `dtest.engine`, `dtest.evaluate`, or any
research driver script that logs to `evaluate.hypothesis_log`. Code here exists
to answer a specific question once (e.g. "how much of a result changes under a
different execution assumption?") and is kept deliberately separate so it can
never be mistaken for, or accidentally substituted into, the shipped harness.

If a diagnostic here ever needs to become a real, reusable rule, it gets
promoted into `dtest.engine`/`dtest.config` properly - with the same tests and
guards as everything else, not left living here.
"""
