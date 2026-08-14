"""Determinism tests - the project's foundational claim.

If these fail, no other result in the repository can be trusted, because the
same inputs stopped producing the same outputs.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from dtest import load_config, set_seeds
from dtest.determinism import (
    RunManifest, assert_reproducible, config_hash, frame_hash, set_seeds as _ss,
)


def _frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(50, 4)),
        columns=list("dcba"),
        index=pd.date_range("2020-01-01", periods=50, freq="D"),
    )


def test_frame_hash_is_stable_across_calls():
    df = _frame()
    assert frame_hash(df) == frame_hash(df.copy())


def test_frame_hash_ignores_column_order_but_not_content():
    """Same data in a different column order is the same data. Different values
    are not - including a difference far below display precision."""
    df = _frame()
    assert frame_hash(df) == frame_hash(df[sorted(df.columns)])

    tweaked = df.copy()
    tweaked.iloc[0, 0] += 1e-12
    assert frame_hash(df) != frame_hash(tweaked)


def test_frame_hash_detects_index_change():
    df = _frame()
    shifted = df.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    assert frame_hash(df) != frame_hash(shifted)


def test_frame_hash_detects_dtype_change():
    """float64 vs float32 is a real difference; the values are not identical."""
    df = _frame()
    assert frame_hash(df) != frame_hash(df.astype("float32"))


def test_config_hash_is_key_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_seeded_rng_reproduces():
    set_seeds(42)
    import random as _r
    a = [_r.random() for _ in range(5)], np.random.rand(5).tolist()
    set_seeds(42)
    b = [_r.random() for _ in range(5)], np.random.rand(5).tolist()
    assert a == b


def test_manifest_round_trip(tmp_path):
    m = RunManifest(run_id="t", as_of="2026-08-13", config={"x": 1})
    m.record_frame("f", _frame())
    m.finish()
    m.write(tmp_path)
    loaded = RunManifest.load(tmp_path)
    assert loaded["run_id"] == "t"
    assert loaded["outputs"]["f"] == frame_hash(_frame())
    assert loaded["config_sha"] == config_hash({"x": 1})


def test_assert_reproducible_flags_the_diverging_output():
    a = RunManifest(run_id="a", as_of="2026-08-13")
    b = RunManifest(run_id="b", as_of="2026-08-13")
    a.record_frame("prices", _frame(1))
    b.record_frame("prices", _frame(1))
    a.record_frame("universe", _frame(2))
    b.record_frame("universe", _frame(3))

    assert_reproducible(a, a)                      # identical: fine
    with pytest.raises(AssertionError, match="universe"):
        assert_reproducible(a, b)


# --------------------------------------------------------------------------
# Config invariants. These guard the corrections the project exists to make.
# --------------------------------------------------------------------------

def test_config_forbids_same_bar_fills():
    """Signal at bar T's close cannot fill at bar T's close.

    This is the single largest source of inflated results in the predecessor
    project. The loader raises rather than warns, so it cannot be optimised away.
    """
    cfg = load_config()
    assert cfg.execution.fill_at == "next_open"


def test_splits_do_not_overlap_and_respect_as_of():
    cfg = load_config()
    for split in cfg.splits.values():
        split.validate(cfg.as_of)                  # raises on overlap / overrun
        assert split.train_end < split.val_start
        assert split.val_end < split.test_start
        assert split.test_end <= cfg.as_of


def test_no_module_calls_today():
    """Nothing in the package may read the wall clock.

    A function that calls `date.today()` gives a different answer tomorrow for
    no reason a manifest can record, which makes reproduction impossible by
    construction.

    Checked by parsing the AST, not by substring search: the first version of
    this test matched the word `date.today()` inside `__init__.py`'s own
    docstring explaining the rule, i.e. it failed on its own documentation. A
    guard that cannot tell code from prose will be silenced rather than obeyed.
    """
    import ast
    from pathlib import Path

    import dtest

    root = Path(dtest.__file__).parent
    # Files allowed a wall-clock read because it stamps METADATA about when
    # something happened (a run, a logged hypothesis) rather than feeding a
    # computation. Keep this list short and each entry justified - it is the
    # one place this guard can be silenced, so it must not become a dumping
    # ground for "computation happened to need today's date".
    METADATA_ONLY = {
        "determinism.py",   # RunManifest.started_utc / finished_utc
        "hypothesis_log.py",  # HypothesisEntry.logged_utc
    }
    offenders: list[str] = []
    for py in sorted(root.rglob("*.py")):
        if py.name in METADATA_ONLY:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr in ("today", "now", "utcnow"):
                offenders.append(f"{py.relative_to(root)}:{node.lineno} "
                                 f"{ast.unparse(fn)}()")
    assert not offenders, "wall-clock reads found: " + "; ".join(offenders)
