"""Pair trade event walk, fixtures numerically verified against the real
z-score output before being locked in here, not hand-guessed - same
discipline every other signal fixture in this project used."""

from __future__ import annotations

import pandas as pd

from dtest.signals.pairs_reversion import pair_trade_events

NOISE = [0.0, 0.3, -0.2, 0.25, -0.15, 0.2, -0.25, 0.15, -0.2, 0.1]


def _reverting_fixture():
    idx = pd.bdate_range("2020-01-06", periods=22)
    a = [100.0 + n for n in NOISE] + [112.0, 111.0] + [105.0, 101.0, 100.2, 100.0] + [100.0] * 6
    b = [100.0] * 22
    return idx, pd.Series(a[:22], index=idx), pd.Series(b, index=idx)


def _non_reverting_fixture():
    idx = pd.bdate_range("2020-01-06", periods=25)
    a = [100.0 + n for n in NOISE] + [100.0 + 3 * i for i in range(1, 16)]
    b = [100.0] * 25
    return idx, pd.Series(a[:25], index=idx), pd.Series(b, index=idx)


def test_pair_trade_fires_and_resolves_on_reversion():
    idx, a, b = _reverting_fixture()
    trades = pair_trade_events(a, b, idx[0], idx[-1], "A", "B",
                               zscore_window=5, z_entry=1.5, z_exit=0.5, max_hold_days=5)
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_date == pd.Timestamp("2020-01-20")
    assert t.exit_date == pd.Timestamp("2020-01-22")
    assert t.exit_reason == "reverted"
    assert t.held_days == 2


def test_pair_trade_direction_is_long_the_cheap_leg():
    """A rose sharply relative to B -> A is rich -> long B, short A."""
    idx, a, b = _reverting_fixture()
    t = pair_trade_events(a, b, idx[0], idx[-1], "A", "B",
                          zscore_window=5, z_entry=1.5, z_exit=0.5, max_hold_days=5)[0]
    assert t.long_symbol == "B"
    assert t.short_symbol == "A"


def test_pair_trade_exits_on_time_when_never_reverting():
    idx, a, b = _non_reverting_fixture()
    trades = pair_trade_events(a, b, idx[0], idx[-1], "A", "B",
                               zscore_window=5, z_entry=1.5, z_exit=0.5, max_hold_days=5)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "time"
    assert t.held_days == 5


def test_pair_trade_exits_on_window_end_not_mislabelled_as_time():
    """Same reverting fixture, but the pair's own validity window ends
    right at the entry date - must be distinguished from a max_hold_days
    time exit, not silently folded into the same label."""
    idx, a, b = _reverting_fixture()
    trades = pair_trade_events(a, b, idx[0], idx[10], "A", "B",
                               zscore_window=5, z_entry=1.5, z_exit=0.5, max_hold_days=5)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "window_end"
    assert t.held_days == 0


def test_pair_trade_one_position_at_a_time():
    """No new entry may open while a trade for this pair is still live -
    same busy_until convention `engine/simulate.py` uses for single-leg
    trades."""
    idx, a, b = _reverting_fixture()
    trades = pair_trade_events(a, b, idx[0], idx[-1], "A", "B",
                               zscore_window=5, z_entry=1.5, z_exit=0.5, max_hold_days=5)
    for i in range(len(trades) - 1):
        assert trades[i + 1].entry_date > trades[i].exit_date


def test_pair_trade_does_not_fire_below_entry_threshold():
    idx, a, b = _reverting_fixture()
    trades = pair_trade_events(a, b, idx[0], idx[-1], "A", "B",
                               zscore_window=5, z_entry=10.0, z_exit=0.5, max_hold_days=5)
    assert trades == []


def test_pair_trade_deterministic():
    idx, a, b = _reverting_fixture()
    kwargs = dict(zscore_window=5, z_entry=1.5, z_exit=0.5, max_hold_days=5)
    t1 = pair_trade_events(a, b, idx[0], idx[-1], "A", "B", **kwargs)
    t2 = pair_trade_events(a, b, idx[0], idx[-1], "A", "B", **kwargs)
    assert len(t1) == 1 and t1 == t2   # non-trivial: both runs found the real trade
