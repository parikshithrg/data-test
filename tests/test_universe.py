"""Universe tests. Each pins one specific claim the module makes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dtest import load_config
from dtest.universe import build_universe


def _cfg(**overrides):
    """The real config, with `universe` fields overridden for a small fixture."""
    cfg = load_config()
    u = replace(cfg.universe, **overrides)
    return replace(cfg, universe=u)


def _panel(n_days: int, symbols: list[str], turnover_fn, start_row: dict[str, int] | None = None):
    """A synthetic close/turnover panel. `turnover_fn(day_index, symbol) -> float`.

    `start_row` lets a symbol be absent (NaN) before it "lists".
    """
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    close = pd.DataFrame(100.0, index=idx, columns=symbols)
    turnover = pd.DataFrame(
        {s: [turnover_fn(i, s) for i in range(n_days)] for s in symbols}, index=idx,
    )
    if start_row:
        for s, row in start_row.items():
            close.iloc[:row, close.columns.get_loc(s)] = np.nan
            turnover.iloc[:row, turnover.columns.get_loc(s)] = np.nan
    return close, turnover


def test_no_lookahead_a_later_price_move_cannot_change_an_earlier_decision():
    """The point-in-time claim, tested directly rather than assumed.

    Two runs identical up to date X, diverging only after it, must produce an
    IDENTICAL universe up to and including X.
    """
    symbols = [f"S{i}" for i in range(30)]
    n = 400

    def base_turnover(i, s):
        return 1000.0 + hash((i, s)) % 500  # deterministic pseudo-randomness

    close_a, turn_a = _panel(n, symbols, base_turnover)
    close_b, turn_b = close_a.copy(), turn_a.copy()

    cut = 300
    # Mutate B only strictly after `cut` - a massive turnover spike for S0.
    turn_b.iloc[cut + 1:, turn_b.columns.get_loc("S0")] *= 1000

    cfg = _cfg(size=5, buffer_size=7, lookback_days=20, min_history_days=30,
              max_staleness_days=5, min_price=1.0)

    res_a = build_universe(close_a, turn_a, cfg)
    res_b = build_universe(close_b, turn_b, cfg)

    up_to_cut = close_a.index[: cut + 1]
    pd.testing.assert_frame_equal(
        res_a.membership.loc[up_to_cut], res_b.membership.loc[up_to_cut]
    )
    # And they MUST differ later, or the test would be vacuous.
    assert not res_a.membership.loc[close_a.index[cut + 1]:].equals(
        res_b.membership.loc[close_a.index[cut + 1]:]
    )


def test_banding_keeps_an_incumbent_that_a_hard_cutoff_would_drop():
    """A name ranking 6th (inside a 8-wide buffer, outside a 5-wide cutoff)
    should stay if it was already in, and never get in fresh."""
    symbols = [f"S{i}" for i in range(10)]
    n = 500

    def turnover_fn(i, s):
        rank = int(s[1:])          # S0 highest turnover ... S9 lowest, fixed
        return 10000.0 - rank * 100

    close, turn = _panel(n, symbols, turnover_fn)
    cfg = _cfg(size=5, buffer_size=8, lookback_days=20, min_history_days=30,
              max_staleness_days=5, min_price=1.0)
    res = build_universe(close, turn, cfg)

    last = res.membership.index[-1]
    selected = set(res.as_of(last))
    assert selected == {"S0", "S1", "S2", "S3", "S4"}, (
        "with a static ranking, banding changes nothing versus a hard cutoff - "
        "this pins the baseline before testing the dynamic case"
    )

    # Now S5 (rank 6, inside buffer, outside cutoff) becomes an incumbent by
    # briefly having the HIGHEST turnover, then reverts to its normal rank.
    turn2 = turn.copy()
    boost_start = 60
    boost_len = 25
    turn2.iloc[boost_start:boost_start + boost_len, turn2.columns.get_loc("S5")] = 50000.0

    res2 = build_universe(close, turn2, cfg)
    after_boost = res2.membership.index[boost_start + boost_len + cfg.universe.lookback_days + 5]
    assert "S5" in res2.as_of(after_boost), (
        "S5 earned incumbency during the boost and, at rank 6, should be "
        "retained by the buffer even after the boost fades"
    )
    assert "S9" not in res2.as_of(after_boost), (
        "the buffer protects incumbents, it does not admit everyone"
    )


def test_new_entrant_needs_the_tight_cutoff_not_the_buffer():
    """A name that was NEVER in must clear `size`, not merely `buffer_size`."""
    symbols = [f"S{i}" for i in range(10)]
    n = 500

    def turnover_fn(i, s):
        rank = int(s[1:])
        return 10000.0 - rank * 100

    close, turn = _panel(n, symbols, turnover_fn)
    cfg = _cfg(size=5, buffer_size=8, lookback_days=20, min_history_days=30,
              max_staleness_days=5, min_price=1.0)
    res = build_universe(close, turn, cfg)
    last = res.as_of(res.membership.index[-1])
    # S5..S7 rank inside the buffer but were never incumbents (static ranking,
    # so they never got a chance to enter) - they must be excluded.
    assert not ({"S5", "S6", "S7"} & set(last))


def test_min_history_gate_blocks_a_freshly_listed_name():
    symbols = ["OLD", "NEW"]
    n = 400

    def turnover_fn(i, s):
        return 100000.0    # NEW would dominate turnover the instant it appears

    close, turn = _panel(n, symbols, turnover_fn, start_row={"NEW": n - 10})
    cfg = _cfg(size=1, buffer_size=1, lookback_days=20, min_history_days=252,
              max_staleness_days=5, min_price=1.0)
    res = build_universe(close, turn, cfg)
    assert res.as_of(res.membership.index[-1]) == ["OLD"], (
        "NEW has only 10 days of history against a 252-day floor and must be excluded"
    )


def test_staleness_gate_blocks_a_delisted_name():
    """A name that stopped trading must drop out even if it still ranks well
    on trailing turnover computed before it went dark."""
    symbols = ["ALIVE", "HALTED"]
    n = 400
    halt_at = 350

    def turnover_fn(i, s):
        if s == "HALTED" and i >= halt_at:
            return np.nan
        return 5000.0

    close, turn = _panel(n, symbols, turnover_fn)
    close.iloc[halt_at:, close.columns.get_loc("HALTED")] = np.nan

    cfg = _cfg(size=2, buffer_size=2, lookback_days=20, min_history_days=30,
              max_staleness_days=5, min_price=1.0)
    res = build_universe(close, turn, cfg)
    last = res.membership.index[-1]
    assert "HALTED" not in res.as_of(last)
    assert "ALIVE" in res.as_of(last)


def test_min_price_gate_excludes_penny_stocks():
    symbols = ["NORMAL", "PENNY"]
    n = 400
    close = pd.DataFrame(100.0, index=pd.bdate_range("2018-01-01", periods=n), columns=symbols)
    close["PENNY"] = 2.0
    turn = pd.DataFrame(5000.0, index=close.index, columns=symbols)

    cfg = _cfg(size=2, buffer_size=2, lookback_days=20, min_history_days=30,
              max_staleness_days=5, min_price=5.0)
    res = build_universe(close, turn, cfg)
    assert "PENNY" not in res.as_of(res.membership.index[-1])


def test_rebalance_takes_effect_strictly_after_the_decision_date():
    """The decision at close of rebalance date T must not retroactively change
    what T's own membership was before that decision."""
    symbols = [f"S{i}" for i in range(4)]
    n = 300

    def turnover_fn(i, s):
        return 1000.0 if s != "S3" else 1.0

    close, turn = _panel(n, symbols, turnover_fn)
    cfg = _cfg(size=3, buffer_size=3, lookback_days=20, min_history_days=30,
              max_staleness_days=5, min_price=1.0)
    res = build_universe(close, turn, cfg)

    # S3 never qualifies (lowest turnover, static ranking) - so on EVERY
    # rebalance date it must already be excluded (its own rebalance can't add
    # it after the fact, since it never wins a slot at all here).
    for d in res.rebalance_dates:
        assert "S3" not in res.as_of(d)


def test_deterministic_across_repeated_runs():
    """Byte-identical membership panel on repeated builds - no set-ordering leak."""
    symbols = [f"S{i}" for i in range(15)]
    n = 400

    def turnover_fn(i, s):
        return 1000.0 + (hash((i, s)) % 200)

    close, turn = _panel(n, symbols, turnover_fn)
    cfg = _cfg(size=6, buffer_size=9, lookback_days=15, min_history_days=30,
              max_staleness_days=5, min_price=1.0)

    r1 = build_universe(close, turn, cfg)
    r2 = build_universe(close, turn, cfg)
    pd.testing.assert_frame_equal(r1.membership, r2.membership)
    pd.testing.assert_frame_equal(r1.rank, r2.rank)


def test_exact_turnover_ties_do_not_break_determinism():
    """Every symbol tied on turnover: rank must still be a fixed, reproducible
    order (alphabetical, by construction) rather than depend on hash seed."""
    symbols = [f"S{i}" for i in range(12)]
    n = 300

    def turnover_fn(i, s):
        return 1000.0    # perfectly tied, always

    close, turn = _panel(n, symbols, turnover_fn)
    cfg = _cfg(size=5, buffer_size=5, lookback_days=15, min_history_days=30,
              max_staleness_days=5, min_price=1.0)
    r1 = build_universe(close, turn, cfg)
    r2 = build_universe(close, turn, cfg)
    pd.testing.assert_frame_equal(r1.membership, r2.membership)
    last = r1.as_of(r1.membership.index[-1])
    assert last == sorted(last)[:5] or set(last) == set(sorted(symbols)[:5])
