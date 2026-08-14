"""Placebo tests. The decisive property: a placebo must be NEUTRAL when
selection carries no information, and BEATABLE when it genuinely does -
otherwise the noise floor itself cannot be trusted."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dtest import load_config
from dtest.engine.simulate import ExitRule, simulate_trades, trades_to_frame
from dtest.evaluate.metrics import summary_stats
from dtest.evaluate.placebo import run_placebos


@pytest.fixture
def cfg():
    c = load_config()
    return replace(c, execution=replace(c.execution, max_participation_pct=100.0))


def _market(n_days: int, n_symbols: int, seed: int = 0, drift_symbols: set[str] | None = None):
    """A synthetic market: n_symbols random-walking names, optionally a subset
    with genuine upward drift baked into their fill-day OPEN prices."""
    idx = pd.bdate_range("2020-01-06", periods=n_days)
    symbols = [f"S{i}" for i in range(n_symbols)]
    rng = np.random.default_rng(seed)

    close = pd.DataFrame(index=idx, columns=symbols, dtype=float)
    for s in symbols:
        drift = 0.003 if (drift_symbols and s in drift_symbols) else 0.0
        rets = rng.normal(drift, 0.01, size=n_days)
        close[s] = 100.0 * np.cumprod(1 + rets)

    open_ = close.shift(1).fillna(close.iloc[0])   # next open ~= prior close, simple
    high = close * 1.01
    low = close * 0.99
    volume = pd.DataFrame(1_000_000.0, index=idx, columns=symbols)
    return idx, symbols, open_, high, low, close, volume


def test_placebo_matches_real_signal_dates_and_counts(cfg):
    idx, symbols, o, h, l, c, v = _market(120, 20)
    rng = np.random.default_rng(1)
    real = pd.DataFrame(False, index=idx, columns=symbols)
    fire_dates = idx[10:100:15]
    for d in fire_dates:
        picks = rng.choice(symbols, size=3, replace=False)
        real.loc[d, picks] = True

    eligible = pd.DataFrame(True, index=idx, columns=symbols)
    rule = ExitRule(max_hold_days=5)
    result = run_placebos(real, eligible, "long", rule, open_=o, high=h, low=l,
                          close=c, volume=v, atr_panel=None,
                          target_value_per_trade=5_000, cfg=cfg, n_seeds=5)

    placebo_sig = result.per_seed  # doesn't directly expose signals, so re-derive
    # Re-run the internal signal builder directly to check date/count matching.
    from dtest.evaluate.placebo import _placebo_signals
    ps = _placebo_signals(real, eligible, c, seed=42)
    assert (ps.sum(axis=1) == real.sum(axis=1)).all(), (
        "placebo must fire exactly as many names as the real signal, on the same dates"
    )
    assert (ps.index[ps.any(axis=1)] == real.index[real.any(axis=1)]).all()


def test_placebo_is_neutral_when_no_symbol_has_real_edge(cfg):
    """With no genuine drift anywhere, the real hypothesis (an arbitrary,
    information-free selection rule) should look statistically like a placebo -
    its mean should sit inside a wide band around the placebo mean, not
    systematically beat every seed."""
    idx, symbols, o, h, l, c, v = _market(250, 30, seed=7)
    rng = np.random.default_rng(2)
    real = pd.DataFrame(False, index=idx, columns=symbols)
    for d in idx[20:230:10]:
        picks = rng.choice(symbols, size=4, replace=False)
        real.loc[d, picks] = True

    eligible = pd.DataFrame(True, index=idx, columns=symbols)
    rule = ExitRule(max_hold_days=5)
    real_trades = trades_to_frame(simulate_trades(
        real, "long", rule, open_=o, high=h, low=l, close=c, volume=v,
        atr_panel=None, target_value_per_trade=5_000, cfg=cfg,
    ))
    real_stats = summary_stats(real_trades)

    result = run_placebos(real, eligible, "long", rule, open_=o, high=h, low=l,
                          close=c, volume=v, atr_panel=None,
                          target_value_per_trade=5_000, cfg=cfg, n_seeds=20)
    cmp = result.compare(real_stats, "mean_net_pct")
    # Not a hard assertion on beating/losing (both sides are possible by chance
    # with no real edge) - the property under test is that the real result does
    # NOT sit at an extreme percentile every time. A generous band avoids flakes.
    assert 5.0 <= cmp["percentile_vs_placebos"] <= 95.0, (
        f"an information-free selection landed at the {cmp['percentile_vs_placebos']}th "
        "percentile of its own placebo distribution - the comparison itself is suspect"
    )


def test_real_edge_beats_the_placebo_band(cfg):
    """The other half of the same property: when a real edge exists (a subset
    of symbols with genuine drift, and the hypothesis selects ONLY those), the
    result must clearly clear the placebo band - otherwise the placebo
    machinery has no power to detect a real signal at all."""
    n_symbols = 40
    drift_symbols = {f"S{i}" for i in range(8)}   # matches _market's naming scheme
    idx, symbols, o, h, l, c, v = _market(300, n_symbols, seed=3,
                                          drift_symbols=drift_symbols)
    real = pd.DataFrame(False, index=idx, columns=symbols)
    # The hypothesis "knows" to only ever pick the drifting names.
    for d in idx[20:280:8]:
        real.loc[d, symbols[:4]] = True

    eligible = pd.DataFrame(True, index=idx, columns=symbols)
    rule = ExitRule(max_hold_days=5)
    real_trades = trades_to_frame(simulate_trades(
        real, "long", rule, open_=o, high=h, low=l, close=c, volume=v,
        atr_panel=None, target_value_per_trade=5_000, cfg=cfg,
    ))
    real_stats = summary_stats(real_trades)

    result = run_placebos(real, eligible, "long", rule, open_=o, high=h, low=l,
                          close=c, volume=v, atr_panel=None,
                          target_value_per_trade=5_000, cfg=cfg, n_seeds=20)
    cmp = result.compare(real_stats, "mean_net_pct")
    assert cmp["beats_best_placebo"], (
        f"real mean {cmp['real']:.3f}% did not clear the best of 20 placebos "
        f"({cmp['placebo_max']:.3f}%) despite a genuine, constructed drift edge - "
        "the placebo machinery has no detection power"
    )


def test_placebo_result_seeds_are_reproducible(cfg):
    idx, symbols, o, h, l, c, v = _market(100, 15, seed=5)
    real = pd.DataFrame(False, index=idx, columns=symbols)
    real.loc[idx[30], symbols[:3]] = True
    real.loc[idx[60], symbols[3:6]] = True
    eligible = pd.DataFrame(True, index=idx, columns=symbols)
    rule = ExitRule(max_hold_days=4)

    r1 = run_placebos(real, eligible, "long", rule, open_=o, high=h, low=l,
                      close=c, volume=v, atr_panel=None,
                      target_value_per_trade=5_000, cfg=cfg, n_seeds=8)
    r2 = run_placebos(real, eligible, "long", rule, open_=o, high=h, low=l,
                      close=c, volume=v, atr_panel=None,
                      target_value_per_trade=5_000, cfg=cfg, n_seeds=8)
    pd.testing.assert_frame_equal(r1.per_seed, r2.per_seed)


def test_placebo_respects_eligible_pool_not_the_full_symbol_list(cfg):
    """A placebo must only draw from that date's point-in-time universe - not
    from every symbol that ever existed in the panel."""
    idx, symbols, o, h, l, c, v = _market(80, 10, seed=9)
    real = pd.DataFrame(False, index=idx, columns=symbols)
    real.loc[idx[20], symbols[:2]] = True

    # Only the first 3 symbols are ever "eligible".
    eligible = pd.DataFrame(False, index=idx, columns=symbols)
    eligible[symbols[:3]] = True

    from dtest.evaluate.placebo import _placebo_signals
    ps = _placebo_signals(real, eligible, c, seed=42)
    picked = ps.columns[ps.loc[idx[20]]]
    assert set(picked) <= set(symbols[:3])


def test_placebo_caps_k_when_pool_smaller_than_requested_count(cfg):
    """If the real signal fired more names than the eligible pool has, the
    placebo must degrade gracefully (fewer picks) rather than crash."""
    idx, symbols, o, h, l, c, v = _market(50, 10, seed=11)
    real = pd.DataFrame(False, index=idx, columns=symbols)
    real.loc[idx[10], symbols[:5]] = True   # 5 signals

    eligible = pd.DataFrame(False, index=idx, columns=symbols)
    eligible[symbols[:2]] = True            # but only 2 are ever eligible

    from dtest.evaluate.placebo import _placebo_signals
    ps = _placebo_signals(real, eligible, c, seed=42)
    assert ps.loc[idx[10]].sum() == 2
