"""Corporate-action tests on hand-built fixtures.

Every case here is one that real data produced. The two guard tests in
particular encode measurements: a 1:2 split must survive detection, and a
settlement quirk that moves most of the market's prev_close must not be mistaken
for hundreds of simultaneous corporate actions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtest.data.corporate_actions import (
    adjusted_close, daily_returns, detect_actions, previous_traded_close,
)


def _panel(values: dict[str, list[float]], n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(values, index=idx)


def _with_quiet_market(values: dict[str, list[float]], n: int, n_filler: int = 30):
    """A fixture with a realistic cross-section around the symbols under test.

    Necessary, not decorative: the date-level guards are SHARE-based, so in a
    two-symbol fixture a single corporate action is 50% of the market and is
    correctly suppressed as a mass event. The first version of these tests failed
    for exactly that reason - the guard was right and the fixture was wrong.
    The filler names are flat and consistent, so they never flag.
    """
    close = dict(values)
    prev = {}
    for k, v in values.items():
        prev[k] = None
    for i in range(n_filler):
        close[f"F{i}"] = [10.0] * n
    close_df = _panel(close, n)
    prev_df = close_df.shift(1)
    prev_df.iloc[0] = close_df.iloc[0]
    return close_df, prev_df


def test_no_actions_on_a_clean_series():
    close = _panel({"A": [100.0, 101.0, 102.0, 103.0]}, 4)
    prev = _panel({"A": [99.0, 100.0, 101.0, 102.0]}, 4)
    rep = detect_actions(close, prev)
    assert rep.n_actions == 0

    r = daily_returns(close, prev, rep)
    # Ordinary days reduce to a plain close-to-close return.
    assert r["A"].iloc[1] == pytest.approx(0.01)
    assert r["A"].iloc[2] == pytest.approx(102 / 101 - 1)


def _split_fixture():
    """Day 2 is a 1:2 split: prior close 100 restated to 50, then closes 51."""
    close, prev = _with_quiet_market({"A": [98.0, 100.0, 51.0, 52.0]}, 4)
    prev.loc[prev.index[2], "A"] = 50.0
    prev.loc[prev.index[3], "A"] = 51.0
    return close, prev


def test_detects_a_one_for_two_split_and_neutralises_the_fake_crash():
    """The headline case: price halves, but nobody lost half their money."""
    close, prev = _split_fixture()

    rep = detect_actions(close, prev)
    assert rep.n_actions == 1
    action = rep.actions.iloc[0]
    assert action["symbol"] == "A"
    assert action["factor"] == pytest.approx(0.5)

    r = daily_returns(close, prev, rep)
    raw = close / previous_traded_close(close) - 1.0
    assert raw["A"].iloc[2] == pytest.approx(-0.49)     # what the raw series says
    assert r["A"].iloc[2] == pytest.approx(0.02)        # what actually happened


def test_mass_prev_close_disagreement_suppresses_the_whole_date():
    """The 2004-04-19 case, reproduced in miniature.

    On that date 698 of 730 symbols had a prev_close inconsistent with the
    previous traded close - most by only a few percent. A specific-only detector
    saw ~100 of them and emitted ~100 phantom actions. The sensitive date-level
    guard rejects the date outright.
    """
    n = 3
    syms = [f"S{i}" for i in range(20)]
    close = pd.DataFrame(100.0, index=pd.bdate_range("2020-01-01", periods=n), columns=syms)
    prev = close.shift(1).fillna(100.0)
    # Day 2: shift most of the market's prev_close by a few percent, and push
    # three of them past the 10% "specific" threshold.
    prev.iloc[2, :] = 104.0
    prev.iloc[2, :3] = 115.0

    rep = detect_actions(close, prev)
    assert rep.n_actions == 0, "a market-wide prev_close shift is not 20 corporate actions"
    assert len(rep.rejected_dates) == 1
    assert rep.rejected_dates.iloc[0]["reason"] == "prev_close untrusted"


def test_small_restatement_is_not_an_action():
    """Publication rounding and minor quirks must not become adjustments."""
    close = _panel({"A": [100.0, 101.0, 102.0]}, 3)
    prev = _panel({"A": [99.0, 100.4, 101.0]}, 3)   # 0.4% off on day 2
    assert detect_actions(close, prev).n_actions == 0


def test_uses_previous_traded_close_not_previous_row():
    """A symbol that did not trade yesterday compares against its last real close.

    A plain `shift(1)` would give NaN on the day AFTER a non-trading day and
    silently drop that observation - which is how a thin name quietly leaves a
    dataset. Here index 1 is untraded, so index 2 must still reference index 0.
    """
    close = _panel({"A": [100.0, np.nan, 102.0]}, 3)
    prior = previous_traded_close(close)
    assert np.isnan(prior["A"].iloc[0])                 # nothing before the start
    assert prior["A"].iloc[1] == pytest.approx(100.0)
    assert prior["A"].iloc[2] == pytest.approx(100.0)   # skips the untraded bar

    plain_shift = close["A"].shift(1)
    assert np.isnan(plain_shift.iloc[2])                # what we are avoiding


def test_adjusted_close_is_continuous_and_anchored_to_the_last_real_price():
    """Back-adjustment must not move today's tradeable price."""
    close, prev = _split_fixture()
    adj = adjusted_close(close, prev)

    # The most recent price is the real one.
    assert adj["A"].iloc[-1] == pytest.approx(52.0)
    # And the split no longer looks like a 49% crash.
    adj_ret = adj["A"].pct_change()
    assert adj_ret.iloc[2] == pytest.approx(0.02, abs=1e-9)
    # Pre-split prices are restated onto the post-split basis (~half).
    assert adj["A"].iloc[1] == pytest.approx(50.0, rel=1e-9)


def test_adjusted_close_survives_rounding_that_broke_the_predecessor():
    """BAJFINANCE printed `1.00 -> 0.50` six times from 2-decimal rounding.

    Because the adjusted series compounds RETURNS in float64 rather than storing
    rounded prices, a long back-adjustment does not collapse into ticks.
    """
    n = 60
    idx = pd.bdate_range("2000-01-01", periods=n)
    # A steady 1%/day riser ending near Rs 1000, i.e. a ~550x total move.
    closes = 1000.0 / (1.01 ** np.arange(n - 1, -1, -1))
    close = pd.DataFrame({"A": closes}, index=idx)
    prev = close.shift(1)
    prev.iloc[0] = closes[0] / 1.01

    adj = adjusted_close(close, prev)
    r = daily_returns(close, prev)
    # Every daily return stays a clean 1% - no quantisation.
    assert np.allclose(r["A"].iloc[1:].to_numpy(), 0.01, atol=1e-9)
    assert adj["A"].iloc[-1] == pytest.approx(1000.0)
    assert adj["A"].iloc[0] > 0.0


def test_factors_frame_is_one_everywhere_except_actions():
    from dtest.data.corporate_actions import adjustment_factors

    close, prev = _split_fixture()
    f = adjustment_factors(close, prev)
    assert (f["F0"] == 1.0).all()
    assert f["A"].iloc[2] == pytest.approx(0.5)
    assert f["A"].iloc[1] == 1.0
