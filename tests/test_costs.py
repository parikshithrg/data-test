"""Cost model tests. These pin the errors that were actually made before.

Two of them exist because the predecessor project got them wrong in production:
the turnover-vs-position units trap, and charging stamp duty on both legs.
"""

from __future__ import annotations

import pytest

from dtest import load_config
from dtest.engine.costs import CostModel


@pytest.fixture(scope="module")
def cm() -> CostModel:
    return CostModel.from_config(load_config())


def test_statutory_round_trip_matches_independent_derivation(cm):
    """0.222% of position, statutory only.

    This figure was derived independently in the predecessor project from the
    same public schedule. Two separate implementations agreeing is the reason to
    trust the number; if this test breaks, a rate changed and every result in the
    project needs re-pricing.
    """
    assert cm.round_trip_pct(include_slippage=False) == pytest.approx(0.22224, abs=1e-5)


def test_full_round_trip_with_default_slippage(cm):
    """0.322% of position at 5 bps/side - the shipped assumption."""
    assert cm.round_trip_pct(include_slippage=True) == pytest.approx(0.32224, abs=1e-5)


def test_position_quoted_is_double_turnover_quoted(cm):
    """THE UNITS TRAP.

    A rate quoted against turnover is half the same rate quoted against
    position, because turnover is buy + sell. Mixing them makes costs look
    survivable when they are not. Pinned so nobody can 'simplify' one into the
    other.
    """
    rt = cm.round_trip(100_000.0, 100_000.0)
    assert rt.pct_of_position == pytest.approx(2 * rt.pct_of_turnover, rel=1e-12)


def test_stamp_duty_is_buy_side_only(cm):
    """Stamp duty is a buy-side levy. Charging both legs is quietly expensive."""
    buy = cm.leg(100_000.0, "buy")
    sell = cm.leg(100_000.0, "sell")
    assert buy.stamp > 0
    assert sell.stamp == 0.0


def test_stt_is_charged_on_both_legs_for_delivery(cm):
    """Delivery pays STT twice. An intraday schedule pays it once, sell-side.

    Anything held overnight is delivery, including a one-session hold - assuming
    otherwise silently halves the largest single component of the cost.
    """
    buy = cm.leg(100_000.0, "buy")
    sell = cm.leg(100_000.0, "sell")
    assert buy.stt == pytest.approx(100.0)      # 0.1% of 1,00,000
    assert sell.stt == pytest.approx(100.0)


def test_gst_excludes_stt_and_stamp(cm):
    """GST applies to service charges only, never to STT or stamp duty."""
    leg = cm.leg(100_000.0, "buy")
    expected = (leg.brokerage + leg.exchange + leg.sebi) * 0.18
    assert leg.gst == pytest.approx(expected, rel=1e-12)


def test_costs_scale_linearly_with_value(cm):
    """A fixed-percentage schedule must be scale free (brokerage is zero here).

    This is what makes post-hoc cost application exact rather than approximate:
    the charge is the same fraction of every trade, so it cannot change which
    names get picked or where a stop sits.
    """
    small = cm.round_trip(10_000.0, 10_000.0)
    large = cm.round_trip(1_000_000.0, 1_000_000.0)
    assert small.pct_of_position == pytest.approx(large.pct_of_position, rel=1e-12)


def test_winner_pays_more_on_the_sell_leg(cm):
    """Real per-trade costing uses actual leg values, not a flat approximation."""
    flat = cm.round_trip(100_000.0, 100_000.0)
    winner = cm.round_trip(100_000.0, 120_000.0)
    assert winner.total > flat.total
    # ...and the headline flat figure understates a winner's rupee cost, which
    # is the conservative direction only for the percentage, not the rupees.
    assert winner.pct_of_position > flat.pct_of_position


def test_slippage_is_separable(cm):
    """Slippage is the one assumption; it must be isolable for a sweep."""
    rt = cm.round_trip(100_000.0, 100_000.0)
    assert rt.total == pytest.approx(rt.statutory + rt.slippage, rel=1e-12)
    assert rt.slippage > 0


def test_rejects_bad_side(cm):
    with pytest.raises(ValueError):
        cm.leg(1000.0, "short")


def test_rejects_negative_value(cm):
    with pytest.raises(ValueError):
        cm.leg(-1.0, "buy")
