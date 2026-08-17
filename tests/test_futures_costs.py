"""Futures cost model tests. Pin the structural differences from cash-equity
delivery costs.py - these are not the same schedule with different numbers,
the STRUCTURE differs (sell-side-only STT, real non-zero brokerage)."""

from __future__ import annotations

import pytest

from dtest import load_config
from dtest.engine.futures_costs import FuturesCostModel


@pytest.fixture(scope="module")
def fcm() -> FuturesCostModel:
    return FuturesCostModel.from_config(load_config())


def test_stt_is_sell_side_only_unlike_cash_equity_delivery(fcm):
    """THE structural difference this whole module exists to get right.
    Cash-equity delivery charges STT on BOTH legs; futures charges it on
    the sell leg only. A buy leg must pay ZERO STT."""
    buy = fcm.leg(100_000.0, "buy")
    sell = fcm.leg(100_000.0, "sell")
    assert buy.stt == 0.0
    assert sell.stt == pytest.approx(100_000.0 * 0.0125 / 100.0)


def test_brokerage_is_real_and_non_zero_unlike_delivery_equity(fcm):
    """Zerodha cash-equity delivery is free; F&O brokerage is not."""
    leg = fcm.leg(10_000.0, "buy")
    assert leg.brokerage > 0.0
    assert leg.brokerage == pytest.approx(10_000.0 * 0.03 / 100.0)


def test_brokerage_caps_at_the_configured_ceiling(fcm):
    """Rs 20 flat or 0.03%, whichever is LOWER - a large enough trade must
    hit the cap, not keep scaling proportionally."""
    small = fcm.leg(10_000.0, "buy")     # 0.03% = Rs 3, well under the cap
    large = fcm.leg(1_000_000.0, "buy")  # 0.03% = Rs 300, must cap at Rs 20
    assert small.brokerage == pytest.approx(3.0)
    assert large.brokerage == pytest.approx(20.0)


def test_stamp_duty_is_buy_side_only_at_futures_own_rate(fcm):
    buy = fcm.leg(100_000.0, "buy")
    sell = fcm.leg(100_000.0, "sell")
    assert buy.stamp == pytest.approx(100_000.0 * 0.002 / 100.0)
    assert sell.stamp == 0.0


def test_gst_excludes_stt_and_stamp(fcm):
    leg = fcm.leg(100_000.0, "sell")
    expected = (leg.brokerage + leg.exchange + leg.sebi) * 0.18
    assert leg.gst == pytest.approx(expected, rel=1e-12)


def test_long_round_trip_opens_with_buy_closes_with_sell(fcm):
    rt = fcm.round_trip(100_000.0, 105_000.0, direction="long")
    assert rt.open.stt == 0.0        # opening buy leg: no STT
    assert rt.close.stt > 0.0        # closing sell leg: STT charged
    assert rt.open.stamp > 0.0       # opening buy leg: stamp duty
    assert rt.close.stamp == 0.0


def test_short_round_trip_opens_with_sell_closes_with_buy(fcm):
    """The mirror of long - a short OPENS by selling, CLOSES by buying.
    Get this backwards and every short trade's costs are silently wrong."""
    rt = fcm.round_trip(100_000.0, 95_000.0, direction="short")
    assert rt.open.stt > 0.0         # opening sell leg: STT charged
    assert rt.close.stt == 0.0       # closing buy leg: no STT
    assert rt.open.stamp == 0.0
    assert rt.close.stamp > 0.0      # closing buy leg: stamp duty


def test_slippage_is_separable(fcm):
    rt = fcm.round_trip(100_000.0, 100_000.0, direction="long")
    assert rt.total == pytest.approx(rt.statutory + rt.slippage, rel=1e-12)
    assert rt.slippage > 0


def test_rejects_bad_action(fcm):
    with pytest.raises(ValueError):
        fcm.leg(1000.0, "short")   # 'short' is a direction, not an action


def test_rejects_bad_direction(fcm):
    with pytest.raises(ValueError):
        fcm.round_trip(1000.0, 1000.0, direction="sideways")


def test_rejects_negative_value(fcm):
    with pytest.raises(ValueError):
        fcm.leg(-1.0, "buy")


def test_describe_reports_both_directions(fcm):
    s = fcm.describe()
    assert "long" in s and "short" in s
