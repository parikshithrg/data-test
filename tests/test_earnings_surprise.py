"""Earnings-surprise (PEAD) signal, hand-computable."""

from __future__ import annotations

import pandas as pd

from dtest.signals.earnings_surprise import earnings_surprise_signal


def test_fires_only_on_the_filing_date_when_it_is_a_trading_day():
    calendar = pd.bdate_range("2020-01-06", periods=10)   # Mon 2020-01-06 .. Fri 2020-01-17
    sue = {"SYM": pd.Series([2.0], index=pd.to_datetime(["2020-01-08"]))}   # a Wednesday
    signal = earnings_surprise_signal(sue, calendar, threshold=1.0)

    assert signal.loc[pd.Timestamp("2020-01-08"), "SYM"]
    assert signal["SYM"].sum() == 1   # exactly one firing, not a plateau


def test_below_threshold_never_fires():
    calendar = pd.bdate_range("2020-01-06", periods=10)
    sue = {"SYM": pd.Series([0.5], index=pd.to_datetime(["2020-01-08"]))}
    signal = earnings_surprise_signal(sue, calendar, threshold=1.0)
    assert not signal["SYM"].any()


def test_rolls_forward_to_next_trading_day_when_filing_lands_on_a_weekend():
    calendar = pd.bdate_range("2020-01-06", periods=10)   # Mon..Fri, Mon..Fri
    # 2020-01-11 is a Saturday - not in a business-day calendar.
    sue = {"SYM": pd.Series([3.0], index=pd.to_datetime(["2020-01-11"]))}
    signal = earnings_surprise_signal(sue, calendar, threshold=1.0)

    assert signal.loc[pd.Timestamp("2020-01-13"), "SYM"]   # next trading day (Monday)
    assert not signal.loc[pd.Timestamp("2020-01-10"), "SYM"]
    assert signal["SYM"].sum() == 1


def test_intraday_filing_timestamp_does_not_roll_to_the_next_day():
    # A post-market-close disclosure timestamp on a real trading day must
    # still map to THAT SAME day, not spuriously roll forward because the
    # time component sorts after midnight.
    calendar = pd.bdate_range("2020-01-06", periods=10)
    sue = {"SYM": pd.Series([2.0], index=pd.to_datetime(["2020-01-08 20:15:00"]))}
    signal = earnings_surprise_signal(sue, calendar, threshold=1.0)
    assert signal.loc[pd.Timestamp("2020-01-08"), "SYM"]
    assert not signal.loc[pd.Timestamp("2020-01-09"), "SYM"]


def test_multiple_symbols_independent():
    calendar = pd.bdate_range("2020-01-06", periods=10)
    sue = {
        "A": pd.Series([2.0], index=pd.to_datetime(["2020-01-07"])),
        "B": pd.Series([0.1], index=pd.to_datetime(["2020-01-07"])),
    }
    signal = earnings_surprise_signal(sue, calendar, threshold=1.0)
    assert signal.loc[pd.Timestamp("2020-01-07"), "A"]
    assert not signal.loc[pd.Timestamp("2020-01-07"), "B"]


def test_filing_after_calendar_end_is_dropped_not_raised():
    calendar = pd.bdate_range("2020-01-06", periods=5)
    sue = {"SYM": pd.Series([2.0], index=pd.to_datetime(["2020-06-01"]))}
    signal = earnings_surprise_signal(sue, calendar, threshold=1.0)
    assert not signal["SYM"].any()
