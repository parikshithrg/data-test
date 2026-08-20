"""Quality / profitability-trend signal, hand-computable."""

from __future__ import annotations

import pandas as pd

from dtest.signals.quality import quality_signal


def test_fires_only_on_the_filing_date_when_it_is_a_trading_day():
    calendar = pd.bdate_range("2020-01-06", periods=10)
    trend = {"SYM": pd.Series([2.0], index=pd.to_datetime(["2020-01-08"]))}
    signal = quality_signal(trend, calendar, threshold=1.0)

    assert signal.loc[pd.Timestamp("2020-01-08"), "SYM"]
    assert signal["SYM"].sum() == 1


def test_below_threshold_never_fires():
    calendar = pd.bdate_range("2020-01-06", periods=10)
    trend = {"SYM": pd.Series([0.2], index=pd.to_datetime(["2020-01-08"]))}
    signal = quality_signal(trend, calendar, threshold=1.0)
    assert not signal["SYM"].any()


def test_rolls_forward_to_next_trading_day_when_filing_lands_on_a_weekend():
    calendar = pd.bdate_range("2020-01-06", periods=10)
    trend = {"SYM": pd.Series([3.0], index=pd.to_datetime(["2020-01-11"]))}   # a Saturday
    signal = quality_signal(trend, calendar, threshold=1.0)
    assert signal.loc[pd.Timestamp("2020-01-13"), "SYM"]
    assert signal["SYM"].sum() == 1


def test_multiple_symbols_independent():
    calendar = pd.bdate_range("2020-01-06", periods=10)
    trend = {
        "A": pd.Series([2.0], index=pd.to_datetime(["2020-01-07"])),
        "B": pd.Series([0.1], index=pd.to_datetime(["2020-01-07"])),
    }
    signal = quality_signal(trend, calendar, threshold=1.0)
    assert signal.loc[pd.Timestamp("2020-01-07"), "A"]
    assert not signal.loc[pd.Timestamp("2020-01-07"), "B"]
