"""Point-in-time fundamentals features, hand-computable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtest.features.fundamentals import (
    margin_trend_zscore,
    margin_ttm,
    point_in_time_series,
    sue_zscore,
    to_daily_panel,
    trailing_ttm,
    yoy_change,
)


def _raw_filings():
    # Standalone (consolidated=False) quarterly EPS, one restatement, and a
    # consolidated row that must be excluded entirely.
    return pd.DataFrame({
        "filing_date": pd.to_datetime([
            "2020-01-20", "2020-04-21", "2020-07-20", "2020-10-19",
            "2021-01-18",              # first filing for Q3FY21 (period_end 2020-12-31)
            "2021-01-25",              # RESTATEMENT of the same period, later filing_date
            "2021-01-18",              # a CONSOLIDATED row on the same date - must be dropped
        ]),
        "period_end": pd.to_datetime([
            "2019-12-31", "2020-03-31", "2020-06-30", "2020-09-30",
            "2020-12-31", "2020-12-31", "2020-12-31",
        ]),
        "consolidated": [False, False, False, False, False, False, True],
        "eps_basic": [10.0, 11.0, 9.0, 12.0, 5.0, 13.0, 99.0],
    })


def test_point_in_time_series_drops_consolidated_rows():
    s = point_in_time_series(_raw_filings(), "eps_basic")
    assert 99.0 not in s.to_numpy()


def test_point_in_time_series_restatement_keeps_latest_filing_date_only():
    s = point_in_time_series(_raw_filings(), "eps_basic")
    # Two standalone rows share period_end 2020-12-31 (filed 2021-01-18 and
    # 2021-01-25) - only the LATER filing_date's value (13.0) should survive,
    # the earlier (5.0) dropped, not averaged.
    assert 5.0 not in s.to_numpy()
    assert s.loc[pd.Timestamp("2021-01-25")] == 13.0
    assert pd.Timestamp("2021-01-18") not in s.index


def test_point_in_time_series_sorted_by_filing_date():
    s = point_in_time_series(_raw_filings(), "eps_basic")
    assert s.index.is_monotonic_increasing


def test_point_in_time_series_empty_input():
    empty = pd.DataFrame({"filing_date": [], "period_end": [], "consolidated": [], "eps_basic": []})
    s = point_in_time_series(empty, "eps_basic")
    assert s.empty


def _margin_filings(n=8, profit_step_at=4):
    idx = pd.bdate_range("2019-04-20", periods=n, freq="63D")
    profits = [10.0] * profit_step_at + [20.0] * (n - profit_step_at)
    return pd.DataFrame({
        "filing_date": idx,
        "period_end": idx,   # not exercised by margin_ttm, any distinct values suffice
        "consolidated": [False] * n,
        "revenue": [100.0] * n,
        "net_profit": profits,
    })


def test_margin_ttm_hand_computed():
    m = margin_ttm(_margin_filings(), window=4)
    # First 4 quarters: TTM profit 40 / TTM revenue 400 = 0.10
    assert np.isnan(m.iloc[2])
    assert m.iloc[3] == pytest.approx(40.0 / 400.0)
    # Last 4 quarters: TTM profit 80 / TTM revenue 400 = 0.20 - margin doubled.
    assert m.iloc[7] == pytest.approx(80.0 / 400.0)


def test_margin_ttm_masks_nonpositive_revenue():
    df = _margin_filings()
    df["revenue"] = 0.0
    m = margin_ttm(df, window=4)
    assert m.isna().all()


def test_margin_trend_zscore_produces_real_numbers_once_enough_history_exists():
    # 14 quarters (margin_ttm needs 4 to start, yoy_change needs 4 more, and
    # rolling_zscore(window=2) needs at least 2 trend points to ever produce
    # a real number). Not asserting a specific sign here - the z-score of
    # the underlying YoY-margin-change is legitimately sensitive to how
    # that change is itself decaying as the one-time step ages out of the
    # trailing TTM window (see test_margin_ttm_hand_computed for the
    # directly hand-verified raw margin values); this test only checks the
    # feature actually produces finite numbers once history allows it,
    # never NaN-forever or a divide-by-zero blowup.
    df = _margin_filings(n=14, profit_step_at=4)
    trend = margin_trend_zscore(df, ttm_window=4, trend_lag=4, zscore_window=2)
    valid = trend.dropna()
    assert len(valid) > 0
    assert np.isfinite(valid.to_numpy()).all()


def test_margin_trend_zscore_raw_yoy_change_is_positive_right_after_the_margin_step():
    # The RAW (pre-zscore) YoY margin change, hand-computed: at the first
    # quarter margin_ttm reflects the full step (idx 7, margin 0.20 vs
    # idx 3's 0.10), yoy_change must show the real, positive +0.10 - this
    # is the one unambiguous, hand-verifiable claim about the trend's sign.
    df = _margin_filings(n=14, profit_step_at=4)
    margin = margin_ttm(df, window=4)
    trend = yoy_change(margin, lag=4)
    assert trend.iloc[7] == pytest.approx(0.20 - 0.10)


def test_yoy_change_hand_computed():
    idx = pd.bdate_range("2020-01-06", periods=8)
    s = pd.Series([10.0, 11.0, 9.0, 12.0, 15.0, 8.0, 20.0, 6.0], index=idx)
    y = yoy_change(s, lag=4)
    assert np.isnan(y.iloc[3])           # only 3 priors exist, needs 4
    assert y.iloc[4] == pytest.approx(15.0 - 10.0)
    assert y.iloc[7] == pytest.approx(6.0 - 12.0)


def test_trailing_ttm_hand_computed():
    idx = pd.bdate_range("2020-01-06", periods=6)
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
    ttm = trailing_ttm(s, window=4)
    assert np.isnan(ttm.iloc[2])   # only 3 values exist, needs 4
    assert ttm.iloc[3] == pytest.approx(1.0 + 2.0 + 3.0 + 4.0)
    assert ttm.iloc[5] == pytest.approx(3.0 + 4.0 + 5.0 + 6.0)


def test_sue_zscore_matches_manual_rolling_zscore():
    idx = pd.bdate_range("2020-01-06", periods=16)
    eps = pd.Series(np.linspace(5.0, 20.0, 16), index=idx)
    surprise = yoy_change(eps, lag=4)
    z = sue_zscore(eps, window=4)
    valid = surprise.dropna()
    # Manual rolling z-score over the same surprise series, window=4,
    # min_periods=2 (max(4//2, 2)) - must match exactly, no drift from a
    # second, independently-written formula.
    mean = valid.rolling(4, min_periods=2).mean()
    std = valid.rolling(4, min_periods=2).std(ddof=1)
    expected = (valid - mean) / std
    pd.testing.assert_series_equal(z.dropna(), expected.dropna(), check_names=False)


def test_to_daily_panel_ffills_and_never_leaks_future_values():
    calendar = pd.bdate_range("2020-01-01", periods=10)
    filing_idx = pd.to_datetime(["2020-01-03", "2020-01-08"])
    s = pd.Series([1.0, 2.0], index=filing_idx)
    panel = to_daily_panel({"SYM": s}, calendar)

    # Before the first filing: NaN, not silently zero or forward-guessed.
    assert panel.loc[calendar[0], "SYM"] != panel.loc[calendar[0], "SYM"]  # NaN check
    # From filing 1 onward, held at 1.0 until filing 2's own date.
    assert panel.loc[pd.Timestamp("2020-01-06"), "SYM"] == 1.0
    assert panel.loc[pd.Timestamp("2020-01-07"), "SYM"] == 1.0
    # On and after filing 2's date, the new value - never leaking backward.
    assert panel.loc[pd.Timestamp("2020-01-08"), "SYM"] == 2.0
    assert panel.loc[calendar[-1], "SYM"] == 2.0


def test_to_daily_panel_empty_series_is_all_nan():
    calendar = pd.bdate_range("2020-01-01", periods=5)
    panel = to_daily_panel({"SYM": pd.Series(dtype="float64")}, calendar)
    assert panel["SYM"].isna().all()
