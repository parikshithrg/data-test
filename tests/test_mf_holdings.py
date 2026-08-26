"""dtest.features.mf_holdings - point-in-time MF-accumulation feature
construction, verified against small synthetic panels covering each real
edge case the module docstring states (new-position exclusion, gap
guarding, filing-lag event mapping)."""

from __future__ import annotations

import pandas as pd

from dtest.features.mf_holdings import (
    aggregate_monthly_quantity,
    breadth_panel,
    build_isin_symbol_map,
    new_entrant_flag,
    quantity_pct_change,
    to_event_panel,
)


def test_build_isin_symbol_map_takes_most_recent_symbol():
    bhav = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        "symbol": ["OLDNAME", "OLDNAME", "NEWNAME"],
        "isin": ["INE000000001", "INE000000001", "INE000000001"],
    })
    m = build_isin_symbol_map(bhav)
    assert m == {"INE000000001": "NEWNAME"}


def test_build_isin_symbol_map_drops_null_isin():
    bhav = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "symbol": ["A", "B"],
        "isin": ["INE111111111", None],
    })
    m = build_isin_symbol_map(bhav)
    assert m == {"INE111111111": "A"}


def test_aggregate_monthly_quantity_sums_across_schemes_and_amcs():
    holdings = pd.DataFrame({
        "period_end": pd.to_datetime(["2024-01-31"] * 3),
        "isin": ["INE1", "INE1", "INE1"],
        "quantity": [1000, 2000, 500],
        "scheme_name": ["Axis Fund A", "Axis Fund B", "SBI Fund C"],
    })
    agg = aggregate_monthly_quantity(holdings, {"INE1": "RELIANCE"})
    assert len(agg) == 1
    assert agg.iloc[0]["symbol"] == "RELIANCE"
    assert agg.iloc[0]["total_quantity"] == 3500
    assert agg.iloc[0]["n_schemes"] == 3


def test_aggregate_monthly_quantity_drops_unmapped_isin():
    holdings = pd.DataFrame({
        "period_end": pd.to_datetime(["2024-01-31"]),
        "isin": ["INE_UNKNOWN"],
        "quantity": [1000],
        "scheme_name": ["Some Fund"],
    })
    agg = aggregate_monthly_quantity(holdings, {"INE1": "RELIANCE"})
    assert agg.empty


def test_quantity_pct_change_normal_case():
    monthly = pd.DataFrame({
        "period_end": pd.to_datetime(["2024-01-31", "2024-02-29"]),
        "symbol": ["RELIANCE", "RELIANCE"],
        "total_quantity": [1000, 1200],
        "n_schemes": [2, 2],
    })
    pct = quantity_pct_change(monthly)
    assert pd.isna(pct.loc[pd.Timestamp("2024-01-31"), "RELIANCE"])
    assert abs(pct.loc[pd.Timestamp("2024-02-29"), "RELIANCE"] - 0.2) < 1e-9


def test_quantity_pct_change_new_position_is_nan_not_infinite():
    # RELIANCE held only from Feb onward - no prior month to compare against.
    monthly = pd.DataFrame({
        "period_end": pd.to_datetime(["2024-01-31", "2024-02-29"]),
        "symbol": ["TCS", "RELIANCE"],
        "total_quantity": [1000, 500],
        "n_schemes": [1, 1],
    })
    pct = quantity_pct_change(monthly)
    assert pd.isna(pct.loc[pd.Timestamp("2024-02-29"), "RELIANCE"])


def test_quantity_pct_change_respects_max_gap_days():
    # A 3-month gap between disclosed periods should NOT be treated as an
    # ordinary one-month change.
    monthly = pd.DataFrame({
        "period_end": pd.to_datetime(["2024-01-31", "2024-04-30"]),
        "symbol": ["RELIANCE", "RELIANCE"],
        "total_quantity": [1000, 1500],
        "n_schemes": [1, 1],
    })
    pct = quantity_pct_change(monthly, max_gap_days=40)
    assert pd.isna(pct.loc[pd.Timestamp("2024-04-30"), "RELIANCE"])


def test_to_event_panel_fires_on_first_trading_day_at_or_after_filing_lag():
    calendar = pd.date_range("2024-02-01", "2024-02-20", freq="B")  # business days
    monthly_signal = pd.DataFrame(
        {"RELIANCE": [0.25]}, index=pd.to_datetime(["2024-01-31"]))
    panel = to_event_panel(monthly_signal, calendar, lag_days=10)
    # period_end + 10 days = 2024-02-10 (a Saturday) -> first business day >= it is 2024-02-12
    fired = panel["RELIANCE"].dropna()
    assert len(fired) == 1
    assert str(fired.index[0].date()) == "2024-02-12"
    assert fired.iloc[0] == 0.25


def test_to_event_panel_drops_filing_dates_beyond_calendar_end():
    calendar = pd.date_range("2024-01-01", "2024-01-31", freq="B")
    monthly_signal = pd.DataFrame(
        {"RELIANCE": [0.25]}, index=pd.to_datetime(["2024-06-30"]))
    panel = to_event_panel(monthly_signal, calendar, lag_days=10)
    assert panel["RELIANCE"].notna().sum() == 0


def test_to_event_panel_accepts_a_boolean_monthly_signal():
    # Regression: a plain float-NaN-initialized frame raised
    # LossySetitemError on modern pandas the first time a boolean panel
    # (e.g. new_entrant_flag's output) was assigned into it.
    calendar = pd.date_range("2024-02-01", "2024-02-20", freq="B")
    monthly_signal = pd.DataFrame(
        {"RELIANCE": [True], "TCS": [False]}, index=pd.to_datetime(["2024-01-31"]))
    panel = to_event_panel(monthly_signal, calendar, lag_days=10)
    fired = panel.dropna(how="all")
    assert len(fired) == 1
    assert bool(fired.iloc[0]["RELIANCE"]) is True
    assert bool(fired.iloc[0]["TCS"]) is False


def test_new_entrant_flag_true_when_absent_then_held():
    monthly = pd.DataFrame({
        "period_end": pd.to_datetime(["2024-01-31", "2024-02-29"]),
        "symbol": ["TCS", "RELIANCE"],
        "total_quantity": [1000, 500],
        "n_schemes": [1, 1],
    })
    flag = new_entrant_flag(monthly)
    assert flag.loc[pd.Timestamp("2024-02-29"), "RELIANCE"] == True
    assert flag.loc[pd.Timestamp("2024-01-31"), "TCS"] == False  # no valid prior period at all


def test_new_entrant_flag_false_when_held_both_months():
    monthly = pd.DataFrame({
        "period_end": pd.to_datetime(["2024-01-31", "2024-02-29"]),
        "symbol": ["RELIANCE", "RELIANCE"],
        "total_quantity": [1000, 1200],
        "n_schemes": [1, 1],
    })
    flag = new_entrant_flag(monthly)
    assert flag.loc[pd.Timestamp("2024-02-29"), "RELIANCE"] == False


def test_new_entrant_flag_true_again_after_a_real_sell_and_rebuy():
    # TCS establishes that Feb genuinely was a disclosed month (real ~30-day
    # adjacency to both Jan and Mar) - RELIANCE is absent that month (sold
    # out), then reappears in March, which should count as new again.
    monthly = pd.DataFrame({
        "period_end": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
        "symbol": ["RELIANCE", "TCS", "RELIANCE"],
        "total_quantity": [1000, 2000, 500],
        "n_schemes": [1, 1, 1],
    })
    flag = new_entrant_flag(monthly)
    assert flag.loc[pd.Timestamp("2024-03-31"), "RELIANCE"] == True


def test_new_entrant_flag_respects_max_gap_days():
    monthly = pd.DataFrame({
        "period_end": pd.to_datetime(["2024-01-31", "2024-06-30"]),
        "symbol": ["RELIANCE", "RELIANCE"],
        "total_quantity": [1000, 500],
        "n_schemes": [1, 1],
    })
    flag = new_entrant_flag(monthly, max_gap_days=40)
    # too large a gap to conclude anything about "last month" - not flagged
    assert flag.loc[pd.Timestamp("2024-06-30"), "RELIANCE"] == False


def test_breadth_panel_pivots_n_schemes():
    monthly = pd.DataFrame({
        "period_end": pd.to_datetime(["2024-01-31", "2024-01-31"]),
        "symbol": ["RELIANCE", "TCS"],
        "total_quantity": [1000, 2000],
        "n_schemes": [5, 2],
    })
    panel = breadth_panel(monthly)
    assert panel.loc[pd.Timestamp("2024-01-31"), "RELIANCE"] == 5
    assert panel.loc[pd.Timestamp("2024-01-31"), "TCS"] == 2
