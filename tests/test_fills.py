"""Fill engine tests. Each pins a specific failure mode a live account would hit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtest import load_config
from dtest.engine.fills import (
    REJECT_NO_NEXT_DAY, REJECT_NO_PRICE, REJECT_NO_VOLUME, REJECT_ZERO_AFTER_CAP,
    next_trading_day, resolve_fill, shares_for_value,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _panels(n=10):
    idx = pd.bdate_range("2020-01-06", periods=n)   # Monday start
    open_ = pd.DataFrame({"A": [100.0] * n}, index=idx)
    vol = pd.DataFrame({"A": [10_000.0] * n}, index=idx)
    return idx, open_, vol


def test_fills_at_next_open_never_same_bar(cfg):
    """The core correction: T's signal fills at T+1's open, never T's own price."""
    idx, open_, vol = _panels()
    signal_date = idx[2]
    res = resolve_fill(signal_date, "A", "buy", 10, open_, vol, idx, cfg)
    assert res.filled
    assert res.fill_date == idx[3]
    assert res.fill_date > signal_date
    assert res.fill_price == 100.0


def test_signal_on_last_bar_cannot_fill(cfg):
    idx, open_, vol = _panels()
    res = resolve_fill(idx[-1], "A", "buy", 10, open_, vol, idx, cfg)
    assert res.rejected
    assert res.reject_reason == REJECT_NO_NEXT_DAY
    assert res.fill_date is None


def test_missing_open_price_is_rejected_not_silently_zero(cfg):
    """A halted or delisted name the next day cannot be traded - not a Rs 0 fill."""
    idx, open_, vol = _panels()
    open_.loc[idx[3], "A"] = np.nan
    res = resolve_fill(idx[2], "A", "buy", 10, open_, vol, idx, cfg)
    assert res.rejected
    assert res.reject_reason == REJECT_NO_PRICE
    assert res.fill_date == idx[3]     # the date IS known, just not tradeable


def test_zero_volume_day_is_rejected(cfg):
    idx, open_, vol = _panels()
    vol.loc[idx[3], "A"] = 0.0
    res = resolve_fill(idx[2], "A", "buy", 10, open_, vol, idx, cfg)
    assert res.rejected
    assert res.reject_reason == REJECT_NO_VOLUME


def test_participation_cap_scales_a_large_order_down(cfg):
    """A backtest cannot buy more of the day's volume than max_participation_pct allows."""
    idx, open_, vol = _panels()
    vol.loc[idx[3], "A"] = 1000.0          # thin day
    cap_pct = cfg.execution.max_participation_pct
    huge_order = 10_000
    res = resolve_fill(idx[2], "A", "buy", huge_order, open_, vol, idx, cfg)
    assert res.was_capped
    assert res.filled_shares == int(cap_pct / 100.0 * 1000.0)
    assert res.filled_shares < huge_order
    assert res.participation_pct == pytest.approx(cap_pct, abs=0.5)


def test_small_order_is_never_capped(cfg):
    idx, open_, vol = _panels()
    res = resolve_fill(idx[2], "A", "buy", 5, open_, vol, idx, cfg)
    assert not res.was_capped
    assert res.filled_shares == 5


def test_order_capped_to_zero_shares_is_rejected(cfg):
    idx, open_, vol = _panels()
    vol.loc[idx[3], "A"] = 1.0             # cap floors to 0 shares at any reasonable pct
    res = resolve_fill(idx[2], "A", "buy", 100, open_, vol, idx, cfg)
    assert res.rejected
    assert res.reject_reason == REJECT_ZERO_AFTER_CAP


def test_requires_whole_shares(cfg):
    idx, open_, vol = _panels()
    with pytest.raises(ValueError):
        resolve_fill(idx[2], "A", "buy", 0, open_, vol, idx, cfg)
    with pytest.raises(ValueError):
        resolve_fill(idx[2], "A", "buy", -5, open_, vol, idx, cfg)


def test_rejects_bad_side(cfg):
    idx, open_, vol = _panels()
    with pytest.raises(ValueError):
        resolve_fill(idx[2], "A", "long", 5, open_, vol, idx, cfg)


def test_next_trading_day_skips_gaps_not_just_calendar_days():
    """A trading calendar with a holiday gap must be respected, not assumed daily."""
    idx = pd.DatetimeIndex(["2020-01-06", "2020-01-07", "2020-01-10"])  # Fri->Mon gap
    assert next_trading_day(idx, idx[1]) == idx[2]
    assert next_trading_day(idx, idx[2]) is None


def test_shares_for_value_floors_never_rounds_up():
    assert shares_for_value(999.0, 100.0) == 9
    assert shares_for_value(1000.0, 100.0) == 10
    assert shares_for_value(50.0, 100.0) == 0
    assert shares_for_value(100.0, 0.0) == 0


def test_symbol_missing_from_panel_is_rejected_not_a_crash(cfg):
    idx, open_, vol = _panels()
    res = resolve_fill(idx[2], "GHOST", "buy", 5, open_, vol, idx, cfg)
    assert res.rejected
    assert res.reject_reason == REJECT_NO_PRICE
