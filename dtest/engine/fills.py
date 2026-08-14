"""Execution: a signal at bar T's close becomes a fill at bar T+1's open.

This is the single most important correction versus the predecessor project,
which entered at the CLOSE of the day the signal was computed from - a price
that cannot be known until after the signal already exists. At a 6-day hold
that flattered results by ~17% of the trade; at a 1.2-day hold (the rank-buffer
exit), ~80%. `config.execution.fill_at` refuses any value but `next_open` for
exactly this reason (see `config.py::Execution.validate`).

PARTICIPATION CAP, and why it is in SHARES, not value. Turnover (rupees) would
require dividing by a price to get shares, compounding a rounding error into
the cap. The bhavcopy exposes real traded share volume directly
(`TOTTRDQTY`/`TtlTradgVol`), so the cap is applied in the same unit the
exchange actually reports - a fill that would consume more than
`max_participation_pct` of that day's ENTIRE session volume did not happen, and
is rejected or scaled down rather than silently assumed.

A fill can fail for three separate, distinguishable reasons, and the result
says which: no next trading day exists (signal on the last bar of history), the
open price is missing (halt/delisting), or the day's volume is zero (no
liquidity at all). Each is a real, different failure mode a live account would
hit, and conflating them into one "couldn't fill" would hide which one is
actually happening in a given backtest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from dtest.config import Config

REJECT_NO_NEXT_DAY = "no_next_trading_day"
REJECT_NO_PRICE = "no_open_price"
REJECT_NO_VOLUME = "no_volume"
REJECT_ZERO_AFTER_CAP = "capped_to_zero_shares"


@dataclass(frozen=True)
class FillResult:
    signal_date: pd.Timestamp
    fill_date: pd.Timestamp | None
    symbol: str
    side: str                      # "buy" | "sell"
    requested_shares: int
    filled_shares: int
    fill_price: float | None
    value: float
    day_volume: float | None
    participation_pct: float | None
    was_capped: bool
    rejected: bool
    reject_reason: str | None

    @property
    def filled(self) -> bool:
        return not self.rejected and self.filled_shares > 0


def next_trading_day(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    """The first calendar date strictly after `date`, or None if `date` is last.

    `calendar` must be the TRADING calendar (already gap-free of weekends and
    holidays - see `dtest.data.prices._trading_calendar` / the bhav store's own
    day list) so "next" here always means "next session", not "next day".
    """
    pos = calendar.searchsorted(date, side="right")
    if pos >= len(calendar):
        return None
    return calendar[pos]


def _reject(signal_date, symbol, side, requested, fill_date, reason,
           price=None, volume=None) -> FillResult:
    return FillResult(
        signal_date=signal_date, fill_date=fill_date, symbol=symbol, side=side,
        requested_shares=requested, filled_shares=0, fill_price=price, value=0.0,
        day_volume=volume, participation_pct=None, was_capped=False,
        rejected=True, reject_reason=reason,
    )


def resolve_fill(
    signal_date: pd.Timestamp,
    symbol: str,
    side: str,
    requested_shares: int,
    open_panel: pd.DataFrame,
    volume_panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    cfg: Config,
) -> FillResult:
    """Turn one signal into one fill (or a reasoned rejection).

    `requested_shares` is always positive; `side` carries direction. Shares are
    whole numbers - Indian cash equity has no fractional-share trading, so a
    backtest that allows them is describing an order nobody could place.
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if requested_shares <= 0:
        raise ValueError(f"requested_shares must be positive, got {requested_shares}")

    fill_date = next_trading_day(calendar, signal_date)
    if fill_date is None:
        return _reject(signal_date, symbol, side, requested_shares, None, REJECT_NO_NEXT_DAY)

    price = open_panel.at[fill_date, symbol] if symbol in open_panel.columns else float("nan")
    if pd.isna(price) or price <= 0:
        return _reject(signal_date, symbol, side, requested_shares, fill_date, REJECT_NO_PRICE)

    volume = volume_panel.at[fill_date, symbol] if symbol in volume_panel.columns else float("nan")
    if pd.isna(volume) or volume <= 0:
        return _reject(signal_date, symbol, side, requested_shares, fill_date,
                       REJECT_NO_VOLUME, price=price, volume=volume)

    cap_shares = math.floor(cfg.execution.max_participation_pct / 100.0 * volume)
    filled = min(requested_shares, max(cap_shares, 0))
    was_capped = filled < requested_shares

    if filled <= 0:
        return _reject(signal_date, symbol, side, requested_shares, fill_date,
                       REJECT_ZERO_AFTER_CAP, price=price, volume=volume)

    return FillResult(
        signal_date=signal_date, fill_date=fill_date, symbol=symbol, side=side,
        requested_shares=requested_shares, filled_shares=filled, fill_price=float(price),
        value=filled * float(price), day_volume=float(volume),
        participation_pct=100.0 * filled / volume, was_capped=was_capped,
        rejected=False, reject_reason=None,
    )


@dataclass(frozen=True)
class PricePeek:
    fill_date: pd.Timestamp | None
    price: float | None
    rejected: bool
    reject_reason: str | None


def peek_fill_price(
    signal_date: pd.Timestamp,
    symbol: str,
    open_panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> PricePeek:
    """The price a fill WOULD execute at, with no share count or cap involved.

    Exists so a caller can size an order (rupee target -> whole shares) BEFORE
    calling `resolve_fill` with the real share count. The alternative - probing
    with an arbitrarily large requested size to discover the price - works but
    is a code smell that couples price discovery to the participation cap for
    no reason; this keeps the two concerns separate.
    """
    fill_date = next_trading_day(calendar, signal_date)
    if fill_date is None:
        return PricePeek(None, None, True, REJECT_NO_NEXT_DAY)
    price = open_panel.at[fill_date, symbol] if symbol in open_panel.columns else float("nan")
    if pd.isna(price) or price <= 0:
        return PricePeek(fill_date, None, True, REJECT_NO_PRICE)
    return PricePeek(fill_date, float(price), False, None)


def shares_for_value(target_value: float, price: float) -> int:
    """Whole shares affordable at `price` for `target_value` of capital.

    Floors, never rounds - an account cannot spend more than it targets, and
    rounding up would silently size a position larger than intended.
    """
    if price <= 0:
        return 0
    return math.floor(target_value / price)
