"""Indian stock-FUTURES cost model. Statutory rates, same discipline as
`costs.py` but a genuinely different schedule, not the same numbers
relabelled - see `config.toml`'s `[futures_costs]` section for exactly
which rates differ and why, and its own note on why this schedule is less
battle-tested than `[costs]` (no independent cross-check yet).

THE ONE STRUCTURAL FACT WORTH STATING TWICE: futures STT is charged on the
SELL side ONLY. Cash-equity delivery ([costs]) charges STT on BOTH legs.
Applying the delivery schedule to a futures leg by accident would silently
double-charge STT on the buy side and get the sell-side rate wrong too -
exactly the class of unit/schedule-mixing error `costs.py`'s own docstring
warns about for turnover-vs-position, here for equity-vs-futures instead.

MARGIN IS NOT MODELLED HERE. A real futures position ties up margin, not
full notional - this module prices the TRANSACTION, the same statutory-
charges-per-leg question `costs.py` answers for equity. How much capital a
short leg actually consumes (SPAN + exposure margin) is a portfolio-level,
sizing-DEPENDENT question this project has no margin data to answer
honestly yet - same category of gap `engine/portfolio.py` already states
for why cash-equity shorting was excluded outright. The TRADE-LEVEL
(sizing-independent) simulator this cost model feeds does not need margin
to report an honest per-trade percentage return; a portfolio-level account
simulation for pairs would need it and is explicitly not attempted yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from dtest.config import Config, FuturesCosts

BPS = 1e-4


@dataclass(frozen=True)
class FuturesLegCost:
    """Charges on ONE side of a futures trade, in rupees, against that
    leg's own notional value."""

    value: float
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float
    slippage: float

    @property
    def statutory(self) -> float:
        return (self.brokerage + self.stt + self.exchange
                + self.sebi + self.stamp + self.gst)

    @property
    def total(self) -> float:
        return self.statutory + self.slippage

    @property
    def pct_of_leg(self) -> float:
        return 100.0 * self.total / self.value if self.value else 0.0


@dataclass(frozen=True)
class FuturesRoundTrip:
    """A complete open-then-close futures leg, costed. Percentages are
    quoted against the OPEN leg's notional, matching `costs.RoundTrip`'s
    `pct_of_position` convention so the two schedules stay comparable."""

    open: FuturesLegCost
    close: FuturesLegCost

    @property
    def total(self) -> float:
        return self.open.total + self.close.total

    @property
    def statutory(self) -> float:
        return self.open.statutory + self.close.statutory

    @property
    def slippage(self) -> float:
        return self.open.slippage + self.close.slippage

    @property
    def pct_of_position(self) -> float:
        return 100.0 * self.total / self.open.value if self.open.value else 0.0


class FuturesCostModel:
    """Charges both legs of a stock-futures round trip (open a position,
    close it - either direction, since a short OPENS with a sell and
    CLOSES with a buy, the mirror of a long)."""

    def __init__(self, costs: FuturesCosts):
        self.c = costs

    @classmethod
    def from_config(cls, cfg: Config) -> "FuturesCostModel":
        return cls(cfg.futures_costs)

    def _brokerage(self, value: float) -> float:
        rate = self.c.brokerage_pct_per_side / 100.0
        return min(value * rate, self.c.brokerage_cap_inr) if rate else 0.0

    def leg(self, value: float, action: str) -> FuturesLegCost:
        """Cost of one leg. `action` is 'buy' or 'sell' - which physical
        order this leg is, NOT whether the position is long or short (a
        short's OPENING leg is a 'sell', its CLOSING leg is a 'buy' -
        callers pass the real order side, same convention `costs.leg`
        uses for cash equity)."""
        if action not in ("buy", "sell"):
            raise ValueError(f"action must be 'buy' or 'sell', got {action!r}")
        if value < 0:
            raise ValueError(f"leg value must be non-negative, got {value}")

        brokerage = self._brokerage(value)
        # STT: sell side only - see module docstring. A buy leg pays zero STT.
        stt = value * self.c.stt_sell_pct / 100.0 if action == "sell" else 0.0
        exchange = value * self.c.exchange_txn_pct_per_side / 100.0
        sebi = value * self.c.sebi_pct_per_side / 100.0
        # Stamp duty is a BUY-side levy only, same convention as cash equity,
        # futures' own (lower) rate.
        stamp = value * self.c.stamp_duty_buy_pct / 100.0 if action == "buy" else 0.0
        gst = (brokerage + exchange + sebi) * self.c.gst_pct / 100.0
        slippage = value * self.c.slippage_bps_per_side * BPS

        return FuturesLegCost(value=value, brokerage=brokerage, stt=stt, exchange=exchange,
                              sebi=sebi, stamp=stamp, gst=gst, slippage=slippage)

    def round_trip(self, open_value: float, close_value: float, *, direction: str) -> FuturesRoundTrip:
        """`direction` is 'long' (opens with a buy, closes with a sell) or
        'short' (opens with a sell, closes with a buy) - the one place this
        model needs to know which side of the pair trade it is pricing."""
        if direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
        open_action, close_action = (("buy", "sell") if direction == "long" else ("sell", "buy"))
        return FuturesRoundTrip(open=self.leg(open_value, open_action),
                                close=self.leg(close_value, close_action))

    def round_trip_pct(self, direction: str, *, include_slippage: bool = True) -> float:
        """Round-trip cost as a % of position, for a flat round trip
        (open value == close value) - the headline figure, same
        `costs.CostModel.round_trip_pct` convention."""
        rt = self.round_trip(1.0, 1.0, direction=direction)
        total = rt.total if include_slippage else rt.statutory
        return 100.0 * total

    def describe(self) -> str:
        long_full = self.round_trip_pct("long")
        short_full = self.round_trip_pct("short")
        slip = self.c.slippage_bps_per_side
        return (f"futures round trip: long {long_full:.3f}% / short {short_full:.3f}% "
               f"of position ({slip:.1f} bps/side slippage included)")
