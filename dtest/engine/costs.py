"""Indian cash-equity DELIVERY cost model. Statutory rates, swept slippage.

Costs are the reason this project exists in its current form. The predecessor
ran three full sessions of parameter sweeps - ranking terms, exit constants, a
rank-buffer exit mechanism - every one of them measured with NO transaction
costs at all, and then had to retroactively re-score the lot. The best result in
that whole investigation (a rank-buffer exit, the first to beat its placebo band
out of sample) died the moment a toll that was calculable on day one was
applied: 0.183%/day of drag against 0.134%/day of gross return.

So: costs are charged here, at the bottom of the engine, and every metric in
this project is net by default. There is no gross-only path to accidentally
report.

---------------------------------------------------------------------------
THE UNITS TRAP - read this before touching any number below.

Charges divide naturally by TURNOVER (buy value + sell value). P&L divides by
POSITION (buy value alone). Turnover is ~2x position, so a rate quoted against
turnover is HALF the same rate quoted against position. Mixing the two makes
costs look survivable when they are not.

The predecessor project hit exactly this: it measured realised charges at
4.26 bps of TURNOVER and briefly read that as the delivery cost, which is below
the statutory delivery floor and only possible because those trades had largely
been intraday.

Everything in this module is POSITION-QUOTED at the public API. Internally,
per-leg figures are quoted against that leg's own value, which is the natural
unit for a statutory schedule.
---------------------------------------------------------------------------

WHY THE RATES ARE LOOKED UP AND NOT FITTED. The statutory schedule is public and
exact. Only slippage is genuinely unknown - it cannot be estimated from daily
OHLC - so it is a single named constant, deliberately loud, that re-prices every
result in the project with one edit. It sits ON TOP of the execution model's
optimism rather than excusing it.

Cross-check: this schedule reproduces the predecessor project's independently
derived figures exactly - 0.222% statutory round trip, 0.322% with 5 bps/side
slippage. Two separate implementations agreeing is the reason to trust it.
"""

from __future__ import annotations

from dataclasses import dataclass

from dtest.config import Config, Costs

BPS = 1e-4


@dataclass(frozen=True)
class LegCost:
    """Charges on ONE side of a trade, in rupees, against that leg's value."""

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
        """Everything the exchange/government takes. Excludes slippage."""
        return (self.brokerage + self.stt + self.exchange
                + self.sebi + self.stamp + self.gst)

    @property
    def total(self) -> float:
        return self.statutory + self.slippage

    @property
    def pct_of_leg(self) -> float:
        return 100.0 * self.total / self.value if self.value else 0.0


@dataclass(frozen=True)
class RoundTrip:
    """A complete buy-then-sell, costed. All percentages are POSITION-quoted."""

    buy: LegCost
    sell: LegCost

    @property
    def total(self) -> float:
        return self.buy.total + self.sell.total

    @property
    def statutory(self) -> float:
        return self.buy.statutory + self.sell.statutory

    @property
    def slippage(self) -> float:
        return self.buy.slippage + self.sell.slippage

    @property
    def pct_of_position(self) -> float:
        """The number that matters: cost as a % of capital deployed."""
        return 100.0 * self.total / self.buy.value if self.buy.value else 0.0

    @property
    def pct_of_turnover(self) -> float:
        """Provided ONLY so a turnover-quoted figure never has to be improvised.
        Never compare this against a P&L percentage - see the units trap above."""
        t = self.buy.value + self.sell.value
        return 100.0 * self.total / t if t else 0.0


class CostModel:
    """Charges both legs of a delivery trade under the statutory schedule.

    Anything held overnight is DELIVERY, including a one-session hold. There is
    no intraday-rate version of a strategy that carries positions, and assuming
    one is how a backtest quietly halves its own STT.
    """

    def __init__(self, costs: Costs):
        self.c = costs

    @classmethod
    def from_config(cls, cfg: Config) -> "CostModel":
        return cls(cfg.costs)

    def _brokerage(self, value: float) -> float:
        rate = self.c.brokerage_pct_per_side / 100.0
        return min(value * rate, self.c.brokerage_cap_inr) if rate else 0.0

    def leg(self, value: float, side: str) -> LegCost:
        """Cost of one leg. `side` is 'buy' or 'sell'."""
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if value < 0:
            raise ValueError(f"leg value must be non-negative, got {value}")

        brokerage = self._brokerage(value)
        stt_rate = (self.c.stt_buy_pct if side == "buy" else self.c.stt_sell_pct)
        stt = value * stt_rate / 100.0
        exchange = value * self.c.exchange_txn_pct_per_side / 100.0
        sebi = value * self.c.sebi_pct_per_side / 100.0
        # Stamp duty is a BUY-side levy only. Charging it on both legs is a
        # common and quietly expensive error.
        stamp = value * self.c.stamp_duty_buy_pct / 100.0 if side == "buy" else 0.0
        # GST applies to the service charges, NOT to STT or stamp duty.
        gst = (brokerage + exchange + sebi) * self.c.gst_pct / 100.0
        slippage = value * self.c.slippage_bps_per_side * BPS

        return LegCost(value=value, brokerage=brokerage, stt=stt, exchange=exchange,
                       sebi=sebi, stamp=stamp, gst=gst, slippage=slippage)

    def round_trip(self, buy_value: float, sell_value: float) -> RoundTrip:
        return RoundTrip(buy=self.leg(buy_value, "buy"),
                         sell=self.leg(sell_value, "sell"))

    def round_trip_pct(self, *, include_slippage: bool = True) -> float:
        """Round-trip cost as a % of position, for a flat round trip.

        Assumes sell value equals buy value. That is exactly right for the
        headline figure and slightly conservative for a winner (whose sell leg is
        larger, so its sell-side charges are larger too) - the per-trade path
        through `round_trip()` uses real values and does not approximate.
        """
        rt = self.round_trip(1.0, 1.0)
        total = rt.total if include_slippage else rt.statutory
        return 100.0 * total

    def describe(self) -> str:
        stat = self.round_trip_pct(include_slippage=False)
        full = self.round_trip_pct(include_slippage=True)
        slip = self.c.slippage_bps_per_side
        return (f"delivery round trip {full:.3f}% of position "
                f"(statutory {stat:.3f}% + {slip:.1f} bps/side slippage)")
