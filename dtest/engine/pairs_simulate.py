"""Honest two-leg pairs simulator: T+1-open fills on BOTH legs, real costs
on both (cash-equity `CostModel` for the long leg, `FuturesCostModel` for
the short leg), and a rollover-aware forced exit on the short leg's
futures contract.

DELIBERATELY LAYERED ON TOP OF `signals/pairs_reversion.py`, UNCHANGED.
`pair_trade_events` still decides WHEN a spread crosses and reverts -
that logic is cash-equity-close-based, matching every other signal in this
project, and does not change here. This module answers a different
question: given those signal-dated events, how does each one actually
FILL and COST - replacing the approximate, no-cost, same-day-close
convention `diagnostic_pairs_reversion.py`'s screening pass used.

THE ROLLOVER RULE, stated once, precisely, because it is the one place
this simulator deviates from the strict T+1-open convention every other
fill in this project uses. If the short leg's futures contract would roll
over on or before the signal's own exit date, the trade is FORCE-CLOSED
at the MARK (settle) price on the LAST trading day the position's actual
contract was still front month - not a T+1-open fill on the new contract,
which would silently price the position on an instrument it was never
actually in. This is not a look-ahead: a real account MUST decide to roll
or close before expiry, and the exchange's own settlement price is exactly
the authoritative EOD number that decision is made against. Every other
exit (spread reversion, max_hold_days, window_end) fills at the honest
T+1 open on both legs, unchanged from the rest of this project's
convention.

P&L CONVENTION: same as the screening diagnostic - net P&L is the average
of the long leg's and the short leg's own net return (after each leg's
own real costs), the return on capital committed per leg. Portfolio-level
margin/leverage economics for a pairs book are NOT modelled - see
`futures_costs.py`'s own docstring on why that is a separate, harder,
not-yet-attempted question.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dtest.config import Config
from dtest.engine.costs import CostModel
from dtest.engine.fills import next_trading_day
from dtest.engine.futures_costs import FuturesCostModel
from dtest.signals.pairs_reversion import PairTrade

EXIT_REVERTED = "reverted"
EXIT_TIME = "time"
EXIT_WINDOW_END = "window_end"
EXIT_ROLLOVER = "rollover"


@dataclass
class TwoLegTrade:
    symbol_a: str
    symbol_b: str
    long_symbol: str
    short_symbol: str
    signal_entry_date: pd.Timestamp
    entry_fill_date: pd.Timestamp | None
    signal_exit_date: pd.Timestamp
    exit_fill_date: pd.Timestamp | None
    exit_reason: str
    long_entry_price: float | None
    long_exit_price: float | None
    short_entry_price: float | None
    short_exit_price: float | None
    long_net_pct: float | None
    short_net_pct: float | None
    net_pnl_pct: float | None
    long_cost_pct: float | None
    short_cost_pct: float | None


def _first_rollover_on_or_before(fut_rollover: pd.DataFrame, symbol: str,
                                 start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp | None:
    if symbol not in fut_rollover.columns:
        return None
    window = fut_rollover.loc[start:end, symbol]
    hits = window[window.fillna(False)]
    return hits.index[0] if len(hits) else None


def simulate_pair_trades(
    pair_trades: list[PairTrade],
    *,
    cash_open: pd.DataFrame,
    fut_open: pd.DataFrame,
    fut_mark: pd.DataFrame,
    fut_rollover: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    cfg: Config,
) -> list[TwoLegTrade]:
    """Turn signal-dated `PairTrade`s into honestly-filled, honestly-costed
    two-leg trades. `cash_open` covers BOTH legs (the long leg trades cash
    equity regardless of which symbol in the pair it is); `fut_open`/
    `fut_mark`/`fut_rollover` are the short leg's futures panels from
    `dtest.data.fno_price`, pivoted to (date x symbol).
    """
    cost_model = CostModel.from_config(cfg)
    fut_cost_model = FuturesCostModel.from_config(cfg)

    results: list[TwoLegTrade] = []
    for t in pair_trades:
        entry_fill = next_trading_day(calendar, t.entry_date)
        signal_exit_fill = next_trading_day(calendar, t.exit_date)

        if entry_fill is None:
            results.append(TwoLegTrade(
                symbol_a=t.symbol_a, symbol_b=t.symbol_b, long_symbol=t.long_symbol,
                short_symbol=t.short_symbol, signal_entry_date=t.entry_date,
                entry_fill_date=None, signal_exit_date=t.exit_date, exit_fill_date=None,
                exit_reason="no_fill", long_entry_price=None, long_exit_price=None,
                short_entry_price=None, short_exit_price=None, long_net_pct=None,
                short_net_pct=None, net_pnl_pct=None, long_cost_pct=None, short_cost_pct=None,
            ))
            continue

        roll_date = _first_rollover_on_or_before(
            fut_rollover, t.short_symbol, entry_fill,
            signal_exit_fill if signal_exit_fill is not None else calendar[-1],
        )

        forced_rollover = roll_date is not None and (
            signal_exit_fill is None or roll_date <= signal_exit_fill)

        if forced_rollover:
            # Force-close at the mark on the LAST day before the roll - see
            # module docstring. roll_date is where is_rollover FIRST reads
            # True, so the last valid front-month day is the trading
            # session immediately before it.
            pos = calendar.get_loc(roll_date)
            exit_fill = calendar[pos - 1] if pos > 0 else None
            exit_reason = EXIT_ROLLOVER
            short_exit_is_mark = True
        else:
            exit_fill = signal_exit_fill
            exit_reason = t.exit_reason
            short_exit_is_mark = False

        if exit_fill is None or exit_fill <= entry_fill:
            results.append(TwoLegTrade(
                symbol_a=t.symbol_a, symbol_b=t.symbol_b, long_symbol=t.long_symbol,
                short_symbol=t.short_symbol, signal_entry_date=t.entry_date,
                entry_fill_date=entry_fill, signal_exit_date=t.exit_date, exit_fill_date=None,
                exit_reason="no_fill", long_entry_price=None, long_exit_price=None,
                short_entry_price=None, short_exit_price=None, long_net_pct=None,
                short_net_pct=None, net_pnl_pct=None, long_cost_pct=None, short_cost_pct=None,
            ))
            continue

        def _get(panel, date, sym):
            if date is None or sym not in panel.columns or date not in panel.index:
                return None
            v = panel.at[date, sym]
            return float(v) if pd.notna(v) and v > 0 else None

        long_entry = _get(cash_open, entry_fill, t.long_symbol)
        long_exit = _get(cash_open, exit_fill, t.long_symbol)
        short_entry = _get(fut_open, entry_fill, t.short_symbol)
        short_exit = (_get(fut_mark, exit_fill, t.short_symbol) if short_exit_is_mark
                     else _get(fut_open, exit_fill, t.short_symbol))

        if None in (long_entry, long_exit, short_entry, short_exit):
            results.append(TwoLegTrade(
                symbol_a=t.symbol_a, symbol_b=t.symbol_b, long_symbol=t.long_symbol,
                short_symbol=t.short_symbol, signal_entry_date=t.entry_date,
                entry_fill_date=entry_fill, signal_exit_date=t.exit_date, exit_fill_date=exit_fill,
                exit_reason="no_fill", long_entry_price=long_entry, long_exit_price=long_exit,
                short_entry_price=short_entry, short_exit_price=short_exit, long_net_pct=None,
                short_net_pct=None, net_pnl_pct=None, long_cost_pct=None, short_cost_pct=None,
            ))
            continue

        long_gross_ret = long_exit / long_entry - 1.0
        short_gross_ret = -(short_exit / short_entry - 1.0)

        long_rt = cost_model.round_trip(long_entry * 1.0, long_exit * 1.0)
        short_rt = fut_cost_model.round_trip(short_entry * 1.0, short_exit * 1.0, direction="short")

        long_net_pct = 100.0 * long_gross_ret - long_rt.pct_of_position
        short_net_pct = 100.0 * short_gross_ret - short_rt.pct_of_position
        net_pnl_pct = (long_net_pct + short_net_pct) / 2.0

        results.append(TwoLegTrade(
            symbol_a=t.symbol_a, symbol_b=t.symbol_b, long_symbol=t.long_symbol,
            short_symbol=t.short_symbol, signal_entry_date=t.entry_date,
            entry_fill_date=entry_fill, signal_exit_date=t.exit_date, exit_fill_date=exit_fill,
            exit_reason=exit_reason, long_entry_price=long_entry, long_exit_price=long_exit,
            short_entry_price=short_entry, short_exit_price=short_exit,
            long_net_pct=long_net_pct, short_net_pct=short_net_pct, net_pnl_pct=net_pnl_pct,
            long_cost_pct=long_rt.pct_of_position, short_cost_pct=short_rt.pct_of_position,
        ))

    return results


def trades_to_frame(trades: list[TwoLegTrade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([vars(t) for t in trades])
