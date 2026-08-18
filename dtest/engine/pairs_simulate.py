"""Honest two-leg pairs simulator: T+1-open fills on BOTH legs, real costs
on both (cash-equity `CostModel` for the long leg, `FuturesCostModel` for
the short leg), a rollforward-at-entry rule for the short leg's contract
choice, and a rollover-aware forced exit when that chosen contract still
runs out before the signal's own exit.

DELIBERATELY LAYERED ON TOP OF `signals/pairs_reversion.py`, UNCHANGED.
`pair_trade_events` still decides WHEN a spread crosses and reverts -
that logic is cash-equity-close-based, matching every other signal in this
project, and does not change here. This module answers a different
question: given those signal-dated events, how does each one actually
FILL and COST - replacing the approximate, no-cost, same-day-close
convention `diagnostic_pairs_reversion.py`'s screening pass used.

ROLLFORWARD-AT-ENTRY, added 2026-08-18 after diagnosing the first honest
run (2026-08-17): a plain front-month entry rollover-forced 73/197 (37%)
of trades, and the cause was NOT the position being held too long against
`pairs_reversion.py`'s own 20-day cap (only 1 of 197 trades ever reached
it) - it was entering INTO a contract that was already nearly expired.
Rollover-forced trades entered with a median 8 days to expiry versus 21
for trades that reverted normally; of trades entered with <=10 days to
expiry, 70% ended up rollover-forced versus 22% otherwise. `MIN_DTE_FOR_
ENTRY_DAYS = 10` (half of `pairs_reversion.MAX_HOLD_DAYS` - a position
that could run the strategy's own full hold budget needs at least half a
contract's remaining life to have a real chance of finishing in the
contract it opened in) - if the front-month contract has fewer days to
expiry than this at entry, the short leg opens in the NEXT contract
instead, mirroring what a real desk does rather than opening a fresh
position days before its instrument expires. Chosen deliberately BEFORE
looking at how much it improves the result, not swept/picked for the best
number - the same discipline this project's exit-geometry and entry-delay
diagnostics used. Single-hop only: once chosen, a trade's contract is not
re-rolled again mid-life, so a position that runs the full 20-day budget
in an entered-with-exactly-10-days contract can still hit a second roll -
a stated, not hidden, limitation.

THE ROLLOVER RULE, stated once, precisely, because it is the one place
this simulator deviates from the strict T+1-open convention every other
fill in this project uses. If the CHOSEN contract (front month, or the
next one under the rollforward rule above) would expire on or before the
signal's own exit date, the trade is FORCE-CLOSED at the MARK (settle)
price on the LAST trading day that specific contract still traded - not a
T+1-open fill on yet another contract, which would silently price the
position on an instrument it was never actually in. This is not a
look-ahead: a real account MUST decide to roll or close before expiry, and
the exchange's own settlement price is exactly the authoritative EOD
number that decision is made against. Every other exit (spread reversion,
max_hold_days, window_end) fills at the honest T+1 open on both legs,
unchanged from the rest of this project's convention.

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

# Half of pairs_reversion.MAX_HOLD_DAYS (20) - see the module docstring's
# ROLLFORWARD-AT-ENTRY section for the reasoning.
MIN_DTE_FOR_ENTRY_DAYS = 10


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
    short_contract_expiry: pd.Timestamp | None
    rolled_forward_at_entry: bool


def _no_fill(t: PairTrade, entry_fill=None, exit_fill=None, long_entry=None, long_exit=None,
            short_entry=None, short_exit=None, contract_expiry=None, rolled=False) -> TwoLegTrade:
    return TwoLegTrade(
        symbol_a=t.symbol_a, symbol_b=t.symbol_b, long_symbol=t.long_symbol,
        short_symbol=t.short_symbol, signal_entry_date=t.entry_date,
        entry_fill_date=entry_fill, signal_exit_date=t.exit_date, exit_fill_date=exit_fill,
        exit_reason="no_fill", long_entry_price=long_entry, long_exit_price=long_exit,
        short_entry_price=short_entry, short_exit_price=short_exit, long_net_pct=None,
        short_net_pct=None, net_pnl_pct=None, long_cost_pct=None, short_cost_pct=None,
        short_contract_expiry=contract_expiry, rolled_forward_at_entry=rolled,
    )


def _valid(v) -> float | None:
    return float(v) if pd.notna(v) and v > 0 else None


def _select_entry_contract(by_symbol_date: pd.DataFrame, symbol: str, date: pd.Timestamp,
                           min_dte: int) -> tuple[pd.Series | None, bool]:
    """The live contract(s) for `symbol` on `date`, ranked by expiry. Returns
    (chosen row, whether this is a rollforward pick) or (None, False) if the
    short leg has no futures data that day at all."""
    key = (symbol, date)
    if key not in by_symbol_date.index:
        return None, False
    rows = by_symbol_date.loc[[key]].sort_values("rank")
    front = rows.iloc[0]
    if front["days_to_expiry"] < min_dte and len(rows) > 1:
        return rows.iloc[1], True
    return front, False


def simulate_pair_trades(
    pair_trades: list[PairTrade],
    *,
    cash_open: pd.DataFrame,
    fut_contracts: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    cfg: Config,
    min_dte_for_entry_days: int = MIN_DTE_FOR_ENTRY_DAYS,
) -> list[TwoLegTrade]:
    """Turn signal-dated `PairTrade`s into honestly-filled, honestly-costed
    two-leg trades. `cash_open` covers BOTH legs (the long leg trades cash
    equity regardless of which symbol in the pair it is); `fut_contracts` is
    the short leg's full contract table from
    `dtest.data.fno_price.load_stock_futures_contracts` (NOT collapsed to
    front month - this function needs to see the next contract too).
    """
    cost_model = CostModel.from_config(cfg)
    fut_cost_model = FuturesCostModel.from_config(cfg)

    by_symbol_date = fut_contracts.set_index(["symbol", "date"]).sort_index()
    by_symbol_expiry = fut_contracts.set_index(["symbol", "expiry_date", "date"]).sort_index()

    results: list[TwoLegTrade] = []
    for t in pair_trades:
        entry_fill = next_trading_day(calendar, t.entry_date)
        signal_exit_fill = next_trading_day(calendar, t.exit_date)

        if entry_fill is None:
            results.append(_no_fill(t))
            continue

        entry_contract, rolled = _select_entry_contract(
            by_symbol_date, t.short_symbol, entry_fill, min_dte_for_entry_days)
        if entry_contract is None:
            results.append(_no_fill(t, entry_fill=entry_fill))
            continue

        chosen_expiry = entry_contract["expiry_date"]
        short_entry = _valid(entry_contract["open_price"])

        life = by_symbol_expiry.loc[(t.short_symbol, chosen_expiry)]
        last_contract_date = life.index.max()

        if signal_exit_fill is None or last_contract_date < signal_exit_fill:
            # The CHOSEN contract (possibly already rolled forward once)
            # still runs out before the signal's own exit - force-close at
            # the mark on the last day it traded. See module docstring.
            exit_fill = last_contract_date if last_contract_date >= entry_fill else None
            exit_reason = EXIT_ROLLOVER
            short_exit_is_mark = True
        else:
            exit_fill = signal_exit_fill
            exit_reason = t.exit_reason
            short_exit_is_mark = False

        if exit_fill is None or exit_fill <= entry_fill:
            results.append(_no_fill(t, entry_fill=entry_fill, short_entry=short_entry,
                                    contract_expiry=chosen_expiry, rolled=rolled))
            continue

        long_entry = _valid(cash_open.at[entry_fill, t.long_symbol]) \
            if t.long_symbol in cash_open.columns and entry_fill in cash_open.index else None
        long_exit = _valid(cash_open.at[exit_fill, t.long_symbol]) \
            if t.long_symbol in cash_open.columns and exit_fill in cash_open.index else None
        short_exit_col = "price" if short_exit_is_mark else "open_price"
        short_exit = _valid(life.at[exit_fill, short_exit_col]) if exit_fill in life.index else None

        if None in (long_entry, long_exit, short_entry, short_exit):
            results.append(_no_fill(t, entry_fill=entry_fill, exit_fill=exit_fill,
                                    long_entry=long_entry, long_exit=long_exit,
                                    short_entry=short_entry, short_exit=short_exit,
                                    contract_expiry=chosen_expiry, rolled=rolled))
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
            short_contract_expiry=chosen_expiry, rolled_forward_at_entry=rolled,
        ))

    return results


def trades_to_frame(trades: list[TwoLegTrade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([vars(t) for t in trades])
