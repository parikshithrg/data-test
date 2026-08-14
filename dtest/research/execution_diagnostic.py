"""DIAGNOSTIC ONLY - same-bar-close entry, to isolate how much of a result
depends on execution timing.

`config.execution.fill_at` refuses anything but `next_open` for real hypothesis
tests, for good reason: entering at the close of the bar a signal was computed
from is a price that could not have been known until after the signal already
existed. This module deliberately BREAKS that rule, on purpose, to answer one
question: for a signal whose result differs sharply from a predecessor
project's claim for the "same" strategy, how much of the gap is the execution
correction itself versus something else (a different test window, different
parameters)? That is a legitimate thing to WANT TO KNOW; it is not a legitimate
thing to trade on, which is why this lives in `dtest.research`, is never
imported by production code, and never writes to the hypothesis log.

WHY THE FIRST ELIGIBLE STOP/TARGET BAR IS T+1, NOT T. Entering at bar T's
CLOSE means the position exists for perhaps the last few seconds of session T.
That bar's own HIGH and LOW already happened - they describe price action
BEFORE the entry, not after it. Checking T's own range for a stop/target touch
would be checking a path the position was never actually exposed to. So the
intrabar walk here starts at T+1, exactly one bar later than the production
simulator's walk (which starts at the entry bar itself, because there entry
happens at that bar's OPEN, before its own range unfolds - see
`engine/simulate.py`'s docstring). `max_hold_days` is interpreted the same way:
sessions of REAL exposure, so it also counts from T+1.

PARTICIPATION CAP is applied against the SAME bar's full-session volume, which
is itself optimistic for a closing-price fill (a real closing-auction fill is
capped by auction liquidity, not the whole day's volume) - stated as a further
source of optimism in the same-bar number, not corrected for, since the point
is to isolate ENTRY TIMING as one variable, not to also model auction
microstructure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dtest.config import Config
from dtest.engine.costs import CostModel
from dtest.engine.fills import next_trading_day, shares_for_value
from dtest.engine.simulate import (
    EXIT_NO_FILL, EXIT_STOP, EXIT_TARGET, EXIT_TIME, EXIT_UNRESOLVED, ExitRule, Trade,
)


def _same_bar_close_fill(signal_date, symbol, close, volume, cfg) -> tuple[float | None, float | None]:
    """(price, day_volume) at the SIGNAL bar's own close, or (None, None) if
    untradeable that bar (missing price/volume)."""
    if symbol not in close.columns:
        return None, None
    price = close.at[signal_date, symbol]
    vol = volume.at[signal_date, symbol] if symbol in volume.columns else float("nan")
    if pd.isna(price) or price <= 0 or pd.isna(vol) or vol <= 0:
        return None, None
    return float(price), float(vol)


def simulate_same_bar_close(
    signals: pd.DataFrame,
    direction: str,
    exit_rule: ExitRule,
    *,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    atr_panel: pd.DataFrame | None,
    target_value_per_trade: float,
    cfg: Config,
) -> list[Trade]:
    """Mirrors `engine.simulate.simulate_trades` exactly, except the entry fills
    at the SIGNAL bar's own close rather than the next bar's open. Everything
    else - one position per symbol via a busy_until cursor, stop-wins-a-tie,
    time exit at the next available open, cost model - is identical, so a
    difference between this and the production result isolates entry timing.
    """
    if direction != "long":
        raise ValueError("this diagnostic only implements the long side (matches "
                         "mean_reversion, the strategy under investigation)")
    calendar = close.index
    costs = CostModel.from_config(cfg)
    needs_stop = exit_rule.atr_stop_multiple is not None
    if needs_stop and atr_panel is None:
        raise ValueError("exit_rule requires a stop/target but no atr_panel was given")
    cap_pct = cfg.execution.max_participation_pct / 100.0

    trades: list[Trade] = []
    busy_until: dict[str, pd.Timestamp] = {}

    sig_rows, sig_cols = np.where(signals.to_numpy())
    order = np.lexsort((sig_cols, sig_rows))

    for k in order:
        d = calendar[sig_rows[k]]
        sym = signals.columns[sig_cols[k]]

        blocked = busy_until.get(sym)
        if blocked is not None and d < blocked:
            continue

        price, day_vol = _same_bar_close_fill(d, sym, close, volume, cfg)
        if price is None:
            trades.append(Trade(
                symbol=sym, signal_date=d, entry_date=d, entry_price=None,
                stop_price=None, target_price=None, exit_date=None, exit_price=None,
                exit_reason=EXIT_NO_FILL, held_days=0, shares=0,
                gross_pnl_pct=None, net_pnl_pct=None, cost_pct=None,
            ))
            continue

        req_shares = shares_for_value(target_value_per_trade, price)
        cap_shares = int(cap_pct * day_vol)
        shares = min(req_shares, max(cap_shares, 0))
        if shares <= 0:
            trades.append(Trade(
                symbol=sym, signal_date=d, entry_date=d, entry_price=None,
                stop_price=None, target_price=None, exit_date=None, exit_price=None,
                exit_reason=EXIT_NO_FILL, held_days=0, shares=0,
                gross_pnl_pct=None, net_pnl_pct=None, cost_pct=None,
            ))
            continue

        entry_date, entry_price = d, price

        stop_price = target_price = None
        if needs_stop:
            a = atr_panel.at[d, sym] if sym in atr_panel.columns else np.nan
            if pd.notna(a) and a > 0:
                risk = exit_rule.atr_stop_multiple * a
                stop_price = entry_price - risk
                target_price = entry_price + exit_rule.risk_reward * risk

        entry_idx = calendar.get_loc(entry_date)
        # First bar with REAL exposure is entry_idx + 1 - see module docstring.
        first_exposed_idx = entry_idx + 1
        last_hold_idx = min(entry_idx + exit_rule.max_hold_days, len(calendar) - 1)

        exit_date = exit_price = None
        exit_reason = EXIT_UNRESOLVED
        if first_exposed_idx > last_hold_idx:
            # max_hold_days too small to reach a single exposed bar - degenerate,
            # but handled rather than silently skipped.
            nxt = next_trading_day(calendar, entry_date)
            if nxt is not None and nxt in open_.index and sym in open_.columns:
                px = open_.at[nxt, sym]
                if pd.notna(px) and px > 0:
                    exit_date, exit_price, exit_reason = nxt, float(px), EXIT_TIME
        else:
            for i in range(first_exposed_idx, last_hold_idx + 1):
                bar_date = calendar[i]
                if sym not in high.columns:
                    break
                hi, lo = high.at[bar_date, sym], low.at[bar_date, sym]
                if pd.isna(hi) or pd.isna(lo):
                    continue
                stop_hit = stop_price is not None and lo <= stop_price
                target_hit = target_price is not None and hi >= target_price
                if stop_hit:
                    exit_date, exit_price, exit_reason = bar_date, stop_price, EXIT_STOP
                    break
                if target_hit:
                    exit_date, exit_price, exit_reason = bar_date, target_price, EXIT_TARGET
                    break
            else:
                nxt = next_trading_day(calendar, calendar[last_hold_idx])
                if nxt is not None and nxt in open_.index and sym in open_.columns:
                    px = open_.at[nxt, sym]
                    if pd.notna(px) and px > 0:
                        exit_date, exit_price, exit_reason = nxt, float(px), EXIT_TIME

        gross_pnl_pct = net_pnl_pct = cost_pct = held_days = None
        if exit_price is not None:
            held_days = calendar.get_loc(exit_date) - entry_idx
            gross_ret = exit_price / entry_price - 1.0
            rt = costs.round_trip(entry_price * shares, exit_price * shares)
            cost_pct = rt.pct_of_position
            gross_pnl_pct = gross_ret * 100.0
            net_pnl_pct = gross_pnl_pct - cost_pct

        trades.append(Trade(
            symbol=sym, signal_date=d, entry_date=entry_date, entry_price=entry_price,
            stop_price=stop_price, target_price=target_price,
            exit_date=exit_date, exit_price=exit_price, exit_reason=exit_reason,
            held_days=held_days if held_days is not None else 0, shares=shares,
            gross_pnl_pct=gross_pnl_pct, net_pnl_pct=net_pnl_pct, cost_pct=cost_pct,
        ))
        busy_until[sym] = exit_date if exit_date is not None else pd.Timestamp.max

    return trades
