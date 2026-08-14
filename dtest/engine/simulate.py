"""Trade-level simulator: entry signals -> costed, executable trades.

SIZING-INDEPENDENT BY DESIGN. This simulator reports per-trade percentages, not
rupees or portfolio weights - the predecessor project's rule, learned the hard
way: "sum of per-trade percentages is not a portfolio return" produced a
phantom 14.5M-rupee result in one of its own investigations. Sizing-dependent
questions (does this survive at Rs 50,000 with 5 concurrent slots?) are a
SEPARATE, later check in `engine/portfolio.py` - answered only after a signal
has already cleared the sizing-independent bar.

ONE POSITION PER SYMBOL AT A TIME, enforced with a `busy_until` cursor per
symbol rather than a flag set-and-cleared within one resolution step. Signals
are processed in chronological order and each is resolved fully (entry through
exit) before the next is considered, so a naive "mark busy, then clear it once
resolved" flag is a no-op - by the time a LATER signal for the same symbol is
reached in the loop, the earlier trade has already been fully resolved and its
flag already cleared, even though the earlier trade's real holding period may
still cover the later signal's date. `busy_until[symbol]` instead records the
actual exit date, and any signal dated strictly before it is skipped. A
position that never resolves within available data (`EXIT_UNRESOLVED`) blocks
that symbol permanently, matching that its real-world fate is unknown.

EXIT PRECEDENCE, checked in this order on every held bar:
  1. stop level (intrabar - a stop can be touched by LOW without closing there)
  2. target level (intrabar, same reasoning)
  3. max hold reached -> exit at the NEXT session's open (a time-based exit is
     still a fresh decision, so it waits for an open like an entry does)
  Both stop AND target touched on the same bar -> STOP WINS, by construction:
  a path that touches both is at least as likely to have hit the adverse level
  first, and assuming the favourable order every time is exactly the kind of
  optimism this project exists to remove.

STOP/TARGET FILL AT THE TOUCHED LEVEL (a resting order the exchange fills when
price reaches it), while ENTRY and a TIME exit fill at the next bar's OPEN (a
fresh decision that cannot execute until the market reopens). Mixing these two
conventions is deliberate: a resting order and a decision made from a close are
different mechanisms and are costed the same way but filled differently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dtest.config import Config
from dtest.engine.costs import CostModel
from dtest.engine.fills import next_trading_day, peek_fill_price, resolve_fill, shares_for_value

EXIT_STOP = "stop"
EXIT_TARGET = "target"
EXIT_TIME = "time"
EXIT_NO_FILL = "no_fill"          # entry signal never got filled at all
EXIT_UNRESOLVED = "unresolved"    # still open at the end of available data


@dataclass(frozen=True)
class Trade:
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp | None
    entry_price: float | None
    stop_price: float | None
    target_price: float | None
    exit_date: pd.Timestamp | None
    exit_price: float | None
    exit_reason: str
    held_days: int
    shares: int
    gross_pnl_pct: float | None
    net_pnl_pct: float | None
    cost_pct: float | None


@dataclass(frozen=True)
class ExitRule:
    """How a trade closes, once opened. `max_hold_days` counts trading SESSIONS
    the position may be held, inclusive of the entry day itself - so
    `max_hold_days=7` means the entry day plus 6 more sessions are eligible for
    a stop/target touch, and if still unresolved after the 7th session's close,
    the position exits at the 8th session's open.

    `atr_stop_multiple=None` disables stop/target entirely - a pure time exit.
    Providing one without the other is invalid: a target with no stop has no
    risk basis to size it from (`risk = entry - stop`), validated at
    construction rather than failing confusingly mid-simulation.
    """

    max_hold_days: int
    atr_window: int = 14
    atr_stop_multiple: float | None = None
    risk_reward: float | None = None

    def __post_init__(self):
        if self.max_hold_days <= 0:
            raise ValueError("max_hold_days must be positive")
        has_stop = self.atr_stop_multiple is not None
        has_target = self.risk_reward is not None
        if has_stop != has_target:
            raise ValueError(
                "atr_stop_multiple and risk_reward must both be set or both be "
                "None - a target needs a stop to define risk, and a stop with "
                "no target is a rule this simulator does not yet support"
            )


def _direction_sign(direction: str) -> int:
    if direction == "long":
        return 1
    if direction == "short":
        return -1
    raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")


def _no_fill_trade(sym, signal_date, fill_date) -> Trade:
    return Trade(
        symbol=sym, signal_date=signal_date, entry_date=fill_date, entry_price=None,
        stop_price=None, target_price=None, exit_date=None, exit_price=None,
        exit_reason=EXIT_NO_FILL, held_days=0, shares=0,
        gross_pnl_pct=None, net_pnl_pct=None, cost_pct=None,
    )


def simulate_trades(
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
    """Walk every signal to a resolved trade.

    `signals` is a (date x symbol) boolean panel: True means "a signal fired at
    this bar's CLOSE" - filled at the next bar's open, per `engine/fills.py`.
    `target_value_per_trade` is a fixed rupee sizing basis, equal for every
    trade, precisely so results stay sizing-independent - each trade is "what
    if exactly this much were risked", not a claim about a compounding account.

    `atr_panel` is required when `exit_rule` has a stop/target; it must be
    computed by the CALLER (`features.technical.atr`) and passed in, so a
    caller who already built it for a signal does not pay for it twice, and so
    the ATR window choice is visible at the call site.
    """
    sign = _direction_sign(direction)
    calendar = close.index
    costs = CostModel.from_config(cfg)
    needs_stop = exit_rule.atr_stop_multiple is not None
    if needs_stop and atr_panel is None:
        raise ValueError("exit_rule requires a stop/target but no atr_panel was given")

    trades: list[Trade] = []
    busy_until: dict[str, pd.Timestamp] = {}   # symbol -> date the position is confirmed clear from

    sig_rows, sig_cols = np.where(signals.to_numpy())
    order = np.lexsort((sig_cols, sig_rows))     # chronological, symbol tie-broken

    for k in order:
        d = calendar[sig_rows[k]]
        sym = signals.columns[sig_cols[k]]

        blocked_until = busy_until.get(sym)
        if blocked_until is not None and d < blocked_until:
            continue    # an earlier trade on this symbol is still open

        peek = peek_fill_price(d, sym, open_, calendar)
        if peek.rejected:
            trades.append(_no_fill_trade(sym, d, peek.fill_date))
            continue

        req_shares = shares_for_value(target_value_per_trade, peek.price)
        if req_shares <= 0:
            trades.append(_no_fill_trade(sym, d, peek.fill_date))
            continue

        entry_fill = resolve_fill(d, sym, "buy" if sign > 0 else "sell", req_shares,
                                  open_, volume, calendar, cfg)
        if entry_fill.rejected:
            trades.append(_no_fill_trade(sym, d, entry_fill.fill_date))
            continue

        entry_date = entry_fill.fill_date
        entry_price = entry_fill.fill_price
        shares = entry_fill.filled_shares

        stop_price = target_price = None
        if needs_stop:
            a = atr_panel.at[d, sym] if sym in atr_panel.columns else np.nan
            if pd.notna(a) and a > 0:
                risk = exit_rule.atr_stop_multiple * a
                stop_price = entry_price - sign * risk
                target_price = entry_price + sign * exit_rule.risk_reward * risk
            # else: no usable ATR at signal time - falls through to a pure
            # time exit rather than placing a zero-width stop that fires
            # instantly on the entry bar's own noise.

        entry_idx = calendar.get_loc(entry_date)
        last_hold_idx = min(entry_idx + exit_rule.max_hold_days - 1, len(calendar) - 1)

        exit_date = exit_price = None
        exit_reason = EXIT_UNRESOLVED
        for i in range(entry_idx, last_hold_idx + 1):
            bar_date = calendar[i]
            if sym not in high.columns:
                break
            hi, lo = high.at[bar_date, sym], low.at[bar_date, sym]
            if pd.isna(hi) or pd.isna(lo):
                continue   # symbol didn't trade this bar; hold through it

            stop_hit = stop_price is not None and (
                (sign > 0 and lo <= stop_price) or (sign < 0 and hi >= stop_price)
            )
            target_hit = target_price is not None and (
                (sign > 0 and hi >= target_price) or (sign < 0 and lo <= target_price)
            )
            if stop_hit:                      # stop wins a same-bar tie
                exit_date, exit_price, exit_reason = bar_date, stop_price, EXIT_STOP
                break
            if target_hit:
                exit_date, exit_price, exit_reason = bar_date, target_price, EXIT_TARGET
                break
        else:
            # Held through every eligible bar with no stop/target touch: exit
            # at the NEXT session's open, exactly like an entry decision.
            nxt = next_trading_day(calendar, calendar[last_hold_idx])
            if nxt is not None and nxt in open_.index and sym in open_.columns:
                px = open_.at[nxt, sym]
                if pd.notna(px) and px > 0:
                    exit_date, exit_price, exit_reason = nxt, float(px), EXIT_TIME

        gross_pnl_pct = net_pnl_pct = cost_pct = held_days = None
        if exit_price is not None:
            held_days = calendar.get_loc(exit_date) - entry_idx
            gross_ret = sign * (exit_price / entry_price - 1.0)
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

        # Block this symbol until the position is confirmed clear. An
        # unresolved trade (ran off the end of available data) blocks forever;
        # everything else clears at its own exit date.
        busy_until[sym] = exit_date if exit_date is not None else pd.Timestamp.max

    return trades


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    cols = list(Trade.__dataclass_fields__.keys())
    if not trades:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([vars(t) for t in trades])[cols]
