"""Account-level simulation: real capital, real slots, real drawdown.

`engine/simulate.py` is deliberately sizing-independent - every trade risks a
fixed rupee amount so its per-trade percentages are comparable regardless of
account size. That is the right tool for deciding whether a signal carries
information. It is the WRONG tool for the final question this project actually
cares about: does this survive as an account? The predecessor project's own
history shows why the two diverge - mean_reversion had the BEST per-trade
expectancy of its three systems and the WORST portfolio Sharpe (0.119, max
drawdown -46.8%), because its ~8 concurrent oversold positions were correlated
and crashed together. A sizing-independent check cannot see that; only a real
account simulation can.

So this module is a genuinely separate simulation, not a wrapper around
`simulate_trades`'s output - it walks the calendar day by day (not
signal by signal) because slot availability, cash, and sector exposure are all
STATE that only exists at the portfolio level and changes with every entry and
exit, in an order that matters:

  each day, in order:
    1. TIME exits scheduled for today's open (yesterday was the last hold day)
    2. NEW entries scheduled for today's open (signal fired at yesterday's
       close), sized to equal-weight of CURRENT equity, gated on an open slot,
       available cash, and the sector cap
    3. STOP/TARGET checks against today's high/low for every open position,
       including ones that just entered today
    4. mark-to-market equity recorded at today's close

Steps 1 and 2 both execute "at today's open" because a daily-bar simulation
has no finer time resolution to separate them - a position closing at the same
moment a new one opens is a modelling simplification, stated rather than
hidden. Every fill still goes through `engine.fills.resolve_fill`, so the
participation cap applies here exactly as it does in the trade-level simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dtest.config import Config
from dtest.engine.costs import CostModel
from dtest.engine.fills import resolve_fill, shares_for_value
from dtest.engine.simulate import EXIT_STOP, EXIT_TARGET, EXIT_TIME, EXIT_UNRESOLVED, ExitRule

TRADING_DAYS_PER_YEAR = 252   # explicit, never inferred - see module docstring
                              # on the predecessor's periods-per-year bug class


@dataclass
class _OpenPosition:
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float | None
    target_price: float | None
    max_hold_last_idx: int
    sector: str


@dataclass(frozen=True)
class PortfolioTrade:
    symbol: str
    sector: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float | None
    target_price: float | None
    exit_date: pd.Timestamp | None
    exit_price: float | None
    exit_reason: str
    held_days: int | None
    gross_pnl_pct: float | None
    net_pnl_pct: float | None
    cost_rupees: float | None


@dataclass(frozen=True)
class PortfolioResult:
    equity_curve: pd.DataFrame     # date, cash, positions_value, equity, n_open
    trades: pd.DataFrame
    skipped_no_slot: int
    skipped_sector_cap: int
    skipped_no_cash: int
    skipped_no_fill: int

    def metrics(self) -> dict:
        return portfolio_metrics(self.equity_curve.set_index("date")["equity"])


def portfolio_metrics(equity: pd.Series) -> dict:
    """CAGR, Sharpe, max drawdown from a daily equity curve.

    `TRADING_DAYS_PER_YEAR = 252` is a fixed constant, never inferred from bar
    spacing - the predecessor project had a shared `core/metrics.py` that
    inferred periods-per-year from an intraday bar-counting formula applied
    unconditionally, silently understating Sharpe by ~2.3x for every daily
    curve in that codebase until it was caught. Fixed by never inferring it.
    """
    equity = equity.dropna()
    if len(equity) < 2:
        return {"cagr_pct": float("nan"), "sharpe": float("nan"),
               "max_drawdown_pct": float("nan"), "n_days": len(equity)}

    n_days = len(equity)
    total_return = equity.iloc[-1] / equity.iloc[0]
    cagr = total_return ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0

    daily_ret = equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
             if daily_ret.std(ddof=1) > 0 else float("nan"))

    drawdown = equity / equity.cummax() - 1.0
    return {
        "cagr_pct": float(cagr * 100.0),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "n_days": n_days,
    }


def benchmark_equity_curve(benchmark_close: pd.Series, initial_capital: float,
                           calendar: pd.DatetimeIndex) -> pd.Series:
    """A buy-and-hold curve on the SAME capital and SAME dates, for a like-for-like
    comparison via `portfolio_metrics`. No costs applied - a single buy-and-hold
    entry's cost is negligible against a multi-year curve and immaterial to the
    comparison; stated rather than silently assumed away."""
    aligned = benchmark_close.reindex(calendar).ffill()
    first_valid = aligned.dropna().iloc[0]
    return initial_capital * aligned / first_valid


def run_portfolio(
    signals: pd.DataFrame,
    direction: str,
    exit_rule: ExitRule,
    sector_map: dict[str, str],
    *,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    atr_panel: pd.DataFrame | None,
    cfg: Config,
) -> PortfolioResult:
    """Walk the calendar day by day, running a real account against `signals`.

    `sector_map` is symbol -> sector label; a symbol absent from the map is
    treated as its own singleton sector "Unknown:<symbol>" rather than lumped
    into one giant "Unknown" bucket - the latter would let unmapped symbols
    silently exempt each other from the sector cap by sharing a fake sector.
    """
    if cfg.portfolio.position_sizing != "equal_weight":
        raise ValueError(f"unsupported position_sizing: {cfg.portfolio.position_sizing!r}")
    if direction != "long":
        # This account is cash-equity DELIVERY, which cannot carry a short
        # overnight - the predecessor project's own reasoning for making
        # long-only the default. `simulate.py` (sizing-independent, informational
        # only) may still explore a short's per-trade characteristics, but this
        # module claims to model a REAL account, and a short cash position has no
        # real settlement mechanics here to model - no margin, no borrow cost, no
        # uncapped-loss cash accounting. Pretending otherwise would be worse than
        # refusing.
        raise ValueError(
            f"direction={direction!r} is not supported at the portfolio level: "
            "cash-equity delivery cannot short. Use simulate.py for a "
            "sizing-independent look at a hypothetical short's per-trade behaviour."
        )
    calendar = close.index
    costs = CostModel.from_config(cfg)
    p = cfg.portfolio
    needs_stop = exit_rule.atr_stop_multiple is not None
    if needs_stop and atr_panel is None:
        raise ValueError("exit_rule requires a stop/target but no atr_panel was given")

    cash = p.initial_capital
    positions: dict[str, _OpenPosition] = {}
    trades: list[PortfolioTrade] = []
    equity_rows = []
    pending_time_exits: dict[int, list[str]] = {}
    skipped = {"no_slot": 0, "sector_cap": 0, "no_cash": 0, "no_fill": 0}

    sig_by_date: dict[pd.Timestamp, list[str]] = {
        d: signals.columns[signals.loc[d].to_numpy()].tolist()
        for d in signals.index[signals.any(axis=1)]
    }

    def sector_of(sym: str) -> str:
        return sector_map.get(sym, f"Unknown:{sym}")

    def close_position(sym: str, exit_date, exit_price: float, reason: str):
        pos = positions.pop(sym)
        rt = costs.round_trip(pos.entry_price * pos.shares, exit_price * pos.shares)
        gross_pct = (exit_price / pos.entry_price - 1.0) * 100.0     # long only, see guard above
        net_pct = gross_pct - rt.pct_of_position
        nonlocal cash
        cash += pos.shares * exit_price - rt.total
        trades.append(PortfolioTrade(
            symbol=sym, sector=pos.sector, signal_date=pos.signal_date,
            entry_date=pos.entry_date, entry_price=pos.entry_price, shares=pos.shares,
            stop_price=pos.stop_price, target_price=pos.target_price,
            exit_date=exit_date, exit_price=exit_price, exit_reason=reason,
            held_days=calendar.get_loc(exit_date) - calendar.get_loc(pos.entry_date),
            gross_pnl_pct=gross_pct, net_pnl_pct=net_pct, cost_rupees=rt.total,
        ))

    for i, d in enumerate(calendar):
        # 1. time exits due today
        for sym in pending_time_exits.pop(i, []):
            if sym not in positions:
                continue
            price = open_.at[d, sym] if sym in open_.columns else np.nan
            if pd.isna(price) or price <= 0:
                continue   # cannot exit today; remains open, will be caught as unresolved
            close_position(sym, d, float(price), EXIT_TIME)

        # 2. new entries: signals from yesterday's close, filled at today's open
        if i > 0:
            prev = calendar[i - 1]
            for sym in sig_by_date.get(prev, []):
                if sym in positions:
                    continue
                if len(positions) >= p.max_positions:
                    skipped["no_slot"] += 1
                    continue

                positions_value = sum(
                    pos.shares * (close.at[prev, s] if prev in close.index and s in close.columns
                                 and pd.notna(close.at[prev, s]) else pos.entry_price)
                    for s, pos in positions.items()
                )
                equity_now = cash + positions_value
                target_value = equity_now / p.max_positions

                sector = sector_of(sym)
                sector_value = sum(
                    pos.shares * (close.at[prev, s] if prev in close.index and s in close.columns
                                 and pd.notna(close.at[prev, s]) else pos.entry_price)
                    for s, pos in positions.items() if pos.sector == sector
                )
                if equity_now > 0 and (sector_value + target_value) / equity_now * 100.0 > p.max_sector_weight_pct:
                    skipped["sector_cap"] += 1
                    continue

                price = open_.at[d, sym] if sym in open_.columns else np.nan
                if pd.isna(price) or price <= 0:
                    skipped["no_fill"] += 1
                    continue
                req_shares = shares_for_value(min(target_value, cash), float(price))
                if req_shares <= 0:
                    skipped["no_cash"] += 1
                    continue

                fill = resolve_fill(prev, sym, "buy", req_shares, open_, volume, calendar, cfg)
                if fill.rejected:
                    skipped["no_fill"] += 1
                    continue

                stop_price = target_price = None
                if needs_stop:
                    a = atr_panel.at[prev, sym] if sym in atr_panel.columns else np.nan
                    if pd.notna(a) and a > 0:
                        risk = exit_rule.atr_stop_multiple * a
                        stop_price = fill.fill_price - risk
                        target_price = fill.fill_price + exit_rule.risk_reward * risk

                cash -= fill.value
                entry_idx = i
                last_hold_idx = min(entry_idx + exit_rule.max_hold_days - 1, len(calendar) - 1)
                positions[sym] = _OpenPosition(
                    symbol=sym, signal_date=prev, entry_date=d, entry_price=fill.fill_price,
                    shares=fill.filled_shares, stop_price=stop_price, target_price=target_price,
                    max_hold_last_idx=last_hold_idx, sector=sector,
                )
                if last_hold_idx + 1 < len(calendar):
                    pending_time_exits.setdefault(last_hold_idx + 1, []).append(sym)

        # 3. stop/target checks against today's range, including today's entries
        for sym in list(positions.keys()):
            pos = positions[sym]
            if sym not in high.columns or sym not in low.columns:
                continue
            hi, lo = high.at[d, sym], low.at[d, sym]
            if pd.isna(hi) or pd.isna(lo):
                continue
            stop_hit = pos.stop_price is not None and lo <= pos.stop_price
            target_hit = pos.target_price is not None and hi >= pos.target_price
            if stop_hit:
                close_position(sym, d, pos.stop_price, EXIT_STOP)
            elif target_hit:
                close_position(sym, d, pos.target_price, EXIT_TARGET)

        # 4. mark-to-market at today's close
        mtm = sum(
            pos.shares * (close.at[d, s] if s in close.columns and pd.notna(close.at[d, s])
                         else pos.entry_price)
            for s, pos in positions.items()
        )
        equity_rows.append({"date": d, "cash": cash, "positions_value": mtm,
                            "equity": cash + mtm, "n_open": len(positions)})

    for sym, pos in list(positions.items()):
        trades.append(PortfolioTrade(
            symbol=sym, sector=pos.sector, signal_date=pos.signal_date,
            entry_date=pos.entry_date, entry_price=pos.entry_price, shares=pos.shares,
            stop_price=pos.stop_price, target_price=pos.target_price,
            exit_date=None, exit_price=None, exit_reason=EXIT_UNRESOLVED,
            held_days=None, gross_pnl_pct=None, net_pnl_pct=None, cost_rupees=None,
        ))

    trade_cols = list(PortfolioTrade.__dataclass_fields__.keys())
    trades_df = (pd.DataFrame([vars(t) for t in trades])[trade_cols]
                if trades else pd.DataFrame(columns=trade_cols))

    return PortfolioResult(
        equity_curve=pd.DataFrame(equity_rows),
        trades=trades_df,
        skipped_no_slot=skipped["no_slot"],
        skipped_sector_cap=skipped["sector_cap"],
        skipped_no_cash=skipped["no_cash"],
        skipped_no_fill=skipped["no_fill"],
    )
