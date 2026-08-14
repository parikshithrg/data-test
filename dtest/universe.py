"""Point-in-time tradeable universe. The fix for the predecessor's survivorship bias.

The old project ranked stocks out of TODAY's Nifty 500 back to 2004, so a name
that delisted, merged, or fell out of the index simply never existed in its
2004-2010 backtests. Measured on the one window where a comparison is possible:
22.5% of symbols trading before 2020 were gone by 2026.

So the universe here is not a list of names. It is a RULE - re-run at every
rebalance date using only data available as of that date - which makes it
point-in-time correct by construction rather than by record-keeping. It will
disagree with "the Nifty 500" on any given day, and that is the point: this
universe can be recomputed standing at 2009 without knowing what happens after.

REBALANCE MECHANICS, and why each piece exists:

  Monthly, on the last trading day of the month. A rebalance decided at close
  of date T uses data THROUGH T and takes effect starting the next trading
  session - it never reaches back to reprice a day it was decided on.

  Ranked on trailing TURNOVER (the exchange-published rupee value traded), not
  price or volume alone - turnover is what determines whether a real fill is
  possible at this account's size.

  BANDED (buffer_size > size): an incumbent stays in the universe while it
  ranks anywhere inside the buffer; only a NEW name has to clear the tighter
  `size` cutoff to get in. Without banding, a stock oscillating around rank 200
  enters and exits every month, each flip a real 0.32% round-trip cost for
  nothing. This is a liquidity-driven turnover cost, not a modelling nicety.

  Eligibility gates (independent of rank): enough history to compute anything
  about the stock (`min_history_days`), traded recently enough to be alive
  (`max_staleness_days`), positive turnover (a halted or dead-quote name is not
  tradeable regardless of what NaN-handling would otherwise let through), and
  priced high enough that a fixed-percentage cost model still means something
  (`min_price` - the old project had 45 symbols under Rs 5, where a single tick
  is a meaningful fraction of the "return").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dtest.config import Config


@dataclass(frozen=True)
class UniverseResult:
    """The point-in-time universe, plus the audit trail for how it was built."""

    membership: pd.DataFrame      # date x symbol, bool - forward-filled daily
    rebalance_dates: list[pd.Timestamp]
    rank: pd.DataFrame            # date x symbol, turnover rank AT rebalance dates only
    log: pd.DataFrame             # one row per rebalance: date, n_selected, n_entries, n_exits

    def as_of(self, date) -> list[str]:
        d = pd.Timestamp(date)
        if d not in self.membership.index:
            raise KeyError(f"{d} is not in the membership index")
        row = self.membership.loc[d]
        return sorted(row.index[row].tolist())

    def size_series(self) -> pd.Series:
        return self.membership.sum(axis=1)


def _month_end_dates(calendar: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each calendar month present in the index."""
    s = pd.Series(calendar, index=calendar)
    return sorted(s.groupby([calendar.year, calendar.month]).max().tolist())


def _last_traded_date(live: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """For each (date, symbol): the most recent date <= that date the symbol traded.

    Built per-column (`idx.where(col).ffill()`) rather than by tiling the
    calendar into a full-size DataFrame - the tiled version is a needless
    date x symbol array of an 8-byte datetime purely to do a subtraction, which
    at ~5,500 x ~1,800 is tens of millions of cells for no reason.
    """
    idx = pd.Series(calendar, index=calendar)
    return live.apply(lambda col: idx.where(col).ffill())


def _rank_by_turnover(trailing: pd.Series) -> pd.Index:
    """Deterministic rank order: turnover descending, symbol name as tie-break.

    Exact float ties are rare but not impossible (e.g. two halted names both at
    zero, or a data artifact), and `Series.sort_values`' default quicksort is
    not stable - without an explicit tie-break, rank assignment for tied names
    could vary run to run, silently breaking the determinism guarantee this
    project is built on.
    """
    tmp = pd.DataFrame({"turnover": trailing.to_numpy(), "symbol": trailing.index})
    tmp = tmp.sort_values(["turnover", "symbol"], ascending=[False, True], kind="stable")
    return pd.Index(tmp["symbol"].to_numpy())


def build_universe(close: pd.DataFrame, turnover: pd.DataFrame, cfg: Config) -> UniverseResult:
    """Build the point-in-time universe from raw price/turnover panels.

    `close` and `turnover` must share an index and columns. Both are used
    RAW, not corporate-action-adjusted - eligibility (min_price, staleness) is a
    question about the tradeable instrument today, not about its long-run
    return series.
    """
    u = cfg.universe
    u.validate()
    calendar = close.index
    rebalances = _month_end_dates(calendar)

    live = close.notna()
    last_seen = _last_traded_date(live, calendar)
    idx = pd.Series(calendar, index=calendar)
    staleness_days = last_seen.apply(lambda col: (idx - col).dt.days)
    history_days = live.cumsum()

    trailing_turnover = turnover.rolling(
        u.lookback_days, min_periods=max(u.lookback_days // 2, 1)
    ).median()

    membership = pd.DataFrame(False, index=calendar, columns=close.columns)
    rank_frame = pd.DataFrame(np.nan, index=calendar, columns=close.columns)
    log_rows = []
    incumbents: set[str] = set()

    for i, d in enumerate(rebalances):
        eligible = (
            (history_days.loc[d] >= u.min_history_days)
            & (staleness_days.loc[d] <= u.max_staleness_days)
            & (close.loc[d] >= u.min_price)
            & trailing_turnover.loc[d].notna()
            & (trailing_turnover.loc[d] > 0)
        )
        pool = trailing_turnover.loc[d].where(eligible).dropna()
        ranked_index = _rank_by_turnover(pool)
        rank_frame.loc[d, ranked_index] = np.arange(1, len(ranked_index) + 1)

        top_size = set(ranked_index[: u.size])
        top_buffer = set(ranked_index[: u.buffer_size])

        selected = top_size | (incumbents & top_buffer)
        if len(selected) > u.buffer_size:
            # Truncate deterministically: highest-ranked members survive, using
            # the same tie-broken order the rank itself was built from.
            ordered = [s for s in ranked_index if s in selected]
            selected = set(ordered[: u.buffer_size])

        entries = selected - incumbents
        exits = incumbents - selected
        log_rows.append({
            "date": d, "n_selected": len(selected),
            "n_entries": len(entries), "n_exits": len(exits),
            "n_eligible_pool": len(ranked_index),
        })

        start, end = d, (rebalances[i + 1] if i + 1 < len(rebalances) else calendar[-1])
        if i == 0:
            # Seed the very first window inclusive of its own start date, so
            # day one of the calendar has a universe. Every later rebalance
            # takes effect strictly AFTER the date it was decided on.
            window = calendar[(calendar >= start) & (calendar <= end)]
        elif i + 1 < len(rebalances):
            window = calendar[(calendar > start) & (calendar <= end)]
        else:
            window = calendar[calendar > start]

        if selected:
            membership.loc[window, sorted(selected)] = True

        incumbents = selected

    return UniverseResult(
        membership=membership, rebalance_dates=rebalances,
        rank=rank_frame, log=pd.DataFrame(log_rows),
    )
