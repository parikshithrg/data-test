"""Corporate actions, using the exchange's `prev_close` as a SIGNAL - not as a
return denominator.

WHAT WAS MEASURED, AND WHY THE OBVIOUS DESIGN IS WRONG.

NSE's bhavcopy publishes a `prev_close` for every symbol every day, and it is
restated when a split or bonus takes effect. The tempting design - which this
module originally used - is to treat `close[t] / prev_close[t] - 1` as the true
daily return, correct across actions by construction.

Measured against real data, that design breaks. On 2004-04-19 only **32 of 730**
symbols had `prev_close` equal to the prior session's close; 698 did not. The
next session (2004-04-20) was perfectly consistent again. Whatever the cause -
settlement-calendar quirks, special sessions, archive irregularities in the
mid-2000s - the empirical fact is that `prev_close` sometimes refers to a
session other than the immediately preceding one. Using it as the return
denominator on those dates silently produces a multi-day return labelled as a
one-day return, for most of the market at once.

THE SEPARATION IS CLEAN, so the fix is not a compromise:

  real actions (2024-06-24, verified against known events):
      ONGC 0.329 (1:3), WIPRO 0.655, BPCL 0.670, BIOCON 0.679 (1:1.5),
      ELECTCAST 0.507, KOTHARIPRO 0.505 (1:2)          -> all <= 0.68
  artefacts (2004-04-19, 572 symbols at once):
      1.0176, 0.9934, 1.0507, 1.0050, 0.9926, 1.0362   -> all within 5% of 1

A 10% threshold separates them completely, and a second guard catches the rest:
**a real corporate action affects a handful of symbols on a given day. A date
where 78% of the market "has an action" is a data artifact.**

SO THE RULE IS:
  * returns are computed against the previous TRADED close (contiguity-correct)
  * except on credible action days, where the exchange's restated `prev_close`
    supplies the adjustment factor (action-correct)
  * "credible" means a large factor AND an isolated event, both measured

Neither failure mode survives: a split does not read as a crash, and a
settlement quirk does not manufacture 572 phantom actions.

NOT COVERED, stated rather than discovered later: ordinary cash dividends. NSE
does not restate `prev_close` for a normal dividend, so this is a PRICE series,
not a total-return series. Indian yields (~1-1.5%) make that a modest,
one-directional understatement of long-horizon returns - but an understatement,
and any multi-year compounding claim must say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A real split/bonus moves the reference price by far more than this. Set from
# the measured separation above: real actions <= 0.68, artefacts within 5% of 1.
MIN_ACTION_MOVE = 0.10

# If more than this share of the day's trading symbols flag at once, the date is
# a data artifact, not simultaneous corporate actions across the whole market.
MAX_ACTION_SHARE_PER_DAY = 0.05

# Two thresholds, doing two different jobs - this distinction is the whole fix.
#
# SENSITIVE (below): "is this date's prev_close column trustworthy at all?"
# Any disagreement beyond publication rounding counts. On 2004-04-19, 698 of 730
# symbols disagreed with the previous traded close - most by only a few percent,
# so a 10% test saw only ~100 of them and the date passed the mass-event guard
# while still emitting ~100 phantom actions. Measuring disagreement sensitively
# catches the date outright.
#
# SPECIFIC (MIN_ACTION_MOVE above): "is THIS symbol's restatement a real action?"
# Only applied on dates that passed the sensitive test.
DISAGREEMENT_TOLERANCE = 0.005      # beyond 2-decimal publication rounding
MAX_DISAGREEMENT_SHARE = 0.20       # above this, the date's prev_close is unusable

# Reporting labels only. The factor always comes from the exchange; it is never
# snapped to one of these.
COMMON_FACTORS = {
    0.5: "1:1 bonus or 1:2 split", 0.2: "1:5 split", 0.1: "1:10 split",
    0.25: "1:4 split", 1/3: "1:3 split", 2/3: "2:3 (1:1.5)", 0.6: "3:5",
    0.05: "1:20 split", 0.02: "1:50 split", 0.01: "1:100 split",
}


@dataclass(frozen=True)
class ActionReport:
    actions: pd.DataFrame          # credible, isolated actions
    rejected_dates: pd.DataFrame   # dates rejected by the mass-event guard
    n_symbols: int

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def by_year(self) -> pd.Series:
        if self.actions.empty:
            return pd.Series(dtype=int)
        return self.actions.groupby(self.actions["date"].dt.year).size()

    def label(self) -> pd.DataFrame:
        if self.actions.empty:
            return self.actions
        out = self.actions.copy()

        def _lbl(f: float) -> str:
            for ref, name in COMMON_FACTORS.items():
                if abs(f - ref) <= 0.03 * ref:
                    return name
            return "other"

        out["likely"] = out["factor"].map(_lbl)
        return out


def previous_traded_close(close: pd.DataFrame) -> pd.DataFrame:
    """Each symbol's own last traded close strictly before each date.

    `ffill().shift(1)` rather than `shift(1)`: a symbol that did not trade
    yesterday must compare against the last day it actually traded, which is
    also what the exchange's own `prev_close` refers to. A plain shift would
    yield NaN and silently drop the observation.
    """
    return close.ffill().shift(1)


def detect_actions(
    close: pd.DataFrame,
    prev_close: pd.DataFrame,
    *,
    min_move: float = MIN_ACTION_MOVE,
    max_share_per_day: float = MAX_ACTION_SHARE_PER_DAY,
    disagreement_tolerance: float = DISAGREEMENT_TOLERANCE,
    max_disagreement_share: float = MAX_DISAGREEMENT_SHARE,
) -> ActionReport:
    """Find credible, isolated corporate actions.

    Three guards, each measured into existence rather than assumed:
      1. the date's `prev_close` column must be broadly consistent with the
         previous traded close (SENSITIVE test - catches settlement quirks)
      2. the restatement must be large (`min_move`, SPECIFIC test)
      3. the date must not still be a mass event (`max_share_per_day`)
    """
    prior = previous_traded_close(close)
    valid = prior.notna() & prev_close.notna() & (prior > 0)
    ratio = (prev_close / prior).where(valid)

    traded = close.notna().sum(axis=1).replace(0, np.nan)

    # Guard 1 - is this date's prev_close usable at all?
    disagree = valid & ((ratio - 1.0).abs() > disagreement_tolerance)
    disagree_share = disagree.sum(axis=1) / traded
    untrusted = (disagree_share > max_disagreement_share).fillna(False)

    # Guard 2 - specific enough to be a real action.
    big = valid & ((ratio - 1.0).abs() > min_move)

    # Guard 3 - even on a broadly-consistent date, a whole-market flag is data.
    mass = ((big.sum(axis=1) / traded) > max_share_per_day).fillna(False)

    bad = untrusted | mass
    rejected = pd.DataFrame({
        "date": close.index[bad],
        "flagged": big.sum(axis=1)[bad].to_numpy(),
        "disagreeing": disagree.sum(axis=1)[bad].to_numpy(),
        "traded": traded[bad].to_numpy(),
        "disagree_share": disagree_share[bad].to_numpy(),
        "reason": np.where(untrusted[bad].to_numpy(), "prev_close untrusted", "mass event"),
    })

    keep = big & ~bad.to_numpy()[:, None]
    idx = np.argwhere(keep.to_numpy())
    rows = [{
        "date": close.index[i],
        "symbol": close.columns[j],
        "factor": float(ratio.iat[i, j]),
        "prior_close": float(prior.iat[i, j]),
        "prev_close": float(prev_close.iat[i, j]),
    } for i, j in idx]

    actions = pd.DataFrame(rows, columns=["date", "symbol", "factor",
                                          "prior_close", "prev_close"])
    if not actions.empty:
        actions = actions.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
    return ActionReport(actions=actions, rejected_dates=rejected, n_symbols=close.shape[1])


def adjustment_factors(close: pd.DataFrame, prev_close: pd.DataFrame,
                       report: ActionReport | None = None) -> pd.DataFrame:
    """A (date x symbol) frame of 1.0 everywhere except on credible action days.

    Kept as its own artifact so a backtest can be re-run with actions on or off
    and the difference attributed, rather than the correction being buried
    inside a return calculation.
    """
    rep = report or detect_actions(close, prev_close)
    factors = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    for _, a in rep.actions.iterrows():
        factors.at[a["date"], a["symbol"]] = a["factor"]
    return factors


def daily_returns(close: pd.DataFrame, prev_close: pd.DataFrame,
                  report: ActionReport | None = None) -> pd.DataFrame:
    """THE way returns are computed in this project.

        r[t] = close[t] / (previous_traded_close[t] * factor[t]) - 1

    `factor` is 1.0 on ordinary days, so this reduces to a plain close-to-close
    return; on a credible action day it rescales the reference price by the
    exchange's own restatement. Never divides two raw closes across an action,
    and never treats a settlement quirk as a return.
    """
    prior = previous_traded_close(close)
    factors = adjustment_factors(close, prev_close, report)
    reference = prior * factors
    r = close / reference - 1.0
    return r.where(reference.gt(0) & close.notna())


def adjusted_close(close: pd.DataFrame, prev_close: pd.DataFrame,
                   report: ActionReport | None = None) -> pd.DataFrame:
    """Continuous back-adjusted price series, anchored to the LAST real close.

    Built by compounding action-correct returns, never by dividing raw prices
    across an action. Anchoring at the end keeps recent prices equal to the real,
    tradeable ones and pushes accumulated adjustment into the distant past.

    This inherits the precision of the RETURNS, not of the stored prices, so it
    does not reproduce the predecessor's rounding collapse - a 2004 price becomes
    a compounded float64 quantity rather than a number rounded to two decimals.
    """
    r = daily_returns(close, prev_close, report)
    traded = close.notna()
    growth = (1.0 + r.fillna(0.0)).cumprod().where(traded)

    out = {}
    for sym in close.columns:
        c = close[sym].dropna()
        g = growth[sym].dropna()
        if c.empty or g.empty:
            out[sym] = pd.Series(np.nan, index=close.index)
            continue
        anchor = c.index[-1]
        scale = c.iloc[-1] / g.loc[anchor]
        out[sym] = (growth[sym] * scale).where(traded[sym])
    return pd.DataFrame(out, index=close.index).sort_index(axis=1)


def verify(close: pd.DataFrame, prev_close: pd.DataFrame) -> pd.DataFrame:
    """Evidence table for the whole approach. Run it; do not trust the docstring.

    The decisive row is the last: on credible action days the RAW close-to-close
    move is huge and the corrected return is ordinary. If those two are similar,
    the correction is doing nothing and something is wrong.
    """
    rep = detect_actions(close, prev_close)
    prior = previous_traded_close(close)
    valid = prior.notna() & prev_close.notna() & (prior > 0)

    raw = (close / prior - 1.0).where(valid)
    adj = daily_returns(close, prev_close, rep).where(valid)

    if rep.n_actions:
        mask = np.zeros(close.shape, dtype=bool)
        ri = {d: i for i, d in enumerate(close.index)}
        ci = {c: j for j, c in enumerate(close.columns)}
        for _, a in rep.actions.iterrows():
            mask[ri[a["date"]], ci[a["symbol"]]] = True
        raw_on = np.abs(raw.to_numpy()[mask])
        adj_on = np.abs(adj.to_numpy()[mask])
    else:
        raw_on = adj_on = np.array([np.nan])

    return pd.DataFrame([
        {"metric": "valid (date, symbol) pairs", "value": int(valid.to_numpy().sum())},
        {"metric": "credible actions detected", "value": rep.n_actions},
        {"metric": "dates rejected as mass events", "value": len(rep.rejected_dates)},
        {"metric": "median |raw return| on action days %",
         "value": round(float(np.nanmedian(raw_on)) * 100, 2)},
        {"metric": "median |corrected return| on action days %",
         "value": round(float(np.nanmedian(adj_on)) * 100, 2)},
    ])
