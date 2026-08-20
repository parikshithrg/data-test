"""Point-in-time fundamentals features, built on top of
`dtest.data.financial_results`'s per-symbol filing history.

STANDALONE (NON-CONSOLIDATED) ONLY, deliberately, not a default silently
picked. Both `dtest.data.financial_results.load_financials`'s
`consolidated` flag values exist for most modern filings, and picking
"whichever is more recent" or "prefer consolidated" would silently switch
which entity a stock's own earnings series describes partway through its
history (many companies only started filing consolidated results in the
mid-2010s - before that, standalone is the ONLY series that exists at
all). Standalone gives the LONGEST unbroken single-entity series for the
most symbols, and avoids a regime change (M&A/subsidiary consolidation
entering the numbers) contaminating a trailing-history-based surprise
measure. A consolidated-only variant is a legitimate, separately-scoped
alternative construction if this line is revisited - not built here.

YoY SURPRISE, NOT A CONSENSUS-BEATING SURPRISE - stated plainly, since this
is the standard PEAD/SUE construction's whole premise and there is no
analyst-consensus data anywhere in this project. `yoy_change` compares a
filing's own value against the SAME field 4 filings earlier (t-4) -
correct under the "strictly quarterly, one consolidation type" assumption
enforced by `point_in_time_series`'s own filtering, not a calendar-date
lookup. `sue_zscore` standardizes that YoY change against the stock's OWN
trailing history of such changes via `features.technical.rolling_zscore` -
reused, not reinvented, same "no new statistical machinery" precedent
`features/pairs.py` already established for this project.

CAUSALITY: every function here operates on a Series indexed by
`filing_date` (already the point-in-time key - see
`financial_results.py`'s own module docstring for why `period_end` must
never be used for this). `to_daily_panel` is the ONLY place ffill happens
- forward-fill ALONE is sufficient for causality here, no extra shift on
top (see that function's own docstring for why a filing event differs
from a price-derived rolling feature in this respect) - a caller must
never ffill a fundamentals series itself.
"""

from __future__ import annotations

import pandas as pd

from dtest.features.technical import rolling_zscore

YOY_LAG_FILINGS = 4
SUE_WINDOW = 8


def point_in_time_series(df: pd.DataFrame, field: str) -> pd.Series:
    """A Series indexed by `filing_date`, one row per STANDALONE
    (non-consolidated) quarterly filing, sorted, and de-duplicated by
    `period_end` keeping the LATEST `filing_date` for a repeated period (a
    later filing for the same quarter is a restatement/revision that
    supersedes the earlier one - never averaged, never the first one kept).
    Empty Series (not an error) if the symbol has no standalone data."""
    if df.empty:
        return pd.Series(dtype="float64", name=field)
    standalone = df[~df["consolidated"]].sort_values(
        ["period_end", "filing_date"], kind="stable")
    standalone = standalone.drop_duplicates(subset="period_end", keep="last")
    standalone = standalone.sort_values("filing_date", kind="stable")
    s = pd.Series(standalone[field].to_numpy(), index=standalone["filing_date"], name=field)
    return s[~s.index.duplicated(keep="last")]


def yoy_change(series: pd.Series, lag: int = YOY_LAG_FILINGS) -> pd.Series:
    """value[t] - value[t - lag] in FILING order (not calendar time) - lag=4
    is "same quarter last year" only under the strictly-quarterly,
    one-consolidation-type assumption `point_in_time_series` already
    enforces upstream."""
    return series - series.shift(lag)


def sue_zscore(series: pd.Series, window: int = SUE_WINDOW) -> pd.Series:
    """Standardized unexpected earnings: `yoy_change(series)`, z-scored
    against its own trailing `window` prior surprises (reuses
    `rolling_zscore` - no new statistical machinery)."""
    surprise = yoy_change(series)
    return rolling_zscore(surprise, window=window, min_periods=max(window // 2, 2))


def trailing_ttm(series: pd.Series, window: int = 4) -> pd.Series:
    """Trailing-twelve-month sum: the last `window` (default 4, one year of
    quarters) filing-indexed values summed - the standard TTM convention
    for a per-quarter flow figure like EPS, avoiding the seasonality a
    single quarter's own EPS would introduce into a P/E-style ratio.
    NaN until `window` real filings exist, same "no partial-window value"
    convention every rolling feature in this project uses."""
    return series.rolling(window, min_periods=window).sum()


def margin_ttm(df: pd.DataFrame, window: int = 4) -> pd.Series:
    """Trailing-twelve-month net profit margin: `trailing_ttm(net_profit) /
    trailing_ttm(revenue)`, both built from the SAME standalone filing rows
    (`point_in_time_series` on `net_profit` and `revenue` from the same
    `df` shares an identical filing_date index by construction - both
    fields come from the same deduplicated row set), so the two TTM sums
    are always over the SAME four quarters, never mismatched. TTM (not a
    single quarter's own margin) smooths quarter-to-quarter seasonality,
    matching the "multi-quarter trend, not one snapshot" framing this
    feature exists for. Masked to `revenue_ttm > 0` - a margin over zero
    or negative trailing revenue is not a meaningful ratio, screened out
    rather than inverted or clipped, same convention `signals/value.py`
    uses for a non-positive P/E denominator."""
    net_profit = point_in_time_series(df, "net_profit")
    revenue = point_in_time_series(df, "revenue")
    revenue_ttm = trailing_ttm(revenue, window=window)
    profit_ttm = trailing_ttm(net_profit, window=window)
    return (profit_ttm / revenue_ttm).where(revenue_ttm > 0)


def margin_trend_zscore(df: pd.DataFrame, ttm_window: int = 4, trend_lag: int = YOY_LAG_FILINGS,
                        zscore_window: int = SUE_WINDOW) -> pd.Series:
    """Quality/profitability-TREND feature: how much TTM net margin has
    changed versus `trend_lag` filings ago (default 4 - TTM-margin-now vs
    TTM-margin-a-year-ago, a genuinely multi-quarter comparison, not a
    single-quarter snapshot), z-scored against its own trailing history
    (reuses `rolling_zscore`, no new statistical machinery). A POSITIVE
    reading means margin is IMPROVING relative to its own recent trend -
    the caller decides the sign/threshold to act on (see
    `signals/quality.py`), this function only builds the feature."""
    margin = margin_ttm(df, window=ttm_window)
    trend = yoy_change(margin, lag=trend_lag)
    return rolling_zscore(trend, window=zscore_window, min_periods=max(zscore_window // 2, 2))


def to_daily_panel(per_symbol: dict[str, pd.Series], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex a dict of {symbol: filing-indexed Series} onto the full
    daily trading calendar via forward-fill (a value is known from its
    OWN filing_date onward, held constant until the next filing).
    Deliberately NO extra shift on top of that, unlike a price-derived
    rolling feature: a price panel's day-T bar only finishes existing at
    T's own close, so a feature built from it needs an explicit
    `.shift(1)`/prior-N-day convention to keep T's OWN bar out of what T's
    decision is allowed to see (see `vol_squeeze_breakout.py`'s
    `prior_high = close.shift(1)...`). A filing's `filing_date` is already
    a real, one-time disclosure EVENT, not a continuously-updating bar -
    ffill alone means day T's panel value can only ever come from a
    filing_date <= T, which is exactly the same "knowable by T's own
    close, fillable at T+1's open" convention every signal in this project
    already uses. No further shift is needed or correct here."""
    cols = {}
    for symbol, s in per_symbol.items():
        if s.empty:
            cols[symbol] = pd.Series(float("nan"), index=calendar)
            continue
        cols[symbol] = s.reindex(calendar.union(s.index)).ffill().reindex(calendar)
    return pd.DataFrame(cols, index=calendar)
