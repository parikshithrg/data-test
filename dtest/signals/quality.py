"""Quality / profitability trend: buy a name whose trailing net profit
margin has genuinely IMPROVED relative to its own recent trend - not
whether the margin is high or low in absolute terms, whether it is getting
BETTER.

WHO IS ON THE OTHER SIDE, AND WHY THEY'RE WRONG (the mandatory story): a
company whose margin is expanding faster than its own recent normal is
disproportionately doing something structurally better - pricing power,
cost discipline, an improving mix - and that operational improvement
takes the market time to fully re-rate into the price, the same
"information diffuses gradually" logic `momentum.py` uses for price
trends and `earnings_surprise.py` uses for a one-quarter EPS surprise,
here applied to a MULTI-QUARTER fundamental trend instead of either a
price series or a single-quarter number. The trader fading a genuine
margin acceleration on no more information than "it already re-rated
some" is betting the improvement was already fully priced with nothing
specific behind that view.

STRUCTURALLY DIFFERENT FROM THE OTHER TWO FUNDAMENTALS SIGNALS in this
project, not a re-labelled variant of either: `earnings_surprise.py`
reacts to a single quarter's EPS surprise (a discrete EVENT, fires once
per filing); `value.py` reacts to the P/E RATIO's level relative to its
own history (dominated by daily PRICE movement, since EPS updates only
quarterly - flagged there as a real overlap risk with mean_reversion).
This signal uses NO price data in its trigger at all - `margin_trend_
zscore` (`features/fundamentals.py`) is built entirely from TTM revenue
and TTM net profit, comparing this year's trailing margin to last year's -
a pure fundamentals-trend claim, immune to the price-mechanically-drives-
the-ratio confound `value.py` has to flag.

FIRES ONCE PER FILING, AS A SHARP EVENT, same convention `earnings_
surprise.py` established and for the identical reason: a filing is a
one-time disclosure, and re-firing on every day the ffill'd trend stays
elevated would re-enter into an increasingly stale, no-longer-fresh
reading of "did margin just improve."
"""

from __future__ import annotations

import pandas as pd

TREND_THRESHOLD = 1.0


def quality_signal(
    trend_per_symbol: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
    threshold: float = TREND_THRESHOLD,
) -> pd.DataFrame:
    """True on exactly one trading day per qualifying filing (margin_trend_
    zscore > `threshold`): the filing's own date if it is a trading day,
    else the next trading day after it - identical mapping convention
    `earnings_surprise.py` uses, for the identical point-in-time reason.
    """
    symbols = list(trend_per_symbol.keys())
    signal = pd.DataFrame(False, index=calendar, columns=symbols)
    for symbol, s in trend_per_symbol.items():
        fires = s[s > threshold]
        for filing_date in fires.index:
            day = pd.Timestamp(filing_date).normalize()
            pos = calendar.searchsorted(day)
            if pos >= len(calendar):
                continue
            signal.loc[calendar[pos], symbol] = True
    return signal
