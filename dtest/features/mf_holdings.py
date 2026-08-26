"""Point-in-time mutual-fund-holdings features, built on top of the real
per-stock equity holdings `dtest.data.amc_holdings`/`scripts/
build_amc_equity_holdings.py` already extracted from SBI and Axis's own
monthly portfolio disclosures (`data/amc_portfolios/equity_holdings.csv`).

SCOPE IS TWO FUND HOUSES, NOT MARKET-WIDE MF FLOW - stated here again,
not just in the data-layer docstrings, since every function below
inherits this limitation directly: `total_quantity` is the sum across
every Axis and SBI scheme's own disclosed holding of a stock, NOT the
mutual fund industry's aggregate position (~50 AMCs exist; 2 are
covered). A real, accepted scope decision (see `project_data_test_
dataset_priority_queue` memory, 2026-08-26) - not silently generalized to
"mutual funds" in any docstring, variable name, or printed result below.

ISIN -> SYMBOL MAPPING REUSES THE PROJECT'S OWN BHAVCOPY DATA, NO NEW
SOURCE - NSE's own bhavcopy already carries a real `isin` column
(`dtest.data.bhavcopy.COLUMNS`), confirmed unused for this purpose
anywhere else in the project. `build_isin_symbol_map` takes the
MOST-RECENT symbol seen for a given ISIN across the whole price history -
a real, accepted simplification: an ISIN surviving a symbol RENAME would
have its pre-rename price history correctly attributed to the (different)
old symbol name in `bhav_store`'s own long-format rows, so a stock that
renamed mid-window has its pre-rename months of holdings mapped to a
symbol whose OWN historical bhavcopy rows exist under a different string
- those months' holdings would then find no matching price data and drop
out silently via the normal reindex-to-NaN path, not corrupt anything.
Not investigated further this pass - a real, stated limitation, not
fixed.

MONTH-OVER-MONTH CHANGE, ACCUMULATION ONLY - `quantity_pct_change`
requires a stock to have been held (quantity > 0) in the IMMEDIATELY
PRECEDING disclosed month too; a stock newly appearing this month (no
prior disclosed holding) gets NaN, not a change from zero. This is a
deliberate scope choice (see the 2026-08-26 AskUserQuestion: "MF
accumulation" was chosen over "MF new-entrant" as the first hypothesis) -
conflating "grew an existing position" with "opened a brand-new one"
would blend two different economic stories into one signal. `max_gap_
days` additionally guards against comparing two DISCLOSED months that
are not actually calendar-adjacent (a stock genuinely absent from the
equity book for a stretch, then reappearing) - treated as not comparable,
NaN, never as an ordinary one-month change.

FILING LAG IS THE SAME SHARED ASSUMPTION `amc_portfolios.py` ALREADY
USES FOR BOTH AMCS (`_ASSUMED_DISCLOSURE_LAG_DAYS = 10`, SEBI's
regulatory filing deadline) - `to_event_panel` marks a disclosed month's
cross-section as "known" starting `period_end + 10 days`, mapped to the
first real TRADING day on/after that date (a calendar date can land on a
weekend/holiday). Deliberately NOT forward-filled onto every day until
the next disclosure (unlike `dtest.features.fundamentals.to_daily_panel`,
which correctly does persist a fundamentals reading as a continuous
STATE) - this is a discrete EVENT signal, matching `delivery_breakout`/
`oi_momentum`'s one-day-spike shape, not a regime read, so it fires True
on exactly one trading day per disclosed month and NaN/False every other
day.
"""

from __future__ import annotations

import pandas as pd

FILING_LAG_DAYS = 10
DEFAULT_MAX_GAP_DAYS = 40


def build_isin_symbol_map(bhav_long: pd.DataFrame) -> dict[str, str]:
    """`bhav_long` must carry `date`, `symbol`, `isin` columns (e.g.
    `dtest.data.bhav_store.load_long(..., columns=BHAV_COLUMNS)`). Returns
    {isin: most-recently-seen symbol} - see module docstring for the real
    symbol-rename caveat this simplification carries."""
    recent = (bhav_long.dropna(subset=["isin"])
              .sort_values("date")
              .drop_duplicates(subset="isin", keep="last"))
    return dict(zip(recent["isin"], recent["symbol"]))


def aggregate_monthly_quantity(holdings: pd.DataFrame, isin_symbol_map: dict[str, str]) -> pd.DataFrame:
    """`holdings` is `equity_holdings.csv` already loaded (columns include
    `period_end`, `isin`, `quantity`, `scheme_name`). Sums `quantity`
    across EVERY scheme and BOTH AMCs for a given (period_end, symbol) -
    the real, stated 2-AMC scope this module's own docstring states.
    Rows whose ISIN has no match in `isin_symbol_map` (delisted, or the
    real rename caveat above) are dropped, not guessed at."""
    df = holdings.copy()
    df["symbol"] = df["isin"].map(isin_symbol_map)
    df = df.dropna(subset=["symbol"])
    agg = (df.groupby(["period_end", "symbol"])
           .agg(total_quantity=("quantity", "sum"), n_schemes=("scheme_name", "nunique"))
           .reset_index())
    return agg


def quantity_pct_change(monthly: pd.DataFrame, max_gap_days: int = DEFAULT_MAX_GAP_DAYS) -> pd.DataFrame:
    """(period_end x symbol) panel of month-over-month %% change in
    `total_quantity` - NaN wherever the symbol wasn't held in the
    immediately preceding DISCLOSED month (a new position, or a genuine
    absence) or that preceding month is more than `max_gap_days` calendar
    days back (not really "the prior month" - see module docstring)."""
    wide = monthly.pivot(index="period_end", columns="symbol", values="total_quantity")
    pct = wide.pct_change()

    prior_period = wide.index.to_series().shift(1)
    gap_days = (wide.index.to_series() - prior_period).dt.days
    valid_gap = (gap_days <= max_gap_days).reindex(pct.index)
    return pct.where(valid_gap)


def to_event_panel(monthly_signal: pd.DataFrame, calendar: pd.DatetimeIndex,
                    lag_days: int = FILING_LAG_DAYS) -> pd.DataFrame:
    """Maps each `monthly_signal` row (indexed by `period_end`) onto the
    first trading day in `calendar` that is >= `period_end + lag_days` -
    a single-day event per disclosed month, NEVER forward-filled (see
    module docstring for why this differs from `fundamentals.
    to_daily_panel`'s persistent-state ffill). A `period_end` whose filing
    date falls after `calendar`'s last day is dropped (not yet knowable
    within this window)."""
    out = pd.DataFrame(float("nan"), index=calendar, columns=monthly_signal.columns)
    filing_dates = monthly_signal.index + pd.Timedelta(days=lag_days)
    positions = calendar.searchsorted(filing_dates)
    for row_pos, target_pos in enumerate(positions):
        if target_pos < len(calendar):
            out.iloc[target_pos] = monthly_signal.iloc[row_pos]
    return out
