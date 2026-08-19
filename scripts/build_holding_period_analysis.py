"""DIAGNOSTIC ONLY - for every signal's real entry points (same signal logic,
same point-in-time universe, same T+1-open entry fill as every other script
in this project), replace the ATR stop/target/7-day exit with a PURE
TIME-BASED hold and record the raw close price - and the return from entry -
at 1 through 12 months held. No new hypothesis, nothing logged to
hypothesis_log.csv: this re-uses each signal's own real entries to ask a
different, exploratory question than any accept/reject test does - "is there
a holding horizon where price behaves differently than the 7-day exit window
already tested can see."

    python scripts/build_holding_period_analysis.py

SCOPE, EACH CHOICE CONFIRMED WITH THE USER BEFORE BUILDING (2026-08-19):
- Signals: mean_reversion, delivery_breakout, oi_momentum, participant_tilt,
  vol_squeeze_breakout (immediate AND delay=2), price_action LONG, momentum -
  every long-only single-leg signal in the project. Pairs/same_sector_pairing
  excluded - "equity delivery trades only", no futures short leg.
- Data scope: FULL available price history, train+val+test combined, not
  restricted to any split's train window - this is descriptive/exploratory,
  never used to accept or reject a hypothesis, so the train/val/test embargo
  discipline that protects THAT decision does not apply here. Universe and
  signal construction are otherwise completely unchanged from the real
  hypothesis tests.
- "1 month" = 21 trading sessions, this project's own existing convention
  (momentum's own monthly hold is already exactly 21 sessions;
  TRADING_DAYS_PER_YEAR=252 is used everywhere else). 12 months = 252
  trading sessions forward of the entry fill date.
- Entry fill: T+1 open, same as every other script - `next_trading_day`
  reduces to literally the next row of the same trading calendar since
  `signal_date` is always already a row in it; entry is valid only if that
  day has a real, positive open price (the same criterion
  `engine/fills.py::peek_fill_price` uses - reimplemented here vectorized
  over tens of thousands of entries rather than called row-by-row, no
  volume/participation cap applied since this is a price-behavior read, not
  a sizing decision).
- price_Xm is the RAW close price X months after entry, no cost adjustment -
  this answers "at what price did it exit", not "what would the net P&L have
  been". return_Xm is the resulting % move from the entry (T+1 open) price,
  also raw/uncosted for the same reason.

NIFTY50 SAME-DATES COMPARISON (added 2026-08-19, on request, before any of
the above was read as a real finding): nifty_price_Xm / nifty_return_Xm walk
NIFTY50's own close price forward from the SAME entry_date each stock entry
used, over the identical 21-trading-day-per-month checkpoints - close-to-
close, matching `dtest.evaluate.metrics.benchmark_excess`'s own established
convention exactly (entry/exit priced off `benchmark_close.reindex(date)`),
not a new one invented for this script. excess_return_Xm = return_Xm minus
nifty_return_Xm is the number that actually answers "did the entry add
anything over just holding the index from the same date" - the raw
return_Xm columns on their own cannot distinguish a real signal from
riding a multi-year secular uptrend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config, set_seeds
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.data.delivery import load_delivery_long
from dtest.data.fno_oi import load_front_month_oi
from dtest.data.participant_flow import load_fii_net_index_flow
from dtest.signals.delivery_breakout import delivery_breakout_signal
from dtest.signals.mean_reversion import mean_reversion_signal
from dtest.signals.momentum import LOOKBACK_DAYS, SKIP_DAYS, TOP_QUANTILE, momentum_signal
from dtest.signals.oi_momentum import oi_momentum_signal
from dtest.signals.participant_tilt import participant_tilt_signal
from dtest.signals.price_action import price_action_signal
from dtest.signals.vol_squeeze_breakout import vol_squeeze_breakout_signal
from dtest.universe import build_universe

TRADING_DAYS_PER_MONTH = 21
N_MONTHS = 12
RUNS = Path(__file__).resolve().parent.parent / "runs" / "holding_period_analysis"


def _load_price_panels(cfg):
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in ("open", "high", "low", "close",
                                                 "volume", "turnover")}
    return panels, stocks


def _entries_from_signal(signal: pd.DataFrame, open_arr: np.ndarray,
                          close_arr: np.ndarray, dates: pd.DatetimeIndex,
                          symbols: np.ndarray, nifty_close: np.ndarray) -> pd.DataFrame:
    """Vectorized T+1-open entry fill + forward close price at every
    1..N_MONTHS trading-day checkpoint, for every True cell in `signal`,
    plus the NIFTY50 same-dates comparison (close-to-close from the same
    entry_date, matching `benchmark_excess`'s own convention)."""
    T = len(dates)
    sig_arr = signal.to_numpy(dtype=bool)
    rows, cols = np.where(sig_arr)
    fill_rows = rows + 1
    keep = fill_rows < T
    rows, cols, fill_rows = rows[keep], cols[keep], fill_rows[keep]

    entry_price = open_arr[fill_rows, cols]
    valid = ~np.isnan(entry_price) & (entry_price > 0)
    rows, cols, fill_rows, entry_price = (rows[valid], cols[valid],
                                           fill_rows[valid], entry_price[valid])

    nifty_entry = nifty_close[fill_rows]

    out = {
        "symbol": symbols[cols],
        "signal_date": dates[rows],
        "entry_date": dates[fill_rows],
        "entry_price": entry_price,
        "nifty_entry_price": nifty_entry,
    }
    for k in range(1, N_MONTHS + 1):
        target = fill_rows + TRADING_DAYS_PER_MONTH * k
        in_range = target < T
        clipped = np.clip(target, 0, T - 1)
        price_k = np.where(in_range, close_arr[clipped, cols], np.nan)
        nifty_k = np.where(in_range, nifty_close[clipped], np.nan)
        return_k = (price_k / entry_price - 1.0) * 100.0
        nifty_return_k = (nifty_k / nifty_entry - 1.0) * 100.0
        out[f"price_{k}m"] = price_k
        out[f"return_{k}m_pct"] = return_k
        out[f"nifty_price_{k}m"] = nifty_k
        out[f"nifty_return_{k}m_pct"] = nifty_return_k
        out[f"excess_return_{k}m_pct"] = return_k - nifty_return_k

    return pd.DataFrame(out)


def main() -> int:
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()
    RUNS.mkdir(parents=True, exist_ok=True)

    print("loading full price panels (no window restriction) ...")
    panels, stocks = _load_price_panels(cfg)
    close, open_, high, low, volume, turnover = (
        panels["close"], panels["open"], panels["high"], panels["low"],
        panels["volume"], panels["turnover"],
    )
    print(f"price panels: {close.shape[0]} sessions x {close.shape[1]} symbols "
          f"({close.index[0].date()} .. {close.index[-1].date()})")

    stock_close, stock_open = close[stocks], open_[stocks]
    stock_high, stock_low = high[stocks], low[stocks]
    stock_volume, stock_turnover = volume[stocks], turnover[stocks]
    dates = stock_close.index
    symbols = np.array(stock_close.columns)
    close_arr = stock_close.to_numpy(dtype=float)
    open_arr = stock_open.to_numpy(dtype=float)

    print("building point-in-time universe (full history) ...")
    uni = build_universe(stock_close, stock_turnover, cfg)
    eligible = uni.membership

    print("loading NIFTY50 for the same-dates benchmark comparison ...")
    nifty_df = pd.read_csv(cfg.paths.price_dir / "NIFTY50_DAILY.csv",
                            parse_dates=["date"]).set_index("date").sort_index()
    nifty_df = nifty_df.loc[nifty_df.index <= pd.Timestamp(cfg.as_of)]
    nifty_close = nifty_df["close"].reindex(dates).ffill().to_numpy(dtype=float)

    print("loading delivery / OI / FII data ...")
    deliv_aligned = to_panel(load_delivery_long(cfg.paths.fno_db), "delivery_pct").reindex(
        index=stock_close.index, columns=stock_close.columns)
    oi_long = load_front_month_oi(cfg.paths.fno_db)
    oi_chg_aligned = to_panel(oi_long, "oi_chg_pct").reindex(
        index=stock_close.index, columns=stock_close.columns)
    dte_aligned = to_panel(oi_long, "days_to_expiry").reindex(
        index=stock_close.index, columns=stock_close.columns)
    fii_aligned = load_fii_net_index_flow(cfg.paths.fno_db).reindex(stock_close.index)

    signals: dict[str, pd.DataFrame] = {}

    signals["mean_reversion"] = mean_reversion_signal(stock_close)
    signals["delivery_breakout"] = delivery_breakout_signal(stock_close, deliv_aligned)
    signals["oi_momentum"] = oi_momentum_signal(stock_close, oi_chg_aligned, dte_aligned)
    signals["participant_tilt"] = participant_tilt_signal(stock_close, fii_aligned)
    signals["vol_squeeze_breakout"] = vol_squeeze_breakout_signal(stock_high, stock_low, stock_close)
    delay2 = signals["vol_squeeze_breakout"].shift(2).fillna(False).astype(bool)
    signals["vol_squeeze_breakout_delay2"] = delay2
    long_raw, _short_raw = price_action_signal(stock_high, stock_low, stock_close, stock_volume)
    signals["price_action_long"] = long_raw
    rebalance_dates = list(uni.rebalance_dates)
    signals["momentum"] = momentum_signal(
        stock_close, uni.membership, rebalance_dates,
        lookback_days=LOOKBACK_DAYS, skip_days=SKIP_DAYS, top_quantile=TOP_QUANTILE,
    )

    summary_rows = []
    all_frames = []
    for name, raw_signal in signals.items():
        sig = raw_signal.fillna(False).astype(bool) & eligible
        n_raw = int(raw_signal.to_numpy(dtype=bool).sum())
        n_elig = int(sig.to_numpy().sum())
        print(f"\n{name}: {n_elig} eligible firings (before universe filter: {n_raw})")
        if n_elig == 0:
            continue

        df = _entries_from_signal(sig, open_arr, close_arr, dates, symbols, nifty_close)
        df.insert(0, "signal", name)
        n_full_12m = int(df["price_12m"].notna().sum())
        print(f"  {len(df)} entries with a valid T+1 fill; {n_full_12m} have a full 12-month "
              f"forward window ({n_full_12m / len(df):.1%})")

        out_path = RUNS / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path}")

        row = {"signal": name, "n_signal_firings_raw": n_raw, "n_eligible_firings": n_elig,
               "n_entries_filled": len(df), "n_with_full_12m_window": n_full_12m}
        for m in (1, 3, 6, 12):
            row[f"mean_return_{m}m_pct"] = df[f"return_{m}m_pct"].mean()
            row[f"mean_nifty_return_{m}m_pct"] = df[f"nifty_return_{m}m_pct"].mean()
            row[f"mean_excess_return_{m}m_pct"] = df[f"excess_return_{m}m_pct"].mean()
            row[f"pct_entries_beating_nifty_{m}m"] = (
                df[f"excess_return_{m}m_pct"] > 0).mean() * 100.0
        summary_rows.append(row)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = RUNS / "all_signals.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nWrote combined file: {combined_path} ({len(combined)} total entries)")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RUNS / "summary.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    print("\nNOT logged to hypothesis_log.csv - exploratory holding-period read, tests no new claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
