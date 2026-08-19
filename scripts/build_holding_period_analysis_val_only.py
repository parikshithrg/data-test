"""DIAGNOSTIC ONLY - out-of-sample robustness check on
`build_holding_period_analysis.py`'s finding (7/8 signals underperform
NIFTY50 by a widening margin the longer a pure time-based hold runs, no
ATR/stop/target at all). That earlier run used FULL history (train+val+test
combined) since it was exploratory. This restricts ENTRY SIGNALS to fire
only within each split's own VAL window (with the same embargo every real
hypothesis test applies) - same universe/eligibility/entry-fill logic,
same 12-month forward-price lookup against the FULL panel (a val-window
entry still needs real future prices to hold 12 months into, which exist
in the full panel regardless of where the window officially ends).

    python scripts/build_holding_period_analysis_val_only.py

Runs BOTH val windows (primary/val 2017-2021, delivery/val 2023-07..2025-03)
for all 8 signal treatments - 16 combinations. NOT logged to
hypothesis_log.csv: still a descriptive read on real entries, not a new
hypothesis. Output kept local only (gitignored), same as the full-history
run - see that script's own docstring for why.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

from build_holding_period_analysis import N_MONTHS, _entries_from_signal, _load_price_panels  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs" / "holding_period_analysis_val_only"


def _val_mask(cfg, dates: pd.DatetimeIndex, split_name: str) -> tuple[np.ndarray, str]:
    split = cfg.split(split_name)
    start, end = split.window("val")
    after_start = dates[dates >= pd.Timestamp(start)]
    if len(after_start) <= split.embargo_days:
        raise RuntimeError(f"not enough sessions after {start} for {split.embargo_days}-day embargo")
    embargoed_start = after_start[split.embargo_days]
    mask = (dates >= embargoed_start) & (dates <= pd.Timestamp(end))
    label = f"{split_name}/val ({embargoed_start.date()}..{end}, {split.embargo_days}d embargo applied)"
    return mask, label


def main() -> int:
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()
    RUNS.mkdir(parents=True, exist_ok=True)

    print("loading full price panels ...")
    panels, stocks = _load_price_panels(cfg)
    close, open_, high, low, volume, turnover = (
        panels["close"], panels["open"], panels["high"], panels["low"],
        panels["volume"], panels["turnover"],
    )
    stock_close, stock_open = close[stocks], open_[stocks]
    stock_high, stock_low = high[stocks], low[stocks]
    stock_volume, stock_turnover = volume[stocks], turnover[stocks]
    dates = stock_close.index
    symbols = np.array(stock_close.columns)
    close_arr = stock_close.to_numpy(dtype=float)
    open_arr = stock_open.to_numpy(dtype=float)

    print("building point-in-time universe (full history) ...")
    uni = build_universe(stock_close, stock_turnover, cfg)
    eligible_full = uni.membership

    print("loading NIFTY50 ...")
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

    raw_signals: dict[str, pd.DataFrame] = {}
    raw_signals["mean_reversion"] = mean_reversion_signal(stock_close)
    raw_signals["delivery_breakout"] = delivery_breakout_signal(stock_close, deliv_aligned)
    raw_signals["oi_momentum"] = oi_momentum_signal(stock_close, oi_chg_aligned, dte_aligned)
    raw_signals["participant_tilt"] = participant_tilt_signal(stock_close, fii_aligned)
    raw_signals["vol_squeeze_breakout"] = vol_squeeze_breakout_signal(stock_high, stock_low, stock_close)
    raw_signals["vol_squeeze_breakout_delay2"] = (
        raw_signals["vol_squeeze_breakout"].shift(2).fillna(False).astype(bool))
    long_raw, _short_raw = price_action_signal(stock_high, stock_low, stock_close, stock_volume)
    raw_signals["price_action_long"] = long_raw
    rebalance_dates_all = list(uni.rebalance_dates)
    raw_signals["momentum"] = momentum_signal(
        stock_close, uni.membership, rebalance_dates_all,
        lookback_days=LOOKBACK_DAYS, skip_days=SKIP_DAYS, top_quantile=TOP_QUANTILE,
    )

    summary_rows = []
    for split_name in ("primary", "delivery"):
        mask, label = _val_mask(cfg, dates, split_name)
        print(f"\n########## {label} ##########")
        for name, raw_signal in raw_signals.items():
            sig = raw_signal.fillna(False).astype(bool) & eligible_full & mask[:, None]
            n_elig = int(sig.to_numpy().sum())
            print(f"\n{name} [{split_name}/val]: {n_elig} eligible firings")
            if n_elig == 0:
                summary_rows.append({"split": split_name, "signal": name, "n_entries": 0})
                continue

            df = _entries_from_signal(sig, open_arr, close_arr, dates, symbols, nifty_close)
            df.insert(0, "split_window", f"{split_name}/val")
            df.insert(0, "signal", name)
            n_full_12m = int(df["price_12m"].notna().sum())
            print(f"  {len(df)} entries filled; {n_full_12m} have a full 12m window "
                  f"({n_full_12m / len(df):.1%})")

            out_path = RUNS / f"{split_name}_val_{name}.csv"
            df.to_csv(out_path, index=False)

            row = {"split": split_name, "signal": name, "n_entries": len(df),
                   "n_full_12m_window": n_full_12m}
            for m in (1, 3, 6, 12):
                row[f"mean_return_{m}m_pct"] = df[f"return_{m}m_pct"].mean()
                row[f"mean_nifty_return_{m}m_pct"] = df[f"nifty_return_{m}m_pct"].mean()
                row[f"mean_excess_return_{m}m_pct"] = df[f"excess_return_{m}m_pct"].mean()
                row[f"pct_beating_nifty_{m}m"] = (df[f"excess_return_{m}m_pct"] > 0).mean() * 100.0
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RUNS / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n\n=== SUMMARY (both val windows) ===")
    cols = ["split", "signal", "n_entries", "mean_excess_return_1m_pct",
            "mean_excess_return_3m_pct", "mean_excess_return_6m_pct",
            "mean_excess_return_12m_pct", "pct_beating_nifty_12m"]
    print(summary[cols].to_string(index=False))

    print(f"\nWrote {RUNS}")
    print("NOT logged to hypothesis_log.csv - exploratory holding-period read on val windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
