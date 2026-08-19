"""Cross-asset stress-regime gate, TERCILE variant of `diagnostic_stress_gate.py`
- same composite, same 6 signals, same placebo discipline, but gates on the
extreme thirds of the composite's own distribution instead of a straight
median split.

    python scripts/diagnostic_stress_gate_tercile.py

WHY THIS EXISTS: the median-split screen (2026-08-19) found a real,
mechanistically coherent pattern - stressed-gating hurt every signal,
calm-gating roughly neutralized 5 of 6 - but NONE of the 18 variants beat
their own placebo band. A 50/50 median split could be diluting a real effect
that's concentrated in the genuine extremes rather than spread evenly across
"below vs above average" - this tests that specific possibility, not a
fishing expedition across arbitrary thresholds.

TERCILE DEFINITION, decided before running: `dims["composite"]` is already a
percentile score, but it is the AVERAGE of six percentiles, so its own
distribution is bell-shaped around 50 (compressed tails), not uniform -
cutting it at a raw 33.3/66.7 would NOT select genuine thirds of its own
observed distribution. Fixed by taking a SECOND causal percentile of the
composite itself (`causal_percentile(dims["composite"])` - the exact same
function, reused, not a new one), which IS uniformly distributed by
construction, THEN cutting that at 33.3/66.7. The MIDDLE THIRD is excluded
from both gates entirely (neither calm nor stressed - genuinely ambiguous,
same "unknown blocks entry, no free pass" convention every gate in this
project uses), not folded into either side.
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
from dtest.engine.simulate import ExitRule
from dtest.features.stress import causal_percentile, cross_asset_stress_composite
from dtest.features.technical import atr
from dtest.signals.delivery_breakout import delivery_breakout_signal
from dtest.signals.mean_reversion import mean_reversion_signal
from dtest.signals.oi_momentum import oi_momentum_signal
from dtest.signals.participant_tilt import participant_tilt_signal
from dtest.signals.price_action import price_action_signal
from dtest.signals.vol_squeeze_breakout import vol_squeeze_breakout_signal
from dtest.universe import build_universe

from diagnostic_stress_gate import _load_price_panels, _run_variant  # noqa: E402

LOWER_TERCILE = 33.333
UPPER_TERCILE = 66.667


def main() -> int:
    pd.set_option("display.width", 160)
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()

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

    print("building point-in-time universe ...")
    uni = build_universe(stock_close, stock_turnover, cfg)
    eligible_full = uni.membership

    print("loading delivery / OI / FII data ...")
    deliv_aligned = to_panel(load_delivery_long(cfg.paths.fno_db), "delivery_pct").reindex(
        index=stock_close.index, columns=stock_close.columns)
    oi_long = load_front_month_oi(cfg.paths.fno_db)
    oi_chg_aligned = to_panel(oi_long, "oi_chg_pct").reindex(
        index=stock_close.index, columns=stock_close.columns)
    dte_aligned = to_panel(oi_long, "days_to_expiry").reindex(
        index=stock_close.index, columns=stock_close.columns)
    fii_aligned = load_fii_net_index_flow(cfg.paths.fno_db).reindex(stock_close.index)

    print("loading macro stress series ...")
    india_vix = pd.read_csv(cfg.paths.price_dir / "INDIAVIX_DAILY.csv",
                             parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    us_vix = pd.read_csv(cfg.paths.macro_dir / "VIX_DAILY.csv",
                          parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    usdinr = pd.read_csv(cfg.paths.macro_dir / "USDINR_DAILY.csv",
                          parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    dxy = pd.read_csv(cfg.paths.macro_dir / "DXY_DAILY.csv",
                       parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    gold = pd.read_csv(cfg.paths.macro_dir / "GOLD_DAILY.csv",
                        parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()

    print("computing point-in-time breadth from this project's own universe ...")
    sma200 = stock_close.rolling(200, min_periods=200).mean()
    above = (stock_close > sma200) & eligible_full
    denom = eligible_full.sum(axis=1).astype(float).replace(0.0, np.nan)
    breadth_pct = (above.sum(axis=1).astype(float) / denom * 100.0)

    print("building the 6-dimension stress composite ...")
    dims = cross_asset_stress_composite(india_vix, us_vix, breadth_pct, usdinr, dxy, gold)

    print("re-percentiling the composite itself for genuine causal terciles ...")
    composite_pctile = causal_percentile(dims["composite"])
    valid_from = composite_pctile.first_valid_index()
    valid_to = composite_pctile.last_valid_index()
    n_valid = int(composite_pctile.notna().sum())
    print(f"  composite tercile-rank valid {valid_from.date() if valid_from is not None else None} .. "
          f"{valid_to.date() if valid_to is not None else None} ({n_valid} sessions)")

    calm_mask = (composite_pctile <= LOWER_TERCILE).fillna(False)
    stressed_mask = (composite_pctile >= UPPER_TERCILE).fillna(False)
    middle_mask = composite_pctile.notna() & ~calm_mask & ~stressed_mask
    print(f"  {int(calm_mask.sum())} calm-tercile sessions, {int(stressed_mask.sum())} "
          f"stressed-tercile sessions, {int(middle_mask.sum())} middle-third sessions excluded "
          f"from both gates ({LOWER_TERCILE:.1f}/{UPPER_TERCILE:.1f} split)")

    a = atr(stock_high, stock_low, stock_close, window=14)
    rule = ExitRule(max_hold_days=7, atr_stop_multiple=2.0, risk_reward=2.5)
    n_seeds = cfg.placebo_seeds

    signal_configs = [
        ("mean_reversion", "primary", lambda: mean_reversion_signal(stock_close)),
        ("delivery_breakout", "delivery", lambda: delivery_breakout_signal(stock_close, deliv_aligned)),
        ("oi_momentum", "primary", lambda: oi_momentum_signal(stock_close, oi_chg_aligned, dte_aligned)),
        ("participant_tilt", "delivery", lambda: participant_tilt_signal(stock_close, fii_aligned)),
        ("vol_squeeze_breakout", "primary",
         lambda: vol_squeeze_breakout_signal(stock_high, stock_low, stock_close)),
        ("price_action_long", "primary",
         lambda: price_action_signal(stock_high, stock_low, stock_close, stock_volume)[0]),
    ]

    kwargs = dict(open_=stock_open, high=stock_high, low=stock_low, close=stock_close,
                   volume=stock_volume, atr_panel=a)

    all_results = []
    for name, split_name, build_raw in signal_configs:
        split = cfg.split(split_name)
        train_start, train_end = split.window("train")
        window_mask = (dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end))
        eligible = eligible_full & window_mask[:, None]

        print(f"\n=== {name} ({split_name}/train, {train_start}..{train_end}) ===")
        raw_signal = build_raw().fillna(False).astype(bool)

        ungated_signal = raw_signal & eligible
        calm_signal = raw_signal & eligible & calm_mask.to_numpy()[:, None]
        stressed_signal = raw_signal & eligible & stressed_mask.to_numpy()[:, None]

        for label, sig in (("UNGATED (baseline)", ungated_signal),
                            ("CALM-TERCILE-GATED", calm_signal),
                            ("STRESSED-TERCILE-GATED", stressed_signal)):
            r = _run_variant(label, sig, eligible, rule, cfg, n_seeds, **kwargs)
            if r:
                r.update({"signal": name, "split": split_name})
                all_results.append(r)

    out = pd.DataFrame(all_results)
    out_dir = Path(cfg.paths.runs) / "diagnostic_stress_gate_tercile"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "summary.csv", index=False)

    print("\n\n=== FULL SUMMARY ===")
    cols = ["signal", "label", "n_trades", "mean_net_pct", "t_stat",
            "percentile_vs_placebos", "beats_best_placebo"]
    print(out[cols].to_string(index=False))

    print(f"\nWrote {out_dir}")
    print("\nNOT logged to hypothesis_log.csv - train-only screening diagnostic. "
          "A candidate must be chosen HERE, before looking at val, then confirmed "
          "via its own real test script - never promoted just for looking better "
          "than baseline on this screen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
