"""Cross-asset stress-regime gate - does gating any of the 6 core reactive
signals by systemic stress (calm vs stressed) help, on TRAIN first, cheaply
(trade-level + placebo, no portfolio yet) - same discipline
`diagnostic_regime_gate.py` (Phase 0, trailing-return regime) already used.

    python scripts/diagnostic_stress_gate.py

THE PREMISE UNDER TEST, stated before running anything: the economic story
is genuinely ambiguous either direction, so BOTH are tested as parallel
screens rather than assuming one - "stressed markets keep falling, don't buy
into panic" (CALM-gated should help) vs. "capitulation dips ARE the real
opportunity" (STRESSED-gated should help). Composite defined in
`dtest.features.stress` (6 dimensions: India VIX, US VIX, breadth-inverted,
USDINR 20d change, DXY, gold 20d return - the same construction Local
Terminal's own Black Swan Radar dashboard already built, now run through
this project's causal/point-in-time discipline). Gate threshold is a plain
median split (composite >= 50 = stressed, < 50 = calm) - a pre-stated,
round, non-fitted convention, same spirit as Phase 0's trailing_return > 0
/ < 0 split, decided before this script ever ran.

REAL DATA CONSTRAINT, stated plainly: India VIX's own local history starts
2009-03-02, and the composite additionally needs a 252-session warm-up on
top of that before any dimension is valid - so the gate has no opinion
before roughly 2010, cutting primary/train's usable span (2004-2016) down
to about 2010-2016 for the 4 primary-split signals below. delivery/train
(2019-2023) sits entirely inside the gate's valid range.

NOT logged to hypothesis_log.csv - screening only. A candidate must be
chosen HERE, before ever looking at val, and confirmed via its own real
test_<signal>_stress_gated.py script - never promoted just for looking good
on this screen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config, set_seeds
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.data.delivery import load_delivery_long
from dtest.data.fno_oi import load_front_month_oi
from dtest.data.participant_flow import load_fii_net_index_flow
from dtest.engine.simulate import ExitRule, simulate_trades, trades_to_frame
from dtest.evaluate.metrics import non_overlapping_tstat, summary_stats
from dtest.evaluate.placebo import run_placebos
from dtest.features.stress import cross_asset_stress_composite
from dtest.features.technical import atr
from dtest.signals.delivery_breakout import delivery_breakout_signal
from dtest.signals.mean_reversion import mean_reversion_signal
from dtest.signals.oi_momentum import oi_momentum_signal
from dtest.signals.participant_tilt import participant_tilt_signal
from dtest.signals.price_action import price_action_signal
from dtest.signals.vol_squeeze_breakout import vol_squeeze_breakout_signal
from dtest.universe import build_universe

TARGET_VALUE_PER_TRADE = 10_000.0
GATE_THRESHOLD = 50.0  # pre-stated median split, decided before any run


def _load_price_panels(cfg):
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in ("open", "high", "low", "close",
                                                 "volume", "turnover")}
    return panels, stocks


def _run_variant(label, signal, eligible, rule, cfg, n_seeds, *, open_, high, low,
                  close, volume, atr_panel):
    n_sig = int(signal.to_numpy().sum())
    print(f"    {label}: {n_sig} signals", end="")
    if n_sig == 0:
        print(" - skipped (zero signals)")
        return None

    trades = trades_to_frame(simulate_trades(
        signal, "long", rule, open_=open_, high=high, low=low, close=close,
        volume=volume, atr_panel=atr_panel, target_value_per_trade=TARGET_VALUE_PER_TRADE,
        cfg=cfg,
    ))
    stats = summary_stats(trades)
    tstat = non_overlapping_tstat(trades)
    placebos = run_placebos(
        signal, eligible, "long", rule, open_=open_, high=high, low=low, close=close,
        volume=volume, atr_panel=atr_panel, target_value_per_trade=TARGET_VALUE_PER_TRADE,
        cfg=cfg, n_seeds=n_seeds,
    )
    cmp = placebos.compare(stats, "mean_net_pct")
    print(f"  n={stats.n_resolved} mean_net%={stats.mean_net_pct:.4f} "
          f"t={tstat['t_stat']:.3f} pctile={cmp['percentile_vs_placebos']:.1f} "
          f"beats_best={cmp['beats_best_placebo']}")

    return {"label": label, "n_trades": stats.n_resolved, "mean_net_pct": stats.mean_net_pct,
            "t_stat": tstat["t_stat"], "n_buckets": tstat["n_buckets"],
            "placebo_mean": cmp["placebo_mean"], "placebo_max": cmp["placebo_max"],
            "beats_best_placebo": cmp["beats_best_placebo"],
            "percentile_vs_placebos": cmp["percentile_vs_placebos"]}


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
    import numpy as np
    sma200 = stock_close.rolling(200, min_periods=200).mean()
    above = (stock_close > sma200) & eligible_full
    denom = eligible_full.sum(axis=1).astype(float).replace(0.0, np.nan)
    breadth_pct = (above.sum(axis=1).astype(float) / denom * 100.0)

    print("building the 6-dimension stress composite ...")
    dims = cross_asset_stress_composite(india_vix, us_vix, breadth_pct, usdinr, dxy, gold)
    valid_from = dims["composite"].first_valid_index()
    valid_to = dims["composite"].last_valid_index()
    n_valid = int(dims["composite"].notna().sum())
    print(f"  composite valid {valid_from.date() if valid_from is not None else None} .. "
          f"{valid_to.date() if valid_to is not None else None} ({n_valid} sessions)")

    calm_mask = (dims["composite"] < GATE_THRESHOLD).fillna(False)
    stressed_mask = (dims["composite"] >= GATE_THRESHOLD).fillna(False)
    print(f"  {int(calm_mask.sum())} calm sessions, {int(stressed_mask.sum())} stressed sessions "
          f"(threshold={GATE_THRESHOLD})")

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
                            ("CALM-GATED", calm_signal),
                            ("STRESSED-GATED", stressed_signal)):
            r = _run_variant(label, sig, eligible, rule, cfg, n_seeds, **kwargs)
            if r:
                r.update({"signal": name, "split": split_name})
                all_results.append(r)

    out = pd.DataFrame(all_results)
    out_dir = Path(cfg.paths.runs) / "diagnostic_stress_gate"
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
