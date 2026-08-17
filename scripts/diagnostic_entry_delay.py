"""DIAGNOSTIC ONLY - does delaying entry N sessions after the signal fires
reduce the stop-hit skew that has now shown up in 4 of 5 rejected signals?

    python scripts/diagnostic_entry_delay.py --split primary --window train

Tests `vol_squeeze_breakout` specifically because its entry condition IS a
live volatility dislocation (the cleanest case of the entry-timing
hypothesis of any signal built so far - see [[project-data-test-status]]'s
2026-08-17 entry). Reuses the PRODUCTION `simulate_trades`/`run_placebos`
unchanged - no new fill mechanics needed, unlike `execution_diagnostic.py`
(which changed HOW a bar fills). Here only WHEN the signal is dated changes:
`raw_signal.shift(delay_days)` moves each firing forward `delay_days`
trading rows before eligibility is applied, so a delay=1 signal enters at
the open one session later than delay=0 would have, resolving on the exact
same intrabar walk/cost model/exit rule as the honest baseline. Eligibility
(universe membership + window) is checked at the DELAYED date, matching
where production checks it against whatever date a signal is dated - not
at the date the original dislocation was detected.

This deliberately never writes to the hypothesis log. A parameter sweep
across delay values, picked to make one of them look best, is exactly the
kind of search this project's discipline exists to prevent - this is a
diagnostic asking "is timing the mechanism", not a hypothesis test asking
"should we trade delay=N".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config, set_seeds
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.engine.simulate import ExitRule, simulate_trades, trades_to_frame
from dtest.evaluate.metrics import non_overlapping_tstat, summary_stats
from dtest.evaluate.placebo import run_placebos
from dtest.features.technical import atr
from dtest.signals.vol_squeeze_breakout import vol_squeeze_breakout_signal
from dtest.universe import build_universe

TARGET_VALUE_PER_TRADE = 10_000.0
DELAYS_TO_TEST = [0, 1, 2, 3]


def _load_price_panels(cfg):
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in ("open", "high", "low", "close",
                                                 "volume", "turnover")}
    return panels, stocks


def _stop_target_rates(stats):
    d = stats.as_dict()
    hit = d["hit_rate_pct"]
    expiry = d["expiry_rate_pct"]
    stop = 100.0 - hit - expiry
    return stop, hit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="primary", choices=["primary", "delivery"])
    ap.add_argument("--window", default=None, choices=["train", "val", "test"], required=True)
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()

    pd.set_option("display.width", 150)
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()

    print(f"=== entry-delay diagnostic: vol_squeeze_breakout ({args.split}/{args.window}) ===")

    panels, stocks = _load_price_panels(cfg)
    close, open_, high, low, volume, turnover = (
        panels["close"], panels["open"], panels["high"], panels["low"],
        panels["volume"], panels["turnover"],
    )

    split = cfg.split(args.split)
    start, end = split.window(args.window)
    if args.window in ("val", "test"):
        after_start = close.index[close.index >= pd.Timestamp(start)]
        start = after_start[split.embargo_days]
    mask = (close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))

    end_pos = close.index.searchsorted(pd.Timestamp(end), side="right")
    buffer_pos = min(end_pos + split.embargo_days, len(close.index))
    keep = close.index[:buffer_pos]
    close, open_, high, low = close.loc[keep], open_.loc[keep], high.loc[keep], low.loc[keep]
    volume, turnover = volume.loc[keep], turnover.loc[keep]
    mask = mask[:buffer_pos]

    stock_close, stock_open = close[stocks], open_[stocks]
    stock_high, stock_low = high[stocks], low[stocks]
    stock_volume, stock_turnover = volume[stocks], turnover[stocks]

    print("building point-in-time universe ...")
    uni = build_universe(stock_close, stock_turnover, cfg)
    eligible = uni.membership & mask[:, None]

    raw_signal = vol_squeeze_breakout_signal(stock_high, stock_low, stock_close)
    a = atr(stock_high, stock_low, stock_close, window=14)
    rule = ExitRule(max_hold_days=7, atr_stop_multiple=2.0, risk_reward=2.5)
    n_seeds = args.seeds if args.seeds is not None else cfg.placebo_seeds

    print(f"\n{'delay':>6} {'n_trades':>9} {'mean_net%':>10} {'t_stat':>8} "
         f"{'stop%':>7} {'target%':>8} {'plc_stop%':>10} {'plc_tgt%':>9} {'pctile':>7}")
    results = []
    for delay in DELAYS_TO_TEST:
        delayed_raw = raw_signal.shift(delay) if delay > 0 else raw_signal
        signal = delayed_raw.fillna(False).astype(bool) & eligible
        n_sig = int(signal.to_numpy().sum())
        if n_sig == 0:
            print(f"{delay:>6} zero signals - skipped")
            continue

        trades = trades_to_frame(simulate_trades(
            signal, "long", rule, open_=stock_open, high=stock_high, low=stock_low,
            close=stock_close, volume=stock_volume, atr_panel=a,
            target_value_per_trade=TARGET_VALUE_PER_TRADE, cfg=cfg,
        ))
        stats = summary_stats(trades)
        tstat = non_overlapping_tstat(trades)
        stop_pct, hit_pct = _stop_target_rates(stats)

        placebos = run_placebos(
            signal, eligible, "long", rule,
            open_=stock_open, high=stock_high, low=stock_low, close=stock_close,
            volume=stock_volume, atr_panel=a, target_value_per_trade=TARGET_VALUE_PER_TRADE,
            cfg=cfg, n_seeds=n_seeds,
        )
        cmp = placebos.compare(stats, "mean_net_pct")
        plc_hit = placebos.per_seed["hit_rate_pct"].mean()
        plc_expiry = placebos.per_seed["expiry_rate_pct"].mean()
        plc_stop = 100.0 - plc_hit - plc_expiry

        print(f"{delay:>6} {stats.n_resolved:>9} {stats.mean_net_pct:>10.4f} "
             f"{tstat['t_stat']:>8.3f} {stop_pct:>7.2f} {hit_pct:>8.2f} "
             f"{plc_stop:>10.2f} {plc_hit:>9.2f} {cmp['percentile_vs_placebos']:>7.3f}")

        results.append({
            "delay": delay, "n_trades": stats.n_resolved,
            "mean_net_pct": stats.mean_net_pct, "t_stat": tstat["t_stat"],
            "stop_pct": stop_pct, "hit_pct": hit_pct,
            "placebo_stop_pct": plc_stop, "placebo_hit_pct": plc_hit,
            "percentile_vs_placebos": cmp["percentile_vs_placebos"],
            "beats_best_placebo": cmp["beats_best_placebo"],
        })

    out = pd.DataFrame(results)
    out_dir = Path(cfg.paths.runs) / "diagnostic_entry_delay"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"vol_squeeze_breakout_{args.split}_{args.window}.csv", index=False)
    print(f"\nWrote {out_dir}")
    print("\nNOT logged to hypothesis_log.csv - this is a diagnostic, not a hypothesis test.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
