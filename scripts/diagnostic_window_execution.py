"""DIAGNOSTIC: is mean_reversion's rejection about EXECUTION, WINDOW, or both?

    python scripts/diagnostic_window_execution.py

The 08-14 execution-timing diagnostic isolated same-bar-close vs T+1-open on
this project's own primary/train window (2004-2016) and found only a small
gap (-0.97% -> -0.72%, still worse than the best of 15 placebos). That left
one variable untested: the predecessor validated this same signal on a
completely different, much shorter window (2021-01-01 .. 2026-08-13, per
`market_gate/trade_recommendations_backtest.py`'s TRAIN_START/TEST_START) and
reported it POSITIVE there (net +1.036%/trade train, +0.516% test - both
sides of ITS split). This script crosses window x execution in a 2x2 so the
two variables stop being confounded:

                    2004-2016 (this project's train)   2021-2026 (predecessor's window)
    T+1 open (honest)        [already logged, rejected]        NEW
    same-bar close (diag)    [already run, still fails ]       NEW - closest reproduction
                                                                 of the predecessor's own setup

If the bottom-right cell alone turns positive, the predecessor's result was
real-but-narrow (a short, favourable window), not a fabrication of loose
execution - the top-right cell then separates out how much of THAT is honest
execution taking it back down again. Universe, costs, corporate-action
handling and the exact signal implementation all still differ from the
predecessor's own code, so this will not reproduce their number exactly - it
tests whether the WINDOW alone moves this project's own honest number from
solidly rejected to competitive.

Same-bar-close is diagnostic-only (see `dtest.research.execution_diagnostic`
module docstring) and neither cell here is written to the hypothesis log -
this script answers "which variable explains the gap", not "should this
ship". A verdict on the 2021-2026 window under honest T+1 execution, if one
is warranted, belongs in a real hypothesis-log entry using
`scripts/test_mean_reversion.py`, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config, set_seeds
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.engine.simulate import ExitRule, simulate_trades, trades_to_frame
from dtest.evaluate.metrics import benchmark_excess, non_overlapping_tstat, summary_stats
from dtest.evaluate.placebo import run_placebos
from dtest.features.technical import atr
from dtest.research.execution_diagnostic import simulate_same_bar_close
from dtest.signals.mean_reversion import mean_reversion_signal
from dtest.universe import build_universe

TARGET_VALUE_PER_TRADE = 10_000.0
N_SEEDS = 15

# The predecessor's own window (trade_recommendations_backtest.py:
# TRAIN_START/TRAIN_END/TEST_START), combined - it reported positive numbers
# on BOTH sides of that split, so the combined span is the fair comparison.
RECENT_START = pd.Timestamp("2021-01-01")


def _windowed_panels(close, open_, high, low, volume, turnover, embargo_days, start, end):
    """Same truncate-at-the-back pattern as test_mean_reversion.py /
    diagnostic_execution_timing.py: keep every bar from the PANEL START (so
    lookback indicators have full runway) through `end` plus an
    embargo-days resolution buffer. Only the back is cut.
    """
    mask = (close.index >= start) & (close.index <= end)
    end_pos = close.index.searchsorted(pd.Timestamp(end), side="right")
    buffer_pos = min(end_pos + embargo_days, len(close.index))
    keep = close.index[:buffer_pos]
    return (close.loc[keep], open_.loc[keep], high.loc[keep], low.loc[keep],
            volume.loc[keep], turnover.loc[keep], mask[:buffer_pos], keep)


def _report(label: str, trades: pd.DataFrame, bench: pd.Series | None,
           placebos_cmp: dict | None) -> dict:
    stats = summary_stats(trades)
    tstat = non_overlapping_tstat(trades)
    excess = None
    if bench is not None:
        e = benchmark_excess(trades, bench)
        excess = float(e.mean()) if len(e) else float("nan")
    print(f"\n--- {label} ---")
    print(f"  n_resolved={stats.n_resolved}  mean_net_pct={stats.mean_net_pct:.4f}  "
         f"win_rate={stats.win_rate_pct:.2f}%  hit_rate={stats.hit_rate_pct}  "
         f"mean_held_days={stats.mean_held_days:.2f}")
    print(f"  t_stat={tstat['t_stat']:.3f} (n_buckets={tstat['n_buckets']})", end="")
    if excess is not None:
        print(f"  excess_vs_NIFTY50={excess:.4f}pp", end="")
    print()
    if placebos_cmp is not None:
        print(f"  placebo band=[{placebos_cmp['placebo_min']:.4f}, "
             f"{placebos_cmp['placebo_max']:.4f}]  "
             f"beats_best_placebo={placebos_cmp['beats_best_placebo']}")
    return {"label": label, "stats": stats, "tstat": tstat, "excess": excess,
            "placebo": placebos_cmp}


def main() -> int:
    pd.set_option("display.width", 150)
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()

    split = cfg.split("primary")
    embargo_days = split.embargo_days

    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in
             ("open", "high", "low", "close", "volume", "turnover")}
    close, open_, high, low, volume, turnover = (
        panels["close"][stocks], panels["open"][stocks], panels["high"][stocks],
        panels["low"][stocks], panels["volume"][stocks], panels["turnover"][stocks],
    )
    print(f"panels: {close.shape[0]} sessions x {close.shape[1]} symbols "
         f"({close.index[0].date()} .. {close.index[-1].date()})")

    bench = None
    bench_path = cfg.paths.price_dir / "NIFTY50_DAILY.csv"
    if bench_path.exists():
        bdf = pd.read_csv(bench_path, parse_dates=["date"]).set_index("date").sort_index()
        bench = bdf["close"].reindex(close.index).ffill()

    recent_end = pd.Timestamp(cfg.as_of)
    windows = [
        ("2004-2016 (this project's primary/train)", split.train_start, split.train_end),
        ("2021-2026 (predecessor's own window)", RECENT_START, recent_end),
    ]

    rule = ExitRule(max_hold_days=7, atr_stop_multiple=2.0, risk_reward=2.5)
    results: dict[tuple[str, str], dict] = {}

    for win_label, start, end in windows:
        wclose, wopen, whigh, wlow, wvolume, wturnover, wmask, keep = _windowed_panels(
            close, open_, high, low, volume, turnover, embargo_days,
            pd.Timestamp(start), pd.Timestamp(end),
        )
        print(f"\n=== WINDOW: {win_label} "
             f"({pd.Timestamp(start).date()} .. {pd.Timestamp(end).date()}, "
             f"panels {keep[0].date()}..{keep[-1].date()}) ===")

        uni = build_universe(wclose, wturnover, cfg)
        raw_signal = mean_reversion_signal(wclose)
        a = atr(whigh, wlow, wclose, window=14)
        eligible = uni.membership & wmask[:, None]
        signal = raw_signal & eligible
        n_sig = int(signal.to_numpy().sum())
        print(f"  signal: {n_sig} firings after universe/window intersection")
        if n_sig == 0:
            print("  zero signals - skipping this window")
            continue

        wbench = bench.reindex(wclose.index) if bench is not None else None

        for exec_label, sim_fn in (
            ("T+1 open (honest)", simulate_trades),
            ("same-bar close (diagnostic)", simulate_same_bar_close),
        ):
            trades = trades_to_frame(sim_fn(
                signal, "long", rule, open_=wopen, high=whigh, low=wlow,
                close=wclose, volume=wvolume, atr_panel=a,
                target_value_per_trade=TARGET_VALUE_PER_TRADE, cfg=cfg,
            ))
            placebos = run_placebos(
                signal, eligible, "long", rule, open_=wopen, high=whigh, low=wlow,
                close=wclose, volume=wvolume, atr_panel=a,
                target_value_per_trade=TARGET_VALUE_PER_TRADE, cfg=cfg, n_seeds=N_SEEDS,
            )
            cmp = placebos.compare(summary_stats(trades), "mean_net_pct")
            r = _report(f"{win_label} x {exec_label}", trades, wbench, cmp)
            results[(win_label, exec_label)] = r

    print("\n\n=== 2x2 SUMMARY (mean_net_pct, %) ===")
    header = f"{'window':42s}"
    exec_labels = ["T+1 open (honest)", "same-bar close (diagnostic)"]
    for el in exec_labels:
        header += f"{el:>30s}"
    print(header)
    for win_label, _, _ in windows:
        row = f"{win_label:42s}"
        for el in exec_labels:
            r = results.get((win_label, el))
            row += f"{r['stats'].mean_net_pct:30.4f}" if r else f"{'--':>30s}"
        print(row)

    print("\n=== 2x2 SUMMARY (beats best placebo) ===")
    for win_label, _, _ in windows:
        row = f"{win_label:42s}"
        for el in exec_labels:
            r = results.get((win_label, el))
            row += f"{str(r['placebo']['beats_best_placebo']):>30s}" if r else f"{'--':>30s}"
        print(row)

    print("\n[DIAGNOSTIC ONLY - not logged to hypothesis_log. Isolates window vs "
         "execution as independent variables; a hypothesis test on the "
         "2021-2026 window under honest T+1 execution, if warranted by this "
         "table, belongs in scripts/test_mean_reversion.py as a real logged run.]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
