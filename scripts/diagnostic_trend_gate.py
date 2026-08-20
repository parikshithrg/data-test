"""Per-stock trend gate: does gating mean_reversion to a STOCK'S OWN trailing
trend help - on TRAIN first, cheaply (trade-level + placebo, no portfolio yet).

    python scripts/diagnostic_trend_gate.py --split primary --window train

THE PREMISE UNDER TEST, stated before running anything (this file's own
existence is the pre-registration). This is a DIFFERENT claim from
`diagnostic_regime_gate.py`'s market-wide NIFTY50 bull/bear gate (2026-08-17,
tested and closed - directionally correct but insufficient, see
[[project-data-test-status]]). That gate asks "is the MARKET calm enough to
buy any dip." This one asks "is THIS STOCK, specifically, already in a real
uptrend" - motivated by the 2026-08-19 top-performer diagnosis, which found
that most of mean_reversion/oi_momentum/etc.'s best-performing symbols were
signal-timed bounces in stocks that did NOT actually beat buy-and-hold NIFTY
over the same span, while the two genuine exceptions (TATAELXSI, ADANIPOWER,
post-2021) had real, verified secular uptrends underneath their trades. The
hypothesis: a dip bought in a stock whose own trailing trend is already up is
disproportionately a normal pullback inside a real winner; the identical dip
bought in a stock whose own trend is down is disproportionately the next leg
of a real decline - "catching a falling knife," the specific failure mode the
per-symbol worst-list (UNITECH, PUNJLLOYD, RELCAPITAL, GLODYNE - all matched
to real, dated corporate distress) already showed concretely.

WHY THIS IS NOT THE SAME TEST AS PHASE 0, just with a different lookback:
Phase 0 gates every stock identically off ONE shared NIFTY50 reading (a
single time-series problem: is TODAY a bull day). This gates each stock off
ITS OWN price history (a cross-sectional problem: is THIS NAME, right now,
a winner or a loser) - two stocks can get opposite gate values on the same
day. Untested combination in this project until now.

REGIME LABEL: `dtest.features.regime.trailing_return`, applied to the
STOCK PANEL (a DataFrame, not NIFTY50's single Series) - the function
already supports this per its own docstring and `test_trailing_return_works_
on_a_dataframe_not_just_a_series` (tests/test_regime.py), so no new feature
code is needed, only a new caller. Same 63-session lookback as Phase 0 (not
freshly tuned here, same "re-deriving it is out of scope for a screening
pass" reasoning). UP = trailing return > 0. A NaN read (insufficient own
history - common for names that listed partway through the window) blocks
entry in BOTH directions, same "unknown does not default to a free pass"
convention every gate in this project uses - never lumped into whichever
gate is the complement.

WHY MEAN_REVERSION. Same signal Phase 0 used (most data of any signal in the
project, and the signal directly implicated in the top-performer finding
that motivated this test).
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
from dtest.features.regime import trailing_return
from dtest.features.technical import atr
from dtest.signals.mean_reversion import mean_reversion_signal
from dtest.universe import build_universe

TARGET_VALUE_PER_TRADE = 10_000.0
TREND_LOOKBACK = 63


def _load_panels(cfg):
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
    print(f"\n--- {label}: {n_sig} signals ---")
    if n_sig == 0:
        print("  zero signals - skipped")
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

    print(f"  n_resolved         {stats.n_resolved}")
    print(f"  mean_net_pct       {stats.mean_net_pct:.4f}")
    print(f"  win_rate_pct       {stats.win_rate_pct:.2f}")
    print(f"  hit_rate_pct       {stats.hit_rate_pct if stats.hit_rate_pct is not None else float('nan'):.2f}")
    print(f"  t_stat             {tstat['t_stat']:.3f}  (n_buckets={tstat['n_buckets']})")
    print(f"  placebo_max        {cmp['placebo_max']:.4f}")
    print(f"  beats_best_placebo {cmp['beats_best_placebo']}")
    print(f"  percentile         {cmp['percentile_vs_placebos']:.1f}")

    return {"label": label, "n_trades": stats.n_resolved, "mean_net_pct": stats.mean_net_pct,
            "win_rate_pct": stats.win_rate_pct, "t_stat": tstat["t_stat"],
            "placebo_max": cmp["placebo_max"], "beats_best_placebo": cmp["beats_best_placebo"],
            "percentile_vs_placebos": cmp["percentile_vs_placebos"]}


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

    print(f"=== per-stock trend-gate diagnostic: mean_reversion ({args.split}/{args.window}) ===")

    panels, stocks = _load_panels(cfg)
    close, open_, high, low, volume, turnover = (
        panels["close"], panels["open"], panels["high"], panels["low"],
        panels["volume"], panels["turnover"],
    )

    split = cfg.split(args.split)
    start, end = split.window(args.window)
    if args.window in ("val", "test"):
        after_start = close.index[close.index >= pd.Timestamp(start)]
        if len(after_start) <= split.embargo_days:
            print(f"not enough sessions after {start} for a {split.embargo_days}-day embargo")
            return 1
        start = after_start[split.embargo_days]
        print(f"  embargo applied: {args.window} signals start {start.date()}")
    mask = (close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))
    if not mask.any():
        print(f"no data for {args.split}/{args.window} ({start}..{end})")
        return 1

    end_pos = close.index.searchsorted(pd.Timestamp(end), side="right")
    buffer_pos = min(end_pos + split.embargo_days, len(close.index))
    keep = close.index[:buffer_pos]
    close, open_, high, low = close.loc[keep], open_.loc[keep], high.loc[keep], low.loc[keep]
    volume, turnover = volume.loc[keep], turnover.loc[keep]
    mask = mask[:buffer_pos]
    print(f"  panels truncated to {keep[0].date()}..{keep[-1].date()} "
         f"({len(keep)} sessions: window + {split.embargo_days}-session resolution buffer)")

    stock_close, stock_open = close[stocks], open_[stocks]
    stock_high, stock_low = high[stocks], low[stocks]
    stock_volume, stock_turnover = volume[stocks], turnover[stocks]

    print("building point-in-time universe ...")
    uni = build_universe(stock_close, stock_turnover, cfg)
    eligible = uni.membership & mask[:, None]

    # Per-stock, not market-wide: one trailing-return reading PER SYMBOL,
    # computed from that symbol's own close panel - two stocks can land on
    # opposite sides of the gate on the same date.
    trail_ret = trailing_return(stock_close, lookback=TREND_LOOKBACK)
    up_mask = (trail_ret > 0).fillna(False)
    down_mask = (trail_ret < 0).fillna(False)

    eligible_np = eligible.to_numpy() if hasattr(eligible, "to_numpy") else eligible
    elig_cells = int(eligible_np.sum())
    n_up = int((up_mask.to_numpy() & eligible_np).sum())
    n_down = int((down_mask.to_numpy() & eligible_np).sum())
    print(f"  own-trend coverage over eligible (date,symbol) cells: "
         f"{n_up}/{elig_cells} up ({100 * n_up / elig_cells:.1f}%), "
         f"{n_down}/{elig_cells} down ({100 * n_down / elig_cells:.1f}%), "
         f"{TREND_LOOKBACK}-session trailing OWN return")

    raw_signal = mean_reversion_signal(stock_close)
    a = atr(stock_high, stock_low, stock_close, window=14)
    rule = ExitRule(max_hold_days=7, atr_stop_multiple=2.0, risk_reward=2.5)
    n_seeds = args.seeds if args.seeds is not None else cfg.placebo_seeds

    ungated_signal = raw_signal & eligible
    up_gated_signal = raw_signal & eligible & up_mask.to_numpy()
    down_gated_signal = raw_signal & eligible & down_mask.to_numpy()

    kwargs = dict(open_=stock_open, high=stock_high, low=stock_low, close=stock_close,
                 volume=stock_volume, atr_panel=a)
    ungated = _run_variant("UNGATED (baseline)", ungated_signal, eligible, rule, cfg, n_seeds, **kwargs)
    up_gated = _run_variant("OWN-TREND-UP-GATED", up_gated_signal, eligible, rule, cfg, n_seeds, **kwargs)
    down_gated = _run_variant("OWN-TREND-DOWN-GATED", down_gated_signal, eligible, rule, cfg, n_seeds, **kwargs)

    print("\n=== SUMMARY ===")
    for r in (ungated, up_gated, down_gated):
        if r:
            print(f"  {r['label']:22s} n={r['n_trades']:6d} mean_net%={r['mean_net_pct']:8.4f} "
                 f"t={r['t_stat']:7.3f} pctile={r['percentile_vs_placebos']:6.1f} "
                 f"beats_best={r['beats_best_placebo']}")

    out_dir = Path(cfg.paths.runs) / "diagnostic_trend_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for r in (ungated, up_gated, down_gated) if r]).to_csv(
        out_dir / f"mean_reversion_{args.split}_{args.window}.csv", index=False)
    print(f"\nWrote {out_dir}")
    print("\nNOT logged to hypothesis_log.csv - this is a train-only screening "
         "diagnostic. A candidate chosen here must be confirmed on val, not "
         "just look better than baseline on train, before it means anything.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
