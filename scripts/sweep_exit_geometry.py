"""Exit-geometry sweep: has this project's exit rule (7-day hold, 2.0x ATR
stop, 1:2.5 R:R) ever been tested on its OWN signals, or just inherited?

    python scripts/sweep_exit_geometry.py --split primary --window train

ANSWER GOING IN: never. Every signal built here (mean_reversion, delivery_
breakout, oi_momentum, participant_tilt, vol_squeeze_breakout, and its
delay=2 variant) used the identical ExitRule(7, 2.0x, 2.5) - a default
carried over, not decided the way this project decides everything else
(train, placebo, held-out). market_gate swept its own 21-cell exit grid
and found "live values optimal on train, nothing works OOS" - but that was
on momentum/breakout signals over a survivorship-biased universe. Whether
that finding transfers to a mean-reversion signal on THIS project's honest
universe is an open, untested question, not an established one.

TWO-PASS DESIGN, to avoid the same free-parameter-tuned-to-outcome trap
already caught twice this project (market_gate's driver-smoothing episode;
this project's own entry-delay diagnostic naming it explicitly before
picking delay=2 on a mechanism criterion rather than a score). PASS 1
(this script): a full grid, TRADE-LEVEL ONLY (no placebo, no portfolio -
cheap), reported in FULL, not filtered down to whatever looks best. PASS 2
(a follow-up script, run only on a SHORTLISTED cell chosen on a stated
principle - never "whichever scored highest here"): full placebo +
portfolio-level, the real hypothesis test, logged if it clears the bar.

Tests mean_reversion because it has the most data (20,499 trades at the
current default) of any signal in the project.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config, set_seeds
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.engine.simulate import ExitRule, simulate_trades, trades_to_frame
from dtest.evaluate.metrics import non_overlapping_tstat, summary_stats
from dtest.features.technical import atr
from dtest.signals.mean_reversion import mean_reversion_signal
from dtest.universe import build_universe

TARGET_VALUE_PER_TRADE = 10_000.0

HOLD_DAYS_GRID = [3, 5, 7, 10, 15]
STOP_MULTIPLE_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
RISK_REWARD_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
CURRENT_DEFAULT = (7, 2.0, 2.5)


def _load_price_panels(cfg):
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in ("open", "high", "low", "close",
                                                 "volume", "turnover")}
    return panels, stocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="primary", choices=["primary", "delivery"])
    ap.add_argument("--window", default=None, choices=["train", "val", "test"], required=True)
    args = ap.parse_args()

    pd.set_option("display.width", 150)
    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()

    print(f"=== exit-geometry sweep: mean_reversion ({args.split}/{args.window}) ===")

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

    raw_signal = mean_reversion_signal(stock_close)
    signal = raw_signal & eligible
    print(f"signal: {int(signal.to_numpy().sum())} firings (fixed across the whole sweep)")

    a = atr(stock_high, stock_low, stock_close, window=14)

    grid = list(itertools.product(HOLD_DAYS_GRID, STOP_MULTIPLE_GRID, RISK_REWARD_GRID))
    print(f"\nsweeping {len(grid)} cells (hold_days x stop_multiple x risk_reward), "
         f"trade-level only, no placebo ...\n")
    print(f"{'hold':>5} {'stop':>5} {'R:R':>5} {'n':>7} {'mean_net%':>10} {'t_stat':>8} {'win%':>7} {'hit%':>7}")

    results = []
    for hold, stop_mult, rr in grid:
        rule = ExitRule(max_hold_days=hold, atr_stop_multiple=stop_mult, risk_reward=rr)
        trades = trades_to_frame(simulate_trades(
            signal, "long", rule, open_=stock_open, high=stock_high, low=stock_low,
            close=stock_close, volume=stock_volume, atr_panel=a,
            target_value_per_trade=TARGET_VALUE_PER_TRADE, cfg=cfg,
        ))
        stats = summary_stats(trades)
        tstat = non_overlapping_tstat(trades)
        row = {
            "hold_days": hold, "atr_stop_multiple": stop_mult, "risk_reward": rr,
            "n_resolved": stats.n_resolved, "mean_net_pct": stats.mean_net_pct,
            "t_stat": tstat["t_stat"], "win_rate_pct": stats.win_rate_pct,
            "hit_rate_pct": stats.hit_rate_pct if stats.hit_rate_pct is not None else float("nan"),
        }
        results.append(row)
        marker = " <- CURRENT DEFAULT" if (hold, stop_mult, rr) == CURRENT_DEFAULT else ""
        print(f"{hold:>5} {stop_mult:>5.1f} {rr:>5.1f} {stats.n_resolved:>7} "
             f"{stats.mean_net_pct:>10.4f} {tstat['t_stat']:>8.3f} "
             f"{stats.win_rate_pct:>7.2f} {row['hit_rate_pct']:>7.2f}{marker}")

    out = pd.DataFrame(results).sort_values("mean_net_pct", ascending=False)
    out_dir = Path(cfg.paths.runs) / "diagnostic_exit_geometry_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mean_reversion_{args.split}_{args.window}.csv"
    out.to_csv(out_path, index=False)

    print(f"\n=== TOP 10 BY mean_net_pct (full grid: {len(grid)} cells, all saved) ===")
    print(out.head(10).to_string(index=False))

    default_row = out[(out.hold_days == CURRENT_DEFAULT[0]) &
                       (out.atr_stop_multiple == CURRENT_DEFAULT[1]) &
                       (out.risk_reward == CURRENT_DEFAULT[2])]
    print(f"\nCurrent default (7, 2.0x, 2.5): mean_net_pct = "
         f"{default_row['mean_net_pct'].iloc[0]:.4f}")
    best = out.iloc[0]
    print(f"Best cell: hold={best.hold_days:.0f} stop={best.atr_stop_multiple:.1f} "
         f"R:R={best.risk_reward:.1f}  mean_net_pct={best.mean_net_pct:.4f}")

    print(f"\nWrote {out_path}")
    print("\nNOT logged to hypothesis_log.csv - this is a screening sweep. Any "
         "candidate must be chosen on a stated principle (not 'scored highest "
         "here') and re-tested with full placebo + portfolio-level before it "
         "means anything.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
