"""Liquidity-filtered momentum: does restricting 12-1 month momentum to the
MOST liquid slice of the eligible universe fix its train-window rejection -
on TRAIN first, cheaply (trade-level + placebo, no portfolio yet).

    python scripts/diagnostic_liquidity_momentum.py --split primary --window train

THE PREMISE UNDER TEST, stated before running anything (this file's own
existence is the pre-registration). `momentum` (2026-08-18) was rejected on
primary/train not because its raw mean was negative (it was POSITIVE,
+0.601%) but because it sat at only the 10th percentile of 30 placebo seeds
drawn blindly from the SAME ~200-name eligible pool on the SAME dates - top-
quintile trailing-return selection actively UNDERPERFORMED picking blindly.
Momentum is one of the most-replicated findings in equity market
literature, but it is also well-documented there as concentrated in liquid
names and prone to REVERSE among illiquid/small-float ones (lottery-demand
and liquidity-constraint effects crowd out the diffusion-of-information
story momentum's own STORY field rests on). This project's `universe.py`
already selects a turnover-ranked top-200(-buffered-to-250) pool - liquid
relative to the full listed market, but nothing has tested whether momentum
specifically needs a MUCH narrower, more liquid slice of even that pool to
show a real selection effect instead of an anti-selection one.

CONSTRUCTION: identical `momentum_signal` (dtest/signals/momentum.py, same
12-1 lookback/skip, same top-quintile-of-eligible-pool ranking - no change
to the signal itself) but called with a NARROWER `universe_membership`:
only the `LIQUID_N` most liquid names (by `UniverseResult.rank`, the exact
same trailing-turnover rank the point-in-time universe itself is built
from - no new liquidity metric invented) at each rebalance date, instead of
the full ~200-name banded pool. The placebo comparison draws from the SAME
narrowed pool, so this remains an apples-to-apples "does selection beat
blind picking WITHIN the liquid slice" test, not a different question.

LIQUID_N = 50 (a quartile of the full size=200 universe) - a round,
pre-stated cut, not tuned against this run's own result. If this screens
well, a broader N-sweep would be a legitimate NEXT question with its own
train/val discipline, not folded into this first pass.
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
from dtest.signals.momentum import LOOKBACK_DAYS, SKIP_DAYS, TOP_QUANTILE, momentum_signal
from dtest.universe import build_universe

TARGET_VALUE_PER_TRADE = 10_000.0
HOLD_DAYS = 21
LIQUID_N = 50


def _load_panels(cfg):
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in ("open", "high", "low", "close",
                                                 "volume", "turnover")}
    return panels, stocks


def _run_variant(label, signal, eligible, rule, cfg, n_seeds, *, open_, high, low,
                 close, volume):
    n_sig = int(signal.to_numpy().sum())
    print(f"\n--- {label}: {n_sig} signals ---")
    if n_sig == 0:
        print("  zero signals - skipped")
        return None

    trades = trades_to_frame(simulate_trades(
        signal, "long", rule, open_=open_, high=high, low=low, close=close,
        volume=volume, atr_panel=None, target_value_per_trade=TARGET_VALUE_PER_TRADE,
        cfg=cfg,
    ))
    stats = summary_stats(trades)
    tstat = non_overlapping_tstat(trades)
    placebos = run_placebos(
        signal, eligible, "long", rule, open_=open_, high=high, low=low, close=close,
        volume=volume, atr_panel=None, target_value_per_trade=TARGET_VALUE_PER_TRADE,
        cfg=cfg, n_seeds=n_seeds,
    )
    cmp = placebos.compare(stats, "mean_net_pct")

    print(f"  n_resolved         {stats.n_resolved}")
    print(f"  mean_net_pct       {stats.mean_net_pct:.4f}")
    print(f"  win_rate_pct       {stats.win_rate_pct:.2f}")
    print(f"  t_stat             {tstat['t_stat']:.3f}  (n_buckets={tstat['n_buckets']})")
    print(f"  placebo_mean       {cmp['placebo_mean']:.4f}")
    print(f"  placebo_max        {cmp['placebo_max']:.4f}")
    print(f"  beats_best_placebo {cmp['beats_best_placebo']}")
    print(f"  percentile         {cmp['percentile_vs_placebos']:.1f}")

    return {"label": label, "n_trades": stats.n_resolved, "mean_net_pct": stats.mean_net_pct,
            "win_rate_pct": stats.win_rate_pct, "t_stat": tstat["t_stat"],
            "placebo_mean": cmp["placebo_mean"], "placebo_max": cmp["placebo_max"],
            "beats_best_placebo": cmp["beats_best_placebo"],
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

    print(f"=== liquidity-filtered momentum diagnostic ({args.split}/{args.window}) ===")

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

    rebalances_in_window = [d for d in uni.rebalance_dates
                            if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    print(f"{len(rebalances_in_window)} rebalance dates in window")

    # `uni.rank` is the exact trailing-turnover rank the universe itself was
    # built from (1 = most liquid), populated only at rebalance dates - a
    # narrower liquid membership, not a new liquidity metric.
    liquid_membership = uni.membership & (uni.rank <= LIQUID_N)
    n_full = int(uni.membership.loc[rebalances_in_window].sum().sum())
    n_liquid = int(liquid_membership.loc[rebalances_in_window].sum().sum())
    print(f"  eligible-pool size at rebalances: full={n_full}, "
         f"liquid(top {LIQUID_N})={n_liquid}")

    full_signal = momentum_signal(
        stock_close, uni.membership, rebalances_in_window,
        lookback_days=LOOKBACK_DAYS, skip_days=SKIP_DAYS, top_quantile=TOP_QUANTILE,
    )
    liquid_signal = momentum_signal(
        stock_close, liquid_membership, rebalances_in_window,
        lookback_days=LOOKBACK_DAYS, skip_days=SKIP_DAYS, top_quantile=TOP_QUANTILE,
    )

    eligible_full = uni.membership & mask[:, None]
    eligible_liquid = liquid_membership & mask[:, None]

    rule = ExitRule(max_hold_days=HOLD_DAYS, atr_stop_multiple=None, risk_reward=None)
    n_seeds = args.seeds if args.seeds is not None else cfg.placebo_seeds

    kwargs = dict(open_=stock_open, high=stock_high, low=stock_low, close=stock_close,
                 volume=stock_volume)
    full = _run_variant(
        "FULL UNIVERSE (baseline, ~200 names)", full_signal & eligible_full,
        eligible_full, rule, cfg, n_seeds, **kwargs,
    )
    liquid = _run_variant(
        f"LIQUID-{LIQUID_N} (top {LIQUID_N} by turnover)", liquid_signal & eligible_liquid,
        eligible_liquid, rule, cfg, n_seeds, **kwargs,
    )

    print("\n=== SUMMARY ===")
    for r in (full, liquid):
        if r:
            print(f"  {r['label']:34s} n={r['n_trades']:6d} mean_net%={r['mean_net_pct']:8.4f} "
                 f"t={r['t_stat']:7.3f} pctile={r['percentile_vs_placebos']:6.1f} "
                 f"beats_best={r['beats_best_placebo']}")

    out_dir = Path(cfg.paths.runs) / "diagnostic_liquidity_momentum"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for r in (full, liquid) if r]).to_csv(
        out_dir / f"momentum_{args.split}_{args.window}.csv", index=False)
    print(f"\nWrote {out_dir}")
    print("\nNOT logged to hypothesis_log.csv - this is a train-only screening "
         "diagnostic. A candidate chosen here must be confirmed on val, not "
         "just look better than baseline on train, before it means anything.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
