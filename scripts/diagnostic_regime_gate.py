"""Phase 0: does gating mean_reversion to bull regimes actually help - on
TRAIN first, cheaply (trade-level + placebo, no portfolio yet).

    python scripts/diagnostic_regime_gate.py --split primary --window train

THE PREMISE UNDER TEST, stated before running anything (this file's own
existence is the pre-registration): "long positions only in a bull regime"
should be an easier signal to trade than an ungated one, because it removes
entries fighting a falling market. market_gate's own causal regime study
(`research/regime_periods.py`, 2026-08-13) found this exact idea real in
DESCRIPTIVE terms (leading_lagging genuinely is a calm-uptrend strategy) but
NOT tradeable as a rotation rule - a regime-conditional rule scored train
Sharpe 0.059 against test Sharpe 0.952, the signature of a rule that only
looked good because it was chosen after seeing the test-equivalent window.
This diagnostic is deliberately built NOT to repeat that mistake: it reports
TRAIN here, and only becomes a logged hypothesis (`scripts/
test_mean_reversion_regime_gated.py`, not this file) if a candidate is
chosen BEFORE looking at val - never after.

REGIME LABEL: `dtest.features.regime.trailing_return`, NIFTY50's own trailing
63-session return AS OF that day's own close - the same causal construction
market_gate's study used (not smoothed HMM, which leaks). Bull = trailing
return > 0. A NaN read (insufficient history) is treated as NOT bull, same
"unknown regime blocks entry, does not default to a free pass" convention
market_gate's own regime gate uses.

WHY MEAN_REVERSION. Most data of any signal in the project (20,499 trades at
default), and the pushback that prompted this ("wouldn't gating by regime
help") followed directly from mean_reversion's own -0.97%/trade result.
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
REGIME_LOOKBACK = 63


def _load_panels(cfg):
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in ("open", "high", "low", "close",
                                                 "volume", "turnover")}
    return panels, stocks


def _nifty_close(cfg, calendar) -> pd.Series | None:
    path = cfg.paths.price_dir / "NIFTY50_DAILY.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df.loc[df.index <= pd.Timestamp(cfg.as_of)]
    return df["close"].reindex(calendar).ffill()


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

    print(f"=== regime-gate diagnostic: mean_reversion ({args.split}/{args.window}) ===")

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

    nifty = _nifty_close(cfg, close.index)
    if nifty is None:
        print("NIFTY50_DAILY.csv not found - cannot build a regime label")
        return 1
    trail_ret = trailing_return(nifty, lookback=REGIME_LOOKBACK)
    # Three-way split, not a bull/~bull complement: an unknown regime (NaN
    # trailing return, insufficient history) blocks entry in BOTH directions,
    # same "unknown does not default to a free pass" convention market_gate's
    # own regime gate uses - NOT lumped into whichever gate is the complement.
    bull_mask = (trail_ret > 0).fillna(False)
    bear_mask = (trail_ret < 0).fillna(False)
    n_bull = int(bull_mask.reindex(close.index[mask]).sum())
    n_bear = int(bear_mask.reindex(close.index[mask]).sum())
    n_total = int(mask.sum())
    print(f"  regime: {n_bull}/{n_total} bull ({100 * n_bull / n_total:.1f}%), "
         f"{n_bear}/{n_total} bear ({100 * n_bear / n_total:.1f}%), "
         f"{REGIME_LOOKBACK}-session trailing NIFTY50 return")

    raw_signal = mean_reversion_signal(stock_close)
    a = atr(stock_high, stock_low, stock_close, window=14)
    rule = ExitRule(max_hold_days=7, atr_stop_multiple=2.0, risk_reward=2.5)
    n_seeds = args.seeds if args.seeds is not None else cfg.placebo_seeds

    ungated_signal = raw_signal & eligible
    bull_gated_signal = raw_signal & eligible & bull_mask.to_numpy()[:, None]
    bear_gated_signal = raw_signal & eligible & bear_mask.to_numpy()[:, None]

    kwargs = dict(open_=stock_open, high=stock_high, low=stock_low, close=stock_close,
                 volume=stock_volume, atr_panel=a)
    ungated = _run_variant("UNGATED (baseline)", ungated_signal, eligible, rule, cfg, n_seeds, **kwargs)
    gated = _run_variant("BULL-GATED", bull_gated_signal, eligible, rule, cfg, n_seeds, **kwargs)
    bear_gated = _run_variant("BEAR-GATED", bear_gated_signal, eligible, rule, cfg, n_seeds, **kwargs)

    print("\n=== SUMMARY ===")
    for r in (ungated, gated, bear_gated):
        if r:
            print(f"  {r['label']:22s} n={r['n_trades']:6d} mean_net%={r['mean_net_pct']:8.4f} "
                 f"t={r['t_stat']:7.3f} pctile={r['percentile_vs_placebos']:6.1f} "
                 f"beats_best={r['beats_best_placebo']}")

    out_dir = Path(cfg.paths.runs) / "diagnostic_regime_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for r in (ungated, gated, bear_gated) if r]).to_csv(
        out_dir / f"mean_reversion_{args.split}_{args.window}.csv", index=False)
    print(f"\nWrote {out_dir}")
    print("\nNOT logged to hypothesis_log.csv - this is a train-only screening "
         "diagnostic. A candidate chosen here must be confirmed on val, not "
         "just look better than baseline on train, before it means anything.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
