"""DIAGNOSTIC / REPORTING ONLY - computes the full per-hypothesis diagnosis
matrix (trade-count, win rate, expectancy, drawdown, CAGR, Sharpe, regime
breakdown, ...) for every hypothesis with real, honestly-simulated trades on
disk, from the ACTUAL saved trade files - nothing here is estimated or
fabricated. Feeds `scripts/build_hypothesis_report_pdf.py`.

    python scripts/build_hypothesis_report_data.py

Reuses `MANIFEST` from `monte_carlo_hypotheses.py` as the single source of
truth for which file backs which hypothesis_id, rather than re-deriving it -
that mapping was already verified once (every resolved-row count checked
against hypothesis_log.csv's own n_trades before being trusted) and should
not drift into a second, silently-different copy.

BULL/BEAR/SIDEWAYS SPLIT - a DESCRIPTIVE breakdown for this report only, not
a new hypothesis test. Uses the same causal feature and lookback Phase 0's
regime-gate screen already established (`dtest.features.regime.trailing_return`,
NIFTY50, 63-session lookback) but adds a +/-5% deadband for a genuine third
"sideways" bucket - Phase 0 itself only ever used a two-way >0/<0 split. The
5% deadband is a round, pre-stated descriptive convention, not fitted to any
outcome - stated here so it is never mistaken for a tuned parameter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dtest import load_config
from dtest.engine.portfolio import portfolio_metrics
from dtest.features.regime import trailing_return
from monte_carlo_hypotheses import MANIFEST  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"
SIDEWAYS_DEADBAND_PCT = 5.0  # +/- this trailing-63-session-return %, descriptive only

# Hypotheses whose trades are two-leg (long+short simultaneously) rather than
# single-leg long-only - drives which columns are computable at all.
PAIRS_HYPOTHESES = {
    "0b11b017cef9", "c55019a896e7", "1d82fec2bbbc", "5dbcd00310b3",
    "a7f9414d3392", "71357c1af8cd", "99a2610cabee", "f68079c5b0b8",
}


def _regime_bucket(entry_dates: pd.Series, regime: pd.Series) -> pd.Series:
    r = regime.reindex(pd.to_datetime(entry_dates)).to_numpy()
    out = np.full(len(r), "unknown", dtype=object)
    out[r > SIDEWAYS_DEADBAND_PCT / 100.0] = "bull"
    out[r < -SIDEWAYS_DEADBAND_PCT / 100.0] = "bear"
    out[(r >= -SIDEWAYS_DEADBAND_PCT / 100.0) & (r <= SIDEWAYS_DEADBAND_PCT / 100.0)] = "sideways"
    return pd.Series(out, index=entry_dates.index)


def _trade_stats(resolved: pd.DataFrame, gross_col: str, cost_total: pd.Series,
                  date_col: str, exit_date_col: str | None) -> dict:
    net = resolved["net_pnl_pct"]
    gross = resolved[gross_col] if gross_col in resolved.columns else net + cost_total

    wins = net[net > 0]
    losses = net[net < 0]
    gross_wins = gross[gross > 0].sum()
    gross_losses = gross[gross < 0].sum()

    if exit_date_col and exit_date_col in resolved.columns:
        held = (pd.to_datetime(resolved[exit_date_col]) - pd.to_datetime(resolved[date_col])).dt.days
        avg_held = float(held.mean())
    elif "held_days" in resolved.columns:
        avg_held = float(resolved["held_days"].mean())
    else:
        avg_held = float("nan")

    total_cost = float(cost_total.sum())
    total_gross = float(gross.sum())

    return {
        "n_trades": int(len(resolved)),
        "win_rate_pct": float((net > 0).mean() * 100.0),
        "avg_winner_pct": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loser_pct": float(losses.mean()) if len(losses) else float("nan"),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf"),
        "gross_expectancy_pct": float(gross.mean()),
        "net_expectancy_pct": float(net.mean()),
        "avg_holding_days": avg_held,
        "costs_pct_of_abs_gross": (total_cost / abs(total_gross) * 100.0
                                    if total_gross != 0 else float("nan")),
    }


def main() -> int:
    cfg = load_config()
    log = pd.read_csv(cfg.paths.runs / "hypothesis_log.csv")

    print("loading NIFTY50 close for the regime breakdown ...")
    nifty = pd.read_csv(cfg.paths.price_dir / "NIFTY50_DAILY.csv",
                         parse_dates=["date"]).set_index("date").sort_index()["close"]
    regime = trailing_return(nifty, lookback=63)

    rows = []
    for hyp_id in log["hypothesis_id"]:
        if hyp_id not in MANIFEST:
            continue
        rel_path, date_col = MANIFEST[hyp_id]
        fpath = RUNS / rel_path
        raw = pd.read_csv(fpath)
        resolved = raw[raw["net_pnl_pct"].notna()].copy()
        is_pairs = hyp_id in PAIRS_HYPOTHESES

        if is_pairs:
            cost_total = resolved["long_cost_pct"] + resolved["short_cost_pct"]
            stats = _trade_stats(resolved, "gross_pnl_pct", cost_total,
                                  date_col, "exit_fill_date")
            long_result = float(resolved["long_net_pct"].mean())
            short_result = float(resolved["short_net_pct"].mean())
        else:
            cost_total = resolved["cost_pct"]
            stats = _trade_stats(resolved, "gross_pnl_pct", cost_total,
                                  date_col, "exit_date" if "exit_date" in resolved.columns else None)
            long_result = stats["net_expectancy_pct"]
            short_result = float("nan")  # long-only design, no honest short leg simulated

        # portfolio-level (single-leg only - no portfolio simulation exists for pairs)
        port_dir = RUNS / Path(rel_path).parent
        eq_path = port_dir / "equity_curve.csv"
        if not is_pairs and eq_path.exists():
            eq = pd.read_csv(eq_path, parse_dates=["date"]).set_index("date")["equity"]
            pm = portfolio_metrics(eq)
        else:
            pm = {"cagr_pct": float("nan"), "sharpe": float("nan"), "max_drawdown_pct": float("nan")}

        bucket = _regime_bucket(resolved[date_col], regime)
        regime_means = resolved.groupby(bucket)["net_pnl_pct"].mean()

        row = {
            "hypothesis_id": hyp_id,
            **stats,
            "max_drawdown_pct": pm["max_drawdown_pct"],
            "cagr_pct": pm["cagr_pct"],
            "sharpe": pm["sharpe"],
            "long_only_result_pct": long_result,
            "short_only_result_pct": short_result,
            "bull_result_pct": float(regime_means.get("bull", float("nan"))),
            "bear_result_pct": float(regime_means.get("bear", float("nan"))),
            "sideways_result_pct": float(regime_means.get("sideways", float("nan"))),
            "is_pairs": is_pairs,
        }
        rows.append(row)
        print(f"  {hyp_id}: n={stats['n_trades']:6d}  win%={stats['win_rate_pct']:6.2f}  "
              f"net_exp%={stats['net_expectancy_pct']:7.3f}  CAGR%={pm['cagr_pct']:7.2f}")

    out = pd.DataFrame(rows).merge(
        log[["hypothesis_id", "title", "split", "window", "decision", "t_stat",
             "beats_best_placebo", "n_buckets"]],
        on="hypothesis_id", how="left",
    )

    out_dir = RUNS / "hypothesis_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "diagnosis_matrix.csv", index=False)
    print(f"\nWrote {out_dir / 'diagnosis_matrix.csv'} ({len(out)} hypotheses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
