"""DIAGNOSTIC ONLY - block-bootstrap Monte Carlo over every hypothesis already
logged in `runs/hypothesis_log.csv`, answering a different question than the
t-stat/placebo pair already computed for each one: not "is the mean
distinguishable from zero" but "if history had shuffled differently, what
fraction of alternate outcomes would still have been profitable, and how wide
is that band."

    python scripts/monte_carlo_hypotheses.py

Resamples the REAL, ALREADY-SIMULATED trades saved under `runs/<hypothesis>/`
for each logged entry - no new backtest, no new data pull, no new hypothesis.
Nothing here is written to hypothesis_log.csv: this re-expresses evidence
that already exists, it does not test a new claim.

METHOD - circular block bootstrap over entry-WEEK buckets, not raw trades.
`dtest/evaluate/metrics.py::non_overlapping_tstat` already establishes why:
trades entered the same week substantially share one market draw, so
resampling individual trades as if independent overstates how much distinct
evidence exists. This script resamples the SAME weekly bucket means that
statistic is built on, in contiguous blocks of BLOCK_SIZE_BUCKETS consecutive
weeks (~1 month), so any real week-to-week clustering in the win/loss
sequence survives into the simulated paths - a plain i.i.d. trade resample
would not preserve that.

TWO NUMBERS REPORTED PER HYPOTHESIS, deliberately not conflated:
1. `prob_mean_positive` - across N resampled histories, the fraction where
   the resampled sample mean itself comes out positive. This is the
   sizing-independent, primary read (same convention `metrics.py` uses
   everywhere else in this project) - a bootstrap analogue of the existing
   t-stat, answering "how much does the positive/negative call above wobble
   under resampling."
2. `prob_compounded_positive` / `compounded_median_pct` - treats the
   resampled weekly-bucket sequence as if 100% of capital rotated through it
   sequentially, compounding week to week. Reported ONLY as an intuitive
   "what would a rupee have turned into" narrative number, NOT a portfolio
   simulation - it has no position limits, no concurrent-capital constraint,
   no slot cap, the exact caveat already attached to momentum's own
   portfolio-level CAGR in this project's 2026-08-18 entry (5-slot cap
   starving a ~40-name signal). Read it as a rough sense of magnitude, not
   as a claim this is a tradeable equity curve.

Both numbers are centered on the REAL observed sample, so they describe the
uncertainty AROUND the measured edge (a bootstrap confidence read), not an
independent probability the strategy is "truly" profitable - stated plainly
so this isn't oversold relative to what a bootstrap can actually claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config

N_SIMS = 10_000
BLOCK_SIZE_BUCKETS = 4          # ~1 month of weekly buckets per contiguous block
BASE_SEED = 20260819            # date this script was first written, fixed thereafter
MIN_BUCKETS_FOR_MC = 5          # below this, block bootstrap has too few blocks to mean anything

RUNS = Path(__file__).resolve().parent.parent / "runs"

# hypothesis_id -> (trades file relative to runs/, entry-date column)
# Every row verified against hypothesis_log.csv's own n_trades count before
# being trusted here (resolved-row count matches exactly, not assumed).
MANIFEST: dict[str, tuple[str, str]] = {
    "5d7650b1840e": ("mean_reversion_primary_train/trades.csv", "entry_date"),
    "61fcd3023c4b": ("delivery_breakout_delivery_train/trades.csv", "entry_date"),
    "80aff81a749e": ("oi_momentum_primary_train/trades.csv", "entry_date"),
    "ee447825d3a7": ("participant_tilt_delivery_train/trades.csv", "entry_date"),
    "663cebb9f054": ("vol_squeeze_breakout_primary_train/trades.csv", "entry_date"),
    "e39de46a0f24": ("vol_squeeze_breakout_delay2_primary_train/trades.csv", "entry_date"),
    # cdd796d6e171 (pairs_reversion, pre-rollforward-fix) has NO surviving raw
    # trades file - the 2026-08-18 rollforward fix re-ran and overwrote
    # runs/pairs_reversion_honest/ in place, and 0b11b017cef9 (below)
    # explicitly supersedes it per the log's own notes. Skipped, not guessed.
    "ddc14822fb70": ("price_action_long_primary_train/trades.csv", "entry_date"),
    "0b11b017cef9": ("pairs_reversion_honest/real_trades_primary_train.csv", "entry_fill_date"),
    "c55019a896e7": ("same_sector_pairing/random_primary_train.csv", "entry_fill_date"),
    "1d82fec2bbbc": ("same_sector_pairing/liquidity_primary_train.csv", "entry_fill_date"),
    "5dbcd00310b3": ("same_sector_pairing/random_primary_val.csv", "entry_fill_date"),
    "a7f9414d3392": ("same_sector_pairing/liquidity_primary_val.csv", "entry_fill_date"),
    "71357c1af8cd": ("same_sector_pairing/random_delivery_train.csv", "entry_fill_date"),
    "99a2610cabee": ("same_sector_pairing/liquidity_delivery_train.csv", "entry_fill_date"),
    "fac149cb6b0b": ("vol_squeeze_breakout_delivery_train/trades.csv", "entry_date"),
    "423c548cc0d3": ("momentum_primary_train/trades.csv", "entry_date"),
    # 2026-08-19: five hypotheses that had only ever been tested on `primary`,
    # newly run on `delivery`/train (see [[project-data-test-status]]).
    "0d8f2ac002c2": ("mean_reversion_delivery_train/trades.csv", "entry_date"),
    "c9d3ab9ffc36": ("oi_momentum_delivery_train/trades.csv", "entry_date"),
    "fc8303956df7": ("price_action_long_delivery_train/trades.csv", "entry_date"),
    "0151b95c725e": ("momentum_delivery_train/trades.csv", "entry_date"),
    "f68079c5b0b8": ("pairs_reversion_honest/real_trades_delivery_train.csv", "entry_fill_date"),
    # 2026-08-19: delivery/val confirmation run for momentum (train result
    # above beat every placebo) - the discipline this project always applies
    # before trusting a train-only result.
    "ee89fa192a3e": ("momentum_delivery_val/trades.csv", "entry_date"),
}


def _bucket_means(trades: pd.DataFrame, date_col: str) -> np.ndarray:
    """Chronologically-ordered entry-week bucket means, same unit
    `non_overlapping_tstat` bases its own t-stat on."""
    resolved = trades[trades["net_pnl_pct"].notna()].copy()
    resolved["bucket"] = pd.to_datetime(resolved[date_col]).dt.to_period("W")
    means = resolved.groupby("bucket")["net_pnl_pct"].mean().sort_index()
    return means.to_numpy(), float(resolved["net_pnl_pct"].mean()), len(resolved)


def _circular_block_bootstrap(bucket_means: np.ndarray, block_size: int,
                               n_sims: int, seed: int) -> np.ndarray:
    """N resampled histories of the same length as `bucket_means`, built from
    contiguous circular blocks of `block_size` consecutive buckets - preserves
    local week-to-week clustering, only the BLOCK SEQUENCE is randomized."""
    rng = np.random.default_rng(seed)
    n = len(bucket_means)
    n_blocks_needed = -(-n // block_size)  # ceil
    out = np.empty((n_sims, n), dtype=float)
    starts = rng.integers(0, n, size=(n_sims, n_blocks_needed))
    for i in range(n_sims):
        pieces = [bucket_means[np.arange(s, s + block_size) % n] for s in starts[i]]
        out[i] = np.concatenate(pieces)[:n]
    return out


def _simulate_one(hyp_id: str, rel_path: str, date_col: str, seed: int) -> dict | None:
    fpath = RUNS / rel_path
    trades = pd.read_csv(fpath)
    bucket_means, real_mean, n_resolved = _bucket_means(trades, date_col)
    n_buckets = len(bucket_means)
    if n_buckets < MIN_BUCKETS_FOR_MC:
        return None

    paths = _circular_block_bootstrap(bucket_means, BLOCK_SIZE_BUCKETS, N_SIMS, seed)
    sim_means = paths.mean(axis=1)
    sim_compounded = (np.prod(1.0 + paths / 100.0, axis=1) - 1.0) * 100.0

    mean_lo, mean_med, mean_hi = np.percentile(sim_means, [2.5, 50, 97.5])
    comp_lo, comp_med, comp_hi = np.percentile(sim_compounded, [2.5, 50, 97.5])

    return {
        "hypothesis_id": hyp_id,
        "n_trades": n_resolved,
        "n_buckets": n_buckets,
        "real_mean_pct": real_mean,
        "prob_mean_positive_pct": float((sim_means > 0).mean() * 100.0),
        "mean_ci_lo_pct": float(mean_lo),
        "mean_ci_median_pct": float(mean_med),
        "mean_ci_hi_pct": float(mean_hi),
        "prob_compounded_positive_pct": float((sim_compounded > 0).mean() * 100.0),
        "compounded_ci_lo_pct": float(comp_lo),
        "compounded_ci_median_pct": float(comp_med),
        "compounded_ci_hi_pct": float(comp_hi),
    }


def main() -> int:
    pd.set_option("display.width", 200)
    cfg = load_config()

    log = pd.read_csv(cfg.paths.runs / "hypothesis_log.csv")

    rows = []
    skipped = []
    for hyp_id in log["hypothesis_id"]:
        if hyp_id not in MANIFEST:
            skipped.append(hyp_id)
            continue
        rel_path, date_col = MANIFEST[hyp_id]
        seed = BASE_SEED + list(MANIFEST).index(hyp_id)
        result = _simulate_one(hyp_id, rel_path, date_col, seed)
        if result is None:
            skipped.append(hyp_id)
            continue
        rows.append(result)

    out = pd.DataFrame(rows).merge(
        log[["hypothesis_id", "title", "split", "window", "decision", "t_stat"]],
        on="hypothesis_id", how="left",
    )
    cols = ["hypothesis_id", "title", "split", "window", "decision", "n_trades", "n_buckets",
            "real_mean_pct", "t_stat", "prob_mean_positive_pct",
            "mean_ci_lo_pct", "mean_ci_median_pct", "mean_ci_hi_pct",
            "prob_compounded_positive_pct",
            "compounded_ci_lo_pct", "compounded_ci_median_pct", "compounded_ci_hi_pct"]
    out = out[cols]

    print(f"=== Monte Carlo (block bootstrap, N={N_SIMS}, block={BLOCK_SIZE_BUCKETS} weekly buckets) "
          f"over {len(out)}/{len(log)} logged hypotheses ===\n")
    display = out.copy()
    display["title"] = display["title"].str.slice(0, 38)
    for c in ["real_mean_pct", "t_stat", "prob_mean_positive_pct", "mean_ci_lo_pct",
              "mean_ci_median_pct", "mean_ci_hi_pct", "prob_compounded_positive_pct",
              "compounded_ci_lo_pct", "compounded_ci_median_pct", "compounded_ci_hi_pct"]:
        display[c] = display[c].round(2)
    print(display.to_string(index=False))

    if skipped:
        print(f"\nSkipped ({len(skipped)}): {', '.join(skipped)} - "
              f"no reusable raw-trades file on disk (see MANIFEST comment) or too few weekly buckets.")

    out_dir = RUNS / "monte_carlo_hypotheses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nWrote {out_dir / 'summary.csv'}")
    print("\nNOT logged to hypothesis_log.csv - re-expresses existing evidence, tests no new claim.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
