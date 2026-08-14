"""The noise floor. A positive result means nothing until it clears this.

A placebo run fires the SAME NUMBER of signals on the SAME DATES as the real
hypothesis, but the names are drawn uniformly at random from that date's
point-in-time ELIGIBLE POOL rather than chosen by the hypothesis's own logic.
Matching dates and counts isolates the one question that matters: does the
SELECTION carry information, or would picking blindly from the same pool that
day have done just as well? A placebo with different dates or different counts
would also differ in market exposure, which is not the question being asked.

30 seeds is the floor, not a comfortable margin. The predecessor project ran 6
seeds once, got an empirical p of only ~0.14, and called it "suggestive, never
established" - `config.research.placebo_seeds = 30` exists specifically so that
mistake requires deliberately overriding a config value to repeat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dtest.config import Config
from dtest.engine.simulate import ExitRule, simulate_trades
from dtest.evaluate.metrics import SummaryStats, summary_stats


@dataclass(frozen=True)
class PlaceboResult:
    per_seed: pd.DataFrame          # one row per seed, every SummaryStats field
    n_seeds: int

    def band(self, metric: str) -> dict:
        """The min/mean/max of `metric` across placebo seeds - the noise floor."""
        vals = self.per_seed[metric].dropna()
        if vals.empty:
            return {"min": float("nan"), "mean": float("nan"), "max": float("nan"), "n": 0}
        return {"min": float(vals.min()), "mean": float(vals.mean()),
               "max": float(vals.max()), "n": int(len(vals))}

    def compare(self, real: SummaryStats, metric: str = "mean_net_pct") -> dict:
        """Where the real result sits relative to the placebo distribution."""
        real_val = getattr(real, metric)
        b = self.band(metric)
        vals = self.per_seed[metric].dropna().to_numpy()
        rank_pct = float((vals < real_val).mean() * 100.0) if len(vals) else float("nan")
        return {
            "metric": metric, "real": real_val, **{f"placebo_{k}": v for k, v in b.items()},
            "beats_best_placebo": bool(real_val > b["max"]) if b["n"] else None,
            "beats_mean_placebo": bool(real_val > b["mean"]) if b["n"] else None,
            "percentile_vs_placebos": rank_pct,
        }


def _placebo_signals(
    real_signals: pd.DataFrame,
    eligible_pool: pd.DataFrame,
    close: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """One random draw: same dates, same per-date counts, uniform-random names
    from that date's tradeable, priced pool."""
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(False, index=real_signals.index, columns=real_signals.columns)
    fired = real_signals.index[real_signals.any(axis=1)]

    cols = real_signals.columns
    for d in fired:
        k = int(real_signals.loc[d].sum())
        if k == 0:
            continue
        eligible = (eligible_pool.loc[d].reindex(cols, fill_value=False)
                   if d in eligible_pool.index else pd.Series(False, index=cols))
        priced = (close.loc[d].reindex(cols).notna()
                 if d in close.index else pd.Series(False, index=cols))
        mask = (eligible & priced).to_numpy()
        pool = cols[mask]
        if len(pool) == 0:
            continue
        k_eff = min(k, len(pool))
        pick = rng.choice(pool.to_numpy(), size=k_eff, replace=False)
        out.loc[d, pick] = True
    return out


def run_placebos(
    real_signals: pd.DataFrame,
    eligible_pool: pd.DataFrame,
    direction: str,
    exit_rule: ExitRule,
    *,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    atr_panel: pd.DataFrame | None,
    target_value_per_trade: float,
    cfg: Config,
    n_seeds: int | None = None,
    base_seed: int = 42,
) -> PlaceboResult:
    """Run `n_seeds` (default: `cfg.research.placebo_seeds`) random-selection
    placebos under IDENTICAL mechanics to the real hypothesis - same exit rule,
    same cost model, same execution assumptions. Only the name selection differs.
    """
    n = n_seeds if n_seeds is not None else cfg.placebo_seeds
    rows = []
    for i in range(n):
        sig = _placebo_signals(real_signals, eligible_pool, close, base_seed + i)
        trades = simulate_trades(
            sig, direction, exit_rule, open_=open_, high=high, low=low, close=close,
            volume=volume, atr_panel=atr_panel,
            target_value_per_trade=target_value_per_trade, cfg=cfg,
        )
        from dtest.engine.simulate import trades_to_frame
        stats = summary_stats(trades_to_frame(trades))
        rows.append({"seed": base_seed + i, **stats.as_dict()})

    return PlaceboResult(per_seed=pd.DataFrame(rows), n_seeds=n)
