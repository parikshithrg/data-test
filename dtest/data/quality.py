"""Data quality and corporate-action audit. Run before trusting any backtest.

This exists because of two specific holes in the predecessor project:

**Corporate actions were never audited at all.** If a 1:2 split or a 1:1 bonus is
not price-adjusted, the price halves overnight and every momentum signal reads a
-50% crash that never happened. Nobody ever checked. The check is cheap: an
unadjusted split leaves a pile-up of returns sitting almost exactly on simple
fractions (1/2, 1/5, 1/10), which random price moves do not do.

**Bad rows were found by crashing on them.** `NIFTY50DIVPOINT_DAILY.csv` is
zero-filled for 55 of 632 rows, and it was discovered when a full-universe run
divided by a zero price. Index "volume" is 0 for 99.9% of history, which silently
turned a rolling z-score into nonsense. Both are found here now, before a run,
rather than by a traceback during one.

Nothing here mutates data. The audit REPORTS; excluding a symbol is a universe
decision, made under a rule, not a silent repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dtest.data.prices import Panels

# A move this large in one session is possible but rare enough to be worth
# inspecting. NSE circuit limits are commonly 20%, so anything past ~35% in a
# liquid name is more likely to be a data artifact than a trade.
JUMP_THRESHOLD = 0.35

# Ratios a price lands on after an unadjusted split or bonus. A 1:1 bonus and a
# 1:2 split both halve the price - indistinguishable here, and it does not
# matter, because the remedy is the same.
SPLIT_RATIOS = (1/2, 1/3, 1/4, 1/5, 1/10, 2/3, 3/4, 2/5, 3/5, 1/20, 1/100)
RATIO_TOLERANCE = 0.02   # within 2% of an exact fraction


@dataclass
class QualityReport:
    n_symbols: int
    n_dates: int
    issues: pd.DataFrame            # one row per (symbol, check) with a count
    split_candidates: pd.DataFrame  # one row per suspected unadjusted action
    coverage: pd.DataFrame          # per symbol: first/last date, rows, gaps
    notes: list[str] = field(default_factory=list)

    def symbols_failing(self, check: str) -> list[str]:
        if self.issues.empty:
            return []
        hit = self.issues[(self.issues["check"] == check) & (self.issues["count"] > 0)]
        return sorted(hit["symbol"].unique())

    def summary(self) -> pd.DataFrame:
        """Counts per check, across the universe."""
        if self.issues.empty:
            return pd.DataFrame(columns=["check", "symbols_affected", "rows_affected"])
        g = self.issues[self.issues["count"] > 0].groupby("check")
        return (
            pd.DataFrame({
                "symbols_affected": g["symbol"].nunique(),
                "rows_affected": g["count"].sum(),
            })
            .reset_index()
            .sort_values("rows_affected", ascending=False, kind="stable")
            .reset_index(drop=True)
        )


def _ratio_is_split_like(ratio: float) -> float | None:
    """Return the matched fraction if `ratio` sits on one, else None."""
    for r in SPLIT_RATIOS:
        if abs(ratio - r) <= RATIO_TOLERANCE * r:
            return r
    return None


def audit(panels: Panels) -> QualityReport:
    o, h, l, c, v = (panels.open, panels.high, panels.low, panels.close, panels.volume)
    rows: list[dict] = []
    notes: list[str] = []

    present = c.notna()

    checks = {
        # A traded bar cannot have a non-positive price. This is the
        # zero-filled-row signature.
        "nonpositive_price": present & (c <= 0),
        # High below low is impossible; it means the columns are wrong.
        "high_lt_low": present & (h < l),
        # The bar's range must contain its own open and close.
        "ohlc_inconsistent": present & ((h < o) | (h < c) | (l > o) | (l > c)),
        # Zero volume on a bar with a real price: an index series, or a halt.
        # Not an error - but a z-score over a mostly-zero window is meaningless,
        # so anything ranking on volume must know.
        "zero_volume": present & (v.fillna(0) <= 0),
        "missing_volume": present & v.isna(),
        # All four fields equal. The signature of this dataset's synthetic
        # "index-close" source, which records a closing level and back-fills the
        # other three. A bar with no range cannot fill a stop or a target
        # honestly, so any simulator touching intraday levels must know.
        "flat_ohlc": present & (h == l) & (o == c) & (h == o),
    }
    for name, mask in checks.items():
        counts = mask.sum()
        for sym, n in counts.items():
            if n:
                rows.append({"symbol": sym, "check": name, "count": int(n)})

    # ---- corporate actions -------------------------------------------------
    ret = c / c.shift(1)
    big = (ret.notna()) & ((ret - 1).abs() > JUMP_THRESHOLD)
    split_rows: list[dict] = []
    for sym in c.columns:
        idx = big.index[big[sym].to_numpy()]
        for d in idx:
            r = float(ret.at[d, sym])
            matched = _ratio_is_split_like(r)
            prev = c[sym].shift(1).at[d]
            split_rows.append({
                "symbol": sym,
                "date": d,
                "prev_close": float(prev),
                "close": float(c.at[d, sym]),
                "ratio": r,
                "pct_move": (r - 1) * 100,
                "matched_fraction": matched,
                "likely_unadjusted_action": matched is not None,
                # A real corporate action changes the price without a
                # corresponding surge in traded value. A genuine -40% news day
                # normally trades heavily. Weak evidence, recorded not acted on.
                "volume_ratio_vs_20d": _vol_ratio(v[sym], d),
            })
    split_candidates = pd.DataFrame(split_rows)
    if not split_candidates.empty:
        split_candidates = split_candidates.sort_values(
            ["likely_unadjusted_action", "symbol", "date"],
            ascending=[False, True, True], kind="stable",
        ).reset_index(drop=True)

    n_flagged = int(split_candidates["likely_unadjusted_action"].sum()) if not split_candidates.empty else 0
    n_big = len(split_candidates)
    if n_big:
        share = n_flagged / n_big
        notes.append(
            f"{n_big} single-session moves over {JUMP_THRESHOLD:.0%}; {n_flagged} "
            f"({share:.1%}) land on a split-like fraction."
        )
        # Random large moves hit an exact fraction only by chance. The tolerance
        # bands cover roughly 8% of the plausible ratio range below 1, so a share
        # far above that is evidence of unadjusted actions rather than of news.
        notes.append(
            "Interpretation: a share near ~8% is chance; well above it means the "
            "series carries unadjusted splits/bonuses and is unsafe for returns."
        )

    # ---- coverage ----------------------------------------------------------
    cov_rows = []
    for sym in c.columns:
        s = c[sym].dropna()
        if s.empty:
            cov_rows.append({"symbol": sym, "first": pd.NaT, "last": pd.NaT,
                             "rows": 0, "span_days": 0, "gap_rate": np.nan})
            continue
        first, last = s.index[0], s.index[-1]
        # Bars this symbol is missing while the market as a whole traded.
        market_days = c.index[(c.index >= first) & (c.index <= last)]
        cov_rows.append({
            "symbol": sym,
            "first": first,
            "last": last,
            "rows": int(s.size),
            "span_days": int(len(market_days)),
            "gap_rate": float(1 - s.size / len(market_days)) if len(market_days) else np.nan,
        })
    coverage = pd.DataFrame(cov_rows).sort_values("symbol", kind="stable").reset_index(drop=True)

    return QualityReport(
        n_symbols=c.shape[1],
        n_dates=c.shape[0],
        issues=pd.DataFrame(rows, columns=["symbol", "check", "count"]),
        split_candidates=split_candidates,
        coverage=coverage,
        notes=notes,
    )


def _vol_ratio(vol: pd.Series, on: pd.Timestamp) -> float:
    """Volume on `on` relative to the prior 20 sessions' median. NaN if unknown."""
    pos = vol.index.get_loc(on)
    if not isinstance(pos, int) or pos < 20:
        return float("nan")
    window = vol.iloc[pos - 20:pos]
    med = window.median()
    if not np.isfinite(med) or med <= 0:
        return float("nan")
    return float(vol.iloc[pos] / med)
