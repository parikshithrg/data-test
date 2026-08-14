"""Price loading: source CSVs -> aligned wide panels, content-addressed.

Four properties this module guarantees. The first two were missing in the
predecessor project and caused real bugs there; the last two were found by
auditing this data on 2026-08-14 and are documented at the point of the fix.

**Nothing past `as_of` is ever loaded.** Not filtered downstream - not loaded.
`universe.read_local_daily` in the old project filtered only `df.index >= cutoff`
with no upper bound, so an ATR helper silently returned TODAY's value regardless
of the historical date it was asked for. A lower bound alone is not a time
machine.

**The cache is keyed by content, not by name.** The cache path derives from the
size+mtime of every source file plus `as_of`, so a changed CSV cannot be served
from a stale cache. The old project's caches were keyed by symbol and had to be
invalidated by hand.

**Dates are normalised to midnight, and same-day duplicates are resolved under a
stated rule.** 29 of 430 source files carry a time component and 25 contain TWO
DIFFERENT SERIES merged together - e.g. NIFTY100LOWVOL30 on 2015-11-16 has a
flat synthetic row (6052.90 for all four fields) at 00:00:00 and a real-OHLC row
at a completely different level (2990.85) at 09:00:07. Deduplicating naively
alternates between the two series and manufactures +/-100% daily moves.

**The trading calendar is defined by a quorum of stocks, not by the union of
every file.** The union runs to 8,458 days over 26 years against NSE's ~6,650,
because ETF and NAV series in this directory publish on their own calendars. A
union calendar puts NaN holes into every stock on days the market never traded,
which then read as gaps.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from dtest.config import Config
from dtest.determinism import RunManifest

FIELDS = ("open", "high", "low", "close", "volume")

# Explicit dtypes. Letting pandas infer means a column of clean ints in one file
# and one stray blank in another produce different dtypes for the same field,
# which changes the frame hash without changing the data.
_DTYPES = {f: "float64" for f in FIELDS}

# A date counts as a trading session when at least this share of the stocks that
# were alive on that date actually printed a close. 0.6 is deliberately loose:
# it must survive genuine partial sessions and thin early-2000s history while
# still rejecting a date that only an ETF NAV series knows about.
CALENDAR_QUORUM = 0.60


@dataclass(frozen=True)
class LoadReport:
    """What loading had to fix. Surfaced, never silent."""

    timed_rows: dict[str, int] = field(default_factory=dict)
    duplicate_days: dict[str, int] = field(default_factory=dict)
    dropped_dates: int = 0
    calendar_days: int = 0
    union_days: int = 0

    def frame(self) -> pd.DataFrame:
        syms = sorted(set(self.timed_rows) | set(self.duplicate_days))
        return pd.DataFrame({
            "symbol": syms,
            "timed_rows": [self.timed_rows.get(s, 0) for s in syms],
            "duplicate_days": [self.duplicate_days.get(s, 0) for s in syms],
        })


@dataclass(frozen=True)
class Panels:
    """Wide (date x symbol) frames, one per OHLCV field, all sharing one index.

    Aligned on a single trading calendar so every field is index-identical - the
    caller can rely on `close.loc[d, s]` and `open.loc[d, s]` describing the same
    bar without re-checking. Missing bars are NaN, never forward-filled: a filled
    bar is a fabricated observation, and in the old project forward-filling into
    a joined frame produced fabricated 0% returns that dragged correlations
    toward zero.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    as_of: date
    inventory_sha: str
    report: LoadReport = field(default_factory=LoadReport)

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def field(self, name: str) -> pd.DataFrame:
        if name not in FIELDS:
            raise KeyError(f"unknown field {name!r}; have {FIELDS}")
        return getattr(self, name)

    def subset(self, symbols: list[str]) -> "Panels":
        keep = [s for s in self.close.columns if s in set(symbols)]
        return Panels(
            **{f: self.field(f)[keep] for f in FIELDS},
            as_of=self.as_of,
            inventory_sha=self.inventory_sha,
            report=self.report,
        )

    def turnover(self) -> pd.DataFrame:
        """Traded value per bar. The liquidity measure the universe rule ranks on.

        close x volume, not a market-cap proxy: shares outstanding are not in this
        dataset and inventing them would be a guess dressed as a measurement.
        """
        return self.close * self.volume


def source_inventory(cfg: Config) -> dict[str, Path]:
    """Every price file, symbol -> path, in sorted order.

    Sorted because iteration order reaches the output: column order changes a
    frame's layout, and an unsorted `glob` is filesystem-dependent.
    """
    files = sorted(Path(cfg.paths.price_dir).glob("*_DAILY.csv"))
    return {f.name.replace("_DAILY.csv", ""): f for f in files}


def _inventory_sha(inventory: dict[str, Path], as_of: date, tag: str) -> str:
    """One hash standing for 'this exact set of source bytes, up to this date'.

    Hashes file SIZE and MTIME rather than full contents: 77 MB of SHA-256 on
    every call would be paid for no benefit, and full content hashes go into the
    run manifest anyway, which is where the audit trail belongs. This value is a
    CACHE KEY, not evidence.
    """
    h = hashlib.sha256()
    h.update(f"{as_of}|{tag}|quorum={CALENDAR_QUORUM}".encode())
    for sym, path in inventory.items():
        st = path.stat()
        h.update(f"{sym}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _resolve_duplicate_days(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple rows on one calendar day to exactly one.

    Rule, applied in order, and chosen to be deterministic rather than clever:

    1. **Prefer a row with real intraday range** (`high > low`) over a flat row
       where all four fields are equal. A flat row is the synthetic signature of
       this dataset's "index-close" EOD source, which records only a closing
       level. Where a file merges that source with a genuine OHLC history, the
       two sit at different price levels and alternating between them fabricates
       enormous daily moves.
    2. **Among survivors, keep the last** in file order, i.e. the most recently
       written vintage.

    This produces ONE internally consistent series. It does not attempt to decide
    which of two merged series is the *correct* one - that is unknowable from the
    file alone, so any symbol needing this is reported and the universe rule is
    left to exclude it.
    """
    day = df.index.normalize()
    if not day.has_duplicates:
        out = df.copy()
        out.index = day
        return out

    has_range = (df["high"] > df["low"]).to_numpy()
    order = np.arange(len(df))
    tmp = pd.DataFrame({"_day": day, "_range": has_range, "_ord": order}, index=df.index)
    # Sort so the preferred row lands last within each day, then keep the last.
    keep_pos = (
        tmp.sort_values(["_day", "_range", "_ord"], kind="stable")
        .groupby("_day", sort=True)["_ord"].last()
        .to_numpy()
    )
    out = df.iloc[np.sort(keep_pos)].copy()
    out.index = out.index.normalize()
    return out


def _read_one(path: Path, as_of: date) -> tuple[pd.DataFrame, int, int]:
    df = pd.read_csv(path, dtype=_DTYPES, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[df.index <= pd.Timestamp(as_of)]   # truncate on the way IN
    if df.empty:
        return df, 0, 0

    n_timed = int((df.index.normalize() != df.index).sum())
    n_dupes = int(df.index.normalize().duplicated().sum())
    return _resolve_duplicate_days(df), n_timed, n_dupes


def _trading_calendar(close: pd.DataFrame) -> pd.DatetimeIndex:
    """Dates on which a quorum of live stocks traded.

    "Live" means between a symbol's own first and last observation, so a symbol
    that had not listed yet never counts against a date's quorum - otherwise the
    early 2000s, when only a handful of names exist, would fail every test.
    """
    present = close.notna()
    first = present.idxmax()
    # idxmax returns the first index for an all-False column too; mask those out.
    any_present = present.any()
    last = present[::-1].idxmax()

    alive = pd.DataFrame(False, index=close.index, columns=close.columns)
    for sym in close.columns:
        if not any_present[sym]:
            continue
        alive.loc[first[sym]:last[sym], sym] = True

    n_alive = alive.sum(axis=1)
    n_traded = (present & alive).sum(axis=1)
    share = np.where(n_alive > 0, n_traded / n_alive.replace(0, np.nan), 0.0)
    return close.index[pd.Series(share, index=close.index).fillna(0) >= CALENDAR_QUORUM]


def build_panels(
    cfg: Config,
    manifest: RunManifest | None = None,
    *,
    symbols: list[str] | None = None,
    calendar_from: list[str] | None = None,
    use_cache: bool = True,
) -> Panels:
    """Load symbols into aligned wide panels, caching by content key.

    `symbols`       - restrict to these (default: everything on disk).
    `calendar_from` - define the trading calendar from this subset only. Pass the
                      stock list: indices and NAV series keep their own calendars
                      and would otherwise inject days the market never traded.
    `manifest`      - if given, record full SHA-256 hashes of every source file
                      into the run's audit record. That is the expensive,
                      evidential hash; the cache key above is the cheap one.
    """
    inventory = source_inventory(cfg)
    if not inventory:
        raise FileNotFoundError(f"no *_DAILY.csv under {cfg.paths.price_dir}")
    if symbols is not None:
        keep = set(symbols)
        inventory = {k: v for k, v in inventory.items() if k in keep}
        if not inventory:
            raise ValueError("none of the requested symbols have price files")

    tag = "all" if symbols is None else hashlib.sha256(
        "|".join(sorted(inventory)).encode()).hexdigest()[:8]
    cal_tag = "self" if calendar_from is None else hashlib.sha256(
        "|".join(sorted(calendar_from)).encode()).hexdigest()[:8]
    sha = _inventory_sha(inventory, cfg.as_of, f"{tag}/{cal_tag}")
    cache_dir = Path(cfg.paths.artifacts) / "panels" / sha

    report = LoadReport()
    if use_cache and (cache_dir / "close.parquet").exists():
        frames = {f: pd.read_parquet(cache_dir / f"{f}.parquet") for f in FIELDS}
    else:
        per_field: dict[str, dict[str, pd.Series]] = {f: {} for f in FIELDS}
        timed: dict[str, int] = {}
        dupes: dict[str, int] = {}
        for sym, path in inventory.items():          # already sorted
            df, n_timed, n_dupes = _read_one(path, cfg.as_of)
            if n_timed:
                timed[sym] = n_timed
            if n_dupes:
                dupes[sym] = n_dupes
            if df.empty:
                continue
            for f in FIELDS:
                per_field[f][sym] = df[f]

        frames = {
            f: pd.DataFrame(cols).sort_index().astype("float64")
            for f, cols in per_field.items()
        }
        union = frames["close"].index
        cal_syms = [s for s in (calendar_from or frames["close"].columns)
                    if s in frames["close"].columns]
        calendar = _trading_calendar(frames["close"][cal_syms])
        frames = {f: v.reindex(calendar) for f, v in frames.items()}

        report = LoadReport(
            timed_rows=timed, duplicate_days=dupes,
            dropped_dates=len(union) - len(calendar),
            calendar_days=len(calendar), union_days=len(union),
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        for f, v in frames.items():
            v.to_parquet(cache_dir / f"{f}.parquet")

    if manifest is not None:
        for sym, path in inventory.items():
            manifest.record_input(f"price:{sym}", path)

    return Panels(**frames, as_of=cfg.as_of, inventory_sha=sha, report=report)
