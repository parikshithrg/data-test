"""Typed config loader. `config/config.toml` is the only place a constant lives.

Validates on load rather than at point of use, so a malformed split or a window
that runs past `as_of` fails immediately with a clear message instead of
producing a quietly truncated backtest three modules later.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Any, Literal

# Per-machine override for the derived-data cache directory, so a different
# machine (or a CI runner) never needs to edit the tracked config.toml just to
# relocate a regenerable cache. Mirrors the predecessor project's
# MARKET_GATE_RESEARCH_CACHE pattern for the identical reason: this working
# tree is OneDrive-synced, and the cache must not be.
ARTIFACTS_ENV_VAR = "DTEST_ARTIFACTS_DIR"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.toml"

WindowName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class Paths:
    price_dir: Path
    fno_db: Path
    industry_map: Path
    artifacts: Path
    runs: Path
    macro_dir: Path
    fundamentals_dir: Path
    shareholding_dir: Path
    insider_trading_dir: Path
    index_reconstitution_dir: Path

    def check_readable(self) -> None:
        """Fail loudly if source data is missing, before any run starts."""
        for label in ("price_dir", "fno_db", "industry_map"):
            p = getattr(self, label)
            if not p.exists():
                raise FileNotFoundError(f"config paths.{label} does not exist: {p}")


@dataclass(frozen=True)
class Split:
    """One train/val/test partition of the timeline.

    `name` matters: a result must always be reported against the split it used,
    because the delivery split has a third of the history and therefore nothing
    like the same statistical power.
    """

    name: str
    description: str
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date
    embargo_days: int

    def window(self, which: WindowName) -> tuple[date, date]:
        return getattr(self, f"{which}_start"), getattr(self, f"{which}_end")

    def validate(self, as_of: date) -> None:
        bounds = [
            ("train", self.train_start, self.train_end),
            ("val", self.val_start, self.val_end),
            ("test", self.test_start, self.test_end),
        ]
        for label, start, end in bounds:
            if start >= end:
                raise ValueError(f"split {self.name}.{label}: start {start} >= end {end}")
            if end > as_of:
                raise ValueError(
                    f"split {self.name}.{label} ends {end}, after as_of {as_of} - "
                    "a window cannot extend past the data"
                )
        for (a_name, _, a_end), (b_name, b_start, _) in zip(bounds, bounds[1:]):
            if b_start <= a_end:
                raise ValueError(
                    f"split {self.name}: {b_name} starts {b_start} on or before "
                    f"{a_name} ends {a_end} - windows must not overlap"
                )


@dataclass(frozen=True)
class Costs:
    brokerage_pct_per_side: float
    brokerage_cap_inr: float
    stt_buy_pct: float
    stt_sell_pct: float
    exchange_txn_pct_per_side: float
    sebi_pct_per_side: float
    stamp_duty_buy_pct: float
    gst_pct: float
    slippage_bps_per_side: float


@dataclass(frozen=True)
class FuturesCosts:
    """Statutory schedule for a STOCK FUTURES leg - deliberately a
    SEPARATE dataclass from `Costs`, not a reused one with different
    numbers plugged in, because the structure itself differs: futures
    STT is sell-side only (equity delivery charges both legs), brokerage
    is real and non-zero (Zerodha delivery is free; F&O is not), and
    stamp duty's rate differs from delivery's. Mixing the two schedules
    by accident would be a silent, expensive error - keeping them as
    distinct types makes that a type error instead."""
    brokerage_pct_per_side: float
    brokerage_cap_inr: float
    stt_sell_pct: float
    exchange_txn_pct_per_side: float
    sebi_pct_per_side: float
    stamp_duty_buy_pct: float
    gst_pct: float
    slippage_bps_per_side: float


@dataclass(frozen=True)
class Universe:
    rebalance: str
    size: int
    buffer_size: int
    lookback_days: int
    min_history_days: int
    max_staleness_days: int
    min_price: float

    def validate(self) -> None:
        if self.buffer_size < self.size:
            raise ValueError(
                f"universe.buffer_size ({self.buffer_size}) < size ({self.size}); "
                "the band must be at least as wide as the universe"
            )
        if self.rebalance != "monthly":
            raise ValueError(f"unsupported universe.rebalance: {self.rebalance!r}")


@dataclass(frozen=True)
class Execution:
    signal_at: str
    fill_at: str
    max_participation_pct: float

    def validate(self) -> None:
        # This is the correction the whole project exists to make. Guard it so
        # nobody can quietly "optimise" back to same-bar fills.
        if self.fill_at != "next_open":
            raise ValueError(
                f"execution.fill_at is {self.fill_at!r}. Only 'next_open' is "
                "supported: a signal computed from bar T's close cannot be "
                "filled at bar T's close, and same-bar fills are the single "
                "largest source of inflated results in the previous project."
            )


@dataclass(frozen=True)
class Portfolio:
    initial_capital: float
    max_positions: int
    position_sizing: str
    max_sector_weight_pct: float


@dataclass(frozen=True)
class Config:
    as_of: date
    project: str
    benchmark: str
    paths: Paths
    splits: dict[str, Split]
    costs: Costs
    futures_costs: FuturesCosts
    universe: Universe
    execution: Execution
    portfolio: Portfolio
    placebo_seeds: int
    raw: dict[str, Any]

    def split(self, name: str = "primary") -> Split:
        if name not in self.splits:
            raise KeyError(f"unknown split {name!r}; have {sorted(self.splits)}")
        return self.splits[name]

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the run manifest."""
        d = asdict(self)
        d.pop("raw")
        return d


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    as_of = raw["meta"]["as_of"]
    embargo = raw["splits"]["embargo_days"]

    splits: dict[str, Split] = {}
    for name, block in raw["splits"].items():
        if not isinstance(block, dict):   # embargo_days sits at this level
            continue
        splits[name] = Split(name=name, embargo_days=embargo, **block)

    paths_raw = dict(raw["paths"])
    env_override = os.environ.get(ARTIFACTS_ENV_VAR)
    if env_override:
        paths_raw["artifacts"] = env_override
    paths = Paths(**{
        k: _resolve(PROJECT_ROOT, v) for k, v in paths_raw.items()
    })

    cfg = Config(
        as_of=as_of,
        project=raw["meta"]["project"],
        benchmark=raw["benchmark"]["symbol"],
        paths=paths,
        splits=splits,
        costs=Costs(**raw["costs"]),
        futures_costs=FuturesCosts(**raw["futures_costs"]),
        universe=Universe(**raw["universe"]),
        execution=Execution(**raw["execution"]),
        portfolio=Portfolio(**raw["portfolio"]),
        placebo_seeds=raw["research"]["placebo_seeds"],
        raw=raw,
    )

    for s in splits.values():
        s.validate(as_of)
    cfg.universe.validate()
    cfg.execution.validate()
    return cfg
