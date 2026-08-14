"""The hypothesis log. Every variant tried, win or lose, append-only.

This exists because the predecessor project ran 15+ variants against the same
train window across three sessions without ever counting them. A result
significant at the 5% level is expected to appear roughly once in twenty by
chance alone - so a project that tries twenty things and reports the one that
worked has, on priors, found nothing. Nobody had a running count to weigh
against.

APPEND-ONLY, and REJECTIONS ARE LOGGED WITH THE SAME WEIGHT AS ACCEPTANCES.
The predecessor's own methodology notes said it plainly: "report rejections as
results" - two of three composite-term candidates being rejected on 2026-08-12
was itself the informative finding, more so than any single accepted change.
An entry, once written, is never edited or deleted; a later re-test of the same
idea is a NEW entry that references the old one, so the full history of what
was tried survives.

THE FILE IS THE STATE. `runs/hypothesis_log.csv`, one row per entry, committed
to git like everything else under `runs/` - so "how many things have been
tried against this split" is answerable by reading a file, not by scrolling
chat history.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DECISIONS = ("accepted", "rejected", "inconclusive")

COLUMNS = [
    "hypothesis_id", "logged_utc", "title", "story", "split", "window",
    "metric", "real_value", "placebo_max", "placebo_mean", "beats_best_placebo",
    "t_stat", "n_buckets", "n_trades", "decision", "notes", "supersedes",
]


@dataclass(frozen=True)
class HypothesisEntry:
    """One row. `story` is mandatory and checked non-empty at construction -
    the whole point of this log is refusing to let a signal search masquerade
    as a hypothesis test, and a blank economic story is the tell."""

    title: str
    story: str                     # the economic story - who's on the other side, and why wrong
    split: str                     # e.g. "primary" or "delivery"
    window: str                    # "train" | "val" | "test"
    metric: str
    real_value: float
    decision: str
    placebo_max: float | None = None
    placebo_mean: float | None = None
    beats_best_placebo: bool | None = None
    t_stat: float | None = None
    n_buckets: int | None = None
    n_trades: int | None = None
    notes: str = ""
    supersedes: str | None = None   # hypothesis_id of an earlier related entry
    hypothesis_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    logged_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.story.strip():
            raise ValueError(
                "story is mandatory: state the economic reasoning BEFORE "
                "logging a result. A signal search without a story is the "
                "exact mistake this log exists to prevent."
            )
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {self.decision!r}")
        if self.window not in ("train", "val", "test"):
            raise ValueError(f"window must be train/val/test, got {self.window!r}")

    def as_row(self) -> dict:
        d = asdict(self)
        return {k: d[k] for k in COLUMNS}


def load_log(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, parse_dates=["logged_utc"])


def append_entry(path: str | Path, entry: HypothesisEntry) -> pd.DataFrame:
    """Append one entry and return the full, updated log.

    Genuinely append-only: existing rows are read back and re-written verbatim,
    never mutated, and the new row is added at the end so file order is also
    chronological order.
    """
    path = Path(path)
    existing = load_log(path)
    new_row = pd.DataFrame([entry.as_row()])
    combined = pd.concat([existing, new_row], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def scoreboard(path: str | Path) -> pd.DataFrame:
    """One row per (split, window): how many hypotheses tried, how many survived.

    The number this project's discipline actually depends on - if train shows
    40 attempts and 3 acceptances, that context belongs next to every accepted
    result, not just in this log.
    """
    log = load_log(path)
    if log.empty:
        return pd.DataFrame(columns=["split", "window", "n_tried", "n_accepted", "n_rejected"])
    g = log.groupby(["split", "window"])
    return (
        pd.DataFrame({
            "n_tried": g.size(),
            "n_accepted": g.apply(lambda d: int((d["decision"] == "accepted").sum())),
            "n_rejected": g.apply(lambda d: int((d["decision"] == "rejected").sum())),
        })
        .reset_index()
        .sort_values(["split", "window"], kind="stable")
        .reset_index(drop=True)
    )
