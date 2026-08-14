"""Reproducibility primitives. Everything else in this package obeys this module.

The rule this project is built on: **the same inputs must produce byte-identical
outputs, and a run must be able to prove which inputs it used.**

That rule exists because of a specific failure in the previous project
(market_gate): `statsmodels.MarkovRegression.fit(search_reps=20)` exposes no seed,
two fits on identical data disagreed enough to relabel symbols, and the
non-determinism reached shipped production code before anyone noticed. A plain
re-run with no new market data could change what the app told the user.

Three things are needed to make a result trustworthy, and all three are enforced
here rather than left to discipline:

1. **Seeds.** Set once, recorded in the manifest.
2. **Input identity.** Content hashes of every file read, so "we re-ran it and got
   a different number" can always be resolved into either a data change or a code
   change - never a mystery.
3. **No wall clock.** Every computation takes an explicit `as_of` date. A function
   that calls `date.today()` produces a different answer tomorrow for no reason
   the manifest can record, which makes reproduction impossible by construction.
   There is no `today()` helper in this package on purpose.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Single seed for the whole project. Recorded in every manifest.
SEED = 42

_HASH_CHUNK = 1 << 20  # 1 MiB


def set_seeds(seed: int = SEED) -> None:
    """Seed every RNG this project can reach.

    Call once at the top of any entry point. Libraries added later that carry
    their own RNG (sklearn's `random_state`, etc.) must be passed `seed`
    explicitly - a global seed does not reach a library that samples at
    construction time, which is exactly how the market_gate bug survived.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def file_hash(path: str | Path) -> str:
    """SHA-256 of a file's bytes, streamed so a 46 GB database does not load."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame: values, index, column names and dtypes.

    Deliberately NOT `to_csv()` - CSV rendering is float-formatting dependent and
    would call two frames equal that differ in the 16th decimal, which is the
    precise class of difference worth catching. `hash_pandas_object` hashes the
    underlying values, so this is exact.

    Columns are sorted before hashing so that column ORDER does not change the
    hash - two frames carrying the same data are the same data. Row order is NOT
    sorted, because for a time series row order is meaning, not presentation.
    """
    cols = sorted(df.columns.astype(str))
    ordered = df[cols] if list(df.columns.astype(str)) != cols else df
    h = hashlib.sha256()
    h.update(json.dumps(
        {"columns": cols, "dtypes": [str(ordered[c].dtype) for c in cols]},
        sort_keys=True,
    ).encode())
    h.update(pd.util.hash_pandas_object(ordered, index=True).to_numpy().tobytes())
    return h.hexdigest()


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash of a config dict. Key order is irrelevant; values are not."""
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()


def _git(*args: str) -> str | None:
    """Run a git command in the project root, or return None if git is unusable."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_state() -> dict[str, Any]:
    """Commit SHA plus whether the tree was dirty when the run started.

    `dirty=True` is not an error, but it means the commit SHA does not fully
    describe the code that ran, so the result is not reproducible from the repo
    alone. Reported rather than blocked - blocking would just push people to
    commit noise mid-experiment.
    """
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": sha,
        "dirty": bool(status) if status is not None else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


@dataclass
class RunManifest:
    """The audit record for one run. Written next to that run's outputs.

    A result without a manifest is not a result in this project - there is no way
    to tell later whether a number came from different data, different code or a
    different config. Small enough to commit; that is the point.
    """

    run_id: str
    as_of: str                                  # explicit; never date.today()
    seed: int = SEED
    started_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_utc: str | None = None
    git: dict[str, Any] = field(default_factory=git_state)
    config: dict[str, Any] = field(default_factory=dict)
    config_sha: str = ""
    inputs: dict[str, str] = field(default_factory=dict)   # label -> sha256
    outputs: dict[str, str] = field(default_factory=dict)  # label -> sha256
    env: dict[str, Any] = field(default_factory=lambda: {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        # BLAS thread count can change floating-point reduction order in linear
        # algebra. Recorded so a cross-machine mismatch is diagnosable instead of
        # being written off as "floating point".
        "threads": {
            v: os.environ.get(v)
            for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    })
    notes: str = ""

    def record_input(self, label: str, path: str | Path) -> None:
        self.inputs[label] = file_hash(path)

    def record_frame(self, label: str, df: pd.DataFrame) -> None:
        self.outputs[label] = frame_hash(df)

    def finish(self) -> None:
        self.finished_utc = datetime.now(timezone.utc).isoformat()
        self.config_sha = config_hash(self.config)

    def write(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True, default=str))
        return path

    @staticmethod
    def load(directory: str | Path) -> dict[str, Any]:
        return json.loads((Path(directory) / "manifest.json").read_text())


def assert_reproducible(a: RunManifest, b: RunManifest) -> None:
    """Raise unless two runs agree on every output hash.

    Used by the determinism test. Reports WHICH output diverged, because "the run
    is not reproducible" is not actionable and "the universe frame diverged but
    prices did not" is.
    """
    diffs = [
        k for k in sorted(set(a.outputs) | set(b.outputs))
        if a.outputs.get(k) != b.outputs.get(k)
    ]
    if diffs:
        raise AssertionError(
            "non-deterministic outputs: "
            + ", ".join(f"{k} ({a.outputs.get(k)} != {b.outputs.get(k)})" for k in diffs)
        )
