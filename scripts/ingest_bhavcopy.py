"""Download NSE cash bhavcopy into the raw cache. Resumable; safe to re-run.

    python scripts/ingest_bhavcopy.py --start 2004-01-01 --end 2026-08-13
    python scripts/ingest_bhavcopy.py --start 2024-06-01 --end 2024-07-15 --probe

Fetch only. Parsing is a separate step (scripts/build_bhav_panels.py) so a
parser change never costs a re-download.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config
from dtest.data.bhavcopy import ingest, load_fetch_log, parse_day


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, type=dt.date.fromisoformat)
    ap.add_argument("--end", required=True, type=dt.date.fromisoformat)
    ap.add_argument("--probe", action="store_true",
                    help="after fetching, parse every day in range and report")
    ap.add_argument("--sleep", type=float, nargs=2, default=(0.9, 1.7),
                    metavar=("MIN", "MAX"))
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )

    cfg = load_config()
    cache = Path(cfg.paths.artifacts) / "bhav"

    n_weekdays = sum(
        1 for i in range((args.end - args.start).days + 1)
        if (args.start + dt.timedelta(days=i)).weekday() < 5
    )
    logging.info("range %s .. %s (%d weekdays) -> %s",
                 args.start, args.end, n_weekdays, cache)

    log_df = ingest(cache, args.start, args.end, sleep_range=tuple(args.sleep))

    if not log_df.empty:
        counts = log_df["status"].value_counts()
        logging.info("fetch log: %s", dict(counts))
        by_kind = log_df[log_df["status"] == "ok"]["kind"].value_counts()
        logging.info("formats: %s", dict(by_kind))
        errs = log_df[log_df["status"] == "error"]
        if not errs.empty:
            logging.warning("%d days errored (re-run to retry):\n%s",
                            len(errs), errs.head(10).to_string(index=False))

    if args.probe:
        ok_days = log_df[log_df["status"] == "ok"]["date"].dt.date.tolist()
        print(f"\nparsing {len(ok_days)} cached days ...")
        for d in ok_days:
            df = parse_day(cache, d)
            if df is None:
                print(f"  {d}  MISSING")
                continue
            eq = df[df["series"] == "EQ"]
            fmt = "udiff" if df["isin"].notna().any() else "legacy"
            print(f"  {d}  {fmt:6s} rows={len(df):5d} EQ={len(eq):5d} "
                  f"close_na={int(eq['close'].isna().sum()):3d} "
                  f"prevclose_na={int(eq['prev_close'].isna().sum()):3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
