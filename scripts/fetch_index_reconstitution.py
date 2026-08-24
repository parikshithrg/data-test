"""One-time acquisition: fetch and parse NSE Indices Limited's full
press-release archive at niftyindices.com/press-release into a single
consolidated events table (symbol-level index additions/exclusions).
NOT a live call inside the deterministic harness - same category as
every other `fetch_*.py` script in this project.

    python scripts/fetch_index_reconstitution.py
    python scripts/fetch_index_reconstitution.py --limit 50   # subset, for testing

ONE LIST PAGE, then one PDF fetch per candidate release - the press-
release LIST itself is fully present in one static HTML page (1,473
entries, 1998-2026, confirmed live 2026-08-24), so unlike every other
fetch script here there is no metadata-pagination step at all. A TITLE
FILTER (`_is_relevant`) skips releases about unrelated fixed-income/debt/
SDL/money-market indices and pure product-launch announcements before
spending a request on them - real, but not equity-constituent-change
content this project's hypothesis category needs.

REAL COVERAGE FLOOR IS ~2010, not 1998 - `dtest/data/index_reconstitution.
py`'s own `parse_press_release_pdf` only emits an event when a release's
own table declared a `Symbol` column (confirmed absent on every pre-2010
sample checked); pre-2010 releases are still fetched (cheap, already
paid for by being on the list) but will correctly contribute zero rows.

RATE LIMITING: same courtesy every other fetch script in this project
extends, applied here too even though niftyindices.com is a different
host from nseindia.com's own endpoints.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config
from dtest.data.index_reconstitution import SCHEMA, parse_press_release_pdf

LIST_URL = "https://www.niftyindices.com/press-release"
DELAY_SECONDS = 0.3
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

_LIST_ITEM_RE = re.compile(
    r'data-date="([^"]+)"[^>]*>\s*<p>[^<]*</p>\s*<a href=\'([^\']+)\'[^>]*>([^<]+)</a>')

# Titles containing any of these substrings are skipped - real releases,
# but not equity-index constituent changes (confirmed by title inspection
# of a 40-symbol sample of the full 1,473-entry list, 2026-08-24).
_EXCLUDE_KEYWORDS = (
    "fixed income", "sdl", "g-sec", "gsec", "bond", "debt", "money market",
    "maturity", "launch of", "methodology", "back-testing", "backtesting",
)
_INCLUDE_KEYWORDS = ("change", "replacement", "replace", "inclusion", "exclusion")


def _is_relevant(title: str) -> bool:
    low = title.lower()
    if any(k in low for k in _EXCLUDE_KEYWORDS):
        return False
    return any(k in low for k in _INCLUDE_KEYWORDS)


def _fetch_list(session: requests.Session) -> list[tuple[pd.Timestamp, str, str]]:
    r = session.get(LIST_URL, timeout=20)
    r.raise_for_status()
    items = []
    seen_urls: set[str] = set()
    n_dupe = 0
    for date_str, url, title in _LIST_ITEM_RE.findall(r.text):
        title = title.replace("&amp;", "&")
        if not _is_relevant(title):
            continue
        try:
            announcement_date = pd.Timestamp(pd.to_datetime(date_str, format="%b %d, %Y"))
        except ValueError:
            continue
        if not url.lower().endswith(".pdf"):
            # a handful of real entries on this list are missing the dot
            # before "pdf" (a genuine typo in NSE's own HTML) - confirmed
            # live, 2026-08-24, e.g. "ind_prs16122008pdf"
            url = re.sub(r'pdf$', '.pdf', url, flags=re.I)
        full_url = url if url.startswith("http") else f"https://www.niftyindices.com{url}"
        if full_url in seen_urls:
            # the list itself has real duplicate entries (confirmed live,
            # 2026-08-24: ind_prs09052017.pdf appears twice, identical date
            # and title, on NSE's own page) - dedupe by url, not a parsing bug
            n_dupe += 1
            continue
        seen_urls.add(full_url)
        items.append((announcement_date, full_url, title))
    if n_dupe:
        print(f"  ({n_dupe} duplicate list entries skipped)")
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the first N releases, for testing")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(cfg.paths.index_reconstitution_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    print("fetching the full press-release list ...")
    items = _fetch_list(session)
    print(f"  {len(items)} relevant releases found (title-filtered)")
    if args.limit:
        items = items[:args.limit]
        print(f"  --limit: processing first {len(items)}")

    all_rows = []
    n_ok, n_zero, n_err = 0, 0, 0
    t0 = time.time()
    for i, (announcement_date, url, title) in enumerate(items):
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            events = parse_press_release_pdf(r.content)
        except Exception as e:
            print(f"    [{i + 1}/{len(items)}] {url} ERROR: {e}")
            n_err += 1
            time.sleep(DELAY_SECONDS)
            continue

        if events:
            n_ok += 1
        else:
            n_zero += 1
        for ev in events:
            all_rows.append({
                "symbol": ev.symbol, "company_name": ev.company_name, "index_name": ev.index_name,
                "action": ev.action, "announcement_date": announcement_date,
                "effective_date": ev.effective_date, "source_url": url,
            })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  ... {i + 1}/{len(items)} releases processed ({elapsed:.0f}s elapsed, "
                 f"{n_ok} yielded events, {n_zero} yielded none, {n_err} errored)")
        time.sleep(DELAY_SECONDS)

    if not all_rows:
        df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in SCHEMA.items()})
    else:
        df = pd.DataFrame(all_rows)
    # A small residual (~0.1% on the full fetch) of exact-duplicate rows
    # remains even after the list-level dedup above - traced to at least
    # one genuine within-document repeat, not investigated further given
    # the scale. Safe to collapse: an exact duplicate (same symbol, index,
    # action, both dates, AND source_url) is definitionally the same
    # disclosed fact, not two different ones.
    n_before = len(df)
    df = df.drop_duplicates()
    if n_before != len(df):
        print(f"  ({n_before - len(df)} exact-duplicate rows collapsed)")
    df = df.sort_values(["announcement_date", "index_name", "action"]).reset_index(drop=True)
    out_path = out_dir / "events.csv"
    df.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. {len(items)} releases: {n_ok} yielded events, "
         f"{n_zero} yielded none, {n_err} errored.")
    print(f"Wrote {len(df)} events to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
