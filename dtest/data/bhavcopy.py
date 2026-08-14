"""NSE cash bhavcopy ingestion - the survivorship fix.

The predecessor project backtested 2004-2026 against 294 CSV files that are
TODAY's Nifty 500 members. Every company that delisted, merged or fell out of
the index in that period is simply absent, so the backtest could only ever pick
from names already known to have survived. Measured on the one window where
comparison is possible (2019+, `cash_delivery_daily`, 4,619 symbols): **22.5% of
the EQ symbols trading before 2020 are gone by 2026** - HDFC, MINDTREE, LTI,
CADILAHC, SRTRANSFIN, PVR and 365 others. A survivor-only dataset shows 0% there.

NSE publishes the full daily cash bhavcopy - every traded symbol, no hindsight -
and it is reachable back to at least 1995. This module ingests it.

TWO FORMATS, and the changeover was measured rather than looked up:
  legacy `cm<DD><MON><YYYY>bhav.csv.zip`   works through 2024-06-03, 404 after
  UDiFF  `BhavCopy_NSE_CM_0_0_0_<YYYYMMDD>_F_0000.csv.zip`  works both sides
Primary format is chosen by date; the other is tried as a fallback, so a genuine
market holiday (both 404) is distinguishable from a format boundary.

WHY PREV_CLOSE MATTERS MORE THAN IT LOOKS. Both formats carry the exchange's own
previous close, which NSE adjusts for splits, bonuses and other corporate
actions. So `prev_close[t] != close[t-1]` is an EXCHANGE-PUBLISHED corporate
action marker with an exact ratio - not a heuristic guess from a suspicious
price jump. Returns computed as `close[t]/prev_close[t] - 1` are corporate-action
correct by construction. The predecessor project had no such marker and its
back-adjusted CSVs had been crushed to the point where BAJFINANCE printed
`1.00 -> 0.50` on six separate dates from pure 2-decimal rounding.

FETCH AND PARSE ARE SEPARATE ON PURPOSE. Raw bytes land on disk exactly as
downloaded and are never touched again; parsing reads from that cache. A parser
change costs a re-parse, never a re-download, and the raw bytes remain the
evidence for what the exchange actually published.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import logging
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

# The date from which the modern format is primary. Measured: legacy 200s at
# 2024-06-03 and 404s at 2024-07-08; UDiFF 200s on both.
UDIFF_FROM = dt.date(2024, 7, 1)

# Polite crawl. The archive is public and static, but this is a few thousand
# sequential requests to someone else's server - so it is deliberately slow,
# jittered, and resumable rather than parallel.
SLEEP_RANGE = (0.9, 1.7)
MAX_RETRIES = 3
RETRY_BACKOFF = 5.0

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# Canonical schema both formats are normalised into.
COLUMNS = ["date", "symbol", "series", "open", "high", "low", "close",
           "last", "prev_close", "volume", "turnover", "trades", "isin"]

STATUS_OK = "ok"
STATUS_HOLIDAY = "holiday"      # both formats 404 - not a trading day
STATUS_ERROR = "error"          # network/parse failure, retried on the next run


def legacy_url(d: dt.date) -> str:
    mon = f"{d:%b}".upper()
    return ("https://nsearchives.nseindia.com/content/historical/EQUITIES/"
            f"{d:%Y}/{mon}/cm{d:%d}{mon}{d:%Y}bhav.csv.zip")


def udiff_url(d: dt.date) -> str:
    return ("https://nsearchives.nseindia.com/content/cm/"
            f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")


def make_session() -> requests.Session:
    """Browser-shaped session with a cookie warm-up.

    The warm-up against www.nseindia.com returns 403 but still sets the cookie
    the archive host wants; measured working. Do not "fix" the 403.
    """
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=20)
    except requests.RequestException as exc:
        log.warning("cookie warm-up failed (continuing): %s", exc)
    return s


@dataclass
class FetchResult:
    day: dt.date
    status: str
    kind: str | None = None       # "legacy" | "udiff"
    nbytes: int = 0
    sha256: str = ""
    error: str = ""


def _raw_path(cache_root: Path, d: dt.date) -> Path:
    # Partitioned by year so a directory never holds 8,000 files.
    return cache_root / "raw" / f"{d:%Y}" / f"{d:%Y%m%d}.zip"


def fetch_day(
    sess: requests.Session,
    day: dt.date,
    cache_root: Path,
    *,
    force: bool = False,
) -> FetchResult:
    """Download one day's bhavcopy into the raw cache. Resumable and idempotent.

    Returns a `FetchResult`; an already-cached day is returned without any
    network call, which is what makes the whole ingest restartable at no cost.
    """
    path = _raw_path(cache_root, day)
    if path.exists() and not force:
        raw = path.read_bytes()
        kind = "udiff" if day >= UDIFF_FROM else "legacy"
        return FetchResult(day, STATUS_OK, kind, len(raw),
                           hashlib.sha256(raw).hexdigest())

    order = ([("udiff", udiff_url), ("legacy", legacy_url)]
             if day >= UDIFF_FROM else
             [("legacy", legacy_url), ("udiff", udiff_url)])

    last_error = ""
    for kind, builder in order:
        url = builder(day)
        for attempt in range(MAX_RETRIES):
            try:
                r = sess.get(url, timeout=45)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue

            if r.status_code == 404:
                break                      # wrong format or a holiday; try next
            if r.status_code != 200:
                last_error = f"HTTP {r.status_code}"
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            if r.content[:2] != b"PK":
                # NSE serves an HTML error page with a 200 when unhappy.
                last_error = "not a zip (HTML error page?)"
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            return FetchResult(day, STATUS_OK, kind, len(r.content),
                               hashlib.sha256(r.content).hexdigest())

    if last_error:
        return FetchResult(day, STATUS_ERROR, error=last_error)
    return FetchResult(day, STATUS_HOLIDAY)


def _unzip(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        return z.read(z.namelist()[0]).decode("utf-8", "replace")


def parse_legacy(text: str, day: dt.date) -> pd.DataFrame:
    """SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP"""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    out = pd.DataFrame({
        "date": pd.Timestamp(day),
        "symbol": df["SYMBOL"].astype(str).str.strip(),
        "series": df["SERIES"].astype(str).str.strip(),
        "open": pd.to_numeric(df["OPEN"], errors="coerce"),
        "high": pd.to_numeric(df["HIGH"], errors="coerce"),
        "low": pd.to_numeric(df["LOW"], errors="coerce"),
        "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
        "last": pd.to_numeric(df["LAST"], errors="coerce"),
        "prev_close": pd.to_numeric(df["PREVCLOSE"], errors="coerce"),
        "volume": pd.to_numeric(df["TOTTRDQTY"], errors="coerce"),
        "turnover": pd.to_numeric(df["TOTTRDVAL"], errors="coerce"),
        # The legacy format carries neither trade count nor ISIN. NaN (not pd.NA)
        # because these are typed float64/string downstream and pd.NA will not
        # cast into a float column.
        "trades": float("nan"),
        "isin": pd.NA,
    })
    return out[COLUMNS].reset_index(drop=True)


def parse_udiff(text: str, day: dt.date) -> pd.DataFrame:
    """Modern UDiFF. Carries every segment, so cash equities must be filtered out.

    `Sgmt == 'CM'` selects the cash market and `FinInstrmTp == 'STK'` selects
    equities specifically - the same file also carries government bonds, ETFs
    and sovereign gold bonds, which are not what this project trades.
    """
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    df = df[(df["Sgmt"].astype(str).str.strip() == "CM")
            & (df["FinInstrmTp"].astype(str).str.strip() == "STK")]
    out = pd.DataFrame({
        "date": pd.Timestamp(day),
        "symbol": df["TckrSymb"].astype(str).str.strip(),
        "series": df["SctySrs"].astype(str).str.strip(),
        "open": pd.to_numeric(df["OpnPric"], errors="coerce"),
        "high": pd.to_numeric(df["HghPric"], errors="coerce"),
        "low": pd.to_numeric(df["LwPric"], errors="coerce"),
        "close": pd.to_numeric(df["ClsPric"], errors="coerce"),
        "last": pd.to_numeric(df["LastPric"], errors="coerce"),
        "prev_close": pd.to_numeric(df["PrvsClsgPric"], errors="coerce"),
        "volume": pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
        "turnover": pd.to_numeric(df["TtlTrfVal"], errors="coerce"),
        "trades": pd.to_numeric(df["TtlNbOfTxsExctd"], errors="coerce"),
        "isin": df["ISIN"].astype(str).str.strip(),
    })
    return out[COLUMNS].reset_index(drop=True)


def parse_day(cache_root: Path, day: dt.date) -> pd.DataFrame | None:
    """Parse one cached day. Returns None if that day is not in the cache.

    Format is detected from CONTENT, not from the date: the file on disk is
    whichever format actually answered, and trusting the date would silently
    mis-parse any day served by the fallback.
    """
    path = _raw_path(cache_root, day)
    if not path.exists():
        return None
    text = _unzip(path.read_bytes())
    header = text.splitlines()[0]
    if "TckrSymb" in header:
        return parse_udiff(text, day)
    return parse_legacy(text, day)


def load_fetch_log(cache_root: Path) -> pd.DataFrame:
    path = cache_root / "fetch_log.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "status", "kind", "nbytes", "sha256", "error"])
    return pd.read_csv(path, parse_dates=["date"])


def save_fetch_log(cache_root: Path, df: pd.DataFrame) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    df.sort_values("date", kind="stable").to_csv(cache_root / "fetch_log.csv", index=False)


def ingest(
    cache_root: Path,
    start: dt.date,
    end: dt.date,
    *,
    sleep_range: tuple[float, float] = SLEEP_RANGE,
    progress_every: int = 50,
) -> pd.DataFrame:
    """Fetch every weekday in [start, end] into the raw cache. Resumable.

    Weekends are skipped without a request. Days already cached, and days already
    logged as holidays, are skipped too - so re-running after an interruption
    costs only the days that are genuinely outstanding.
    """
    cache_root = Path(cache_root)
    log_df = load_fetch_log(cache_root)
    known = {}
    if not log_df.empty:
        known = dict(zip(log_df["date"].dt.date, log_df["status"]))

    sess = make_session()
    results: list[FetchResult] = []
    day = start
    n_done = n_new = n_hol = n_err = 0
    t0 = time.time()

    while day <= end:
        if day.weekday() >= 5:                     # Sat/Sun: never a session
            day += dt.timedelta(days=1)
            continue
        prior = known.get(day)
        cached = _raw_path(cache_root, day).exists()
        if cached or prior == STATUS_HOLIDAY:
            n_done += 1
            day += dt.timedelta(days=1)
            continue

        res = fetch_day(sess, day, cache_root)
        results.append(res)
        if res.status == STATUS_OK:
            n_new += 1
        elif res.status == STATUS_HOLIDAY:
            n_hol += 1
        else:
            n_err += 1
            log.warning("%s: %s", day, res.error)

        if (n_new + n_hol + n_err) % progress_every == 0:
            _flush(cache_root, log_df, results)
            rate = (n_new + n_hol + n_err) / max(time.time() - t0, 1e-9)
            log.info("%s | new=%d holiday=%d err=%d cached=%d | %.2f req/s",
                     day, n_new, n_hol, n_err, n_done, rate)

        time.sleep(random.uniform(*sleep_range))
        day += dt.timedelta(days=1)

    out = _flush(cache_root, log_df, results)
    log.info("ingest done: %d new, %d holidays, %d errors, %d already cached",
             n_new, n_hol, n_err, n_done)
    return out


def _flush(cache_root: Path, log_df: pd.DataFrame,
           results: list[FetchResult]) -> pd.DataFrame:
    if not results:
        return log_df
    new = pd.DataFrame([{
        "date": pd.Timestamp(r.day), "status": r.status, "kind": r.kind,
        "nbytes": r.nbytes, "sha256": r.sha256, "error": r.error,
    } for r in results])
    combined = (
        pd.concat([log_df, new], ignore_index=True)
        .drop_duplicates(subset="date", keep="last")
    )
    # Concatenating onto an EMPTY frame yields an object-dtype date column, so
    # the first run of a fresh cache would lose `.dt` on the returned log while
    # every later run kept it. Normalise unconditionally.
    combined["date"] = pd.to_datetime(combined["date"])
    save_fetch_log(cache_root, combined)
    return combined
