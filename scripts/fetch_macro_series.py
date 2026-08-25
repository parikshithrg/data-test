"""ONE-TIME DATA ACQUISITION, not part of the deterministic backtest
harness - same category as `fetch_gsec_yields.py`. Fetches the 6 real
`api.mospi.gov.in` series `dtest/data/macro_series.py` documents (CPI,
IIP, WPI, GDP, PLFS unemployment, RBI forex reserves), saves each as its
own CSV in `cfg.paths.macro_series_dir`.

    python scripts/fetch_macro_series.py

Run once. Re-run only to refresh with more recent data - every endpoint
here is a live, growing government series (each of today's `raise_for_status`
calls hitting `api.mospi.gov.in` directly, not a cache).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config
from dtest.data.mospi_api import fetch_all

MONTH_TO_NUM = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}


def _month_date(year: object, month: object) -> pd.Timestamp | None:
    m = MONTH_TO_NUM.get(str(month))
    if m is None:
        return None
    return pd.Timestamp(year=int(year), month=m, day=1)


def _num(x: object) -> float | None:
    if x is None or x in ("-", ""):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def fetch_cpi() -> pd.DataFrame:
    rows = fetch_all("/api/cpi/getCPIData", {"base_year": 2024, "division_code": 0})
    df = pd.DataFrame(rows)
    df["date"] = [_month_date(y, m) for y, m in zip(df["year"], df["month"])]
    out = df.rename(columns={"index": "index", "inflation": "inflation"})[
        ["date", "series", "sector", "index", "inflation"]
    ]
    out["index"] = out["index"].map(_num)
    out["inflation"] = out["inflation"].map(_num)
    return out.dropna(subset=["date"])


def fetch_iip() -> pd.DataFrame:
    rows = fetch_all("/api/iip/getIIPData", {"frequency": "Monthly", "type": "General"})
    df = pd.DataFrame(rows)
    df["date"] = [_month_date(y, m) for y, m in zip(df["year"], df["month"])]
    df["index"] = df["index"].map(_num)
    df["growth_rate_pct"] = df["growth_rate"].map(_num)
    return df[["date", "index", "growth_rate_pct"]].dropna(subset=["date"])


def fetch_wpi() -> pd.DataFrame:
    rows = fetch_all("/api/wpi/getWpiRecords", {"major_group_code": "1000000000", "Format": "JSON"})
    df = pd.DataFrame(rows)
    df["date"] = [_month_date(y, m) for y, m in zip(df["year"], df["month"])]
    df["index"] = df["index_value"].map(_num)
    return df[["date", "index"]].dropna(subset=["date"])


def fetch_gdp() -> pd.DataFrame:
    rows = fetch_all("/api/nas/getNASData", {
        "base_year": "2011-12", "series": "Current", "frequency_code": "Quarterly",
        "indicator_code": "5", "Format": "JSON",
    })
    df = pd.DataFrame(rows)
    df["current_price_rs_crore"] = df["current_price"].map(_num)
    df["constant_price_rs_crore"] = df["constant_price"].map(_num)
    return df[["year", "quarter", "current_price_rs_crore", "constant_price_rs_crore"]]


def fetch_unemployment() -> pd.DataFrame:
    rows = fetch_all("/api/plfs/getData", {
        "indicator_code": 3, "frequency_code": 3, "Format": "JSON",
    })
    df = pd.DataFrame(rows)
    df["date"] = [_month_date(y, m) for y, m in zip(df["year"], df["month"])]
    df["unemployment_rate_pct"] = df["value"].map(_num)
    return df.rename(columns={"AgeGroup": "age_group"})[
        ["date", "age_group", "gender", "sector", "unemployment_rate_pct"]
    ].dropna(subset=["date"])


def fetch_forex_reserves() -> pd.DataFrame:
    rows = fetch_all("/api/rbi/getRbiRecords", {"sub_indicator_code": 47})
    df = pd.DataFrame(rows)
    df["date"] = [_month_date(y, m) for y, m in zip(df["year"], df["month"])]
    df["value"] = df["value"].map(_num)
    return df.rename(columns={
        "foreign_exchange_reserve_type": "reserve_type",
        "foreign_exchange_reserve_currency": "currency",
    })[["date", "reserve_type", "currency", "value"]].dropna(subset=["date"])


FETCHERS = {
    "CPI_MONTHLY.csv": fetch_cpi,
    "IIP_MONTHLY.csv": fetch_iip,
    "WPI_MONTHLY.csv": fetch_wpi,
    "GDP_QUARTERLY.csv": fetch_gdp,
    "UNEMPLOYMENT_MONTHLY.csv": fetch_unemployment,
    "FOREX_RESERVES_MONTHLY.csv": fetch_forex_reserves,
}


def main() -> int:
    cfg = load_config()
    out_dir = Path(cfg.paths.macro_series_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, fetcher in FETCHERS.items():
        print(f"fetching {filename} ...")
        df = fetcher()
        out_path = out_dir / filename
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path}: {len(df)} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
