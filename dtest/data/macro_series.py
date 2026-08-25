"""Six national macro series this project sourced itself from
`api.mospi.gov.in` (see `mospi_api.py`'s own docstring for the client and
why it's real, public, and needs a legacy-SSL workaround). Dataset priority
queue Tier 6 items 13-14: RBI forex reserves, and MoSPI's IIP/CPI/WPI/GDP/
unemployment. Deliberately headline-only, not the full disaggregated
hierarchy each source actually supports (CPI down to item level, WPI down
to commodity level, PLFS down to state/religion/education/etc.) - these are
macro REGIME context per the priority queue's own Tier 6 framing ("not
expected to be a per-stock edge alone"), not a per-stock signal, so the
national aggregate is what this project needs. Each `load_*` function
reads the CSV `scripts/fetch_macro_series.py` writes to `macro_series_dir`;
none of them call the network.

NOT COVERED HERE, real reasons stated rather than silently skipped: RBI
repo rate / M3 / bank-credit-growth (Tier 6 item 13's other three asks) -
`api.mospi.gov.in`'s own `rbi` product only carries External Sector
Statistics (BoP, trade, forex, external debt, NRI deposits - confirmed live
by sweeping `sub_indicator_code` 1-110, real data only at 1-48, all of it
external-sector); Money & Banking tables (repo rate, M3, credit) live in a
different part of RBI's own Handbook of Statistics, whose real download
files (`rbidocs.rbi.org.in`) are behind a bot-detection JS challenge
(F5/TSPD cookie, confirmed live: the raw response is an anti-bot challenge
page, not the XLSX) - not reverse-engineered here, same class of
disproportionate-effort call as the AMC-portfolio dead end. FRED (this
project's own fallback for `gsec_yields.py`) was actively refusing
connections all session (`ConnectionResetError`), not merely rate-limited -
untried candidates for a future session: `IRSTCB01INM156N` (central bank
rate, repo proxy), `MABMM301INM189N` (M3), `CRDQINBPABIS` (bank credit).
GST collections (Tier 6 item 15) - GSTN/CBIC's own domain, not MoSPI's;
not investigated this session.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CPI_SCHEMA = {
    "date": "datetime64[ns]", "series": "string", "sector": "string",
    "index": "float64", "inflation": "float64",
}
IIP_SCHEMA = {"date": "datetime64[ns]", "index": "float64", "growth_rate_pct": "float64"}
WPI_SCHEMA = {"date": "datetime64[ns]", "index": "float64"}
GDP_SCHEMA = {
    "year": "string", "quarter": "string",
    "current_price_rs_crore": "float64", "constant_price_rs_crore": "float64",
}
UNEMPLOYMENT_SCHEMA = {
    "date": "datetime64[ns]", "age_group": "string", "gender": "string",
    "sector": "string", "unemployment_rate_pct": "float64",
}
FOREX_RESERVES_SCHEMA = {
    "date": "datetime64[ns]", "reserve_type": "string", "currency": "string",
    "value": "float64",
}


def _empty(schema: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in schema.items()})


def _read(path: Path, schema: dict[str, str], parse_dates: list[str]) -> pd.DataFrame:
    if not path.exists():
        return _empty(schema)
    df = pd.read_csv(path, parse_dates=parse_dates)
    for col, dtype in schema.items():
        if dtype != "datetime64[ns]":
            df[col] = df[col].astype(dtype)
    return df.sort_values(parse_dates or list(schema)[:1], kind="stable").reset_index(drop=True)


def load_cpi(macro_series_dir: Path) -> pd.DataFrame:
    """All-India headline CPI (General), base year 2024, one row per
    (date, series[Current/Back], sector[Rural/Urban/Combined])."""
    return _read(Path(macro_series_dir) / "CPI_MONTHLY.csv", CPI_SCHEMA, ["date"])


def load_iip(macro_series_dir: Path) -> pd.DataFrame:
    """All-India headline Index of Industrial Production (General/General),
    base year 2022-23."""
    return _read(Path(macro_series_dir) / "IIP_MONTHLY.csv", IIP_SCHEMA, ["date"])


def load_wpi(macro_series_dir: Path) -> pd.DataFrame:
    """All-India headline Wholesale Price Index, base year 2022-23. No
    inflation column on this source - a YoY transform is a features-layer
    concern, not this loader's, matching `gsec_yields.py`'s own
    data-vs-features split."""
    return _read(Path(macro_series_dir) / "WPI_MONTHLY.csv", WPI_SCHEMA, ["date"])


def load_gdp(macro_series_dir: Path) -> pd.DataFrame:
    """All-India quarterly GDP (current and constant price), base year
    2011-12, series=Current - the deeper of the two base-year series
    `api.mospi.gov.in` offers (58 quarters, 2011-12..2025-26) vs the newer
    2022-23-base series' 16 quarters; only one was fetched, a real,
    stated scope choice. No `date` column - a fiscal-year string
    (e.g. "2011-12") plus a quarter label is the real granularity."""
    return _read(Path(macro_series_dir) / "GDP_QUARTERLY.csv", GDP_SCHEMA, [])


def load_unemployment(macro_series_dir: Path) -> pd.DataFrame:
    """All-India monthly PLFS Unemployment Rate (Current Weekly Status),
    by age group / gender / sector - the real granularity this source
    publishes at, not aggregated down to one national number."""
    return _read(Path(macro_series_dir) / "UNEMPLOYMENT_MONTHLY.csv", UNEMPLOYMENT_SCHEMA, ["date"])


def load_forex_reserves(macro_series_dir: Path) -> pd.DataFrame:
    """RBI Foreign Exchange Reserves, monthly, all reserve-type components
    (Total, Foreign Currency Assets, Gold, SDRs, Reserve Tranche Position)
    in both US$ and Rs crore. `reserve_type == "Total"` is the headline
    number the eSankhyiki homepage itself highlights."""
    return _read(Path(macro_series_dir) / "FOREX_RESERVES_MONTHLY.csv", FOREX_RESERVES_SCHEMA, ["date"])
