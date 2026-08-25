"""dtest.data.macro_series against small synthetic cached CSVs - never a
live api.mospi.gov.in call in a unit test."""

from __future__ import annotations

import pandas as pd
import pytest

from dtest.data.macro_series import (
    load_cpi, load_iip, load_wpi, load_gdp, load_unemployment, load_forex_reserves,
)


@pytest.fixture
def macro_dir(tmp_path):
    pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01"], "series": ["Current", "Current"],
        "sector": ["Combined", "Combined"], "index": [100.0, 100.5], "inflation": [4.1, 4.3],
    }).to_csv(tmp_path / "CPI_MONTHLY.csv", index=False)
    pd.DataFrame({
        "date": ["2024-01-01"], "index": [123.1], "growth_rate_pct": [7.3],
    }).to_csv(tmp_path / "IIP_MONTHLY.csv", index=False)
    pd.DataFrame({
        "date": ["2024-01-01"], "index": [155.7],
    }).to_csv(tmp_path / "WPI_MONTHLY.csv", index=False)
    pd.DataFrame({
        "year": ["2023-24"], "quarter": ["Q1"],
        "current_price_rs_crore": [6704409.0], "constant_price_rs_crore": [6565078.0],
    }).to_csv(tmp_path / "GDP_QUARTERLY.csv", index=False)
    pd.DataFrame({
        "date": ["2025-12-01"], "age_group": ["15 years and above"], "gender": ["male"],
        "sector": ["rural"], "unemployment_rate_pct": [4.1],
    }).to_csv(tmp_path / "UNEMPLOYMENT_MONTHLY.csv", index=False)
    pd.DataFrame({
        "date": ["2025-06-01"], "reserve_type": ["Total"], "currency": ["₹ Crores"],
        "value": [5986617.92],
    }).to_csv(tmp_path / "FOREX_RESERVES_MONTHLY.csv", index=False)
    return tmp_path


def test_cpi_loads_with_correct_schema(macro_dir):
    df = load_cpi(macro_dir)
    assert list(df.columns) == ["date", "series", "sector", "index", "inflation"]
    assert len(df) == 2
    assert df.iloc[0]["index"] == 100.0


def test_iip_loads(macro_dir):
    df = load_iip(macro_dir)
    assert df.iloc[0]["growth_rate_pct"] == 7.3


def test_wpi_loads_no_inflation_column(macro_dir):
    df = load_wpi(macro_dir)
    assert list(df.columns) == ["date", "index"]


def test_gdp_loads_no_date_column(macro_dir):
    df = load_gdp(macro_dir)
    assert "date" not in df.columns
    assert df.iloc[0]["quarter"] == "Q1"


def test_unemployment_loads_with_demographic_dims(macro_dir):
    df = load_unemployment(macro_dir)
    assert df.iloc[0]["unemployment_rate_pct"] == 4.1
    assert df.iloc[0]["sector"] == "rural"


def test_forex_reserves_loads(macro_dir):
    df = load_forex_reserves(macro_dir)
    assert df.iloc[0]["reserve_type"] == "Total"


@pytest.mark.parametrize("loader,filename,schema_cols", [
    (load_cpi, "CPI_MONTHLY.csv", ["date", "series", "sector", "index", "inflation"]),
    (load_iip, "IIP_MONTHLY.csv", ["date", "index", "growth_rate_pct"]),
    (load_wpi, "WPI_MONTHLY.csv", ["date", "index"]),
    (load_unemployment, "UNEMPLOYMENT_MONTHLY.csv",
     ["date", "age_group", "gender", "sector", "unemployment_rate_pct"]),
    (load_forex_reserves, "FOREX_RESERVES_MONTHLY.csv",
     ["date", "reserve_type", "currency", "value"]),
])
def test_missing_file_returns_empty_with_correct_schema(tmp_path, loader, filename, schema_cols):
    df = loader(tmp_path)   # none of the CSVs exist in this empty tmp_path
    assert df.empty
    assert list(df.columns) == schema_cols


def test_gdp_missing_file_returns_empty_with_correct_schema(tmp_path):
    df = load_gdp(tmp_path)
    assert df.empty
    assert list(df.columns) == ["year", "quarter", "current_price_rs_crore", "constant_price_rs_crore"]
