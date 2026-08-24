"""dtest.data.gsec_yields against small synthetic cached CSVs - never a
live FRED call in a unit test."""

from __future__ import annotations

import pandas as pd
import pytest

from dtest.data.gsec_yields import load_gsec_yields


@pytest.fixture
def gsec_dir(tmp_path):
    pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01"], "yield_pct": [7.1, 7.2],
    }).to_csv(tmp_path / "10Y_MONTHLY.csv", index=False)
    pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01"], "yield_pct": [6.5, 6.6],
    }).to_csv(tmp_path / "3M_MONTHLY.csv", index=False)
    return tmp_path


def test_loads_both_tenors_long_format(gsec_dir):
    df = load_gsec_yields(gsec_dir)
    assert set(df["tenor"]) == {"10Y", "3M"}
    assert len(df) == 4


def test_values_match_source(gsec_dir):
    df = load_gsec_yields(gsec_dir)
    row = df[(df["tenor"] == "10Y") & (df["date"] == pd.Timestamp("2024-01-01"))]
    assert row.iloc[0]["yield_pct"] == 7.1


def test_missing_tenor_file_is_skipped_not_an_error(tmp_path):
    pd.DataFrame({"date": ["2024-01-01"], "yield_pct": [7.1]}).to_csv(
        tmp_path / "10Y_MONTHLY.csv", index=False)
    df = load_gsec_yields(tmp_path)   # 3M_MONTHLY.csv doesn't exist
    assert set(df["tenor"]) == {"10Y"}


def test_no_files_returns_empty_with_correct_schema(tmp_path):
    df = load_gsec_yields(tmp_path)
    assert df.empty
    assert list(df.columns) == ["date", "tenor", "yield_pct"]
