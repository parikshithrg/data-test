"""dtest.data.rbi_rates_credit against small synthetic cached CSVs - never
a live FRED call in a unit test, same pattern as test_gsec_yields.py."""

from __future__ import annotations

import pandas as pd
import pytest

from dtest.data.rbi_rates_credit import load_repo_rate_proxy, load_bank_credit


@pytest.fixture
def rbi_dir(tmp_path):
    pd.DataFrame({
        "observation_date": ["2026-01-01", "2026-02-01"],
        "IRSTCI01INM156N": [5.5, 5.5],
    }).to_csv(tmp_path / "IRSTCI01INM156N.csv", index=False)
    pd.DataFrame({
        "observation_date": ["2025-07-01", "2025-10-01"],
        "CRDQINBPABIS": [196939.058, 211244.677],
    }).to_csv(tmp_path / "CRDQINBPABIS.csv", index=False)
    return tmp_path


def test_repo_rate_proxy_loads(rbi_dir):
    df = load_repo_rate_proxy(rbi_dir)
    assert list(df.columns) == ["date", "call_money_rate_pct"]
    assert len(df) == 2
    assert df.iloc[0]["call_money_rate_pct"] == 5.5


def test_bank_credit_loads(rbi_dir):
    df = load_bank_credit(rbi_dir)
    assert list(df.columns) == ["date", "credit_rs_bn"]
    assert df.iloc[1]["credit_rs_bn"] == 211244.677


def test_repo_rate_proxy_missing_file_returns_empty_with_correct_schema(tmp_path):
    df = load_repo_rate_proxy(tmp_path)
    assert df.empty
    assert list(df.columns) == ["date", "call_money_rate_pct"]


def test_bank_credit_missing_file_returns_empty_with_correct_schema(tmp_path):
    df = load_bank_credit(tmp_path)
    assert df.empty
    assert list(df.columns) == ["date", "credit_rs_bn"]
