"""dtest.data.delivery against a small synthetic sqlite fixture - never the
real 46GB fno.db in a unit test. Mirrors the real cash_delivery_daily
schema exactly so a real schema drift would break this fixture first.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from dtest.data.delivery import load_delivery_long

SCHEMA = """
CREATE TABLE cash_delivery_daily (
    trade_date TEXT, symbol TEXT, series TEXT,
    open REAL, high REAL, low REAL, close REAL, prev_close REAL,
    last_price REAL, avg_price REAL, total_traded_qty INTEGER, turnover REAL,
    no_of_trades INTEGER, delivery_qty INTEGER, delivery_pct REAL
)
"""

ROWS = [
    # date, symbol, series, ..., total_traded_qty, turnover, ..., delivery_qty, delivery_pct
    ("2024-01-02", "AAA", "EQ", 1, 1, 1, 1, 1, 1, 1, 1000, 100.0, 10, 600, 60.0),
    ("2024-01-02", "BBB", "BE", 1, 1, 1, 1, 1, 1, 1, 500, 50.0, 5, 500, 100.0),
    ("2024-01-03", "AAA", "EQ", 1, 1, 1, 1, 1, 1, 1, 1100, 110.0, 11, 550, 50.0),
    ("2024-01-04", "AAA", "EQ", 1, 1, 1, 1, 1, 1, 1, 1200, 120.0, 12, 700, 58.3),
]


@pytest.fixture
def fno_db(tmp_path):
    path = tmp_path / "fno.db"
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    con.executemany(
        "INSERT INTO cash_delivery_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ROWS,
    )
    con.commit()
    con.close()
    return path


def test_load_delivery_long_keeps_only_eq_series(fno_db):
    df = load_delivery_long(fno_db)
    assert set(df["symbol"]) == {"AAA"}
    assert len(df) == 3


def test_load_delivery_long_applies_date_bounds_in_sql(fno_db):
    df = load_delivery_long(fno_db, start=pd.Timestamp("2024-01-03"),
                            end=pd.Timestamp("2024-01-03"))
    assert len(df) == 1
    assert df["date"].iloc[0] == pd.Timestamp("2024-01-03")


def test_load_delivery_long_schema_and_dtypes(fno_db):
    df = load_delivery_long(fno_db)
    assert df["date"].dtype == "datetime64[ns]"
    assert list(df.columns) == [
        "date", "symbol", "total_traded_qty", "delivery_qty", "delivery_pct", "turnover",
    ]
    row = df[df["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert row["delivery_qty"] == 600
    assert row["delivery_pct"] == pytest.approx(60.0)


def test_load_delivery_long_deterministic_row_order(fno_db):
    df1 = load_delivery_long(fno_db)
    df2 = load_delivery_long(fno_db)
    pd.testing.assert_frame_equal(df1, df2)
    assert list(df1["date"]) == sorted(df1["date"])
