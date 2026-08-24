"""dtest.data.options_chain against a small synthetic sqlite fixture -
never the real fno.db in a unit test. Deliberately includes BOTH real
`instrument` code eras found live in the real database (`OPTSTK` pre-
2024-06, `STO` from 2024-06 on) to prove `load_options_chain` is immune to
that column's drift, since it filters on `asset_class`/`contract_type`
instead - see `dtest/data/options_chain.py`'s own module docstring for the
live discovery this guards against.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from dtest.data.options_chain import load_options_chain

SCHEMA = """
CREATE TABLE fno_bhavcopy_full (
    trade_date TEXT, symbol TEXT, instrument TEXT, asset_class TEXT,
    contract_type TEXT, expiry_date TEXT, strike REAL, option_type TEXT,
    open REAL, high REAL, low REAL, close REAL, settle REAL,
    contracts INTEGER, value_lakh REAL, open_interest INTEGER, chg_in_oi INTEGER
)
"""

ROWS = [
    # Old-era code (OPTSTK), pre-2024-06: AAA, two strikes, one expiry.
    ("2024-01-23", "AAA", "OPTSTK", "STOCK", "OPT", "2024-01-25", 100.0, "CE",
     5.0, 6.0, 4.0, 5.5, 5.2, 1000, 1, 200000, 5000),
    ("2024-01-23", "AAA", "OPTSTK", "STOCK", "OPT", "2024-01-25", 100.0, "PE",
     3.0, 4.0, 2.0, 3.5, 3.2, 800, 1, 150000, -2000),
    # New-era code (STO), post-2024-06: same symbol, real 2026 date - the
    # exact shape that would be silently dropped by an `instrument='OPTSTK'` filter.
    ("2026-08-07", "AAA", "STO", "STOCK", "OPT", "2026-08-27", 110.0, "CE",
     8.0, 9.0, 7.0, 8.5, 8.2, 500, 1, 90000, 1000),
    # Must be excluded: a stock FUTURE row, and an INDEX option row.
    ("2024-01-23", "AAA", "STF", "STOCK", "FUT", "2024-01-25", None, None,
     1, 1, 1, 1, 1, 1, 1, 999999, 999999),
    ("2024-01-23", "NIFTY", "OPTIDX", "INDEX", "OPT", "2024-01-25", 20000.0, "CE",
     1, 1, 1, 1, 1, 1, 1, 888888, 888888),
]


@pytest.fixture
def fno_db(tmp_path):
    path = tmp_path / "fno.db"
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    con.executemany(
        "INSERT INTO fno_bhavcopy_full VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ROWS,
    )
    con.commit()
    con.close()
    return path


def test_excludes_futures_and_other_underlyings(fno_db):
    df = load_options_chain(fno_db)
    assert set(df["symbol"]) == {"AAA"}
    assert len(df) == 3   # 2 old-era rows + 1 new-era row, futures/index excluded


def test_immune_to_the_instrument_column_era_drift(fno_db):
    """The whole point of this module: pick up the 2026 row coded `STO`
    exactly as readily as the 2024 rows coded `OPTSTK` - never filter on
    `instrument`."""
    df = load_options_chain(fno_db)
    new_era = df[df["trade_date"] == pd.Timestamp("2026-08-07")]
    assert len(new_era) == 1
    assert new_era.iloc[0]["strike"] == 110.0
    assert new_era.iloc[0]["option_type"] == "CE"


def test_ce_and_pe_both_present_same_strike_different_rows(fno_db):
    df = load_options_chain(fno_db)
    old_era = df[df["trade_date"] == pd.Timestamp("2024-01-23")]
    assert set(old_era["option_type"]) == {"CE", "PE"}
    assert set(old_era["strike"]) == {100.0}


def test_date_bounds_applied(fno_db):
    df = load_options_chain(fno_db, start=pd.Timestamp("2025-01-01"))
    assert len(df) == 1
    assert df.iloc[0]["trade_date"] == pd.Timestamp("2026-08-07")


def test_symbol_filter_pushed_into_sql(fno_db):
    df = load_options_chain(fno_db, symbols=["ZZZ"])
    assert df.empty
    assert list(df.columns) == list(load_options_chain(fno_db).columns)


def test_empty_result_has_correct_schema_and_dtypes(fno_db):
    df = load_options_chain(fno_db, start=pd.Timestamp("2099-01-01"))
    assert df.empty
    assert str(df["strike"].dtype) == "float64"
    assert str(df["trade_date"].dtype) == "datetime64[ns]"
