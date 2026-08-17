"""dtest.data.fno_oi against a small synthetic sqlite fixture - never the
real 46GB fno.db in a unit test. Two overlapping contracts (Jan and Feb
expiry) for one symbol spanning a rollover, mirroring how NSE actually
lists stock futures (near + next month simultaneously).
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from dtest.data.fno_oi import load_front_month_oi

SCHEMA = """
CREATE TABLE fno_bhavcopy_full (
    trade_date TEXT, symbol TEXT, instrument TEXT, asset_class TEXT,
    contract_type TEXT, expiry_date TEXT, strike REAL, option_type TEXT,
    open REAL, high REAL, low REAL, close REAL, settle REAL,
    contracts INTEGER, value_lakh REAL, open_interest INTEGER, chg_in_oi INTEGER
)
"""

# AAA: Jan contract (expiry 01-25) trades through 01-25, Feb contract
# (expiry 02-29) already listed and trading alongside it from 01-24.
# BBB: an OPTION row on the same dates, and an INDEX future row - both must
# be excluded by the asset_class/contract_type filter.
ROWS = [
    ("2024-01-23", "AAA", "FUTSTK", "STOCK", "FUT", "2024-01-25", None, None,
     1, 1, 1, 1, 1, 1, 1, 21176500, -7881250),
    ("2024-01-23", "AAA", "FUTSTK", "STOCK", "FUT", "2024-02-29", None, None,
     1, 1, 1, 1, 1, 1, 1, 30000000, 500000),
    ("2024-01-24", "AAA", "FUTSTK", "STOCK", "FUT", "2024-01-25", None, None,
     1, 1, 1, 1, 1, 1, 1, 12913750, -8262750),
    ("2024-01-24", "AAA", "FUTSTK", "STOCK", "FUT", "2024-02-29", None, None,
     1, 1, 1, 1, 1, 1, 1, 31000000, 1000000),
    ("2024-01-25", "AAA", "FUTSTK", "STOCK", "FUT", "2024-01-25", None, None,
     1, 1, 1, 1, 1, 1, 1, 4323500, -8590250),
    ("2024-01-25", "AAA", "FUTSTK", "STOCK", "FUT", "2024-02-29", None, None,
     1, 1, 1, 1, 1, 1, 1, 33000000, 2000000),
    # Jan contract has expired (no more rows) - Feb contract is now front month.
    ("2024-01-29", "AAA", "FUTSTK", "STOCK", "FUT", "2024-02-29", None, None,
     1, 1, 1, 1, 1, 1, 1, 34120500, 2183250),
    # Must be excluded: an option on AAA, and an index future.
    ("2024-01-23", "AAA", "OPTSTK", "STOCK", "OPT", "2024-01-25", 100.0, "CE",
     1, 1, 1, 1, 1, 1, 1, 999999, 999999),
    ("2024-01-23", "NIFTY", "FUTIDX", "INDEX", "FUT", "2024-01-25", None, None,
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


def test_front_month_excludes_options_and_other_underlyings(fno_db):
    df = load_front_month_oi(fno_db)
    assert set(df["symbol"]) == {"AAA"}
    assert len(df) == 4   # one row per AAA trade_date, front month only


def test_front_month_picks_soonest_unexpired_expiry(fno_db):
    df = load_front_month_oi(fno_db)
    row = df[df["date"] == pd.Timestamp("2024-01-23")].iloc[0]
    assert row["expiry_date"] == pd.Timestamp("2024-01-25")   # Jan, not Feb
    assert row["open_interest"] == 21176500


def test_front_month_switches_to_next_contract_after_expiry(fno_db):
    df = load_front_month_oi(fno_db)
    row = df[df["date"] == pd.Timestamp("2024-01-29")].iloc[0]
    assert row["expiry_date"] == pd.Timestamp("2024-02-29")
    assert row["open_interest"] == 34120500


def test_is_rollover_flags_only_the_contract_switch_day(fno_db):
    df = load_front_month_oi(fno_db).set_index("date")
    assert not df.loc[pd.Timestamp("2024-01-23"), "is_rollover"]
    assert not df.loc[pd.Timestamp("2024-01-24"), "is_rollover"]
    assert not df.loc[pd.Timestamp("2024-01-25"), "is_rollover"]
    assert df.loc[pd.Timestamp("2024-01-29"), "is_rollover"]


def test_oi_chg_pct_is_contract_native_not_a_stitched_diff(fno_db):
    """The rollover day's oi_chg_pct must come straight from the NEW
    contract's own exchange-reported chg_in_oi (2183250 / prev), never from
    diffing the stitched open_interest LEVEL across the Jan->Feb switch
    (4323500 -> 34120500, which would be a nonsense ~689% jump)."""
    df = load_front_month_oi(fno_db).set_index("date")
    roll_row = df.loc[pd.Timestamp("2024-01-29")]
    expected = 2183250 / (34120500 - 2183250) * 100.0
    assert roll_row["oi_chg_pct"] == pytest.approx(expected)
    assert roll_row["oi_chg_pct"] < 10   # nowhere near a stitched-level artifact


def test_days_to_expiry_computed_correctly(fno_db):
    df = load_front_month_oi(fno_db).set_index("date")
    assert df.loc[pd.Timestamp("2024-01-23"), "days_to_expiry"] == 2
    assert df.loc[pd.Timestamp("2024-01-25"), "days_to_expiry"] == 0
    assert df.loc[pd.Timestamp("2024-01-29"), "days_to_expiry"] == 31


def test_load_front_month_oi_deterministic(fno_db):
    df1 = load_front_month_oi(fno_db)
    df2 = load_front_month_oi(fno_db)
    pd.testing.assert_frame_equal(df1, df2)
