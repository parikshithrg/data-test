"""dtest.data.fno_price against a small synthetic sqlite fixture - never the
real 46GB fno.db in a unit test. Mirrors test_fno_oi.py's rollover fixture
shape, but the point under test is different: price must NOT be smoothed
across the roll - it should jump to the new contract's own real level.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from dtest.data.fno_price import load_front_month_price, load_stock_futures_contracts

SCHEMA = """
CREATE TABLE fno_bhavcopy_full (
    trade_date TEXT, symbol TEXT, instrument TEXT, asset_class TEXT,
    contract_type TEXT, expiry_date TEXT, strike REAL, option_type TEXT,
    open REAL, high REAL, low REAL, close REAL, settle REAL,
    contracts INTEGER, value_lakh REAL, open_interest INTEGER, chg_in_oi INTEGER
)
"""

# AAA: Jan contract (expiry 01-25) trades through 01-25 around settle~100-102;
# Feb contract (expiry 02-29) already listed alongside it from 01-24, trading
# at a genuinely different level (~105-108, a real cost-of-carry basis) - the
# point is the switch should NOT be smoothed toward the Jan level.
ROWS = [
    ("2024-01-23", "AAA", "FUTSTK", "STOCK", "FUT", "2024-01-25", None, None,
     100, 101, 99, 100.5, 100.0, 1, 1, 1000, 100),
    ("2024-01-23", "AAA", "FUTSTK", "STOCK", "FUT", "2024-02-29", None, None,
     105, 106, 104, 105.5, 105.0, 1, 1, 2000, 200),
    ("2024-01-24", "AAA", "FUTSTK", "STOCK", "FUT", "2024-01-25", None, None,
     101, 102, 100, 101.5, 101.0, 1, 1, 900, -100),
    ("2024-01-24", "AAA", "FUTSTK", "STOCK", "FUT", "2024-02-29", None, None,
     106, 107, 105, 106.5, 106.0, 1, 1, 2100, 100),
    ("2024-01-25", "AAA", "FUTSTK", "STOCK", "FUT", "2024-01-25", None, None,
     102, 103, 101, 102.5, 102.0, 1, 1, 800, -100),
    ("2024-01-25", "AAA", "FUTSTK", "STOCK", "FUT", "2024-02-29", None, None,
     107, 108, 106, 107.5, 107.0, 1, 1, 2200, 100),
    # Jan contract expired - Feb is now front month, at ITS OWN level (108).
    ("2024-01-29", "AAA", "FUTSTK", "STOCK", "FUT", "2024-02-29", None, None,
     108, 109, 107, 108.5, 108.0, 1, 1, 2300, 100),
    # A thin day where settle is missing (NULL) - must fall back to close.
    ("2024-01-30", "AAA", "FUTSTK", "STOCK", "FUT", "2024-02-29", None, None,
     108, 109, 107, 109.0, None, 1, 1, 2350, 50),
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
    df = load_front_month_price(fno_db)
    assert set(df["symbol"]) == {"AAA"}
    assert len(df) == 5   # one row per AAA trade_date, front month only


def test_front_month_picks_soonest_unexpired_expiry(fno_db):
    df = load_front_month_price(fno_db)
    row = df[df["date"] == pd.Timestamp("2024-01-23")].iloc[0]
    assert row["expiry_date"] == pd.Timestamp("2024-01-25")   # Jan, not Feb
    assert row["price"] == pytest.approx(100.0)                # Jan's own settle
    assert row["open_price"] == pytest.approx(100.0)           # Jan's own open


def test_open_price_also_jumps_to_new_contract_on_rollover(fno_db):
    """Same point as the mark price test, for the FILL price: the open on
    the rollover day must be the Feb contract's own real open (108), not
    the Jan contract's."""
    df = load_front_month_price(fno_db).set_index("date")
    assert df.loc[pd.Timestamp("2024-01-25"), "open_price"] == pytest.approx(102.0)
    assert df.loc[pd.Timestamp("2024-01-29"), "open_price"] == pytest.approx(108.0)


def test_price_jumps_to_new_contracts_own_level_not_smoothed(fno_db):
    """The whole point of this module: price on the rollover day must be
    the FEB contract's own real settle (108.0), not any blend or
    adjustment toward the Jan level (102.0) - a real ~5.9% basis, not an
    artifact to be hidden."""
    df = load_front_month_price(fno_db).set_index("date")
    pre_roll = df.loc[pd.Timestamp("2024-01-25"), "price"]
    post_roll = df.loc[pd.Timestamp("2024-01-29"), "price"]
    assert pre_roll == pytest.approx(102.0)
    assert post_roll == pytest.approx(108.0)
    assert post_roll != pytest.approx(pre_roll)   # real basis, not smoothed away


def test_is_rollover_flags_only_the_contract_switch_day(fno_db):
    df = load_front_month_price(fno_db).set_index("date")
    assert not df.loc[pd.Timestamp("2024-01-23"), "is_rollover"]
    assert not df.loc[pd.Timestamp("2024-01-24"), "is_rollover"]
    assert not df.loc[pd.Timestamp("2024-01-25"), "is_rollover"]
    assert df.loc[pd.Timestamp("2024-01-29"), "is_rollover"]
    assert not df.loc[pd.Timestamp("2024-01-30"), "is_rollover"]


def test_price_falls_back_to_close_when_settle_missing(fno_db):
    df = load_front_month_price(fno_db).set_index("date")
    row = df.loc[pd.Timestamp("2024-01-30")]
    assert row["price"] == pytest.approx(109.0)   # close, since settle is NULL


def test_days_to_expiry_computed_correctly(fno_db):
    df = load_front_month_price(fno_db).set_index("date")
    assert df.loc[pd.Timestamp("2024-01-23"), "days_to_expiry"] == 2
    assert df.loc[pd.Timestamp("2024-01-25"), "days_to_expiry"] == 0
    assert df.loc[pd.Timestamp("2024-01-29"), "days_to_expiry"] == 31


def test_load_front_month_price_deterministic(fno_db):
    df1 = load_front_month_price(fno_db)
    df2 = load_front_month_price(fno_db)
    pd.testing.assert_frame_equal(df1, df2)


def test_stock_futures_contracts_includes_both_live_contracts_not_just_front(fno_db):
    """The whole point of this function: on 2024-01-23/24/25, BOTH the Jan
    and Feb contracts are live and unexpired - `load_front_month_price`
    collapses to Jan only; this must keep both."""
    df = load_stock_futures_contracts(fno_db)
    on_2301 = df[df["date"] == pd.Timestamp("2024-01-23")]
    assert set(on_2301["expiry_date"]) == {pd.Timestamp("2024-01-25"), pd.Timestamp("2024-02-29")}


def test_stock_futures_contracts_ranks_by_soonest_expiry_first(fno_db):
    df = load_stock_futures_contracts(fno_db).set_index(["date", "rank"])
    jan = df.loc[(pd.Timestamp("2024-01-23"), 1)]
    feb = df.loc[(pd.Timestamp("2024-01-23"), 2)]
    assert jan["expiry_date"] == pd.Timestamp("2024-01-25")
    assert jan["price"] == pytest.approx(100.0)
    assert feb["expiry_date"] == pd.Timestamp("2024-02-29")
    assert feb["price"] == pytest.approx(105.0)


def test_stock_futures_contracts_excludes_options_and_other_underlyings(fno_db):
    df = load_stock_futures_contracts(fno_db)
    assert set(df["symbol"]) == {"AAA"}


def test_stock_futures_contracts_days_to_expiry_per_contract(fno_db):
    df = load_stock_futures_contracts(fno_db).set_index(["date", "rank"])
    assert df.loc[(pd.Timestamp("2024-01-23"), 1), "days_to_expiry"] == 2
    assert df.loc[(pd.Timestamp("2024-01-23"), 2), "days_to_expiry"] == 37


def test_stock_futures_contracts_deterministic(fno_db):
    df1 = load_stock_futures_contracts(fno_db)
    df2 = load_stock_futures_contracts(fno_db)
    pd.testing.assert_frame_equal(df1, df2)
