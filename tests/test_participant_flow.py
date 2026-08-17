"""dtest.data.participant_flow against a small synthetic sqlite fixture -
never the real 46GB fno.db in a unit test.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from dtest.data.participant_flow import load_fii_net_index_flow

SCHEMA = """
CREATE TABLE participant_oi_daily (
    trade_date TEXT, participant TEXT,
    future_index_long INTEGER, future_index_short INTEGER,
    future_stock_long INTEGER, future_stock_short INTEGER,
    option_index_call_long INTEGER, option_index_put_long INTEGER,
    option_index_call_short INTEGER, option_index_put_short INTEGER,
    option_stock_call_long INTEGER, option_stock_put_long INTEGER,
    option_stock_call_short INTEGER, option_stock_put_short INTEGER,
    total_long_contracts INTEGER, total_short_contracts INTEGER
)
"""

ROWS = [
    ("2024-01-02", "FII", 100000, 40000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ("2024-01-02", "DII", 20000, 30000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ("2024-01-03", "FII", 90000, 60000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
]


@pytest.fixture
def fno_db(tmp_path):
    path = tmp_path / "fno.db"
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    con.executemany(
        "INSERT INTO participant_oi_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ROWS,
    )
    con.commit()
    con.close()
    return path


def test_load_fii_net_index_flow_only_fii(fno_db):
    net = load_fii_net_index_flow(fno_db)
    assert len(net) == 2   # DII row excluded


def test_load_fii_net_index_flow_is_long_minus_short(fno_db):
    net = load_fii_net_index_flow(fno_db)
    assert net.loc[pd.Timestamp("2024-01-02")] == 60000
    assert net.loc[pd.Timestamp("2024-01-03")] == 30000


def test_load_fii_net_index_flow_date_bounds(fno_db):
    net = load_fii_net_index_flow(fno_db, start=pd.Timestamp("2024-01-03"))
    assert len(net) == 1
    assert net.index[0] == pd.Timestamp("2024-01-03")


def test_load_fii_net_index_flow_raises_on_duplicate_dates(fno_db):
    con = sqlite3.connect(fno_db)
    con.execute(
        "INSERT INTO participant_oi_daily VALUES "
        "('2024-01-02','FII',1,1,0,0,0,0,0,0,0,0,0,0,0,0)"
    )
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="duplicate"):
        load_fii_net_index_flow(fno_db)
