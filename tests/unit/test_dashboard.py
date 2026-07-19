"""Unit tests for utils/dashboard.py's pure data layer (the SQL/shaping functions that back
the live Streamlit dashboard). The Streamlit/plotly render layer is not imported here — it is
loaded lazily inside dashboard.render(), so this test needs only pandas + sqlite3."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

# utils/ is not a package — load the dashboard tool straight from its file (same pattern as
# test_benchmark.py loads benchmark_report). Registering in sys.modules before exec keeps any
# in-module name resolution well-defined.
_DASHBOARD_PATH = Path(__file__).resolve().parents[2] / "utils" / "dashboard.py"
_spec = importlib.util.spec_from_file_location("dashboard", _DASHBOARD_PATH)
dashboard = importlib.util.module_from_spec(_spec)
sys.modules["dashboard"] = dashboard
_spec.loader.exec_module(dashboard)


def _build_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE system_resources (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, resource_type TEXT,
            resource_name TEXT, value REAL, unit TEXT, tag TEXT);
        CREATE TABLE training_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, model_name TEXT,
            stat_name TEXT, value REAL, round_number INTEGER, epoch_number INTEGER,
            tag TEXT, superseded INTEGER DEFAULT 0);
        CREATE TABLE pipeline_stages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, stage TEXT, start_time REAL, end_time REAL,
            duration_s REAL, tag TEXT, metadata TEXT);
        """
    )
    for ts in (100.0, 160.0, 220.0):
        conn.execute(
            "INSERT INTO system_resources (timestamp,resource_type,resource_name,value,unit,tag)"
            " VALUES (?,?,?,?,?,?)",
            (ts, "cpu", "system_total", 50.0, "percent", "t1"),
        )
        conn.execute(
            "INSERT INTO system_resources (timestamp,resource_type,resource_name,value,unit,tag)"
            " VALUES (?,?,?,?,?,?)",
            (ts, "ram", "system_total", 40.0 + ts / 100, "percent", "t1"),
        )
        conn.execute(
            "INSERT INTO system_resources (timestamp,resource_type,resource_name,value,unit,tag)"
            " VALUES (?,?,?,?,?,?)",
            (ts, "gpu", "GPU0_utilization", 30.0, "percent", "t1"),
        )
    for ep, v in enumerate([1.0, 0.5, 0.2], start=1):
        conn.execute(
            "INSERT INTO training_stats"
            " (timestamp,model_name,stat_name,value,round_number,epoch_number,tag,superseded)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (100.0 + ep, "beta_vae", "total_loss", v, 1, ep, "t1", 0),
        )
    # A superseded row that MUST be excluded from the loss curve
    conn.execute(
        "INSERT INTO training_stats"
        " (timestamp,model_name,stat_name,value,round_number,epoch_number,tag,superseded)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (999.0, "beta_vae", "total_loss", 42.0, 1, 1, "t1", 1),
    )
    for stage, st, et, tag in (
        ("train.round_01", 100.0, 220.0, "t1"),
        ("train.round_01.data_generation", 100.0, 160.0, "t1"),
        ("inference.infer_cadence_001", 300.0, 330.0, "t2"),
    ):
        conn.execute(
            "INSERT INTO pipeline_stages (stage,start_time,end_time,duration_s,tag,metadata)"
            " VALUES (?,?,?,?,?,?)",
            (stage, st, et, et - st, tag, None),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "aetherscan.db")
    _build_db(path)
    return path


@pytest.fixture
def conn(db_path):
    c = dashboard.connect_ro(db_path)
    yield c
    c.close()


def test_connect_ro_is_read_only(db_path):
    conn = dashboard.connect_ro(db_path)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute(
            "INSERT INTO pipeline_stages (stage,start_time,end_time,duration_s) VALUES ('x',0,1,1)"
        )
    conn.close()


def test_list_tags(conn):
    assert dashboard.list_tags(conn) == ["t1", "t2"]


def test_load_resources(conn):
    res = dashboard.load_resources(conn, "t1")
    assert len(res) == 9  # 3 timestamps x 3 series
    assert set(res["resource_type"]) == {"cpu", "ram", "gpu"}


def test_load_training_stats_excludes_superseded(conn):
    tr = dashboard.load_training_stats(conn, "t1")
    assert len(tr) == 3  # the superseded row is dropped
    assert list(tr["value"]) == [1.0, 0.5, 0.2]  # ordered by round/epoch


def test_load_stages_derives_depth_and_orders(conn):
    st = dashboard.load_stages(conn, "t1")
    assert len(st) == 2
    assert set(st["depth"]) == {2, 3}
    assert st.iloc[0]["stage"] == "train.round_01"  # earliest start first


def test_run_summary(conn):
    res = dashboard.load_resources(conn, "t1")
    st = dashboard.load_stages(conn, "t1")
    summ = dashboard.run_summary(res, st)
    assert summ["wall_s"] == 120.0  # 220 - 100
    assert summ["n_stages"] == 2
    assert summ["latest_stage"] == "train.round_01.data_generation"  # latest-started span
    assert summ["peak_ram_pct"] == pytest.approx(42.2)


def test_empty_tag_is_safe(conn):
    empty = dashboard.load_stages(conn, "nope")
    assert empty.empty
    summ = dashboard.run_summary(dashboard.load_resources(conn, "nope"), empty)
    assert summ["n_stages"] == 0
    assert summ["latest_stage"] is None
