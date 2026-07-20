"""Unit tests for utils/dashboard.py's pure data layer (the SQL/shaping/PCA/artifact helpers that
back the live Streamlit dashboard). The Streamlit/plotly render layer is not imported here — it is
loaded lazily inside dashboard.render(), so this test needs only numpy + pandas + sqlite3."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

# utils/ is not a package — load the dashboard tool straight from its file (same pattern as
# test_benchmark.py loads benchmark_report).
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
        CREATE TABLE injection_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, stat_name TEXT, value REAL,
            round_number INTEGER, signal_type TEXT, injection_stage TEXT, is_finite INTEGER,
            slope_clamped INTEGER, tag TEXT, superseded INTEGER DEFAULT 0);
        CREATE TABLE latent_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, model_name TEXT,
            round_number INTEGER, epoch_number INTEGER, step_number INTEGER,
            signal_type TEXT, latent_vector TEXT, tag TEXT, superseded INTEGER DEFAULT 0);
        CREATE TABLE inference_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, prediction INTEGER,
            confidence REAL, target TEXT, band TEXT, frequency_mhz REAL, tag TEXT,
            superseded INTEGER DEFAULT 0);
        CREATE TABLE inference_cadences (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, tag TEXT, status TEXT,
            duration_s REAL, n_stamps INTEGER, n_candidates INTEGER, superseded INTEGER DEFAULT 0);
        CREATE TABLE pipeline_stages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, stage TEXT, start_time REAL, end_time REAL,
            duration_s REAL, tag TEXT, metadata TEXT);
        """
    )
    for ts in (100.0, 160.0, 220.0):
        for rt, rn, v in (
            ("cpu", "system_total", 50.0),
            ("ram", "system_total", 40.0 + ts / 100),
            ("gpu", "GPU0_utilization", 30.0),
        ):
            conn.execute(
                "INSERT INTO system_resources (timestamp,resource_type,resource_name,value,unit,tag)"
                " VALUES (?,?,?,?,?,?)",
                (ts, rt, rn, v, "percent", "t1"),
            )
    # beta_vae loss (3 epochs) + a superseded row + an rf-model row that must be excluded
    for ep, v in enumerate([1.0, 0.5, 0.2], start=1):
        conn.execute(
            "INSERT INTO training_stats"
            " (timestamp,model_name,stat_name,value,round_number,epoch_number,tag,superseded)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (100.0 + ep, "beta_vae", "total_loss", v, 1, ep, "t1", 0),
        )
    conn.execute(
        "INSERT INTO training_stats"
        " (timestamp,model_name,stat_name,value,round_number,epoch_number,tag,superseded)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (999.0, "beta_vae", "total_loss", 42.0, 1, 1, "t1", 1),
    )
    conn.execute(
        "INSERT INTO training_stats"
        " (timestamp,model_name,stat_name,value,round_number,epoch_number,tag,superseded)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (105.0, "rf", "total_loss", 7.0, 1, 1, "t1", 0),
    )
    # injection_stats
    for i, (sig, fin, clamp) in enumerate([("eti", 1, 0), ("rfi", 0, 1), ("eti", 1, 0)]):
        conn.execute(
            "INSERT INTO injection_stats"
            " (timestamp,stat_name,value,round_number,signal_type,injection_stage,is_finite,"
            "slope_clamped,tag,superseded) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (100.0 + i, "global_mean", float(i), 1, sig, "A", fin, clamp, "t1", 0),
        )
    # latent snapshots: two frames; the LATEST (round1/epoch2/step0) has 2 rows
    conn.execute(
        "INSERT INTO latent_snapshots"
        " (timestamp,model_name,round_number,epoch_number,step_number,signal_type,latent_vector,tag,superseded)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (1, "beta_vae", 1, 1, 0, "true", json.dumps([0.0, 1.0, 2.0]), "t1", 0),
    )
    conn.execute(
        "INSERT INTO latent_snapshots"
        " (timestamp,model_name,round_number,epoch_number,step_number,signal_type,latent_vector,tag,superseded)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (2, "beta_vae", 1, 2, 0, "true", json.dumps([1.0, 0.0, 1.0]), "t1", 0),
    )
    conn.execute(
        "INSERT INTO latent_snapshots"
        " (timestamp,model_name,round_number,epoch_number,step_number,signal_type,latent_vector,tag,superseded)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (2, "beta_vae", 1, 2, 0, "false", json.dumps([3.0, 3.0, 3.0]), "t1", 0),
    )
    # inference_results: 1 candidate (prediction=1) + 1 non-candidate + 1 superseded candidate
    conn.execute(
        "INSERT INTO inference_results (timestamp,prediction,confidence,target,band,frequency_mhz,tag,superseded)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (1, 1, 0.95, "HIP1", "C", 1420.0, "t1", 0),
    )
    conn.execute(
        "INSERT INTO inference_results (timestamp,prediction,confidence,target,band,frequency_mhz,tag,superseded)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (1, 0, 0.10, "HIP1", "C", 1421.0, "t1", 0),
    )
    conn.execute(
        "INSERT INTO inference_results (timestamp,prediction,confidence,target,band,frequency_mhz,tag,superseded)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (1, 1, 0.99, "HIP1", "C", 1422.0, "t1", 1),
    )
    conn.execute(
        "INSERT INTO inference_cadences (timestamp,tag,status,duration_s,n_stamps,n_candidates,superseded)"
        " VALUES (?,?,?,?,?,?,?)",
        (1, "t1", "inferred", 12.5, 100, 1, 0),
    )
    for stage, st, et in (
        ("train.round_01", 100.0, 220.0),
        ("train.round_01.data_generation", 100.0, 160.0),
    ):
        conn.execute(
            "INSERT INTO pipeline_stages (stage,start_time,end_time,duration_s,tag,metadata)"
            " VALUES (?,?,?,?,?,?)",
            (stage, st, et, et - st, "t1", None),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "db" / "aetherscan.db")
    Path(path).parent.mkdir(parents=True)
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
    assert dashboard.list_tags(conn) == ["t1"]


def test_load_resources(conn):
    res = dashboard.load_resources(conn, "t1")
    assert len(res) == 9
    assert set(res["resource_type"]) == {"cpu", "ram", "gpu"}


def test_load_training_stats_filters_model_superseded_and_statset(conn):
    df = dashboard.load_training_stats(conn, "t1", dashboard._LOSS_STATS)
    # 3 live beta_vae total_loss rows: rf-model row + superseded row excluded
    assert len(df) == 3
    assert list(df["value"]) == [1.0, 0.5, 0.2]
    # a stat not in the requested set is not returned
    assert dashboard.load_training_stats(conn, "t1", ["clipping_rate"]).empty


def test_load_injection_stats_excludes_superseded(conn):
    inj = dashboard.load_injection_stats(conn, "t1")
    assert len(inj) == 3
    assert set(inj["signal_type"]) == {"eti", "rfi"}


def test_load_latent_snapshots_latest_picks_last_frame(conn):
    snaps = dashboard.load_latent_snapshots_latest(conn, "t1")
    # latest frame is round1/epoch2/step0 which has 2 rows (true + false), not the epoch1 row
    assert len(snaps) == 2
    assert set(snaps["signal_type"]) == {"true", "false"}


def test_parse_latent_matrix_and_pca(conn):
    snaps = dashboard.load_latent_snapshots_latest(conn, "t1")
    mat, labels = dashboard.parse_latent_matrix(snaps)
    assert mat.shape == (2, 3)
    assert len(labels) == 2
    proj = dashboard.pca_2d(mat)
    assert proj.shape == (2, 2)


def test_parse_latent_matrix_drops_malformed(conn):
    df = dashboard.load_latent_snapshots_latest(conn, "t1").copy()
    df.loc[len(df)] = {"signal_type": "bad", "latent_vector": "not json"}
    mat, labels = dashboard.parse_latent_matrix(df)
    assert mat.shape[0] == 2  # the malformed row is dropped
    assert "bad" not in labels


def test_parse_latent_matrix_drops_nonfinite(conn):
    # json.loads accepts NaN/Infinity; such rows must be dropped so the downstream SVD (pca_2d)
    # can't blow up ("SVD did not converge") on a live run with unstable latents.
    df = dashboard.load_latent_snapshots_latest(conn, "t1").copy()
    df.loc[len(df)] = {
        "signal_type": "naninf",
        "latent_vector": json.dumps([float("nan"), 1.0, 2.0]),
    }
    mat, labels = dashboard.parse_latent_matrix(df)
    assert mat.shape[0] == 2  # only the 2 finite rows survive
    assert "naninf" not in labels
    assert np.isfinite(mat).all()
    # and the projection is computable (no LinAlgError)
    assert dashboard.pca_2d(mat).shape == (2, 2)


def test_pca_2d_empty_and_1d():
    assert dashboard.pca_2d(np.empty((0, 0))).shape == (0, 2)
    proj = dashboard.pca_2d(np.array([[1.0], [2.0], [3.0]]))  # d==1 -> zero-filled 2nd col
    assert proj.shape == (3, 2)
    assert np.allclose(proj[:, 1], 0.0)


def test_load_inference_results_excludes_superseded(conn):
    res = dashboard.load_inference_results(conn, "t1")
    assert len(res) == 2  # the superseded candidate is dropped
    assert (res["prediction"] == 1).sum() == 1


def test_load_inference_cadences(conn):
    cad = dashboard.load_inference_cadences(conn, "t1")
    assert len(cad) == 1
    assert cad.iloc[0]["status"] == "inferred"


def test_load_stages_derives_depth(conn):
    st = dashboard.load_stages(conn, "t1")
    assert set(st["depth"]) == {2, 3}


def test_missing_tables_return_empty(tmp_path):
    # a DB with only system_resources — every other loader must degrade to an empty frame
    path = str(tmp_path / "bare.db")
    c = sqlite3.connect(path)
    c.executescript(
        "CREATE TABLE system_resources (timestamp REAL, resource_type TEXT,"
        " resource_name TEXT, value REAL, unit TEXT, tag TEXT);"
    )
    c.close()
    conn = dashboard.connect_ro(path)
    assert dashboard.load_stages(conn, "t1").empty
    assert dashboard.load_inference_results(conn, "t1").empty
    assert dashboard.load_inference_cadences(conn, "t1").empty
    assert dashboard.load_latent_snapshots_latest(conn, "t1").empty
    conn.close()


def test_run_summary(conn):
    res = dashboard.load_resources(conn, "t1")
    st = dashboard.load_stages(conn, "t1")
    summ = dashboard.run_summary(res, st)
    assert summ["wall_s"] == 120.0
    assert summ["latest_stage"] == "train.round_01.data_generation"
    assert summ["peak_ram_pct"] == pytest.approx(42.2)


def test_run_summary_no_stages_uses_resources(conn):
    # exercise the `elif not resources.empty` wall-clock branch: no stages, so the span comes from
    # the resource timestamps (220 - 100), and latest_stage is None
    res = dashboard.load_resources(conn, "t1")
    empty_stages = dashboard.load_stages(conn, "nope")
    summ = dashboard.run_summary(res, empty_stages)
    assert summ["wall_s"] == 120.0
    assert summ["n_stages"] == 0
    assert summ["latest_stage"] is None


def test_list_png_artifacts_and_default_dir(tmp_path):
    plots = tmp_path / "plots"
    (plots / "inference" / "t1").mkdir(parents=True)
    (plots / "beta_vae_loss_curves_t1.png").write_bytes(b"x")
    (plots / "inference" / "t1" / "stamp_gallery_t1.png").write_bytes(b"y")
    (plots / "notes.txt").write_text("ignore me")
    arts = dashboard.list_png_artifacts(str(plots))
    names = {a["name"] for a in arts}
    assert names == {"beta_vae_loss_curves_t1.png", "stamp_gallery_t1.png"}  # .txt ignored
    assert dashboard.list_png_artifacts(str(tmp_path / "nonexistent")) == []
    # default_plots_dir derives {output}/plots from {output}/db/aetherscan.db
    assert dashboard.default_plots_dir(str(tmp_path / "db" / "aetherscan.db")) == str(plots)
