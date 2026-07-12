# NOTE: come back to this later

"""Unit tests for aetherscan.db: writer thread lifecycle, flush sentinel protocol, executemany
batching across tables, is_finite sanitization, and query filters / column whitelists — all
against a tmp-path SQLite file."""

from __future__ import annotations

import json
import os
import sqlite3
import time

import numpy as np
import pytest

from aetherscan.config import get_config
from aetherscan.db.db import _SCHEMA_VERSION, Database


@pytest.fixture
def db(tmp_path):
    """A started Database against the tmp output path, with the periodic flush effectively
    disabled (long write_interval / huge buffer) so only the flush sentinel can drain writes —
    making flush() the thing under test rather than a timing accident."""
    config = get_config()
    config.db.write_interval = 300.0
    config.db.write_buffer_max_size = 100_000
    database = Database()
    assert database.db_path.startswith(str(tmp_path))
    database.start()
    yield database
    database.stop()


class TestWriterThreadLifecycle:
    def test_start_spawns_writer_thread(self, db):
        assert db.writer_thread is not None
        assert db.writer_thread.is_alive()

    def test_start_is_idempotent(self, db):
        thread = db.writer_thread
        db.start()
        assert db.writer_thread is thread  # no second thread spawned

    def test_stop_terminates_thread(self, db):
        db.stop()
        assert not db.writer_thread.is_alive()

    def test_restart_after_stop(self, db):
        db.stop()
        db.start()
        assert db.writer_thread.is_alive()
        db.write_training_stat("beta_vae", "total_loss", 1.0, tag="test_v1")
        assert db.flush(timeout=10) is True
        assert len(db.query_training_stat(tag="test_v1")) == 1

    def test_singleton_semantics(self, db):
        assert Database() is db


class TestFlushSentinel:
    def test_flush_drains_queued_writes(self, db):
        for i in range(5):
            db.write_training_stat(
                "beta_vae", "total_loss", float(i), round_number=1, epoch_number=i, tag="test_v1"
            )
        # write_interval is 300 s, buffer cap 100k: only the sentinel can have flushed these.
        assert db.flush(timeout=10) is True
        rows = db.query_training_stat(tag="test_v1")
        assert len(rows) == 5
        assert sorted(r["value"] for r in rows) == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_flush_without_writer_is_noop_success(self):
        database = Database()  # never started
        assert database.flush(timeout=1) is True

    def test_flush_during_shutdown_returns_false(self, db):
        # Deterministic shutdown-in-progress simulation: setting stop_event on the *live*
        # writer races (the thread may exit before flush() checks it, flipping the result
        # to True via the not-running branch). Stop the real writer first, then stand in an
        # always-alive stub so flush() deterministically hits the stop_event check.
        class _AliveStub:
            @staticmethod
            def is_alive():
                return True

        db.stop()
        db.writer_thread = _AliveStub()
        db.stop_event.set()  # stop() already set it; explicit for the reader
        try:
            assert db.flush(timeout=1) is False
        finally:
            db.writer_thread = None  # let the fixture's stop() no-op cleanly


class TestExecutemanyBatching:
    def test_mixed_table_buffer_lands_in_all_tables(self, db):
        """One flush must route grouped executemany() inserts to every destination table."""
        tag = "test_v1"
        db.write_system_resource("cpu", "system_total", 42.0, unit="percent", tag=tag)
        db.write_training_stat("beta_vae", "total_loss", 0.5, 1, 1, tag=tag)
        db.write_injection_stat("eti_snr", 12.5, round_number=1, signal_class="true", tag=tag)
        db.write_latent_snapshot("beta_vae", 1, 1, 10, 0, "true_only_eti", [[0.1] * 4] * 6, tag=tag)
        db.write_inference_result("/tmp/x.npy", 0, 1, 0.99, latent_vector=np.arange(4.0), tag=tag)
        assert db.flush(timeout=10) is True

        assert len(db.query_system_resource(tag=tag)) == 1
        assert len(db.query_training_stat(tag=tag)) == 1
        assert len(db.query_injection_stat(tag=tag)) == 1
        snapshots = db.query_latent_snapshots(tag=tag)
        assert len(snapshots) == 1
        assert json.loads(snapshots[0]["latent_vector"]) == [[0.1] * 4] * 6
        results = db.query_inference_result(tag=tag)
        assert len(results) == 1
        assert json.loads(results[0]["latent_vector"]) == [0.0, 1.0, 2.0, 3.0]

    def test_multiple_rows_per_table_batch(self, db):
        for i in range(20):
            db.write_training_stat("beta_vae", "kl_loss", float(i), 1, i, tag="test_v2")
        assert db.flush(timeout=10) is True
        assert len(db.query_training_stat(tag="test_v2", stat_name="kl_loss")) == 20


class TestIsFiniteSanitization:
    def test_non_finite_values_stored_as_zero_with_flag(self, db):
        tag = "test_v3"
        db.write_injection_stat("global_skew", float("nan"), tag=tag)
        db.write_injection_stat("global_skew", float("inf"), tag=tag)
        db.write_injection_stat("global_skew", float("-inf"), tag=tag)
        db.write_injection_stat("global_skew", 1.25, tag=tag)
        assert db.flush(timeout=10) is True

        finite_only = db.query_injection_stat(tag=tag)  # only_finite defaults to True
        assert [r["value"] for r in finite_only] == [1.25]

        everything = db.query_injection_stat(tag=tag, only_finite=False)
        assert len(everything) == 4
        sanitized = [r for r in everything if r["is_finite"] == 0]
        assert len(sanitized) == 3
        assert all(r["value"] == 0.0 for r in sanitized)

    def test_slope_clamped_flag_round_trip(self, db):
        tag = "test_v4"
        db.write_injection_stat("rfi_drift_rate", 1.0, slope_clamped=True, tag=tag)
        db.write_injection_stat("rfi_drift_rate", 2.0, slope_clamped=False, tag=tag)
        db.write_injection_stat("rfi_drift_rate", 3.0, slope_clamped=None, tag=tag)
        assert db.flush(timeout=10) is True

        clamped = db.query_injection_stat(tag=tag, only_slope_clamped=True)
        assert [r["value"] for r in clamped] == [1.0]
        unclamped = db.query_injection_stat(tag=tag, only_slope_clamped=False)
        assert sorted(r["value"] for r in unclamped) == [2.0, 3.0]  # None defaults to 0

    def test_stability_aggregation(self, db):
        tag = "test_v5"
        db.write_injection_stat("global_skew", float("nan"), round_number=1, tag=tag)
        db.write_injection_stat("global_skew", 1.0, round_number=1, slope_clamped=True, tag=tag)
        db.write_injection_stat("global_skew", 2.0, round_number=2, tag=tag)
        assert db.flush(timeout=10) is True

        rows = db.query_injection_stat_stability(stat_name="global_skew", tag=tag)
        by_round = {r["round_number"]: r for r in rows}
        assert by_round[1]["total_count"] == 2
        assert by_round[1]["non_finite_count"] == 1
        assert by_round[1]["clamped_count"] == 1
        assert by_round[2]["total_count"] == 1
        assert by_round[2]["non_finite_count"] == 0


class TestQueryFiltersAndWhitelists:
    @pytest.fixture(autouse=True)
    def _rows(self, db):
        self.db = db
        now = time.time()
        for round_number in (1, 2, 3):
            for tag in ("test_v1", "test_v2"):
                db.write_training_stat(
                    "beta_vae",
                    "total_loss",
                    round_number * 1.0,
                    round_number,
                    1,
                    tag=tag,
                    timestamp=now + round_number,
                )
        assert db.flush(timeout=10) is True

    def test_scalar_filter_uses_equality(self):
        rows = self.db.query_training_stat(tag="test_v1")
        assert len(rows) == 3
        assert all(r["tag"] == "test_v1" for r in rows)

    def test_list_filter_uses_in(self):
        rows = self.db.query_training_stat(tag=["test_v1", "test_v2"])
        assert len(rows) == 6

    def test_range_filters_inclusive(self):
        rows = self.db.query_training_stat(tag="test_v1", start_round_number=2, end_round_number=3)
        assert sorted(r["round_number"] for r in rows) == [2, 3]

    def test_column_projection(self):
        rows = self.db.query_training_stat(tag="test_v1", columns=["value", "round_number"])
        assert rows
        assert all(set(r.keys()) == {"value", "round_number"} for r in rows)

    def test_invalid_column_rejected(self):
        with pytest.raises(ValueError, match="Invalid column"):
            self.db.query_training_stat(columns=["value; DROP TABLE training_stats;--"])

    def test_invalid_column_rejected_per_table(self):
        # round_number is valid for training_stats but not system_resources.
        with pytest.raises(ValueError, match="Invalid column"):
            self.db.query_system_resource(columns=["round_number"])

    def test_empty_list_filter_matches_everything(self):
        # NOTE: this documents CURRENT behavior, not desired behavior — an empty IN-list is
        # treated as "no filter" (the `if tag:` gate in every query_* skips falsy values), so
        # a caller passing a filtered-to-empty list gets the whole table instead of no rows.
        # If query_* semantics are ever fixed to "empty list matches nothing", update this
        # test deliberately rather than treating the failure as a regression.
        assert len(self.db.query_training_stat(tag=[])) == 6

    def test_confidence_bounds_on_inference_results(self):
        self.db.write_inference_result("/a.npy", 0, 1, 0.99, tag="test_v9")
        self.db.write_inference_result("/a.npy", 1, 0, 0.42, tag="test_v9")
        assert self.db.flush(timeout=10) is True
        rows = self.db.query_inference_result(tag="test_v9", min_confidence=0.9)
        assert [r["confidence"] for r in rows] == [0.99]
        rows = self.db.query_inference_result(tag="test_v9", prediction=0)
        assert [r["confidence"] for r in rows] == [0.42]


class TestMarkSuperseded:
    def _write_rounds(self, db, tag, rounds=(1, 2, 3)):
        for round_number in rounds:
            db.write_training_stat("beta_vae", "total_loss", 1.0, round_number, 1, tag=tag)
            db.write_injection_stat("eti_snr", 12.5, round_number=round_number, tag=tag)
            db.write_latent_snapshot(
                "beta_vae", round_number, 1, 10, 0, "true_only_eti", [[0.1] * 4] * 6, tag=tag
            )

    def test_round_ge_marking_and_default_filtering(self, db):
        """Marking (tag, round >= k) must hide those rows from every default query while
        include_superseded=True still returns them — the resume-of-round-k scenario."""
        tag = "test_v1"
        self._write_rounds(db, tag)

        for table in ("training_stats", "injection_stats", "latent_snapshots"):
            assert db.mark_superseded(table, tag, round_ge=2) is True

        assert sorted(r["round_number"] for r in db.query_training_stat(tag=tag)) == [1]
        assert sorted(r["round_number"] for r in db.query_injection_stat(tag=tag)) == [1]
        assert sorted(r["round_number"] for r in db.query_latent_snapshots(tag=tag)) == [1]

        everything = db.query_training_stat(tag=tag, include_superseded=True)
        assert sorted(r["round_number"] for r in everything) == [1, 2, 3]
        assert {r["round_number"]: r["superseded"] for r in everything} == {1: 0, 2: 1, 3: 1}

    def test_mark_flushes_queued_writes_first(self, db):
        """Rows still sitting in the write queue when mark_superseded() is called were
        written before it, so FIFO ordering must land them in the table AND mark them."""
        tag = "test_v2"
        db.write_training_stat("beta_vae", "total_loss", 1.0, 2, 1, tag=tag)  # not yet flushed
        assert db.mark_superseded("training_stats", tag, round_ge=1) is True

        assert db.query_training_stat(tag=tag) == []
        stale = db.query_training_stat(tag=tag, include_superseded=True)
        assert len(stale) == 1 and stale[0]["superseded"] == 1

    def test_rows_written_after_mark_stay_live(self, db):
        tag = "test_v3"
        db.write_training_stat("beta_vae", "total_loss", 1.0, 2, 1, tag=tag)
        assert db.mark_superseded("training_stats", tag, round_ge=2) is True
        db.write_training_stat("beta_vae", "total_loss", 2.0, 2, 1, tag=tag)  # the re-run's row
        assert db.flush(timeout=10) is True

        live = db.query_training_stat(tag=tag)
        assert [r["value"] for r in live] == [2.0]

    def test_mark_scoped_to_tag(self, db):
        self._write_rounds(db, "test_v4", rounds=(1,))
        self._write_rounds(db, "test_v5", rounds=(1,))
        assert db.mark_superseded("training_stats", "test_v4", round_ge=1) is True
        assert db.query_training_stat(tag="test_v4") == []
        assert len(db.query_training_stat(tag="test_v5")) == 1

    def test_npy_path_marking_on_inference_results(self, db):
        tag = "test_v6"
        db.write_inference_result("/a.npy", 0, 1, 0.99, tag=tag)
        db.write_inference_result("/b.npy", 0, 1, 0.88, tag=tag)
        assert db.mark_superseded("inference_results", tag, npy_path="/a.npy") is True

        live = db.query_inference_result(tag=tag)
        assert [r["npy_path"] for r in live] == ["/b.npy"]
        assert len(db.query_inference_result(tag=tag, include_superseded=True)) == 2

    def test_stability_aggregation_excludes_superseded(self, db):
        tag = "test_v7"
        db.write_injection_stat("global_skew", 1.0, round_number=1, tag=tag)
        db.write_injection_stat("global_skew", 2.0, round_number=2, tag=tag)
        assert db.mark_superseded("injection_stats", tag, round_ge=2) is True

        rows = db.query_injection_stat_stability(stat_name="global_skew", tag=tag)
        assert [r["round_number"] for r in rows] == [1]
        rows = db.query_injection_stat_stability(
            stat_name="global_skew", tag=tag, include_superseded=True
        )
        assert [r["round_number"] for r in rows] == [1, 2]

    def test_snapshot_keys_exclude_superseded(self, db):
        tag = "test_v8"
        self._write_rounds(db, tag, rounds=(1, 2))
        assert db.mark_superseded("latent_snapshots", tag, round_ge=2) is True
        keys = db.query_latent_snapshot_keys(tag=tag)
        assert [k["round_number"] for k in keys] == [1]

    def test_mark_without_writer_thread_runs_inline(self):
        """With no writer thread there is nothing queued to order against, so the UPDATE
        runs synchronously in the caller thread (unit-test and post-shutdown ergonomics)."""
        database = Database()
        database.start()
        database.write_training_stat("beta_vae", "total_loss", 1.0, 1, 1, tag="test_v9")
        assert database.flush(timeout=10) is True
        database.stop()

        assert database.mark_superseded("training_stats", "test_v9", round_ge=1) is True
        assert database.query_training_stat(tag="test_v9") == []
        assert len(database.query_training_stat(tag="test_v9", include_superseded=True)) == 1

    def test_invalid_table_rejected(self, db):
        with pytest.raises(ValueError, match="does not support table"):
            db.mark_superseded("system_resources", "test_v1")

    def test_invalid_filters_rejected(self, db):
        with pytest.raises(ValueError, match="round_ge is not supported"):
            db.mark_superseded("inference_results", "test_v1", round_ge=1)
        with pytest.raises(ValueError, match="npy_path is only supported"):
            db.mark_superseded("training_stats", "test_v1", npy_path="/a.npy")
        with pytest.raises(ValueError, match="non-empty tag"):
            db.mark_superseded("training_stats", "")


class TestInferenceCadences:
    """The per-cadence inference run manifest (schema v2): write/query round-trip, JSON
    field serialization, status filtering, and the supersede-on-retry flow the stage-aware
    resume relies on."""

    def test_write_query_round_trip(self, db):
        tag = "test_v1"
        summary = {"n": 4, "mean": 0.5, "quantiles": {"p50": 0.5}}
        db.write_inference_cadence(
            npy_path="/pre/cad_a.npy",
            status="inferred",
            tag=tag,
            csv_path="/catalog.csv",
            cadence_key=("HIP110750", "AGBT21B_999_31", "L", "0", "1400"),
            n_stamps=4,
            n_candidates=1,
            confidence_summary=summary,
            duration_s=12.5,
        )
        assert db.flush(timeout=10) is True

        rows = db.query_inference_cadences(tag=tag)
        assert len(rows) == 1
        row = rows[0]
        assert row["npy_path"] == "/pre/cad_a.npy"
        assert row["status"] == "inferred"
        assert row["csv_path"] == "/catalog.csv"
        assert json.loads(row["cadence_key"]) == ["HIP110750", "AGBT21B_999_31", "L", "0", "1400"]
        assert row["n_stamps"] == 4
        assert row["n_candidates"] == 1
        assert json.loads(row["confidence_summary"]) == summary
        assert row["duration_s"] == 12.5
        assert row["superseded"] == 0

    def test_optional_fields_default_to_none(self, db):
        db.write_inference_cadence(npy_path="/pre/cad_b.npy", status="preprocessed", tag="test_v1")
        assert db.flush(timeout=10) is True
        row = db.query_inference_cadences(tag="test_v1")[0]
        assert row["cadence_key"] is None
        assert row["confidence_summary"] is None
        assert row["n_candidates"] is None

    def test_status_and_npy_path_filters(self, db):
        tag = "test_v2"
        db.write_inference_cadence(npy_path="/a.npy", status="preprocessed", tag=tag, n_stamps=3)
        db.write_inference_cadence(npy_path="/a.npy", status="inferred", tag=tag, n_stamps=3)
        db.write_inference_cadence(npy_path="/b.npy", status="failed", tag=tag)
        assert db.flush(timeout=10) is True

        inferred = db.query_inference_cadences(tag=tag, status="inferred")
        assert [r["npy_path"] for r in inferred] == ["/a.npy"]
        a_rows = db.query_inference_cadences(tag=tag, npy_path="/a.npy")
        assert sorted(r["status"] for r in a_rows) == ["inferred", "preprocessed"]
        multi = db.query_inference_cadences(tag=tag, status=["inferred", "failed"])
        assert sorted(r["status"] for r in multi) == ["failed", "inferred"]

    def test_supersede_on_retry_flow(self, db):
        """The manifest state machine on a retried cadence: 'preprocessed' + 'failed' rows
        from the dead attempt are superseded before the fresh 'inferred' row is written, so
        the resume query (status='inferred', default superseded filter) flips from empty to
        exactly one row."""
        tag = "test_v3"
        npy = "/pre/cad_c.npy"
        db.write_inference_cadence(npy_path=npy, status="preprocessed", tag=tag, n_stamps=7)
        db.write_inference_cadence(npy_path=npy, status="failed", tag=tag)

        # Before the retry completes: no live 'inferred' row -> the cadence is re-attempted
        assert db.flush(timeout=10) is True
        assert db.query_inference_cadences(tag=tag, npy_path=npy, status="inferred") == []

        # Retry succeeds: supersede old rows, then write the fresh 'inferred' row
        assert db.mark_superseded("inference_cadences", tag, npy_path=npy) is True
        db.write_inference_cadence(
            npy_path=npy, status="inferred", tag=tag, n_stamps=7, n_candidates=0
        )
        assert db.flush(timeout=10) is True

        live = db.query_inference_cadences(tag=tag, npy_path=npy)
        assert [r["status"] for r in live] == ["inferred"]
        all_rows = db.query_inference_cadences(tag=tag, npy_path=npy, include_superseded=True)
        assert len(all_rows) == 3

    def test_supersede_scoped_to_npy_path(self, db):
        tag = "test_v4"
        db.write_inference_cadence(npy_path="/a.npy", status="inferred", tag=tag)
        db.write_inference_cadence(npy_path="/b.npy", status="inferred", tag=tag)
        assert db.mark_superseded("inference_cadences", tag, npy_path="/a.npy") is True
        live = db.query_inference_cadences(tag=tag, status="inferred")
        assert [r["npy_path"] for r in live] == ["/b.npy"]

    def test_invalid_column_rejected(self, db):
        with pytest.raises(ValueError, match="Invalid column"):
            db.query_inference_cadences(columns=["npy_path; DROP TABLE inference_cadences"])


class TestPipelineStages:
    def test_write_and_query_round_trip(self, db):
        tag = "test_v1"
        db.write_pipeline_stage("train.round_01", 100.0, 160.0, tag=tag, metadata='{"a": 1}')
        assert db.flush(timeout=10) is True

        rows = db.query_pipeline_stages(tag=tag)
        assert len(rows) == 1
        row = rows[0]
        assert row["stage"] == "train.round_01"
        assert row["start_time"] == 100.0
        assert row["end_time"] == 160.0
        assert row["duration_s"] == 60.0  # derived at write time
        assert json.loads(row["metadata"]) == {"a": 1}

    def test_query_filters_and_ordering(self, db):
        tag = "test_v1"
        # Inserted out of chronological order on purpose
        db.write_pipeline_stage("train.round_02", 200.0, 260.0, tag=tag)
        db.write_pipeline_stage("train.round_01", 100.0, 160.0, tag=tag)
        db.write_pipeline_stage("inference.viz", 300.0, 310.0, tag=tag)
        db.write_pipeline_stage("train.round_01", 100.0, 150.0, tag="test_v2")
        assert db.flush(timeout=10) is True

        rows = db.query_pipeline_stages(tag=tag)
        assert [r["stage"] for r in rows] == ["train.round_01", "train.round_02", "inference.viz"]

        # stage accepts str or list; start/end bound the row's start_time
        assert len(db.query_pipeline_stages(stage="inference.viz", tag=tag)) == 1
        assert len(db.query_pipeline_stages(stage=["train.round_01", "train.round_02"])) == 3
        assert [
            r["stage"] for r in db.query_pipeline_stages(tag=tag, start_time=150.0, end_time=250.0)
        ] == ["train.round_02"]

    def test_column_projection_and_whitelist(self, db):
        db.write_pipeline_stage("train.rf", 1.0, 2.0, tag="test_v1")
        assert db.flush(timeout=10) is True
        rows = db.query_pipeline_stages(tag="test_v1", columns=["stage", "duration_s"])
        assert rows == [{"stage": "train.rf", "duration_s": 1.0}]
        with pytest.raises(ValueError, match="Invalid column"):
            db.query_pipeline_stages(columns=["stage; DROP TABLE pipeline_stages"])


class TestSchemaMigration:
    # The v0 schema (pre-superseded) for the four tables the migration touches, trimmed to
    # the columns the assertions need plus everything NOT NULL.
    _V0_SCHEMA = """
        CREATE TABLE training_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            model_name TEXT NOT NULL,
            stat_name TEXT NOT NULL,
            value REAL NOT NULL,
            round_number INTEGER,
            epoch_number INTEGER,
            tag TEXT,
            metadata TEXT
        );
        CREATE TABLE injection_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            stat_name TEXT NOT NULL,
            value REAL NOT NULL,
            round_number INTEGER,
            chunk_number INTEGER,
            sample_index INTEGER,
            background_index INTEGER,
            signal_class TEXT,
            signal_type TEXT,
            injection_stage TEXT,
            is_finite INTEGER DEFAULT 1,
            slope_clamped INTEGER DEFAULT 0,
            tag TEXT,
            metadata TEXT
        );
        CREATE TABLE latent_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            model_name TEXT NOT NULL,
            round_number INTEGER NOT NULL,
            epoch_number INTEGER NOT NULL,
            step_number INTEGER NOT NULL,
            cadence_index INTEGER NOT NULL,
            signal_type TEXT NOT NULL,
            latent_vector TEXT NOT NULL,
            snr_base INTEGER,
            snr_range INTEGER,
            tag TEXT,
            metadata TEXT
        );
        CREATE TABLE inference_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            npy_path TEXT NOT NULL,
            snippet_index INTEGER NOT NULL,
            prediction INTEGER NOT NULL,
            confidence REAL NOT NULL,
            latent_vector TEXT,
            target TEXT,
            session TEXT,
            cadence_id INTEGER,
            band TEXT,
            frequency_mhz REAL,
            timestamp_observed REAL,
            h5_path TEXT,
            tag TEXT,
            metadata TEXT
        );
    """

    _MIGRATED_TABLES = (
        "training_stats",
        "injection_stats",
        "latent_snapshots",
        "inference_results",
    )

    def _create_v0_db(self, config):
        """Lay down an old-schema (pre-superseded, user_version 0) db file with one row,
        at the exact path Database will open."""
        db_path = os.path.join(config.output_path, "db", "aetherscan.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(self._V0_SCHEMA)
        conn.execute(
            "INSERT INTO training_stats (timestamp, model_name, stat_name, value, round_number,"
            " epoch_number, tag, metadata) VALUES (1.0, 'beta_vae', 'total_loss', 0.5, 1, 1,"
            " 'test_v1', NULL)"
        )
        conn.commit()
        conn.close()
        return db_path

    def _column_names(self, db_path, table):
        conn = sqlite3.connect(db_path)
        try:
            return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        finally:
            conn.close()

    def _user_version(self, db_path):
        conn = sqlite3.connect(db_path)
        try:
            return conn.execute("PRAGMA user_version").fetchone()[0]
        finally:
            conn.close()

    def _table_names(self, db_path):
        conn = sqlite3.connect(db_path)
        try:
            return {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        finally:
            conn.close()

    def test_old_schema_gains_superseded_column(self):
        db_path = self._create_v0_db(get_config())
        assert self._user_version(db_path) == 0

        database = Database()  # _init_database() runs the migration
        try:
            for table in self._MIGRATED_TABLES:
                assert "superseded" in self._column_names(db_path, table)
            assert self._user_version(db_path) == _SCHEMA_VERSION

            # Pre-migration rows default to superseded = 0 and stay visible
            rows = database.query_training_stat(tag="test_v1")
            assert len(rows) == 1
            assert rows[0]["superseded"] == 0
        finally:
            database.stop()

    def test_old_schema_gains_inference_cadences_table(self):
        """A pre-versioning database (v0: no inference_cadences table) must come out of
        _init_database() with the v2 manifest table present and usable."""
        db_path = self._create_v0_db(get_config())
        assert "inference_cadences" not in self._table_names(db_path)

        database = Database()
        database.start()
        try:
            assert "inference_cadences" in self._table_names(db_path)
            assert self._user_version(db_path) == _SCHEMA_VERSION
            database.write_inference_cadence(npy_path="/a.npy", status="inferred", tag="test_v1")
            assert database.flush(timeout=10) is True
            assert len(database.query_inference_cadences(tag="test_v1")) == 1
        finally:
            database.stop()

    def test_old_schema_gains_pipeline_stages_table(self):
        """A pre-versioning database (v0: no pipeline_stages table) must come out of
        _init_database() with the v4 stage-timing table present and usable."""
        db_path = self._create_v0_db(get_config())
        assert "pipeline_stages" not in self._table_names(db_path)

        database = Database()
        database.start()
        try:
            assert "pipeline_stages" in self._table_names(db_path)
            assert self._user_version(db_path) == _SCHEMA_VERSION
            database.write_pipeline_stage("train.round_01", 1.0, 2.5, tag="test_v1")
            assert database.flush(timeout=10) is True
            rows = database.query_pipeline_stages(tag="test_v1")
            assert len(rows) == 1
            assert rows[0]["duration_s"] == 1.5
        finally:
            database.stop()

    def test_v2_schema_migrates_to_v4(self):
        """A database stamped at v2 (superseded columns + inference_cadences already
        present, no config_fingerprint, no pipeline_stages) must gain the v3
        config_fingerprint column, the v4 pipeline_stages table, and the v4 stamp."""
        config = get_config()
        db_path = os.path.join(config.output_path, "db", "aetherscan.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(self._V0_SCHEMA)
        for table in self._MIGRATED_TABLES:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN superseded INTEGER DEFAULT 0")
        conn.execute(
            "CREATE TABLE inference_cadences (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " timestamp REAL NOT NULL, tag TEXT, csv_path TEXT, cadence_key TEXT,"
            " npy_path TEXT NOT NULL, status TEXT NOT NULL, n_stamps INTEGER,"
            " n_candidates INTEGER, confidence_summary TEXT, duration_s REAL,"
            " superseded INTEGER DEFAULT 0)"
        )
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        conn.close()

        database = Database()
        try:
            # v3 ALTER: the pre-existing inference_cadences table gains config_fingerprint
            assert "config_fingerprint" in self._column_names(db_path, "inference_cadences")
            # v4: the new stage-timing table is created
            assert "pipeline_stages" in self._table_names(db_path)
            assert self._user_version(db_path) == _SCHEMA_VERSION
        finally:
            database.stop()

    def test_migration_is_idempotent_across_reopens(self):
        db_path = self._create_v0_db(get_config())

        # First open migrates; second open must be a clean no-op (no duplicate-column error)
        first = Database()
        first.stop()
        Database._reset()
        second = Database()
        try:
            for table in self._MIGRATED_TABLES:
                assert "superseded" in self._column_names(db_path, table)
            assert self._user_version(db_path) == _SCHEMA_VERSION
            assert len(second.query_training_stat(tag="test_v1")) == 1
        finally:
            second.stop()

    def test_fresh_db_created_at_current_version(self, db):
        for table in self._MIGRATED_TABLES:
            assert "superseded" in self._column_names(db.db_path, table)
        assert self._user_version(db.db_path) == _SCHEMA_VERSION
