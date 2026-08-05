# NOTE: come back to this later

"""Unit tests for aetherscan.db: writer thread lifecycle, flush sentinel protocol, executemany
batching across tables, is_finite sanitization, and query filters / column whitelists — all
against a tmp-path SQLite file."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager

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


class TestConnectionPragmas:
    def test_temp_store_is_memory(self, db):
        """_get_connection sets temp_store=MEMORY so large GROUP BY/ORDER BY/window sorts in the
        teardown/plot passes stay in RAM instead of spilling to an on-disk temp b-tree
        (2 == MEMORY; 0 == DEFAULT, 1 == FILE)."""
        with db._get_connection() as conn:
            assert conn.execute("PRAGMA temp_store").fetchone()[0] == 2

    def test_synchronous_is_normal(self, db):
        """The pre-existing synchronous=NORMAL pragma (#277) must still apply alongside the new
        temp_store setting (1 == NORMAL)."""
        with db._get_connection() as conn:
            assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1


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

    def test_latent_snapshots_bulk_matches_per_row_semantics(self, db):
        """write_latent_snapshots_bulk (one call per snapshot capture) must land rows with
        the same content per-row write_latent_snapshot calls produce."""
        tag = "test_v3"
        vector = np.round(np.arange(24, dtype=np.float64).reshape(6, 4) / 7.0, 8)
        db.write_latent_snapshots_bulk(
            model_name="beta_vae",
            round_number=2,
            epoch_number=3,
            step_number=5,
            snr_base=10,
            snr_range=20,
            tag=tag,
            timestamp=123.0,
            snapshots=[(0, "true_only_eti", vector), (1, "false_with_rfi", vector + 1.0)],
        )
        assert db.flush(timeout=10) is True
        snapshots = db.query_latent_snapshots(tag=tag)
        assert len(snapshots) == 2
        by_idx = {s["cadence_index"]: s for s in snapshots}
        assert json.loads(by_idx[0]["latent_vector"]) == vector.tolist()
        assert json.loads(by_idx[1]["latent_vector"]) == (vector + 1.0).tolist()
        assert by_idx[0]["signal_type"] == "true_only_eti"
        assert by_idx[1]["signal_type"] == "false_with_rfi"


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

    def test_null_round_number_escapes_round_ge(self, db):
        """Documents the mechanism behind #210 / KNOWN_ISSUES entry 17: SQL comparisons against
        NULL never match, so a row written without a round_number (the RF-generation call's old
        default) is invisible to round_ge no matter the threshold — it stays live forever."""
        tag = "test_v_null_round"
        db.write_injection_stat("eti_snr", 12.5, round_number=None, tag=tag)
        assert db.mark_superseded("injection_stats", tag, round_ge=1) is True

        live = db.query_injection_stat(tag=tag)
        assert len(live) == 1
        assert live[0]["superseded"] == 0

    def test_sentinel_round_number_is_supersedable(self, db):
        """The fix for #210: giving RF-generation rows a real round_number (num_training_rounds
        + 1, instead of the None default) makes them reachable by the exact same round_ge
        supersede call _init_run_state already issues on every retry -- no NULL-aware special
        case needed. A stale row from a crashed rf_train attempt (sentinel=21) is marked
        superseded once the resumed attempt's round_ge reaches that sentinel; a live row from a
        completed VAE round below it survives."""
        tag = "test_v_sentinel_round"
        sentinel = 21  # num_training_rounds=20 + 1, as train_random_forest computes it
        db.write_injection_stat("eti_snr", 1.0, round_number=20, tag=tag)  # completed VAE round
        db.write_injection_stat("eti_snr", 2.0, round_number=sentinel, tag=tag)  # stale RF attempt
        assert db.mark_superseded("injection_stats", tag, round_ge=sentinel) is True

        live = db.query_injection_stat(tag=tag)
        assert [r["round_number"] for r in live] == [20]

        everything = db.query_injection_stat(tag=tag, include_superseded=True)
        assert {r["round_number"]: r["superseded"] for r in everything} == {20: 0, sentinel: 1}

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

    # The complete expected index set per table — the v9 audit's final state. Both
    # _init_database() (fresh dbs) and the migration blocks (older dbs) must land exactly
    # here, so tests assert equality, not membership.
    _EXPECTED_INDEXES = {
        "system_resources": {"idx_system_resources_filter"},
        "injection_stats": {
            "idx_injection_stats_filter",
            "idx_injection_stats_by_stat",
            "idx_injection_stats_by_round",
        },
        "training_stats": {"idx_training_stats_filter"},
        "latent_snapshots": {"idx_latent_snapshots_by_key", "idx_latent_snapshots_keys"},
        "inference_results": {"idx_inference_results_filter", "idx_inference_results_supersede"},
        "inference_cadences": {"idx_inference_cadences_filter"},
        "pipeline_stages": {"idx_pipeline_stages_filter"},
    }

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

    def _index_names(self, db_path, table):
        conn = sqlite3.connect(db_path)
        try:
            return {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ?",
                    (table,),
                )
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

    def test_old_schema_gains_training_stats_is_finite(self):
        # v6 (#289): pre-v6 rows were all finite by construction (a non-finite value could
        # never be written — it bound as NULL and blew the NOT NULL constraint), so the
        # DEFAULT 1 backfill is exact and they stay visible under the only_finite default.
        db_path = self._create_v0_db(get_config())
        database = Database()
        try:
            assert "is_finite" in self._column_names(db_path, "training_stats")
            rows = database.query_training_stat(tag="test_v1")
            assert len(rows) == 1
            assert rows[0]["is_finite"] == 1
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
        database.start()
        try:
            # v3 ALTER: the pre-existing inference_cadences table gains config_fingerprint
            assert "config_fingerprint" in self._column_names(db_path, "inference_cadences")
            # v4: the new stage-timing table is created
            assert "pipeline_stages" in self._table_names(db_path)
            assert self._user_version(db_path) == _SCHEMA_VERSION
            # ...and the table created during a v2->v4 migration is actually usable
            database.write_pipeline_stage("train.round_01", 10.0, 12.5, tag="test_v2")
            assert database.flush(timeout=10) is True
            rows = database.query_pipeline_stages(tag="test_v2")
            assert len(rows) == 1
            assert rows[0]["duration_s"] == 2.5
        finally:
            database.stop()

    def test_migration_reaches_final_index_set(self):
        """A pre-v7 database — seeded with the retired (tag, timestamp, ...)
        latent_snapshots index so the v7 DROP path is exercised, not just the CREATEs —
        must come out of _init_database() with exactly the current index set on every
        table, plus the version stamp."""
        db_path = self._create_v0_db(get_config())
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE INDEX idx_latent_snapshots_filter ON latent_snapshots"
            "(tag, timestamp, model_name, round_number, epoch_number, step_number)"
        )
        conn.commit()
        conn.close()

        database = Database()
        try:
            for table, expected in self._EXPECTED_INDEXES.items():
                assert self._index_names(db_path, table) == expected, table
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
        # The fresh-db index set must equal the migrated end state exactly (requirement of
        # the v7 sweep: _init_database() and the migration block may never diverge)
        for table, expected in self._EXPECTED_INDEXES.items():
            assert self._index_names(db.db_path, table) == expected, table

    def test_init_never_runs_the_count_storm(self, monkeypatch):
        """#301: Database.__init__ must not call get_db_stats — its per-table COUNT(*)
        scans cost a measured ~13 min per cold-cache launch on a catalog-scale DB. The
        method stays available for on-demand diagnostics, but startup may only log O(1)
        facts."""

        def _boom(self):
            raise AssertionError("get_db_stats must not run at Database init (#301)")

        monkeypatch.setattr(Database, "get_db_stats", _boom)
        database = Database()
        try:
            # ...and the method itself still works when explicitly requested
            monkeypatch.undo()
            stats = database.get_db_stats()
            assert "db_size_bytes" in stats
        finally:
            database.stop()


def _bulk_rows(n: int, round_number: int | None = 1, value: float = 1.0) -> list[dict]:
    """Injection-stat row dicts for write_injection_stats_bulk (#277 tests)."""
    return [
        {
            "stat_name": "global_mean",
            "value": value,
            "round_number": round_number,
            "chunk_number": 0,
            "sample_index": i,
            "background_index": i,
            "signal_class": "true",
            "signal_type": "true_only_eti",
            "injection_stage": "A",
        }
        for i in range(n)
    ]


def _wait_backlog_empty(database, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if database.injection_backlog_rows() == 0:
            return True
        time.sleep(0.05)
    return False


class TestTrainingStatFiniteHandling:
    """#289: NaN training stats used to bind as SQL NULL, blow the NOT NULL constraint, and
    silently vanish the entire flush batch. They now store as 0.0 with is_finite=0, queries
    filter them by default, and a batch failure falls back to per-row writes."""

    def test_nan_and_none_values_stored_flagged_and_filtered(self, db):
        db.write_training_stat("beta_vae", "good_stat", 1.5, tag="fin_v1")
        db.write_training_stat("beta_vae", "nan_stat", float("nan"), tag="fin_v1")
        db.write_training_stat("beta_vae", "none_stat", None, tag="fin_v1")
        assert db.flush()
        # Default query drops the non-finite rows...
        rows = db.query_training_stat(tag="fin_v1", columns=["stat_name", "value"])
        assert [r["stat_name"] for r in rows] == ["good_stat"]
        # ...but they are recorded (0.0, is_finite=0), not lost — and the good row survived
        # in the same batch (the pre-#289 behavior dropped all three).
        all_rows = db.query_training_stat(
            tag="fin_v1", only_finite=False, columns=["stat_name", "value", "is_finite"]
        )
        assert len(all_rows) == 3
        flagged = {r["stat_name"]: (r["value"], r["is_finite"]) for r in all_rows}
        assert flagged["nan_stat"] == (0.0, 0)
        assert flagged["none_stat"] == (0.0, 0)
        assert flagged["good_stat"] == (1.5, 1)

    def test_flush_falls_back_to_per_row_on_poisoned_batch(self, db):
        # Smuggle a raw record violating NOT NULL (stat_name=None) around the sanitized
        # write path, alongside a good row: the batch insert fails, the fallback lands the
        # good row and drops exactly the bad one.
        now = time.time()
        good = ("training_stats", (now, "m", "ok_stat", 2.0, 1, 1, "fallback_v1", None, 1))
        bad = ("training_stats", (now, "m", None, 3.0, 1, 1, "fallback_v1", None, 1))
        db._flush_buffer(buffer=[bad, good])
        rows = db.query_training_stat(tag="fallback_v1", columns=["stat_name", "value"])
        assert [(r["stat_name"], r["value"]) for r in rows] == [("ok_stat", 2.0)]

    def test_fallback_does_not_duplicate_rows_before_a_mid_batch_poison(self, db):
        # The poisoned row sits MID-batch: executemany steps good rows into the open
        # transaction before failing, and without the SAVEPOINT rollback the per-row retry
        # would re-insert them (duplicates). Exactly one copy of each good row must land.
        now = time.time()
        good1 = ("training_stats", (now, "m", "first_stat", 1.0, 1, 1, "fallback_v2", None, 1))
        bad = ("training_stats", (now, "m", None, 3.0, 1, 1, "fallback_v2", None, 1))
        good2 = ("training_stats", (now, "m", "second_stat", 2.0, 1, 1, "fallback_v2", None, 1))
        db._flush_buffer(buffer=[good1, bad, good2])
        rows = db.query_training_stat(tag="fallback_v2", columns=["stat_name", "value"])
        assert sorted((r["stat_name"], r["value"]) for r in rows) == [
            ("first_stat", 1.0),
            ("second_stat", 2.0),
        ]


class TestInjectionStatTimeSpan:
    """query_injection_stat_time_span: the whole-partition MIN/MAX aggregate that callers use
    to tighten (tag, timestamp) index windows for round-scoped queries."""

    def test_span_by_round_and_missing_tag(self, db):
        rows = _bulk_rows(3, round_number=1) + _bulk_rows(3, round_number=2)
        for i, r in enumerate(rows[:3]):
            r["timestamp"] = 100.0 + i  # round 1: 100..102
        for i, r in enumerate(rows[3:]):
            r["timestamp"] = 200.0 + i  # round 2: 200..202
        db.write_injection_stats_bulk(rows, tag="span_v1")
        assert _wait_backlog_empty(db)
        assert db.query_injection_stat_time_span(tag="span_v1") == (100.0, 202.0)
        assert db.query_injection_stat_time_span(tag="span_v1", start_round_number=2) == (
            200.0,
            202.0,
        )
        assert db.query_injection_stat_time_span(
            tag="span_v1", start_round_number=1, end_round_number=1
        ) == (100.0, 102.0)
        assert db.query_injection_stat_time_span(tag="no_such_tag") is None

    def test_span_includes_superseded_and_non_finite_rows(self, db):
        # The span deliberately ignores the is_finite / superseded filters row-level queries
        # apply: it must be a SUPERSET bound, so intersecting a query window with it can
        # never drop a row that query would have returned.
        rows = _bulk_rows(2, round_number=1)
        rows[0]["timestamp"] = 50.0
        rows[0]["value"] = float("nan")  # sanitized to is_finite=0 at write time
        rows[1]["timestamp"] = 60.0
        db.write_injection_stats_bulk(rows, tag="span_v2")
        assert _wait_backlog_empty(db)
        db.mark_superseded("injection_stats", tag="span_v2", round_ge=1)
        assert db.query_injection_stat_time_span(tag="span_v2") == (50.0, 60.0)


class TestV9IndexPlannerContracts:
    """The v9 by_round index (#375) must serve the supersede/time-span shapes WITHOUT
    stealing the plot-pass plans from idx_injection_stats_by_stat — the `+round_number`
    guards in query_injection_stat / query_injection_stat_stability enforce the latter —
    and the latent partial index must serve the keys query without stealing the per-frame
    or supersede shapes from by_key. Pin every direction with EXPLAIN QUERY PLAN on the
    real schema (no ANALYZE, matching production, where the un-stat'd cost model would
    otherwise prefer a leading-range index). The SQL under test is captured from the REAL
    builders via a connection trace, so dropping a `+` guard in db.py fails here — the
    assertions cover the builders, not hand-written literals that could drift from them."""

    @pytest.fixture()
    def traced(self, db, monkeypatch):
        """(db, captured): every statement the Database's own connections execute lands in
        `captured` — including the inline (no-writer) mark_superseded path."""
        captured: list[str] = []
        orig = Database._get_connection

        @contextmanager
        def traced_connection(db_self):
            with orig(db_self) as conn:
                conn.set_trace_callback(captured.append)
                try:
                    yield conn
                finally:
                    conn.set_trace_callback(None)

        monkeypatch.setattr(Database, "_get_connection", traced_connection)
        return db, captured

    def _plan_of_last(self, db, captured, needle):
        """EXPLAIN QUERY PLAN of the most recent captured statement containing `needle`.
        Works whether the trace delivers placeholder or expanded SQL: unbound placeholders
        plan identically to bound ones, so Nones suffice. The plan string carries the
        SQLite version so a failure years from now reads as 'the toolchain's planner
        changed' rather than 'someone deleted a guard'."""
        stmt = next(
            (s for s in reversed(captured) if needle in s and not s.startswith("EXPLAIN")), None
        )
        assert stmt is not None, (
            f"no captured statement contains {needle!r} — builder refactor? "
            f"{len(captured)} statements captured"
        )
        conn = sqlite3.connect(db.db_path)
        try:
            rows = conn.execute("EXPLAIN QUERY PLAN " + stmt, [None] * stmt.count("?")).fetchall()
        finally:
            conn.close()
        plan = " | ".join(str(row) for row in rows)
        return stmt, f"{plan} [sqlite {sqlite3.sqlite_version}]"

    def _seed_injection(self, db):
        rows = _bulk_rows(6, round_number=1) + _bulk_rows(6, round_number=2)
        db.write_injection_stats_bulk(rows, tag="plan_v1")
        assert _wait_backlog_empty(db)

    def _seed_latent(self, db):
        for step in (10, 20):
            db.write_latent_snapshot(
                model_name="beta_vae",
                round_number=1,
                epoch_number=1,
                step_number=step,
                cadence_index=0,
                signal_type="true_only_eti",
                latent_vector=[0.0] * 8,
                snr_base=1,
                snr_range=99,
                tag="plan_v2",
            )
        assert db.flush(timeout=10.0)

    def test_injection_supersede_update_uses_by_round(self, traced):
        db, captured = traced
        self._seed_injection(db)
        db.stop()  # no writer → mark_superseded executes inline through _get_connection
        db.mark_superseded("injection_stats", tag="plan_v1", round_ge=2)
        stmt, plan = self._plan_of_last(db, captured, "UPDATE injection_stats")
        assert "idx_injection_stats_by_round" in plan, (stmt, plan)

    def test_time_span_uses_by_round_covering(self, traced):
        db, captured = traced
        self._seed_injection(db)
        db.query_injection_stat_time_span(tag="plan_v1", start_round_number=1, end_round_number=2)
        stmt, plan = self._plan_of_last(db, captured, "MIN(timestamp)")
        assert "idx_injection_stats_by_round" in plan, (stmt, plan)
        assert "COVERING" in plan.upper(), plan

    def test_guarded_plot_query_keeps_by_stat(self, traced):
        # The builder must emit `+round_number` (the guard itself) AND the resulting plan
        # must keep the v7 stat-scoped index — positive and negative pinned together.
        db, captured = traced
        self._seed_injection(db)
        db.query_injection_stat(
            tag="plan_v1",
            stat_name="eti_snr",
            injection_stage="post",
            start_round_number=1,
            end_round_number=2,
        )
        stmt, plan = self._plan_of_last(db, captured, "FROM injection_stats")
        # COUNT, not substring: both the start- and end-bound terms must carry the guard —
        # deleting either one alone would still satisfy an `in` check. The negative states
        # the invariant directly: no unguarded round_number term reaches the planner.
        assert stmt.count("+round_number") == 2, stmt
        assert "AND round_number" not in stmt, stmt
        assert "idx_injection_stats_by_round" not in plan, (stmt, plan)
        assert "idx_injection_stats_by_stat" in plan, (stmt, plan)

    def test_guarded_stability_query_keeps_by_stat(self, traced):
        # No timestamp bounds — the hardest case: with a bare GROUP BY round_number the
        # planner picks by_round(tag=?) purely for its free grouping order, so the builder
        # guards the GROUP BY/ORDER BY with `+` as well.
        db, captured = traced
        self._seed_injection(db)
        db.query_injection_stat_stability(
            tag="plan_v1", stat_name="eti_snr", start_round_number=1, end_round_number=2
        )
        stmt, plan = self._plan_of_last(db, captured, "GROUP BY")
        # All four guarded sites, individually deletable, individually pinned: the two WHERE
        # bounds plus GROUP BY plus ORDER BY — and no unguarded round_number term anywhere
        # (the negative also catches a guard MOVED rather than deleted).
        assert stmt.count("+round_number") == 4, stmt
        assert "GROUP BY +round_number" in stmt, stmt
        assert "ORDER BY +round_number" in stmt, stmt
        assert "AND round_number" not in stmt, stmt
        assert "GROUP BY round_number" not in stmt, stmt
        assert "ORDER BY round_number" not in stmt, stmt
        assert "idx_injection_stats_by_round" not in plan, (stmt, plan)
        assert "idx_injection_stats_by_stat" in plan, (stmt, plan)

    def test_latent_keys_query_is_covering_on_partial_index(self, traced):
        db, captured = traced
        self._seed_latent(db)
        db.query_latent_snapshot_keys(tag="plan_v2", start_time=0.0)
        stmt, plan = self._plan_of_last(db, captured, "SELECT DISTINCT")
        assert "idx_latent_snapshots_keys" in plan, (stmt, plan)
        assert "COVERING" in plan.upper(), plan
        # Index-only AND sort-free: DISTINCT + ORDER BY must come from index order, not a
        # temp b-tree — the full "index-only scan" claim in DATABASE.md.
        assert "TEMP B-TREE" not in plan.upper(), plan

    def test_latent_per_frame_query_keeps_by_key(self, traced):
        # The new partial index is also predicate-eligible here (bare superseded = 0);
        # by_key's full six-column seek must keep winning.
        db, captured = traced
        self._seed_latent(db)
        db.query_latent_snapshots(
            model_name="beta_vae",
            round_number=1,
            epoch_number=1,
            step_number=10,
            tag="plan_v2",
            start_time=0.0,
            end_time=99.0,
        )
        stmt, plan = self._plan_of_last(db, captured, "FROM latent_snapshots")
        assert "idx_latent_snapshots_by_key" in plan, (stmt, plan)
        assert "idx_latent_snapshots_keys" not in plan, (stmt, plan)

    def test_latent_supersede_update_keeps_by_key(self, traced):
        db, captured = traced
        self._seed_latent(db)
        db.stop()
        db.mark_superseded("latent_snapshots", tag="plan_v2", round_ge=1)
        stmt, plan = self._plan_of_last(db, captured, "UPDATE latent_snapshots")
        assert "idx_latent_snapshots_by_key" in plan, (stmt, plan)
        assert "idx_latent_snapshots_keys" not in plan, (stmt, plan)


class TestBulkLane:
    """#277: high-volume injection stats ride a bounded bulk lane separate from the
    foreground queue, with per-round pending accounting."""

    def test_bulk_rows_round_trip(self, db):
        db.write_injection_stats_bulk(_bulk_rows(25), tag="bulk_v1")
        assert _wait_backlog_empty(db)
        rows = db.query_injection_stat(tag="bulk_v1", columns=["sample_index", "value"])
        assert len(rows) == 25

    def test_bulk_inline_when_writer_not_running(self):
        database = Database()
        # Writer never started: rows are written inline with zero backlog
        database.write_injection_stats_bulk(_bulk_rows(7), tag="bulk_inline")
        assert database.injection_backlog_rows() == 0
        assert len(database.query_injection_stat(tag="bulk_inline")) == 7

    def test_bulk_sanitizes_non_finite(self, db):
        rows = _bulk_rows(2)
        rows[0]["value"] = float("nan")
        db.write_injection_stats_bulk(rows, tag="bulk_nan")
        assert _wait_backlog_empty(db)
        assert len(db.query_injection_stat(tag="bulk_nan", only_finite=True)) == 1
        all_rows = db.query_injection_stat(
            tag="bulk_nan", only_finite=False, columns=["value", "is_finite"]
        )
        assert len(all_rows) == 2
        coerced = [r for r in all_rows if r["is_finite"] == 0]
        assert len(coerced) == 1 and coerced[0]["value"] == 0.0

    def test_bulk_chunking_and_backlog_accounting(self):
        database = Database()
        database.bulk_chunk_rows = 10

        class _AliveStub:
            def is_alive(self):
                return True

        # Stub writer: chunks are enqueued but never consumed, so the chunk/backlog math
        # is directly observable
        database.writer_thread = _AliveStub()
        try:
            database.write_injection_stats_bulk(_bulk_rows(25, round_number=2), tag="bulk_chunks")
            assert database.bulk_queue.qsize() == 3  # 10 + 10 + 5
            assert database.injection_backlog_rows() == 25
            assert database.injection_backlog_rows(max_round=1) == 0
            assert database.injection_backlog_rows(max_round=2) == 25
            # None round_number counts against every max_round (conservative)
            database.write_injection_stats_bulk(_bulk_rows(3, round_number=None), tag="bulk_chunks")
            assert database.injection_backlog_rows(max_round=1) == 3
        finally:
            database.writer_thread = None

    def test_flush_does_not_wait_for_bulk_lane(self, db, monkeypatch):
        # Slow every commit so the bulk backlog outlives the foreground flush
        real_flush = db._flush_buffer

        def slow_flush(buffer=None):
            time.sleep(0.3)
            real_flush(buffer)

        monkeypatch.setattr(db, "_flush_buffer", slow_flush)
        db.bulk_chunk_rows = 50
        for _ in range(6):
            db.write_injection_stats_bulk(_bulk_rows(50), tag="bulk_load")
        db.write_training_stat(model_name="m", stat_name="s", value=1.0, tag="bulk_load_fg")

        assert db.flush(timeout=10.0) is True
        # The foreground flush completed while bulk chunks were still queued — the whole
        # point of the two-lane design (#277 problem 2)
        assert db.injection_backlog_rows() > 0
        assert _wait_backlog_empty(db)  # let the fixture teardown finish cleanly

    def test_stability_query_round_bounds(self, db):
        for round_number in (1, 2, 3):
            db.write_injection_stats_bulk(_bulk_rows(4, round_number=round_number), tag="rb")
        assert _wait_backlog_empty(db)
        rows = db.query_injection_stat_stability(
            stat_name="global_mean", start_round_number=1, end_round_number=2, tag="rb"
        )
        assert [r["round_number"] for r in rows] == [1, 2]
        assert all(r["total_count"] == 4 for r in rows)


class TestShutdownDrain:
    """#277 problem 4: stop() must drain both lanes instead of silently dropping the queue."""

    def test_stop_drains_foreground_backlog(self, db, monkeypatch):
        # Slow, small-batch writer: most rows are still queue-resident when stop() is called
        # (the old stop() dropped exactly those rows)
        real_flush = db._flush_buffer

        def slow_flush(buffer=None):
            time.sleep(0.01)
            real_flush(buffer)

        monkeypatch.setattr(db, "_flush_buffer", slow_flush)
        db.write_buffer_max_size = 100
        n = 2000
        for i in range(n):
            db.write_training_stat(model_name="m", stat_name="s", value=float(i), tag="drain_fg")
        db.stop()
        assert len(db.query_training_stat(tag="drain_fg", columns=["value"])) == n

    def test_stop_drains_bulk_backlog(self, db):
        db.bulk_chunk_rows = 100
        for _ in range(5):
            db.write_injection_stats_bulk(_bulk_rows(100), tag="drain_bulk")
        db.stop()
        assert db.injection_backlog_rows() == 0
        assert len(db.query_injection_stat(tag="drain_bulk")) == 500


class TestTwoPassColumns:
    """#282: schema v5 — inference_results carries the two-pass scores."""

    def test_write_and_query_two_pass_scores(self, db):
        db.write_inference_result(
            npy_path="/x/a.npy",
            snippet_index=3,
            prediction=1,
            confidence=0.97,
            tag="tp_v1",
            screening_proba=0.91,
            mc_mean=0.97,
            mc_std=0.03,
        )
        assert db.flush()
        [row] = db.query_inference_result(
            tag="tp_v1", columns=["screening_proba", "mc_mean", "mc_std"]
        )
        assert row["screening_proba"] == pytest.approx(0.91)
        assert row["mc_mean"] == pytest.approx(0.97)
        assert row["mc_std"] == pytest.approx(0.03)

    def test_two_pass_columns_default_null(self, db):
        db.write_inference_result(
            npy_path="/x/b.npy", snippet_index=0, prediction=1, confidence=0.99, tag="tp_v2"
        )
        assert db.flush()
        [row] = db.query_inference_result(tag="tp_v2", columns=["mc_mean", "mc_std"])
        assert row["mc_mean"] is None and row["mc_std"] is None
