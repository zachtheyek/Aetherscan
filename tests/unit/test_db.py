"""Unit tests for aetherscan.db: writer thread lifecycle, flush sentinel protocol, executemany
batching across tables, is_finite sanitization, and query filters / column whitelists — all
against a tmp-path SQLite file."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from aetherscan.config import get_config
from aetherscan.db.db import Database


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
