"""Unit tests for main.py glue that isn't reachable through the higher-level commands: the
terminal training-status / exit-code contract (_report_final_training_status), non-retryable
streaming-inference failures, and the stage-aware inference retry state machine (manifest-driven
skip, per-cadence failure containment, supersede-on-retry) with the GPU pipeline and
preprocessing stubbed out."""

from __future__ import annotations

import json
import logging
import types

import numpy as np
import pytest

from aetherscan import main
from aetherscan.config import get_config
from aetherscan.main import NonRetryableInferenceError, _run_streaming_csv_inference
from aetherscan.preprocessing import CadenceGroup, CadenceResult, DataPreprocessor, PendingCadence
from aetherscan.run_state import STAGE_RF_PLOTS, STAGE_VAE_PLOTS, TrainingRunState


def _pipeline_with(stages_failed):
    state = TrainingRunState(tag="t", run_start_time=1.0, stages_failed=list(stages_failed))
    return types.SimpleNamespace(run_state=state)


class TestReportFinalTrainingStatus:
    def test_success_when_no_failed_stages(self):
        # No permanently-failed stage -> returns normally, no SystemExit.
        main._report_final_training_status(_pipeline_with([]))

    @pytest.mark.parametrize(
        "failed", [[STAGE_VAE_PLOTS], [STAGE_RF_PLOTS], [STAGE_VAE_PLOTS, STAGE_RF_PLOTS]]
    )
    def test_exits_nonzero_on_failed_plot_stage(self, failed):
        with pytest.raises(SystemExit) as exc:
            main._report_final_training_status(_pipeline_with(failed))
        assert exc.value.code == 1

    def test_exits_nonzero_when_pipeline_is_none(self):
        # Degenerate no-pipeline path must never report a false success.
        with pytest.raises(SystemExit) as exc:
            main._report_final_training_status(None)
        assert exc.value.code == 1

    def test_rf_skip_annotates_success_instead_of_unqualified(self, caplog):
        # A run whose RF stage was skipped (pre-loaded already-trained RF, issue #142) still
        # exits 0, but the terminal message must be the skip warning, not plain success.
        pipeline = _pipeline_with([])
        pipeline.rf_training_skipped_from_tag = "test_v27"
        with caplog.at_level(logging.INFO, logger="aetherscan.main"):
            main._report_final_training_status(pipeline)  # no SystemExit
        assert any("SKIPPED" in r.message and "test_v27" in r.message for r in caplog.records)
        assert not any("completed successfully" in r.message for r in caplog.records)


@pytest.fixture
def initialized_runtime():
    """DataPreprocessor needs live db + manager singletons; conftest tears them down.
    Returns the Database so tests can flush/query the run manifest."""
    from aetherscan.db import init_db  # noqa: PLC0415
    from aetherscan.manager import init_manager  # noqa: PLC0415

    init_manager()
    return init_db()


class TestStreamingInferenceNonRetryable:
    def test_empty_catalog_raises_non_retryable(self, initialized_runtime):
        # No inference_files configured -> plan_cadences yields no units. This is a
        # permanent (config) failure: the retry loop in inference_command re-raises
        # NonRetryableInferenceError immediately instead of burning retry attempts.
        # The raise happens before any model loading, so no strategy is needed.
        preprocessor = DataPreprocessor()
        with pytest.raises(NonRetryableInferenceError, match="No cadence work units"):
            _run_streaming_csv_inference(preprocessor, strategy=None)

    def test_non_retryable_error_is_an_exception_subclass(self):
        # Sanity: it must be catchable as a plain Exception (cleanup paths) while being
        # distinguishable from transient failures by the retry loop.
        assert issubclass(NonRetryableInferenceError, RuntimeError)


class _StubPreprocessor:
    """DataPreprocessor stand-in: fixed work units, canned per-cadence stamp arrays, no
    pools. Writes real .npy/.json artifacts so the resume/viz plumbing sees real files."""

    def __init__(self, tmp_path, keys, n_stamps=4, width=8):
        self.tmp_path = tmp_path
        self.n_stamps = n_stamps
        self.width = width
        self.units = [self._make_unit(key) for key in keys]
        self.processed_keys: list[tuple] = []
        self.loaded_paths: list[str] = []

    def _make_unit(self, key):
        group = CadenceGroup(
            key=key,
            h5_paths=[f"/data/{key[0]}_{i}.h5" for i in range(6)],
            csv_path="catalog.csv",
            expected_obs=6,
            is_valid=True,
        )
        return PendingCadence(group=group, npy_path=str(self.tmp_path / f"{key[0]}.npy"))

    def plan_cadences(self):
        return list(self.units)

    def start_energy_detection_pool(self):
        pass

    def stop_energy_detection_pool(self):
        pass

    def process_pending_cadence(self, unit):
        self.processed_keys.append(unit.group.key)
        rng = np.random.default_rng(3)
        stamps = rng.random((self.n_stamps, 6, 16, self.width)).astype(np.float32)
        np.save(unit.npy_path, stamps)
        metadata_path = DataPreprocessor.cadence_metadata_path(unit.npy_path)
        with open(metadata_path, "w") as f:
            json.dump({"h5_paths": unit.group.h5_paths, "key": list(unit.group.key)}, f)
        return CadenceResult(
            npy_path=unit.npy_path,
            h5_paths=unit.group.h5_paths,
            key=unit.group.key,
            n_hits=self.n_stamps,
            metadata_path=metadata_path,
        )

    def load_inference_data(self, override_filepaths):
        self.loaded_paths.extend(override_filepaths)
        return np.load(override_filepaths[0])


class _StubPipeline:
    """InferencePipeline stand-in recording which cadences reached the inference stage.
    Raises for any npy_path in fail_paths (simulating a mid-cadence death)."""

    instances: list = []

    def __init__(self, strategy=None):
        self.strategy = strategy
        self.inferred_paths: list[str] = []
        self.fail_paths: set[str] = set(type(self)._fail_paths)
        type(self).instances.append(self)

    _fail_paths: set = set()

    def run_inference(self, data, npy_path, **provenance):
        if npy_path in self.fail_paths:
            raise RuntimeError("simulated mid-cadence death")
        self.inferred_paths.append(npy_path)
        n = data.shape[0]
        proba = np.linspace(0.05, 0.95, n)
        predictions = (proba > 0.9).astype(int)
        return {
            "n_cadence_snippets": n,
            "n_processed": n,
            "n_candidates": int(predictions.sum()),
            "proba_true": proba,
            "predictions": predictions,
            "latents": np.zeros((n * 6, 8), dtype=np.float32),
        }


@pytest.fixture
def stubbed_streaming(tmp_path, initialized_runtime, monkeypatch):
    """Wire the streaming loop to the stubs; viz disabled (smoke-tested separately in
    test_inference_viz.py). Returns (db, make_preprocessor)."""
    db = initialized_runtime
    config = get_config()
    config.inference.inference_viz_enabled = False
    config.checkpoint.save_tag = "test_v1"
    monkeypatch.setattr(main, "InferencePipeline", _StubPipeline)
    _StubPipeline.instances = []
    _StubPipeline._fail_paths = set()

    def make_preprocessor(keys=(("A", "1"), ("B", "2"))):
        return _StubPreprocessor(tmp_path, list(keys))

    return db, make_preprocessor


class TestStreamingResumeStateMachine:
    def test_fresh_run_infers_all_and_writes_manifest(self, stubbed_streaming):
        db, make_preprocessor = stubbed_streaming
        preprocessor = make_preprocessor()

        totals = _run_streaming_csv_inference(preprocessor, strategy=None)

        assert totals["n_cadences"] == 2
        assert totals["n_skipped"] == 0
        assert totals["n_cadence_snippets"] == 8
        assert db.flush(timeout=10) is True
        rows = db.query_inference_cadences(tag="test_v1", status="inferred")
        assert len(rows) == 2
        assert all(r["n_stamps"] == 4 for r in rows)
        assert all(json.loads(r["confidence_summary"])["n"] == 4 for r in rows)

    def test_inferred_cadence_skipped_on_retry(self, stubbed_streaming):
        """A live 'inferred' manifest row short-circuits the whole cadence: neither
        preprocessing nor inference runs for it, and its stored aggregates fold into the
        totals."""
        db, make_preprocessor = stubbed_streaming
        first = make_preprocessor()
        _run_streaming_csv_inference(first, strategy=None)

        second = make_preprocessor()
        totals = _run_streaming_csv_inference(second, strategy=None)

        assert totals["n_skipped"] == 2
        assert totals["n_cadences"] == 2
        assert totals["n_cadence_snippets"] == 8  # from manifest aggregates
        assert second.processed_keys == []  # preprocessing never ran
        # No second pipeline was even constructed (nothing pending -> no model load)
        assert len(_StubPipeline.instances) == 1

    def test_changed_config_reinfers_instead_of_reusing_stale(self, stubbed_streaming):
        """A reused save-tag with a CHANGED inference config must NOT skip already-inferred
        cadences: the manifest's config_fingerprint mismatches, so they are re-inferred rather
        than silently serving stale results (guards the #122-class sticky-manifest footgun on
        the inference side)."""
        db, make_preprocessor = stubbed_streaming
        config = get_config()

        first = make_preprocessor()
        _run_streaming_csv_inference(first, strategy=None)
        assert db.flush(timeout=10) is True
        fp_before = {
            r["config_fingerprint"]
            for r in db.query_inference_cadences(tag="test_v1", status="inferred")
        }

        # Change a result-affecting inference knob under the SAME save-tag.
        config.inference.classification_threshold = 0.123

        second = make_preprocessor()
        totals = _run_streaming_csv_inference(second, strategy=None)

        assert totals["n_skipped"] == 0  # live 'inferred' rows are NOT reused under the new config
        assert totals["n_cadences"] == 2
        assert second.processed_keys == [("A", "1"), ("B", "2")]  # both re-preprocessed
        assert len(_StubPipeline.instances) == 2  # a second pipeline was built (re-inference ran)

        assert db.flush(timeout=10) is True
        live = db.query_inference_cadences(tag="test_v1", status="inferred")
        assert len(live) == 2  # stale rows superseded, fresh rows live
        fp_after = {r["config_fingerprint"] for r in live}
        assert fp_after.isdisjoint(fp_before)  # fresh rows carry the new config fingerprint

    def test_failed_cadence_recorded_and_retried_alone(self, stubbed_streaming):
        """Inference-stage containment: cadence B's failure doesn't abort cadence A; the
        pass raises so the retry loop re-attempts, and the retry re-runs ONLY B."""
        db, make_preprocessor = stubbed_streaming
        first = make_preprocessor()
        fail_path = first.units[1].npy_path
        _StubPipeline._fail_paths = {fail_path}

        with pytest.raises(RuntimeError, match="failed for 1 cadence"):
            _run_streaming_csv_inference(first, strategy=None)

        assert db.flush(timeout=10) is True
        assert [
            r["npy_path"] for r in db.query_inference_cadences(tag="test_v1", status="inferred")
        ] == [first.units[0].npy_path]
        assert [
            r["npy_path"] for r in db.query_inference_cadences(tag="test_v1", status="failed")
        ] == [fail_path]

        # Retry pass: A skipped via manifest, B re-attempted and now succeeds
        _StubPipeline._fail_paths = set()
        second = make_preprocessor()
        totals = _run_streaming_csv_inference(second, strategy=None)

        assert totals["n_skipped"] == 1
        assert totals["n_cadences"] == 2
        retry_pipeline = _StubPipeline.instances[-1]
        assert retry_pipeline.inferred_paths == [fail_path]
        # B's 'failed' row was superseded by the fresh 'inferred' row
        assert db.flush(timeout=10) is True
        b_rows = db.query_inference_cadences(tag="test_v1", npy_path=fail_path)
        assert [r["status"] for r in b_rows] == ["inferred"]

    def test_stale_inference_results_superseded_on_retry(self, stubbed_streaming):
        """Partial positives written by a dead attempt must be flagged before the re-run's
        rows land, so candidates can't double up."""
        db, make_preprocessor = stubbed_streaming
        preprocessor = make_preprocessor(keys=[("A", "1")])
        npy_path = preprocessor.units[0].npy_path

        # Simulate a dead attempt's partial write for this cadence under the same tag
        db.write_inference_result(npy_path, 0, 1, 0.999, tag="test_v1")
        assert db.flush(timeout=10) is True

        _run_streaming_csv_inference(preprocessor, strategy=None)
        assert db.flush(timeout=10) is True

        live = db.query_inference_result(tag="test_v1", npy_path=npy_path)
        assert all(r["superseded"] == 0 for r in live)
        everything = db.query_inference_result(
            tag="test_v1", npy_path=npy_path, include_superseded=True
        )
        stale = [r for r in everything if r["superseded"] == 1]
        assert len(stale) == 1
        assert stale[0]["confidence"] == 0.999

    def test_preprocessing_artifacts_skip_to_inference(self, stubbed_streaming, tmp_path):
        """A cadence with a stamp .npy but no 'inferred' manifest row (killed between
        stages) must skip preprocessing and go straight to inference — via the real
        DataPreprocessor.process_pending_cadence resume path."""
        db, make_preprocessor = stubbed_streaming
        stub = make_preprocessor(keys=[("A", "1")])
        unit = stub.units[0]
        stub.process_pending_cadence(unit)  # lay down .npy + metadata, no manifest row

        real = DataPreprocessor()
        result = real.process_pending_cadence(unit)
        assert result is not None
        assert result.n_hits == 4  # from the existing .npy, not a re-run

    def test_all_cadences_no_stamps_is_non_retryable(self, stubbed_streaming, monkeypatch):
        db, make_preprocessor = stubbed_streaming
        preprocessor = make_preprocessor()
        monkeypatch.setattr(_StubPreprocessor, "process_pending_cadence", lambda self, unit: None)
        with pytest.raises(NonRetryableInferenceError, match="No cadence results"):
            _run_streaming_csv_inference(preprocessor, strategy=None)
