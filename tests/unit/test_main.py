"""Unit tests for main.py glue that isn't reachable through the higher-level commands: the
terminal training-status / exit-code contract (_report_final_training_status), non-retryable
streaming-inference failures, the stage-aware inference retry state machine (manifest-driven
skip, per-cadence failure containment, supersede-on-retry) with the GPU pipeline and
preprocessing stubbed out, and the end-of-run benchmark-report Slack hook
(_post_benchmark_report)."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import types

import numpy as np
import pytest

from aetherscan import main
from aetherscan.benchmark import stage_timer
from aetherscan.config import get_config
from aetherscan.db import get_db
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
        self.load_parallel_flags: list[bool] = []

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
            # Mirrors the real _process_cadence: a freshly written .npy is prunable (#302)
            freshly_extracted=True,
        )

    def load_inference_data(self, override_filepaths=None, parallel=True):
        # override_filepaths=None mirrors the legacy --test-files call shape, which loads
        # the configured test_files with no arguments
        if override_filepaths is None:
            override_filepaths = [get_config().data.test_files[0]]
        self.loaded_paths.extend(override_filepaths)
        self.load_parallel_flags.append(parallel)
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
        # Mirror the real pipeline's side effect: positives land in inference_results, so
        # supersede-on-retry tests gate on actual candidate rows (not an empty query)
        db = get_db()
        tag = get_config().checkpoint.save_tag
        for idx in np.nonzero(predictions)[0]:
            db.write_inference_result(npy_path, int(idx), 1, float(proba[idx]), tag=tag)
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
        # The streaming loader must run sequentially (parallel=False): the prefetch thread
        # already saturates the CPU with the persistent energy-detection pool.
        assert preprocessor.load_parallel_flags == [False, False]
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

    def test_prune_off_by_default_keeps_stamps(self, stubbed_streaming):
        """#399: the None default resolves OFF — a default-configured run retains every
        stamp .npy so the fingerprint-scoped cache serves re-scores out of the box."""
        db, make_preprocessor = stubbed_streaming
        assert get_config().inference.prune_stamps is None  # default
        preprocessor = make_preprocessor()
        _run_streaming_csv_inference(preprocessor, strategy=None)
        for unit in preprocessor.units:
            assert os.path.exists(unit.npy_path)

    def test_explicit_prune_deletes_npy_keeps_metadata_and_sidecar(self, stubbed_streaming):
        """#302 (+#399 default flip): with --prune-stamps, after a successful pass every
        freshly-extracted stamp .npy is gone, the metadata .json stays, each candidate's
        snippet is sidecarred, and the manifest-driven resume still skips everything on
        the next pass without touching the missing .npy."""
        from aetherscan.candidate_figures import candidate_sidecar_path  # noqa: PLC0415

        db, make_preprocessor = stubbed_streaming
        get_config().inference.prune_stamps = True
        preprocessor = make_preprocessor()

        _run_streaming_csv_inference(preprocessor, strategy=None)

        for unit in preprocessor.units:
            assert not os.path.exists(unit.npy_path)  # pruned
            assert os.path.exists(DataPreprocessor.cadence_metadata_path(unit.npy_path))
            # The stub scores exactly one candidate per cadence (proba > 0.9)
            sidecar = candidate_sidecar_path(unit.npy_path)
            assert os.path.exists(sidecar)
            with np.load(sidecar) as loaded:
                assert loaded["snippet_indices"].tolist() == [3]
                assert loaded["stamps"].shape == (1, 6, 16, 8)

        # Resume rides the DB row, never the pruned .npy
        second = make_preprocessor()
        totals = _run_streaming_csv_inference(second, strategy=None)
        assert totals["n_skipped"] == 2
        assert second.processed_keys == []

    def test_no_prune_flag_keeps_all_stamps(self, stubbed_streaming):
        db, make_preprocessor = stubbed_streaming
        get_config().inference.prune_stamps = False
        preprocessor = make_preprocessor()
        _run_streaming_csv_inference(preprocessor, strategy=None)
        for unit in preprocessor.units:
            assert os.path.exists(unit.npy_path)

    def test_explicit_output_dir_defaults_prune_off(self, stubbed_streaming, tmp_path):
        """An operator-curated --preprocess-output-dir with the default (off) pruning is
        never destroyed — pruning stays OFF unless --prune-stamps is passed explicitly."""
        db, make_preprocessor = stubbed_streaming
        get_config().inference.preprocess_output_dir = str(tmp_path)
        preprocessor = make_preprocessor()
        _run_streaming_csv_inference(preprocessor, strategy=None)
        for unit in preprocessor.units:
            assert os.path.exists(unit.npy_path)

    def test_handed_cache_never_pruned(self, stubbed_streaming, monkeypatch):
        """Only freshly-extracted stamps are prunable: a cadence resumed from a
        pre-existing .npy (freshly_extracted=False) keeps its stamps even with pruning
        ON."""
        db, make_preprocessor = stubbed_streaming
        get_config().inference.prune_stamps = True
        preprocessor = make_preprocessor()
        original = preprocessor.process_pending_cadence

        def resume_like(unit):
            result = original(unit)
            result.freshly_extracted = False
            return result

        monkeypatch.setattr(preprocessor, "process_pending_cadence", resume_like)
        _run_streaming_csv_inference(preprocessor, strategy=None)
        for unit in preprocessor.units:
            assert os.path.exists(unit.npy_path)

    def test_resolve_prune_stamps_matrix(self):
        """#399: None resolves OFF regardless of the cache dir; explicit flags win."""
        config = get_config()
        config.inference.prune_stamps = None
        config.inference.preprocess_output_dir = None
        assert main._resolve_prune_stamps(config) is False  # default: keep stamps
        config.inference.preprocess_output_dir = "/some/dir"
        assert main._resolve_prune_stamps(config) is False
        config.inference.prune_stamps = True
        assert main._resolve_prune_stamps(config) is True  # explicit ON beats explicit dir
        config.inference.prune_stamps = True
        config.inference.preprocess_output_dir = None
        assert main._resolve_prune_stamps(config) is True  # explicit ON, default dir
        config.inference.prune_stamps = False
        assert main._resolve_prune_stamps(config) is False  # explicit OFF

    def test_prune_failure_keeps_stamps_and_run_continues(self, stubbed_streaming, monkeypatch):
        """Pruning is best-effort: a sidecar-write failure must keep the stamps and never
        fail the cadence or the pass."""
        db, make_preprocessor = stubbed_streaming

        def boom(npy_path, snippet_indices):
            raise OSError("simulated sidecar write failure")

        monkeypatch.setattr(main, "write_candidate_snippet_sidecar", boom)
        preprocessor = make_preprocessor()
        totals = _run_streaming_csv_inference(preprocessor, strategy=None)
        assert totals["n_cadences"] == 2
        for unit in preprocessor.units:
            assert os.path.exists(unit.npy_path)  # kept on failure

    def test_prefetch_depth_2_completes_all_cadences(self, stubbed_streaming):
        """#298 N2 + #401: at depth 2 every cadence is inferred exactly once and the
        totals/manifest are complete — consumption order is completion order, so only
        the SET of inferred cadences is contractual, not their sequence."""
        db, make_preprocessor = stubbed_streaming
        get_config().inference.prefetch_depth = 2
        preprocessor = make_preprocessor(keys=[("A", "1"), ("B", "2"), ("C", "3"), ("D", "4")])

        totals = _run_streaming_csv_inference(preprocessor, strategy=None)

        assert totals["n_cadences"] == 4
        pipeline = _StubPipeline.instances[-1]
        assert sorted(pipeline.inferred_paths) == sorted(u.npy_path for u in preprocessor.units)
        assert db.flush(timeout=10) is True
        assert len(db.query_inference_cadences(tag="test_v1", status="inferred")) == 4

    def test_completion_order_consumes_fast_cadence_before_straggler(self, stubbed_streaming):
        """#401: with depth 2, a slow first cadence must NOT head-of-line-block the fast
        second one — the fast cadence is inferred while the straggler still preprocesses.
        (Under the old strict-catalog-order consumption this asserted the reverse.)"""
        import time as _time  # noqa: PLC0415

        db, make_preprocessor = stubbed_streaming
        get_config().inference.prefetch_depth = 2
        preprocessor = make_preprocessor(keys=[("SLOW", "1"), ("FAST", "2")])
        slow_path = preprocessor.units[0].npy_path
        original = preprocessor.process_pending_cadence

        def stalling(unit):
            if unit.npy_path == slow_path:
                _time.sleep(0.8)
            return original(unit)

        preprocessor.process_pending_cadence = stalling

        totals = _run_streaming_csv_inference(preprocessor, strategy=None)

        assert totals["n_cadences"] == 2
        pipeline = _StubPipeline.instances[-1]
        fast_path = preprocessor.units[1].npy_path
        assert pipeline.inferred_paths == [fast_path, slow_path]

    def test_prefetch_load_failure_falls_back_to_inference_thread(self, stubbed_streaming):
        """#298 I5-overlap: a prefetch-side load failure must not abort the pass one
        iteration later — the inference thread retries the load under its own per-cadence
        containment."""
        db, make_preprocessor = stubbed_streaming
        preprocessor = make_preprocessor()
        fail_once_path = preprocessor.units[0].npy_path
        original_load = preprocessor.load_inference_data
        failed: list[str] = []

        def flaky_load(override_filepaths=None, parallel=True):
            if override_filepaths == [fail_once_path] and fail_once_path not in failed:
                failed.append(fail_once_path)
                raise OSError("simulated transient read failure")
            return original_load(override_filepaths=override_filepaths, parallel=parallel)

        preprocessor.load_inference_data = flaky_load

        totals = _run_streaming_csv_inference(preprocessor, strategy=None)

        assert totals["n_cadences"] == 2
        assert failed == [fail_once_path]  # prefetch failed once, fallback succeeded
        assert db.flush(timeout=10) is True
        assert len(db.query_inference_cadences(tag="test_v1", status="inferred")) == 2

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

        # Exactly the fresh attempt's candidate row is live (the stub writes one positive
        # per cadence at confidence 0.95): no double-up, and it wasn't swept up by the
        # supersede that retired the stale row
        live = db.query_inference_result(tag="test_v1", npy_path=npy_path)
        assert len(live) == 1
        assert live[0]["superseded"] == 0
        assert live[0]["confidence"] == pytest.approx(0.95)
        everything = db.query_inference_result(
            tag="test_v1", npy_path=npy_path, include_superseded=True
        )
        stale = [r for r in everything if r["superseded"] == 1]
        assert len(stale) == 1
        assert stale[0]["confidence"] == 0.999

    def test_preprocessing_artifact_resume_skips_reprocessing(self, stubbed_streaming, tmp_path):
        """A cadence with a stamp .npy on disk resumes off it: the real
        DataPreprocessor.process_pending_cadence returns the stored stamp count without
        re-running energy detection. (Preprocessing-artifact resume only — the streaming
        loop's handoff of the resumed cadence to the inference stage is not exercised
        here.)"""
        db, make_preprocessor = stubbed_streaming
        stub = make_preprocessor(keys=[("A", "1")])
        unit = stub.units[0]
        stub.process_pending_cadence(unit)  # lay down .npy + metadata, no manifest row

        real = DataPreprocessor()
        result = real.process_pending_cadence(unit)
        assert result is not None
        assert result.n_hits == 4  # from the existing .npy, not a re-run

    def test_viz_collection_failure_never_fails_the_pass(self, stubbed_streaming, monkeypatch):
        """A collector bug must degrade the plots, not the science: cadences still complete
        (and resume) normally when record_processed / record_skipped raise."""
        db, make_preprocessor = stubbed_streaming
        get_config().inference.inference_viz_enabled = True

        class _ExplodingCollector:
            def __init__(self, *args, **kwargs):
                self.records: list = []

            def record_processed(self, *args, **kwargs):
                raise RuntimeError("simulated collector bug")

            def record_skipped(self, *args, **kwargs):
                raise RuntimeError("simulated collector bug")

        monkeypatch.setattr(main, "InferenceVizCollector", _ExplodingCollector)
        monkeypatch.setattr(main, "render_inference_visualizations", lambda *a, **k: None)

        totals = _run_streaming_csv_inference(make_preprocessor(), strategy=None)
        assert totals["n_cadences"] == 2  # record_processed raised for both; pass unaffected

        # Retry pass resumes both cadences off the manifest -> exercises the
        # record_skipped guard the same way
        totals = _run_streaming_csv_inference(make_preprocessor(), strategy=None)
        assert totals["n_skipped"] == 2


class TestLegacyTestFilesSupersede:
    def test_stale_rows_superseded_before_rerun(self, stubbed_streaming, monkeypatch):
        """The legacy --test-files path must retire a dead attempt's partial positives
        before the re-run's rows land — one live set for the npy_path, no duplicates
        (mirrors _infer_cadence's supersede-on-retry step on the streaming path)."""
        db, make_preprocessor = stubbed_streaming
        stub = make_preprocessor(keys=[("A", "1")])
        unit = stub.units[0]
        stub.process_pending_cadence(unit)  # lay down a real .npy for the no-arg load
        get_config().data.test_files = [unit.npy_path]

        # Simulate a dead attempt's partial write for this file under the same tag
        db.write_inference_result(unit.npy_path, 0, 1, 0.999, tag="test_v1")
        assert db.flush(timeout=10) is True

        def fake_run_inference_pipeline(cadence_data, npy_path, strategy):
            db.write_inference_result(npy_path, 1, 1, 0.7, tag="test_v1")
            n = int(cadence_data.shape[0])
            return {"n_cadence_snippets": n, "n_processed": n, "n_candidates": 1}

        monkeypatch.setattr(main, "run_inference_pipeline", fake_run_inference_pipeline)

        results = main._run_legacy_test_files_inference(stub, strategy=None)
        assert results["n_candidates"] == 1

        assert db.flush(timeout=10) is True
        live = db.query_inference_result(tag="test_v1", npy_path=unit.npy_path)
        assert len(live) == 1  # exactly the fresh attempt's row survives un-superseded
        assert live[0]["superseded"] == 0
        assert live[0]["confidence"] == pytest.approx(0.7)
        everything = db.query_inference_result(
            tag="test_v1", npy_path=unit.npy_path, include_superseded=True
        )
        stale = [r for r in everything if r["superseded"] == 1]
        assert [r["confidence"] for r in stale] == [0.999]

    def test_all_cadences_no_stamps_is_non_retryable(self, stubbed_streaming, monkeypatch):
        db, make_preprocessor = stubbed_streaming
        preprocessor = make_preprocessor()
        monkeypatch.setattr(_StubPreprocessor, "process_pending_cadence", lambda self, unit: None)
        with pytest.raises(NonRetryableInferenceError, match="No cadence results"):
            _run_streaming_csv_inference(preprocessor, strategy=None)


class TestPostBenchmarkReport:
    """End-of-run benchmark-report hook: flush -> render PNG -> Slack upload, fully guarded."""

    @pytest.fixture
    def slack_upload(self, monkeypatch):
        """Stub main.get_logger with a recording uploader; returns the recorded calls."""
        calls = []

        def upload(png_path, title=None, message=None, **kwargs):
            calls.append({"png_path": png_path, "title": title, "message": message})
            return True

        fake_logger = types.SimpleNamespace(upload_image_to_slack=upload)
        monkeypatch.setattr(main, "get_logger", lambda: fake_logger)
        return calls

    def _png_path(self, tag):
        return os.path.join(get_config().output_path, "plots", f"benchmark_report_{tag}.png")

    def test_renders_png_and_uploads(self, initialized_runtime, slack_upload):
        # The span is still sitting in the async DB write queue when the hook runs — the
        # hook's own db.flush() must drain it before the report tool reads the DB
        with stage_timer("train.load_backgrounds", tag="test_v1"):
            pass

        main._post_benchmark_report("test_v1")

        png_path = self._png_path("test_v1")
        assert os.path.exists(png_path)
        assert len(slack_upload) == 1
        assert slack_upload[0]["png_path"] == png_path
        assert "test_v1" in slack_upload[0]["title"]

    def test_disabled_by_config_skips_everything(self, initialized_runtime, slack_upload):
        get_config().monitor.benchmark_report_enabled = False
        with stage_timer("train.load_backgrounds", tag="test_v1"):
            pass

        main._post_benchmark_report("test_v1")

        assert slack_upload == []
        assert not os.path.exists(self._png_path("test_v1"))

    def test_no_stage_rows_skips_upload(self, initialized_runtime, slack_upload):
        main._post_benchmark_report("test_v1")
        assert slack_upload == []
        assert not os.path.exists(self._png_path("test_v1"))

    def test_exception_inside_hook_is_swallowed(self, initialized_runtime, monkeypatch):
        # Any blow-up inside the hook (here: resolving the Slack logger) must never
        # escape and fail an otherwise-finished run
        def boom():
            raise RuntimeError("boom")

        monkeypatch.setattr(main, "get_logger", boom)
        with stage_timer("train.load_backgrounds", tag="test_v1"):
            pass
        main._post_benchmark_report("test_v1")  # must not raise

    def test_system_exit_from_report_tool_is_swallowed(self, monkeypatch, tmp_path):
        # A DB file predating the benchmarking schema (no pipeline_stages table) makes the
        # report tool's load_rows raise SystemExit — which is NOT an Exception subclass,
        # so the hook must swallow it explicitly rather than kill the run
        legacy_db = tmp_path / "legacy.db"
        sqlite3.connect(str(legacy_db)).close()
        fake_db = types.SimpleNamespace(db_path=str(legacy_db), flush=lambda timeout=None: True)
        monkeypatch.setattr(main, "get_db", lambda: fake_db)
        main._post_benchmark_report("test_v1")  # must not raise

    def test_missing_report_script_skips_without_raising(
        self, initialized_runtime, slack_upload, monkeypatch
    ):
        # A pip-installed package without the repo checkout alongside has no utils/
        # directory next to src/aetherscan — the report_path.exists() guard must skip
        # gracefully (warn + return) rather than blow up on the missing file.
        monkeypatch.setattr(main.Path, "exists", lambda self: False)
        with stage_timer("train.load_backgrounds", tag="test_v1"):
            pass
        main._post_benchmark_report("test_v1")  # must not raise
        assert slack_upload == []


class TestPrefetchRamPreflight:
    """#408: catalog-derived RAM preflight — warn, never clamp. The estimate keys on the
    bands actually present in the pending catalog, so an X-band-only catalog stays quiet
    where a C-band one warns on the same small host."""

    def _units(self, bands, group_by_cols=None):
        cols = group_by_cols or ["Target", "Session", "Band", "Cadence ID", "Frequency"]
        band_slot = next((i for i, c in enumerate(cols) if c.strip().lower() == "band"), None)
        units = []
        for i, band in enumerate(bands):
            key = [f"T{i}", "S1", "?", str(i), "1400"][: len(cols)]
            if band_slot is not None:
                key[band_slot] = band
            group = CadenceGroup(
                key=tuple(key),
                h5_paths=[f"/x/{i}_{j}.h5" for j in range(6)],
                csv_path="catalog.csv",
                expected_obs=6,
                is_valid=True,
            )
            units.append(PendingCadence(group=group, npy_path=f"/x/{i}.npy", index=i + 1))
        return units, cols

    def test_c_band_catalog_warns_on_small_host_with_suggestion(self):
        units, cols = self._units(["C", "X"])
        # depth 4 -> 5 in-flight x 65 GB = 325 GB > 90% of 288 GB
        message = main._prefetch_ram_preflight(units, cols, depth=4, total_ram_gb=288.0)
        assert message is not None
        assert "325 GB" in message
        assert "driven by band C" in message
        # 0.9 * 288 // 65 = 3 -> suggested depth 2 (3 in-flight fit the budget)
        assert "--prefetch-depth 2" in message

    def test_x_band_only_catalog_quiet_on_same_host(self):
        units, cols = self._units(["X", "X", "X"])
        # 5 in-flight x 8 GB = 40 GB, far under the budget
        assert main._prefetch_ram_preflight(units, cols, depth=4, total_ram_gb=288.0) is None

    def test_lowercase_band_column_matches(self):
        # Group-by column names are user-supplied CSV headers; the lookup must be
        # case-insensitive like derive_cadence_provenance (review note) — a catalog
        # headed 'band' must NOT fall to the conservative unknown-band branch
        units, cols = self._units(
            ["X"], group_by_cols=["Target", "Session", "band", "Cadence ID", "Frequency"]
        )
        assert main._prefetch_ram_preflight(units, cols, depth=4, total_ram_gb=288.0) is None

    def test_large_host_quiet_even_for_c_band(self):
        units, cols = self._units(["C"])
        # blpc3-scale: 325 GB < 0.9 * 503 GB
        assert main._prefetch_ram_preflight(units, cols, depth=4, total_ram_gb=503.0) is None

    def test_unknown_band_and_missing_band_column_are_conservative(self):
        units, cols = self._units(["Q"])  # not in the table -> C-band worst case
        assert main._prefetch_ram_preflight(units, cols, depth=4, total_ram_gb=288.0) is not None
        # No band group-by column at all -> conservative too
        units2, no_band_cols = self._units(["X"], group_by_cols=["Target", "Session", "Cadence ID"])
        assert (
            main._prefetch_ram_preflight(units2, no_band_cols, depth=4, total_ram_gb=288.0)
            is not None
        )

    def test_empty_pending_and_degenerate_inputs_return_none(self):
        units, cols = self._units(["C"])
        assert main._prefetch_ram_preflight([], cols, depth=4, total_ram_gb=288.0) is None
        assert main._prefetch_ram_preflight(units, cols, depth=0, total_ram_gb=288.0) is None
        assert main._prefetch_ram_preflight(units, cols, depth=4, total_ram_gb=0.0) is None

    def test_no_step_down_edge_says_so_instead_of_suggesting_current_depth(self):
        units, cols = self._units(["C"])
        # 65 GB host: even depth 1 (2 in-flight = 130 GB) exceeds the budget; the old
        # floor would have "suggested" the depth the user is already at (review note) —
        # the honest message is that no depth fits
        message = main._prefetch_ram_preflight(units, cols, depth=1, total_ram_gb=65.0)
        assert message is not None
        assert "no --prefetch-depth fits" in message
        assert "consider --prefetch-depth" not in message

    def test_nothing_fits_at_depth_ge_2_does_not_suggest_a_rejected_depth(self):
        # Second-pass review catch: on a 100 GB host at depth 4 the budget holds exactly
        # one in-flight cadence (90 // 65 = 1, raw candidate 0) — the pre-fix clamp lifted
        # 0 to 1 and recommended a depth whose own worst case (2 x 65 = 130 GB) the
        # arithmetic rejects. The unclamped test must fall through to the honest branch.
        units, cols = self._units(["C"])
        message = main._prefetch_ram_preflight(units, cols, depth=4, total_ram_gb=100.0)
        assert message is not None
        assert "no --prefetch-depth fits" in message
        assert "consider --prefetch-depth" not in message

    def test_unknown_band_message_names_the_assumption_not_a_question_mark(self):
        # Catalogs grouped without a band column must not render "driven by band ? of ?"
        units, no_band_cols = self._units(["X"], group_by_cols=["Target", "Session", "Cadence ID"])
        message = main._prefetch_ram_preflight(units, no_band_cols, depth=4, total_ram_gb=288.0)
        assert message is not None
        assert "unknown-band worst case" in message
        assert "band ? of" not in message

    def test_call_site_wiring_logs_warning_through_streaming_loop(
        self, stubbed_streaming, monkeypatch, caplog
    ):
        """The whole preflight call is exception-guarded, so a wiring regression (argument
        order, units) would silently degrade to an INFO line — pin the happy path end to
        end: a small-host run through _run_streaming_csv_inference must emit the WARNING
        (the stub cadences' 2-tuple keys have no band slot -> conservative branch)."""
        import psutil  # noqa: PLC0415

        db, make_preprocessor = stubbed_streaming
        fake = types.SimpleNamespace(total=int(100e9))  # 100 GB host, C worst case
        monkeypatch.setattr(psutil, "virtual_memory", lambda: fake)
        preprocessor = make_preprocessor()
        with caplog.at_level(logging.WARNING, logger="aetherscan.main"):
            _run_streaming_csv_inference(preprocessor, strategy=None)
        assert any("RAM preflight" in record.message for record in caplog.records)
