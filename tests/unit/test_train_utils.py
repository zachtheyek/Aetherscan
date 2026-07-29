# NOTE: come back to this later

"""Unit tests for aetherscan.train pure-logic helpers: checkpoint tag resolution, curriculum
schedules, directory archiving, encoder-trained heuristics, the val-AUC quality floor, SHAP
output normalization, the rf_train dataset producer-await/fallback composition
(_obtain_rf_dataset), the rf_plots overlap coordinator (plot_rf_diagnostics), and the training
stage machine (skip-if-done / record-failure semantics against a stub pipeline)."""

from __future__ import annotations

import logging
import math
import os
import threading
import types

import numpy as np
import pytest

from aetherscan.config import get_config
from aetherscan.round_data import RoundDataPaths
from aetherscan.run_state import (
    STAGE_FINAL_SAVE,
    STAGE_HF_UPLOAD,
    STAGE_RF_PLOTS,
    STAGE_RF_TRAIN,
    STAGE_VAE_PLOTS,
    STAGE_VAE_ROUNDS,
    TRAINING_STAGES,
    TrainingRunState,
    config_fingerprint,
    run_state_path,
    save_run_state,
)
from aetherscan.train import (
    TrainingPipeline,
    _execute_training_stages,
    _resolve_load_tag,
    archive_directory,
    build_epoch_history,
    check_encoder_trained,
    check_posterior_collapse,
    check_screening_threshold,
    check_val_auc_floor,
    compute_expected_std,
)


def _touch_pair(checkpoints_dir, tag):
    """Create a matching encoder/decoder checkpoint pair for `tag`."""
    os.makedirs(checkpoints_dir, exist_ok=True)
    for prefix in ("vae_encoder", "vae_decoder"):
        with open(os.path.join(checkpoints_dir, f"{prefix}_{tag}.keras"), "w") as f:
            f.write("stub")


class TestResolveLoadTag:
    """The load_models() tag decision (issue #142): an explicit missing tag must fail loud
    instead of silently substituting the latest tag present in base_dir."""

    def test_explicit_existing_tag_returned(self, tmp_path):
        _touch_pair(tmp_path, "round_03")
        assert _resolve_load_tag(str(tmp_path), "round_03") == "round_03"

    def test_explicit_missing_tag_raises_instead_of_falling_back(self, tmp_path):
        # The old fallback would have loaded the stale test_v27 model and reported success.
        _touch_pair(tmp_path, "test_v27")
        with pytest.raises(FileNotFoundError, match="round_01"):
            _resolve_load_tag(str(tmp_path), "round_01")

    def test_missing_round_tag_message_hints_checkpoints(self, tmp_path):
        # The per-round-checkpoint hint is relevant only for a round_XX tag.
        _touch_pair(tmp_path, "test_v27")
        with pytest.raises(FileNotFoundError, match="--load-dir checkpoints"):
            _resolve_load_tag(str(tmp_path), "round_01")

    def test_missing_non_round_tag_message_omits_checkpoints_hint(self, tmp_path):
        # For a non-round explicit tag (e.g. a typo'd full run tag) the checkpoints hint is a
        # red herring and must not appear.
        _touch_pair(tmp_path, "train_20260101_120000")
        with pytest.raises(FileNotFoundError) as excinfo:
            _resolve_load_tag(str(tmp_path), "train_20260101_130000")
        assert "--load-dir checkpoints" not in str(excinfo.value)

    def test_default_prefers_final(self, tmp_path):
        _touch_pair(tmp_path, "final")
        _touch_pair(tmp_path, "round_09")
        assert _resolve_load_tag(str(tmp_path), None) == "final"

    def test_default_no_final_raises_loudly(self, tmp_path):
        # tag=None loads only the conventional "final" model — it never scans for the "latest"
        # tag present (which could be a stale, unrelated run's model). No "final" → fail loudly.
        _touch_pair(tmp_path, "round_02")
        _touch_pair(tmp_path, "train_20260101_120000")
        with pytest.raises(FileNotFoundError, match="final"):
            _resolve_load_tag(str(tmp_path), None)

    def test_default_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _resolve_load_tag(str(tmp_path), None)


class TestTrainingPlotsDir:
    """_training_plots_dir centralizes this run's plots base:
    {output_path}/plots/training/{save_tag}[/subdir]."""

    def _pipeline(self, tag):
        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline.config = get_config()
        pipeline.config.checkpoint.save_tag = tag
        return pipeline

    def test_base_dir(self):
        pipeline = self._pipeline("run_a")
        assert pipeline._training_plots_dir() == os.path.join(
            pipeline.config.output_path, "plots", "training", "run_a"
        )

    def test_subdir(self):
        pipeline = self._pipeline("run_a")
        assert pipeline._training_plots_dir("checkpoints") == os.path.join(
            pipeline.config.output_path, "plots", "training", "run_a", "checkpoints"
        )


class TestResumeInPlace:
    """Resume-in-place (--load-tag {full} == save_tag) is a continuation of the same run, not a
    user override: _init_run_state must let the manifest's completed_rounds drive _start_round (so
    the round-checkpoint resume fallback is reachable) instead of resetting it to 1 and wiping the
    manifest."""

    def test_manifest_rounds_drive_start_round(self, tmp_path):
        config = get_config()
        config.output_path = str(tmp_path)
        tag = "train_20260101_120000"
        config.checkpoint.save_tag = tag
        config.checkpoint.load_tag = tag  # resume-in-place: the full load-tag was adopted
        config.checkpoint.load_dir = None
        config.checkpoint.start_round = 1  # the default — must NOT clobber the manifest

        # Manifest for THIS tag: rounds 1-3 done, VAE unfinished; fingerprint matches the config.
        state = TrainingRunState(
            tag=tag,
            run_start_time=123.0,
            config_fingerprint=config_fingerprint(config.to_dict()),
            completed_rounds=[1, 2, 3],
        )
        save_run_state(state, run_state_path(str(tmp_path), tag))

        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline.config = config
        pipeline.db = types.SimpleNamespace(mark_superseded=lambda *a, **k: True)
        pipeline._init_run_state()

        # Manifest drives resume: start at round 4 (max completed + 1); rounds are NOT wiped.
        assert pipeline._start_round == 4
        assert pipeline.run_state.completed_rounds == [1, 2, 3]


class TestResumeLoadPlan:
    """__init__'s checkpoint-load decision, factored into TrainingPipeline._resume_load_plan so
    it is testable without building the TF graph. Composes with TestResumeInPlace: _init_run_state
    derives _start_round from the manifest, then _resume_load_plan turns it into the actual
    (tag, dir) to load — resume-in-place with no final weights must fall back to the last
    completed round, or fail loudly."""

    def _pipeline(self, tmp_path, *, load_tag, save_tag, start_round, final_on_disk):
        config = get_config()
        config.model_path = str(tmp_path)
        config.checkpoint.load_tag = load_tag
        config.checkpoint.load_dir = None
        config.checkpoint.save_tag = save_tag
        if final_on_disk:
            _touch_pair(str(tmp_path), save_tag)  # final weights at the model root
        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline.config = config
        pipeline._start_round = start_round
        return pipeline

    def test_resume_in_place_no_final_falls_back_to_last_round(self, tmp_path):
        # The whole point of the refactor: no final weights yet → load round_{start_round-1}.
        tag = "train_20260101_120000"
        pipeline = self._pipeline(
            tmp_path, load_tag=tag, save_tag=tag, start_round=4, final_on_disk=False
        )
        assert pipeline._resume_load_plan() == ("round_03", "checkpoints")

    def test_resume_in_place_no_final_no_rounds_raises(self, tmp_path):
        # No final weights and no completed rounds → fail loudly, never load another run's model.
        tag = "train_20260101_120000"
        pipeline = self._pipeline(
            tmp_path, load_tag=tag, save_tag=tag, start_round=1, final_on_disk=False
        )
        with pytest.raises(FileNotFoundError, match="Cannot resume run"):
            pipeline._resume_load_plan()

    def test_resume_in_place_with_final_loads_final(self, tmp_path):
        # Final weights present (VAE finished, a later stage died) → load them from the model
        # root (dir=None), not a per-round checkpoint.
        tag = "train_20260101_120000"
        pipeline = self._pipeline(
            tmp_path, load_tag=tag, save_tag=tag, start_round=21, final_on_disk=True
        )
        assert pipeline._resume_load_plan() == (tag, None)

    def test_fresh_run_returns_none(self, tmp_path):
        # No --load-tag, start_round 1 → nothing to load.
        pipeline = self._pipeline(
            tmp_path,
            load_tag=None,
            save_tag="train_20260101_120000",
            start_round=1,
            final_on_disk=False,
        )
        assert pipeline._resume_load_plan() is None


class TestTrainRandomForestSkipIsLoud:
    def test_pretrained_rf_skip_warns_and_records_source_tag(self, caplog):
        """The is_trained early-return (issue #142) must warn loudly, name the tag the stale
        RF was loaded from, and set the marker main.py uses to qualify the terminal status."""
        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline.config = get_config()
        pipeline._resumed = False
        pipeline.rf_model = types.SimpleNamespace(is_trained=True)
        pipeline._rf_loaded_from_tag = "test_v27"
        pipeline.rf_training_skipped_from_tag = None
        # The skip path winds down a pending producer RF pre-generation (moot once the
        # stage is skipped) — none exists here, the shutdown must be a clean no-op
        pipeline._round_producer = None
        pipeline._rf_producer_request = None

        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            pipeline.train_random_forest()

        assert pipeline.rf_training_skipped_from_tag == "test_v27"
        assert any(
            "RF training SKIPPED" in r.message and "test_v27" in r.message for r in caplog.records
        )


class _FakeRfProducer:
    """await_round/shutdown recorder standing in for RoundDataProducer."""

    def __init__(self, log, error=None):
        self._log = log
        self._error = error
        self.shutdown_calls = 0

    def await_round(self, round_idx):
        self._log.append(("await", round_idx))
        if self._error is not None:
            raise self._error
        return {"n_samples": 8}

    def shutdown(self):
        self.shutdown_calls += 1
        self._log.append(("shutdown",))


class _FakeRfDataGenerator:
    """generate_round recorder standing in for DataGenerator on the in-process path."""

    def __init__(self, log):
        self._log = log

    def generate_round(self, paths, n_samples, snr_base, snr_range, round_num=None):
        self._log.append(("generate", paths.round_dir, n_samples, snr_base, snr_range, round_num))


class TestObtainRfDataset:
    """_obtain_rf_dataset composes producer await -> shutdown -> manifest reuse ->
    in-process fallback. Every resume/fallback path is pinned: producer-success,
    producer-error (falls back in-process), no-producer fresh generation, valid-dir
    reuse, and the defensive request-less-producer wind-down."""

    def _pipeline(self, log, producer=None, request=None):
        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline.config = get_config()
        pipeline.config.training.num_training_rounds = 20
        pipeline._round_producer = producer
        pipeline._rf_producer_request = request
        pipeline.data_generator = _FakeRfDataGenerator(log)
        return pipeline

    def _patch_validate(self, monkeypatch, log, manifest):
        def _validate(paths, expected_n_samples=None, expected_array_dtype=None):
            log.append(("validate", paths.round_dir, expected_n_samples, expected_array_dtype))
            return manifest

        monkeypatch.setattr("aetherscan.train.validate_done_manifest", _validate)

    def _rf_paths(self, tmp_path):
        return RoundDataPaths(round_dir=os.path.join(str(tmp_path), "rf"), round_idx=0)

    def test_producer_success_reuses_result_without_regenerating(self, tmp_path, monkeypatch):
        log = []
        producer = _FakeRfProducer(log)
        pipeline = self._pipeline(log, producer=producer, request=21)
        self._patch_validate(monkeypatch, log, manifest={"n_samples": 8})
        rf_paths = self._rf_paths(tmp_path)

        pipeline._obtain_rf_dataset(rf_paths, 8, 10.0, 40.0)

        # Await precedes the manifest check (validating mid-write would race the
        # fallback's regeneration against live producer writes), shutdown lands exactly
        # once, and the producer's dataset is consumed without a regeneration
        assert log == [
            ("await", 21),
            ("shutdown",),
            ("validate", rf_paths.round_dir, 8, pipeline.config.training.round_array_dtype),
        ]
        assert producer.shutdown_calls == 1
        assert pipeline._round_producer is None
        assert pipeline._rf_producer_request is None

    def test_producer_error_falls_back_to_in_process_generation(self, tmp_path, monkeypatch):
        log = []
        producer = _FakeRfProducer(log, error=RuntimeError("producer exploded"))
        pipeline = self._pipeline(log, producer=producer, request=21)
        self._patch_validate(monkeypatch, log, manifest=None)
        rf_paths = self._rf_paths(tmp_path)

        pipeline._obtain_rf_dataset(rf_paths, 8, 10.0, 40.0)

        # Producer failure still shuts it down exactly once, then the unchanged
        # in-process path runs with the same num_training_rounds+1 sentinel round_num
        assert log == [
            ("await", 21),
            ("shutdown",),
            ("validate", rf_paths.round_dir, 8, pipeline.config.training.round_array_dtype),
            ("generate", rf_paths.round_dir, 8, 10.0, 40.0, 21),
        ]
        assert producer.shutdown_calls == 1
        assert pipeline._round_producer is None

    def test_no_producer_generates_in_process(self, tmp_path, monkeypatch):
        # Overlap disabled, sequential mode, or a resumed run entering rf_train directly
        # (startup cleanup deleted any stale rf dir, so validation fails and regenerates)
        log = []
        pipeline = self._pipeline(log, producer=None, request=None)
        self._patch_validate(monkeypatch, log, manifest=None)
        rf_paths = self._rf_paths(tmp_path)

        pipeline._obtain_rf_dataset(rf_paths, 8, 10.0, 40.0)

        assert log == [
            ("validate", rf_paths.round_dir, 8, pipeline.config.training.round_array_dtype),
            ("generate", rf_paths.round_dir, 8, 10.0, 40.0, 21),
        ]

    def test_no_producer_valid_dir_is_reused(self, tmp_path, monkeypatch):
        log = []
        pipeline = self._pipeline(log, producer=None, request=None)
        self._patch_validate(monkeypatch, log, manifest={"n_samples": 8})

        pipeline._obtain_rf_dataset(self._rf_paths(tmp_path), 8, 10.0, 40.0)

        assert [entry[0] for entry in log] == ["validate"]

    def test_producer_without_pending_request_is_still_shut_down(self, tmp_path, monkeypatch):
        # Defensive: a live producer with no RF request pending must be wound down, not
        # leaked (and not awaited — there is nothing to wait for)
        log = []
        producer = _FakeRfProducer(log)
        pipeline = self._pipeline(log, producer=producer, request=None)
        self._patch_validate(monkeypatch, log, manifest=None)
        rf_paths = self._rf_paths(tmp_path)

        pipeline._obtain_rf_dataset(rf_paths, 8, 10.0, 40.0)

        assert log[0] == ("shutdown",)
        assert not any(entry[0] == "await" for entry in log)
        assert producer.shutdown_calls == 1


class _PipelineStub:
    """Just enough of TrainingPipeline to drive _calculate_curriculum_snr."""

    # NOTE: this stub intentionally carries only `.config` because _calculate_curriculum_snr
    # reads nothing else today. If that method ever touches another attribute (e.g.
    # self.logger), the stub fails with a bare AttributeError rather than a clear message —
    # extend it (or build a real instance via TrainingPipeline.__new__) at that point.

    def __init__(self, config):
        self.config = config


def _curriculum(round_idx):
    return TrainingPipeline._calculate_curriculum_snr(_PipelineStub(get_config()), round_idx)


class TestCalculateCurriculumSnr:
    @pytest.fixture(autouse=True)
    def _setup_config(self):
        config = get_config()
        config.training.num_training_rounds = 5
        config.training.snr_base = 10
        config.training.initial_snr_range = 40
        config.training.final_snr_range = 10

    def test_linear_endpoints_and_monotonicity(self):
        get_config().training.curriculum_schedule = "linear"
        ranges = [_curriculum(i)[1] for i in range(5)]
        assert ranges[0] == 40
        assert ranges[-1] == 10
        assert all(a >= b for a, b in zip(ranges, ranges[1:], strict=False))
        assert all(base == 10 for base, _ in (_curriculum(i) for i in range(5)))

    def test_linear_midpoint(self):
        get_config().training.curriculum_schedule = "linear"
        # progress = 2/4 = 0.5 -> 40 - 0.5 * 30 = 25
        assert _curriculum(2)[1] == 25

    def test_exponential_endpoints_exact(self):
        config = get_config()
        config.training.curriculum_schedule = "exponential"
        config.training.exponential_decay_rate = -3.0
        assert _curriculum(0)[1] == 40
        assert _curriculum(4)[1] == 10

    def test_exponential_decays_faster_than_linear(self):
        config = get_config()
        config.training.curriculum_schedule = "exponential"
        config.training.exponential_decay_rate = -3.0
        ranges = [_curriculum(i)[1] for i in range(5)]
        assert all(a >= b for a, b in zip(ranges, ranges[1:], strict=False))
        # Exponential front-loads the difficulty ramp: below linear at the midpoint.
        assert ranges[2] < 25

    def test_exponential_rejects_nonnegative_decay_rate(self):
        config = get_config()
        config.training.curriculum_schedule = "exponential"
        config.training.exponential_decay_rate = 0.5
        with pytest.raises(ValueError, match="must be < 0"):
            _curriculum(1)

    def test_step_schedule(self):
        config = get_config()
        config.training.curriculum_schedule = "step"
        config.training.step_easy_rounds = 2
        config.training.step_hard_rounds = 3
        assert [_curriculum(i)[1] for i in range(5)] == [40, 40, 10, 10, 10]

    def test_step_schedule_rejects_bad_sum(self):
        config = get_config()
        config.training.curriculum_schedule = "step"
        config.training.step_easy_rounds = 2
        config.training.step_hard_rounds = 2  # 2 + 2 != 5
        with pytest.raises(ValueError, match="must equal total_rounds"):
            _curriculum(0)

    def test_single_round_returns_initial_range(self):
        config = get_config()
        config.training.num_training_rounds = 1
        config.training.curriculum_schedule = "linear"
        assert _curriculum(0) == (10, 40)

    def test_unknown_schedule_raises(self):
        get_config().training.curriculum_schedule = "sigmoid"
        with pytest.raises(ValueError, match="invalid"):
            _curriculum(0)


class TestArchiveDirectory:
    def test_empty_directory_is_noop(self, tmp_path):
        base = tmp_path / "plots"
        archive_directory(str(base))
        assert base.exists()
        assert list(base.iterdir()) == []

    def test_fresh_run_moves_files_to_archive(self, tmp_path):
        base = tmp_path / "plots"
        base.mkdir()
        (base / "a.png").write_text("a")
        (base / "b.png").write_text("b")
        (base / "subdir").mkdir()  # not in target_dirs -> untouched

        archive_directory(str(base), round_num=1)

        remaining = {p.name for p in base.iterdir()}
        assert remaining == {"archive", "subdir"}
        archived = list((base / "archive").iterdir())
        assert len(archived) == 1  # one timestamped snapshot
        assert {p.name for p in archived[0].iterdir()} == {"a.png", "b.png"}

    def test_resume_copies_then_deletes_rounds_geq(self, tmp_path):
        base = tmp_path / "checkpoints"
        base.mkdir()
        for tag in ("round_01", "round_02", "round_03"):
            (base / f"vae_encoder_{tag}.keras").write_text("stub")
        (base / "notes.txt").write_text("keep me")

        archive_directory(str(base), round_num=2)

        remaining = {p.name for p in base.iterdir()}
        # Rounds >= 2 deleted; round_01 and non-round files kept.
        assert remaining == {"archive", "vae_encoder_round_01.keras", "notes.txt"}
        # Everything (including the later-deleted files) was backed up first.
        snapshot = next((base / "archive").iterdir())
        assert {p.name for p in snapshot.iterdir()} == {
            "vae_encoder_round_01.keras",
            "vae_encoder_round_02.keras",
            "vae_encoder_round_03.keras",
            "notes.txt",
        }

    def test_target_dirs_moved_and_recreated_empty(self, tmp_path):
        base = tmp_path / "tb"
        (base / "train").mkdir(parents=True)
        (base / "train" / "events.1").write_text("x")
        (base / "validation").mkdir()

        archive_directory(str(base), target_dirs=["train"], round_num=1)

        assert (base / "train").exists()
        assert list((base / "train").iterdir()) == []  # replaced with an empty dir
        assert (base / "validation").exists()  # not a target -> untouched
        snapshot = next((base / "archive").iterdir())
        assert (snapshot / "train" / "events.1").exists()


class TestEncoderTrainedHeuristics:
    @staticmethod
    def _dense_model(initializer, units=256, input_dim=256):
        import tensorflow as tf  # noqa: PLC0415

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(units, kernel_initializer=initializer),
            ]
        )
        return model

    def test_expected_std_he_normal(self):
        from tensorflow.keras.initializers import HeNormal  # noqa: PLC0415

        model = self._dense_model(HeNormal(seed=0), units=4, input_dim=8)
        expected = compute_expected_std(model.layers[-1])
        assert expected == pytest.approx(np.sqrt(2.0 / 8))

    def test_expected_std_glorot_normal(self):
        from tensorflow.keras.initializers import GlorotNormal  # noqa: PLC0415

        model = self._dense_model(GlorotNormal(seed=0), units=4, input_dim=8)
        expected = compute_expected_std(model.layers[-1])
        assert expected == pytest.approx(np.sqrt(2.0 / (8 + 4)))

    def test_expected_std_unknown_initializer_returns_none(self):
        model = self._dense_model("glorot_uniform", units=4, input_dim=8)
        assert compute_expected_std(model.layers[-1]) is None

    def test_fresh_encoder_reports_untrained(self):
        from tensorflow.keras.initializers import HeNormal  # noqa: PLC0415

        # 256x256 kernel: sampling noise on the std is ~0.3%, far under the 20% threshold.
        model = self._dense_model(HeNormal(seed=0))
        assert check_encoder_trained(model) is False

    def test_scaled_weights_report_trained(self):
        from tensorflow.keras.initializers import HeNormal  # noqa: PLC0415

        model = self._dense_model(HeNormal(seed=0))
        layer = model.layers[-1]
        kernel, bias = layer.get_weights()
        layer.set_weights([kernel * 3.0, bias])  # 200% deviation from expected std
        assert check_encoder_trained(model) is True


class TestCheckValAucFloor:
    """check_val_auc_floor: the opt-in RF val-ROC-AUC quality gate (issue #139 Gate 1)."""

    def test_disabled_by_default_computes_nothing(self, caplog):
        # Single-class labels would make roc_auc_score raise — proving the disabled gate
        # (min_val_auc=0.0) returns early without computing any metric.
        labels = np.ones(4, dtype=np.int64)
        probas = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        with caplog.at_level(logging.INFO, logger="aetherscan.train"):
            result = check_val_auc_floor(labels, probas, min_val_auc=0.0, tag="test_v1")
        assert result is None
        assert not caplog.records

    def test_floor_met_returns_auc_without_warning(self, caplog):
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        probas = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)  # perfect separation
        with caplog.at_level(logging.INFO, logger="aetherscan.train"):
            result = check_val_auc_floor(labels, probas, min_val_auc=0.9, tag="test_v1")
        assert result == pytest.approx(1.0)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("Model quality gate passed" in r.message for r in caplog.records)

    def test_floor_unmet_warns_loudly(self, caplog):
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        probas = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)  # inverted separation
        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            result = check_val_auc_floor(labels, probas, min_val_auc=0.9, tag="test_v1")
        assert result == pytest.approx(0.0)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "min_val_auc" in warnings[0].message
        assert "rf_eval_artifacts_test_v1.joblib" in warnings[0].message

    def test_single_class_labels_with_gate_enabled_warns_instead_of_raising(self, caplog):
        # roc_auc_score raises ValueError on single-class labels; with the gate enabled the
        # guard must warn and return None rather than let that escape (mirrors
        # compute_rf_eval_metrics' identical guard in rf_metrics.py).
        labels = np.ones(4, dtype=np.int64)
        probas = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            result = check_val_auc_floor(labels, probas, min_val_auc=0.9, tag="test_v1")
        assert result is None
        assert any(
            "single-class" in r.message and "cannot be evaluated" in r.message
            for r in caplog.records
        )


class TestPlotRfDiagnosticsOverlap:
    """plot_rf_diagnostics overlaps the SHAP computation (background thread) with the five
    non-SHAP plots: the SHAP plots run only after the thread joins, and a SHAP-compute failure
    is recorded once per SHAP plot exactly as the sequential path would have recorded it."""

    NON_SHAP = [
        "plot_rf_confusion_matrices",
        "plot_rf_classification_curves",
        "plot_rf_calibration_curve",
        "plot_rf_ensemble_accuracy_curve",
        "plot_rf_latent_decision_boundary",
    ]
    SHAP = [
        "plot_rf_shap_summary",
        "plot_rf_shap_dependence",
        "plot_rf_shap_interactions",
        "plot_rf_shap_loss_monitoring",
        "plot_rf_shap_explanation_clustering",
    ]

    @staticmethod
    def _recorder(log, name):
        def _record():
            log.append(name)

        return _record

    def _pipeline(self, log, shap_compute=None, artifacts_error=None):
        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        for name in self.NON_SHAP + self.SHAP:
            setattr(pipeline, name, self._recorder(log, name))

        def _load_artifacts(tag=None):
            if artifacts_error is not None:
                raise artifacts_error
            log.append("load_artifacts")
            return {"tag": "test_v1"}

        pipeline._load_rf_eval_artifacts = _load_artifacts
        pipeline._compute_or_load_shap_values = shap_compute or (
            lambda artifacts: log.append("shap_done")
        )
        return pipeline

    def test_shap_plots_wait_for_background_compute(self):
        # The stubbed SHAP compute refuses to finish until the last non-SHAP plot releases it,
        # so "shap_done" preceding every SHAP plot in the log proves the join, and the non-SHAP
        # names preceding "shap_done" prove they were not serialized behind the compute.
        log = []
        release = threading.Event()

        def _shap_compute(artifacts):
            assert release.wait(timeout=30), "non-SHAP plots never released the SHAP stub"
            log.append("shap_done")

        pipeline = self._pipeline(log, shap_compute=_shap_compute)

        def _last_non_shap_plot():
            log.append("plot_rf_latent_decision_boundary")
            release.set()

        pipeline.plot_rf_latent_decision_boundary = _last_non_shap_plot

        pipeline.plot_rf_diagnostics()

        assert log == ["load_artifacts", *self.NON_SHAP, "shap_done", *self.SHAP]

    def test_shap_failure_records_every_shap_plot_and_spares_the_rest(self, caplog):
        log = []

        def _shap_compute(artifacts):
            raise RuntimeError("SHAP exploded")

        pipeline = self._pipeline(log, shap_compute=_shap_compute)
        with (
            caplog.at_level(logging.ERROR, logger="aetherscan.train"),
            pytest.raises(RuntimeError) as excinfo,
        ):
            pipeline.plot_rf_diagnostics()

        # All five SHAP plot names failed, in order — the same bookkeeping the sequential path
        # produces when each plot hits the same compute error — and each got the per-plot
        # "Failed to execute" record; the non-SHAP successes are untouched.
        assert str(excinfo.value) == "5 plot(s) failed: " + ", ".join(self.SHAP)
        assert [entry for entry in log if entry in self.NON_SHAP] == self.NON_SHAP
        assert not any(entry in self.SHAP for entry in log)  # never invoked without values
        for name in self.SHAP:
            assert any(
                f"Failed to execute {name}: SHAP exploded" in r.message for r in caplog.records
            )

    def test_non_shap_failure_is_isolated(self):
        log = []
        pipeline = self._pipeline(log)

        def _boom():
            raise ValueError("bad plot")

        pipeline.plot_rf_calibration_curve = _boom
        with pytest.raises(RuntimeError) as excinfo:
            pipeline.plot_rf_diagnostics()

        assert str(excinfo.value) == "1 plot(s) failed: plot_rf_calibration_curve"
        # The SHAP plots still ran, after the join.
        assert log[-len(self.SHAP) :] == self.SHAP

    def test_artifact_load_failure_falls_back_to_sequential_order(self):
        # Preserves the pre-overlap path: every plot is still attempted individually, in the
        # original interleaved order (in production each would re-raise the load error itself).
        log = []
        pipeline = self._pipeline(log, artifacts_error=FileNotFoundError("no artifacts"))
        pipeline.plot_rf_diagnostics()
        assert log == self.NON_SHAP[:2] + self.SHAP + self.NON_SHAP[2:]


class _StageMachineStub:
    """Duck-typed TrainingPipeline for _execute_training_stages: records the call order,
    raises inside any method named in `fail`, and reports rf-load success per `rf_load_ok`.
    `hf_upload` toggles the opt-in hf_upload stage via the config singleton (which the real
    pipeline also reads through self.config)."""

    def __init__(self, state, fail=(), rf_load_ok=True, hf_upload=False):
        self.run_state = state
        self.calls = []
        self.rf_model = "trained-rf"
        self._fail = set(fail)
        self._rf_load_ok = rf_load_ok
        self.config = get_config()
        self.config.hf.upload_after_training = hf_upload

    def _invoke(self, name):
        self.calls.append(name)
        if name in self._fail:
            raise RuntimeError(f"{name} failed")

    def train_beta_vae(self):
        self._invoke("train_beta_vae")

    def plot_vae_diagnostics(self):
        self._invoke("plot_vae_diagnostics")

    def train_random_forest(self):
        self._invoke("train_random_forest")

    def plot_rf_diagnostics(self):
        self._invoke("plot_rf_diagnostics")

    def final_save(self):
        self._invoke("final_save")

    def upload_to_hf(self):
        self._invoke("upload_to_hf")

    def save_models(self):
        self._invoke("save_models")

    def try_load_rf_for_resume(self):
        self.calls.append("try_load_rf_for_resume")
        return self._rf_load_ok

    def _clear_rf_caches(self):
        self.calls.append("_clear_rf_caches")

    def _clear_latent_viz_data(self):
        self.calls.append("_clear_latent_viz_data")

    # The real methods also persist the manifest; persistence is covered in test_run_state.
    def _mark_stage_done(self, stage):
        self.run_state.mark_stage_done(stage)

    def _record_stage_failure(self, stage):
        self.run_state.record_stage_failure(stage)

    def _clear_stage_failure(self, stage):
        self.calls.append("_clear_stage_failure")
        self.run_state.clear_stage_failure(stage)


class TestExecuteTrainingStages:
    def _state(self, **kwargs):
        return TrainingRunState(tag="test_v1", run_start_time=1.0, **kwargs)

    def test_fresh_run_executes_all_stages_in_order(self):
        stub = _StageMachineStub(self._state())
        _execute_training_stages(stub)
        assert stub.calls == [
            "train_beta_vae",
            "plot_vae_diagnostics",
            "_clear_latent_viz_data",
            "train_random_forest",
            "plot_rf_diagnostics",
            "_clear_rf_caches",
            "final_save",
        ]
        # hf_upload is opt-in and disabled by default: skipped, not marked done
        assert stub.run_state.stages_done == [s for s in TRAINING_STAGES if s != STAGE_HF_UPLOAD]
        assert stub.run_state.stages_failed == []

    def test_done_stages_are_skipped(self):
        state = self._state(
            completed_rounds=[1, 2],
            stages_done=[STAGE_VAE_ROUNDS, STAGE_VAE_PLOTS, STAGE_RF_TRAIN],
        )
        stub = _StageMachineStub(state)
        _execute_training_stages(stub)
        # rf_train skip must reload the persisted RF (rf_plots/final_save need a live model)
        assert stub.calls == [
            "try_load_rf_for_resume",
            "plot_rf_diagnostics",
            "_clear_rf_caches",
            "final_save",
        ]
        assert state.stages_done == [s for s in TRAINING_STAGES if s != STAGE_HF_UPLOAD]

    def test_vae_plot_failure_is_recorded_but_does_not_abort(self):
        stub = _StageMachineStub(self._state(), fail={"plot_vae_diagnostics"})
        _execute_training_stages(stub)
        assert "final_save" in stub.calls
        # The viz batch is freed even when the plot group fails
        assert "_clear_latent_viz_data" in stub.calls
        assert stub.run_state.stages_failed == [STAGE_VAE_PLOTS]
        assert not stub.run_state.is_stage_done(STAGE_VAE_PLOTS)
        assert stub.run_state.is_stage_done(STAGE_FINAL_SAVE)

    def test_rf_plot_failure_is_recorded_and_caches_cleared(self):
        stub = _StageMachineStub(self._state(), fail={"plot_rf_diagnostics"})
        _execute_training_stages(stub)
        assert "_clear_rf_caches" in stub.calls
        assert "final_save" in stub.calls
        assert stub.run_state.stages_failed == [STAGE_RF_PLOTS]

    def test_rf_train_failure_saves_vae_and_propagates(self):
        stub = _StageMachineStub(self._state(), fail={"train_random_forest"})
        with pytest.raises(RuntimeError, match="train_random_forest failed"):
            _execute_training_stages(stub)
        assert stub.rf_model is None  # partial RF state dropped before the best-effort save
        assert stub.calls[-1] == "save_models"
        assert not stub.run_state.is_stage_done(STAGE_RF_TRAIN)
        assert "final_save" not in stub.calls

    def test_rf_train_skip_falls_back_to_retrain_when_reload_fails(self):
        state = self._state(
            stages_done=[STAGE_VAE_ROUNDS, STAGE_VAE_PLOTS, STAGE_RF_TRAIN],
        )
        stub = _StageMachineStub(state, rf_load_ok=False)
        _execute_training_stages(stub)
        assert "train_random_forest" in stub.calls
        assert state.is_stage_done(STAGE_RF_TRAIN)

    def test_relaunch_after_plot_failure_retries_only_failed_stage(self):
        # First run: vae_plots fails, everything else succeeds
        state = self._state()
        first = _StageMachineStub(state, fail={"plot_vae_diagnostics"})
        _execute_training_stages(first)
        assert state.stages_failed == [STAGE_VAE_PLOTS]

        # Relaunch with the persisted state: only vae_plots re-runs (the rf_train skip still
        # reloads the persisted RF — cheap, and by design), and success clears the failure
        second = _StageMachineStub(state)
        _execute_training_stages(second)
        assert second.calls == [
            "plot_vae_diagnostics",
            "_clear_latent_viz_data",
            "try_load_rf_for_resume",
        ]
        assert state.stages_failed == []
        assert state.stages_done[-1] == STAGE_VAE_PLOTS  # re-marked after the retry

    def test_hf_upload_enabled_runs_after_final_save(self):
        stub = _StageMachineStub(self._state(), hf_upload=True)
        _execute_training_stages(stub)
        assert stub.calls[-2:] == ["final_save", "upload_to_hf"]
        assert stub.run_state.stages_done == list(TRAINING_STAGES)
        assert stub.run_state.stages_failed == []

    def test_hf_upload_failure_recorded_but_never_fails_the_run(self):
        stub = _StageMachineStub(self._state(), fail={"upload_to_hf"}, hf_upload=True)
        _execute_training_stages(stub)  # must not raise — weights are already safe locally
        assert stub.run_state.stages_failed == [STAGE_HF_UPLOAD]
        assert not stub.run_state.is_stage_done(STAGE_HF_UPLOAD)
        assert stub.run_state.is_stage_done(STAGE_FINAL_SAVE)

    def test_hf_upload_done_is_skipped_on_relaunch(self):
        state = self._state(stages_done=list(TRAINING_STAGES))
        stub = _StageMachineStub(state, hf_upload=True)
        _execute_training_stages(stub)
        assert "upload_to_hf" not in stub.calls

    def test_disabling_hf_upload_clears_stale_failure(self):
        # A previous --hf-upload attempt failed; the user re-runs without it. The stale
        # failure must be dropped so it can't force a nonzero exit forever.
        state = self._state(
            stages_done=[s for s in TRAINING_STAGES if s != STAGE_HF_UPLOAD],
            stages_failed=[STAGE_HF_UPLOAD],
        )
        stub = _StageMachineStub(state, hf_upload=False)
        _execute_training_stages(stub)
        assert "upload_to_hf" not in stub.calls
        assert "_clear_stage_failure" in stub.calls
        assert state.stages_failed == []
        assert not state.is_stage_done(STAGE_HF_UPLOAD)


class TestBuildEpochHistory:
    """#277 problem 6: the real global-epoch axis — gaps stay gaps, no positional compression."""

    def test_positions_use_real_epoch_numbers(self):
        stats = [
            {"stat_name": "total_loss", "round_number": 1, "epoch_number": 1, "value": 1.0},
            {"stat_name": "total_loss", "round_number": 1, "epoch_number": 2, "value": 2.0},
            {"stat_name": "total_loss", "round_number": 2, "epoch_number": 1, "value": 4.0},
        ]
        epochs, history = build_epoch_history(stats, epochs_per_round=3)
        assert list(epochs) == [1, 2, 3, 4]
        values = history["total_loss"]
        assert values[0] == 1.0 and values[1] == 2.0 and values[3] == 4.0
        # Round 1 epoch 3 never committed -> NaN gap, NOT round 2's value shifted left
        assert math.isnan(values[2])

    def test_gap_is_not_compressed(self):
        stats = [
            {"stat_name": "kl_loss", "round_number": 1, "epoch_number": 1, "value": 1.0},
            {"stat_name": "kl_loss", "round_number": 1, "epoch_number": 5, "value": 5.0},
        ]
        epochs, history = build_epoch_history(stats, epochs_per_round=10)
        assert list(epochs) == [1, 2, 3, 4, 5]
        values = history["kl_loss"]
        assert values[0] == 1.0 and values[4] == 5.0
        assert all(math.isnan(v) for v in values[1:4])

    def test_duplicate_rows_last_wins(self):
        stats = [
            {"stat_name": "lr", "round_number": 1, "epoch_number": 1, "value": 0.1},
            {"stat_name": "lr", "round_number": 1, "epoch_number": 1, "value": 0.2},
        ]
        _, history = build_epoch_history(stats, epochs_per_round=2)
        assert history["lr"] == [0.2]

    def test_rows_without_round_or_epoch_are_skipped(self):
        stats = [
            {"stat_name": "scalar", "round_number": None, "epoch_number": None, "value": 9.0},
            {"stat_name": "total_loss", "round_number": 1, "epoch_number": 1, "value": 1.0},
        ]
        epochs, history = build_epoch_history(stats, epochs_per_round=5)
        assert list(epochs) == [1]
        assert "scalar" not in history

    def test_empty_input(self):
        epochs, history = build_epoch_history([], epochs_per_round=5)
        assert list(epochs) == []
        assert history == {}


class TestPlotFlushGates:
    """#277 problem 5: a failed flush (or a pending bulk backlog) skips the plot — it never
    renders a partial result set. Raising is the skip mechanism: per-round callers catch and
    log; _run_plot_group records a non-critical failure."""

    @staticmethod
    def _pipeline_with_db(flush_ok: bool = True, backlog: int = 0):
        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        config = get_config()
        config.checkpoint.save_tag = "test_20260101_000000"
        pipeline.config = config

        class _DBStub:
            def flush(self, timeout=None):
                return flush_ok

            def injection_backlog_rows(self, max_round=None):
                return backlog

        pipeline.db = _DBStub()
        pipeline.start_time = 0.0
        return pipeline

    def test_loss_curves_skip_on_flush_failure(self):
        pipeline = self._pipeline_with_db(flush_ok=False)
        with pytest.raises(RuntimeError, match="skipping the beta-VAE loss curves"):
            pipeline.plot_beta_vae_loss_curves()

    def test_stability_skip_on_flush_failure(self):
        pipeline = self._pipeline_with_db(flush_ok=False)
        with pytest.raises(RuntimeError, match="skipping the training stability plot"):
            pipeline.plot_beta_vae_training_stability()

    def test_injection_skip_on_bulk_backlog(self):
        pipeline = self._pipeline_with_db(flush_ok=True, backlog=7)
        with pytest.raises(RuntimeError, match="bulk lane"):
            pipeline.plot_injection_stats(round_number=1)

    def test_injection_skip_on_flush_failure(self):
        pipeline = self._pipeline_with_db(flush_ok=False, backlog=0)
        with pytest.raises(RuntimeError, match="skipping injection"):
            pipeline.plot_injection_stats(round_number=1)


class TestCheckPosteriorCollapse:
    """#282: advisory posterior-collapse guard (WARN, never fail)."""

    def test_healthy_round_passes(self, caplog):
        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            flagged = check_posterior_collapse(
                kl_per_dim=np.array([0.5, 0.8, 0.3, 0.6]),
                low_kl_streaks=np.zeros(4),
                kl_epsilon=0.01,
                min_active_fraction=0.5,
                patience=5,
                tag="round_01",
            )
        assert flagged is False
        assert not [r for r in caplog.records if "POSTERIOR COLLAPSE" in r.message]

    def test_low_active_fraction_flags(self, caplog):
        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            flagged = check_posterior_collapse(
                kl_per_dim=np.array([0.5, 0.001, 0.002, 0.003]),
                low_kl_streaks=np.zeros(4),
                kl_epsilon=0.01,
                min_active_fraction=0.5,
                patience=5,
                tag="round_02",
            )
        assert flagged is True
        assert any("POSTERIOR COLLAPSE" in r.message for r in caplog.records)

    def test_stuck_streak_flags_even_with_enough_active(self, caplog):
        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            flagged = check_posterior_collapse(
                kl_per_dim=np.array([0.5, 0.8, 0.3, 0.001]),
                low_kl_streaks=np.array([0, 0, 0, 7]),
                kl_epsilon=0.01,
                min_active_fraction=0.5,
                patience=5,
                tag="round_03",
            )
        assert flagged is True
        assert any("dims stuck below epsilon" in r.message for r in caplog.records)


class TestCheckScreeningThreshold:
    """#282: the cascade's screen must lose ~zero recall vs MC-on-everything."""

    def test_safe_screen_passes(self, caplog):
        labels = np.array([1, 1, 0, 0])
        pass1 = np.array([0.9, 0.8, 0.2, 0.1])
        mc = np.array([0.995, 0.992, 0.3, 0.2])
        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            stats = check_screening_threshold(
                test_labels=labels,
                pass1_probas=pass1,
                mc_mean_probas=mc,
                screening_threshold=0.5,
                science_threshold=0.99,
                recall_tolerance=0.0,
                tag="t",
            )
        assert stats["screen_recall_loss"] == pytest.approx(0.0)
        assert stats["screen_max_safe_threshold"] == pytest.approx(0.8)
        assert not [r for r in caplog.records if "UNSAFE" in r.message]

    def test_lossy_screen_warns_with_numbers(self, caplog):
        labels = np.array([1, 1, 0])
        pass1 = np.array([0.9, 0.3, 0.1])  # second positive rejected by the screen
        mc = np.array([0.995, 0.995, 0.2])  # ...but MC would have promoted it
        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            stats = check_screening_threshold(
                test_labels=labels,
                pass1_probas=pass1,
                mc_mean_probas=mc,
                screening_threshold=0.5,
                science_threshold=0.99,
                recall_tolerance=0.0,
                tag="t",
            )
        assert stats["screen_recall_mc_everything"] == pytest.approx(1.0)
        assert stats["screen_recall_cascade"] == pytest.approx(0.5)
        assert stats["screen_max_safe_threshold"] == pytest.approx(0.3)
        assert any("SCREENING THRESHOLD UNSAFE" in r.message for r in caplog.records)
