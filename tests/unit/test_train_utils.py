# NOTE: come back to this later

"""Unit tests for aetherscan.train pure-logic helpers: checkpoint tag resolution, curriculum
schedules, directory archiving, encoder-trained heuristics, the val-AUC quality floor, SHAP
output normalization, the SHAP disk-cache consistency guards (#359) and input fingerprint
(#414), the rf_train dataset producer-await/fallback composition (_obtain_rf_dataset), the
rf_plots overlap coordinator (plot_rf_diagnostics), and the training stage machine
(skip-if-done / record-failure semantics against a stub pipeline)."""

from __future__ import annotations

import logging
import math
import os
import threading
import types

import joblib
import numpy as np
import pytest

from aetherscan.config import get_config
from aetherscan.db import get_machine_name
from aetherscan.display_tag import display_tag
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
    _rf_artifact_digest,
    _shap_cache_consistent,
    _shap_content_fingerprint,
    _shap_input_fingerprint,
    archive_directory,
    build_epoch_history,
    check_encoder_trained,
    check_posterior_collapse,
    check_screening_threshold,
    check_val_auc_floor,
    compute_expected_std,
)


def _touch_pair(checkpoints_dir, tag):
    """Create a matching encoder/decoder checkpoint pair for `tag`, named exactly as save_models
    lands them on disk: the display tag ({command}_{machine}_{datetime}) for a resolved run tag,
    else `tag` unchanged (round_XX / final / non-run-tags pass through). _resolve_load_tag /
    _model_pair_exists reconstruct the same name from the plain tag on this host."""
    os.makedirs(checkpoints_dir, exist_ok=True)
    fname_tag = display_tag(tag, get_machine_name())
    for prefix in ("vae_encoder", "vae_decoder"):
        with open(os.path.join(checkpoints_dir, f"{prefix}_{fname_tag}.keras"), "w") as f:
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


class TestDisplayTagFilenameInvariant:
    """The join-key guarantee behind the display-tag refactor: save_models writes the model pair
    under the DISPLAY-tagged filename ({command}_{machine}_{datetime}), and the resume/load path
    (_model_pair_exists -> _resolve_load_tag) reconstructs that same name from the plain DB tag +
    this host. So a resume on the writing host locates its own artifacts, while the plain-tagged
    name another host would leave is deliberately NOT accepted (the display tag is f(DB tag,
    local machine); the DB tag alone no longer locates the file)."""

    def test_reader_reconstructs_the_writers_display_name(self, tmp_path):
        tag = "train_20260731_182011"  # a real {command}_{datetime} run tag
        # Writer side: exactly what save_models(tag) lands on disk this host.
        _touch_pair(str(tmp_path), tag)
        expected = os.path.join(
            str(tmp_path), f"vae_encoder_{display_tag(tag, get_machine_name())}.keras"
        )
        assert os.path.exists(expected)
        assert get_machine_name() in os.path.basename(expected)  # the machine token is present
        # Reader side: derives the same display name, then adopts the plain DB tag for identity.
        assert _resolve_load_tag(str(tmp_path), tag) == tag

    def test_plain_tagged_pair_is_not_accepted_for_a_run_tag(self, tmp_path):
        # A plain {command}_{datetime} pair (no machine token — e.g. copied from another host, or
        # a pre-refactor write) must NOT satisfy the display-tag reader on this host.
        tag = "train_20260731_182011"
        for prefix in ("vae_encoder", "vae_decoder"):
            with open(os.path.join(str(tmp_path), f"{prefix}_{tag}.keras"), "w") as f:
                f.write("stub")
        with pytest.raises(FileNotFoundError):
            _resolve_load_tag(str(tmp_path), tag)


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
        # Manifest FILENAME is display-tagged (mirrors _init_run_state); its stored `tag` stays plain.
        save_run_state(state, run_state_path(str(tmp_path), display_tag(tag, get_machine_name())))

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


def _shap_payload(n_summary=8, n_interact=4, n_features=6, n_val=32):
    """A well-formed rf_shap_values_{tag} payload, indices drawn in-range for `n_val`."""
    rng = np.random.default_rng(0)
    return {
        "shap_values_summary": np.zeros((n_summary, n_features), dtype=np.float32),
        "summary_indices": np.sort(rng.choice(n_val, size=n_summary, replace=False)),
        "shap_values_interaction": np.zeros((n_interact, n_features, n_features), dtype=np.float32),
        "interaction_indices": np.sort(rng.choice(n_val, size=n_interact, replace=False)),
        "shap_values_logloss": np.zeros((n_summary, n_features), dtype=np.float32),
        "expected_value": 0.5,
    }


class TestShapCacheConsistent:
    """_shap_cache_consistent (#359): shape-only validation of a disk-cached SHAP payload
    against the eval artifacts + config in hand — the cache is keyed on {tag} alone, so every
    mismatch here is a stale-cache-under-a-reused-tag scenario."""

    def test_matching_payload_accepted(self):
        assert _shap_cache_consistent(
            _shap_payload(), n_val=32, n_features=6, n_summary=8, n_interact=4
        )

    def test_summary_count_mismatch_rejected(self):
        # shap_max_samples_summary changed between the cache write and this run
        assert not _shap_cache_consistent(
            _shap_payload(n_summary=8), n_val=32, n_features=6, n_summary=16, n_interact=4
        )

    def test_interaction_count_mismatch_rejected(self):
        assert not _shap_cache_consistent(
            _shap_payload(n_interact=4), n_val=32, n_features=6, n_summary=8, n_interact=2
        )

    def test_out_of_range_summary_indices_rejected(self):
        # cache written against a larger val split than the current artifacts carry — counts
        # can still match, so the index-range check is what catches it
        payload = _shap_payload(n_summary=8, n_val=32)
        payload["summary_indices"] = np.array([0, 1, 2, 3, 4, 5, 6, 600])
        assert not _shap_cache_consistent(
            payload, n_val=32, n_features=6, n_summary=8, n_interact=4
        )

    def test_feature_width_mismatch_rejected(self):
        # a different latent variant won under the same tag → different feature width
        assert not _shap_cache_consistent(
            _shap_payload(n_features=6), n_val=32, n_features=48, n_summary=8, n_interact=4
        )

    def test_malformed_payloads_rejected_not_raised(self):
        # The guard exists to tolerate whatever is on disk: junk must be inconsistent,
        # never an exception.
        kwargs = {"n_val": 32, "n_features": 6, "n_summary": 8, "n_interact": 4}
        assert not _shap_cache_consistent("not a dict", **kwargs)
        assert not _shap_cache_consistent(None, **kwargs)
        missing_key = _shap_payload()
        del missing_key["expected_value"]
        assert not _shap_cache_consistent(missing_key, **kwargs)
        wrong_type = _shap_payload()
        wrong_type["shap_values_summary"] = [[0.0] * 6] * 8  # list, not ndarray
        assert not _shap_cache_consistent(wrong_type, **kwargs)

    def test_logloss_shape_mismatch_rejected(self):
        payload = _shap_payload(n_summary=8, n_features=6)
        payload["shap_values_logloss"] = np.zeros((8, 5), dtype=np.float32)
        assert not _shap_cache_consistent(
            payload, n_val=32, n_features=6, n_summary=8, n_interact=4
        )


class TestComputeOrLoadShapValuesCacheGuard:
    """_compute_or_load_shap_values (#359 + #414): the on-disk cache is returned only when it
    passes the shape-consistency guard AND its recorded input fingerprint (eval matrix +
    labels + persisted RF artifact) matches the inputs in hand; a stale payload is recomputed
    and its file overwritten, and a fingerprint that cannot be verified (no RF artifact on
    disk) degrades to warn-and-reuse."""

    N_VAL = 32
    N_FEATURES = 6

    def _artifacts(self):
        rng = np.random.default_rng(1)
        return {
            "tag": "test_v1",
            "train_features": rng.normal(size=(64, self.N_FEATURES)).astype(np.float32),
            "val_features": rng.normal(size=(self.N_VAL, self.N_FEATURES)).astype(np.float32),
            "val_binary_labels": (np.arange(self.N_VAL) % 2).astype(np.int64),
        }

    def _pipeline(self, n_summary=8, n_interact=4):
        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline.config = get_config()
        pipeline.config.training.shap_max_samples_summary = n_summary
        pipeline.config.training.shap_max_samples_interaction = n_interact
        pipeline.config.manager.n_processes = 1
        pipeline._rf_shap_cache = {}
        pipeline.rf_model = types.SimpleNamespace(model={"stub": "rf"})
        return pipeline

    def _shap_path(self, config, tag="test_v1"):
        return os.path.join(
            config.model_path, f"rf_shap_values_{display_tag(tag, get_machine_name())}.joblib"
        )

    def _rf_path(self, config, tag="test_v1"):
        return os.path.join(
            config.model_path, f"random_forest_{display_tag(tag, get_machine_name())}.joblib"
        )

    def _dump_rf_and_fingerprint(self, config, artifacts, model=None):
        """Persist an RF artifact and return the #414 input fingerprint a cache written
        against (these artifacts, that RF file) would carry."""
        rf_path = self._rf_path(config)
        os.makedirs(os.path.dirname(rf_path), exist_ok=True)
        joblib.dump(model if model is not None else {"stub": "rf"}, rf_path)
        return _shap_input_fingerprint(
            artifacts["val_features"],
            artifacts["val_binary_labels"],
            _rf_artifact_digest(rf_path),
        )

    def _stub_compute(self, monkeypatch, calls):
        """Replace the SHAP pool + explainer with counting stubs so the recompute path runs
        without the real TreeSHAP machinery."""
        import contextlib as _contextlib  # noqa: PLC0415

        import aetherscan.train as train_module  # noqa: PLC0415

        @_contextlib.contextmanager
        def _fake_pool(rf_path, n_workers, background=None):
            def _run_pass(name, features, y=None):
                calls.append(name)
                if name == "interaction":
                    return np.zeros(
                        (len(features), features.shape[1], features.shape[1]),
                        dtype=np.float32,
                    )
                return np.zeros((len(features), features.shape[1]), dtype=np.float32)

            yield _run_pass

        monkeypatch.setattr(train_module, "shap_pool", _fake_pool)
        monkeypatch.setattr(
            train_module,
            "shap",
            types.SimpleNamespace(
                TreeExplainer=lambda model: types.SimpleNamespace(expected_value=[0.3, 0.7])
            ),
        )

    def test_consistent_cache_returned_without_recompute(self, monkeypatch):
        pipeline = self._pipeline(n_summary=8, n_interact=4)
        artifacts = self._artifacts()
        payload = _shap_payload(
            n_summary=8, n_interact=4, n_features=self.N_FEATURES, n_val=self.N_VAL
        )
        # #414: a servable cache carries the input fingerprint of (these artifacts, the
        # RF artifact on disk)
        payload["input_fingerprint"] = self._dump_rf_and_fingerprint(pipeline.config, artifacts)
        shap_path = self._shap_path(pipeline.config)
        os.makedirs(os.path.dirname(shap_path), exist_ok=True)
        joblib.dump(payload, shap_path)

        calls = []
        self._stub_compute(monkeypatch, calls)
        result = TrainingPipeline._compute_or_load_shap_values(pipeline, artifacts)

        assert calls == []  # never entered the compute path
        assert np.array_equal(result["summary_indices"], payload["summary_indices"])
        assert pipeline._rf_shap_cache["test_v1"] is result

    def test_pre_414_cache_without_fingerprint_recomputed(self, monkeypatch, caplog):
        # A pre-#414 cache passes every shape check but records no input fingerprint —
        # it cannot be tied to the current model/data, so it migrates via one recompute.
        pipeline = self._pipeline(n_summary=8, n_interact=4)
        legacy = _shap_payload(
            n_summary=8, n_interact=4, n_features=self.N_FEATURES, n_val=self.N_VAL
        )
        shap_path = self._shap_path(pipeline.config)
        os.makedirs(os.path.dirname(shap_path), exist_ok=True)
        joblib.dump(legacy, shap_path)

        calls = []
        self._stub_compute(monkeypatch, calls)
        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            result = TrainingPipeline._compute_or_load_shap_values(pipeline, self._artifacts())

        assert any("predate the input fingerprint" in r.message for r in caplog.records)
        assert calls == ["summary", "interaction", "logloss"]
        # The rewritten cache is post-#414: it now carries the fingerprint
        assert "input_fingerprint" in result
        assert joblib.load(shap_path)["input_fingerprint"] == result["input_fingerprint"]

    def test_different_model_same_shapes_recomputed(self, monkeypatch, caplog):
        # The #414 core case: every shape matches (same seeds, same n_val), but the cache
        # was fingerprinted against a DIFFERENT RF artifact — a leftover values cache from
        # an older model must not be served.
        pipeline = self._pipeline(n_summary=8, n_interact=4)
        artifacts = self._artifacts()
        stale = _shap_payload(
            n_summary=8, n_interact=4, n_features=self.N_FEATURES, n_val=self.N_VAL
        )
        stale["input_fingerprint"] = self._dump_rf_and_fingerprint(
            pipeline.config, artifacts, model={"stub": "OLD rf"}
        )
        # The current model's artifact replaces the old one on disk
        joblib.dump({"stub": "rf"}, self._rf_path(pipeline.config))
        shap_path = self._shap_path(pipeline.config)
        os.makedirs(os.path.dirname(shap_path), exist_ok=True)
        joblib.dump(stale, shap_path)

        calls = []
        self._stub_compute(monkeypatch, calls)
        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            result = TrainingPipeline._compute_or_load_shap_values(pipeline, artifacts)

        assert any("input-fingerprint mismatch" in r.message for r in caplog.records)
        assert calls == ["summary", "interaction", "logloss"]
        # The fresh cache is fingerprinted against the CURRENT RF artifact
        assert result["input_fingerprint"] == _shap_input_fingerprint(
            artifacts["val_features"],
            artifacts["val_binary_labels"],
            _rf_artifact_digest(self._rf_path(pipeline.config)),
        )

    def test_different_val_contents_same_shapes_recomputed(self, monkeypatch, caplog):
        # Same shapes, same n_val, same RF — but the val CONTENTS changed (the exact gap
        # _shap_cache_consistent's docstring used to disclaim).
        pipeline = self._pipeline(n_summary=8, n_interact=4)
        artifacts = self._artifacts()
        other = dict(artifacts)
        other["val_features"] = artifacts["val_features"] + 1.0
        stale = _shap_payload(
            n_summary=8, n_interact=4, n_features=self.N_FEATURES, n_val=self.N_VAL
        )
        stale["input_fingerprint"] = self._dump_rf_and_fingerprint(pipeline.config, other)
        shap_path = self._shap_path(pipeline.config)
        os.makedirs(os.path.dirname(shap_path), exist_ok=True)
        joblib.dump(stale, shap_path)

        calls = []
        self._stub_compute(monkeypatch, calls)
        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            TrainingPipeline._compute_or_load_shap_values(pipeline, artifacts)

        assert any("input-fingerprint mismatch" in r.message for r in caplog.records)
        assert calls == ["summary", "interaction", "logloss"]

    def test_missing_rf_artifact_serves_cache_unverified(self, monkeypatch, caplog):
        # Cannot-verify is not verified-stale: with no RF artifact on disk the model
        # component can't be checked (and a recompute would itself need the missing RF) —
        # the cache is served with a warning.
        pipeline = self._pipeline(n_summary=8, n_interact=4)
        payload = _shap_payload(
            n_summary=8, n_interact=4, n_features=self.N_FEATURES, n_val=self.N_VAL
        )
        payload["input_fingerprint"] = "whatever-was-recorded"
        shap_path = self._shap_path(pipeline.config)
        os.makedirs(os.path.dirname(shap_path), exist_ok=True)
        joblib.dump(payload, shap_path)
        assert not os.path.exists(self._rf_path(pipeline.config))

        calls = []
        self._stub_compute(monkeypatch, calls)
        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            result = TrainingPipeline._compute_or_load_shap_values(pipeline, self._artifacts())

        assert any("cannot be verified" in r.message for r in caplog.records)
        assert calls == []
        assert np.array_equal(result["summary_indices"], payload["summary_indices"])

    def test_stale_cache_recomputed_and_overwritten(self, monkeypatch, caplog):
        # The cache on disk was written when shap_max_samples_summary was 8; this run wants 16.
        pipeline = self._pipeline(n_summary=16, n_interact=4)
        stale = _shap_payload(
            n_summary=8, n_interact=4, n_features=self.N_FEATURES, n_val=self.N_VAL
        )
        shap_path = self._shap_path(pipeline.config)
        os.makedirs(os.path.dirname(shap_path), exist_ok=True)
        joblib.dump(stale, shap_path)

        calls = []
        self._stub_compute(monkeypatch, calls)
        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            result = TrainingPipeline._compute_or_load_shap_values(pipeline, self._artifacts())

        assert any("recomputing" in r.message for r in caplog.records)
        assert calls == ["summary", "interaction", "logloss"]
        assert result["shap_values_summary"].shape == (16, self.N_FEATURES)
        assert len(result["summary_indices"]) == 16
        # the stale file was overwritten with the fresh payload
        assert len(joblib.load(shap_path)["summary_indices"]) == 16

    def test_compute_path_without_rf_raises_pointed_error(self, monkeypatch):
        # Any entry into the compute path (stale cache here; no cache behaves identically)
        # needs a loaded RF — the guard's message is the behavior under test.
        pipeline = self._pipeline(n_summary=16, n_interact=4)
        pipeline.rf_model = None
        stale = _shap_payload(
            n_summary=8, n_interact=4, n_features=self.N_FEATURES, n_val=self.N_VAL
        )
        shap_path = self._shap_path(pipeline.config)
        os.makedirs(os.path.dirname(shap_path), exist_ok=True)
        joblib.dump(stale, shap_path)

        with pytest.raises(RuntimeError, match="no trained RF is loaded"):
            TrainingPipeline._compute_or_load_shap_values(pipeline, self._artifacts())


class TestShapClusteringCacheGuard:
    """plot_rf_shap_explanation_clustering (#359): a cached UMAP/KMeans joblib is accepted only
    when its row count matches AND its persisted content fingerprint equals the current SHAP
    matrix's — anything else (stale rows, same-shape different content, pre-fingerprint schema,
    junk) is refit and overwritten instead of IndexError-ing or plotting silently wrong points.
    The clustering joblib and the SHAP-values joblib are separate {tag}-keyed caches that can
    go out of sync when only one is regenerated."""

    N_SUMMARY = 24
    N_VAL = 40
    N_FEATURES = 4

    def _pipeline(self, monkeypatch, umap_fits: list | None = None):
        rng = np.random.default_rng(2)
        subtypes = np.array(["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"])[
            np.arange(self.N_VAL) % 4
        ]
        artifacts = {
            "tag": "test_v1",
            "val_subtype_labels": subtypes,
            "val_binary_labels": (np.arange(self.N_VAL) % 2).astype(np.int64),
            "val_preds": (np.arange(self.N_VAL) % 3 == 0).astype(np.int64),
            "classification_threshold": 0.5,
        }
        shap_data = {
            "shap_values_summary": rng.normal(size=(self.N_SUMMARY, self.N_FEATURES)).astype(
                np.float32
            ),
            "summary_indices": np.sort(rng.choice(self.N_VAL, size=self.N_SUMMARY, replace=False)),
        }

        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline.config = get_config()
        # _training_plots_dir builds the savefig base from config.checkpoint.save_tag (not
        # the tag argument) — without this the plot path joins None and TypeErrors.
        pipeline.config.checkpoint.save_tag = "test_v1"
        pipeline._load_rf_eval_artifacts = lambda tag=None: artifacts
        pipeline._compute_or_load_shap_values = lambda artifacts: shap_data

        # Stub UMAP + KMeans: the guard under test decides whether they run at all, and real
        # fits (numba JIT) would dominate the test's wall time without testing our logic.
        fits = umap_fits if umap_fits is not None else []

        class _FakeUmap:
            def __init__(self, **kwargs):
                pass

            def fit(self, values):
                fits.append("umap")
                return self

            def transform(self, values):
                return np.zeros((len(values), 2), dtype=np.float32)

        class _FakeKmeans:
            def __init__(self, **kwargs):
                pass

            def fit_predict(self, values):
                return np.arange(len(values)) % 4

        import aetherscan.train as train_module  # noqa: PLC0415

        monkeypatch.setattr(train_module, "umap", types.SimpleNamespace(UMAP=_FakeUmap))
        monkeypatch.setattr(train_module, "KMeans", _FakeKmeans)
        return pipeline, shap_data

    def _clustering_path(self, config, tag="test_v1"):
        return os.path.join(
            config.model_path,
            f"rf_shap_clustering_{display_tag(tag, get_machine_name())}.joblib",
        )

    def test_stale_clustering_cache_refit_and_overwritten(self, monkeypatch, caplog):
        fits = []
        pipeline, shap_data = self._pipeline(monkeypatch, umap_fits=fits)
        clustering_path = self._clustering_path(pipeline.config)
        os.makedirs(os.path.dirname(clustering_path), exist_ok=True)
        # Written by an earlier run whose n_summary was 10 — stale against today's 24
        # (and a pre-fingerprint schema: no shap_fingerprint key).
        joblib.dump(
            {
                "embedding": np.zeros((10, 2), dtype=np.float32),
                "cluster_labels": np.zeros(10, dtype=np.int64),
            },
            clustering_path,
        )

        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            TrainingPipeline.plot_rf_shap_explanation_clustering(pipeline, tag="test_v1")

        assert any("refitting" in r.message for r in caplog.records)
        assert fits == ["umap"]
        refit = joblib.load(clustering_path)
        assert len(refit["embedding"]) == self.N_SUMMARY
        assert len(refit["cluster_labels"]) == self.N_SUMMARY
        # the refit persists the fingerprint of the SHAP matrix it consumed, arming the guard
        assert refit["shap_fingerprint"] == _shap_content_fingerprint(
            shap_data["shap_values_summary"]
        )

    def test_same_shape_different_content_refit(self, monkeypatch, caplog):
        # The silent-wrongness case neither row counts nor seeded indices can catch: a cache
        # fit on a DIFFERENT SHAP matrix of identical shape (e.g. a different latent variant
        # winning the sweep under a reused tag) must be refit, not served.
        fits = []
        pipeline, shap_data = self._pipeline(monkeypatch, umap_fits=fits)
        clustering_path = self._clustering_path(pipeline.config)
        os.makedirs(os.path.dirname(clustering_path), exist_ok=True)
        other_matrix = shap_data["shap_values_summary"] + 1.0  # same shape, different content
        joblib.dump(
            {
                "embedding": np.zeros((self.N_SUMMARY, 2), dtype=np.float32),
                "cluster_labels": (np.arange(self.N_SUMMARY) % 4).astype(np.int64),
                "shap_fingerprint": _shap_content_fingerprint(other_matrix),
            },
            clustering_path,
        )

        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            TrainingPipeline.plot_rf_shap_explanation_clustering(pipeline, tag="test_v1")

        assert any("refitting" in r.message for r in caplog.records)
        assert fits == ["umap"]

    def test_pre_fingerprint_schema_with_matching_rows_refit(self, monkeypatch, caplog):
        # The actual migration path for every clustering joblib on disk today: correct row
        # count, no shap_fingerprint key. Pins the key-set precheck in isolation — the
        # stale-cache test above bundles it with a row-count mismatch.
        fits = []
        pipeline, _ = self._pipeline(monkeypatch, umap_fits=fits)
        clustering_path = self._clustering_path(pipeline.config)
        os.makedirs(os.path.dirname(clustering_path), exist_ok=True)
        joblib.dump(
            {
                "embedding": np.zeros((self.N_SUMMARY, 2), dtype=np.float32),
                "cluster_labels": (np.arange(self.N_SUMMARY) % 4).astype(np.int64),
            },
            clustering_path,
        )

        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            TrainingPipeline.plot_rf_shap_explanation_clustering(pipeline, tag="test_v1")

        assert any("refitting" in r.message for r in caplog.records)
        assert fits == ["umap"]
        assert "shap_fingerprint" in joblib.load(clustering_path)

    def test_malformed_clustering_cache_refit_not_raised(self, monkeypatch, caplog):
        fits = []
        pipeline, _ = self._pipeline(monkeypatch, umap_fits=fits)
        clustering_path = self._clustering_path(pipeline.config)
        os.makedirs(os.path.dirname(clustering_path), exist_ok=True)
        joblib.dump("junk, not a dict", clustering_path)

        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            TrainingPipeline.plot_rf_shap_explanation_clustering(pipeline, tag="test_v1")

        assert any("refitting" in r.message for r in caplog.records)
        assert fits == ["umap"]

    def test_matching_clustering_cache_used_without_refit(self, monkeypatch):
        fits = []
        pipeline, shap_data = self._pipeline(monkeypatch, umap_fits=fits)
        clustering_path = self._clustering_path(pipeline.config)
        os.makedirs(os.path.dirname(clustering_path), exist_ok=True)
        # Sentinel values: if the accepted cache were silently overwritten or ignored, the
        # reload below would not still hold them.
        cached = {
            "embedding": np.full((self.N_SUMMARY, 2), 7.0, dtype=np.float32),
            "cluster_labels": (np.arange(self.N_SUMMARY) % 4).astype(np.int64),
            "shap_fingerprint": _shap_content_fingerprint(shap_data["shap_values_summary"]),
        }
        joblib.dump(cached, clustering_path)

        TrainingPipeline.plot_rf_shap_explanation_clustering(pipeline, tag="test_v1")

        assert fits == []  # cache accepted, no refit
        # ...and the accepted cache is untouched on disk (no rewrite happened)
        reloaded = joblib.load(clustering_path)
        assert np.array_equal(reloaded["embedding"], cached["embedding"])
        assert np.array_equal(reloaded["cluster_labels"], cached["cluster_labels"])


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


class TestLatentVariantSelectionPlot:
    """plot_rf_latent_variant_selection renders the #282 variant-selection figure straight from
    the in-memory metrics dict + winner name (no DB, no TF): the recall bar chart with the winner
    highlighted and — when the tie-break passed over a higher-recall variant — the tie band, plus
    the AUC/Brier/ECE/feature-count companions. Driven off a bare __new__ instance."""

    def _pipeline(self, tag):
        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline.config = get_config()
        pipeline.config.checkpoint.save_tag = tag
        return pipeline

    def _variant_metrics(self):
        # Realistic #282 shape: the simplest variant (z_mean) sits a hair BELOW the top recall
        # (z / z_aug), so the tie-break can pick it and the plot's band spans the traded recall.
        recalls = {
            "z_mean": 0.9994,
            "z": 0.9996,
            "z_aug": 0.9996,
            "z_mean_total_kl": 0.9990,
            "z_mean_obs_logvar": 0.9992,
            "z_mean_dim_logvar": 0.9988,
            "z_mean_logvar_active": 0.9985,
            "z_mean_logvar": 0.9989,
        }
        feats = {
            "z_mean": 48,
            "z": 48,
            "z_aug": 48,
            "z_mean_total_kl": 49,
            "z_mean_obs_logvar": 54,
            "z_mean_dim_logvar": 56,
            "z_mean_logvar_active": 72,
            "z_mean_logvar": 96,
        }
        return {
            name: {
                "recall_at_fpr": recalls[name],
                "roc_auc": 0.990 + 0.001 * i,
                "brier": 0.010 + 0.001 * i,
                "ece": 0.020 + 0.001 * i,
                "n_features": feats[name],
            }
            for i, name in enumerate(recalls)
        }

    def test_writes_nonempty_png_with_tiebreak_winner(self):
        # winner (z_mean) has LOWER recall than the best (z): exercises the tie-band path.
        pipeline = self._pipeline("sel_tiebreak")
        pipeline.plot_rf_latent_variant_selection(self._variant_metrics(), "z_mean")
        path = os.path.join(
            pipeline._training_plots_dir(), "latent_variant_selection_sel_tiebreak.png"
        )
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_writes_nonempty_png_when_winner_is_top_recall(self):
        # winner == best (no tie-break): the band collapses; the figure must still render.
        pipeline = self._pipeline("sel_top")
        pipeline.plot_rf_latent_variant_selection(self._variant_metrics(), "z")
        path = os.path.join(pipeline._training_plots_dir(), "latent_variant_selection_sel_top.png")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
