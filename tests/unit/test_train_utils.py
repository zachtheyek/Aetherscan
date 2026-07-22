# NOTE: come back to this later

"""Unit tests for aetherscan.train pure-logic helpers: checkpoint tag resolution, curriculum
schedules, directory archiving, encoder-trained heuristics, the val-AUC quality floor, SHAP
output normalization, and the training stage machine (skip-if-done / record-failure semantics
against a stub pipeline)."""

from __future__ import annotations

import logging
import os
import types

import numpy as np
import pytest

from aetherscan.config import get_config
from aetherscan.run_state import (
    STAGE_FINAL_SAVE,
    STAGE_HF_UPLOAD,
    STAGE_RF_PLOTS,
    STAGE_RF_TRAIN,
    STAGE_VAE_PLOTS,
    STAGE_VAE_ROUNDS,
    TRAINING_STAGES,
    TrainingRunState,
)
from aetherscan.train import (
    TrainingPipeline,
    _execute_training_stages,
    _resolve_load_tag,
    _select_positive_class_shap,
    archive_directory,
    check_encoder_trained,
    check_val_auc_floor,
    compute_expected_std,
    get_latest_tag,
)


def _touch_pair(checkpoints_dir, tag):
    """Create a matching encoder/decoder checkpoint pair for `tag`."""
    os.makedirs(checkpoints_dir, exist_ok=True)
    for prefix in ("vae_encoder", "vae_decoder"):
        with open(os.path.join(checkpoints_dir, f"{prefix}_{tag}.keras"), "w") as f:
            f.write("stub")


class TestGetLatestTag:
    def test_priority_ladder(self, tmp_path):
        d = str(tmp_path / "ckpt")
        for tag in ("test_v3", "20240101_000000", "round_02", "final_v1"):
            _touch_pair(d, tag)
        assert get_latest_tag(d) == "final_v1"

    def test_round_beats_timestamp_and_test(self, tmp_path):
        d = str(tmp_path / "ckpt")
        for tag in ("test_v9", "20991231_235959", "round_02", "round_10"):
            _touch_pair(d, tag)
        # Numeric compare, not lexicographic: round_10 > round_02.
        assert get_latest_tag(d) == "round_10"

    def test_timestamp_beats_test(self, tmp_path):
        d = str(tmp_path / "ckpt")
        for tag in ("test_v9", "20240101_000000", "20250101_000000"):
            _touch_pair(d, tag)
        assert get_latest_tag(d) == "20250101_000000"

    def test_test_tags_ranked_by_version(self, tmp_path):
        d = str(tmp_path / "ckpt")
        for tag in ("test_v2", "test_v17", "test_v9"):
            _touch_pair(d, tag)
        assert get_latest_tag(d) == "test_v17"

    def test_final_ranked_by_version(self, tmp_path):
        d = str(tmp_path / "ckpt")
        for tag in ("final_v1", "final_v12", "final_v3"):
            _touch_pair(d, tag)
        assert get_latest_tag(d) == "final_v12"

    def test_encoder_without_decoder_ignored(self, tmp_path):
        d = str(tmp_path / "ckpt")
        _touch_pair(d, "round_01")
        # Higher-priority final_v2 lacks its decoder — must not win.
        with open(os.path.join(d, "vae_encoder_final_v2.keras"), "w") as f:
            f.write("stub")
        assert get_latest_tag(d) == "round_01"

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="doesn't exist"):
            get_latest_tag(str(tmp_path / "nope"))

    def test_empty_directory_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError, match="No encoder files"):
            get_latest_tag(str(d))

    def test_no_complete_pair_raises(self, tmp_path):
        d = tmp_path / "orphans"
        d.mkdir()
        with open(d / "vae_encoder_round_01.keras", "w") as f:
            f.write("stub")
        with pytest.raises(FileNotFoundError, match="No valid model pairs"):
            get_latest_tag(str(d))


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
        # For a non-round explicit tag (e.g. a typo'd final_v2) the checkpoints hint is a
        # red herring and must not appear.
        _touch_pair(tmp_path, "final_v1")
        with pytest.raises(FileNotFoundError) as excinfo:
            _resolve_load_tag(str(tmp_path), "final_v2")
        assert "--load-dir checkpoints" not in str(excinfo.value)

    def test_default_prefers_final(self, tmp_path):
        _touch_pair(tmp_path, "final")
        _touch_pair(tmp_path, "round_09")
        assert _resolve_load_tag(str(tmp_path), None) == "final"

    def test_default_falls_back_to_latest(self, tmp_path):
        _touch_pair(tmp_path, "round_02")
        _touch_pair(tmp_path, "test_v1")
        assert _resolve_load_tag(str(tmp_path), None) == "round_02"

    def test_default_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _resolve_load_tag(str(tmp_path), None)


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

        with caplog.at_level(logging.WARNING, logger="aetherscan.train"):
            pipeline.train_random_forest()

        assert pipeline.rf_training_skipped_from_tag == "test_v27"
        assert any(
            "RF training SKIPPED" in r.message and "test_v27" in r.message for r in caplog.records
        )


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


class TestSelectPositiveClassShap:
    N, F = 5, 8

    def test_list_of_class_arrays_selects_positive(self):
        neg = np.zeros((self.N, self.F))
        pos = np.ones((self.N, self.F))
        result = _select_positive_class_shap([neg, pos])
        np.testing.assert_array_equal(result, pos)

    def test_trailing_class_axis_values(self):
        values = np.stack(
            [np.zeros((self.N, self.F)), np.ones((self.N, self.F))], axis=-1
        )  # (N, F, 2)
        result = _select_positive_class_shap(values)
        assert result.shape == (self.N, self.F)
        assert np.all(result == 1.0)

    def test_trailing_class_axis_interactions(self):
        values = np.stack(
            [np.zeros((self.N, self.F, self.F)), np.ones((self.N, self.F, self.F))], axis=-1
        )  # (N, F, F, 2)
        result = _select_positive_class_shap(values)
        assert result.shape == (self.N, self.F, self.F)
        assert np.all(result == 1.0)

    def test_single_output_passthrough(self):
        values = np.arange(self.N * self.F, dtype=float).reshape(self.N, self.F)
        np.testing.assert_array_equal(_select_positive_class_shap(values), values)

    def test_log_loss_list_selects_first(self):
        first = np.ones((self.N, self.F))
        result = _select_positive_class_shap([first, np.zeros((self.N, self.F))], log_loss=True)
        np.testing.assert_array_equal(result, first)

    def test_log_loss_trailing_class_axis(self):
        values = np.stack([np.zeros((self.N, self.F)), np.ones((self.N, self.F))], axis=-1)
        result = _select_positive_class_shap(values, log_loss=True)
        assert result.shape == (self.N, self.F)
        assert np.all(result == 1.0)

    def test_log_loss_passthrough(self):
        values = np.arange(self.N * self.F, dtype=float).reshape(self.N, self.F)
        np.testing.assert_array_equal(_select_positive_class_shap(values, log_loss=True), values)


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
