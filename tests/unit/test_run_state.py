"""Unit tests for aetherscan.run_state: manifest round-trip, atomic persistence, corrupt-file
downgrade, and the stage/round bookkeeping helpers that drive the training stage machine."""

from __future__ import annotations

import copy
import json
import os
from unittest import mock

import pytest

from aetherscan.run_state import (
    STAGE_FINAL_SAVE,
    STAGE_HF_UPLOAD,
    STAGE_RF_PLOTS,
    STAGE_RF_TRAIN,
    STAGE_VAE_PLOTS,
    STAGE_VAE_ROUNDS,
    TRAINING_STAGES,
    TrainingRunState,
    config_changed,
    config_fingerprint,
    inference_config_fingerprint,
    load_run_state,
    preprocessing_config_fingerprint,
    run_state_path,
    save_run_state,
)


class TestRunStatePath:
    def test_layout(self, tmp_path):
        assert run_state_path(str(tmp_path), "test_v1") == str(tmp_path / "run_state_test_v1.json")


class TestRoundTrip:
    def test_save_then_load_preserves_all_fields(self, tmp_path):
        path = run_state_path(str(tmp_path), "test_v1")
        state = TrainingRunState(
            tag="test_v1",
            run_start_time=1234.5,
            attempt=3,
            completed_rounds=[1, 2],
            stages_done=[STAGE_VAE_ROUNDS],
            stages_failed=[STAGE_VAE_PLOTS],
        )
        save_run_state(state, path)
        assert load_run_state(path) == state

    def test_save_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "deeper" / "run_state_test_v1.json")
        save_run_state(TrainingRunState(tag="test_v1", run_start_time=1.0), path)
        assert load_run_state(path) is not None

    def test_missing_file_returns_none(self, tmp_path):
        assert load_run_state(str(tmp_path / "run_state_missing.json")) is None

    def test_corrupt_file_downgrades_to_none(self, tmp_path):
        path = str(tmp_path / "run_state_test_v1.json")
        with open(path, "w") as f:
            f.write("{ this is not json")
        assert load_run_state(path) is None

    def test_missing_required_field_downgrades_to_none(self, tmp_path):
        path = str(tmp_path / "run_state_test_v1.json")
        with open(path, "w") as f:
            json.dump({"tag": "test_v1"}, f)  # no run_start_time
        assert load_run_state(path) is None

    def test_completed_rounds_are_sorted_on_load(self, tmp_path):
        path = str(tmp_path / "run_state_test_v1.json")
        with open(path, "w") as f:
            json.dump({"tag": "test_v1", "run_start_time": 1.0, "completed_rounds": [3, 1, 2]}, f)
        assert load_run_state(path).completed_rounds == [1, 2, 3]


class TestAtomicity:
    def test_failed_write_leaves_previous_manifest_intact(self, tmp_path):
        path = run_state_path(str(tmp_path), "test_v1")
        original = TrainingRunState(tag="test_v1", run_start_time=1.0, completed_rounds=[1])
        save_run_state(original, path)

        updated = TrainingRunState(tag="test_v1", run_start_time=1.0, completed_rounds=[1, 2])
        with (
            mock.patch("aetherscan.run_state.json.dump", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            save_run_state(updated, path)

        # The previous manifest survives untouched and no .tmp litter is left behind
        assert load_run_state(path) == original
        assert not os.path.exists(path + ".tmp")

    def test_write_goes_through_tmp_then_replace(self, tmp_path):
        path = run_state_path(str(tmp_path), "test_v1")
        with mock.patch("aetherscan.run_state.os.replace", wraps=os.replace) as mock_replace:
            save_run_state(TrainingRunState(tag="test_v1", run_start_time=1.0), path)
        mock_replace.assert_called_once_with(path + ".tmp", path)


class TestBookkeeping:
    def test_stage_names(self):
        assert TRAINING_STAGES == (
            STAGE_VAE_ROUNDS,
            STAGE_VAE_PLOTS,
            STAGE_RF_TRAIN,
            STAGE_RF_PLOTS,
            STAGE_FINAL_SAVE,
            STAGE_HF_UPLOAD,
        )

    def test_mark_stage_done_is_idempotent(self):
        state = TrainingRunState(tag="test_v1", run_start_time=1.0)
        state.mark_stage_done(STAGE_VAE_ROUNDS)
        state.mark_stage_done(STAGE_VAE_ROUNDS)
        assert state.stages_done == [STAGE_VAE_ROUNDS]
        assert state.is_stage_done(STAGE_VAE_ROUNDS)
        assert not state.is_stage_done(STAGE_VAE_PLOTS)

    def test_stage_success_clears_recorded_failure(self):
        state = TrainingRunState(tag="test_v1", run_start_time=1.0)
        state.record_stage_failure(STAGE_VAE_PLOTS)
        state.record_stage_failure(STAGE_VAE_PLOTS)
        assert state.stages_failed == [STAGE_VAE_PLOTS]
        state.mark_stage_done(STAGE_VAE_PLOTS)
        assert state.stages_failed == []
        assert state.is_stage_done(STAGE_VAE_PLOTS)

    def test_clear_stage_failure_drops_without_marking_done(self):
        state = TrainingRunState(tag="test_v1", run_start_time=1.0)
        state.record_stage_failure(STAGE_HF_UPLOAD)
        state.clear_stage_failure(STAGE_HF_UPLOAD)
        assert state.stages_failed == []
        assert not state.is_stage_done(STAGE_HF_UPLOAD)
        # Clearing an unrecorded stage is a no-op
        state.clear_stage_failure(STAGE_HF_UPLOAD)
        assert state.stages_failed == []

    def test_mark_round_completed_sorted_and_deduped(self):
        state = TrainingRunState(tag="test_v1", run_start_time=1.0)
        state.mark_round_completed(2)
        state.mark_round_completed(1)
        state.mark_round_completed(2)
        assert state.completed_rounds == [1, 2]

    def test_resume_round(self):
        state = TrainingRunState(tag="test_v1", run_start_time=1.0)
        assert state.resume_round == 1
        state.mark_round_completed(1)
        state.mark_round_completed(2)
        assert state.resume_round == 3


class TestConfigFingerprint:
    """config_fingerprint() + config_changed(): the config-drift guard that stops a reused
    --save-tag from silently resuming/skipping under a changed config."""

    BASE = {
        "paths": {"data_path": "/data", "output_path": "/out", "model_path": "/models"},
        "db": {"write_interval": 5.0},
        "manager": {"n_processes": 32, "chunks_per_worker": 4},
        "logger": {"slack_enabled": True},
        "beta_vae": {"beta": 1.5, "alpha": 10.0, "latent_dim": 8},
        "random_forest": {"n_estimators": 1000, "seed": 11},
        "data": {"num_target_backgrounds": 45000},
        "training": {"num_samples_beta_vae": 499200, "max_retries": 3, "retry_delay": 60},
        "gpu": {"num_replicas": None},
        "inference": {"max_retries": 3, "coarse_channel_log_interval": None},
        "hf": {
            "upload_after_training": False,
            "repo_id": "zachtheyek/aetherscan",
            "revision": None,
        },
        "checkpoint": {"save_tag": "final_v1", "load_tag": None, "start_round": 1},
    }

    def _mutate(self, section, key, value):
        d = copy.deepcopy(self.BASE)
        d[section][key] = value
        return d

    def test_stable_for_identical_config(self):
        assert config_fingerprint(self.BASE) == config_fingerprint(copy.deepcopy(self.BASE))

    @pytest.mark.parametrize(
        "section,key,value",
        [
            ("beta_vae", "beta", 2.0),
            ("beta_vae", "latent_dim", 16),
            ("random_forest", "n_estimators", 500),
            ("data", "num_target_backgrounds", 30000),
            ("training", "num_samples_beta_vae", 8192),
            ("gpu", "num_replicas", 6),
        ],
    )
    def test_result_affecting_change_flips_fingerprint(self, section, key, value):
        assert config_fingerprint(self._mutate(section, key, value)) != config_fingerprint(
            self.BASE
        )

    @pytest.mark.parametrize(
        "section,key,value",
        [
            ("paths", "data_path", "/elsewhere"),
            ("paths", "output_path", "/other_out"),
            ("db", "write_interval", 99.0),
            ("manager", "n_processes", 96),
            ("logger", "slack_enabled", False),
            ("inference", "max_retries", 9),
            # HF upload config never affects the training result — toggling --hf-upload or
            # changing its repo/revision must NOT discard the manifest (regression: #130).
            ("hf", "upload_after_training", True),
            ("hf", "repo_id", "other/repo"),
            ("checkpoint", "load_tag", "round_03"),
            ("checkpoint", "start_round", 4),
            ("training", "max_retries", 10),
            ("training", "retry_delay", 5),
        ],
    )
    def test_excluded_change_keeps_fingerprint(self, section, key, value):
        assert config_fingerprint(self._mutate(section, key, value)) == config_fingerprint(
            self.BASE
        )

    def test_config_changed_none_state_is_false(self):
        assert config_changed(None, "abc") is False

    def test_config_changed_matching_is_false(self):
        state = TrainingRunState(tag="t", run_start_time=1.0, config_fingerprint="abc")
        assert config_changed(state, "abc") is False

    def test_config_changed_mismatch_is_true(self):
        state = TrainingRunState(tag="t", run_start_time=1.0, config_fingerprint="abc")
        assert config_changed(state, "xyz") is True

    def test_config_changed_pre_fingerprint_manifest_is_true(self):
        # A manifest written before fingerprinting has "" and must be treated as changed.
        state = TrainingRunState(tag="t", run_start_time=1.0, config_fingerprint="")
        assert config_changed(state, "abc") is True

    def test_fingerprint_round_trips_through_manifest(self, tmp_path):
        path = run_state_path(str(tmp_path), "final_v1")
        state = TrainingRunState(tag="final_v1", run_start_time=1.0, config_fingerprint="deadbeef")
        save_run_state(state, path)
        assert load_run_state(path).config_fingerprint == "deadbeef"


class TestInferenceConfigFingerprint:
    """inference_config_fingerprint(): the inference-side config-drift guard. The denylist must
    keep inert knobs (I/O, batching, retry, viz, and the coarse_channel_log_interval progress-log
    cadence) out of the hash, while result-affecting inference params and data-geometry changes
    flip it."""

    BASE = {
        "inference": {
            "classification_threshold": 0.99,
            "stat_threshold": 2048.0,
            "coarse_channel_log_interval": None,
            "max_retries": 3,
            "inference_viz_enabled": True,
        },
        "data": {
            "downsample_factor": 8,
            "width_bin": 4096,
            "num_observations": 6,
            "time_bins": 16,
        },
    }

    def _mutate(self, section, key, value):
        d = copy.deepcopy(self.BASE)
        d[section][key] = value
        return d

    def test_stable_for_identical_config(self):
        assert inference_config_fingerprint(self.BASE) == inference_config_fingerprint(
            copy.deepcopy(self.BASE)
        )

    @pytest.mark.parametrize(
        "section,key,value",
        [
            ("inference", "classification_threshold", 0.5),
            ("inference", "stat_threshold", 1024.0),
            ("data", "downsample_factor", 4),
            ("data", "width_bin", 2048),
            ("data", "num_observations", 4),
            ("data", "time_bins", 32),
        ],
    )
    def test_result_affecting_change_flips_fingerprint(self, section, key, value):
        assert inference_config_fingerprint(self._mutate(section, key, value)) != (
            inference_config_fingerprint(self.BASE)
        )

    @pytest.mark.parametrize(
        "key,value",
        [
            # coarse_channel_log_interval is inert (progress-log cadence only); changing it must
            # NOT invalidate stage-aware resume. Regression guard for the
            # parallel_coarse_chans -> coarse_channel_log_interval rename: the denylist in
            # run_state.py must track the new field name, else this knob leaks into the hash and
            # a reused --save-tag needlessly re-infers every cadence.
            ("coarse_channel_log_interval", 8),
            ("max_retries", 9),
            ("inference_viz_enabled", False),
            # prefetch_depth is scheduling only (#298 N2): per-cadence results are
            # depth-invariant, so changing it must not force a re-inference on resume.
            ("prefetch_depth", 2),
            # #302/#301: retention + viz-scope knobs are result-invariant; they MUST be
            # denylisted or every 'inferred' resume row stales on upgrade.
            ("prune_stamps", False),
            ("inference_viz_scope", "new"),
        ],
    )
    def test_inert_inference_change_keeps_fingerprint(self, key, value):
        d = copy.deepcopy(self.BASE)
        d["inference"][key] = value
        assert inference_config_fingerprint(d) == inference_config_fingerprint(self.BASE)

    def test_new_keys_absent_hashes_like_present(self):
        """The load-bearing #301/#302 upgrade property the PR claims: a to_dict() that
        GAINS prune_stamps + inference_viz_scope must hash identically to a master-shaped
        dict WITHOUT them, for BOTH inference and preprocessing fingerprints — else the
        upgrade stales every 'inferred' resume row and renames every ED cache dir. This
        pins the claim (previously only asserted live)."""
        master_shaped = copy.deepcopy(self.BASE)  # no prune_stamps / inference_viz_scope
        branch_shaped = copy.deepcopy(self.BASE)
        branch_shaped["inference"]["prune_stamps"] = True
        branch_shaped["inference"]["inference_viz_scope"] = "full"
        assert inference_config_fingerprint(branch_shaped) == inference_config_fingerprint(
            master_shaped
        )
        assert preprocessing_config_fingerprint(branch_shaped) == preprocessing_config_fingerprint(
            master_shaped
        )


class TestPreprocessingConfigFingerprint:
    """preprocessing_config_fingerprint (#298 I3): keys the fingerprint-scoped stamp cache
    directory. Scoring/model knobs must NOT flip it (a new encoder/threshold reuses stamps —
    the whole point); ED-affecting keys and unknown new keys MUST flip it (fail-safe
    denylist: an unrecognized key over-invalidates rather than silently reusing stamps)."""

    BASE = {
        "inference": {
            "encoder_path": "/models/a/vae_encoder.keras",
            "rf_path": "/models/a/random_forest.joblib",
            "classification_threshold": 0.99,
            "screening_threshold": 0.5,
            "mc_draws": 32,
            "reference_cloud_size": 10000,
            "stat_threshold": 2048.0,
            "coarse_channel_width": 1048576,
            "bandpass_method": "pfb",
            "pfb_taps_per_channel": 12,
            "stamp_width": 4096,
            "overlap_fraction": 0.5,
            "cadence_group_by_cols": ["Target", "Session"],
            "max_retries": 3,
        },
        "data": {
            "downsample_factor": 8,
            "width_bin": 4096,
            "num_observations": 6,
            "time_bins": 16,
        },
    }

    def _mutate(self, section, key, value):
        d = copy.deepcopy(self.BASE)
        d[section][key] = value
        return d

    @pytest.mark.parametrize(
        "key,value",
        [
            ("encoder_path", "/models/b/vae_encoder.keras"),
            ("rf_path", "/models/b/random_forest.joblib"),
            ("classification_threshold", 0.5),
            ("screening_threshold", 0.25),
            ("mc_draws", 64),
            ("reference_cloud_size", 0),
            ("max_retries", 9),
        ],
    )
    def test_scoring_and_inert_keys_keep_fingerprint(self, key, value):
        assert preprocessing_config_fingerprint(
            self._mutate("inference", key, value)
        ) == preprocessing_config_fingerprint(self.BASE)

    @pytest.mark.parametrize(
        "section,key,value",
        [
            ("inference", "stat_threshold", 1024.0),
            ("inference", "coarse_channel_width", 524288),
            ("inference", "bandpass_method", "spline"),
            ("inference", "pfb_taps_per_channel", 4),
            ("inference", "stamp_width", 2048),
            ("inference", "overlap_fraction", 0.25),
            ("inference", "cadence_group_by_cols", ["Target"]),
            ("data", "downsample_factor", 4),
            ("data", "width_bin", 2048),
        ],
    )
    def test_ed_affecting_change_flips_fingerprint(self, section, key, value):
        assert preprocessing_config_fingerprint(
            self._mutate(section, key, value)
        ) != preprocessing_config_fingerprint(self.BASE)

    def test_unknown_new_key_over_invalidates(self):
        # The denylist's fail-safe: a future inference field this module has never heard of
        # must flip the fingerprint (forcing a harmless re-preprocess), never reuse stamps.
        d = copy.deepcopy(self.BASE)
        d["inference"]["some_future_ed_knob"] = 7
        assert preprocessing_config_fingerprint(d) != preprocessing_config_fingerprint(self.BASE)

    def test_differs_from_inference_fingerprint_scope(self):
        # Scoring changes flip the RESUME fingerprint (re-infer) while keeping the
        # PREPROCESSING fingerprint (reuse stamps) — the two guards protect different things.
        changed = self._mutate("inference", "classification_threshold", 0.5)
        assert inference_config_fingerprint(changed) != inference_config_fingerprint(self.BASE)
        assert preprocessing_config_fingerprint(changed) == preprocessing_config_fingerprint(
            self.BASE
        )
