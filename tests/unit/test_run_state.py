"""Unit tests for aetherscan.run_state: manifest round-trip, atomic persistence, corrupt-file
downgrade, and the stage/round bookkeeping helpers that drive the training stage machine."""

from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from aetherscan.run_state import (
    STAGE_FINAL_SAVE,
    STAGE_RF_PLOTS,
    STAGE_RF_TRAIN,
    STAGE_VAE_PLOTS,
    STAGE_VAE_ROUNDS,
    TRAINING_STAGES,
    TrainingRunState,
    load_run_state,
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
