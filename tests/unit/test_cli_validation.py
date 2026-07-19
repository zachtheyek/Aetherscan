# NOTE: come back to this later

"""Unit tests for aetherscan.cli: tag pattern, validation matrix, cross-param solver, and
apply_saved_config precedence (defaults < saved config < CLI args)."""

from __future__ import annotations

import collections
import json

import pytest

from aetherscan import cli
from aetherscan.cli import (
    _TAG_PATTERN,
    _check_cross_constraints,
    _solve_cross_param_constraints,
    apply_args_to_config,
    apply_saved_config,
    collect_validation_errors,
    setup_argument_parser,
)
from aetherscan.config import get_config

# The blpc3 smoke config from the repo's known-good runbook: divisible for exactly 5 replicas.
# Mirrored by _SMOKE_FLAGS in tests/integration/test_train_smoke.py (which runs it for real on
# the cluster) — keep the two in sync.
_SMOKE_FLAGS_5_REPLICAS = [
    "--per-replica-batch-size",
    "4",
    "--per-replica-val-batch-size",
    "4",
    "--effective-batch-size",
    "20",
    "--num-samples-beta-vae",
    "200",
    "--num-samples-rf",
    "200",
    "--latent-viz-num-cadences-per-type",
    "5",
]


def _parse(argv):
    return setup_argument_parser().parse_args(argv)


def _cross_param_errors(args, num_replicas):
    errors = collect_validation_errors(args, num_replicas)
    return [e for e in errors if e.fix_kind == "cross_param"]


@pytest.fixture(autouse=True)
def _ample_disk_space(monkeypatch):
    """The round-data disk-budget check reads the real filesystem via shutil.disk_usage (the
    full-scale default config needs ~650 GB free — more than CI runners or dev boxes have).
    Pin it to a huge value so validation tests are machine-independent; tests that exercise
    the check itself (TestRoundDataFlags) patch their own values on top."""
    usage = collections.namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda path: usage(2**60, 0, 2**60))


class TestTagPattern:
    @pytest.mark.parametrize(
        "tag",
        ["20260712_123456", "final_v1", "final_v12", "round_01", "round_5", "test_v1", "test_v17"],
    )
    def test_accepted_formats(self, tag):
        assert _TAG_PATTERN.match(tag)

    @pytest.mark.parametrize(
        "tag",
        [
            "smoke_blackwell",  # free-form slug
            "final_1",  # missing v
            "final_v",  # missing version number
            "test_17",  # missing v
            "TEST_V1",  # wrong case
            "2026_0712",  # malformed timestamp
            "20260712-123456",  # wrong separator
            "20260712_12345",  # HHMMS (5 digits)
            "round_",  # missing round number
            " test_v1",  # leading whitespace
            "test_v1 ",  # trailing whitespace
            "",
        ],
    )
    def test_rejected_formats(self, tag):
        assert not _TAG_PATTERN.match(tag)


class TestCrossReplicaDivisibilityMatrix:
    """Default config divides cleanly across 4 or 6 replicas but NOT 5; the blpc3 smoke config
    is the inverse. collect_validation_errors must reproduce that matrix exactly."""

    @pytest.mark.parametrize("num_replicas", [4, 6])
    def test_defaults_valid_for_4_and_6_replicas(self, num_replicas):
        assert _cross_param_errors(_parse(["train"]), num_replicas) == []

    def test_defaults_invalid_for_5_replicas(self):
        errors = _cross_param_errors(_parse(["train"]), 5)
        violated = {e.field for e in errors}
        assert violated == {
            "training.effective_batch_size",
            "training.num_samples_beta_vae",
            "training.num_samples_rf",
            "training.latent_viz_num_cadences_per_type",
        }

    def test_smoke_config_valid_for_5_replicas(self):
        assert _cross_param_errors(_parse(["train", *_SMOKE_FLAGS_5_REPLICAS]), 5) == []

    @pytest.mark.parametrize("num_replicas", [4, 6])
    def test_smoke_config_invalid_for_4_and_6_replicas(self, num_replicas):
        assert _cross_param_errors(_parse(["train", *_SMOKE_FLAGS_5_REPLICAS]), num_replicas)

    def test_unknown_replica_count_skips_cross_checks(self):
        # num_replicas=None (no GPUs detectable) must skip the divisibility section entirely,
        # even for a config that would fail on any replica count.
        assert _cross_param_errors(_parse(["train", *_SMOKE_FLAGS_5_REPLICAS]), None) == []


class TestSemanticChecks:
    def test_save_tag_format_error(self):
        errors = collect_validation_errors(_parse(["train", "--save-tag", "bogus"]), None)
        assert any(e.field == "checkpoint.save_tag" and e.fix_kind == "format" for e in errors)

    def test_load_tag_format_error(self):
        errors = collect_validation_errors(_parse(["train", "--load-tag", "bogus"]), None)
        assert any(e.field == "checkpoint.load_tag" and e.fix_kind == "format" for e in errors)

    def test_num_samples_divisible_by_4(self):
        errors = collect_validation_errors(_parse(["train", "--num-samples-beta-vae", "202"]), None)
        assert any(e.field == "training.num_samples_beta_vae" and e.divisor == 4 for e in errors)

    def test_curriculum_schedule_enum(self):
        errors = collect_validation_errors(
            _parse(["train", "--curriculum-schedule", "sigmoid"]), None
        )
        assert any(
            e.field == "training.curriculum_schedule" and e.fix_kind == "enum" for e in errors
        )

    def test_step_schedule_sum_constraint(self):
        argv = [
            "train",
            "--curriculum-schedule",
            "step",
            "--num-training-rounds",
            "10",
            "--step-easy-rounds",
            "3",
            "--step-hard-rounds",
            "4",
        ]
        errors = collect_validation_errors(_parse(argv), None)
        assert any(
            e.field == "training.step_easy_rounds" and e.fix_kind == "cross_param" for e in errors
        )

    def test_snr_curriculum_ordering(self):
        argv = ["train", "--initial-snr-range", "5", "--final-snr-range", "10"]
        errors = collect_validation_errors(_parse(argv), None)
        assert any(e.field == "training.initial_snr_range" for e in errors)

    def test_missing_train_files_reported(self):
        errors = collect_validation_errors(_parse(["train"]), None)
        # Default train_files don't exist under the tmp data path.
        file_errors = [e for e in errors if e.fix_kind == "file_exists"]
        assert {e.field for e in file_errors} == {"data.train_files"}
        assert len(file_errors) == len(get_config().data.train_files)

    def test_existing_train_files_pass(self, tmp_path):
        config = get_config()
        for filename in config.data.train_files:
            (tmp_path / "data" / "training" / filename).touch()
        errors = collect_validation_errors(_parse(["train"]), None)
        assert [e for e in errors if e.fix_kind == "file_exists"] == []

    def test_inference_requires_model_artifacts(self):
        errors = collect_validation_errors(_parse(["inference"]), None)
        fields = {e.field for e in errors if e.fix_kind == "file_exists"}
        assert {
            "inference.encoder_path",
            "inference.rf_path",
            "inference.config_path",
        } <= fields

    def test_inference_stamp_width_must_match_width_bin(self, make_inference_csv):
        csv_path = make_inference_csv("subset.csv")
        argv = [
            "inference",
            "--inference-files",
            csv_path.name,
            "--stamp-width",
            "2048",  # width_bin default is 4096
        ]
        errors = collect_validation_errors(_parse(argv), None)
        assert any(e.field == "inference.stamp_width" for e in errors)


class TestCrossParamSolver:
    _VALID_BASE = {
        # The repo defaults: valid for 4 and 6 replicas.
        "num_samples_beta_vae": 499200,
        "num_samples_rf": 99840,
        "train_val_split": 0.8,
        "per_replica_batch_size": 128,
        "effective_batch_size": 3072,
        "per_replica_val_batch_size": 80,
    }

    def test_check_cross_constraints_matrix(self):
        assert _check_cross_constraints(**self._VALID_BASE, num_replicas_list=[4])
        assert _check_cross_constraints(**self._VALID_BASE, num_replicas_list=[6])
        assert not _check_cross_constraints(**self._VALID_BASE, num_replicas_list=[5])

    def test_solver_returns_base_when_already_valid(self):
        assert _solve_cross_param_constraints(self._VALID_BASE, [4, 6]) == self._VALID_BASE

    def test_solver_respects_candidate_budget(self):
        # The default search ranges exceed a tiny budget, so the solver must bail with None
        # rather than grind through the grid.
        base = dict(self._VALID_BASE)
        base["effective_batch_size"] = 3070  # invalidate so the grid search is attempted
        assert _solve_cross_param_constraints(base, [4], max_candidates=10) is None

    def test_solver_finds_nearest_valid_config(self, monkeypatch):
        # Shrink the search ranges to a tractable grid and verify the solver picks the
        # L1-nearest satisfying combination.
        monkeypatch.setattr(
            cli,
            "_SEARCH_RANGES",
            {
                "num_samples_beta_vae": (160, 320, 80),
                "num_samples_rf": (40, 80, 20),
                "per_replica_batch_size": (4, 8, 4),
                "effective_batch_size": (16, 64, 16),
                "per_replica_val_batch_size": (4, 8, 4),
            },
        )
        base = {
            "num_samples_beta_vae": 250,
            "num_samples_rf": 50,
            "train_val_split": 0.8,
            "per_replica_batch_size": 5,
            "effective_batch_size": 20,
            "per_replica_val_batch_size": 5,
        }
        solution = _solve_cross_param_constraints(base, [4])
        assert solution is not None
        assert solution["train_val_split"] == base["train_val_split"]  # held fixed
        assert _check_cross_constraints(**solution, num_replicas_list=[4])
        # Every solved field must come from the (patched) search grid.
        for field_name, (lo, hi, step) in cli._SEARCH_RANGES.items():
            assert solution[field_name] in range(lo, hi + 1, step)
        # And the solution must be L1-minimal among all valid grid points.
        assert self._l1(solution, base) == min(
            self._l1(candidate, base) for candidate in self._valid_grid_points(base)
        )

    @staticmethod
    def _l1(candidate, base):
        return sum(
            abs(candidate[f] - base[f])
            for f in (
                "num_samples_beta_vae",
                "num_samples_rf",
                "per_replica_batch_size",
                "effective_batch_size",
                "per_replica_val_batch_size",
            )
        )

    @staticmethod
    def _valid_grid_points(base):
        from itertools import product  # noqa: PLC0415

        ranges = {f: range(lo, hi + 1, step) for f, (lo, hi, step) in cli._SEARCH_RANGES.items()}
        for nsb, nsr, prb, eb, prvb in product(
            ranges["num_samples_beta_vae"],
            ranges["num_samples_rf"],
            ranges["per_replica_batch_size"],
            ranges["effective_batch_size"],
            ranges["per_replica_val_batch_size"],
        ):
            candidate = {
                "num_samples_beta_vae": nsb,
                "num_samples_rf": nsr,
                "train_val_split": base["train_val_split"],
                "per_replica_batch_size": prb,
                "effective_batch_size": eb,
                "per_replica_val_batch_size": prvb,
            }
            if _check_cross_constraints(**candidate, num_replicas_list=[4]):
                yield candidate


class TestApplySavedConfigPrecedence:
    def test_saved_config_overrides_defaults(self, tmp_path):
        saved = {
            "training": {"num_training_rounds": 7, "snr_base": 33},
            "checkpoint": {"save_tag": "final_v9"},
            "data_path": "/saved/data/path",
        }
        path = tmp_path / "saved_config.json"
        path.write_text(json.dumps(saved))

        apply_saved_config(str(path))
        config = get_config()
        assert config.training.num_training_rounds == 7
        assert config.training.snr_base == 33
        # Documented sharp edge: the saved file clobbers checkpoint.save_tag too.
        assert config.checkpoint.save_tag == "final_v9"
        assert config.data_path == "/saved/data/path"

    def test_cli_args_override_saved_config(self, tmp_path):
        saved = {"training": {"snr_base": 33, "num_training_rounds": 7}}
        path = tmp_path / "saved_config.json"
        path.write_text(json.dumps(saved))
        apply_saved_config(str(path))

        args = _parse(["train", "--snr-base", "44"])
        apply_args_to_config(args)
        config = get_config()
        assert config.training.snr_base == 44  # CLI wins over saved
        assert config.training.num_training_rounds == 7  # saved wins over default (20)

    def test_unknown_keys_and_fields_skipped(self, tmp_path):
        saved = {
            "not_a_section": {"whatever": 1},
            "training": {"not_a_field": 123, "snr_base": 21},
        }
        path = tmp_path / "saved_config.json"
        path.write_text(json.dumps(saved))
        apply_saved_config(str(path))
        config = get_config()
        assert config.training.snr_base == 21
        assert not hasattr(config.training, "not_a_field")
        assert not hasattr(config, "not_a_section")

    def test_missing_file_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            apply_saved_config("/nonexistent/config.json")

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            apply_saved_config(str(path))


class TestApplyArgsToConfig:
    def test_none_args_leave_defaults(self):
        config = get_config()
        default_rounds = config.training.num_training_rounds
        apply_args_to_config(_parse(["train"]))
        assert config.training.num_training_rounds == default_rounds

    def test_mode_scoped_flags_route_to_correct_section(self):
        # --per-replica-batch-size / --max-retries exist in both subparsers; the command
        # gates which config section receives them.
        config = get_config()
        apply_args_to_config(_parse(["train", "--per-replica-batch-size", "64"]))
        assert config.training.per_replica_batch_size == 64

        inference_default = config.inference.per_replica_batch_size
        apply_args_to_config(_parse(["inference", "--per-replica-batch-size", "32"]))
        assert config.inference.per_replica_batch_size == 32
        assert config.training.per_replica_batch_size == 64  # untouched by inference flag
        assert inference_default != 32

    def test_load_tag_infers_start_round(self):
        config = get_config()
        apply_args_to_config(_parse(["train", "--load-tag", "round_03"]))
        assert config.checkpoint.start_round == 4


class TestRoundDataFlags:
    """Flags and validation for the disk-backed round-data pipeline (round_data.py)."""

    def test_flags_apply_to_config(self):
        apply_args_to_config(
            _parse(
                [
                    "train",
                    "--round-data-dir",
                    "/scratch/rounds",
                    "--no-overlap-data-generation",
                    "--keep-round-data",
                    "--data-gen-task-size",
                    "128",
                ]
            )
        )
        config = get_config()
        assert config.training.round_data_dir == "/scratch/rounds"
        assert config.training.overlap_data_generation is False
        assert config.training.keep_round_data is True
        assert config.training.data_gen_task_size == 128

    def test_defaults_preserved_when_omitted(self):
        apply_args_to_config(_parse(["train"]))
        config = get_config()
        assert config.training.round_data_dir is None
        assert config.training.overlap_data_generation is True
        assert config.training.keep_round_data is False
        assert config.training.data_gen_task_size == 256

    def test_data_gen_task_size_below_one_rejected(self):
        errors = collect_validation_errors(_parse(["train", "--data-gen-task-size", "0"]), None)
        assert any(
            e.field == "training.data_gen_task_size" and e.fix_kind == "clamp_low" for e in errors
        )

    def _patch_free_bytes(self, monkeypatch, free_bytes):
        usage = collections.namedtuple("usage", ["total", "used", "free"])
        monkeypatch.setattr(
            cli.shutil, "disk_usage", lambda path: usage(free_bytes * 2, free_bytes, free_bytes)
        )

    def test_disk_budget_error_when_insufficient(self, monkeypatch):
        config = get_config()
        round_nbytes = cli._estimate_round_data_nbytes(
            config.training.num_samples_beta_vae,
            config.data.num_observations,
            config.data.time_bins,
            config.data.width_bin // config.data.downsample_factor,
        )
        # Between 1.1x and 2.2x one round: fails with overlap (default), passes without
        self._patch_free_bytes(monkeypatch, int(1.5 * round_nbytes))

        errors = collect_validation_errors(_parse(["train"]), None)
        disk_errors = [e for e in errors if e.field == "training.round_data_dir"]
        assert len(disk_errors) == 1
        assert "GB free" in disk_errors[0].message

        errors = collect_validation_errors(_parse(["train", "--no-overlap-data-generation"]), None)
        assert not any(e.field == "training.round_data_dir" for e in errors)

    def test_disk_budget_ok_when_sufficient(self, monkeypatch):
        config = get_config()
        round_nbytes = cli._estimate_round_data_nbytes(
            config.training.num_samples_beta_vae,
            config.data.num_observations,
            config.data.time_bins,
            config.data.width_bin // config.data.downsample_factor,
        )
        self._patch_free_bytes(monkeypatch, int(10 * round_nbytes))
        errors = collect_validation_errors(_parse(["train"]), None)
        assert not any(e.field == "training.round_data_dir" for e in errors)

    def test_estimate_scales_with_sample_count(self):
        one = cli._estimate_round_data_nbytes(4, 6, 16, 512)
        two = cli._estimate_round_data_nbytes(8, 6, 16, 512)
        assert two == 2 * one
        # 3 arrays x n x 6 x 16 x 512 float32 + n x U20 labels
        assert one == 3 * 4 * 6 * 16 * 512 * 4 + 4 * 80

    def test_nearest_existing_ancestor(self, tmp_path):
        missing = tmp_path / "a" / "b" / "c"
        assert cli._nearest_existing_ancestor(str(missing)) == str(tmp_path)
        assert cli._nearest_existing_ancestor(str(tmp_path)) == str(tmp_path)
