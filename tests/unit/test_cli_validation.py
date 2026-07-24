# NOTE: come back to this later

"""Unit tests for aetherscan.cli: tag pattern, validation matrix, cross-param solver, and
apply_saved_config precedence (defaults < saved config < CLI args)."""

from __future__ import annotations

import collections
import json

import pytest

from aetherscan import cli
from aetherscan.cli import (
    _LOAD_TAG_PATTERN,
    _SAVE_TAG_PATTERN,
    _build_suggestion_block,
    _check_cross_constraints,
    _solve_cross_param_constraints,
    apply_args_to_config,
    apply_saved_config,
    collect_validation_errors,
    resolve_save_tag,
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
    @pytest.mark.parametrize("tag", ["test", "train", "inf", "bench"])
    def test_save_tag_accepted_prefixes(self, tag):
        assert _SAVE_TAG_PATTERN.match(tag)

    @pytest.mark.parametrize(
        "tag",
        ["final_v1", "test_v1", "round_01", "train_20260101_000000", "TRAIN", " train", "", "prod"],
    )
    def test_save_tag_rejected(self, tag):
        # --save-tag is a bare command prefix only; a full/round tag or old-scheme tag is rejected.
        assert not _SAVE_TAG_PATTERN.match(tag)

    @pytest.mark.parametrize(
        "tag",
        [
            "train_20260712_123456",
            "test_20260101_000000",
            "inf_20991231_235959",
            "bench_20260712_123456",
            "round_01",
            "round_5",
        ],
    )
    def test_load_tag_accepted(self, tag):
        assert _LOAD_TAG_PATTERN.match(tag)

    @pytest.mark.parametrize(
        "tag",
        [
            "final_v1",  # old scheme, retired
            "test_v1",  # old scheme
            "20260712_123456",  # bare datetime, retired
            "train",  # a bare prefix isn't a full load tag
            "train_2026",  # incomplete datetime
            "TRAIN_20260101_000000",  # wrong case
            "20260712-123456",  # wrong separator
            "round_",  # missing round number
            " train_20260101_000000",  # leading whitespace
            "",
        ],
    )
    def test_load_tag_rejected(self, tag):
        assert not _LOAD_TAG_PATTERN.match(tag)


class TestResolveSaveTag:
    def test_prefix_gets_datetime_appended(self):
        tag = resolve_save_tag("train", "test", None)
        assert tag.startswith("test_") and _LOAD_TAG_PATTERN.match(tag)

    def test_omitted_defaults_to_subcommand(self):
        assert resolve_save_tag("train", None, None).startswith("train_")
        assert resolve_save_tag("inference", None, None).startswith("inf_")

    def test_full_load_tag_is_adopted_verbatim(self):
        # A full {cmd}_{datetime} --load-tag resumes that run in place → its tag is adopted.
        assert (
            resolve_save_tag("train", "train", "train_20260101_120000") == "train_20260101_120000"
        )

    def test_round_load_tag_does_not_adopt(self):
        # round_XX seeds a fresh run — the save-tag is freshly stamped, not adopted.
        tag = resolve_save_tag("train", "train", "round_05")
        assert tag.startswith("train_") and tag != "round_05"


class TestCrossReplicaDivisibilityMatrix:
    """The default config divides cleanly across 4, 5, AND 6 replicas (issue #254 — Option B
    defaults); the blpc3 smoke config is valid for 5 only. collect_validation_errors must
    reproduce that matrix exactly."""

    @pytest.mark.parametrize("num_replicas", [4, 5, 6])
    def test_defaults_valid_for_4_5_6_replicas(self, num_replicas):
        assert _cross_param_errors(_parse(["train"]), num_replicas) == []

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

    def test_round_load_tag_without_checkpoints_load_dir_rejected(self):
        # The #142 resume footgun: per-round checkpoints live under checkpoints/, so a bare
        # `--load-tag round_XX` would search the models root and silently load a stale model.
        errors = collect_validation_errors(_parse(["train", "--load-tag", "round_01"]), None)
        assert any(e.field == "checkpoint.load_dir" and e.fix_kind == "cross_param" for e in errors)

    def test_round_load_tag_with_checkpoints_load_dir_passes(self):
        errors = collect_validation_errors(
            _parse(["train", "--load-tag", "round_01", "--load-dir", "checkpoints"]), None
        )
        assert not any(e.field == "checkpoint.load_dir" for e in errors)

    def test_full_load_tag_without_load_dir_passes(self):
        # A full run tag's final model legitimately lives in the models root — no load-dir required.
        errors = collect_validation_errors(
            _parse(["train", "--load-tag", "train_20260101_120000"]), None
        )
        assert not any(e.field == "checkpoint.load_dir" for e in errors)

    def test_num_samples_divisible_by_4(self):
        errors = collect_validation_errors(_parse(["train", "--num-samples-beta-vae", "202"]), None)
        assert any(e.field == "training.num_samples_beta_vae" and e.divisor == 4 for e in errors)

    def test_negative_seed_rejected(self):
        errors = collect_validation_errors(_parse(["train", "--seed", "-1"]), None)
        assert any(e.field == "training.seed" and e.fix_kind == "clamp_low" for e in errors)

    @pytest.mark.parametrize("seed", ["0", "42"])
    def test_non_negative_seed_passes(self, seed):
        errors = collect_validation_errors(_parse(["train", "--seed", seed]), None)
        assert not any(e.field == "training.seed" for e in errors)

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

    def test_stamp_gallery_top_k_bounds(self):
        errors = collect_validation_errors(
            _parse(["inference", "--stamp-gallery-top-k", "0"]), None
        )
        assert any(e.field == "inference.stamp_gallery_top_k" for e in errors)

    def test_max_candidate_plots_bounds(self):
        errors = collect_validation_errors(
            _parse(["inference", "--max-candidate-plots", "-1"]), None
        )
        assert any(e.field == "inference.max_candidate_plots" for e in errors)

    def test_viz_flags_route_to_inference_config(self):
        config = get_config()
        apply_args_to_config(
            _parse(
                [
                    "inference",
                    "--no-inference-viz",
                    "--stamp-gallery-top-k",
                    "6",
                    "--max-candidate-plots",
                    "10",
                ]
            )
        )
        assert config.inference.inference_viz_enabled is False
        assert config.inference.stamp_gallery_top_k == 6
        assert config.inference.max_candidate_plots == 10

    def test_viz_flags_omitted_keep_defaults(self):
        config = get_config()
        apply_args_to_config(_parse(["inference"]))
        assert config.inference.inference_viz_enabled is True
        assert config.inference.stamp_gallery_top_k == 12
        assert config.inference.max_candidate_plots == 50

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

    def test_inference_artifact_trio_omitted_is_valid(self):
        # No local artifact paths -> resolution is handled upstream (HF download in main.py),
        # so validation must not demand them.
        errors = collect_validation_errors(_parse(["inference"]), None)
        artifact_fields = {"inference.encoder_path", "inference.rf_path", "inference.config_path"}
        assert not artifact_fields & {e.field for e in errors}

    def test_inference_artifact_partial_trio_rejected(self, tmp_path):
        encoder = tmp_path / "vae_encoder_test_v1.keras"
        encoder.touch()
        errors = collect_validation_errors(
            _parse(["inference", "--encoder-path", str(encoder)]), None
        )
        fields = {e.field for e in errors if e.fix_kind == "file_exists"}
        # The two missing paths are reported; the provided one passes.
        assert {"inference.rf_path", "inference.config_path"} <= fields
        assert "inference.encoder_path" not in fields

    def test_inference_artifact_partial_trio_from_config_rejected(self, tmp_path):
        # The trio check must count paths sourced from a loaded saved config (the
        # `_resolve(args, ..., config.inference.*)` fallback), not just CLI flags: a
        # config-side encoder_path with no flags is still a partial trio.
        encoder = tmp_path / "vae_encoder_test_v1.keras"
        encoder.touch()
        get_config().inference.encoder_path = str(encoder)
        errors = collect_validation_errors(_parse(["inference"]), None)
        fields = {e.field for e in errors if e.fix_kind == "file_exists"}
        assert {"inference.rf_path", "inference.config_path"} <= fields
        assert "inference.encoder_path" not in fields

    def test_inference_artifact_full_trio_must_exist_on_disk(self):
        errors = collect_validation_errors(
            _parse(
                [
                    "inference",
                    "--encoder-path",
                    "/nonexistent/e.keras",
                    "--rf-path",
                    "/nonexistent/r.joblib",
                    "--config-path",
                    "/nonexistent/c.json",
                ]
            ),
            None,
        )
        fields = {e.field for e in errors if e.fix_kind == "file_exists"}
        assert {
            "inference.encoder_path",
            "inference.rf_path",
            "inference.config_path",
        } <= fields

    @pytest.mark.parametrize("command", ["train", "inference"])
    def test_hf_repo_id_format_rejected(self, command):
        errors = collect_validation_errors(_parse([command, "--hf-repo-id", "not-a-repo-id"]), None)
        assert any(e.field == "hf.repo_id" for e in errors)

    @pytest.mark.parametrize("repo_id", ["zachtheyek/aetherscan", "org-name/repo.name-1"])
    def test_hf_repo_id_valid_values_pass(self, repo_id):
        errors = collect_validation_errors(_parse(["train", "--hf-repo-id", repo_id]), None)
        assert [e for e in errors if e.field == "hf.repo_id"] == []

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

    def test_inference_stamp_width_must_be_divisible_by_downsample_factor(self, make_inference_csv):
        # store_downsampled_stamps defaults on, so the stored width (stamp_width //
        # downsample_factor) must be exact; a stamp_width indivisible by the default
        # downsample_factor (8) is rejected. Distinct from the equality check above, which the
        # 2048 case only exercised because 2048 happens to divide 8.
        csv_path = make_inference_csv("subset.csv")
        argv = [
            "inference",
            "--inference-files",
            csv_path.name,
            "--stamp-width",
            "4095",  # not a multiple of the default downsample_factor (8)
        ]
        errors = collect_validation_errors(_parse(argv), None)
        assert any("divisible by" in e.message and "downsample-factor" in e.message for e in errors)

    def test_colliding_stamp_width_fixes_both_survive_in_suggestions(self, make_inference_csv):
        # --stamp-width 2047 violates BOTH stamp_width checks (equality with the default
        # width_bin 4096 and divisibility by the default downsample_factor 8) on the same
        # field. The suggestion block keys proposals on (field, fix_kind), so both proposed
        # fixes must appear — 4096 from the equality fix, 2048 from the divisibility fix —
        # instead of one silently overwriting the other.
        csv_path = make_inference_csv("subset.csv")
        argv = ["inference", "--inference-files", csv_path.name, "--stamp-width", "2047"]
        errors = collect_validation_errors(_parse(argv), None)
        stamp_errors = [e for e in errors if e.field == "inference.stamp_width"]
        assert len(stamp_errors) == 2

        block = _build_suggestion_block(errors, None)
        assert "--stamp-width 4096" in block
        assert "--stamp-width 2048" in block

    def test_bandpass_method_enum(self):
        errors = collect_validation_errors(
            _parse(["inference", "--bandpass-method", "median"]), None
        )
        assert any(e.field == "inference.bandpass_method" for e in errors)

    @pytest.mark.parametrize("method", ["pfb", "spline"])
    def test_bandpass_method_valid_values_pass(self, method):
        errors = collect_validation_errors(_parse(["inference", "--bandpass-method", method]), None)
        assert [e for e in errors if e.field == "inference.bandpass_method"] == []

    def test_pfb_taps_per_channel_must_be_positive(self):
        errors = collect_validation_errors(
            _parse(["inference", "--pfb-taps-per-channel", "0"]), None
        )
        assert any(e.field == "inference.pfb_taps_per_channel" for e in errors)


class TestCrossParamSolver:
    # A config valid for 4 and 6 replicas but not 5 — the shape of the pre-Option-B defaults.
    _BASE_4_6 = {
        "num_samples_beta_vae": 499200,
        "num_samples_rf": 99840,
        "train_val_split": 0.8,
        "per_replica_batch_size": 128,
        "effective_batch_size": 3072,
        "per_replica_val_batch_size": 80,
    }
    _LATENT_TOTAL = 960  # latent_viz_num_cadences_per_type (240) * 4

    def test_check_cross_constraints_matrix(self):
        assert _check_cross_constraints(**self._BASE_4_6, num_replicas_list=[4])
        assert _check_cross_constraints(**self._BASE_4_6, num_replicas_list=[6])
        assert not _check_cross_constraints(**self._BASE_4_6, num_replicas_list=[5])

    def test_check_cross_constraints_latent_is_optional_and_enforced(self):
        # per_replica_val_batch_size 13 -> global 78 on 6 GPUs, which divides the val split and
        # num_samples_rf (both 99840) but NOT latent_viz*4 = 960.
        base = {**self._BASE_4_6, "per_replica_val_batch_size": 13}
        assert _check_cross_constraints(**base, num_replicas_list=[6])  # latent skipped by default
        assert not _check_cross_constraints(**base, num_replicas_list=[6], latent_total=960)

    def test_solver_returns_base_when_already_valid(self):
        assert _solve_cross_param_constraints(self._BASE_4_6, [4, 6]) == self._BASE_4_6

    def test_solver_keeps_data_and_batch_when_fixing_5_replicas(self):
        # The signature failure (valid for 6, not 5): the solver must keep the data sizes and the
        # throughput-optimal per-replica batch, moving only the two divisibility-bound batches.
        base = {**self._BASE_4_6, "latent_total": self._LATENT_TOTAL}
        sol = _solve_cross_param_constraints(base, [5])
        assert sol is not None
        assert sol["num_samples_beta_vae"] == base["num_samples_beta_vae"]
        assert sol["num_samples_rf"] == base["num_samples_rf"]
        assert sol["per_replica_batch_size"] == base["per_replica_batch_size"]
        assert sol["train_val_split"] == base["train_val_split"]
        assert _check_cross_constraints(
            **sol, num_replicas_list=[5], latent_total=self._LATENT_TOTAL
        )

    def test_solver_is_latent_aware(self):
        # Every proposed global val batch must divide latent_total when it is supplied.
        base = {**self._BASE_4_6, "effective_batch_size": 3070, "latent_total": self._LATENT_TOTAL}
        sol = _solve_cross_param_constraints(base, [5])
        assert sol is not None
        assert self._LATENT_TOTAL % (sol["per_replica_val_batch_size"] * 5) == 0

    def test_solver_satisfies_all_requested_replica_counts(self):
        # A multi-count solve must hold for every requested replica count simultaneously.
        base = {**self._BASE_4_6, "effective_batch_size": 3070, "latent_total": self._LATENT_TOTAL}
        sol = _solve_cross_param_constraints(base, [4, 5, 6])
        assert sol is not None
        fields = {k: sol[k] for k in self._BASE_4_6}
        for nr in (4, 5, 6):
            assert _check_cross_constraints(
                **fields, num_replicas_list=[nr], latent_total=self._LATENT_TOTAL
            )

    def test_solver_respects_candidate_budget(self):
        # The budget is a backstop; zero means bail before checking any candidate.
        base = {**self._BASE_4_6, "effective_batch_size": 3070}
        assert _solve_cross_param_constraints(base, [5], max_candidates=0) is None

    def test_solver_picks_the_l1_nearest_not_the_first_valid(self):
        # Strict L1-minimality guard: for the 5-GPU fix, keeping the data sizes + per-replica batch,
        # the nearest valid effective batch to 3072 is 2560 (2560 divides the 399360 train split and
        # is a multiple of 128*5=640; 3840 is the next one up and is farther), and the val batch lands
        # 16 from 80 (64 or 96). A refactor that returned the first valid candidate instead of the
        # closest would drift off these exact values.
        base = {**self._BASE_4_6, "latent_total": self._LATENT_TOTAL}
        sol = _solve_cross_param_constraints(base, [5])
        assert sol is not None
        assert sol["per_replica_batch_size"] == 128
        assert sol["effective_batch_size"] == 2560
        assert abs(sol["per_replica_val_batch_size"] - 80) == 16


class TestApplySavedConfigPrecedence:
    def test_saved_config_overrides_defaults(self, tmp_path):
        saved = {
            "training": {"num_training_rounds": 7, "snr_base": 33},
            "data_path": "/saved/data/path",
        }
        path = tmp_path / "saved_config.json"
        path.write_text(json.dumps(saved))

        apply_saved_config(str(path))
        config = get_config()
        assert config.training.num_training_rounds == 7
        assert config.training.snr_base == 33
        assert config.data_path == "/saved/data/path"

    def test_checkpoint_section_is_never_applied(self, tmp_path):
        """Regression: a saved *training* config's checkpoint section (most damagingly
        save_tag) must never leak onto the singleton — an inference run layering the
        training config would otherwise masquerade under the training run's tag."""
        config = get_config()
        config.checkpoint.save_tag = "20260712_010203"  # this run's own tag
        original_load_dir = config.checkpoint.load_dir
        original_load_tag = config.checkpoint.load_tag
        original_start_round = config.checkpoint.start_round

        saved = {
            "checkpoint": {
                "save_tag": "final_v9",
                "load_dir": "/somewhere/else",
                "load_tag": "round_05",
                "start_round": 6,
            },
            "beta_vae": {"latent_dim": 16},
        }
        path = tmp_path / "saved_config.json"
        path.write_text(json.dumps(saved))
        apply_saved_config(str(path))

        assert config.checkpoint.save_tag == "20260712_010203"
        assert config.checkpoint.load_dir == original_load_dir
        assert config.checkpoint.load_tag == original_load_tag
        assert config.checkpoint.start_round == original_start_round
        # Non-checkpoint sections still layer normally.
        assert config.beta_vae.latent_dim == 16

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

    def test_seed_flags_apply_to_training_config(self):
        config = get_config()
        assert config.training.seed is None  # default: OS entropy
        assert config.training.tf_deterministic_ops is False
        apply_args_to_config(_parse(["train", "--seed", "123", "--tf-deterministic-ops"]))
        assert config.training.seed == 123
        assert config.training.tf_deterministic_ops is True
        # Omitting both flags leaves the applied values untouched (None-guarded application)
        apply_args_to_config(_parse(["train"]))
        assert config.training.seed == 123
        assert config.training.tf_deterministic_ops is True
        apply_args_to_config(_parse(["train", "--no-tf-deterministic-ops"]))
        assert config.training.tf_deterministic_ops is False

    def test_bandpass_flags_apply_to_inference_config(self):
        config = get_config()
        # Preset True so --no-bandpass-debug-plot has something to force off.
        config.inference.bandpass_debug_plot = True
        apply_args_to_config(
            _parse(
                [
                    "inference",
                    "--bandpass-method",
                    "spline",
                    "--pfb-taps-per-channel",
                    "8",
                    "--no-bandpass-debug-plot",
                ]
            )
        )
        assert config.inference.bandpass_method == "spline"
        assert config.inference.pfb_taps_per_channel == 8
        assert config.inference.bandpass_debug_plot is False

    def test_bandpass_debug_plot_omitted_preserves_default(self):
        # BooleanOptionalAction default=None: omitting the flag must not clobber the config
        # value (the "is not None" apply guard is what preserves it).
        config = get_config()
        config.inference.bandpass_debug_plot = True
        apply_args_to_config(_parse(["inference"]))
        assert config.inference.bandpass_debug_plot is True

    def test_hf_flags_route_to_hf_section(self):
        config = get_config()
        assert config.hf.upload_after_training is False  # default: local-only
        apply_args_to_config(_parse(["train", "--hf-upload", "--hf-repo-id", "other/repo"]))
        assert config.hf.upload_after_training is True
        assert config.hf.repo_id == "other/repo"

        apply_args_to_config(_parse(["train", "--no-hf-upload"]))
        assert config.hf.upload_after_training is False

        apply_args_to_config(_parse(["inference", "--hf-revision", "v0.1.0"]))
        assert config.hf.revision == "v0.1.0"

    def test_force_tag_routes_to_checkpoint_section(self):
        config = get_config()
        assert config.checkpoint.force_tag is False
        apply_args_to_config(_parse(["train", "--force-tag"]))
        assert config.checkpoint.force_tag is True
        # Omitting the flag leaves the config value untouched (tri-state BooleanOptionalAction)
        apply_args_to_config(_parse(["inference"]))
        assert config.checkpoint.force_tag is True

    def test_benchmark_report_flag_routes_to_monitor_section(self):
        config = get_config()
        assert config.monitor.benchmark_report_enabled is True  # default: on
        apply_args_to_config(_parse(["train", "--no-benchmark-report"]))
        assert config.monitor.benchmark_report_enabled is False
        # Omitting the flag leaves the config value untouched (tri-state BooleanOptionalAction)
        apply_args_to_config(_parse(["inference"]))
        assert config.monitor.benchmark_report_enabled is False
        apply_args_to_config(_parse(["inference", "--benchmark-report"]))
        assert config.monitor.benchmark_report_enabled is True


class TestLatentTraversalFlags:
    """Flags and validation for the latent-dimension traversal plot (PLAN PR-05)."""

    def test_flags_apply_to_config(self):
        apply_args_to_config(
            _parse(
                [
                    "train",
                    "--latent-traversal-every-round",
                    "--latent-traversal-num-steps",
                    "9",
                    "--latent-traversal-max-sigma",
                    "2.5",
                ]
            )
        )
        config = get_config()
        assert config.training.latent_traversal_every_round is True
        assert config.training.latent_traversal_num_steps == 9
        assert config.training.latent_traversal_max_sigma == 2.5

    def test_defaults_preserved_when_omitted(self):
        apply_args_to_config(_parse(["train"]))
        config = get_config()
        assert config.training.latent_traversal_every_round is False
        assert config.training.latent_traversal_num_steps == 7
        assert config.training.latent_traversal_max_sigma == 3.0

    def test_no_flag_forces_off(self):
        apply_args_to_config(_parse(["train", "--no-latent-traversal-every-round"]))
        assert get_config().training.latent_traversal_every_round is False

    @staticmethod
    def _traversal_errors(argv):
        return [
            e
            for e in collect_validation_errors(_parse(argv), None)
            if e.field.startswith("training.latent_traversal")
        ]

    @pytest.mark.parametrize("steps", [3, 5, 7, 9])
    def test_odd_step_counts_accepted(self, steps):
        assert self._traversal_errors(["train", "--latent-traversal-num-steps", str(steps)]) == []

    @pytest.mark.parametrize("steps", [-1, 0, 1, 2, 4, 6, 8])
    def test_even_or_too_small_step_counts_rejected(self, steps):
        errors = self._traversal_errors(["train", "--latent-traversal-num-steps", str(steps)])
        assert len(errors) == 1
        assert errors[0].field == "training.latent_traversal_num_steps"
        assert "odd" in errors[0].message

    @pytest.mark.parametrize("max_sigma", ["0.5", "1", "3.0", "10"])
    def test_positive_max_sigma_accepted(self, max_sigma):
        assert self._traversal_errors(["train", "--latent-traversal-max-sigma", max_sigma]) == []

    @pytest.mark.parametrize("max_sigma", ["0", "0.0", "-1.5"])
    def test_nonpositive_max_sigma_rejected(self, max_sigma):
        errors = self._traversal_errors(["train", "--latent-traversal-max-sigma", max_sigma])
        assert len(errors) == 1
        assert errors[0].field == "training.latent_traversal_max_sigma"

    def test_defaults_pass_validation(self):
        assert self._traversal_errors(["train"]) == []


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
        assert config.training.data_gen_task_size == 64

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


class TestMaxRetriesValidation:
    """--max-retries must be >= 1: 0 would run zero attempts and exit 0 having trained nothing."""

    def test_zero_rejected(self):
        errors = collect_validation_errors(_parse(["train", "--max-retries", "0"]), None)
        assert any(
            e.field == "training.max_retries" and e.fix_kind == "clamp_low" and e.min_val == 1
            for e in errors
        )

    def test_negative_rejected(self):
        errors = collect_validation_errors(_parse(["train", "--max-retries", "-2"]), None)
        assert any(e.field == "training.max_retries" for e in errors)

    def test_one_accepted(self):
        errors = collect_validation_errors(_parse(["train", "--max-retries", "1"]), None)
        assert not any(e.field == "training.max_retries" for e in errors)
