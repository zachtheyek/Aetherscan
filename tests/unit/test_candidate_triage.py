"""Unit tests for aetherscan.candidate_triage (#395/#397): frequency-exclusion
partitioning semantics — report-time only, so correctness here is about tallies and
Slack surfaces, never about what a snippet scores."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from aetherscan.candidate_triage import (
    candidate_latent_matrix,
    frequency_excluded,
    mahalanobis_ood,
    normalized_frequency_ranges,
    partition_candidates_by_frequency,
    survey_ood_scores,
    training_ood_scores,
    triage_sort_rows,
)


class TestNormalizedFrequencyRanges:
    def test_none_and_empty_normalize_to_no_ranges(self):
        assert normalized_frequency_ranges(None) == []
        assert normalized_frequency_ranges([]) == []

    def test_pairs_normalize_sorted_as_float_tuples(self):
        ranges = normalized_frequency_ranges([[1616, 1626.5], [1575.0, 1576.0]])
        assert ranges == [(1575.0, 1576.0), (1616.0, 1626.5)]

    @pytest.mark.parametrize(
        "bad",
        [
            [[1626.5, 1616]],  # start >= end
            [[1616, 1616]],  # zero-width
            [[-5, 10]],  # non-positive start
            [[0, 10]],  # zero start
            [[float("inf"), 2000]],  # inf parses as a float on the CLI
            [[1616, float("nan")]],  # so does nan
            [["iridium", 1626.5]],  # non-numeric
            [[1616]],  # missing end
            ["1616", "1626.5"],  # flat string pair: '1616'[0] would index characters
        ],
    )
    def test_malformed_ranges_raise(self, bad):
        with pytest.raises(ValueError):
            normalized_frequency_ranges(bad)


class TestFrequencyExcluded:
    RANGES = [(1575.0, 1576.0), (1616.0, 1626.5)]

    @pytest.mark.parametrize("freq", [1575.0, 1576.0, 1575.42, 1616.0, 1626.5, 1620.0])
    def test_inside_and_boundary_excluded(self, freq):
        # Inclusive at both edges: a candidate exactly on an allocation boundary is the
        # allocation's
        assert frequency_excluded(freq, self.RANGES) is True

    @pytest.mark.parametrize("freq", [1574.999, 1576.001, 1615.9, 1626.6, 8438.0])
    def test_outside_reported(self, freq):
        assert frequency_excluded(freq, self.RANGES) is False

    def test_none_frequency_never_excluded(self):
        # No basis to filter -> reporting is the safe side
        assert frequency_excluded(None, self.RANGES) is False


class TestPartitionCandidatesByFrequency:
    def test_no_ranges_reports_everything(self):
        rows = [{"frequency_mhz": 1575.42}, {"frequency_mhz": 8438.0}]
        reported, excluded = partition_candidates_by_frequency(rows, [])
        assert reported == rows
        assert excluded == []

    def test_partition_preserves_order_within_sides(self):
        rows = [
            {"frequency_mhz": 1575.42, "id": "gps"},
            {"frequency_mhz": 8438.0, "id": "clean-1"},
            {"frequency_mhz": 1626.0, "id": "iridium"},
            {"frequency_mhz": None, "id": "no-freq"},
            {"frequency_mhz": 4091.0, "id": "clean-2"},
        ]
        ranges = [(1575.0, 1576.0), (1616.0, 1626.5)]
        reported, excluded = partition_candidates_by_frequency(rows, ranges)
        assert [r["id"] for r in reported] == ["clean-1", "no-freq", "clean-2"]
        assert [r["id"] for r in excluded] == ["gps", "iridium"]


class TestMahalanobisOod:
    def test_outlier_scores_high_inlier_low(self):
        rng = np.random.default_rng(11)
        reference = rng.normal(size=(500, 8))
        candidates = np.vstack([np.zeros(8), np.full(8, 25.0)])
        distances, percentiles = mahalanobis_ood(candidates, reference)
        assert distances[1] > distances[0]
        assert percentiles[1] == 100.0  # farther than every reference row
        assert percentiles[0] < 50.0  # the mean point is a deep inlier

    def test_degenerate_reference_does_not_raise(self):
        # Collapsed dimension + duplicated rows: the ridge + pinv must keep this finite
        reference = np.zeros((10, 4))
        reference[:, 0] = np.arange(10)
        distances, percentiles = mahalanobis_ood(np.ones((2, 4)), reference)
        assert np.isfinite(distances).all()
        assert np.isfinite(percentiles).all()


class TestCandidateLatentMatrix:
    def test_parses_json_payloads_and_skips_bad_rows(self):
        rows = [
            {"latent_vector": json.dumps([1.0, 2.0, 3.0])},
            {"latent_vector": None},  # skipped
            {"latent_vector": json.dumps([4.0, 5.0])},  # width mismatch -> skipped
            {"latent_vector": [6.0, 7.0, 8.0]},  # already a list
            {"latent_vector": "not json"},  # unparseable -> skipped
        ]
        matrix, indices = candidate_latent_matrix(rows)
        assert matrix.shape == (2, 3)
        assert indices == [0, 3]
        assert np.allclose(matrix[1], [6.0, 7.0, 8.0])

    def test_empty_rows(self):
        matrix, indices = candidate_latent_matrix([{"latent_vector": None}])
        assert matrix.size == 0
        assert indices == []

    def test_expected_width_filters_per_row_not_anchor_first(self):
        # One anomalous-width FIRST row must not veto the normal rows behind it when the
        # reference dimensionality is known (audit fix: anchor-first let a single corrupt
        # row kill every OOD column for the run)
        rows = [
            {"latent_vector": json.dumps([1.0, 2.0])},  # anomalous width
            {"latent_vector": json.dumps([1.0] * 4)},
            {"latent_vector": json.dumps([2.0] * 4)},
        ]
        matrix, indices = candidate_latent_matrix(rows, expected_width=4)
        assert matrix.shape == (2, 4)
        assert indices == [1, 2]


class TestTriageSortRows:
    def test_confidence_first_then_ood_then_mc_std(self):
        rows = [
            {"npy_path": "a", "snippet_index": 0, "confidence": 1.0, "mc_std": 0.01},
            {"npy_path": "a", "snippet_index": 1, "confidence": 1.0, "mc_std": 0.002},
            {"npy_path": "a", "snippet_index": 2, "confidence": 0.995, "mc_std": 0.001},
            {"npy_path": "a", "snippet_index": 3, "confidence": 1.0, "mc_std": None},
        ]
        # Snippet 1 is most survey-OOD -> reviews first within the P=1.0 tie; snippet 3
        # (no MC spread) sorts after snippet 0; lower confidence stays last regardless
        survey_ood = {("a", 1): (9.0, 99.9), ("a", 0): (1.0, 50.0), ("a", 3): (1.0, 50.0)}
        ordered = triage_sort_rows(rows, survey_ood)
        assert [r["snippet_index"] for r in ordered] == [1, 0, 3, 2]


class TestOodScoreSources:
    def test_survey_ood_requires_latent_mean(self, tmp_path):
        rows = [
            {"npy_path": "n", "snippet_index": 0, "latent_vector": json.dumps([0.0] * 8)},
        ]
        old_cloud = tmp_path / "cloud_old.npz"
        np.savez_compressed(old_cloud, mc_mean=np.zeros(5), mc_std=np.zeros(5))
        assert survey_ood_scores(rows, str(old_cloud)) == {}

        rng = np.random.default_rng(3)
        new_cloud = tmp_path / "cloud_new.npz"
        np.savez_compressed(new_cloud, latent_mean=rng.normal(size=(200, 8)))
        scores = survey_ood_scores(rows, str(new_cloud))
        assert ("n", 0) in scores
        distance, percentile = scores[("n", 0)]
        assert distance >= 0.0
        assert 0.0 <= percentile <= 100.0

    def test_survey_ood_missing_cloud_skips(self, tmp_path):
        assert survey_ood_scores([], str(tmp_path / "absent.npz")) == {}

    def test_training_ood_from_eval_artifacts(self, tmp_path):
        train_tag = "train_fake"
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"checkpoint": {"save_tag": train_tag}}))
        rng = np.random.default_rng(5)
        joblib.dump(
            {
                "latent_variant": "z_mean",
                "train_features": rng.normal(size=(300, 8)),
                "train_binary_labels": np.array([1] * 200 + [0] * 100),
            },
            tmp_path / f"rf_eval_artifacts_{train_tag}.joblib",
        )
        rows = [
            {"npy_path": "n", "snippet_index": 0, "latent_vector": json.dumps([0.1] * 8)},
        ]
        scores = training_ood_scores(rows, str(tmp_path), str(config_path))
        assert ("n", 0) in scores

    def test_training_ood_skips_non_z_mean_variant(self, tmp_path):
        train_tag = "train_fake"
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"checkpoint": {"save_tag": train_tag}}))
        joblib.dump(
            {
                "latent_variant": "z_mean_logvar",
                "train_features": np.zeros((10, 16)),
                "train_binary_labels": np.ones(10),
            },
            tmp_path / f"rf_eval_artifacts_{train_tag}.joblib",
        )
        rows = [{"npy_path": "n", "snippet_index": 0, "latent_vector": json.dumps([0.1] * 8)}]
        assert training_ood_scores(rows, str(tmp_path), str(config_path)) == {}

    def test_training_ood_missing_artifact_skips(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"checkpoint": {"save_tag": "train_missing"}}))
        assert training_ood_scores([], str(tmp_path), str(config_path)) == {}

    def test_training_ood_finds_display_tagged_artifact(self, tmp_path):
        # train.py names the artifact with the TRAINING host's display tag
        # (rf_eval_artifacts_train_{machine}_{datetime}.joblib); inference may run on a
        # different host, so the glob fallback must find it (audit blocker: the plain-tag
        # lookup missed every display-tagged artifact in production)
        train_tag = "train_20260729_152426"
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"checkpoint": {"save_tag": train_tag}}))
        rng = np.random.default_rng(5)
        joblib.dump(
            {
                "latent_variant": "z_mean",
                "train_features": rng.normal(size=(300, 8)),
                "train_binary_labels": np.array([1] * 200 + [0] * 100),
            },
            tmp_path / "rf_eval_artifacts_train_otherhost_20260729_152426.joblib",
        )
        rows = [{"npy_path": "n", "snippet_index": 0, "latent_vector": json.dumps([0.1] * 8)}]
        assert ("n", 0) in training_ood_scores(rows, str(tmp_path), str(config_path))

    def test_module_import_is_stdlib_only(self):
        # cli.py imports the range validator from this module, and the weekly docs
        # workflow imports cli.py on a bare interpreter with no numpy installed — the
        # MODULE import must not pull numpy (deferred into the OOD functions)
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415

        code = (
            "import sys; sys.modules['numpy'] = None; "
            "import aetherscan.candidate_triage; print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src"},
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout
