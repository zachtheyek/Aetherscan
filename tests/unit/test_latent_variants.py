"""Unit tests for aetherscan.latent_variants (#282): variant feature builders, active-unit
detection, selection metrics (recall@FPR, ECE, minimum-margin winner), and the probability
calibrator. TF-free by design."""

from __future__ import annotations

import numpy as np
import pytest

from aetherscan.latent_variants import (
    VARIANT_ORDER,
    active_latent_dims,
    apply_probability_calibrator,
    build_variant_features,
    build_z_aug_training_set,
    expected_calibration_error,
    fit_probability_calibrator,
    per_cadence_kl,
    recall_at_fpr,
    sample_z_flat,
    select_winner,
    variant_feature_count,
    variant_feature_names,
)

_NUM_OBS = 6
_LATENT_DIM = 8
_BASE = _NUM_OBS * _LATENT_DIM


def _blocks(n=10, seed=0):
    rng = np.random.default_rng(seed)
    z_mean = rng.normal(size=(n, _BASE)).astype(np.float32)
    z_log_var = rng.normal(-2, 0.3, size=(n, _BASE)).astype(np.float32)
    return z_mean, z_log_var


class TestFeatureBuilders:
    @pytest.mark.parametrize("variant", VARIANT_ORDER)
    def test_feature_count_matches_declared(self, variant):
        z_mean, z_log_var = _blocks()
        active = [0, 3, 5]
        features = build_variant_features(variant, z_mean, z_log_var, _NUM_OBS, _LATENT_DIM, active)
        assert features.shape == (
            10,
            variant_feature_count(variant, _NUM_OBS, _LATENT_DIM, active),
        )

    def test_lead_block_always_first(self):
        # Layout contract: the lead (z_mean or draw) block occupies the first BASE columns
        z_mean, z_log_var = _blocks()
        for variant in VARIANT_ORDER:
            features = build_variant_features(
                variant, z_mean, z_log_var, _NUM_OBS, _LATENT_DIM, [0]
            )
            np.testing.assert_array_equal(features[:, :_BASE], z_mean)

    def test_total_kl_column_matches_closed_form(self):
        z_mean, z_log_var = _blocks()
        features = build_variant_features(
            "z_mean_total_kl", z_mean, z_log_var, _NUM_OBS, _LATENT_DIM
        )
        expected = -0.5 * (
            1
            + z_log_var.astype(np.float64)
            - z_mean.astype(np.float64) ** 2
            - np.exp(z_log_var.astype(np.float64))
        )
        np.testing.assert_allclose(features[:, -1], expected.sum(axis=1), rtol=1e-5)
        np.testing.assert_allclose(
            per_cadence_kl(z_mean, z_log_var), expected.sum(axis=1), rtol=1e-5
        )

    def test_obs_and_dim_aggregates(self):
        z_mean, z_log_var = _blocks()
        reshaped = z_log_var.reshape(-1, _NUM_OBS, _LATENT_DIM)
        obs_features = build_variant_features(
            "z_mean_obs_logvar", z_mean, z_log_var, _NUM_OBS, _LATENT_DIM
        )
        np.testing.assert_allclose(obs_features[:, _BASE:], reshaped.mean(axis=2), rtol=1e-6)
        dim_features = build_variant_features(
            "z_mean_dim_logvar", z_mean, z_log_var, _NUM_OBS, _LATENT_DIM
        )
        np.testing.assert_allclose(dim_features[:, _BASE:], reshaped.mean(axis=1), rtol=1e-6)

    def test_active_variant_selects_the_right_columns(self):
        z_mean, z_log_var = _blocks()
        active = [1, 4]
        features = build_variant_features(
            "z_mean_logvar_active", z_mean, z_log_var, _NUM_OBS, _LATENT_DIM, active
        )
        expected_cols = [
            obs * _LATENT_DIM + dim for obs in range(_NUM_OBS) for dim in sorted(active)
        ]
        np.testing.assert_array_equal(features[:, _BASE:], z_log_var[:, expected_cols])

    def test_active_variant_degenerates_to_z_mean_when_all_collapsed(self):
        z_mean, z_log_var = _blocks()
        features = build_variant_features(
            "z_mean_logvar_active", z_mean, z_log_var, _NUM_OBS, _LATENT_DIM, []
        )
        np.testing.assert_array_equal(features, z_mean)

    def test_unknown_variant_raises(self):
        z_mean, z_log_var = _blocks()
        with pytest.raises(ValueError, match="Unknown latent variant"):
            build_variant_features("nope", z_mean, z_log_var, _NUM_OBS, _LATENT_DIM)

    def test_sample_z_flat_is_seeded_and_scales_with_log_var(self):
        z_mean, z_log_var = _blocks()
        draw_a = sample_z_flat(z_mean, z_log_var, np.random.default_rng(5))
        draw_b = sample_z_flat(z_mean, z_log_var, np.random.default_rng(5))
        np.testing.assert_array_equal(draw_a, draw_b)
        # Negligible variance collapses the draw onto the mean
        tight = sample_z_flat(z_mean, np.full_like(z_log_var, -100.0), np.random.default_rng(5))
        np.testing.assert_allclose(tight, z_mean, atol=1e-5)

    def test_z_aug_training_set_shapes_and_label_replication(self):
        z_mean, z_log_var = _blocks(n=4)
        labels = np.array([0, 1, 0, 1])
        features, stacked = build_z_aug_training_set(
            z_mean, z_log_var, labels, draws=3, rng=np.random.default_rng(0)
        )
        assert features.shape == (16, _BASE)  # (draws + 1) * n rows
        np.testing.assert_array_equal(stacked, np.tile(labels, 4))
        np.testing.assert_array_equal(features[:4], z_mean)  # deterministic row group first


class TestActiveLatentDims:
    def test_detects_collapsed_dims(self):
        rng = np.random.default_rng(0)
        n = 200
        per_obs = rng.normal(size=(n * _NUM_OBS, _LATENT_DIM)).astype(np.float32)
        per_obs[:, 2] = 0.0  # collapsed: zero variance
        per_obs[:, 6] = 0.05  # collapsed: constant
        flat = per_obs.reshape(n, _BASE)
        active = active_latent_dims(flat, _NUM_OBS, _LATENT_DIM, threshold=0.01)
        assert 2 not in active and 6 not in active
        assert len(active) == _LATENT_DIM - 2


class TestSelectionMetrics:
    def test_recall_at_fpr_perfect_separation(self):
        labels = np.array([0] * 100 + [1] * 50)
        scores = np.concatenate([np.linspace(0, 0.4, 100), np.linspace(0.6, 1.0, 50)])
        assert recall_at_fpr(labels, scores, 0.01) == 1.0

    def test_recall_at_fpr_single_class_is_nan(self):
        assert np.isnan(recall_at_fpr(np.ones(10), np.linspace(0, 1, 10), 0.01))

    def test_ece_zero_for_perfectly_calibrated_bins(self):
        rng = np.random.default_rng(1)
        probas = rng.uniform(0.05, 0.95, size=20_000)
        labels = (rng.uniform(size=20_000) < probas).astype(int)
        assert expected_calibration_error(labels, probas) < 0.02

    def test_select_winner_prefers_simpler_on_tie(self):
        rng = np.random.default_rng(2)
        labels = np.array([0] * 400 + [1] * 100)
        base = np.concatenate([rng.uniform(0, 0.5, 400), rng.uniform(0.5, 1.0, 100)])
        # Identical scores => statistically tied => the simpler (earlier-ordered) wins
        scores = {"z_mean": base, "z_mean_logvar": base.copy()}
        winner, recalls = select_winner(labels, scores, 0.05, 50, np.random.default_rng(3))
        assert winner == "z_mean"
        assert set(recalls) == {"z_mean", "z_mean_logvar"}

    def test_select_winner_takes_clear_improvement(self):
        rng = np.random.default_rng(4)
        labels = np.array([0] * 400 + [1] * 100)
        weak = np.concatenate([rng.uniform(0, 0.6, 400), rng.uniform(0.3, 0.9, 100)])
        strong = np.concatenate([rng.uniform(0, 0.3, 400), rng.uniform(0.7, 1.0, 100)])
        winner, _ = select_winner(
            labels, {"z_mean": weak, "z_mean_logvar": strong}, 0.05, 100, np.random.default_rng(5)
        )
        assert winner == "z_mean_logvar"


class TestCalibrator:
    def test_isotonic_when_large_and_improves_ece(self):
        rng = np.random.default_rng(6)
        n = 5000
        # Systematically overconfident scores: true P(y=1) = raw**2
        raw = rng.uniform(size=n)
        labels = (rng.uniform(size=n) < raw**2).astype(int)
        calibrator = fit_probability_calibrator(raw, labels, min_isotonic=1000)
        assert calibrator["method"] == "isotonic"
        calibrated = apply_probability_calibrator(calibrator, raw)
        assert expected_calibration_error(labels, calibrated) < expected_calibration_error(
            labels, raw
        )

    def test_sigmoid_below_isotonic_floor(self):
        rng = np.random.default_rng(7)
        raw = rng.uniform(size=200)
        labels = (rng.uniform(size=200) < raw).astype(int)
        calibrator = fit_probability_calibrator(raw, labels, min_isotonic=1000)
        assert calibrator["method"] == "sigmoid"
        calibrated = apply_probability_calibrator(calibrator, raw)
        assert calibrated.shape == raw.shape
        assert np.all((calibrated >= 0) & (calibrated <= 1))

    def test_identity_when_none(self):
        raw = np.linspace(0, 1, 11)
        np.testing.assert_array_equal(apply_probability_calibrator(None, raw), raw)

    def test_ece_hand_computed_quantile_bins(self):
        # 4 probabilities, 2 quantile bins of 2: bin1 = {0.1, 0.2} (mean 0.15, frac pos 0.5),
        # bin2 = {0.8, 0.9} (mean 0.85, frac pos 0.5) ->
        # ECE = 0.5*|0.5-0.15| + 0.5*|0.5-0.85| = 0.35
        labels = np.array([0, 1, 0, 1])
        probas = np.array([0.1, 0.2, 0.8, 0.9])
        assert expected_calibration_error(labels, probas, n_bins=2) == pytest.approx(0.35)


class TestVariantFeatureNames:
    """variant_feature_names must stay column-for-column in lockstep with
    build_variant_features — a length mismatch is an IndexError inside shap's plots
    (how the hardcoded 48-name list broke on a 54-feature z_mean_obs_logvar winner)."""

    @pytest.mark.parametrize("variant", VARIANT_ORDER)
    @pytest.mark.parametrize("active_dims", [[], [1, 3], [0, 1, 2, 3, 4, 5, 6, 7]])
    def test_names_match_feature_columns(self, variant, active_dims):
        num_obs, latent_dim, n = 6, 8, 4
        rng = np.random.default_rng(11)
        lead = rng.normal(size=(n, num_obs * latent_dim)).astype(np.float32)
        logvar = rng.normal(size=(n, num_obs * latent_dim)).astype(np.float32)
        features = build_variant_features(
            variant, lead, logvar, num_obs, latent_dim, active_dims=active_dims
        )
        names = variant_feature_names(variant, num_obs, latent_dim, active_dims=active_dims)
        assert len(names) == features.shape[1]
        assert len(set(names)) == len(names)  # no duplicate column names

    def test_lead_block_matches_historical_convention(self):
        names = variant_feature_names("z_mean", 6, 8)
        assert names[0] == "ON-1_dim-0"
        assert names[8] == "OFF-1_dim-0"
        assert names[-1] == "OFF-3_dim-7"
