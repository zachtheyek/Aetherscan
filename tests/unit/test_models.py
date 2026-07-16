# NOTE: come back to this later

"""Unit tests for aetherscan.models: latent feature layout, RandomForestModel behavior, the
Sampling layer, and encoder/decoder symmetry."""

from __future__ import annotations

import numpy as np
import pytest

from aetherscan.config import get_config
from aetherscan.models import RandomForestModel, Sampling, prepare_latent_features


class TestPrepareLatentFeatures:
    def test_row_major_layout(self):
        """features[i] must be the row-major concatenation of cadence i's 6 latents."""
        n_cadences, num_obs, latent_dim = 3, 6, 4
        # Encode (cadence, obs, dim) into each value so misordering is unambiguous.
        latents = np.zeros((n_cadences * num_obs, latent_dim))
        for cad in range(n_cadences):
            for obs in range(num_obs):
                for dim in range(latent_dim):
                    latents[cad * num_obs + obs, dim] = cad * 1000 + obs * 10 + dim

        features = prepare_latent_features(latents, num_observations=num_obs)

        assert features.shape == (n_cadences, num_obs * latent_dim)
        for cad in range(n_cadences):
            expected = np.concatenate([latents[cad * num_obs + obs] for obs in range(num_obs)])
            np.testing.assert_array_equal(features[cad], expected)
            # Spot-check the row-major invariant: feature column obs*latent_dim + dim.
            assert features[cad, 1 * latent_dim + 2] == cad * 1000 + 10 + 2

    def test_indivisible_row_count_raises(self):
        with pytest.raises(ValueError, match="Not divisible"):
            prepare_latent_features(np.zeros((7, 8)), num_observations=6)

    def test_single_cadence(self):
        latents = np.arange(6 * 8, dtype=float).reshape(6, 8)
        features = prepare_latent_features(latents, num_observations=6)
        np.testing.assert_array_equal(features, latents.ravel()[None, :])


def _toy_latents(n_per_class=16, latent_dim=8, num_obs=6, seed=0):
    """Separable toy data: class-1 latents cluster at +2, class-0 at -2."""
    rng = np.random.default_rng(seed)
    neg = rng.normal(-2.0, 0.3, size=(n_per_class * num_obs, latent_dim))
    pos = rng.normal(+2.0, 0.3, size=(n_per_class * num_obs, latent_dim))
    latents = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)
    return latents, labels


class TestRandomForestModel:
    @pytest.fixture(autouse=True)
    def _small_forest(self):
        # 1000 trees is overkill for toy data; shrink for speed.
        get_config().rf.n_estimators = 10

    def test_train_and_predict_separable_data(self):
        latents, labels = _toy_latents()
        model = RandomForestModel()
        assert model.is_trained is False
        model.train(latents, labels)
        assert model.is_trained is True

        predictions = model.predict(latents)
        np.testing.assert_array_equal(predictions, labels)

    def test_predict_proba_shape_and_normalization(self):
        latents, labels = _toy_latents()
        model = RandomForestModel()
        model.train(latents, labels)
        probas = model.predict_proba(latents)
        assert probas.shape == (len(labels), 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0)

    def test_predict_verbose_confidence_of_predicted_class(self):
        latents, labels = _toy_latents()
        model = RandomForestModel()
        model.train(latents, labels)
        predictions, confidences = model.predict_verbose(latents)
        probas = model.predict_proba(latents)
        np.testing.assert_array_equal(predictions, labels)
        # Confidence is P(predicted class) — high for confident negatives too.
        expected = np.where(predictions, probas[:, 1], probas[:, 0])
        np.testing.assert_array_equal(confidences, expected)
        assert confidences.min() >= 0.5

    def test_threshold_semantics(self):
        latents, labels = _toy_latents()
        model = RandomForestModel()
        model.train(latents, labels)
        # Strict inequality: P(1) > 1.0 is impossible, so everything is class 0.
        assert model.predict(latents, threshold=1.0).sum() == 0
        # Threshold 0 flags anything with nonzero positive probability.
        assert model.predict(latents, threshold=0.0).sum() >= labels.sum()

    def test_feature_label_mismatch_raises(self):
        latents, _ = _toy_latents(n_per_class=4)
        wrong_labels = np.zeros(3, dtype=int)
        model = RandomForestModel()
        with pytest.raises(ValueError, match="mismatch"):
            model.train(latents, wrong_labels)

    def test_save_load_round_trip(self, tmp_path):
        latents, labels = _toy_latents()
        model = RandomForestModel()
        model.train(latents, labels)
        path = str(tmp_path / "rf.joblib")
        model.save(path)

        restored = RandomForestModel()
        restored.load(path)
        assert restored.is_trained is True
        np.testing.assert_array_equal(restored.predict_proba(latents), model.predict_proba(latents))


class TestSamplingLayer:
    def test_output_shape(self):
        import tensorflow as tf  # noqa: PLC0415

        z_mean = tf.zeros((5, 8))
        z_log_var = tf.zeros((5, 8))
        z = Sampling()([z_mean, z_log_var])
        assert z.shape == (5, 8)

    def test_collapses_to_mean_at_negligible_variance(self):
        import tensorflow as tf  # noqa: PLC0415

        z_mean = tf.constant(np.arange(10, dtype=np.float32).reshape(2, 5))
        z_log_var = tf.fill((2, 5), -100.0)  # std = exp(-50) ~ 0
        z = Sampling()([z_mean, z_log_var])
        np.testing.assert_allclose(z.numpy(), z_mean.numpy(), atol=1e-6)

    def test_variance_injects_noise(self):
        import tensorflow as tf  # noqa: PLC0415

        z_mean = tf.zeros((4, 8))
        z_log_var = tf.zeros((4, 8))  # std = 1
        z1 = Sampling()([z_mean, z_log_var]).numpy()
        z2 = Sampling()([z_mean, z_log_var]).numpy()
        assert not np.allclose(z1, z2)
        assert np.std(z1) > 0.1


@pytest.mark.slow
class TestEncoderDecoderSymmetry:
    """Builds the full Beta-VAE graph on CPU — slow but CI-safe."""

    def test_symmetry_and_forward_pass(self):
        import tensorflow as tf  # noqa: PLC0415

        from aetherscan.models import create_beta_vae_model  # noqa: PLC0415

        vae = create_beta_vae_model()
        config = get_config()
        latent_dim = config.beta_vae.latent_dim
        dense_size = config.beta_vae.dense_layer_size

        # Encoder consumes single observations; decoder must emit the exact mirror shape.
        assert vae.encoder.input_shape == (None, 16, dense_size, 1)
        z_mean_shape, z_log_var_shape, z_shape = vae.encoder.output_shape
        assert z_mean_shape == (None, latent_dim)
        assert z_log_var_shape == (None, latent_dim)
        assert z_shape == (None, latent_dim)
        assert vae.decoder.input_shape == (None, latent_dim)
        assert vae.decoder.output_shape == vae.encoder.input_shape

        # Forward pass round-trips the cadence shape (same graph — built once per test run).
        batch = tf.random.uniform((2, 6, 16, dense_size))
        reconstruction, z_mean, z_log_var, z = vae(batch, training=False)
        assert reconstruction.shape == (2, 6, 16, dense_size)
        # The encoder operates per observation: 2 cadences * 6 observations.
        assert z_mean.shape == (12, latent_dim)
        assert z.shape == (12, latent_dim)
        # Sigmoid output stays within [0, 1] like the log-normed inputs.
        recon = reconstruction.numpy()
        assert recon.min() >= 0.0
        assert recon.max() <= 1.0
