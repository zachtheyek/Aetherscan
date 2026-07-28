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


@pytest.mark.slow
class TestMixedPrecisionDtypeIslands:
    """beta_vae.mixed_precision: under the keras mixed_bfloat16 policy the interior conv
    stack computes in bf16, while the numerically sensitive islands — the z_mean/z_log_var
    heads, the Sampling layer, and the decoder's sigmoid output (which feeds the BCE loss
    math) — are pinned to fp32 in models/vae.py."""

    def test_bf16_policy_keeps_fp32_islands(self):
        import tensorflow as tf  # noqa: PLC0415

        from aetherscan.models.vae import build_decoder, build_encoder  # noqa: PLC0415

        previous = tf.keras.mixed_precision.global_policy().name
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
            encoder = build_encoder()
            decoder = build_decoder()
        finally:
            tf.keras.mixed_precision.set_global_policy(previous)

        # fp32 islands: the latent heads, the sampled z, and the sigmoid reconstruction.
        z_mean, z_log_var, z = encoder.outputs
        assert tf.as_dtype(z_mean.dtype) == tf.float32
        assert tf.as_dtype(z_log_var.dtype) == tf.float32
        assert tf.as_dtype(z.dtype) == tf.float32
        assert tf.as_dtype(decoder.outputs[0].dtype) == tf.float32
        sampling = next(layer for layer in encoder.layers if isinstance(layer, Sampling))
        assert sampling.compute_dtype == "float32"

        # The interior conv stack actually picked up the bf16 compute dtype.
        encoder_convs = [
            layer for layer in encoder.layers if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        assert encoder_convs
        assert all(layer.compute_dtype == "bfloat16" for layer in encoder_convs)
        # Variables stay fp32 under keras mixed precision (why bf16 needs no loss scaling).
        assert all(v.dtype == tf.float32 for v in encoder.trainable_variables)
        assert all(v.dtype == tf.float32 for v in decoder.trainable_variables)

    def test_default_policy_builds_all_float32(self):
        """Regression pin: with the flag off (no policy call), every layer is fp32 — the
        dtype="float32" island kwargs must be no-ops under the default policy."""
        import tensorflow as tf  # noqa: PLC0415

        from aetherscan.models.vae import build_decoder, build_encoder  # noqa: PLC0415

        assert tf.keras.mixed_precision.global_policy().name == "float32"
        encoder = build_encoder()
        decoder = build_decoder()
        for model in (encoder, decoder):
            for layer in model.layers:
                assert layer.compute_dtype == "float32", layer.name
                assert layer.dtype == "float32", layer.name


class TestSeededSamplingReproducibility:
    """#279: seed_tensorflow makes the Sampling layer's draws reproducible — the mechanism
    that makes inference candidate sets repeatable (the layer used to be entirely unseeded
    on the inference path)."""

    def test_same_stream_key_reproduces_sampled_z(self):
        import tensorflow as tf  # noqa: PLC0415

        from aetherscan.seeding import seed_tensorflow  # noqa: PLC0415

        layer = Sampling()
        z_mean = tf.zeros((4, 8))
        z_log_var = tf.zeros((4, 8))

        seed_tensorflow(207, False, 1, 5)
        first = layer([z_mean, z_log_var]).numpy()
        seed_tensorflow(207, False, 1, 5)
        second = layer([z_mean, z_log_var]).numpy()
        np.testing.assert_array_equal(first, second)

        # A different sub-key (e.g. another cadence) yields a different, but equally
        # reproducible, draw
        seed_tensorflow(207, False, 1, 6)
        third = layer([z_mean, z_log_var]).numpy()
        assert not np.array_equal(first, third)

    def test_unseeded_root_returns_none_and_leaves_entropy(self):
        import tensorflow as tf  # noqa: PLC0415

        from aetherscan.seeding import seed_tensorflow  # noqa: PLC0415

        layer = Sampling()
        z_mean = tf.zeros((4, 8))
        z_log_var = tf.zeros((4, 8))
        assert seed_tensorflow(None, False, 1) is None
        first = layer([z_mean, z_log_var]).numpy()
        assert seed_tensorflow(None, False, 1) is None
        second = layer([z_mean, z_log_var]).numpy()
        assert not np.array_equal(first, second)


class TestWeightInitReproducibility:
    """#279 acceptance ("same seed => byte-identical VAE weights") — the criterion that was
    NOT actually met before: tf_keras initializers seed from Python's global `random`, not
    from tf.random.set_seed, so weight init drifted run-to-run even with --seed pinned."""

    @staticmethod
    def _build_kernel():
        import tensorflow as tf  # noqa: PLC0415
        from tensorflow import keras  # noqa: PLC0415

        del tf  # imported for parity with the production build path
        inputs = keras.Input(shape=(8,))
        outputs = keras.layers.Dense(16, kernel_initializer=keras.initializers.HeNormal())(inputs)
        return keras.Model(inputs, outputs).get_weights()[0]

    def test_same_root_seed_reproduces_initial_weights(self):
        from aetherscan.seeding import seed_tensorflow  # noqa: PLC0415

        seed_tensorflow(207, False, 0)
        first = self._build_kernel()
        seed_tensorflow(207, False, 0)
        second = self._build_kernel()
        np.testing.assert_array_equal(first, second)

    def test_different_root_seed_changes_initial_weights(self):
        from aetherscan.seeding import seed_tensorflow  # noqa: PLC0415

        seed_tensorflow(207, False, 0)
        first = self._build_kernel()
        seed_tensorflow(208, False, 0)
        second = self._build_kernel()
        assert not np.array_equal(first, second)

    def test_tf_set_seed_alone_is_insufficient(self):
        # Regression guard on the ROOT CAUSE: if a future refactor drops the Python-random
        # seeding from seed_tensorflow (e.g. "tf.random.set_seed is enough"), this test
        # documents why that is wrong on the tf_keras stack. Two builds under only
        # tf.random.set_seed must differ — if they ever stop differing, tf_keras changed its
        # initializer seeding and seed_tensorflow's workaround can be revisited.
        import random  # noqa: PLC0415

        import tensorflow as tf  # noqa: PLC0415

        random.seed(1234)  # fixed start so the assertion below is deterministic
        tf.random.set_seed(207)
        first = self._build_kernel()
        tf.random.set_seed(207)
        second = self._build_kernel()
        assert not np.array_equal(first, second)
