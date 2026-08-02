"""Unit tests for aetherscan.inference: the bucketed numpy-slice encode path (#298 I2+I4 —
order preservation, tail padding/truncation, bounded trace shapes), the batched MC cascade
(#298 I8), and the confidence-summary math backing the inference_cadences manifest."""

from __future__ import annotations

import types

import numpy as np
import pytest
import tensorflow as tf

from aetherscan.config import get_config
from aetherscan.inference import (
    _MIN_ENCODE_BUCKET,
    InferencePipeline,
    ReferenceCloudReservoir,
    _batched_mc_scores,
    summarize_confidences,
)
from aetherscan.latent_variants import (
    apply_probability_calibrator,
    build_variant_features,
    fit_probability_calibrator,
    sample_z_flat,
)


def _make_data(n: int) -> np.ndarray:
    # Give every sample a unique signature so padding/ordering is verifiable
    return np.arange(n, dtype=np.float32)[:, None, None, None] * np.ones(
        (n, 6, 4, 8), dtype=np.float32
    )


class TestDeriveEncodeModel:
    """#301 E6 + #305 pass-2: the encode-only submodel drops the discarded Sampling head;
    the head-count guard checks the SOURCE encoder (3 heads) and falls back to the full
    encoder for any other count, instead of the [:2] slice silently taking the wrong two."""

    def _fake_encoder(self, n_outputs: int):
        inp = tf.keras.Input(shape=(4,))
        outs = [tf.keras.layers.Dense(2, name=f"h{i}")(inp) for i in range(n_outputs)]
        return tf.keras.Model(inp, outs if n_outputs != 1 else outs[0])

    def test_three_head_encoder_yields_two_output_submodel(self):
        from aetherscan.inference import _derive_encode_model  # noqa: PLC0415

        enc = self._fake_encoder(3)  # [z_mean, z_log_var, z]
        sub = _derive_encode_model(enc)
        assert sub is not enc
        assert len(sub.outputs) == 2  # sampling head dropped

    @pytest.mark.parametrize("n_outputs", [2, 4])
    def test_wrong_head_count_falls_back_to_full_encoder(self, n_outputs):
        # A 2-head OR a 4-head encoder must fall back loudly (not mis-slice): the guard the
        # first attempt checked on the DERIVED model missed the 4-head add-a-head case.
        from aetherscan.inference import _derive_encode_model  # noqa: PLC0415

        enc = self._fake_encoder(n_outputs)
        assert _derive_encode_model(enc) is enc


class TestEncodeBucket:
    """#298 I2+I4: the final-partial-step bucket must cover the remainder, stay a power of
    two in [_MIN_ENCODE_BUCKET, max_bucket], and bound padding waste."""

    def test_small_remainders_take_the_floor_bucket(self):
        assert InferencePipeline._encode_bucket(1, 1, 256) == _MIN_ENCODE_BUCKET
        assert InferencePipeline._encode_bucket(5, 5, 256) == _MIN_ENCODE_BUCKET

    def test_bucket_covers_the_per_replica_need(self):
        # remaining=900 over 5 replicas -> need 180 -> next power of two 256
        assert InferencePipeline._encode_bucket(900, 5, 256) == 256
        # remaining=300 over 5 replicas -> need 60 -> 64
        assert InferencePipeline._encode_bucket(300, 5, 256) == 64

    def test_bucket_never_exceeds_the_configured_cap(self):
        assert InferencePipeline._encode_bucket(10_000, 1, 256) == 256
        assert InferencePipeline._encode_bucket(3, 1, 2) == 2

    def test_bucket_ladder_is_bounded(self):
        # Whatever the catalog mix, the set of shapes is the power-of-two ladder
        buckets = {InferencePipeline._encode_bucket(r, 5, 256) for r in range(1, 256 * 5)}
        assert buckets <= {16, 32, 64, 128, 256}


def _bare_pipeline(per_replica_batch: int, latent_dim: int = 1) -> InferencePipeline:
    """Minimal pipeline wired for _distributed_encode on the default (CPU) strategy."""
    config = get_config()
    config.data.time_bins = 4
    config.data.width_bin = 64
    config.data.downsample_factor = 8  # encode reshape width: 64 // 8 = 8

    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.config = config
    pipeline.strategy = tf.distribute.get_strategy()
    pipeline.num_replicas = 1
    pipeline.encoder = _StubEncoder()
    pipeline.latent_dim = latent_dim
    pipeline.num_observations = 6
    pipeline.per_replica_inf_batch_size = per_replica_batch
    pipeline._encode_step = None
    return pipeline


class TestDistributedEncodeSlicing:
    """#298 I2 option (b): the numpy-slice encode must return exactly the real snippets'
    latent rows, in snippet-major order, whatever the bucket/padding geometry."""

    def _signatures(self, z_mean: np.ndarray, n: int) -> None:
        # _StubEncoder emits each observation's mean power == the snippet's signature value
        np.testing.assert_allclose(z_mean[:, 0], np.repeat(np.arange(n, dtype=np.float32), 6))

    @pytest.mark.parametrize("n_samples", [1, 3, 16, 21, 47])
    def test_all_real_rows_in_order_no_padding_leak(self, n_samples):
        pipeline = _bare_pipeline(per_replica_batch=16)
        z_mean, z_log_var = pipeline._distributed_encode(_make_data(n_samples))
        assert z_mean.shape == (n_samples * 6, 1)
        assert z_log_var.shape == (n_samples * 6, 1)
        self._signatures(z_mean, n_samples)

    def test_multi_step_full_buckets_then_partial(self):
        # 40 samples at per-replica 16: two full steps (16, 16) + one padded step (8 -> 16)
        pipeline = _bare_pipeline(per_replica_batch=16)
        z_mean, _ = pipeline._distributed_encode(_make_data(40))
        assert z_mean.shape == (240, 1)
        self._signatures(z_mean, 40)

    def test_zero_samples_raises_in_run_inference(self):
        pipeline = _bare_pipeline(per_replica_batch=16)
        pipeline.rf_model = _RecordingRF()
        pipeline.threshold = 0.99
        pipeline.screening_threshold = 0.5
        pipeline.mc_draws = 2
        pipeline.calibrator = None
        pipeline._reference_reservoir = ReferenceCloudReservoir(
            capacity=0, rng=np.random.default_rng(0)
        )
        pipeline._write_inference_results = lambda **kwargs: 0
        with pytest.raises(ValueError, match="Not enough samples"):
            pipeline.run_inference(
                data=np.zeros((0, 6, 4, 8), dtype=np.float32), npy_path="/fake.npy"
            )


class _StubEncoder:
    """Keras-encoder stand-in usable inside the tf.function encode step: one latent
    dimension per observation carrying the observation's mean power. _make_data gives
    sample i the constant value i, so every latent row is traceable to its snippet."""

    def __call__(self, reshaped, training=False):
        z = tf.reduce_mean(reshaped, axis=[1, 2])  # (batch * 6, 4, 8, 1) -> (batch * 6, 1)
        return z, z, z


class _RecordingRF:
    """predict_proba stub recording how many feature rows reached the classifier. Since
    #282 the pipeline calls rf_model.model.predict_proba with CADENCE-level variant
    features, so the stub exposes itself as its own .model."""

    def __init__(self):
        self.n_rows_seen: int | None = None
        self.model = self

    def predict_proba(self, features):
        self.n_rows_seen = features.shape[0]
        return np.full((features.shape[0], 2), [0.7, 0.3])  # P(true)=0.3: never a candidate


class TestLatentPaddingRowTruncation:
    """run_inference must drop the encoded padding rows before the RF sees them
    (inference.py: latents = latents[: n_samples * self.num_observations]) — the
    encoder-side half of the tail-drop fix, complementing the dataset-padding tests above."""

    def test_encoded_padding_rows_are_dropped(self):
        config = get_config()
        config.data.time_bins = 4
        config.data.width_bin = 64
        config.data.downsample_factor = 8  # encode reshape width: 64 // 8 = 8

        # Bypass __init__ (which loads real models); wire only what run_inference touches.
        pipeline = InferencePipeline.__new__(InferencePipeline)
        pipeline.config = config
        pipeline.strategy = tf.distribute.get_strategy()
        pipeline.num_replicas = 1
        pipeline.encoder = _StubEncoder()
        pipeline.rf_model = _RecordingRF()
        pipeline.latent_dim = 1
        pipeline.num_observations = 6
        pipeline.per_replica_inf_batch_size = 2  # global batch 2: n=5 pads to 6
        pipeline.threshold = 0.99
        pipeline.screening_threshold = 0.5
        pipeline.mc_draws = 2
        pipeline.calibrator = None
        pipeline._reference_reservoir = ReferenceCloudReservoir(
            capacity=0, rng=np.random.default_rng(0)
        )
        pipeline._encode_step = None
        # DB writes are covered elsewhere; keep this test on the truncation seam.
        pipeline._write_inference_results = lambda **kwargs: 0

        data = _make_data(5)
        results = pipeline.run_inference(data=data, npy_path="/fake/cadence.npy")

        # 5 samples pad to 6 -> 36 latent rows encoded; only the 30 real ones survive, and
        # since #282 the RF consumes CADENCE-level variant features: one row per snippet.
        assert pipeline.rf_model.n_rows_seen == 5
        assert results["latents"].shape == (5 * 6, 1)
        assert results["n_cadence_snippets"] == 5
        assert results["n_processed"] == 5
        # Row signatures prove the survivors are the real snippets in snippet-major order
        # (the dropped tail re-encoded sample 0 and would have carried signature 0.0).
        np.testing.assert_allclose(
            results["latents"][:, 0], np.repeat(np.arange(5, dtype=np.float32), 6)
        )


class TestSummarizeConfidences:
    def test_summary_math(self):
        proba = np.linspace(0.0, 1.0, 101)  # p50 = 0.5, p01 = 0.01, ... exactly
        summary = summarize_confidences(proba, threshold=0.9)

        assert summary["n"] == 101
        assert summary["threshold"] == 0.9
        assert summary["n_above_threshold"] == 10  # 0.91 .. 1.00 (strictly above)
        assert summary["mean"] == pytest.approx(0.5)
        assert summary["min"] == 0.0
        assert summary["max"] == 1.0
        assert summary["quantiles"]["p50"] == pytest.approx(0.5)
        assert summary["quantiles"]["p01"] == pytest.approx(0.01)
        assert summary["quantiles"]["p99"] == pytest.approx(0.99)
        assert list(summary["quantiles"]) == ["p01", "p05", "p25", "p50", "p75", "p95", "p99"]

    def test_single_value(self):
        summary = summarize_confidences(np.array([0.42]), threshold=0.5)
        assert summary["n"] == 1
        assert summary["n_above_threshold"] == 0
        assert all(v == pytest.approx(0.42) for v in summary["quantiles"].values())

    def test_json_serializable(self):
        import json  # noqa: PLC0415

        summary = summarize_confidences(np.random.default_rng(1).random(50), threshold=0.99)
        round_tripped = json.loads(json.dumps(summary))
        assert round_tripped == summary  # plain floats/ints only, no numpy scalars

    def test_threshold_boundary_is_strict(self):
        # Matches the RF prediction rule: prediction = proba > threshold, not >=
        summary = summarize_confidences(np.array([0.99, 0.991]), threshold=0.99)
        assert summary["n_above_threshold"] == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one confidence"):
            summarize_confidences(np.array([]), threshold=0.5)


class TestBatchedMcScores:
    """#298 I8: one stacked predict_proba call must reproduce the retired per-draw loop
    bit-for-bit (n_jobs=1) — same values, same positions, same RNG stream consumption."""

    NUM_OBS, LATENT_DIM = 4, 2

    def _tiny_rf(self, n_features):
        import types  # noqa: PLC0415

        from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

        rng = np.random.default_rng(5)
        features = rng.standard_normal((80, n_features))
        labels = (features[:, 0] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=16, random_state=0, n_jobs=1).fit(
            features, labels
        )
        # _batched_mc_scores only touches rf_model.model.predict_proba
        return types.SimpleNamespace(model=model)

    def _blocks(self, n):
        rng = np.random.default_rng(9)
        base = self.NUM_OBS * self.LATENT_DIM
        mean_flat = rng.standard_normal((n, base)).astype(np.float32)
        logvar_flat = (rng.standard_normal((n, base)) - 2.0).astype(np.float32)
        return mean_flat, logvar_flat

    def _reference_loop(
        self, rf, calibrator, variant, mean_flat, logvar_flat, active_dims, draws, rng
    ):
        """Verbatim re-implementation of the pre-#298 per-draw loop."""
        out = np.empty((draws, len(mean_flat)))
        for draw_index in range(draws):
            draw_flat = sample_z_flat(mean_flat, logvar_flat, rng)
            feats = build_variant_features(
                variant, draw_flat, logvar_flat, self.NUM_OBS, self.LATENT_DIM, active_dims
            )
            out[draw_index] = apply_probability_calibrator(
                calibrator, rf.model.predict_proba(feats)[:, 1]
            )
        return out

    @pytest.mark.parametrize("variant", ["z_mean", "z_mean_obs_logvar"])
    def test_matches_per_draw_loop_bitwise(self, variant):
        base = self.NUM_OBS * self.LATENT_DIM
        n_features = base if variant == "z_mean" else base + self.NUM_OBS
        rf = self._tiny_rf(n_features)
        mean_flat, logvar_flat = self._blocks(12)

        expected = self._reference_loop(
            rf, None, variant, mean_flat, logvar_flat, None, 8, np.random.default_rng(77)
        )
        actual = _batched_mc_scores(
            rf,
            None,
            variant,
            mean_flat,
            logvar_flat,
            self.NUM_OBS,
            self.LATENT_DIM,
            None,
            8,
            np.random.default_rng(77),
        )
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (8, 12)

    def test_matches_with_calibrator(self):
        base = self.NUM_OBS * self.LATENT_DIM
        rf = self._tiny_rf(base)
        mean_flat, logvar_flat = self._blocks(10)
        cal_rng = np.random.default_rng(3)
        calibrator = fit_probability_calibrator(
            cal_rng.random(200), (cal_rng.random(200) > 0.5).astype(int), min_isotonic=50
        )

        expected = self._reference_loop(
            rf, calibrator, "z_mean", mean_flat, logvar_flat, None, 6, np.random.default_rng(21)
        )
        actual = _batched_mc_scores(
            rf,
            calibrator,
            "z_mean",
            mean_flat,
            logvar_flat,
            self.NUM_OBS,
            self.LATENT_DIM,
            None,
            6,
            np.random.default_rng(21),
        )
        np.testing.assert_array_equal(actual, expected)


class TestReferenceCloudReservoir:
    """#282: seeded uniform reservoir over pass-1 rejects."""

    def _rows(self, n, offset=0):
        mean = np.arange(n * 4, dtype=np.float32).reshape(n, 4) + offset
        log_var = np.full((n, 4), -2.0, dtype=np.float32)
        screening = np.linspace(0, 0.4, n).astype(np.float32)
        return mean, log_var, screening

    def test_fills_to_capacity_then_stays_bounded(self):
        reservoir = ReferenceCloudReservoir(capacity=5, rng=np.random.default_rng(0))
        reservoir.offer(*self._rows(3))
        reservoir.offer(*self._rows(10, offset=100))
        mean, log_var, screening = reservoir.arrays()
        assert mean.shape == (5, 4) and log_var.shape == (5, 4) and screening.shape == (5,)
        assert reservoir.seen == 13

    def test_seeded_reservoir_is_reproducible(self):
        def build():
            reservoir = ReferenceCloudReservoir(capacity=4, rng=np.random.default_rng(42))
            for chunk in range(5):
                reservoir.offer(*self._rows(7, offset=chunk * 10))
            return reservoir.arrays()

        first_mean, _, first_screening = build()
        second_mean, _, second_screening = build()
        np.testing.assert_array_equal(first_mean, second_mean)
        np.testing.assert_array_equal(first_screening, second_screening)

    def test_zero_capacity_collects_nothing(self):
        reservoir = ReferenceCloudReservoir(capacity=0, rng=np.random.default_rng(0))
        reservoir.offer(*self._rows(6))
        _, _, screening = reservoir.arrays()
        assert len(screening) == 0

    def test_rng_consumption_pattern_is_frozen(self):
        # GOLDEN test: pins the exact rng-consumption pattern of offer()'s vectorized
        # replacement phase (one rng.random(batch) acceptance draw, then one
        # rng.integers(batch) slot draw, ALWAYS both, regardless of acceptance count).
        # A same-seed-twice comparison cannot catch a refactor that reorders these calls —
        # both sides would drift together — so the surviving items are frozen here.
        # Regenerate the goldens ONLY for a deliberate, documented stream change.
        reservoir = ReferenceCloudReservoir(capacity=3, rng=np.random.default_rng(123))
        for chunk in range(4):
            mean = (np.arange(5, dtype=np.float32) + 10 * chunk).reshape(5, 1)
            log_var = np.full((5, 1), -2.0, dtype=np.float32)
            screening = (np.arange(5, dtype=np.float32) + 10 * chunk) / 100.0
            reservoir.offer(mean, log_var, screening.astype(np.float32))
        mean_rows, _, screening_vals = reservoir.arrays()
        assert reservoir.seen == 20
        np.testing.assert_array_equal(mean_rows[:, 0], [14.0, 20.0, 22.0])
        np.testing.assert_allclose(screening_vals, [0.14, 0.20, 0.22], atol=1e-6)


class TestRfFeatureLayoutGuard:
    """#282 hardening: init_models fails loud on a config↔RF feature-count mismatch —
    same-width variants (z/z_mean/z_aug all = num_obs*latent_dim) make it silent otherwise."""

    def _pipeline(self, variant, active_dims, n_features, recorded_variant=None):
        config = get_config()
        config.data.num_observations = 6
        config.beta_vae.latent_dim = 8
        config.rf.latent_variant = variant
        config.rf.active_dims = active_dims
        pipeline = InferencePipeline.__new__(InferencePipeline)
        pipeline.config = config
        model = types.SimpleNamespace()
        if n_features is not None:
            model.n_features_in_ = n_features
        if recorded_variant is not None:
            model.aetherscan_latent_variant_ = recorded_variant
        pipeline.rf_model = types.SimpleNamespace(model=model)
        return pipeline

    def test_matching_count_passes(self):
        # z_mean at num_obs=6, latent_dim=8 -> 6*8 = 48 features
        self._pipeline("z_mean", list(range(8)), 48)._check_rf_feature_layout()

    def test_mismatched_count_raises(self):
        # config declares z_mean (48) but the loaded forest carries z_mean_logvar (96) features
        pipeline = self._pipeline("z_mean", list(range(8)), 96)
        with pytest.raises(ValueError, match="feature-count mismatch"):
            pipeline._check_rf_feature_layout()

    def test_absent_n_features_is_a_noop(self):
        # an unfitted/stub forest without n_features_in_ -> the guard skips silently
        self._pipeline("z_mean", list(range(8)), None)._check_rf_feature_layout()

    def test_matching_variant_stamp_passes(self):
        # forest stamped with the same variant the config declares -> passes
        self._pipeline(
            "z_mean", list(range(8)), 48, recorded_variant="z_mean"
        )._check_rf_feature_layout()

    def test_mismatched_variant_stamp_raises(self):
        # SAME width (z and z_mean are both 48) so the count check passes, but the #318 identity
        # stamp catches the confusion the count check cannot
        pipeline = self._pipeline("z_mean", list(range(8)), 48, recorded_variant="z")
        with pytest.raises(ValueError, match="latent-variant mismatch"):
            pipeline._check_rf_feature_layout()

    def test_absent_variant_stamp_is_a_noop(self):
        # a forest without the #318 stamp (e.g. the v1.0.0 weights) -> identity check skips
        self._pipeline(
            "z_mean", list(range(8)), 48, recorded_variant=None
        )._check_rf_feature_layout()
