"""Unit tests for aetherscan.inference: the tail-padding fix in
prepare_distributed_inf_dataset (regression for the silent partial-batch drop), the
encoded-padding-row truncation in run_inference, the InfDataHolder clear semantics, and the
confidence-summary math backing the inference_cadences manifest."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from aetherscan.config import get_config
from aetherscan.inference import (
    InfDataHolder,
    InferencePipeline,
    ReferenceCloudReservoir,
    prepare_distributed_inf_dataset,
    summarize_confidences,
)


def _collect_batches(result: dict) -> np.ndarray:
    """Materialize inf_steps batches from the (infinite, repeated) distributed dataset."""
    iterator = iter(result["inf_dataset"])
    batches = [next(iterator).numpy() for _ in range(result["inf_steps"])]
    return np.concatenate(batches, axis=0)


def _make_data(n: int) -> np.ndarray:
    # Give every sample a unique signature so padding/ordering is verifiable
    return np.arange(n, dtype=np.float32)[:, None, None, None] * np.ones(
        (n, 6, 4, 8), dtype=np.float32
    )


class TestPrepareDistributedInfDataset:
    @pytest.fixture
    def strategy(self):
        return tf.distribute.get_strategy()  # default no-op strategy (1 replica, CPU-safe)

    def test_partial_tail_batch_is_padded_not_dropped(self, strategy):
        # Regression: 5 samples with global batch 2 used to yield inf_steps = 2 with
        # drop_remainder=True — the 5th sample was silently never processed.
        data = _make_data(5)
        result = prepare_distributed_inf_dataset(
            data=data,
            n_samples=5,
            per_replica_inf_batch_size=2,
            num_replicas=1,
            strategy=strategy,
        )
        assert result["n_samples"] == 5
        assert result["n_padded"] == 6
        assert result["inf_steps"] == 3

        out = _collect_batches(result)
        assert out.shape[0] == 6
        np.testing.assert_array_equal(out[:5], data)
        # Padding duplicates rows cycled from the front
        np.testing.assert_array_equal(out[5], data[0])

    def test_cadence_smaller_than_one_batch_processes_everything(self, strategy):
        # Regression: with per-cadence batches, a cadence with fewer stamps than one global
        # batch used to process *nothing* (inf_steps == 0).
        data = _make_data(3)
        result = prepare_distributed_inf_dataset(
            data=data,
            n_samples=3,
            per_replica_inf_batch_size=8,
            num_replicas=1,
            strategy=strategy,
        )
        assert result["inf_steps"] == 1
        assert result["n_padded"] == 8

        out = _collect_batches(result)
        np.testing.assert_array_equal(out[:3], data)
        # 5 pad rows cycle deterministically over the 3 real samples
        np.testing.assert_array_equal(out[3:], data[np.arange(5) % 3])

    def test_exact_multiple_needs_no_padding(self, strategy):
        data = _make_data(4)
        result = prepare_distributed_inf_dataset(
            data=data,
            n_samples=4,
            per_replica_inf_batch_size=2,
            num_replicas=1,
            strategy=strategy,
        )
        assert result["n_padded"] == 4
        assert result["inf_steps"] == 2
        np.testing.assert_array_equal(_collect_batches(result), data)

    def test_zero_samples_raises(self, strategy):
        with pytest.raises(ValueError, match="Not enough samples"):
            prepare_distributed_inf_dataset(
                data=np.zeros((0, 6, 4, 8), dtype=np.float32),
                n_samples=0,
                per_replica_inf_batch_size=2,
                num_replicas=1,
                strategy=strategy,
            )

    def test_order_is_preserved(self, strategy):
        data = _make_data(7)
        result = prepare_distributed_inf_dataset(
            data=data,
            n_samples=7,
            per_replica_inf_batch_size=3,
            num_replicas=1,
            strategy=strategy,
        )
        out = _collect_batches(result)
        np.testing.assert_array_equal(out[:7, 0, 0, 0], np.arange(7, dtype=np.float32))

    @pytest.mark.parametrize("num_replicas", [1, 2])
    def test_multi_replica_global_batch_geometry(self, strategy, num_replicas):
        """global_batch = per_replica x num_replicas — the production geometry the tail-drop
        fix targets. num_replicas is a plain argument to the padding/step math (independent
        of the CPU default strategy used here); true cross-replica distribution is only
        exercised by the gpu-marked integration smoke."""
        data = _make_data(5)
        result = prepare_distributed_inf_dataset(
            data=data,
            n_samples=5,
            per_replica_inf_batch_size=2,
            num_replicas=num_replicas,
            strategy=strategy,
        )
        global_batch = 2 * num_replicas
        expected_padded = -(-5 // global_batch) * global_batch  # 6 for 1 replica, 8 for 2
        assert result["n_padded"] == expected_padded
        assert result["inf_steps"] == expected_padded // global_batch

        out = _collect_batches(result)
        assert out.shape[0] == expected_padded
        np.testing.assert_array_equal(out[:5], data)
        # Padding rows cycle deterministically from the front, whatever the replica count
        np.testing.assert_array_equal(out[5:], data[np.arange(expected_padded - 5) % 5])


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


class TestInfDataHolder:
    def test_clear_is_idempotent(self):
        holder = InfDataHolder(np.ones(3))
        holder.clear()
        assert holder.data is None
        holder.clear()  # second clear is a no-op
        assert holder.data is None


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
