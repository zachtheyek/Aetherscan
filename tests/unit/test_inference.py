"""Unit tests for aetherscan.inference: the tail-padding fix in
prepare_distributed_inf_dataset (regression for the silent partial-batch drop) and the
InfDataHolder clear semantics."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from aetherscan.inference import InfDataHolder, prepare_distributed_inf_dataset


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


class TestInfDataHolder:
    def test_clear_is_idempotent(self):
        holder = InfDataHolder(np.ones(3))
        holder.clear()
        assert holder.data is None
        holder.clear()  # second clear is a no-op
        assert holder.data is None
