"""Unit tests for train.py's batched distributed dataset builders: whole-global-batch yields,
exact index coverage, stratified splits, the shuffle=False alignment contract that
train_random_forest depends on, and the viz dataset's order guarantee."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from aetherscan.train import (
    TrainDataHolder,
    prepare_distributed_train_dataset,
    prepare_distributed_viz_dataset,
)

# Tiny sample shape — the builders never hardcode (6, 16, 512)
_SAMPLE_SHAPE = (2, 3, 4)
_SIGNAL_TYPES = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]

pytestmark = pytest.mark.slow  # builds TF graphs on CPU


def _make_data(n_samples=40):
    """Arrays whose every sample is a constant equal to (offset + row index), so a yielded
    batch reveals exactly which rows it gathered."""
    base = np.arange(n_samples, dtype=np.float32)[:, None, None, None]
    ones = np.ones((n_samples, *_SAMPLE_SHAPE), dtype=np.float32)
    labels = np.array([_SIGNAL_TYPES[(4 * i) // n_samples] for i in range(n_samples)], dtype="U20")
    return {
        "concatenated": base * ones,
        "true": (base + 1000.0) * ones,
        "false": (base + 2000.0) * ones,
        "labels": labels,
    }


def _build(data, shuffle, train_val_split=0.8, prb=4, eb=8, prvb=4, rng=None):
    return prepare_distributed_train_dataset(
        data=data,
        train_val_split=train_val_split,
        per_replica_batch_size=prb,
        effective_batch_size=eb,
        per_replica_val_batch_size=prvb,
        num_replicas=1,
        strategy=tf.distribute.get_strategy(),
        shuffle=shuffle,
        rng=rng,
    )


def _batch_row_ids(batch):
    """Recover the source row indices a yielded ((c, t, f), y) batch gathered."""
    (concat, true, false), y = batch
    concat_ids = concat.numpy()[:, 0, 0, 0]
    np.testing.assert_array_equal(concat_ids, y.numpy()[:, 0, 0, 0])
    np.testing.assert_array_equal(concat_ids + 1000.0, true.numpy()[:, 0, 0, 0])
    np.testing.assert_array_equal(concat_ids + 2000.0, false.numpy()[:, 0, 0, 0])
    return concat_ids.astype(np.int64)


class TestPrepareDistributedTrainDataset:
    def test_step_counts_and_batched_shapes(self):
        results = _build(_make_data(), shuffle=True)
        assert results["n_train_trimmed"] == 32
        assert results["n_val_trimmed"] == 8
        assert results["train_steps"] == 4  # 32 / effective(8)
        assert results["accumulation_steps"] == 2  # effective(8) / global(4)
        assert results["val_steps"] == 2  # 8 / global val(4)

        iterator = iter(results["train_dataset"])
        batch = next(iterator)
        (concat, true, false), y = batch
        # Whole global batches: leading batch dim is the global batch size (no .batch() op)
        assert concat.shape == (4, *_SAMPLE_SHAPE)
        assert true.shape == (4, *_SAMPLE_SHAPE)
        assert false.shape == (4, *_SAMPLE_SHAPE)
        assert y.shape == (4, *_SAMPLE_SHAPE)
        results["_train_holder"].clear()

    def test_indivisible_effective_batch_raises(self):
        """Runtime backstop for the validation-time skip (#143): an effective batch size that
        isn't a multiple of per_replica_batch_size * num_replicas must fail loudly with the
        validator's guidance, not emit a partial trailing global batch every epoch."""
        with pytest.raises(
            ValueError,
            match=r"--effective-batch-size \(8\) must be divisible by "
            r"per_replica_batch_size \* num_replicas \(3\)",
        ):
            _build(_make_data(), shuffle=True, prb=3, eb=8)

    def test_epoch_covers_train_indices_exactly_once(self):
        results = _build(_make_data(), shuffle=True)
        n_batches_per_epoch = results["train_steps"] * results["accumulation_steps"]
        iterator = iter(results["train_dataset"])
        seen = []
        for _ in range(n_batches_per_epoch):
            batch_ids = _batch_row_ids(next(iterator))
            # Within-batch sorted gathers (read locality); membership is epoch-shuffled
            assert np.all(np.diff(batch_ids) >= 0)
            seen.extend(batch_ids.tolist())
        assert sorted(seen) == sorted(results["train_indices"].tolist())
        results["_train_holder"].clear()

    def test_shuffle_false_yields_train_indices_order(self):
        """The alignment contract train_random_forest depends on: with shuffle=False the i-th
        yielded cadence is row train_indices[i], across exactly train_steps x accumulation_steps
        batches per epoch pass."""
        results = _build(_make_data(), shuffle=False)
        n_batches_per_epoch = results["train_steps"] * results["accumulation_steps"]
        iterator = iter(results["train_dataset"])
        seen = []
        for _ in range(n_batches_per_epoch):
            seen.extend(_batch_row_ids(next(iterator)).tolist())
        assert seen == results["train_indices"].tolist()
        # A second epoch pass repeats the same order (.repeat() on an ordered generator)
        second = []
        for _ in range(n_batches_per_epoch):
            second.extend(_batch_row_ids(next(iterator)).tolist())
        assert second == seen
        results["_train_holder"].clear()

    def test_val_yields_val_indices_order(self):
        results = _build(_make_data(), shuffle=True)
        iterator = iter(results["val_dataset"])
        seen = []
        for _ in range(results["val_steps"]):
            seen.extend(_batch_row_ids(next(iterator)).tolist())
        assert seen == results["val_indices"].tolist()
        results["_train_holder"].clear()

    def test_split_is_stratified_and_disjoint(self):
        data = _make_data()
        results = _build(data, shuffle=True)
        train_indices = results["train_indices"]
        val_indices = results["val_indices"]
        assert set(train_indices).isdisjoint(set(val_indices))
        labels = data["labels"]
        for signal_type in _SIGNAL_TYPES:
            # 40 samples, 10 per type, 0.8 split -> 8 train / 2 val of each type
            assert int(np.sum(labels[train_indices] == signal_type)) == 8
            assert int(np.sum(labels[val_indices] == signal_type)) == 2
        results["_train_holder"].clear()

    def test_seeded_rng_reproduces_split_and_epoch_order(self):
        """Reproducibility contract (issue #49): the same seeded Generator reproduces the
        stratified split AND every epoch's shuffled batch order; successive epochs still
        differ from each other (the rng advances, randomness is not removed)."""

        def _run():
            results = _build(_make_data(), shuffle=True, rng=np.random.default_rng(11))
            n_batches = results["train_steps"] * results["accumulation_steps"]
            iterator = iter(results["train_dataset"])
            epochs = [
                [_batch_row_ids(next(iterator)).tolist() for _ in range(n_batches)]
                for _ in range(2)
            ]
            out = (results["train_indices"].tolist(), results["val_indices"].tolist(), epochs)
            results["_train_holder"].clear()
            return out

        first, second = _run(), _run()
        assert first == second
        # Epoch-level randomness survives seeding: epoch 2's batch membership differs
        assert first[2][0] != first[2][1]

    def test_memmap_inputs_supported(self, tmp_path):
        """Round data arrives as np.load(mmap_mode='r') memmaps — gathers must produce plain
        in-RAM batches from them."""
        data = _make_data()
        mmap_data = {"labels": data["labels"]}
        for key in ("concatenated", "true", "false"):
            path = tmp_path / f"{key}.npy"
            np.save(path, data[key])
            mmap_data[key] = np.load(path, mmap_mode="r")
        results = _build(mmap_data, shuffle=False)
        iterator = iter(results["train_dataset"])
        batch_ids = _batch_row_ids(next(iterator))
        assert batch_ids.tolist() == results["train_indices"][:4].tolist()
        results["_train_holder"].clear()

    def test_holder_clear_drops_references(self):
        holder = TrainDataHolder(np.zeros(2), np.zeros(2), np.zeros(2))
        holder.clear()
        assert holder.concat is None and holder.true is None and holder.false is None
        holder.clear()  # idempotent
        assert holder._cleared


class TestPrepareDistributedVizDataset:
    def _build_viz(self, concat, prib=4, rng=None):
        return prepare_distributed_viz_dataset(
            concat_data=concat,
            per_replica_inf_batch_size=prib,
            num_replicas=1,
            strategy=tf.distribute.get_strategy(),
            rng=rng,
        )

    def test_order_preserved_with_padding(self):
        n = 10
        base = np.arange(n, dtype=np.float32)[:, None, None, None]
        concat = base * np.ones((n, *_SAMPLE_SHAPE), dtype=np.float32)
        results = self._build_viz(concat)
        assert results["n_samples"] == 10
        assert results["n_padded"] == 12
        assert results["viz_steps"] == 3

        iterator = iter(results["viz_dataset"])
        seen = []
        for _ in range(results["viz_steps"]):
            batch = next(iterator)
            assert batch.shape == (4, *_SAMPLE_SHAPE)  # whole global batches
            seen.extend(batch.numpy()[:, 0, 0, 0].astype(np.int64).tolist())
        # WARN-comment contract: the first n_samples yields are the original cadence order —
        # plot_latent_space_gif and _capture_latent_snapshot truncate the padded tail
        assert seen[:n] == list(range(n))
        # Padding rows duplicate real cadences
        assert all(0 <= idx < n for idx in seen[n:])
        results["_viz_holder"].clear()

    def test_no_padding_when_divisible(self):
        n = 8
        concat = np.ones((n, *_SAMPLE_SHAPE), dtype=np.float32)
        results = self._build_viz(concat)
        assert results["n_padded"] == n
        assert results["viz_steps"] == 2
        results["_viz_holder"].clear()

    def test_seeded_rng_reproduces_padding(self):
        n = 10
        base = np.arange(n, dtype=np.float32)[:, None, None, None]
        concat = base * np.ones((n, *_SAMPLE_SHAPE), dtype=np.float32)

        def _padded_tail():
            results = self._build_viz(concat, rng=np.random.default_rng(7))
            iterator = iter(results["viz_dataset"])
            seen = []
            for _ in range(results["viz_steps"]):
                seen.extend(next(iterator).numpy()[:, 0, 0, 0].astype(np.int64).tolist())
            results["_viz_holder"].clear()
            return seen[n:]

        assert _padded_tail() == _padded_tail()
