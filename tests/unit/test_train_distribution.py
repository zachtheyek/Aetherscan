"""Multi-replica distribution semantics of the per-replica-element datasets (#276
follow-up), on 2 logical CPU devices in a subprocess (logical devices must be configured
before TF initializes, which only a fresh interpreter guarantees).

Pins the contract everything downstream relies on: replica r of global batch g receives
rows [r*B:(r+1)*B] of the sorted global batch — so experimental_local_results concatenated
in replica order reproduces the exact global-batch stream, which is what train.py's
_distributed_encode does and what the train_random_forest alignment (encoded row i ==
train_indices[i]) requires on multi-GPU hosts."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.slow

_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import tensorflow as tf

    cpus = tf.config.list_physical_devices("CPU")
    tf.config.set_logical_device_configuration(
        cpus[0],
        [tf.config.LogicalDeviceConfiguration(), tf.config.LogicalDeviceConfiguration()],
    )
    strategy = tf.distribute.MirroredStrategy(["/CPU:0", "/CPU:1"])
    assert strategy.num_replicas_in_sync == 2

    from aetherscan.train import prepare_distributed_train_dataset

    n = 40
    base = np.arange(n, dtype=np.float32)[:, None, None, None]
    ones = np.ones((n, 2, 3, 4), dtype=np.float32)
    types = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]
    data = {
        "concatenated": base * ones,
        "true": (base + 1000.0) * ones,
        "false": (base + 2000.0) * ones,
        "labels": np.array([types[(4 * i) // n] for i in range(n)], dtype="U20"),
    }

    for shuffle in (False, True):
        results = prepare_distributed_train_dataset(
            data=data,
            train_val_split=0.8,
            per_replica_batch_size=4,
            effective_batch_size=16,
            per_replica_val_batch_size=2,
            num_replicas=2,
            strategy=strategy,
            shuffle=shuffle,
            rng=np.random.default_rng(3),
        )
        n_batches = results["train_steps"] * results["accumulation_steps"]
        iterator = iter(results["train_dataset"])
        seen = []
        for _ in range(n_batches):
            (concat, true, false), y = next(iterator)
            ids = []
            for part, offset in ((concat, 0.0), (true, 1000.0), (false, 2000.0)):
                local = strategy.experimental_local_results(part)
                assert len(local) == 2, len(local)
                assert all(tuple(x.shape) == (4, 2, 3, 4) for x in local)
                part_ids = np.concatenate(
                    [x.numpy()[:, 0, 0, 0] for x in local]
                ) - offset
                ids.append(part_ids.astype(np.int64))
            # The three streams and y stay row-aligned across the replica split
            assert (ids[0] == ids[1]).all() and (ids[0] == ids[2]).all()
            y_ids = np.concatenate(
                [x.numpy()[:, 0, 0, 0] for x in strategy.experimental_local_results(y)]
            ).astype(np.int64)
            assert (ids[0] == y_ids).all()
            # Whole global batch, sorted (read locality) — replica r holds rows r*B:(r+1)*B
            assert (np.diff(ids[0]) >= 0).all()
            seen.extend(ids[0].tolist())
        if shuffle:
            assert sorted(seen) == sorted(results["train_indices"].tolist())
        else:
            # Exact alignment: concatenated local results reproduce train_indices order
            assert seen == results["train_indices"].tolist()

        val_iterator = iter(results["val_dataset"])
        val_seen = []
        for _ in range(results["val_steps"]):
            (concat, _, _), _ = next(val_iterator)
            val_seen.extend(
                np.concatenate(
                    [
                        x.numpy()[:, 0, 0, 0]
                        for x in strategy.experimental_local_results(concat)
                    ]
                )
                .astype(np.int64)
                .tolist()
            )
        assert val_seen == results["val_indices"].tolist()

        # Teardown in the order bench_input_pipeline documents (iterators -> datasets ->
        # holder.clear): dropping the holder first leaves from_generator threads blocked
        # mid-yield and the interpreter exit hangs (observed: all asserts pass, then hang)
        holder = results["_train_holder"]
        del iterator, val_iterator
        results = None
        import gc

        gc.collect()
        holder.clear()

    # ---- The accumulated train step's CROSS-REPLICA semantics (the part R=1 tests can't
    # see): gradient accumulators are aggregation=SUM and the update scales by
    # 1/(K * num_replicas). A mutation to aggregation=MEAN, or dropping num_replicas from
    # the scale, is invisible at R=1 and silently mis-scales every gradient by R on the
    # cluster — this pins the applied weight delta against hand-computed math at R=2.
    from types import SimpleNamespace

    from aetherscan.train import TrainingPipeline

    class StubModel(tf.Module):
        def __init__(self):
            super().__init__()
            with strategy.scope():
                self.w = tf.Variable(2.0, dtype=tf.float32, name="w")
                self.optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

        def compute_total_loss(self, main_data, true_data, false_data, y, training=False):
            del true_data, false_data, y, training
            # dL/dw = mean(main_batch): the gradient IS the per-replica batch mean, so the
            # applied update is exactly the mean over all K*R per-replica batch means
            total = self.w * tf.reduce_mean(main_data)
            return {
                "total_loss": total,
                "reconstruction_loss": total,
                "kl_loss": total,
                "true_loss": total,
                "false_loss": total,
                "kl_per_dim": tf.fill([2], total),
            }

    model = StubModel()
    tp = TrainingPipeline.__new__(TrainingPipeline)
    tp.strategy = strategy
    tp.vae = model
    tp.config = SimpleNamespace(beta_vae=SimpleNamespace(latent_dim=2))
    tp._grad_accumulators = None
    tp._train_loss_accumulators = None
    tp._val_loss_accumulator = None
    tp._unconnected_grad_indices = set()
    tp._accumulated_train_step_fns = {}
    tp._val_loop_fns = {}
    tp._ensure_accumulation_state()

    results = prepare_distributed_train_dataset(
        data=data,
        train_val_split=0.8,
        per_replica_batch_size=4,
        effective_batch_size=16,
        per_replica_val_batch_size=2,
        num_replicas=2,
        strategy=strategy,
        shuffle=False,
        rng=np.random.default_rng(3),
    )
    k = results["accumulation_steps"]
    assert k == 2, k
    iterator = iter(results["train_dataset"])
    step_fn = tp._get_accumulated_train_step(k)
    losses, global_norm, applied = step_fn(iterator)
    assert bool(applied.numpy())

    # Hand-computed reference from the known shuffle=False stream: micro-batch m gives
    # replica r rows train_indices[(m*2 + r)*4 : (m*2 + r + 1)*4], whose sample values
    # equal the row indices; grad = mean over the K*R per-replica means, SGD lr=1, w0=2.
    ti = results["train_indices"]
    per_replica_means = [ti[i * 4 : (i + 1) * 4].mean() for i in range(k * 2)]
    grad = float(np.mean(per_replica_means)) * 1.0  # dL/dw at w — w-independent here
    clip = min(1.0, 1.0 / abs(grad))  # tf.clip_by_global_norm at 1.0, single scalar grad
    expected_w = 2.0 - 1.0 * grad * clip
    got_w = float(strategy.experimental_local_results(model.w)[0].numpy())
    assert abs(got_w - expected_w) < 1e-4, (got_w, expected_w, grad)
    assert abs(float(global_norm.numpy()) - abs(grad)) < 1e-4
    # Loss accumulators are aggregation=MEAN across replicas, averaged over K in-graph
    expected_loss = 2.0 * float(np.mean(per_replica_means))
    assert abs(float(losses["total"].numpy()) - expected_loss) < 1e-3

    holder = results["_train_holder"]
    del iterator
    results = None
    gc.collect()
    holder.clear()

    print("DISTRIBUTION_OK", flush=True)
    """
)


def test_two_replica_split_matches_global_batch_order():
    env = dict(os.environ)
    repo_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    env["PYTHONPATH"] = os.path.abspath(repo_src) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only: logical-device split of CPU:0
    result = subprocess.run(  # noqa: PLW1510 — returncode asserted below
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "DISTRIBUTION_OK" in result.stdout
