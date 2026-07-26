#!/usr/bin/env python3
"""
Input-pipeline benchmark: the real memmap -> tf.data -> distribute -> train-step path (#276).

`bench_gpu.py` drives synthetic in-memory tensors, so it measures the GPUs' ceiling but not the
host-side input path that feeds them — the path the 2026-07 release run showed to be the actual
bottleneck (GPUs idle ~75-80% of every epoch; see issue #276). This script closes that gap: it
synthesizes an on-disk round of the true production layout (three float32 memmaps of shape
(N, 6, 16, 512) + labels), then measures the pipeline at three depths:

    gather   raw numpy fancy-index gathers from the memmap, single-threaded — the lower bound
             for one batch's host work, isolating numpy/page-cache from tf.data.
    iterate  tf.data throughput of the full dataset builder (no GPU consumption): batches/s and
             effective MB/s. `--variant legacy` reproduces the pre-#276 single-generator builder
             verbatim; `--variant current` uses train.prepare_distributed_train_dataset as it is
             today (index generators + parallel deterministic gather map). Same seed => both
             variants produce identical batch streams, so the delta is pure pipeline overhead.
    step     end-to-end: the real Beta-VAE training step (same accumulate-then-apply cadence as
             train.py's _train_epoch) consuming the real input pipeline on --num-gpus GPUs.
             Compare cadences/s against bench_gpu.py's synthetic number on the same host: the
             gap between them IS the input-pipeline cost.

Contention/determinism knobs for the #276 audit:
    --gil-load N          spawn N background threads doing write_injection_stat-shaped pure-Python
                          work (dict/tuple packing + queue puts), quantifying the GIL interaction
                          with the DB drainer flood documented in issue #277.
    --deterministic-ops   call tf.config.experimental.enable_op_determinism() first, isolating the
                          cost of --tf-deterministic-ops on the input path.
    --profile DIR         (step mode) capture a TensorFlow profiler trace over the timed steps —
                          the authoritative host-vs-device timeline; open with TensorBoard.

Cluster usage (the synthetic round defaults to ~12 GB on disk — point --data-dir at scratch):

    ./utils/run_container.sh python benchmarks/bench_input_pipeline.py --mode gather \
        --data-dir /datax/scratch/$USER/bench_input
    ./utils/run_container.sh python benchmarks/bench_input_pipeline.py --mode iterate \
        --variant legacy --data-dir /datax/scratch/$USER/bench_input
    ./utils/run_container.sh python benchmarks/bench_input_pipeline.py --mode iterate \
        --variant current --data-dir /datax/scratch/$USER/bench_input
    ./utils/run_container.sh python benchmarks/bench_input_pipeline.py --mode step \
        --variant current --num-gpus 5 --data-dir /datax/scratch/$USER/bench_input

The synthetic arrays are written once and reused across invocations (delete with --regen or by
removing --data-dir). After the first write the pages are OS-page-cache resident, matching the
live-run regime the audit measured (wa=0, bi=0, round resident in cache).

Writes a JSON result to benchmarks/results/ (or --output), like the other benchmarks.
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import threading
import time

import numpy as np
from _common import machine_info, write_result

_CADENCE_SHAPE = (6, 16, 512)
_SAMPLE_BYTES = int(np.prod(_CADENCE_SHAPE)) * 4  # float32
_SIGNAL_TYPES = ("false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi")
_CLIP_NORM = 1.0  # matches train.py `_apply_gradients`


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return ivalue


def _synthesize_round(data_dir: str, n_samples: int, regen: bool) -> dict:
    """Create (or reuse) the three on-disk memmaps + labels of a production-layout round."""
    os.makedirs(data_dir, exist_ok=True)
    paths = {
        name: os.path.join(data_dir, f"{name}.npy") for name in ("concatenated", "true", "false")
    }
    labels_path = os.path.join(data_dir, "labels.npy")

    def _stale(path: str) -> bool:
        if not os.path.exists(path):
            return True
        try:
            return np.load(path, mmap_mode="r").shape[0] != n_samples
        except Exception:
            return True

    if regen or any(_stale(p) for p in paths.values()) or _stale(labels_path):
        rng = np.random.default_rng(0)
        for path in paths.values():
            print(f"writing {path} ({n_samples} samples, {n_samples * _SAMPLE_BYTES / 1e9:.1f} GB)")
            arr = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(n_samples, *_CADENCE_SHAPE)
            )
            chunk = 2048
            for start in range(0, n_samples, chunk):
                stop = min(start + chunk, n_samples)
                arr[start:stop] = rng.random((stop - start, *_CADENCE_SHAPE), dtype=np.float32)
            arr.flush()
            del arr
        labels = np.array(
            [_SIGNAL_TYPES[i % len(_SIGNAL_TYPES)] for i in range(n_samples)], dtype="U20"
        )
        np.save(labels_path, labels)

    return {
        "concatenated": np.load(paths["concatenated"], mmap_mode="r"),
        "true": np.load(paths["true"], mmap_mode="r"),
        "false": np.load(paths["false"], mmap_mode="r"),
        "labels": np.load(labels_path),
    }


def _legacy_prepare(
    data,
    train_val_split,
    per_replica_batch_size,
    effective_batch_size,
    per_replica_val_batch_size,
    num_replicas,
    strategy,
    shuffle=True,
    rng=None,
):
    """
    Verbatim replica of the PRE-#276 prepare_distributed_train_dataset dataset construction
    (master @ 0e7e6ab): a single Python generator does the per-batch triple memmap gather,
    then from_generator(...).repeat().prefetch(AUTOTUNE). Split/trim/shuffle logic is identical
    to the current builder, so with the same seeded rng both variants yield the same batches.
    """
    import tensorflow as tf  # noqa: PLC0415

    from aetherscan.train import TrainDataHolder  # noqa: PLC0415

    global_train_batch_size = per_replica_batch_size * num_replicas
    global_val_batch_size = per_replica_val_batch_size * num_replicas
    if rng is None:
        rng = np.random.default_rng()

    labels = np.asarray(data["labels"])
    train_indices, val_indices = [], []
    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
        n_label_train = int(len(label_indices) * train_val_split)
        train_indices.append(label_indices[:n_label_train])
        val_indices.append(label_indices[n_label_train:])
    train_indices = np.concatenate(train_indices)
    val_indices = np.concatenate(val_indices)

    n_train_trimmed = (len(train_indices) // effective_batch_size) * effective_batch_size
    n_val_trimmed = (len(val_indices) // global_val_batch_size) * global_val_batch_size
    if n_train_trimmed < len(train_indices):
        train_indices = rng.choice(train_indices, size=n_train_trimmed, replace=False)
    if n_val_trimmed < len(val_indices):
        val_indices = rng.choice(val_indices, size=n_val_trimmed, replace=False)
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)

    train_holder = TrainDataHolder(data["concatenated"], data["true"], data["false"])

    def train_generator():
        while True:
            with train_holder._lock:
                if train_holder._cleared:
                    return
                concat = train_holder.concat
                true = train_holder.true
                false = train_holder.false
            indices = train_indices.copy()
            if shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), global_train_batch_size):
                batch_indices = indices[start : start + global_train_batch_size]
                if shuffle:
                    batch_indices = np.sort(batch_indices)
                concat_batch = concat[batch_indices]
                yield (concat_batch, true[batch_indices], false[batch_indices]), concat_batch
            del concat, true, false

    sample_shape = data["concatenated"].shape[1:]
    batch_spec = tf.TensorSpec(shape=(None, *sample_shape), dtype=tf.float32)
    output_signature = ((batch_spec, batch_spec, batch_spec), batch_spec)

    train_dataset = (
        tf.data.Dataset.from_generator(train_generator, output_signature=output_signature)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    return {
        "train_dataset": strategy.experimental_distribute_dataset(train_dataset),
        "n_train_trimmed": n_train_trimmed,
        "train_steps": n_train_trimmed // effective_batch_size,
        "accumulation_steps": effective_batch_size // global_train_batch_size,
        "_train_holder": train_holder,
    }


def _build_datasets(variant, data, args, strategy, num_replicas):
    rng = np.random.default_rng(42)
    kwargs = {
        "data": data,
        "train_val_split": 0.8,
        "per_replica_batch_size": args.per_replica_batch_size,
        "effective_batch_size": args.per_replica_batch_size
        * num_replicas
        * args.accumulation_steps,
        "per_replica_val_batch_size": 64,
        "num_replicas": num_replicas,
        "strategy": strategy,
        "shuffle": True,
        "rng": rng,
    }
    if variant == "legacy":
        return _legacy_prepare(**kwargs)
    from aetherscan.train import prepare_distributed_train_dataset  # noqa: PLC0415

    return prepare_distributed_train_dataset(**kwargs)


def _gil_hog(stop_event: threading.Event) -> None:
    """Pure-Python work shaped like the DB drainer's write_injection_stat flood (#277)."""
    q: queue.Queue = queue.Queue()
    while not stop_event.is_set():
        for i in range(1000):
            metadata = {"hostname": "bench", "ip": "0.0.0.0", "i": i}
            q.put(
                (
                    "injection_stats",
                    (
                        time.time(),
                        "intensity_mean",
                        1.0,
                        1,
                        2,
                        i,
                        3,
                        "true_only_eti",
                        "narrowband",
                        "post",
                        1,
                        0,
                        "bench",
                        str(metadata),
                    ),
                )
            )
        while not q.empty():
            q.get()


def _run_gather(data, args) -> dict:
    """Raw single-threaded numpy gather throughput (the host-work lower bound per batch)."""
    rng = np.random.default_rng(42)
    n = data["concatenated"].shape[0]
    global_batch = args.per_replica_batch_size * args.num_gpus
    concat, true, false = data["concatenated"], data["true"], data["false"]

    def one_batch():
        idx = np.sort(rng.choice(n, size=global_batch, replace=False))
        return concat[idx], true[idx], false[idx]

    for _ in range(args.warmup):
        one_batch()
    start = time.perf_counter()
    for _ in range(args.steps):
        one_batch()
    elapsed = time.perf_counter() - start
    batches_per_s = args.steps / elapsed
    return {
        "batches_per_s": batches_per_s,
        "samples_per_s": batches_per_s * global_batch,
        "gathered_mb_per_s": batches_per_s * global_batch * _SAMPLE_BYTES * 3 / 1e6,
        "elapsed_s": elapsed,
    }


def _run_iterate(data, args, variant) -> dict:
    """tf.data pipeline throughput without GPU consumption (micro-batches per second)."""
    import tensorflow as tf  # noqa: PLC0415

    strategy = tf.distribute.get_strategy()  # default (no-op) strategy: measures the producer
    built = _build_datasets(variant, data, args, strategy, num_replicas=1)
    iterator = iter(built["train_dataset"])
    global_batch = args.per_replica_batch_size

    for _ in range(args.warmup):
        next(iterator)
    start = time.perf_counter()
    for _ in range(args.steps):
        next(iterator)
    elapsed = time.perf_counter() - start
    # Teardown order mirrors train.py (_train_epoch / train_round finally blocks): drop the
    # iterator and dataset BEFORE clearing the holder, so tf.data finalizes the generator
    # thread instead of leaving it blocked mid-yield at interpreter exit (observed hang)
    del iterator
    built.pop("train_dataset", None)
    built["_train_holder"].clear()
    import gc  # noqa: PLC0415

    gc.collect()
    batches_per_s = args.steps / elapsed
    return {
        "variant": variant,
        "batches_per_s": batches_per_s,
        "samples_per_s": batches_per_s * global_batch,
        "gathered_mb_per_s": batches_per_s * global_batch * _SAMPLE_BYTES * 3 / 1e6,
        "elapsed_s": elapsed,
    }


def _run_step(data, args, variant) -> dict:
    """End-to-end: real VAE train step (accumulate-then-apply cadence) fed by the pipeline."""
    import tensorflow as tf  # noqa: PLC0415

    from aetherscan.models import create_beta_vae_model  # noqa: PLC0415

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("--mode step requires physical GPUs; none are visible.")
    if args.num_gpus > len(gpus):
        raise SystemExit(f"--num-gpus {args.num_gpus} requested but only {len(gpus)} visible.")
    use = gpus[: args.num_gpus]
    tf.config.set_visible_devices(use, "GPU")
    for gpu in use:
        tf.config.experimental.set_memory_growth(gpu, True)

    strategy = tf.distribute.MirroredStrategy(devices=[f"/GPU:{i}" for i in range(args.num_gpus)])
    with strategy.scope():
        model = create_beta_vae_model()

    built = _build_datasets(variant, data, args, strategy, num_replicas=args.num_gpus)
    iterator = iter(built["train_dataset"])
    accumulation_steps = built["accumulation_steps"]
    global_batch = args.per_replica_batch_size * args.num_gpus

    @tf.function
    def micro_step(batch):
        # Mirrors train.py's _distributed_train_step: per-replica loss+grads, MEAN-reduced
        def step_fn(batch_data):
            (main, true_, false_), target = batch_data
            with tf.GradientTape() as tape:
                losses = model.compute_total_loss(main, true_, false_, target, training=True)
            grads = tape.gradient(losses["total_loss"], model.trainable_variables)
            return losses["total_loss"], grads

        per_loss, per_grads = strategy.run(step_fn, args=(batch,))
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_loss, axis=None)
        grads = [strategy.reduce(tf.distribute.ReduceOp.MEAN, g, axis=None) for g in per_grads]
        return loss, grads

    def optimizer_step():
        # Mirrors _train_epoch: sum reduced grads over accumulation sub-steps in Python,
        # average, clip by global norm, apply once
        accumulated = None
        for _ in range(accumulation_steps):
            _, grads = micro_step(next(iterator))
            if accumulated is None:
                accumulated = grads
            else:
                accumulated = [a + g for a, g in zip(accumulated, grads, strict=False)]
        averaged = [a / accumulation_steps for a in accumulated]
        clipped, _ = tf.clip_by_global_norm(averaged, _CLIP_NORM)
        model.optimizer.apply_gradients(zip(clipped, model.trainable_variables, strict=False))

    for _ in range(args.warmup):
        optimizer_step()
    model.trainable_variables[0].numpy()  # drain queued device work (see bench_gpu.py sync note)

    if args.profile:
        tf.profiler.experimental.start(args.profile)
    start = time.perf_counter()
    for _ in range(args.steps):
        optimizer_step()
    model.trainable_variables[0].numpy()
    elapsed = time.perf_counter() - start
    if args.profile:
        tf.profiler.experimental.stop()

    # Same teardown order as _run_iterate (and train.py): iterator/dataset first, then
    # clear (None-assignment rather than del: optimizer_step closes over the name)
    iterator = None  # noqa: F841
    built.pop("train_dataset", None)
    built["_train_holder"].clear()
    import gc  # noqa: PLC0415

    gc.collect()
    micro_batches = args.steps * accumulation_steps
    return {
        "variant": variant,
        "num_gpus": args.num_gpus,
        "accumulation_steps": accumulation_steps,
        "optimizer_steps_per_s": args.steps / elapsed,
        "aggregate_cadences_per_s": micro_batches * global_batch / elapsed,
        "elapsed_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gather", "iterate", "step"], required=True)
    parser.add_argument(
        "--variant",
        choices=["legacy", "current"],
        default="current",
        help="Dataset builder: 'legacy' = pre-#276 single-generator replica; "
        "'current' = train.prepare_distributed_train_dataset as checked out.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "bench_input_pipeline"),
        help="Where the synthetic round memmaps live (~12 GB at the default --n-samples).",
    )
    parser.add_argument("--n-samples", type=_positive_int, default=20_000)
    parser.add_argument("--regen", action="store_true", help="Rewrite the synthetic round.")
    parser.add_argument(
        "--cleanup", action="store_true", help="Delete the synthetic round when done."
    )
    parser.add_argument("--per-replica-batch-size", type=_positive_int, default=128)
    parser.add_argument("--accumulation-steps", type=_positive_int, default=1)
    parser.add_argument("--num-gpus", type=_positive_int, default=1)
    parser.add_argument("--warmup", type=_positive_int, default=5)
    parser.add_argument("--steps", type=_positive_int, default=50)
    parser.add_argument(
        "--gil-load",
        type=int,
        default=0,
        help="Background threads doing DB-drainer-shaped pure-Python work during the "
        "timed loop (quantifies the #277 GIL interaction).",
    )
    parser.add_argument(
        "--deterministic-ops",
        action="store_true",
        help="Enable tf.config.experimental.enable_op_determinism() first.",
    )
    parser.add_argument(
        "--profile", default=None, help="(step mode) TF profiler trace output directory."
    )
    parser.add_argument("--output", default=None, help="Result JSON path.")
    args = parser.parse_args()

    data = _synthesize_round(args.data_dir, args.n_samples, args.regen)

    if args.mode != "gather":
        import tensorflow as tf  # noqa: PLC0415  (deferred so gather mode stays numpy-only)

        from aetherscan.config import init_config  # noqa: PLC0415

        init_config()
        if args.deterministic_ops:
            tf.config.experimental.enable_op_determinism()

    stop_event = threading.Event()
    hogs = [
        threading.Thread(target=_gil_hog, args=(stop_event,), daemon=True)
        for _ in range(args.gil_load)
    ]
    for hog in hogs:
        hog.start()

    try:
        if args.mode == "gather":
            results = _run_gather(data, args)
        elif args.mode == "iterate":
            results = _run_iterate(data, args, args.variant)
        else:
            results = _run_step(data, args, args.variant)
    finally:
        stop_event.set()
        for hog in hogs:
            hog.join(timeout=5)

    results["gil_load_threads"] = args.gil_load
    results["deterministic_ops"] = args.deterministic_ops

    for key, value in results.items():
        print(f"{key}: {value}")

    path = write_result(
        f"bench_input_pipeline_{args.mode}",
        {
            "mode": args.mode,
            "variant": args.variant if args.mode != "gather" else None,
            "n_samples": args.n_samples,
            "per_replica_batch_size": args.per_replica_batch_size,
            "accumulation_steps": args.accumulation_steps,
            "num_gpus": args.num_gpus,
            "warmup": args.warmup,
            "steps": args.steps,
            "host": machine_info()["hostname"],
        },
        results,
        args.output,
    )
    print(f"Result written to {path}")

    if args.cleanup:
        shutil.rmtree(args.data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
