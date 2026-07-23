#!/usr/bin/env python3
"""
GPU benchmark: Beta-VAE training step and encoder inference — throughput and peak VRAM,
single- or multi-GPU.

Unlike the other scripts in this directory (which time hot *CPU* kernels and run anywhere),
this one needs a physical GPU and the full TensorFlow / aetherscan stack, so run it inside the
NGC container on a cluster:

    # largest per-replica batch that fits one GPU, with throughput + VRAM at each size:
    ./utils/run_container.sh python benchmarks/bench_gpu.py --mode train  --find-max
    ./utils/run_container.sh python benchmarks/bench_gpu.py --mode encode --find-max
    # explicit sizes, and multi-GPU scaling across all 6 replicas:
    ./utils/run_container.sh python benchmarks/bench_gpu.py --mode train --batch-sizes 128,256
    ./utils/run_container.sh python benchmarks/bench_gpu.py --mode train --num-gpus 6 --batch-sizes 128
    # model the real accumulate-then-apply cadence (one apply per 4 micro-batches):
    ./utils/run_container.sh python benchmarks/bench_gpu.py --mode train --num-gpus 6 --batch-sizes 128 --accumulation-steps 4

It builds the real Beta-VAE (`create_beta_vae_model`) under a `MirroredStrategy` over the first
`--num-gpus` GPUs and drives synthetic batches of the true pipeline shapes. A training example is
a *cadence* `(6, 16, 512)`, so the four training inputs (`main`, `true`, `false`, `target`) are
each `(B, 6, 16, 512)` and one step runs `18*B` per-observation encoder forwards (`6B` main +
`6B` true + `6B` false) plus `6B` decoder forwards — the same work as `_distributed_train_step`.
Inference encodes individual observations, so `encode` mode drives `(B, 16, 512, 1)`.

For each per-replica batch `B` it reports **aggregate** throughput across the replicas
(train: cadences/s; encode: observations/s) and **per-GPU** peak VRAM. `--find-max` doubles `B`
until a GPU OOMs and reports the largest power of two that fits — use it to size
`--per-replica-batch-size` to the card (VRAM is per GPU; each replica holds a full model copy).

What this does and does NOT cover:
  - Covers the VAE training step (composite loss + backprop + clipped Adam), encoder inference,
    the decoder (inside the train step), and — with `--num-gpus > 1` — the MirroredStrategy
    cross-replica gradient all-reduce and multi-GPU scaling. fp32, matching the pipeline (which
    uses no mixed precision or XLA).
  - Models gradient accumulation with `--accumulation-steps K`: one optimizer step accumulates
    all-reduced grads over K micro-batches then applies once with the global-norm clip, exactly as
    train.py's `_train_epoch` (K=1, the default, is a plain apply-every-step). Peak VRAM then
    includes the persistent accumulator and throughput reflects the once-per-K apply cadence.
  - Does NOT model the input pipeline (synthetic in-memory tensors, no `tf.data`-from-memmap +
    prefetch + host->device copy) or the CPU-side Random Forest stage (see bench_rf.py).

Note: `--find-max` is capped at `--max-batch` (default 4096) because an encoder forward whose
conv feature maps exceed ~2^31 elements (roughly batch >= 8192, an intermediate map is
batch*16*512*channels) trips a TensorFlow int32 launch-config overflow that aborts the process
uncatchably rather than raising a clean OOM. 4096 covers the training VRAM ceiling and a generous
inference range (the pipeline's inference default is 2048).

Writes a JSON result to benchmarks/results/ (or --output), like the CPU benchmarks.
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import tensorflow as tf
from _common import machine_info, write_result

from aetherscan.config import get_config, init_config
from aetherscan.models import create_beta_vae_model

# Fixed pipeline shapes (see models/vae.py). A training example is a cadence of 6 observations;
# BetaVAE.call reshapes (B, 6, 16, 512) -> (B*6, 16, 512, 1) to encode. Inference encodes single
# observations of shape (16, 512, 1).
_CADENCE_SHAPE = (6, 16, 512)
_OBS_SHAPE = (16, 512, 1)
_CLIP_NORM = 1.0  # matches train.py `_apply_gradients` (tf.clip_by_global_norm(..., 1.0))
# Module-level RNG so successive _rand() draws differ (main/true/false/target aren't byte-identical).
# Values don't affect GPU timing or VRAM (both shape-dependent) — this only avoids a degenerate loss.
_RNG = np.random.default_rng(0)


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return ivalue


def _batch_list(value: str) -> list[int]:
    try:
        sizes = [int(x) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--batch-sizes must be comma-separated integers, got {value!r}"
        ) from exc
    if not sizes or any(b < 1 for b in sizes):
        raise argparse.ArgumentTypeError(f"--batch-sizes must be positive integers, got {value!r}")
    return sizes


def _select_gpus(num_gpus: int) -> list[str]:
    """Restrict TF to the first `num_gpus` visible GPUs with memory growth on (mirrors main.py so
    get_memory_info reflects real usage rather than a pre-grabbed pool). Returns device names."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("bench_gpu requires a physical GPU; none are visible.")
    if num_gpus > len(gpus):
        raise SystemExit(f"--num-gpus {num_gpus} requested but only {len(gpus)} GPU(s) visible.")
    use = gpus[:num_gpus]
    tf.config.set_visible_devices(use, "GPU")
    for gpu in use:
        tf.config.experimental.set_memory_growth(gpu, True)
    return [f"GPU:{i}" for i in range(num_gpus)]


def _rand(shape: tuple[int, ...]) -> np.ndarray:
    return _RNG.random(shape, dtype=np.float32)


def _peak_vram_gb(devices: list[str]) -> float:
    # SI GB (1e9 B); nvidia-smi reports base-2 MiB/GiB (GiB = GB / 1.074). Peak across replicas,
    # which are ~equal. get_memory_info tracks real allocation because memory growth is on.
    return max(tf.config.experimental.get_memory_info(d)["peak"] for d in devices) / 1e9


def _make_encode_step_fn(model):
    """Per-replica encoder forward, run via strategy.run for `--mode encode`. (The train path builds
    its own gradient-accumulating step inside _measure, so it does not use this.)"""

    def step_fn(inputs):
        z_mean, _, _ = model.encoder(inputs[0], training=False)
        return z_mean

    return step_fn


def _measure(
    strategy, devices, model, step_fn, mode, batch, warmup, steps, accumulation_steps
) -> dict | None:
    """Run `warmup` + `steps` optimizer steps at per-replica batch `batch` on every replica. A
    train optimizer step accumulates all-reduced grads over `accumulation_steps` micro-batches then
    applies once with the global-norm clip (train.py's `_train_epoch`); an encode step is a single
    encoder forward (`accumulation_steps` ignored). Returns aggregate throughput (per_replica_batch
    * num_replicas * micro_batches / elapsed) and per-GPU peak VRAM, or None if a GPU OOMs (the
    caller stops the sweep there)."""
    shape = _CADENCE_SHAPE if mode == "train" else _OBS_SHAPE
    n_inputs = 4 if mode == "train" else 1

    def value_fn(_ctx):
        return tuple(tf.constant(_rand((batch, *shape))) for _ in range(n_inputs))

    @tf.function
    def encode_run(dist_inputs):
        return strategy.run(step_fn, args=(dist_inputs,))

    if mode == "train":
        # Per-replica gradient accumulators (created in scope so they mirror across replicas).
        with strategy.scope():
            accumulators = [
                tf.Variable(tf.zeros_like(v), trainable=False) for v in model.trainable_variables
            ]

        def _accumulate(inputs):
            main, true_, false_, target = inputs
            with tf.GradientTape() as tape:
                losses = model.compute_total_loss(main, true_, false_, target, training=True)
            grads = tape.gradient(losses["total_loss"], model.trainable_variables)
            for acc, g in zip(accumulators, grads, strict=False):
                acc.assign_add(g)

        def _apply_and_reset():
            grads = [acc / accumulation_steps for acc in accumulators]
            clipped, _ = tf.clip_by_global_norm(grads, _CLIP_NORM)
            model.optimizer.apply_gradients(zip(clipped, model.trainable_variables, strict=False))
            for acc in accumulators:
                acc.assign(tf.zeros_like(acc))

        @tf.function
        def run(dist_inputs):
            # Accumulate grads over accumulation_steps micro-batches into the per-replica
            # accumulators, then apply once. The apply runs INSIDE strategy.run, so MirroredStrategy
            # all-reduces on apply — the correct multi-GPU accumulate-then-apply cadence, and it
            # sidesteps the cross-device error from applying pre-reduced grads in cross-replica context.
            for _ in range(accumulation_steps):
                strategy.run(_accumulate, args=(dist_inputs,))
            strategy.run(_apply_and_reset)
    else:
        run = encode_run

    def sync(out) -> None:
        # Force queued device work to finish before the clock is read. Train applies mutate the
        # model variables, so reading one is a sufficient barrier; encode syncs on its output.
        if mode == "train":
            model.trainable_variables[0].numpy()
        elif out is not None:
            strategy.experimental_local_results(out)[0].numpy()

    try:
        dist_inputs = strategy.experimental_distribute_values_from_function(value_fn)
        out = None
        for _ in range(warmup):
            out = run(dist_inputs)
        sync(out)  # also absorbs tracing
        for d in devices:
            tf.config.experimental.reset_memory_stats(d)
        start = time.perf_counter()
        for _ in range(steps):
            out = run(dist_inputs)
        sync(out)  # force queued steps to finish
        elapsed = time.perf_counter() - start
    except tf.errors.ResourceExhaustedError:
        return None

    peak = _peak_vram_gb(devices)
    gc.collect()
    n = len(devices)
    micro_batches = steps * (accumulation_steps if mode == "train" else 1)
    aggregate = (batch * n * micro_batches) / elapsed if elapsed > 0 else float("inf")
    return {
        "per_replica_batch_size": batch,
        "num_gpus": n,
        "steps": steps,
        "accumulation_steps": accumulation_steps if mode == "train" else 1,
        "elapsed_s": elapsed,
        "aggregate_samples_per_s": aggregate,
        "peak_vram_gb_per_gpu": peak,
    }


def _batch_schedule(args) -> list[int]:
    if args.batch_sizes:
        return args.batch_sizes
    # --find-max (explicit or the default when --batch-sizes is unset): powers of two from
    # --start-batch to the cap; the sweep stops at the first OOM, so the last success is the
    # largest power of two that fits.
    sizes, b = [], args.start_batch
    while b <= args.max_batch:
        sizes.append(b)
        b *= 2
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "encode"], default="train")
    parser.add_argument(
        "--num-gpus",
        type=_positive_int,
        default=1,
        help="Replica count (MirroredStrategy over the first N visible GPUs). Default 1.",
    )
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--batch-sizes",
        type=_batch_list,
        default=None,
        help="Comma-separated per-replica batch sizes to measure (mutually exclusive with --find-max).",
    )
    size_group.add_argument(
        "--find-max",
        action="store_true",
        help="Double the per-replica batch until a GPU OOMs (the default when --batch-sizes is unset).",
    )
    parser.add_argument(
        "--start-batch",
        type=_positive_int,
        default=64,
        help="First per-replica size for --find-max.",
    )
    parser.add_argument(
        "--max-batch",
        type=_positive_int,
        default=4096,
        help="Cap for --find-max (default 4096; see the int32 launch-config note in the header).",
    )
    parser.add_argument(
        "--warmup", type=_positive_int, default=3, help="Untimed warmup steps per size."
    )
    parser.add_argument("--steps", type=_positive_int, default=20, help="Timed steps per size.")
    parser.add_argument(
        "--accumulation-steps",
        type=_positive_int,
        default=1,
        help="Train only: micro-batches accumulated per optimizer step (mirrors _train_epoch). "
        "Default 1 = apply every step. Ignored for --mode encode.",
    )
    parser.add_argument("--output", default=None, help="Result JSON path.")
    args = parser.parse_args()

    devices = _select_gpus(args.num_gpus)
    init_config()
    config = get_config()
    # Build the model once (weights are batch-independent) and reuse across sizes.
    strategy = tf.distribute.MirroredStrategy(devices=[f"/{d}" for d in devices])
    with strategy.scope():
        model = create_beta_vae_model()
    step_fn = _make_encode_step_fn(model)

    unit = "cadences/s" if args.mode == "train" else "obs/s"
    shape = _CADENCE_SHAPE if args.mode == "train" else _OBS_SHAPE
    train_note = (
        f"  accumulation_steps={args.accumulation_steps}  "
        f"(18*B encoder + 6*B decoder forwards/micro-batch)"
        if args.mode == "train"
        else ""
    )
    print(
        f"mode={args.mode}  num_gpus={len(devices)}  latent_dim={config.beta_vae.latent_dim}  "
        f"example_shape={shape}" + train_note
    )
    print(
        f"throughput = aggregate across {len(devices)} GPU(s); VRAM = peak per GPU (SI GB, 1e9 B)"
    )
    print(f"{'batch':>8}  {'agg ' + unit:>16}  {'VRAM/GPU (GB)':>14}")

    measured: list[dict] = []
    max_fit: dict | None = None
    for batch in _batch_schedule(args):
        res = _measure(
            strategy,
            devices,
            model,
            step_fn,
            args.mode,
            batch,
            args.warmup,
            args.steps,
            args.accumulation_steps,
        )
        if res is None:
            print(f"{batch:>8}  {'OOM':>16}  {'-':>14}")
            break
        measured.append(res)
        max_fit = res
        print(
            f"{batch:>8}  {res['aggregate_samples_per_s']:>16.0f}  {res['peak_vram_gb_per_gpu']:>14.2f}"
        )

    results = {
        "measurements": measured,
        "max_fit_batch": max_fit["per_replica_batch_size"] if max_fit else None,
        "max_fit_peak_vram_gb_per_gpu": max_fit["peak_vram_gb_per_gpu"] if max_fit else None,
    }
    if max_fit:
        print(
            f"\nlargest fitting per-replica batch: {max_fit['per_replica_batch_size']} "
            f"({max_fit['peak_vram_gb_per_gpu']:.2f} GB/GPU peak, "
            f"{max_fit['aggregate_samples_per_s']:.0f} agg {unit})"
        )
    path = write_result(
        f"bench_gpu_{args.mode}",
        {
            "mode": args.mode,
            "num_gpus": len(devices),
            "example_shape": list(shape),
            "latent_dim": config.beta_vae.latent_dim,
            "warmup": args.warmup,
            "steps": args.steps,
            "accumulation_steps": args.accumulation_steps if args.mode == "train" else 1,
            "gpu": machine_info()["hostname"],
        },
        results,
        args.output,
    )
    print(f"Result written to {path}")


if __name__ == "__main__":
    main()
