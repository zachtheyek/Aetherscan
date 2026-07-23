#!/usr/bin/env python3
"""
GPU micro-benchmark: Beta-VAE training step and encoder inference — throughput and peak VRAM.

Unlike the other scripts in this directory (which time hot *CPU* kernels and run anywhere),
this one needs a physical GPU and the full TensorFlow / aetherscan stack, so run it inside the
NGC container on a cluster:

    ./utils/run_container.sh python benchmarks/bench_gpu.py --mode train  --find-max
    ./utils/run_container.sh python benchmarks/bench_gpu.py --mode encode --find-max
    ./utils/run_container.sh python benchmarks/bench_gpu.py --mode train  --batch-sizes 128,256,512

It builds the real Beta-VAE (`create_beta_vae_model`) on a single GPU and drives synthetic
batches of the true pipeline shapes. A training example is a *cadence* `(6, 16, 512)`, so the
four training inputs (`main`, `true`, `false`, `target`) are each `(B, 6, 16, 512)` and one step
runs `18*B` encoder passes (`6B` main + `6B` true + `6B` false) plus `6B` decoder passes,
exactly like `_distributed_train_step`. Inference encodes individual observations, so `encode`
mode drives `(B, 16, 512, 1)`. For each per-replica batch size it reports throughput
(train: cadences/s; encode: observations/s) and peak VRAM.

`--find-max` doubles the batch size until the GPU OOMs and reports the largest power of two that
fits — use it to size `--per-replica-batch-size` (train) and `--per-replica-batch-size`
(inference) so a large-VRAM card (e.g. a 96 GB Blackwell) is not left training at a batch sized
for a 16 GB card. Peak VRAM is per GPU (each replica holds a full copy), so the combined figure
is this number times the replica count.

Note: the sweep is capped at `--max-batch` (default 4096) because a single encoder forward pass
whose conv feature maps exceed ~2^31 elements (roughly batch >= 8192, since an intermediate map
is batch*16*512*channels) trips a TensorFlow int32 launch-config overflow that aborts the process
uncatchably rather than raising a clean OOM. 4096 comfortably covers the training VRAM ceiling and
a generous inference range (the pipeline's inference default is 2048).

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
_DEVICE = "GPU:0"


def _select_single_gpu() -> None:
    """Restrict TF to the first visible GPU with memory growth on, mirroring main.py so that
    `get_memory_info` reflects real usage rather than a pre-grabbed whole-device pool."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("bench_gpu requires a physical GPU; none are visible.")
    tf.config.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)


def _rand(shape: tuple[int, ...]) -> tf.Tensor:
    return tf.constant(np.random.default_rng(0).random(shape, dtype=np.float32))


def _peak_vram_gb() -> float:
    return tf.config.experimental.get_memory_info(_DEVICE)["peak"] / 1e9


def _reset_vram_peak() -> None:
    tf.config.experimental.reset_memory_stats(_DEVICE)


def _make_train_step(model):
    @tf.function
    def train_step(main, true_, false_, target):
        with tf.GradientTape() as tape:
            losses = model.compute_total_loss(main, true_, false_, target, training=True)
        grads = tape.gradient(losses["total_loss"], model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, _CLIP_NORM)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables, strict=False))
        return losses["total_loss"]

    return train_step


def _make_encode_step(model):
    @tf.function
    def encode_step(obs):
        z_mean, _, _ = model.encoder(obs, training=False)
        return z_mean

    return encode_step


def _measure(mode: str, step_fn, batch: int, warmup: int, steps: int) -> dict | None:
    """Run `warmup` + `steps` iterations at per-replica batch size `batch`; return timing/VRAM,
    or None if the GPU ran out of memory (the caller stops the sweep there)."""
    if mode == "train":
        # main, true, false, target — each a batch of cadences (B, 6, 16, 512).
        inputs = tuple(_rand((batch, *_CADENCE_SHAPE)) for _ in range(4))
    else:
        # encode: a batch of single observations (B, 16, 512, 1).
        inputs = (_rand((batch, *_OBS_SHAPE)),)

    try:
        for _ in range(warmup):
            out = step_fn(*inputs)
        out.numpy()  # sync after warmup (also absorbs tf.function tracing) before timing
        _reset_vram_peak()
        start = time.perf_counter()
        for _ in range(steps):
            out = step_fn(*inputs)
        out.numpy()  # force the queued steps to finish before stopping the clock
        elapsed = time.perf_counter() - start
    except tf.errors.ResourceExhaustedError:
        # `inputs` is dropped when this frame returns; the sweep stops at the first OOM.
        return None

    peak = _peak_vram_gb()
    del inputs
    gc.collect()
    # batch dim = cadences (train) / observations (encode); see the header line for the unit.
    samples_per_s = (batch * steps) / elapsed if elapsed > 0 else float("inf")
    return {
        "per_replica_batch_size": batch,
        "steps": steps,
        "elapsed_s": elapsed,
        "samples_per_s": samples_per_s,
        "peak_vram_gb": peak,
    }


def _batch_schedule(args) -> list[int]:
    if args.batch_sizes:
        return [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    # --find-max (default): powers of two from --start-batch up to the cap; the sweep stops at
    # the first OOM, so the last success is the largest power of two that fits.
    sizes, b = [], args.start_batch
    while b <= args.max_batch:
        sizes.append(b)
        b *= 2
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "encode"], default="train")
    parser.add_argument(
        "--batch-sizes",
        default=None,
        help="Comma-separated per-replica batch sizes to measure (overrides --find-max).",
    )
    parser.add_argument(
        "--find-max",
        action="store_true",
        help="Double the batch size until the GPU OOMs (the default when --batch-sizes is unset).",
    )
    parser.add_argument("--start-batch", type=int, default=64, help="First size for --find-max.")
    parser.add_argument(
        "--max-batch",
        type=int,
        default=4096,
        help="Cap for --find-max (default 4096; see the int32 launch-config note in the header).",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmup steps per size.")
    parser.add_argument("--steps", type=int, default=20, help="Timed steps per size.")
    parser.add_argument("--output", default=None, help="Result JSON path.")
    args = parser.parse_args()

    _select_single_gpu()
    init_config()
    config = get_config()
    # Build the model once (weights are batch-independent); reuse across sizes.
    strategy = tf.distribute.OneDeviceStrategy(f"/{_DEVICE}")
    with strategy.scope():
        model = create_beta_vae_model()
    step_fn = _make_train_step(model) if args.mode == "train" else _make_encode_step(model)

    unit = "cadences/s" if args.mode == "train" else "obs/s"
    shape = _CADENCE_SHAPE if args.mode == "train" else _OBS_SHAPE
    print(
        f"mode={args.mode}  latent_dim={config.beta_vae.latent_dim}  example_shape={shape}"
        + ("  (18*B encoder + 6*B decoder passes/step)" if args.mode == "train" else "")
    )
    print(f"{'batch':>8}  {unit:>12}  {'peak VRAM (GB)':>15}")

    measured: list[dict] = []
    max_fit: dict | None = None
    for batch in _batch_schedule(args):
        res = _measure(args.mode, step_fn, batch, args.warmup, args.steps)
        if res is None:
            print(f"{batch:>8}  {'OOM':>12}  {'-':>15}")
            break
        measured.append(res)
        max_fit = res
        print(f"{batch:>8}  {res['samples_per_s']:>12.0f}  {res['peak_vram_gb']:>15.2f}")

    results = {
        "measurements": measured,
        "max_fit_batch": max_fit["per_replica_batch_size"] if max_fit else None,
        "max_fit_peak_vram_gb": max_fit["peak_vram_gb"] if max_fit else None,
    }
    if max_fit:
        print(
            f"\nlargest fitting per-replica batch: {max_fit['per_replica_batch_size']} "
            f"({max_fit['peak_vram_gb']:.2f} GB peak, {max_fit['samples_per_s']:.0f} {unit})"
        )
    path = write_result(
        f"bench_gpu_{args.mode}",
        {
            "mode": args.mode,
            "obs_shape": list(_OBS_SHAPE),
            "latent_dim": config.beta_vae.latent_dim,
            "warmup": args.warmup,
            "steps": args.steps,
            "gpu": machine_info()["hostname"],
        },
        results,
        args.output,
    )
    print(f"Result written to {path}")


if __name__ == "__main__":
    main()
