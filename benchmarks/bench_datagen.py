#!/usr/bin/env python3
"""
Round-generation benchmark: the real pooled generate_round_to_memmap path (the producer wall).

bench_injection.py times the single-process new_cadence kernel; this script measures the
whole round-generation machinery at reduced scale exactly as the producer runs it — a
worker pool initialized via _init_worker against shared-memory background plates, per-task
seed derivation from the root seed, batched memmap-writing tasks, per-chunk barriers — and
then sha256s every output array. Two invocations with identical arguments on different
code (e.g. master vs a candidate branch) must produce identical checksums: that is the
byte-compatibility gate for any generation-path optimization, alongside the wall-time delta.

    # arm A (checkout master), then arm B (checkout branch), same args:
    ./utils/run_container.sh python benchmarks/bench_datagen.py \
        --n-samples 8192 --workers 32 --seed 11 \
        --data-dir /datax/scratch/$USER/data/aetherscan/bench/datagen --output /tmp/arm_a.json

Compare: python -c "import json,sys; a,b=(json.load(open(p)) for p in sys.argv[1:3]); \
    print('IDENTICAL' if a['checksums']==b['checksums'] else 'MISMATCH')" /tmp/arm_a.json /tmp/arm_b.json

Stats callbacks are exercised with a counting stub (no DB) — the DB drain is off the
generation critical path (#277) and singleton bootstrap is out of scope for a benchmark.
Writes a JSON result to benchmarks/results/ (or --output), like the other benchmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from _common import write_result

from aetherscan.data_generation import _init_worker, generate_round_to_memmap
from aetherscan.round_data import RoundDataPaths

# Production cadence geometry (DataConfig defaults after 8x downsampling)
NUM_OBSERVATIONS = 6
TIME_BINS = 16
WIDTH_BIN = 512
FREQ_RESOLUTION = 2.7939677238464355  # Hz
TIME_RESOLUTION = 18.25361108  # seconds


def _default_data_dir(sub: str) -> str:
    """Default bench-data location: {AETHERSCAN_DATA_PATH}/bench/{sub}. Reads the same env var
    config.data_path honors (with config.py's literal default as the fallback, so the two agree)
    without importing config — keeps this script framework-light. Benchmark data lands
    per-user/host under the pipeline's data root instead of scattering across /tmp or CWD."""
    return os.path.join(
        os.environ.get("AETHERSCAN_DATA_PATH", "/datax/scratch/zachy/data/aetherscan"),
        "bench",
        sub,
    )


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=8192, help="Cadences per array (%%4==0)")
    parser.add_argument("--n-plates", type=int, default=4096, help="Synthetic background plates")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Samples per chunk (%%4==0)")
    parser.add_argument("--task-size", type=int, default=64, help="Cadences per pool task")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Pool size (1 = in-process)"
    )
    parser.add_argument(
        "--seed", type=int, default=11, help="Root seed (plates + per-task streams)"
    )
    parser.add_argument("--snr-base", type=float, default=10.0)
    parser.add_argument("--snr-range", type=float, default=40.0)
    parser.add_argument(
        "--data-dir",
        default=_default_data_dir("datagen"),
        help="Scratch dir for the generated round (default: {AETHERSCAN_DATA_PATH}/bench/"
        "datagen). Deleted on exit unless --keep is passed — pass --keep to persist the round "
        "under the bench dir.",
    )
    parser.add_argument("--keep", action="store_true", help="Keep the round dir (default: delete)")
    parser.add_argument(
        "--preload-tf",
        action="store_true",
        help="Import TensorFlow (CUDA-blanked) before forking the pool, so workers inherit "
        "the same import graph the production producer's workers carry — without it, "
        "gc/interpreter costs read optimistically low vs production",
    )
    parser.add_argument("--output", default=None, help="Result JSON path")
    args = parser.parse_args()

    if args.preload_tf:
        # Mirror the producer: TF importable but never CUDA-initialized in generation workers
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        import tensorflow  # noqa: F401, PLC0415

        print("Preloaded TensorFlow into the parent (workers fork with the full import graph)")

    # Synthetic chi2 plates — seeded, so identical across arms with identical args
    rng = np.random.default_rng(args.seed)
    plates = rng.chisquare(
        df=4, size=(args.n_plates, NUM_OBSERVATIONS, TIME_BINS, WIDTH_BIN)
    ).astype(np.float32)
    print(f"Plates: {plates.shape} float32 ({plates.nbytes / 2**20:.0f} MiB), seed {args.seed}")

    paths = RoundDataPaths.for_round(args.data_dir, 1)
    stats_counts = {"segments": 0, "samples": 0}

    def stats_cb(segment: dict) -> None:
        stats_counts["segments"] += 1
        stats_counts["samples"] += len(segment["stats_list"])

    shm = None
    pool = None
    try:
        if args.workers > 1:
            shm = SharedMemory(create=True, size=plates.nbytes, name=f"bench_datagen_{os.getpid()}")
            shared = np.ndarray(plates.shape, dtype=plates.dtype, buffer=shm.buf)
            shared[:] = plates[:]
            pool = Pool(
                processes=args.workers,
                initializer=_init_worker,
                initargs=(shm.name, plates.shape, plates.dtype),
            )
        print(f"Generating {args.n_samples} samples x3 arrays with {args.workers} worker(s)")
        wall_start = time.time()
        manifest = generate_round_to_memmap(
            paths,
            args.n_samples,
            args.snr_base,
            args.snr_range,
            width_bin=WIDTH_BIN,
            num_observations=NUM_OBSERVATIONS,
            time_bins=TIME_BINS,
            chunk_size=args.chunk_size,
            task_size=args.task_size,
            freq_resolution=FREQ_RESOLUTION,
            time_resolution=TIME_RESOLUTION,
            pool=pool,
            backgrounds=None if pool is not None else plates,
            round_num=1,
            seed=args.seed,
            stats_cb=stats_cb,
        )
        wall = time.time() - wall_start
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if shm is not None:
            shm.close()
            shm.unlink()  # creator-only unlink, per the shared-memory rules

    checksums = {}
    for name, path in paths.array_paths.items():
        checksums[name] = _sha256(path)
    for name, path in paths.lognorm_paths.items():
        checksums[f"{name}_lognorm"] = _sha256(path)
    checksums["labels"] = _sha256(paths.labels_path)

    throughput = args.n_samples / wall
    print(f"Wall: {wall:.1f} s ({throughput:.1f} samples/s x3 arrays)")
    print(
        f"Stats callbacks: {stats_counts['segments']} segments, {stats_counts['samples']} samples"
    )
    for key in sorted(checksums):
        print(f"  sha256 {key}: {checksums[key][:16]}")

    params = {
        k: getattr(args, k)
        for k in (
            "n_samples",
            "n_plates",
            "chunk_size",
            "task_size",
            "workers",
            "seed",
            "snr_base",
            "snr_range",
        )
    }
    results = {
        "wall_s": wall,
        "samples_per_s": throughput,
        "manifest_wall_s": manifest["wall_time_s"],
        "stats_segments": stats_counts["segments"],
        "stats_samples": stats_counts["samples"],
        "checksums": checksums,
    }
    write_result("bench_datagen", params, results, args.output)

    if not args.keep:
        shutil.rmtree(paths.round_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
