#!/usr/bin/env python3
"""
Micro-benchmark: the two per-cadence CPU transforms of the load path — frequency
downsampling (skimage downscale_local_mean, as in _downsample_worker / stamp extraction)
and log-normalization (data_generation.log_norm, as in load_inference_data).

    python benchmarks/bench_lognorm_downsample.py [--cadences 64] [--repeats 3]

Prints cadences/s for each transform and writes a JSON result to benchmarks/results/
(or --output).
"""

from __future__ import annotations

import argparse

import numpy as np
from _common import summarize, time_repeats, write_result
from skimage.transform import downscale_local_mean

from aetherscan.data_generation import log_norm

NUM_OBSERVATIONS = 6
TIME_BINS = 16
WIDTH_BIN = 4096
DOWNSAMPLE_FACTOR = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadences", type=int, default=64, help="Cadences per repeat")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default=None, help="Result JSON path")
    args = parser.parse_args()

    rng = np.random.default_rng(11)
    full = rng.chisquare(df=4, size=(args.cadences, NUM_OBSERVATIONS, TIME_BINS, WIDTH_BIN)).astype(
        np.float32
    )
    final_width = WIDTH_BIN // DOWNSAMPLE_FACTOR
    print(
        f"{args.cadences} cadences of shape ({NUM_OBSERVATIONS}, {TIME_BINS}, {WIDTH_BIN}), "
        f"downsample x{DOWNSAMPLE_FACTOR} -> {final_width} bins"
    )

    def run_downsample() -> np.ndarray:
        # Mirrors _downsample_worker: per-observation downscale_local_mean over frequency
        out = np.zeros((args.cadences, NUM_OBSERVATIONS, TIME_BINS, final_width), dtype=np.float32)
        for cadence_idx in range(args.cadences):
            for obs_idx in range(NUM_OBSERVATIONS):
                out[cadence_idx, obs_idx] = downscale_local_mean(
                    full[cadence_idx, obs_idx], (1, DOWNSAMPLE_FACTOR)
                ).astype(np.float32)
        return out

    downsampled = run_downsample()

    results = {}
    durations = time_repeats(run_downsample, args.repeats)
    results["downsample"] = summarize(durations, args.cadences)
    print(
        f"downsample:  {results['downsample']['ops_per_s']:>10.1f} cadences/s "
        f"(best {results['downsample']['best_s']:.3f}s)"
    )

    def run_lognorm() -> None:
        # Mirrors load_inference_data: per-cadence log_norm over the downsampled stamps
        for cadence_idx in range(args.cadences):
            log_norm(downsampled[cadence_idx])

    durations = time_repeats(run_lognorm, args.repeats)
    results["lognorm"] = summarize(durations, args.cadences)
    print(
        f"lognorm:     {results['lognorm']['ops_per_s']:>10.1f} cadences/s "
        f"(best {results['lognorm']['best_s']:.3f}s)"
    )

    path = write_result(
        "bench_lognorm_downsample",
        {
            "cadences": args.cadences,
            "width_bin": WIDTH_BIN,
            "downsample_factor": DOWNSAMPLE_FACTOR,
        },
        results,
        args.output,
    )
    print(f"Result written to {path}")


if __name__ == "__main__":
    main()
