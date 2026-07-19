#!/usr/bin/env python3
"""
Micro-benchmark: setigen narrowband signal injection (data_generation.new_cadence) into a
stacked synthetic cadence — the per-sample kernel of training data generation.

    python benchmarks/bench_injection.py [--injections 200] [--repeats 3]

Prints injections/s and writes a JSON result to benchmarks/results/ (or --output).
"""

from __future__ import annotations

import argparse

import numpy as np
from _common import summarize, time_repeats, write_result

from aetherscan.data_generation import new_cadence

# Defaults mirror DataConfig: 6 obs x 16 time bins stacked, 4096 fine bins,
# GBT-resolution axis scales
NUM_OBSERVATIONS = 6
TIME_BINS = 16
WIDTH_BIN = 4096
FREQ_RESOLUTION = 2.7939677238464355  # Hz
TIME_RESOLUTION = 18.25361108  # seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--injections", type=int, default=200, help="Injections per repeat")
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default=None, help="Result JSON path")
    args = parser.parse_args()

    rng = np.random.default_rng(11)
    stacked = rng.chisquare(df=4, size=(NUM_OBSERVATIONS * TIME_BINS, WIDTH_BIN)).astype(np.float64)
    print(
        f"Injecting {args.injections} signals/repeat into a stacked "
        f"({stacked.shape[0]}, {stacked.shape[1]}) cadence at SNR {args.snr}"
    )

    def run() -> None:
        for _ in range(args.injections):
            # Copy per injection, as batch generation does (each sample gets a fresh plate)
            new_cadence(stacked.copy(), args.snr, WIDTH_BIN, FREQ_RESOLUTION, TIME_RESOLUTION)

    durations = time_repeats(run, args.repeats)
    results = {"injection": summarize(durations, args.injections)}
    print(
        f"injection:   {results['injection']['ops_per_s']:>10.1f} injections/s "
        f"(best {results['injection']['best_s']:.3f}s for {args.injections})"
    )

    path = write_result(
        "bench_injection",
        {
            "injections": args.injections,
            "snr": args.snr,
            "width_bin": WIDTH_BIN,
            "stacked_shape": list(stacked.shape),
        },
        results,
        args.output,
    )
    print(f"Result written to {path}")


if __name__ == "__main__":
    main()
