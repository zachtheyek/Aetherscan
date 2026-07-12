#!/usr/bin/env python3
"""
Micro-benchmark: vectorized sliding-window normality test (_sliding_normality_k2) vs the
historical per-window scipy.stats.normaltest Python loop — the #1 cost of inference
preprocessing before PR-06 vectorized it.

    python benchmarks/bench_normality.py [--width 1048576] [--repeats 3] [--skip-baseline]

Prints windows/s for both implementations plus the speedup, and writes a JSON result to
benchmarks/results/ (or --output).
"""

from __future__ import annotations

import argparse

import numpy as np
from _common import summarize, time_repeats, write_result
from scipy import stats

from aetherscan.preprocessing import _sliding_normality_k2


def scipy_loop(channel: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """The historical implementation: one scipy.stats.normaltest call per window."""
    width = channel.shape[1]
    out = []
    for start in range(0, width - window_size, step_size):
        window = channel[:, start : start + window_size].flatten()
        out.append(stats.normaltest(window).statistic)
    return np.asarray(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1_048_576, help="Channel width (fine bins)")
    parser.add_argument("--time-bins", type=int, default=16)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--step", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip the (slow) scipy per-window loop"
    )
    parser.add_argument("--output", default=None, help="Result JSON path")
    args = parser.parse_args()

    rng = np.random.default_rng(11)
    channel = rng.chisquare(df=4, size=(args.time_bins, args.width)).astype(np.float64)
    n_windows = len(range(0, args.width - args.window, args.step))
    print(
        f"Channel ({args.time_bins}, {args.width}), window={args.window}, step={args.step} "
        f"-> {n_windows} windows"
    )

    results = {}

    durations = time_repeats(
        lambda: _sliding_normality_k2(channel, args.window, args.step), args.repeats
    )
    results["vectorized"] = summarize(durations, n_windows)
    print(
        f"vectorized:  {results['vectorized']['ops_per_s']:>12.0f} windows/s "
        f"(best {results['vectorized']['best_s']:.3f}s)"
    )

    if not args.skip_baseline:
        durations = time_repeats(lambda: scipy_loop(channel, args.window, args.step), args.repeats)
        results["scipy_loop"] = summarize(durations, n_windows)
        print(
            f"scipy loop:  {results['scipy_loop']['ops_per_s']:>12.0f} windows/s "
            f"(best {results['scipy_loop']['best_s']:.3f}s)"
        )
        speedup = results["vectorized"]["ops_per_s"] / results["scipy_loop"]["ops_per_s"]
        results["speedup"] = speedup
        print(f"speedup:     {speedup:.1f}x")

    path = write_result(
        "bench_normality",
        {
            "width": args.width,
            "time_bins": args.time_bins,
            "window": args.window,
            "step": args.step,
            "n_windows": n_windows,
        },
        results,
        args.output,
    )
    print(f"Result written to {path}")


if __name__ == "__main__":
    main()
