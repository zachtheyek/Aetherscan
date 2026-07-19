#!/usr/bin/env python3
"""
Micro-benchmark: the two bandpass flatteners applied per coarse channel during energy
detection — the static PFB response divide (pfb.equalize_passband, the default) vs the
historical per-channel spline fit (preprocessing._spline_flatten_bandpass).

    python benchmarks/bench_pfb_vs_spline.py [--width 1048576] [--repeats 3]

Prints channels/s for both methods plus the speedup, and writes a JSON result to
benchmarks/results/ (or --output). The one-time PFB response FFT is reported separately
(it is paid once per run, not per channel).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from _common import summarize, time_repeats, write_result

from aetherscan.pfb import equalize_passband, gen_coarse_channel_response
from aetherscan.preprocessing import _spline_flatten_bandpass

TIME_BINS = 16
NUM_COARSE = 64  # response fold width param (matches a typical GBT file's coarse count)
TAPS_PER_CHANNEL = 12  # GBT / Breakthrough Listen backend default
SPLINE_ORDER = 16  # InferenceConfig.spline_order default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--width", type=int, default=1_048_576, help="Coarse channel width (fine bins)"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default=None, help="Result JSON path")
    args = parser.parse_args()

    # One-time response generation (lru_cached in production, one FFT per run)
    response_start = time.perf_counter()
    response = gen_coarse_channel_response(args.width, NUM_COARSE, TAPS_PER_CHANNEL)
    response_s = time.perf_counter() - response_start

    # Synthetic coarse channel: chi-2 noise shaped by the PFB response, so both methods
    # flatten a realistic scalloped passband
    rng = np.random.default_rng(11)
    channel = (rng.chisquare(df=4, size=(TIME_BINS, args.width)) * response[np.newaxis, :]).astype(
        np.float64
    )
    print(
        f"Coarse channel ({TIME_BINS}, {args.width}), taps={TAPS_PER_CHANNEL}, "
        f"spline order={SPLINE_ORDER} (response FFT: {response_s:.2f}s one-time)"
    )

    results = {"response_generation_s": response_s}

    durations = time_repeats(lambda: equalize_passband(channel, response), args.repeats)
    results["pfb"] = summarize(durations, 1)
    print(
        f"pfb:         {results['pfb']['ops_per_s']:>8.2f} channels/s "
        f"(best {results['pfb']['best_s']:.3f}s/channel)"
    )

    durations = time_repeats(lambda: _spline_flatten_bandpass(channel, SPLINE_ORDER), args.repeats)
    results["spline"] = summarize(durations, 1)
    print(
        f"spline:      {results['spline']['ops_per_s']:>8.2f} channels/s "
        f"(best {results['spline']['best_s']:.3f}s/channel)"
    )

    speedup = results["pfb"]["ops_per_s"] / results["spline"]["ops_per_s"]
    results["speedup"] = speedup
    print(f"speedup:     {speedup:.1f}x")

    path = write_result(
        "bench_pfb_vs_spline",
        {
            "width": args.width,
            "time_bins": TIME_BINS,
            "num_coarse": NUM_COARSE,
            "taps_per_channel": TAPS_PER_CHANNEL,
            "spline_order": SPLINE_ORDER,
        },
        results,
        args.output,
    )
    print(f"Result written to {path}")


if __name__ == "__main__":
    main()
