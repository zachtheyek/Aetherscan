#!/usr/bin/env python3
"""
GPU compute occupancy from a TensorFlow profiler XPlane trace (#276 audit tooling).

Reads the xplane.pb a `--profile DIR` run of bench_input_pipeline.py writes (or any
tf.profiler capture), merges each GPU device plane's event intervals so overlapping streams
don't double-count, and reports per-GPU busy time and occupancy over the observed window
(global min event start .. global max event end across all GPU planes). This is the stricter
device-side measurement the #276 audit used: nvidia-smi's sampled "utilization" marks a whole
sample busy if any kernel ran in it, so it reads high against true kernel occupancy.

Usage (container — needs the tensorflow proto stubs, no GPU):
    ./utils/run_container.sh python benchmarks/parse_xplane_occupancy.py <profile_dir>

Large traces parse slowly (a ~360 MB 5-GPU capture takes a few minutes of pure-Python event
iteration); run it once per capture and keep the summary next to the trace.
"""

from __future__ import annotations

import glob
import sys

from tensorflow.core.profiler.protobuf import xplane_pb2


def _merged_busy_ps(intervals: list[tuple[int, int]]) -> int:
    """Total covered picoseconds of possibly-overlapping (start, end) intervals."""
    intervals.sort()
    busy = 0
    cur_start: int | None = None
    cur_end: int | None = None
    for start, end in intervals:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                busy += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        busy += cur_end - cur_start
    return busy


def main(profile_dir: str) -> None:
    paths = glob.glob(f"{profile_dir}/plugins/profile/*/*.xplane.pb")
    if not paths:
        raise SystemExit(f"no xplane.pb under {profile_dir}")
    space = xplane_pb2.XSpace()
    with open(paths[0], "rb") as f:
        space.ParseFromString(f.read())

    gpu_planes = [p for p in space.planes if p.name.startswith("/device:GPU:")]
    if not gpu_planes:
        raise SystemExit("no GPU planes in trace")

    window_start: int | None = None
    window_end: int | None = None
    per_gpu: dict[str, int] = {}
    for plane in gpu_planes:
        intervals: list[tuple[int, int]] = []
        for line in plane.lines:
            for event in line.events:
                start = line.timestamp_ns * 1000 + event.offset_ps
                end = start + event.duration_ps
                intervals.append((start, end))
                window_start = start if window_start is None else min(window_start, start)
                window_end = end if window_end is None else max(window_end, end)
        per_gpu[plane.name] = _merged_busy_ps(intervals)

    window = window_end - window_start
    print(f"window: {window / 1e12:.2f} s (across all GPU events)")
    total = 0.0
    for name, busy in sorted(per_gpu.items()):
        frac = busy / window
        total += frac
        print(f"{name}: busy {busy / 1e12:.2f} s  occupancy {frac * 100:.1f}%")
    print(f"mean GPU occupancy: {total / len(per_gpu) * 100:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1])
