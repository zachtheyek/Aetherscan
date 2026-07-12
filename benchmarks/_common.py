"""
Shared plumbing for the micro-benchmark scripts: src/ importability, machine metadata,
wall-clock timing of repeated callables, and JSON result output. Deliberately framework-free
(see benchmarks/README.md) — each bench_*.py is a plain script that prints ops/s and writes
one JSON file.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

# Make src/ importable when running directly: benchmarks/bench_*.py
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_BENCH_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

RESULTS_DIR = os.path.join(_BENCH_DIR, "results")


def machine_info() -> dict:
    """Hostname / platform / CPU info stamped into every result JSON."""
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def time_repeats(func, repeats: int) -> list[float]:
    """Call func() `repeats` times, returning each call's wall-clock seconds."""
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - start)
    return durations


def summarize(durations: list[float], ops_per_call: float) -> dict:
    """Best/mean seconds plus ops/s (best-run basis, the conventional benchmark number)."""
    best = min(durations)
    return {
        "repeats": len(durations),
        "best_s": best,
        "mean_s": sum(durations) / len(durations),
        "ops_per_call": ops_per_call,
        "ops_per_s": ops_per_call / best if best > 0 else float("inf"),
    }


def write_result(name: str, params: dict, results: dict, output: str | None = None) -> str:
    """Write the benchmark result JSON (default benchmarks/results/{name}_{host}.json)."""
    payload = {
        "benchmark": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine": machine_info(),
        "params": params,
        "results": results,
    }
    if output is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output = os.path.join(RESULTS_DIR, f"{name}_{platform.node()}.json")
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    return output
