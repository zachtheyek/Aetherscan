"""Unit test for benchmarks/bench_datagen.py's stdlib-only bench-data-dir helper (Task B).

The default --data-dir must resolve under {AETHERSCAN_DATA_PATH}/bench/datagen, reading the same
env var config.data_path honors. bench_datagen.py's heavy generation deps (setigen via
aetherscan.data_generation, plus round_data / _common) are stubbed here so the pure helper stays
importable without TensorFlow or the full dependency stack.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_BENCH_DATAGEN = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_datagen.py"


def _load_bench_datagen() -> types.ModuleType:
    """Import bench_datagen with its non-stdlib imports stubbed (stdlib + numpy only)."""
    data_gen = types.ModuleType("aetherscan.data_generation")
    data_gen._init_worker = lambda *a, **k: None
    data_gen.generate_round_to_memmap = lambda *a, **k: None
    round_data = types.ModuleType("aetherscan.round_data")
    round_data.RoundDataPaths = object
    common = types.ModuleType("_common")
    common.write_result = lambda *a, **k: None
    stubs = {
        "aetherscan.data_generation": data_gen,
        "aetherscan.round_data": round_data,
        "_common": common,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("bench_datagen", _BENCH_DATAGEN)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


def test_default_data_dir_honors_env(monkeypatch):
    module = _load_bench_datagen()
    monkeypatch.setenv("AETHERSCAN_DATA_PATH", "/scratch/somebody/data/aetherscan")
    result = module._default_data_dir("datagen")
    assert result == "/scratch/somebody/data/aetherscan/bench/datagen"
    assert result.endswith("bench/datagen")
