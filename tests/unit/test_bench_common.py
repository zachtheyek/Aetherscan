"""Unit test for benchmarks/_common.py's stdlib-only bench-data-dir helper.

default_bench_data_dir(sub) is the shared default for both bench_datagen.py's --data-dir and
bench_input_pipeline.py's --data-dir: it must resolve under {AETHERSCAN_DATA_PATH}/bench/{sub},
reading the same env var config.data_path honors and falling back to config.py's literal default
when the var is unset (config.py:688). That fallback parity is the load-bearing invariant — it is
what keeps the bench scripts pointed at the same data root the pipeline uses without importing the
config singleton. _common.py is stdlib-only, so it loads straight from its file (no TensorFlow or
dependency-stack stubbing needed), the same file-path import the other benchmark tests use.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_COMMON_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "_common.py"
_spec = importlib.util.spec_from_file_location("_common", _COMMON_PATH)
_common = importlib.util.module_from_spec(_spec)
sys.modules["_common"] = _common
_spec.loader.exec_module(_common)


def test_default_bench_data_dir_honors_env(monkeypatch):
    monkeypatch.setenv("AETHERSCAN_DATA_PATH", "/scratch/somebody/data/aetherscan")
    assert (
        _common.default_bench_data_dir("datagen")
        == "/scratch/somebody/data/aetherscan/bench/datagen"
    )


def test_default_bench_data_dir_falls_back_to_config_default(monkeypatch):
    monkeypatch.delenv("AETHERSCAN_DATA_PATH", raising=False)
    result = _common.default_bench_data_dir("input")
    # Parity with config.py's literal default (config.py:688), so both bench scripts land under
    # the same data root the pipeline uses even when the env var is unset.
    assert result.endswith("/bench/input")
    assert result.startswith("/datax/scratch/zachy/data/aetherscan")
