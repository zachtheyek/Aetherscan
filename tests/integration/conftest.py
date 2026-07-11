"""Shared fixtures for the cluster integration smokes.

Both smoke tests launch `python -m aetherscan.main ...` as a subprocess against the real
cluster environment; the repo root, the AETHERSCAN_* path resolution (env var or the cluster
defaults baked into utils/run_container.sh), and the PYTHONPATH-injected subprocess launcher
live here so the two tests can't drift apart.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SUBPROCESS_TIMEOUT_SECONDS = 7200


@pytest.fixture
def cluster_paths():
    """(data_path, model_path, output_path) from AETHERSCAN_* env, falling back to the same
    cluster defaults utils/run_container.sh uses."""
    return (
        os.environ.get("AETHERSCAN_DATA_PATH", "/datax/scratch/zachy/data/aetherscan"),
        os.environ.get("AETHERSCAN_MODEL_PATH", "/datax/scratch/zachy/models/aetherscan"),
        os.environ.get("AETHERSCAN_OUTPUT_PATH", "/datax/scratch/zachy/outputs/aetherscan"),
    )


@pytest.fixture
def run_pipeline():
    """Launcher for `python -m aetherscan.main <args...>` from the repo root, with <repo>/src
    prepended to PYTHONPATH so the subprocess can `import aetherscan` both inside the NGC
    container and in a bare env. Returns the CompletedProcess (check=False — callers assert
    on returncode so failures surface the log tail, not a bare CalledProcessError)."""

    def _run(args: list[str]) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        src = os.path.join(_REPO_ROOT, "src")
        env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
        return subprocess.run(
            [sys.executable, "-m", "aetherscan.main", *args],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )

    return _run
