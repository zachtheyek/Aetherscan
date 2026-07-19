"""
Shared pytest fixtures for the Aetherscan test suite.

Provides three things:
1. An autouse fixture that isolates every (non-integration) test: AETHERSCAN_{DATA,MODEL,OUTPUT}_PATH
   point at tmp_path, all singletons are reset via their _reset() hooks, and a fresh Config is
   initialized. Teardown stops any started background threads/pools and resets everything again.
2. Synthetic data factories: tiny .npy background plates, tiny .h5 observation files, and tiny
   inference CSVs — sized for speed, shaped like the real thing.
3. A headless matplotlib backend for CI (set before any aetherscan module imports pyplot).
"""

# TODO: integration tests that verify clean shutdown under adverse conditions
#       (SIGTERM/SIGINT mid-run, resource cleanup) — carried over from the old tests/placeholder
# TODO: build tests that tagged releases deploy properly to various environments
#       (PyPI, HuggingFace, etc.) once release engineering lands — carried over from placeholder

from __future__ import annotations

import contextlib
import csv
import os
import signal
import sys

import numpy as np
import pytest

# Force a non-interactive backend before any test imports matplotlib.pyplot
# (train.py imports it at module level; CI runners have no display).
os.environ.setdefault("MPLBACKEND", "Agg")
# Quiet TF's C++ INFO/WARNING chatter in test output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _reset_all_singletons():
    """Reset every singleton class via its _reset() teardown hook.

    Imports are deferred so integration runs (which skip the isolation fixture entirely)
    never pull TensorFlow into the pytest parent process.
    """
    from aetherscan.config import Config  # noqa: PLC0415
    from aetherscan.db.db import Database  # noqa: PLC0415
    from aetherscan.logger.logger import Logger  # noqa: PLC0415
    from aetherscan.manager.manager import ResourceManager  # noqa: PLC0415
    from aetherscan.monitor.monitor import ResourceMonitor  # noqa: PLC0415

    for cls in (Config, Database, Logger, ResourceManager, ResourceMonitor):
        cls._reset()


def _teardown_singletons():
    """Stop background resources still held by singletons, then reset them all.

    Mirrors ResourceManager.cleanup_all()'s ordering (pools -> shared memory -> monitor ->
    db -> logger) without its logging side effects, so a test that leaked a resource can't
    poison the next test.
    """
    from aetherscan.db.db import Database  # noqa: PLC0415
    from aetherscan.logger.logger import Logger  # noqa: PLC0415
    from aetherscan.manager.manager import ResourceManager  # noqa: PLC0415
    from aetherscan.monitor.monitor import ResourceMonitor  # noqa: PLC0415

    manager = ResourceManager._instance
    if manager is not None:
        for managed_process in list(manager._processes):
            with contextlib.suppress(Exception):
                managed_process.close(timeout=5.0)
        for managed_pool in list(manager._pools):
            with contextlib.suppress(Exception):
                managed_pool.close(timeout=5.0)
        for managed_shm in list(manager._shared_memories):
            with contextlib.suppress(Exception):
                managed_shm.close()
        # Each ResourceManager construction registers an atexit callback; unregister it so
        # per-test instances don't accumulate stale cleanup hooks over the session.
        import atexit  # noqa: PLC0415

        atexit.unregister(manager.cleanup_all)

    monitor = ResourceMonitor._instance
    if monitor is not None:
        with contextlib.suppress(Exception):
            monitor.stop()

    db = Database._instance
    if db is not None:
        with contextlib.suppress(Exception):
            db.stop()

    logger_instance = Logger._instance
    if logger_instance is not None:
        with contextlib.suppress(Exception):
            logger_instance.stop()

    _reset_all_singletons()


@pytest.fixture(autouse=True)
def aetherscan_isolated_env(request, tmp_path, monkeypatch):
    """Autouse: fresh singletons + tmp-path-scoped env for every test.

    Integration tests are exempt (marker `integration`): they exercise the real pipeline as a
    subprocess and must inherit the real AETHERSCAN_* environment, and their pytest parent
    process should never import TensorFlow.
    """
    if request.node.get_closest_marker("integration"):
        yield
        return

    data_path = tmp_path / "data"
    model_path = tmp_path / "models"
    output_path = tmp_path / "outputs"
    for sub in ("training", "testing", "inference"):
        (data_path / sub).mkdir(parents=True, exist_ok=True)
    model_path.mkdir(exist_ok=True)
    output_path.mkdir(exist_ok=True)

    monkeypatch.setenv("AETHERSCAN_DATA_PATH", str(data_path))
    monkeypatch.setenv("AETHERSCAN_MODEL_PATH", str(model_path))
    monkeypatch.setenv("AETHERSCAN_OUTPUT_PATH", str(output_path))
    # Keep unit tests from ever talking to Slack, whatever the host env holds.
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_CHANNEL", raising=False)

    # ResourceManager's constructor installs SIGINT/SIGTERM handlers; snapshot and restore
    # them so per-test instances don't leave stale bound-method handlers behind.
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_stdout, original_stderr = sys.stdout, sys.stderr

    from aetherscan.config import init_config  # noqa: PLC0415

    _reset_all_singletons()
    init_config()

    yield

    _teardown_singletons()
    signal.signal(signal.SIGINT, original_sigint)
    signal.signal(signal.SIGTERM, original_sigterm)
    sys.stdout, sys.stderr = original_stdout, original_stderr


# NOTE: come back to this later
@pytest.fixture
def make_background_npy(tmp_path):
    """Factory for tiny .npy background plates shaped like real training backgrounds.

    Returns a callable(filename, n_cadences=8, width_bin=512) -> Path that writes a positive
    float32 array of shape (n_cadences, 6, 16, width_bin) under the tmp data/training dir.
    """

    def _make(filename="backgrounds.npy", n_cadences=8, width_bin=512):
        rng = np.random.default_rng(11)
        # Chi-squared-ish positive noise, loosely resembling detected power spectra.
        plate = rng.chisquare(df=4, size=(n_cadences, 6, 16, width_bin)).astype(np.float32)
        path = tmp_path / "data" / "training" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, plate)
        return path

    return _make


# NOTE: come back to this later
@pytest.fixture
def make_h5_observation(tmp_path):
    """Factory for tiny .h5 observation files matching the filterbank-style layout.

    Returns a callable(filename, n_chans=2048, time_bins=16) -> Path that writes an .h5 with a
    'data' dataset of shape (time_bins, 1, n_chans) plus fch1/foff/nchans attrs (plain h5py,
    no bitshuffle).
    """
    import h5py  # noqa: PLC0415

    def _make(filename="obs.h5", n_chans=2048, time_bins=16):
        rng = np.random.default_rng(7)
        data = rng.chisquare(df=4, size=(time_bins, 1, n_chans)).astype(np.float32)
        path = tmp_path / filename
        with h5py.File(path, "w") as hf:
            dset = hf.create_dataset("data", data=data)
            dset.attrs["fch1"] = 8421.386717353016
            dset.attrs["foff"] = -2.7939677238464355e-06
            dset.attrs["nchans"] = n_chans
        return path

    return _make


# NOTE: come back to this later
@pytest.fixture
def make_inference_csv(tmp_path):
    """Factory for tiny inference CSVs in the cadence-grouping layout.

    Returns a callable(filename, groups) -> Path. `groups` is a list of (key_dict, h5_paths)
    pairs; each h5 path becomes one row carrying the group's key columns. The h5-path column
    name is read from InferenceConfig.cadence_h5_path_col on the initialized singleton, and
    the default group keys mirror InferenceConfig.cadence_group_by_cols.
    """
    from aetherscan.config import get_config  # noqa: PLC0415

    def _make(filename="subset.csv", groups=None):
        if groups is None:
            groups = [
                (
                    {
                        "Target": "HIP110750",
                        "Session": "AGBT21B_999_31",
                        "Band": "L",
                        "Cadence ID": "0",
                        "Frequency": "1400",
                    },
                    [f"/data/obs_{i}.h5" for i in range(6)],
                )
            ]
        h5_col = get_config().inference.cadence_h5_path_col
        fieldnames = list(groups[0][0].keys()) + [h5_col]
        path = tmp_path / "data" / "inference" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for key_dict, h5_paths in groups:
                for h5 in h5_paths:
                    writer.writerow({**key_dict, h5_col: h5})
        return path

    return _make
