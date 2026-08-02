"""Unit tests for the TF_USE_LEGACY_KERAS default set in ``aetherscan/__init__.py`` (#323).

The v1.0.0 ``.keras`` weights are Keras-2 (tf_keras) format, so ``from tensorflow import keras``
must resolve to tf_keras — importing the package sets ``TF_USE_LEGACY_KERAS=1`` early, before any
submodule imports TF. These assertions run in a *subprocess* because ``aetherscan`` is already
imported (and the ``setdefault`` already run) in the test process, so the flag's value here would
not reflect a fresh import. Same subprocess pattern as ``test_train_distribution.py``. The child
imports only ``aetherscan`` + ``os`` (no TensorFlow), so these stay fast and CPU-only.
"""

from __future__ import annotations

import os
import subprocess
import sys

# Print the flag the child sees right after importing the package.
_SNIPPET = "import aetherscan, os; print(os.environ.get('TF_USE_LEGACY_KERAS'))"


def _import_aetherscan_with(env_overrides: dict[str, str]) -> str:
    """Run ``import aetherscan`` in a clean child and return the flag it observed."""
    env = {k: v for k, v in os.environ.items() if k != "TF_USE_LEGACY_KERAS"}
    # Make aetherscan importable in the child exactly as it is here — CI runs the suite via
    # PYTHONPATH=src (no installed dist), so propagate this process's import path.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip()


def test_import_sets_legacy_keras_when_unset():
    # With the var unset, importing aetherscan defaults it to "1" so `from tensorflow import
    # keras` resolves to tf_keras (the Keras-2 lineage the v1.0.0 weights were saved under).
    assert _import_aetherscan_with({}) == "1"


def test_import_preserves_explicit_legacy_keras():
    # setdefault, not assignment: an explicit environment value (or the container's own export)
    # must win, so a deliberate opt-out is never silently overridden.
    assert _import_aetherscan_with({"TF_USE_LEGACY_KERAS": "0"}) == "0"
