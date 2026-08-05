"""Unit tests for the TF_USE_LEGACY_KERAS default (#323).

The v1.0.0 ``.keras`` weights are Keras-2 (tf_keras) format, so ``from tensorflow import keras``
must resolve to tf_keras. Two layers guarantee that:

* ``aetherscan/__init__.py`` sets the flag on import — the runtime path; and
* ``tests/conftest.py`` sets it before any test module imports TensorFlow — the order-proof test
  path (a TF-first module collected early would otherwise pin the session to Keras 3).

The env-var assertions run in a *subprocess* because ``aetherscan`` is already imported (and its
``setdefault`` already run) in the test process, so a fresh import cannot be observed here — same
subprocess pattern as ``test_train_distribution.py``. ``src`` reaches the child via
``pyproject.toml``'s ``[tool.pytest.ini_options] pythonpath = ["src"]``, which is propagated below.
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
    # Reproduce this process's import path in the child (``src`` is on ``sys.path`` via pyproject's
    # pytest ``pythonpath``, not a ``PYTHONPATH`` prefix), then apply the case-specific override.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env.update(env_overrides)
    result = subprocess.run(  # noqa: PLW1510 — returncode asserted below so the child output shows
        [sys.executable, "-c", _SNIPPET],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result.stdout.strip()


def test_import_sets_legacy_keras_when_unset():
    # With the var unset, importing aetherscan defaults it to "1" so `from tensorflow import
    # keras` resolves to tf_keras (the Keras-2 lineage the v1.0.0 weights were saved under).
    assert _import_aetherscan_with({}) == "1"


def test_import_preserves_explicit_legacy_keras():
    # setdefault, not assignment: an explicit environment value (or the container's own export)
    # must win, so a deliberate opt-out is never silently overridden.
    assert _import_aetherscan_with({"TF_USE_LEGACY_KERAS": "0"}) == "0"


def test_tensorflow_resolves_to_tf_keras_in_this_session():
    # The env var is necessary but not sufficient — assert TF actually honored it and that
    # `tensorflow.keras` is the tf_keras lineage (matches the in-container check,
    # `tf_keras.api._v2.keras`). This is the assertion that would have gone red on master
    # before #323. conftest.py sets the flag before this module imports TF.
    import tensorflow as tf  # noqa: PLC0415 — deferred so the module stays importable TF-free

    assert "tf_keras" in tf.keras.__name__
