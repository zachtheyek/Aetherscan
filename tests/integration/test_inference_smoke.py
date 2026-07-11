"""End-to-end CSV-inference smoke test (cluster-only, blpc3).

Runs subset CSV inference against the persisted dummy model (test_v17) as a subprocess of
`python -m aetherscan.main inference` and asserts a clean exit plus a config snapshot. The
subset CSV's raw .h5 files live under /datag (blpc3 only), which run_container.sh does not
bind by default — either bind it explicitly:

    SINGULARITY_BIND=/datag ./utils/run_container.sh python -m pytest tests/ -m "gpu or cluster" -q

or rely on the resume path: preprocessing skips any cadence whose stamp .npy already exists
under <output>/preprocessed, so a previously-preprocessed subset runs without /datag.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from datetime import datetime

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.cluster]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TIMEOUT_SECONDS = 7200
_MODEL_TAG = "test_v17"  # persisted dummy model on blpc3
_CSV_NAME = "subset_test.csv"  # 2 complete 6-observation cadences


def _env_path(var, default):
    return os.environ.get(var, default)


def test_inference_smoke():
    if shutil.which("nvidia-smi") is None:
        pytest.skip("requires a GPU host (nvidia-smi not found)")

    data_path = _env_path("AETHERSCAN_DATA_PATH", "/datax/scratch/zachy/data/aetherscan")
    model_path = _env_path("AETHERSCAN_MODEL_PATH", "/datax/scratch/zachy/models/aetherscan")
    output_path = _env_path("AETHERSCAN_OUTPUT_PATH", "/datax/scratch/zachy/outputs/aetherscan")

    encoder = os.path.join(model_path, f"vae_encoder_{_MODEL_TAG}.keras")
    rf = os.path.join(model_path, f"random_forest_{_MODEL_TAG}.joblib")
    saved_config = os.path.join(model_path, f"config_{_MODEL_TAG}.json")
    csv_path = os.path.join(data_path, "inference", _CSV_NAME)
    for required in (encoder, rf, saved_config, csv_path):
        if not os.path.exists(required):
            pytest.skip(f"required cluster artifact missing: {required}")

    # Raw .h5 reads need /datag; already-preprocessed stamps make it optional.
    preprocessed = glob.glob(os.path.join(output_path, "preprocessed", "subset_test_*.npy"))
    if not os.path.exists("/datag") and not preprocessed:
        pytest.skip("/datag not mounted and no preprocessed subset stamps to resume from")

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd = [
        sys.executable,
        "-m",
        "aetherscan.main",
        "inference",
        "--encoder-path",
        encoder,
        "--rf-path",
        rf,
        "--config-path",
        saved_config,
        "--inference-files",
        _CSV_NAME,
        "--save-tag",
        tag,
        "--max-retries",
        "1",
    ]

    env = dict(os.environ)
    src = os.path.join(_REPO_ROOT, "src")
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src

    proc = subprocess.run(
        cmd,
        check=False,
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
    )
    tail = "\n".join(proc.stdout.splitlines()[-40:])
    assert proc.returncode == 0, f"inference smoke run failed (tag={tag}); last output:\n{tail}"
    assert "Inference completed successfully!" in proc.stdout
    assert os.path.exists(os.path.join(output_path, f"config_{tag}.json"))
