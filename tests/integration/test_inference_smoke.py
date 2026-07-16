# NOTE: come back to this later

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
from datetime import datetime

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.cluster]

# NOTE: this smoke is coupled to a specific dummy model that must exist on the cluster; the
# skip guard below handles its absence gracefully. Future improvement for a contributor who
# wants to decouple it: make the tag configurable via an env var (e.g.
# AETHERSCAN_SMOKE_MODEL_TAG) or a pytest option/marker so the smoke isn't wired to one artifact.
_MODEL_TAG = "test_v17"  # persisted dummy model on blpc3
_CSV_NAME = "subset_test.csv"  # 2 complete 6-observation cadences


def test_inference_smoke(cluster_paths, run_pipeline):
    if shutil.which("nvidia-smi") is None:
        pytest.skip("requires a GPU host (nvidia-smi not found)")

    data_path, model_path, output_path = cluster_paths

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
    proc = run_pipeline(
        [
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
    )

    tail = "\n".join(proc.stdout.splitlines()[-40:])
    assert proc.returncode == 0, f"inference smoke run failed (tag={tag}); last output:\n{tail}"
    assert "Inference completed successfully!" in proc.stdout
    assert os.path.exists(os.path.join(output_path, f"config_{tag}.json"))
