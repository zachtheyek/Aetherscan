# NOTE: come back to this later

"""End-to-end training smoke test (cluster-only).

Runs the known-good blpc3 5-GPU smoke config from the repo runbook as a subprocess of
`python -m aetherscan.main train` and asserts the run exits cleanly with all final model
artifacts on disk. Batch/sample sizes here divide cleanly for exactly 5 replicas — this test
is sized for blpc3 (5x RTX PRO 6000). Run inside the NGC container:

    ./utils/run_container.sh python -m pytest tests/ -m "gpu or cluster" -q
"""

from __future__ import annotations

import os
import re
import shutil

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.cluster]

# The known-good blpc3 smoke config: repo defaults are only divisible for 4 or 6 replicas, so
# the batch/sample geometry is overridden to divide for 5. Mirrored (as the subject under
# test) by _SMOKE_FLAGS_5_REPLICAS in tests/unit/test_cli_validation.py — keep in sync.
_SMOKE_FLAGS = [
    "--max-retries",
    "1",
    "--num-training-rounds",
    "2",
    "--epochs-per-round",
    "2",
    "--per-replica-batch-size",
    "4",
    "--per-replica-val-batch-size",
    "4",
    "--effective-batch-size",
    "20",
    "--num-samples-beta-vae",
    "200",
    "--num-samples-rf",
    "200",
    "--latent-viz-num-cadences-per-type",
    "5",
]


def _next_test_tag(model_path: str) -> str:
    """Scan existing artifacts for test_vNN tags and return the next unused one — reusing a
    tag causes stale-artifact confusion.

    NOTE: the scan-then-pick is not atomic — two runs launched concurrently could both claim
    the same tag. Fine for its intended use (manual, sequential smoke runs on a shared
    cluster); revisit if these smokes are ever launched in parallel.
    """
    versions = [0]
    for root in (model_path, os.path.join(model_path, "checkpoints")):
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            match = re.search(r"test_v(\d+)", entry)
            if match:
                versions.append(int(match.group(1)))
    return f"test_v{max(versions) + 1}"


def test_train_smoke(cluster_paths, run_pipeline):
    if shutil.which("nvidia-smi") is None:
        pytest.skip("requires a GPU host (nvidia-smi not found)")

    data_path, model_path, output_path = cluster_paths
    if not os.path.isdir(os.path.join(data_path, "training")):
        pytest.skip(f"training data not found under {data_path}")

    tag = _next_test_tag(model_path)
    proc = run_pipeline(["train", "--save-tag", tag, *_SMOKE_FLAGS])

    tail = "\n".join(proc.stdout.splitlines()[-40:])
    assert proc.returncode == 0, f"train smoke run failed (tag={tag}); last output:\n{tail}"
    assert "Training completed successfully!" in proc.stdout

    # Final artifacts: encoder/decoder/RF under model_path, config snapshot under output_path.
    for artifact in (
        f"vae_encoder_{tag}.keras",
        f"vae_decoder_{tag}.keras",
        f"random_forest_{tag}.joblib",
    ):
        assert os.path.exists(os.path.join(model_path, artifact)), f"missing {artifact}"
    assert os.path.exists(os.path.join(output_path, f"config_{tag}.json"))

    # End-of-run benchmark report: pins the #203 _post_benchmark_report hook's real
    # db.flush -> render path end-to-end (output_path/plots/benchmark_report_{tag}.png).
    assert os.path.exists(os.path.join(output_path, "plots", f"benchmark_report_{tag}.png"))

    # Per-round checkpoints for both rounds.
    checkpoints = os.path.join(model_path, "checkpoints")
    for round_tag in ("round_01", "round_02"):
        assert os.path.exists(os.path.join(checkpoints, f"vae_encoder_{round_tag}.keras"))
