# NOTE: come back to this later

"""End-to-end CSV-inference smoke test (cluster-only, blpc3).

Runs subset CSV inference against the persisted dummy model (test_v17) as a subprocess of
`python -m aetherscan.main inference` and asserts a clean exit plus a config snapshot. The
subset CSV's raw .h5 files live under /datag (blpc3 only), which run_container.sh does not
bind by default — either bind it explicitly:

    SINGULARITY_BIND=/datag ./utils/run_container.sh python -m pytest tests/ -m "gpu or cluster" -q

or rely on the resume path: preprocessing skips any cadence whose stamp .npy already exists.
Since #298 the default stamp directory is ED-fingerprint-scoped ({data_path}/inference/
preprocessed/<csv_stem>_ed<hash12>/), shared across runs with the same ED config — so a
fresh-tagged run resumes automatically off any prior same-config run's stamps. When /datag
is not mounted and only legacy stamps exist under <output>/preprocessed, the test still
points --preprocess-output-dir at them explicitly (the pinned-directory escape hatch).
"""

from __future__ import annotations

import glob
import os
import re
import shutil

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.cluster]

# NOTE: this smoke is coupled to a specific dummy model that must exist on the cluster; the
# skip guard below handles its absence gracefully. The tag is overridable via the shared
# smoke_model_tag fixture (AETHERSCAN_SMOKE_MODEL_TAG, default test_v17) so this smoke can point
# at whatever tag test_train_smoke just produced instead of being wired to one fixed artifact.
_CSV_NAME = "subset_test.csv"  # 2 complete 6-observation cadences


def test_inference_smoke(cluster_paths, run_pipeline, smoke_model_tag):
    if shutil.which("nvidia-smi") is None:
        pytest.skip("requires a GPU host (nvidia-smi not found)")

    data_path, model_path, output_path = cluster_paths

    encoder = os.path.join(model_path, f"vae_encoder_{smoke_model_tag}.keras")
    rf = os.path.join(model_path, f"random_forest_{smoke_model_tag}.joblib")
    # Training saves config_{tag}.json under OUTPUT_path (tag_guards); only hand-placed
    # legacy artifacts (test_v17) kept a copy in model_path — accept either location so
    # the smoke runs against freshly trained models without manual copying (#298).
    saved_config = next(
        (
            path
            for path in (
                os.path.join(model_path, f"config_{smoke_model_tag}.json"),
                os.path.join(output_path, f"config_{smoke_model_tag}.json"),
            )
            if os.path.exists(path)
        ),
        os.path.join(output_path, f"config_{smoke_model_tag}.json"),
    )
    csv_path = os.path.join(data_path, "inference", _CSV_NAME)
    for required in (encoder, rf, saved_config, csv_path):
        if not os.path.exists(required):
            pytest.skip(f"required cluster artifact missing: {required}")

    # Raw .h5 reads need /datag; already-preprocessed stamps make it optional (and since
    # #298 the fingerprint-scoped default cache resumes same-ED-config stamps on its own);
    # only legacy stamps under <output>/preprocessed still need the explicit directory.
    extra_flags: list[str] = []
    if not os.path.exists("/datag"):
        legacy_dir = os.path.join(output_path, "preprocessed")
        if not glob.glob(os.path.join(legacy_dir, "subset_test_*.npy")):
            pytest.skip("/datag not mounted and no preprocessed subset stamps to resume from")
        extra_flags = ["--preprocess-output-dir", legacy_dir]

    # --save-tag takes a BARE PREFIX since #272 (the run stamps its own datetime); this
    # smoke passed a bare datetime — rejected by validation — and had been silently broken
    # since the tag refactor (#298 repair). The resolved inf_{datetime} tag is recovered
    # from the run's own output below.
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
            "inf",
            "--max-retries",
            "1",
            *extra_flags,
        ]
    )

    tail = "\n".join(proc.stdout.splitlines()[-40:])
    assert proc.returncode == 0, f"inference smoke run failed; last output:\n{tail}"
    assert "Inference completed successfully!" in proc.stdout
    tag_match = re.search(r"\binf_\d{8}_\d{6}\b", proc.stdout)
    assert tag_match, "resolved inf_{datetime} tag not found in the run output"
    tag = tag_match.group(0)
    assert os.path.exists(os.path.join(output_path, f"config_{tag}.json"))

    # End-of-run benchmark report: pins the #203 _post_benchmark_report hook's real
    # db.flush -> render path end-to-end (output_path/plots/benchmark_report_{tag}.png).
    assert os.path.exists(os.path.join(output_path, "plots", f"benchmark_report_{tag}.png"))
