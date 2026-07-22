# NOTE: come back to this later

"""SNR→confidence monotonicity behavioral gate (cluster-only; issue #139 Gate 2).

The core scientific expectation for the trained VAE+RF: higher injected-signal SNR ⇒ higher
P(true). Aggregate metrics (ROC-AUC on a mixed-SNR val set) can miss a model that ignores
signal strength, so this test generates synthetic cadences at a few controlled SNRs with the
same injection code the training pipeline uses (`data_generation.create_true_single` with
`snr_range=0` pins the drawn SNR to `snr_base`), scores them in-process with the persisted
VAE encoder + Random Forest, and asserts the mean P(true) is non-decreasing in SNR within a
tolerance.

Unlike the two end-to-end smokes this does not subprocess `aetherscan.main` — it drives the
generation and scoring seams directly (no config singleton, no DB). All TF/aetherscan imports
are deferred into the test body so collecting this module never pulls TensorFlow into the
pytest parent process.

The tolerance is sized for the persisted smoke-trained dummy model (weakly trained, and the
encoder samples z stochastically); tighten it when gating a production model. Runs only on
the cluster inside the NGC container:

    ./utils/run_container.sh python -m pytest tests/integration -m "gpu or cluster" -q
"""

from __future__ import annotations

import os
import random
import shutil

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.cluster]

# NOTE: coupled to the same persisted dummy model as the smokes (see test_inference_smoke.py
# for the decoupling suggestion).
_MODEL_TAG = "test_v17"  # persisted dummy model on blpc3
_BACKGROUND_FILE = "real_filtered_LARGE_HIP110750.npy"  # first default train file
# Pinned to DataConfig's current defaults (config.py) rather than read live: test_v17 was
# trained against these values, so a future default change must not silently alter what this
# behavioral gate exercises.
_FREQ_RESOLUTION = 2.7939677238464355  # Hz
_TIME_RESOLUTION = 18.25361108  # seconds
# Controlled SNRs spanning the training curriculum (snr_base=10, initial range 40 → the
# model saw SNR 10-50 during training).
_SNRS = (10.0, 20.0, 35.0, 50.0)
_CADENCES_PER_SNR = 32
# Allowed decrease in mean P(true) between consecutive SNR levels; absorbs sampling noise
# from the stochastic encoder z and the finite cadence count.
_TOLERANCE = 0.05
_SEED = 139


def test_snr_confidence_monotonicity(cluster_paths):
    if shutil.which("nvidia-smi") is None:
        pytest.skip("requires a GPU host (nvidia-smi not found)")

    data_path, model_path, _ = cluster_paths
    encoder_path = os.path.join(model_path, f"vae_encoder_{_MODEL_TAG}.keras")
    rf_path = os.path.join(model_path, f"random_forest_{_MODEL_TAG}.joblib")
    background_path = os.path.join(data_path, "training", _BACKGROUND_FILE)
    for required in (encoder_path, rf_path, background_path):
        if not os.path.exists(required):
            pytest.skip(f"required cluster artifact missing: {required}")

    import joblib  # noqa: PLC0415
    import tensorflow as tf  # noqa: PLC0415

    from aetherscan.data_generation import create_true_single  # noqa: PLC0415
    from aetherscan.models import prepare_latent_features  # noqa: PLC0415

    # create_true_single/new_cadence draw from the global legacy RNGs (mirroring
    # data_generation._run_memmap_task's per-task seeding); tf seed pins the encoder's
    # z sampling.
    random.seed(_SEED)
    np.random.seed(_SEED)
    tf.random.set_seed(_SEED)

    # Background plates shaped (n, 6, 16, width_bin_downsampled), same as training. mmap
    # keeps the (potentially multi-GB) file on disk; create_true_single reads one
    # background per cadence.
    plate = np.load(background_path, mmap_mode="r")
    _, num_observations, time_bins, width_bin = plate.shape

    # Load the persisted models the same way inference does (encoder inside strategy scope).
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        encoder = tf.keras.models.load_model(encoder_path)
    rf = joblib.load(rf_path)

    mean_proba_true = []
    for snr in _SNRS:
        # snr_range=0.0 pins the per-cadence drawn SNR exactly to `snr` (the draw is
        # random.random() * snr_range + snr_base).
        cadences = np.stack(
            [
                create_true_single(
                    plate,
                    snr_base=snr,
                    snr_range=0.0,
                    width_bin=width_bin,
                    freq_resolution=_FREQ_RESOLUTION,
                    time_resolution=_TIME_RESOLUTION,
                )[0]
                for _ in range(_CADENCES_PER_SNR)
            ]
        ).astype(np.float32)

        # Score exactly like the inference pipeline: (n, 6, 16, W) -> (n*6, 16, W, 1)
        # snippets, sampled z from the encoder, 6 latents concatenated per cadence, RF
        # P(class=1).
        snippets = cadences.reshape(-1, time_bins, width_bin, 1)
        _, _, z = encoder(snippets, training=False)
        features = prepare_latent_features(z.numpy(), num_observations)
        proba_true = rf.predict_proba(features)[:, 1]
        mean_proba_true.append(float(proba_true.mean()))

    curve = ", ".join(
        f"SNR {snr:g} -> {mean:.4f}" for snr, mean in zip(_SNRS, mean_proba_true, strict=True)
    )
    for i in range(1, len(_SNRS)):
        assert mean_proba_true[i] >= mean_proba_true[i - 1] - _TOLERANCE, (
            f"mean P(true) decreased with SNR beyond tolerance {_TOLERANCE}: "
            f"SNR {_SNRS[i - 1]:g} -> {mean_proba_true[i - 1]:.4f} vs "
            f"SNR {_SNRS[i]:g} -> {mean_proba_true[i]:.4f} (full curve: {curve})"
        )
