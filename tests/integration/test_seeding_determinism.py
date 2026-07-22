# NOTE: come back to this later

"""GPU-path seeding determinism smoke (cluster-only; issue #207's --tf-deterministic-ops).

Pins the promise behind --seed + --tf-deterministic-ops: with the TF global RNG seeded and op
determinism enabled, a run is bit-exact reproducible on GPU. Building the beta-VAE and taking
one optimizer step over identical inputs *twice* must yield byte-identical encoder weights.
This is the only automated coverage that the deterministic-kernel path actually delivers
reproducibility — weight init (HeNormal/GlorotNormal) and the VAE Sampling layer's epsilon both
draw from the seeded global RNG, so a regression in the seeding wiring would desynchronize them.

Like test_model_behavior.py this drives the real seams in-process (not a subprocess of
`aetherscan.main`): it uses the pipeline's own `create_beta_vae_model()` construction path and a
train step mirroring `TrainingPipeline._distributed_train_step` + `_apply_gradients`
(GradientTape -> `compute_total_loss` -> `clip_by_global_norm` -> `optimizer.apply_gradients`).
The inputs are fixed synthetic snippets, so no cluster-resident data is read; the gpu/cluster
markers keep it in the same GPU-host-only selection as the other smokes (deterministic cuDNN
kernels are only meaningful on real GPUs). All TF/aetherscan imports are deferred into the test
body so collecting this module never pulls TensorFlow into the pytest parent process.

    ./utils/run_container.sh python -m pytest tests/integration -m "gpu or cluster" -q
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.cluster]

_SEED = 207
# Two cadences (12 observations) exercise reconstruction + KL + both clustering heads; the shape
# matches the VAE's expected cadence form (batch, 6, 16, 512) — see BetaVAE.call in models/vae.py.
_BATCH = 2
_CADENCE_SHAPE = (_BATCH, 6, 16, 512)


def test_seeding_yields_byte_identical_encoder_weights():
    if shutil.which("nvidia-smi") is None:
        pytest.skip("requires a GPU host (nvidia-smi not found)")

    import tensorflow as tf  # noqa: PLC0415

    from aetherscan.config import get_config, init_config  # noqa: PLC0415
    from aetherscan.models import create_beta_vae_model  # noqa: PLC0415

    # Integration tests skip the autouse config-init fixture, so bootstrap the singleton here,
    # then pin the seed the construction path reads (create_beta_vae_model -> get_config). Single
    # -threaded test, so the mutate-config-post-init caveat does not apply.
    if get_config() is None:
        init_config()
    get_config().training.seed = _SEED
    seed = get_config().training.seed

    # Deterministic cuDNN/reduction kernels (the --tf-deterministic-ops promise). Process-global
    # and only meaningful on a real GPU, hence the gpu/cluster markers + nvidia-smi guard above.
    tf.config.experimental.enable_op_determinism()

    # Fixed synthetic snippets, generated once and reused across both runs so the two builds see
    # byte-identical inputs. Values in [0, 1) match the sigmoid-bounded reconstruction target of
    # the binary-cross-entropy loss; target_data = main_data, as in the real reconstruction path.
    rng = np.random.default_rng(_SEED)
    main_data = tf.constant(rng.random(size=_CADENCE_SHAPE, dtype=np.float32))
    true_data = tf.constant(rng.random(size=_CADENCE_SHAPE, dtype=np.float32))
    false_data = tf.constant(rng.random(size=_CADENCE_SHAPE, dtype=np.float32))
    target_data = main_data

    strategy = tf.distribute.get_strategy()

    def _build_and_train_once() -> list[np.ndarray]:
        # Seed the TF global RNG before any variable creation so weight init and the Sampling
        # layer's epsilon draw the same stream each run — exactly the ordering TrainingPipeline
        # uses (tf.random.set_seed before create_beta_vae_model, inside strategy.scope()).
        tf.random.set_seed(seed)
        with strategy.scope():
            vae = create_beta_vae_model()
            # One real train step: mirrors _distributed_train_step's per-replica body and
            # _apply_gradients (global-norm gradient clipping at 1.0, then Adam apply).
            with tf.GradientTape() as tape:
                losses = vae.compute_total_loss(
                    main_data, true_data, false_data, target_data, training=True
                )
            gradients = tape.gradient(losses["total_loss"], vae.trainable_variables)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            vae.optimizer.apply_gradients(
                zip(clipped_gradients, vae.trainable_variables, strict=False)
            )
            return vae.encoder.get_weights()

    weights_first = _build_and_train_once()
    weights_second = _build_and_train_once()

    assert len(weights_first) == len(weights_second)
    for w_first, w_second in zip(weights_first, weights_second, strict=True):
        np.testing.assert_array_equal(w_first, w_second)
