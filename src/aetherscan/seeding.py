"""
Root-seed stream derivation for reproducible pipeline runs (issues #49, #279).

A single root seed (config.reproducibility.seed, --seed on both subcommands) makes every
random stream in the pipeline deterministic. Rather than seeding numpy's global RNG once
(streams would collide and any consumer could perturb every other), each consumer derives its
own independent stream here via np.random.SeedSequence keyed on (root_seed, stream id,
...extras) — the same key always reproduces the same stream, and distinct keys are
statistically independent. Consumers that need a numpy Generator use derive_rng; APIs that
take an int random_state (sklearn, umap) use derive_seed; TensorFlow's global RNG (weight
init + the VAE Sampling layer) is seeded via seed_tensorflow, called by BOTH pipeline
constructors so training and inference cannot drift (#279 — inference used to be entirely
unseeded, making candidate sets unreproducible).

The Random Forest seed is likewise derived from the root (STREAM_RF); config.rf.seed remains
only as an explicit override for the deprecated --rf-seed flag.

Deliberately NOT seeded (do not "fix" these): uuid4 temp-file names (must stay real entropy),
round_data._array_checksum's content-keyed default_rng (keyed on the array, never the root,
or checksums stop being comparable), and data_generation's per-PID worker init seeding (always
overwritten by the per-task root-derived reseed before any draw).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Stream ids: the leading component of every derived stream key. Each consumer of the root seed
# owns one id (extra key components — e.g. the round index — subdivide it further), so no two
# consumers can ever end up sharing a stream. Add a new id here for any new consumer.
STREAM_DATA_GEN = 0  # per-round worker-task seeds (data_generation.generate_round_to_memmap)
STREAM_DATASET = 1  # stratified split / trim / epoch shuffles (prepare_distributed_train_dataset)
STREAM_VIZ = 2  # latent-viz batch selection + padding (train.TrainingPipeline)
STREAM_PLOT = 3  # plot subsampling (train.TrainingPipeline plot helpers)
STREAM_RF = 4  # RandomForest estimator random_state + pre-fit shuffle (models/random_forest)
STREAM_UMAP = 5  # every umap.UMAP random_state + fit-pool subsampling (sub-keyed per site)
STREAM_KMEANS = 6  # SHAP-space KMeans random_state (train.plot_rf_shap_explanation_clustering)
STREAM_TF = 7  # tf.random.set_seed roots — sub-keyed 0=training (then per round), 1=inference
STREAM_INFERENCE_VIZ = 8  # inference_viz latent budget-fill subsample
STREAM_SHAP_SAMPLES = 9  # SHAP summary/interaction row subsampling (feeds cached artifacts)
STREAM_RF_PLOTS = 10  # RF plot subsamples (sub-keyed: 0=learning curve, 1=decision boundary)

# One-shot flag so an unseeded run warns exactly once instead of per derived stream
_UNSEEDED_WARNED = False


def _warn_if_unseeded(root_seed: int | None) -> None:
    global _UNSEEDED_WARNED
    if root_seed is None and not _UNSEEDED_WARNED:
        logger.warning(
            "reproducibility.seed is None: every derived random stream falls back to OS "
            "entropy, so this run is NOT reproducible. Set --seed (or leave the default) "
            "for reproducible artifacts."
        )
        _UNSEEDED_WARNED = True


def derive_rng(root_seed: int | None, *stream_key: int) -> np.random.Generator:
    """
    Build the numpy Generator for one consumer stream of the pipeline root seed.

    With root_seed None this returns a fresh OS-entropy Generator (the historical
    non-reproducible behavior), warning once per process. Otherwise the Generator is seeded
    from SeedSequence([root_seed, *stream_key]), so the same (root_seed, stream_key) always
    yields an identical stream and different keys yield independent ones. `stream_key` must
    start with one of the STREAM_* ids above.
    """
    _warn_if_unseeded(root_seed)
    if root_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([root_seed, *stream_key]))


def derive_seed(root_seed: int | None, *stream_key: int) -> int:
    """
    The int-valued sibling of derive_rng, for APIs that take an integer random_state
    (sklearn RandomForestClassifier/KMeans/shuffle, umap.UMAP, tf.random.set_seed).

    Deterministic given (root_seed, stream_key); with root_seed None it returns a fresh
    OS-entropy int (warning once) so call sites can keep passing a CONCRETE random_state —
    per #279's constraint, determinism knobs stay set at every site that sets one today,
    only the value's provenance changes. Returns a value in [0, 2**32).
    """
    _warn_if_unseeded(root_seed)
    if root_seed is None:
        return int(np.random.SeedSequence().generate_state(1)[0])
    return int(np.random.SeedSequence([root_seed, *stream_key]).generate_state(1)[0])


def seed_tensorflow(root_seed: int | None, deterministic_ops: bool, *stream_key: int) -> int | None:
    """
    Seed TensorFlow's global RNG from the root and (optionally) force deterministic op
    implementations. Called by BOTH TrainingPipeline and InferencePipeline constructors —
    and again at round/cadence boundaries with extended stream keys, so a resumed or
    partially-rerun pipeline reproduces an uninterrupted one (#279: a single __init__-time
    set_seed is not resume-safe, because skipping rounds shifts the stream position).

    Returns the seed applied, or None when root_seed is None (TF keeps its own entropy;
    warned once). TF is imported lazily so this module stays importable without the
    scientific stack.
    """
    import tensorflow as tf  # noqa: PLC0415

    _warn_if_unseeded(root_seed)
    applied: int | None = None
    if root_seed is not None:
        applied = derive_seed(root_seed, STREAM_TF, *stream_key)
        tf.random.set_seed(applied)

    if deterministic_ops:
        # Deterministic kernels only pin op results; without a seed the draws still differ
        # run to run, so flag the (legal but pointless) combination loudly
        tf.config.experimental.enable_op_determinism()
        logger.info("Enabled deterministic TF op implementations (tf_deterministic_ops)")
        if root_seed is None:
            logger.warning(
                "tf_deterministic_ops is enabled but reproducibility.seed is None: "
                "deterministic kernels alone do not make runs reproducible"
            )
    return applied
