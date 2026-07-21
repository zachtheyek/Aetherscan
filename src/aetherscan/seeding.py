"""
Root-seed stream derivation for reproducible pipeline runs (issue #49).

A single root seed (config.training.seed, --seed) makes every random stream in the training
pipeline deterministic. Rather than seeding numpy's global RNG once (streams would collide and
any consumer could perturb every other), each consumer derives its own independent Generator
here via np.random.SeedSequence keyed on (root_seed, stream id, ...extras) — the same key always
reproduces the same stream, and distinct keys are statistically independent.

TensorFlow's global RNG (weight init + the VAE Sampling layer) is seeded separately with
tf.random.set_seed(root_seed) in TrainingPipeline.__init__, and the Random Forest stage keeps
its own long-standing config.rf.seed.
"""

from __future__ import annotations

import numpy as np

# Stream ids: the leading component of every derived stream key. Each consumer of the root seed
# owns one id (extra key components — e.g. the round index — subdivide it further), so no two
# consumers can ever end up sharing a stream. Add a new id here for any new consumer.
STREAM_DATA_GEN = 0  # per-round worker-task seeds (data_generation.generate_round_to_memmap)
STREAM_DATASET = 1  # stratified split / trim / epoch shuffles (prepare_distributed_train_dataset)
STREAM_VIZ = 2  # latent-viz batch selection + padding (train.TrainingPipeline)
STREAM_PLOT = 3  # plot subsampling (train.TrainingPipeline plot helpers)


def derive_rng(root_seed: int | None, *stream_key: int) -> np.random.Generator:
    """
    Build the numpy Generator for one consumer stream of the pipeline root seed.

    With root_seed None (the config default) this returns a fresh OS-entropy Generator —
    the historical non-reproducible behavior. Otherwise the Generator is seeded from
    SeedSequence([root_seed, *stream_key]), so the same (root_seed, stream_key) always yields
    an identical stream and different keys yield independent ones. `stream_key` must start
    with one of the STREAM_* ids above.
    """
    if root_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([root_seed, *stream_key]))
