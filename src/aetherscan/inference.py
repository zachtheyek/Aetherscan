"""
Inference orchestration for Aetherscan Pipeline
Implements distributed encoding of preprocessed cadence snippets and Random Forest candidate
classification. Supports distributed datasets, per-replica latent generation, and writing
predictions / latent vectors to the database for downstream analysis.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time

import numpy as np
import tensorflow as tf

from aetherscan.benchmark import stage_timer
from aetherscan.config import get_config
from aetherscan.db import get_db
from aetherscan.models import RandomForestModel

logger = logging.getLogger(__name__)

# Quantile levels stored in each cadence's confidence summary (inference_cadences manifest)
_CONFIDENCE_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def summarize_confidences(proba_true: np.ndarray, threshold: float) -> dict:
    """
    Aggregate a cadence's P(true) vector into the JSON-serializable confidence summary
    stored on its inference_cadences manifest row.

    Returns {n, threshold, n_above_threshold, mean, min, max, quantiles} where quantiles maps
    'p01'/'p05'/.../'p99' to the corresponding quantile of proba_true. Keeping the summary in
    the manifest means run-level artifacts don't depend on the positives-only
    inference_results table. Raises ValueError on an empty vector (a cadence with zero
    snippets never reaches inference).
    """
    proba_true = np.asarray(proba_true, dtype=np.float64)
    if proba_true.size == 0:
        raise ValueError("summarize_confidences requires at least one confidence value")

    quantiles = np.quantile(proba_true, _CONFIDENCE_QUANTILES)
    return {
        "n": int(proba_true.size),
        "threshold": float(threshold),
        "n_above_threshold": int((proba_true > threshold).sum()),
        "mean": float(proba_true.mean()),
        "min": float(proba_true.min()),
        "max": float(proba_true.max()),
        "quantiles": {
            f"p{int(q * 100):02d}": float(v)
            for q, v in zip(_CONFIDENCE_QUANTILES, quantiles, strict=True)
        },
    }


# Create data holder objects, to be paired with data generators, for TF's distributed datasets
# Allows for explicit dereferencing of large arrays using DataHolder.clear(), which lets
# Python's garbage collector free up memory on-demand
# Note, DataHolder.clear() is only useful at the end of an epoch, once indices have been exhausted,
# since the data generators' local caches maintain references to the data until then
# This is not an issue in our current implementation, where we only clear & reset resources at the
# end of a round. However, if you require early exit behavior, you may want to remove the _lock and
# use explicit _cleared() checks instead, which negates the need for local caches (see commit hash
# 2a404a4). The trade-off being that you're at risk of race conditions if multiple threads attempt
# to access/clear the DataHolder simultaneously. While this is not the case in our current
# implementation, we opted for a more defensive approach rather than accomodating future design
# patterns. As well, the data should not be modified once the DataHolder has been initialized to
# prevent corrupted state in the DataHolder
# Note, there's a potential deadlock issue with DataHolder's lock contention
# Since the generators acquire locks at the start of every loop iteration, if TF's prefetch threads
# (.prefetch(tf.data.AUTOTUNE)) are blocked waiting on this lock while the main thread is trying to
# call self.data_generator.clear() (which also needs the lock), there could be contention.
# This has not been an issue so far, but if you encounter this in the future, pls update this
# comment with your findings
class InfDataHolder:
    def __init__(self, data):
        self._cleared = False
        self._lock = threading.Lock()
        self.data = data

    def clear(self):
        with self._lock:
            if self._cleared:
                return
            self._cleared = True
            self.data = None


def prepare_distributed_inf_dataset(
    data: np.ndarray,
    n_samples: int,
    per_replica_inf_batch_size: int,
    num_replicas: int,
    strategy: tf.distribute.Strategy,
) -> dict:
    """
    Build a distributed inference dataset from `data` (shape (n_samples, 6, 16, 512)) and return
    a dict with the dataset, padded/real sample counts, step count, and the InfDataHolder.

    Distinct from train.py's RF-training counterpart: signal classes are not assumed to be known
    ahead of time, so the dataset yields raw cadences without label channels. Order is preserved
    (no shuffle) since gradients aren't computed during inference.

    When n_samples isn't divisible by the global batch size, the data is padded with duplicate
    rows up to the next batch multiple (mirroring train.py's viz-dataset padding pattern) so the
    final partial batch is encoded rather than silently dropped — callers truncate the encoded
    latents back to n_samples. The previous behavior (inf_steps = n // global_batch with
    drop_remainder=True and no padding) never processed the tail; with per-cadence batches a
    cadence smaller than one global batch would have processed nothing at all.
    """
    global_inf_batch_size = per_replica_inf_batch_size * num_replicas

    # Sanity check: verify there's at least one sample to run inference on
    if n_samples == 0:
        raise ValueError("Not enough samples (0) to run inference")

    # Pad with duplicate rows (cycled from the front, deterministic) to the next global-batch
    # multiple; the encoded outputs for the padded rows are discarded by the caller.
    # NOTE: np.concatenate materializes a full copy of the cadence's stamps, briefly doubling
    # that cadence's memory footprint. Acceptable at per-cadence scale (the padding itself is
    # under one global batch); revisit if batches ever wrap multi-cadence arrays again.
    n_padded = int(np.ceil(n_samples / global_inf_batch_size)) * global_inf_batch_size
    if n_padded > n_samples:
        pad_count = n_padded - n_samples
        pad_indices = np.arange(pad_count) % n_samples
        inf_data = np.concatenate([data, data[pad_indices]], axis=0)
        logger.info(f"Data alignment: Inf {n_samples}→{n_padded} (padded {pad_count})")
    else:
        inf_data = data
        logger.info(f"Data alignment: Inf {n_samples} (no padding needed)")
    inf_holder = InfDataHolder(inf_data)

    # Create generator function for memory-efficient data loading
    def inf_generator():
        while True:  # Make generator infinite to reset state between passes
            # Acquire lock to check cleared status and capture data references
            # Local references keep data alive even if clear() is called mid-epoch
            with inf_holder._lock:
                if inf_holder._cleared:
                    return  # Exit if data already cleared
                # Cache references while holding lock
                data = inf_holder.data

            # Maintain order on each epoch since shuffling provides no benefits (no gradients
            # are calculated during inference)
            for idx in range(len(data)):
                yield data[idx]

            # Remove cache references for future garbage collection
            del data

    # Determine dataset output signature
    sample_shape = inf_data.shape[1:]
    output_signature = tf.TensorSpec(shape=sample_shape, dtype=tf.float32)

    # Create dataset using generator to reduce GPU memory pressure
    # Data is kept on CPU & transferred to GPU in batches on-demand
    # Note that the dataset yields data in batches before being sharded (distributed) across replicas
    # Hence, we use global batch sizes here to ensure per replica batch sizes match expectations
    logger.info(
        f"Creating infinite dataset from generator with global batch size: {global_inf_batch_size}"
    )

    inf_dataset = (
        tf.data.Dataset.from_generator(inf_generator, output_signature=output_signature)
        .batch(global_inf_batch_size, drop_remainder=True)
        # NOTE: do we need repeat for inf dataset? run test without repeat & see if anything breaks?
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    # Distribute dataset across GPUs
    logger.info(f"Distributing dataset across {num_replicas} GPUs")

    inf_dataset_distributed = strategy.experimental_distribute_dataset(inf_dataset)

    # Calculate steps (n_padded is an exact multiple of the global batch size by construction)
    inf_steps = n_padded // global_inf_batch_size

    return {
        "inf_dataset": inf_dataset_distributed,
        "n_padded": n_padded,
        "n_samples": n_samples,
        "inf_steps": inf_steps,
        "_inf_holder": inf_holder,
    }


class InferencePipeline:
    """Inference pipeline"""

    def __init__(self, strategy: tf.distribute.Strategy = None):
        """
        Initialize the inference pipeline with an optional tf.distribute strategy (defaults to
        the current strategy, i.e. no-op for single-device).
        """
        self.config = get_config()
        if self.config is None:
            raise ValueError("get_config() returned None")

        self.db = get_db()
        if self.db is None:
            raise ValueError("get_db() returned None")
        self.start_time = time.time()

        # Set distributed strategy
        self.strategy = strategy or tf.distribute.get_strategy()
        self.num_replicas = self.strategy.num_replicas_in_sync

        # Initialize models
        self.init_models(
            encoder_path=self.config.inference.encoder_path, rf_path=self.config.inference.rf_path
        )

        self.latent_dim = self.config.beta_vae.latent_dim
        self.num_observations = self.config.data.num_observations

        self.per_replica_inf_batch_size = self.config.inference.per_replica_batch_size
        self.threshold = self.config.inference.classification_threshold

        # Lazily-built tf.function for the distributed encode step, cached so repeated
        # run_inference calls (one per cadence in the streaming loop) reuse one trace
        # instead of accumulating a new concrete function per cadence
        self._encode_step = None

    # NOTE: HuggingFace Hub model loading is handled upstream: when no local artifact paths
    # are given, hf_hub.resolve_inference_artifacts() (called from main()) downloads the
    # revision-pinned artifacts and routes their cache paths into config.inference, so the
    # paths received here are always local files
    def init_models(self, encoder_path: str, rf_path: str):
        """Load the VAE encoder (inside strategy.scope) and the Random Forest classifier from disk."""
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")
        if not os.path.exists(rf_path):
            raise FileNotFoundError(f"Random Forest not found at {rf_path}")

        try:
            # Load encoder within strategy scope
            logger.info(f"Loading encoder from {encoder_path} within strategy scope")
            with self.strategy.scope():
                self.encoder = tf.keras.models.load_model(encoder_path)
            logger.info("Encoder loaded successfully")
        except Exception as e:
            logger.error(f"Error loading encoder: {e}")
            raise  # Re-raise to propagate error

        try:
            # Load Random Forest
            logger.info(f"Loading Random Forest from {rf_path}")
            self.rf_model = RandomForestModel()
            self.rf_model.load(rf_path)
            logger.info("Random Forest loaded successfully")

        except Exception as e:
            logger.error(f"Error loading Random Forest: {e}")
            raise  # Re-raise to propagate error

    def run_inference(
        self,
        data: np.ndarray,
        npy_path: str,
        target: str | None = None,
        session: str | None = None,
        cadence_id: int | None = None,
        band: str | None = None,
        frequency_mhz: float | None = None,
        stamp_frequencies_mhz: list[float] | None = None,
        timestamp_observed: float | None = None,
        h5_path: str | None = None,
    ) -> dict:
        """
        Run inference on preprocessed cadence snippets (shape (n, 6, 16, 512)) sourced from
        npy_path, and return {n_cadence_snippets, n_processed, n_candidates, proba_true,
        predictions, latents}. proba_true is the per-snippet P(true) vector, predictions the
        thresholded 0/1 array, and latents the truncated per-observation latent array of shape
        (n * num_observations, latent_dim) — callers use them for the per-cadence manifest
        summary and the visualization suite, then drop them (they are per-cadence transients,
        never accumulated catalog-wide).

        Encodes each snippet through the VAE encoder under the distribution strategy, then runs
        the Random Forest classifier on the latents and writes positive predictions to the
        database (along with the latent vector and observational provenance: target, session,
        cadence_id, band, frequency, timestamp_observed, h5_path — typically derived by
        preprocessing.derive_cadence_provenance from the cadence's metadata JSON).
        stamp_frequencies_mhz, when given, carries one center frequency per snippet row (the
        metadata's stamp_frequencies_mhz) and takes precedence over the scalar frequency_mhz.

        Safe to call repeatedly on one pipeline instance (the per-cadence streaming loop does):
        models stay loaded, and per-call dataset state is released in the finally block. The
        caller owns tf.keras.backend.clear_session() — see run_inference_pipeline /
        main.inference_command.
        """
        # Sanity check
        if not self.encoder or not self.rf_model:
            raise RuntimeError("Encoder and/or Random Forest not initialized")

        inf_holder = None
        inf_dataset = None

        try:
            n_samples = data.shape[0]
            logger.info(f"Running inference on {n_samples} cadence snippets from {npy_path}")

            if stamp_frequencies_mhz is not None and len(stamp_frequencies_mhz) != n_samples:
                logger.warning(
                    f"stamp_frequencies_mhz has {len(stamp_frequencies_mhz)} entries but "
                    f"{n_samples} snippets were loaded; ignoring per-stamp frequencies"
                )
                stamp_frequencies_mhz = None

            # Prepare distributed dataset for inference (pads to a global-batch multiple)
            results = prepare_distributed_inf_dataset(
                data=data,
                n_samples=n_samples,
                per_replica_inf_batch_size=self.per_replica_inf_batch_size,
                num_replicas=self.num_replicas,
                strategy=self.strategy,
            )

            del data
            gc.collect()

            inf_dataset = results["inf_dataset"]
            n_padded = results["n_padded"]
            inf_steps = results["inf_steps"]
            inf_holder = results["_inf_holder"]

            del results
            gc.collect()

            logger.info(
                f"Generating latents for {n_samples} cadence snippets "
                f"({n_padded} padded) using distributed inference"
            )

            with stage_timer("encode"):
                latents = self._distributed_encode(inf_dataset, n_padded, inf_steps)

            # Drop the encoded padding rows: the first n_samples * num_observations latent
            # rows correspond exactly to the real snippets (row order is snippet-major)
            latents = latents[: n_samples * self.num_observations]

            # Run RF classification. One predict_proba pass (1000 trees per snippet) yields
            # everything downstream: P(true) for the manifest confidence summary and the
            # confidence-distribution figure, plus the same predictions / confidences
            # predict_verbose would have derived from it (probability of the predicted
            # class, so high for confident negatives too).
            logger.info("Running Random Forest classification")
            with stage_timer("rf"):
                probas = self.rf_model.predict_proba(latents)
                proba_true = probas[:, 1]
                predictions = (proba_true > self.threshold).astype(int)
                confidence_scores = np.where(predictions, proba_true, probas[:, 0])

            # Write results to database
            with stage_timer("db_write"):
                n_candidates = self._write_inference_results(
                    npy_path=npy_path,
                    predictions=predictions,
                    confidence_scores=confidence_scores,
                    latents=latents,
                    target=target,
                    session=session,
                    cadence_id=cadence_id,
                    band=band,
                    frequency_mhz=frequency_mhz,
                    stamp_frequencies_mhz=stamp_frequencies_mhz,
                    timestamp_observed=timestamp_observed,
                    h5_path=h5_path,
                )

        except Exception as e:
            logger.error(f"Error in run_inference(): {e}")
            raise  # Re-raise to propagate error

        finally:
            # Clear per-call dataset state (guarded: an early failure may predate creation).
            # Note tf.keras.backend.clear_session() is intentionally NOT called here — the
            # streaming loop reuses this pipeline's loaded models across cadences; the caller
            # clears the session once when the whole run is done.
            if inf_holder is not None:
                inf_holder.clear()
            del inf_dataset
            gc.collect()

        return {
            "n_cadence_snippets": n_samples,
            "n_processed": n_samples,
            "n_candidates": n_candidates,
            "proba_true": proba_true,
            "predictions": predictions,
            "latents": latents,
        }

    def _distributed_encode(
        self,
        dataset: tf.distribute.DistributedDataset,
        n_samples: int,
        n_steps: int,
    ) -> np.ndarray:
        """
        Encode `n_steps` worth of batches from a distributed `dataset` into a pre-allocated
        latent array of shape (n_samples * num_observations, latent_dim). Per-replica results
        are gathered via experimental_local_results + np.concatenate (faster than a strategy-level
        gather over NCCL for the small latent payload).
        """
        # Pre-allocate latent array
        # Use np.empty() instead of np.zeros() so problematic latent values don't fail silently
        latents = np.empty((n_samples * self.num_observations, self.latent_dim), dtype=np.float32)

        if self._encode_step is None:
            # Cache dimensions for tf.function
            time_bins = self.config.data.time_bins
            width_bin = self.config.data.width_bin // self.config.data.downsample_factor

            @tf.function
            def encode_step(batch_data):
                def encode_fn(data):
                    """Per-replica encoding step"""
                    # Reshape for encoder: (batch, 6, 16, 512) -> (batch * 6, 16, 512, 1)
                    reshaped = tf.reshape(data, [-1, time_bins, width_bin, 1])
                    # Encode (returns mean, log_var, z)
                    _, _, z = self.encoder(reshaped, training=False)
                    return z

                # Run encoding on all replicas
                per_replica_z = self.strategy.run(encode_fn, args=(batch_data,))
                return per_replica_z

            self._encode_step = encode_step

        encode_step = self._encode_step

        # Process all batches
        iterator = iter(dataset)
        current_idx = 0

        try:
            for step in range(n_steps):
                batch = next(iterator)

                # Get per-replica latents for this batch
                per_replica_z = encode_step(batch)

                # Extract results from each replica and concatenate
                # This avoids the inefficient gather operation with NCCL
                results = self.strategy.experimental_local_results(per_replica_z)

                # Concatenate results from all replicas
                batch_z = np.concatenate([r.numpy() for r in results], axis=0)

                batch_size = batch_z.shape[0]
                latents[current_idx : current_idx + batch_size] = batch_z

                current_idx += batch_size

                # Log progress
                if (step + 1) % 10 == 0 or (step + 1) == n_steps:
                    logger.info(f"Encoded step {step + 1}/{n_steps}")

                del per_replica_z, results, batch_z
                gc.collect()

        except Exception as e:
            logger.error(f"Error in _distributed_encode(): {e}")
            raise  # Re-raise to propagate error

        finally:
            # NOTE: should check to make sure iterator exist first
            del iterator

        # NOTE: is this the right location for return statement?
        return latents

    def _write_inference_results(
        self,
        npy_path: str,
        predictions: np.ndarray,
        confidence_scores: np.ndarray,
        latents: np.ndarray,
        target: str | None = None,
        session: str | None = None,
        cadence_id: int | None = None,
        band: str | None = None,
        frequency_mhz: float | None = None,
        stamp_frequencies_mhz: list[float] | None = None,
        timestamp_observed: float | None = None,
        h5_path: str | None = None,
    ) -> int:
        """Write inference results to database.

        Predictions/confidences are indexed per snippet (dataset order is preserved end to end:
        the .npy rows, the padded dataset, and the truncated latents all share snippet-major
        ordering, so snippet_index == the row index in npy_path). stamp_frequencies_mhz, when
        given, supplies the per-snippet center frequency (falling back to the scalar
        frequency_mhz otherwise); latents rows are per observation, so snippet idx spans
        latents[idx * num_observations : (idx + 1) * num_observations], flattened to one
        (num_observations * latent_dim,) provenance vector.
        """
        if self.db is None:
            raise RuntimeError("No database instance detected - cannot store inference results")

        tag = self.config.checkpoint.save_tag
        n_candidates = 0

        for idx in range(len(confidence_scores)):
            confidence = float(confidence_scores[idx])
            prediction = int(predictions[idx])

            # NOTE: should we just store everything? benchmark storage requirements
            # Only store candidates above threshold (to reduce db size)
            if prediction == 1:
                n_candidates += 1

                snippet_frequency_mhz = frequency_mhz
                if stamp_frequencies_mhz is not None:
                    snippet_frequency_mhz = float(stamp_frequencies_mhz[idx])

                # One latent row per observation -> flatten the snippet's num_observations
                # rows into a single (num_observations * latent_dim,) vector
                latent_rows = latents[
                    idx * self.num_observations : (idx + 1) * self.num_observations
                ]

                self.db.write_inference_result(
                    npy_path=npy_path,
                    snippet_index=idx,
                    prediction=prediction,
                    confidence=confidence,
                    latent_vector=latent_rows.reshape(-1),
                    target=target,
                    session=session,
                    cadence_id=cadence_id,
                    band=band,
                    frequency_mhz=snippet_frequency_mhz,
                    timestamp_observed=timestamp_observed,
                    h5_path=h5_path,
                    tag=tag,
                )

        logger.info(f"Wrote {n_candidates} candidates to database")
        return n_candidates

    # NOTE: candidate plotting lives in aetherscan.inference_viz (plot_candidate /
    # plot_candidate_gallery), rendered at end of run from the inference_results rows +
    # stamp .npy files so it also covers cadences skipped by the stage-aware resume.


# TODO: add try-except switch statements (see run_training_pipeline())
def run_inference_pipeline(
    cadence_data: np.ndarray,
    npy_path: str,
    strategy: tf.distribute.Strategy,
    target: str | None = None,
    session: str | None = None,
    cadence_id: int | None = None,
    band: str | None = None,
    frequency_mhz: float | None = None,
    stamp_frequencies_mhz: list[float] | None = None,
    timestamp_observed: float | None = None,
    h5_path: str | None = None,
) -> dict:
    """
    Single-shot inference entry point: build an InferencePipeline under `strategy`, run it once
    against `cadence_data` (shape (n, 6, 16, 512)) sourced from `npy_path`, and clear the TF
    session. Used by the legacy --test-files path, which loads one preprocessed array up front;
    the CSV streaming path in main.inference_command instead builds one InferencePipeline and
    calls run_inference per cadence so models load once.

    The optional provenance arguments (target/session/cadence_id/band/frequency_mhz/
    stamp_frequencies_mhz/timestamp_observed/h5_path) are written to the inference_results
    table for any positive candidates. Returns run_inference's results dict
    ({n_cadence_snippets, n_processed, n_candidates, proba_true, predictions, latents}).
    """
    # Create pipeline
    pipeline = InferencePipeline(strategy=strategy)

    try:
        # Run inference. The umbrella span makes run_inference's encode/rf/db_write
        # sub-stages record as "inference.infer.*" on this legacy path (the streaming
        # path opens a per-cadence "inference.infer_cadence_NNN" umbrella instead)
        with stage_timer("inference.infer"):
            results = pipeline.run_inference(
                data=cadence_data,
                npy_path=npy_path,
                target=target,
                session=session,
                cadence_id=cadence_id,
                band=band,
                frequency_mhz=frequency_mhz,
                stamp_frequencies_mhz=stamp_frequencies_mhz,
                timestamp_observed=timestamp_observed,
                h5_path=h5_path,
            )
    finally:
        # Force TensorFlow to release internal references to datasets/iterators. Runs once
        # per pipeline lifetime (after the loaded models are no longer needed) rather than
        # inside run_inference, which may be called repeatedly on live models.
        tf.keras.backend.clear_session()
        logger.info("Cleared TensorFlow session state")

    return results
