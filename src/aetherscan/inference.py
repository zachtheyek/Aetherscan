# TODO: write docstring
"""
Inference orchestration for Aetherscan Pipeline
Implements ...
Supports distributed datasets & latent generation
...
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time

import numpy as np
import tensorflow as tf

from aetherscan.config import get_config
from aetherscan.db import get_db
from aetherscan.models import RandomForestModel, Sampling

logger = logging.getLogger(__name__)


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
    data: dict,
    n_samples: int,
    per_replica_inf_batch_size: int,
    num_replicas: int,
    strategy: tf.distribute.Strategy,
) -> dict:
    """
    Prepare distributed datasets for inference
    Note, this function is not meant for RF training
    It is different from aetherscan.train.prepare_distributed_inf_dataset(),
    since we can't assume signal classes are known ahead of time

    Args:
        data: ndarray with shape (n_samples, 6, 16, 512)
        n_samples: Number of samples in data
        per_replica_inf_batch_size: Batch size per replica for inference
        num_replicas: Number of replicas in strategy
        strategy: TensorFlow distribution strategy

    Returns: {inf_dataset, n_inf_trimmed, inf_steps, _inf_holder}
             Inference distributed dataset, number of samples, number of steps,
              and DataHolder reference
    """
    global_inf_batch_size = per_replica_inf_batch_size * num_replicas

    # NOTE: does trimming/divisibility matter for inference?
    # Trim datasets to fit batch sizes (prevents uneven batches on final step)
    # Note, n_samples should already be divisible by effective_batch_size
    # Trimming here is just a defensive measure to doubly ensure divisibility before creating &
    # distributing our datasets
    # Alternatively, we could also pad the data instead of trimming
    # n_inf_trimmed = (n_samples // global_inf_batch_size) * global_inf_batch_size
    #
    # logger.info(f"Data alignment: Inf {n_samples}→{n_inf_trimmed}")
    n_inf_trimmed = n_samples

    # Sanity check: verify there's enough samples to run inference
    if n_inf_trimmed == 0:
        raise ValueError(
            f"Not enough samples ({n_samples}) for global batch size ({per_replica_inf_batch_size} * {num_replicas})"
            f"Reduce per_replica_batch_size or provide more samples"
        )

    # Prepare data
    inf_data = data[:n_inf_trimmed]
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
        # NOTE: do we need repeat for inf dataset?
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    # Distribute dataset across GPUs
    logger.info(f"Distributing dataset across {num_replicas} GPUs")

    inf_dataset_distributed = strategy.experimental_distribute_dataset(inf_dataset)

    # Calculate steps
    inf_steps = n_inf_trimmed // global_inf_batch_size

    # Sanity check: verify step sizes are valid before returning
    if inf_steps < 1:
        raise ValueError(
            f"inf_steps < 1: n_inf_trimmed ({n_inf_trimmed}) must be >= per_replica_inf_batch_size * num_replicas ({per_replica_inf_batch_size} * {num_replicas})"
        )

    return {
        "inf_dataset": inf_dataset_distributed,
        "n_inf_trimmed": n_inf_trimmed,
        "inf_steps": inf_steps,
        "_inf_holder": inf_holder,
    }


class InferencePipeline:
    """Inference pipeline"""

    def __init__(self, strategy: tf.distribute.Strategy = None):
        """
        Initialize inference pipeline.

        Args:
            strategy: TensorFlow distribution strategy
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

    # TODO: implement model loading from HuggingFace (parametrize args to InferenceConfig)
    def init_models(self, encoder_path: str, rf_path: str):
        """
        Initialize models within strategy scope
        """
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")
        if not os.path.exists(rf_path):
            raise FileNotFoundError(f"Random Forest not found at {rf_path}")

        try:
            # Load encoder within strategy scope
            logger.info(f"Loading encoder from {encoder_path} within strategy scope")
            with self.strategy.scope():
                self.encoder = tf.keras.models.load_model(
                    encoder_path, custom_objects={"Sampling": Sampling}
                )
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

    # TODO: finish writing docstring (how will args be passed?)
    def run_inference(
        self,
        data: np.ndarray,
        npy_path: str,
        # NOTE: how do we pass these in from preproc?
        target: str | None = None,
        session: str | None = None,
        cadence_id: int | None = None,
        band: str | None = None,
        frequency_mhz: float | None = None,
        timestamp_observed: float | None = None,
        h5_path: str | None = None,
    ) -> dict:
        """
        Run inference on preprocessed cadence snippets.

        Args:
            data: Preprocessed cadence snippets, shape (n, 6, 16, 512)
            npy_path: Source .npy file path containing the cadence snippets
            ...

        Returns:
            Dict with inference statistics
        """
        # Sanity check
        if not self.encoder or not self.rf_model:
            raise RuntimeError("Encoder and/or Random Forest not initialized")

        try:
            n_samples = data.shape[0]
            logger.info(f"Running inference on {n_samples} cadence snippets from {npy_path}")

            # Prepare distributed dataset for inference
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
            n_inf_trimmed = results["n_inf_trimmed"]
            inf_steps = results["inf_steps"]
            inf_holder = results["_inf_holder"]

            del results
            gc.collect()

            logger.info(
                f"Generating latents for {n_inf_trimmed} cadence snippets using distributed inference"
            )

            latents = self._distributed_encode(inf_dataset, n_inf_trimmed, inf_steps)

            # Run RF classification
            logger.info("Running Random Forest classification")
            predictions, confidence_scores = self.rf_model.predict_verbose(latents, self.threshold)

            # Write results to database
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
                timestamp_observed=timestamp_observed,
                h5_path=h5_path,
            )

        except Exception as e:
            logger.error(f"Error in run_inference(): {e}")
            raise  # Re-raise to propagate error

        finally:
            # NOTE: should check to make sure holder & dataset exist first
            # Clear intermediate data
            inf_holder.clear()
            del inf_dataset

            # Force TensorFlow to release internal references to datasets/iterators
            # This prevents generator closures from accumulating in memory between rounds
            tf.keras.backend.clear_session()
            logger.info("Cleared TensorFlow session state")

            # NOTE: should check to make sure arrays exist first
            del latents, predictions, confidence_scores
            gc.collect()

        # NOTE: is this the right location for return statement?
        return {
            "n_cadence_snippets": n_samples,
            "n_processed": n_inf_trimmed,
            "n_candidates": n_candidates,
        }

    def _distributed_encode(
        self,
        dataset: tf.distribute.DistributedDataset,
        n_samples: int,
        n_steps: int,
    ) -> np.ndarray:
        """
        Encode cadence snippets using distributed strategy.
        """
        # Pre-allocate latent array
        # Use np.empty() instead of np.zeros() so problematic latent values don't fail silently
        latents = np.empty((n_samples * self.num_observations, self.latent_dim), dtype=np.float32)

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
        timestamp_observed: float | None = None,
        h5_path: str | None = None,
    ) -> int:
        """Write inference results to database."""
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

                self.db.write_inference_result(
                    npy_path=npy_path,
                    # NOTE: is it guaranteed that snippets are processed sequentially?
                    snippet_index=idx,
                    prediction=prediction,
                    confidence=confidence,
                    # NOTE: does latents need to be reshaped first before being passed as arg?
                    latent_vector=latents[idx],
                    # NOTE: how do we pass these in from preproc?
                    target=target,
                    session=session,
                    cadence_id=cadence_id,
                    band=band,
                    frequency_mhz=frequency_mhz,
                    timestamp_observed=timestamp_observed,
                    h5_path=h5_path,
                    tag=tag,
                )

        logger.info(f"Wrote {n_candidates} candidates to database")
        return n_candidates

    # TODO: add plotting functions (remember to call when candidate is found) (full workflow when candidate is found: db write, make plot, save plot, send to slack)
    def plot_candidate(self):
        pass


# TODO: figure out how to pass preproc metadata into InferencePipeline (target, session, cadence_id, band, frequency_mhz, timestamp_observed, h5_path). should we roll these metadata + npy_path into a list/dict from preproc, then unroll them inside run_inference_pipeline()?
# TODO: add try-except switch statements (see run_training_pipeline())
def run_inference_pipeline(
    cadence_data: np.ndarray,
    npy_path: str,
    strategy: tf.distribute.Strategy,
    # NOTE: how do we pass these in from preproc?
    target: str | None = None,
    session: str | None = None,
    cadence_id: int | None = None,
    band: str | None = None,
    frequency_mhz: float | None = None,
    timestamp_observed: float | None = None,
    h5_path: str | None = None,
) -> dict:
    """
    Complete Aetherscan inference pipeline run

    Args:
        cadence_data: Array of preprocessed cadences, shape (n, 6, 16, 512)
        npy_path: Source .npy file path containing the cadence snippets
        strategy: TensorFlow distribution strategy
        ...
    """
    # Create pipeline
    pipeline = InferencePipeline(strategy=strategy)

    # Run inference
    results = pipeline.run_inference(
        data=cadence_data,
        # NOTE: how do we pass these in from preproc?
        npy_path=npy_path,
        target=target,
        session=session,
        cadence_id=cadence_id,
        band=band,
        frequency_mhz=frequency_mhz,
        timestamp_observed=timestamp_observed,
        h5_path=h5_path,
    )

    return results
