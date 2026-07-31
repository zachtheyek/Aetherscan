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
import time

import joblib
import numpy as np
import tensorflow as tf

from aetherscan.benchmark import stage_timer
from aetherscan.config import get_config
from aetherscan.db import get_db
from aetherscan.latent_variants import (
    apply_probability_calibrator,
    build_variant_features,
    sample_z_flat,
)
from aetherscan.models import RandomForestModel, prepare_latent_features
from aetherscan.seeding import (
    STREAM_INFERENCE_MC,
    STREAM_REFERENCE_CLOUD,
    derive_rng,
    seed_tensorflow,
)

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


# Final-partial-step bucket floor for _distributed_encode (#298 I2+I4): per-replica batch
# sizes are powers of two in [_MIN_ENCODE_BUCKET, inference.per_replica_batch_size], so the
# number of distinct traced shapes (and cuDNN autotune events) stays bounded across any
# catalog while tiny cadences never launch degenerate single-digit-row kernels.
_MIN_ENCODE_BUCKET = 16


def _batched_mc_scores(
    rf_model: RandomForestModel,
    calibrator: dict | None,
    variant: str,
    mean_flat: np.ndarray,
    logvar_flat: np.ndarray,
    num_observations: int,
    latent_dim: int,
    active_dims: list[int] | None,
    mc_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Score `mc_draws` reparameterized draws with ONE stacked predict_proba call, returning
    scores of shape (mc_draws, n_rows) — row-for-row what the old per-draw loop produced
    (#298 I8). Correctness rests on three invariants: the draws are generated SEQUENTIALLY
    from `rng` first, so the #279 stream consumption order is byte-identical to the loop;
    sklearn's predict_proba consumes no numpy RNG and scores rows independently, so batch
    composition cannot change any row's probability; and the calibrator is elementwise
    (isotonic interpolation / per-row logistic). What changes is only the fixed per-call
    overhead — check_array + a 1000-tree joblib dispatch — paid once instead of mc_draws
    times (dispatch dominates at typical survivor counts). Bit-identity is pinned by a
    unit test at n_jobs=1 (threaded tree accumulation order is sklearn's own run-to-run
    nondeterminism, present in the old loop too).
    """
    draw_features = [
        build_variant_features(
            variant,
            sample_z_flat(mean_flat, logvar_flat, rng),
            logvar_flat,
            num_observations,
            latent_dim,
            active_dims,
        )
        for _ in range(mc_draws)
    ]
    stacked_features = np.vstack(draw_features)
    # Release the per-draw list before the forest runs: holding it across predict_proba
    # doubled the MC transient (mc_draws x n_rows x n_features, twice) for no reason —
    # the stacked copy is the only thing the predict needs (#301)
    del draw_features
    stacked_scores = apply_probability_calibrator(
        calibrator, rf_model.model.predict_proba(stacked_features)[:, 1]
    )
    return stacked_scores.reshape(mc_draws, -1)


class ReferenceCloudReservoir:
    """
    Seeded uniform reservoir (algorithm R) over the pass-1 rejects' posterior parameters
    (#282): a fixed-size, catalog-representative sample — deliberately NOT near-threshold,
    which would bias the reference cloud toward the boundary and make every candidate look
    ordinary. offer() is O(1) per row; the MC scoring cost at finalize is fixed
    (capacity x mc_draws) regardless of survey size.
    """

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self.seen = 0
        self._mean_rows: list[np.ndarray] = []
        self._log_var_rows: list[np.ndarray] = []
        self._screening: list[float] = []

    def offer(
        self, mean_flat: np.ndarray, log_var_flat: np.ndarray, screening_probas: np.ndarray
    ) -> None:
        if self.capacity <= 0:
            return
        n_rows = len(mean_flat)
        start = 0
        # Fill phase (rare: only until the reservoir reaches capacity)
        while start < n_rows and len(self._mean_rows) < self.capacity:
            self._mean_rows.append(np.array(mean_flat[start]))
            self._log_var_rows.append(np.array(log_var_flat[start]))
            self._screening.append(float(screening_probas[start]))
            self.seen += 1
            start += 1
        if start >= n_rows:
            return

        # Replacement phase, vectorized (this runs for every reject batch of the survey, so
        # the per-row Python loop it replaces was a real cost): item t (1-indexed global
        # count) is accepted with probability capacity/t and lands in a uniform slot —
        # textbook algorithm R, with one rng call per batch instead of one per row
        remaining = n_rows - start
        t_values = self.seen + 1 + np.arange(remaining)
        accepted = self.rng.random(remaining) < (self.capacity / t_values)
        slots = self.rng.integers(0, self.capacity, size=remaining)
        self.seen += remaining
        for offset in np.nonzero(accepted)[0]:
            row_index = start + int(offset)
            slot = int(slots[offset])
            self._mean_rows[slot] = np.array(mean_flat[row_index])
            self._log_var_rows[slot] = np.array(log_var_flat[row_index])
            self._screening[slot] = float(screening_probas[row_index])

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._mean_rows:
            empty = np.empty((0, 0), dtype=np.float32)
            return empty, empty, np.empty(0, dtype=np.float32)
        return (
            np.stack(self._mean_rows),
            np.stack(self._log_var_rows),
            np.asarray(self._screening, dtype=np.float32),
        )


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

        # Reproducibility (#279): inference used to be entirely unseeded — the encoder's
        # Sampling layer drew fresh entropy every run, so the same encoder + RF + stamps
        # could yield a different candidate set on every run. Seed TF's global RNG from the
        # shared root seed (sub-key 1 = the inference stream; training uses 0) and honor
        # tf_deterministic_ops on this path too. run_inference re-seeds per cadence.
        applied_tf_seed = seed_tensorflow(
            self.config.reproducibility.seed,
            self.config.reproducibility.tf_deterministic_ops,
            1,
        )
        if applied_tf_seed is not None:
            logger.info(
                f"Seeded TF global RNG from root seed {self.config.reproducibility.seed} "
                f"(derived inference stream seed {applied_tf_seed})"
            )

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
        self.screening_threshold = self.config.inference.screening_threshold
        self.mc_draws = self.config.inference.mc_draws

        # Reference cloud (#282): a seeded uniform reservoir over the pass-1 REJECTS across
        # the whole run — MC-scored once at finalize so the candidate uncertainty plot
        # compares candidates against the survey, not against other candidates
        self._reference_reservoir = ReferenceCloudReservoir(
            capacity=self.config.inference.reference_cloud_size,
            rng=derive_rng(self.config.reproducibility.seed, STREAM_REFERENCE_CLOUD),
        )

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
            # NOTE: per-layer dtype policies are baked into the saved .keras file, so an
            # encoder trained with beta_vae.mixed_precision infers in bf16 automatically
            # (z_mean/z_log_var/Sampling stay fp32 islands) — no global policy call here.
            logger.info(f"Loading encoder from {encoder_path} within strategy scope")
            with self.strategy.scope():
                self.encoder = tf.keras.models.load_model(encoder_path)
                # Encode-only view (#301): inference consumes z_mean/z_log_var and
                # discards the sampled z, but the Sampling layer's tf.random draw is a
                # STATEFUL op that graph pruning must retain — it executed per step per
                # replica for nothing. Slicing the functional model's first two outputs
                # excludes the sampling branch from the traced graph entirely; z_mean /
                # z_log_var tensors are the same graph nodes, so outputs are unchanged.
                try:
                    self._encode_model = tf.keras.Model(
                        self.encoder.inputs, self.encoder.outputs[:2]
                    )
                except Exception as e:
                    logger.warning(
                        f"Encode-only submodel derivation failed ({e}); using the full "
                        f"encoder (the discarded sampling op will run per step)"
                    )
                    self._encode_model = self.encoder
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

        # Probability calibrator (#282): the saved training config records whether one is
        # active; applying it identically at inference is mandatory (an unapplied calibrator
        # is a silent train/serve mismatch), so a missing artifact is a hard error
        self.calibrator = None
        if self.config.rf.calibration_active:
            calibrator_path = os.path.join(
                os.path.dirname(rf_path),
                os.path.basename(rf_path).replace("random_forest", "rf_calibrator"),
            )
            if not os.path.exists(calibrator_path):
                raise FileNotFoundError(
                    f"config records an active probability calibrator "
                    f"({self.config.rf.calibration_method}) but {calibrator_path} does not "
                    "exist — refusing to score uncalibrated (train/serve mismatch)"
                )
            self.calibrator = joblib.load(calibrator_path)
            logger.info(
                f"Loaded probability calibrator ({self.calibrator['method']}) from "
                f"{calibrator_path}"
            )
        logger.info(
            f"Inference feature layout: latent_variant='{self.config.rf.latent_variant}', "
            f"active_dims={self.config.rf.active_dims}"
        )

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
        seed_key: int | None = None,
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

        # Reproducibility (#279): re-seed TF per cadence when the caller provides a stable
        # key (the streaming loop passes the catalog cadence index), so a cadence's sampled
        # latents depend only on (root seed, cadence) — reproducible even when the catalog
        # is subset or a run resumes partway
        if seed_key is not None:
            seed_tensorflow(self.config.reproducibility.seed, False, 1, seed_key)

        try:
            n_samples = data.shape[0]
            # Sanity check: verify there's at least one sample to run inference on
            if n_samples == 0:
                raise ValueError("Not enough samples (0) to run inference")
            logger.info(f"Running inference on {n_samples} cadence snippets from {npy_path}")

            if stamp_frequencies_mhz is not None and len(stamp_frequencies_mhz) != n_samples:
                logger.warning(
                    f"stamp_frequencies_mhz has {len(stamp_frequencies_mhz)} entries but "
                    f"{n_samples} snippets were loaded; ignoring per-stamp frequencies"
                )
                stamp_frequencies_mhz = None

            # Encode directly from numpy slices (#298 I2+I4): no per-cadence tf.data
            # dataset build / distribute / iter() churn, no full-array pad copy — see
            # _distributed_encode for the bucketed batch geometry and padding semantics.
            # Only the real snippets' latent rows come back (padding never leaves encode).
            with stage_timer("encode"):
                z_mean, z_log_var = self._distributed_encode(data)

            # NOTE: no gc.collect() here — `data` is still referenced by the caller
            # (main._infer_cadence holds cadence_data until after run_inference returns),
            # so a full collection frees nothing material and each one costs ~0.3 s with
            # TF's object graph loaded while holding the GIL against the prefetch thread
            # (#298 I7). The finally-block collect below is the one per-call collection
            # point.
            del data

            # Two-pass cascade (#282). Pass 1 scores EVERY snippet deterministically
            # (features per the saved config's winning variant, z_mean in the lead slot)
            # against the permissive screening threshold; pass 2 re-scores the survivors
            # with mc_draws seeded reparameterized draws and REPLACES their score with the
            # MC mean, which carries the science threshold. Not two ANDed criteria — pass 1
            # only exists to say "definitely not a candidate" cheaply.
            logger.info("Running Random Forest classification (two-pass cascade)")
            with stage_timer("rf"):
                variant = self.config.rf.latent_variant
                active_dims = self.config.rf.active_dims
                latent_dim = self.latent_dim
                num_observations = self.num_observations
                mean_flat = prepare_latent_features(z_mean, num_observations)
                logvar_flat = prepare_latent_features(z_log_var, num_observations)

                pass1_features = build_variant_features(
                    variant, mean_flat, logvar_flat, num_observations, latent_dim, active_dims
                )
                screening_probas = apply_probability_calibrator(
                    self.calibrator, self.rf_model.model.predict_proba(pass1_features)[:, 1]
                ).astype(np.float32)
                survivors = screening_probas > self.screening_threshold

                # Pass 2: seeded MC over the survivors. Draws are keyed on (root seed,
                # cadence seed_key), so a cadence's MC statistics are independent of the
                # rest of the catalog. Spread is computed on the SAME (calibrated when
                # active) scale as the plotted probabilities — documented choice.
                mc_mean = np.full(n_samples, np.nan, dtype=np.float32)
                mc_std = np.full(n_samples, np.nan, dtype=np.float32)
                survivor_idx = np.nonzero(survivors)[0]
                if len(survivor_idx):
                    # None gets the BARE stream key, a real cadence index its own sub-key:
                    # SeedSequence([root, id]) and SeedSequence([root, id, 0]) are distinct,
                    # so a keyless legacy call can never collide with catalog cadence 0
                    mc_key = (
                        (STREAM_INFERENCE_MC,)
                        if seed_key is None
                        else (STREAM_INFERENCE_MC, seed_key)
                    )
                    mc_rng = derive_rng(self.config.reproducibility.seed, *mc_key)
                    draw_scores = _batched_mc_scores(
                        self.rf_model,
                        self.calibrator,
                        variant,
                        mean_flat[survivor_idx],
                        logvar_flat[survivor_idx],
                        num_observations,
                        latent_dim,
                        active_dims,
                        self.mc_draws,
                        mc_rng,
                    )
                    mc_mean[survivor_idx] = draw_scores.mean(axis=0)
                    mc_std[survivor_idx] = draw_scores.std(axis=0)
                    del draw_scores

                # Final score: MC mean for survivors, the pass-1 score otherwise; the
                # science threshold applies to the final score. Confidence of a NEGATIVE is
                # 1 - final score on the DEPLOYED (calibrated when active) scale — a
                # deliberate semantic choice: the old probas[:, 0] equalled 1 - p only on
                # the raw scale, and a calibrator fit on the positive marginal does not
                # commute with the complement (1 - cal(p1) != cal(p0) in general). All
                # persisted/reported probabilities live on one scale this way.
                proba_true = np.where(survivors, np.nan_to_num(mc_mean), screening_probas)
                predictions = (proba_true > self.threshold).astype(int)
                confidence_scores = np.where(predictions, proba_true, 1.0 - proba_true)

                # Reference cloud: feed the pass-1 rejects to the seeded uniform reservoir
                reject_idx = np.nonzero(~survivors)[0]
                if len(reject_idx):
                    self._reference_reservoir.offer(
                        mean_flat[reject_idx],
                        logvar_flat[reject_idx],
                        screening_probas[reject_idx],
                    )

                # The per-observation z_mean rows are the latents consumers see (candidate
                # provenance vectors + the viz collector) — deterministic, reproducible
                latents = z_mean

            # Write results to database
            with stage_timer("db_write"):
                n_candidates = self._write_inference_results(
                    npy_path=npy_path,
                    predictions=predictions,
                    confidence_scores=confidence_scores,
                    latents=latents,
                    screening_probas=screening_probas,
                    mc_mean=mc_mean,
                    mc_std=mc_std,
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
            # One full collection per call (#298 I7 kept exactly this one) — clears any
            # cyclic debris from the encode/cascade path. tf.keras.backend.clear_session()
            # is intentionally NOT called here: the streaming loop reuses this pipeline's
            # loaded models across cadences; the caller clears the session once when the
            # whole run is done.
            gc.collect()

        return {
            "n_cadence_snippets": n_samples,
            "n_processed": n_samples,
            "n_candidates": n_candidates,
            "proba_true": proba_true,
            "predictions": predictions,
            "latents": latents,
            "screening_probas": screening_probas,
            "mc_mean": mc_mean,
            "mc_std": mc_std,
        }

    @staticmethod
    def _encode_bucket(remaining: int, num_replicas: int, max_bucket: int) -> int:
        """
        Per-replica batch size for a final partial encode step: the smallest power of two
        (floor _MIN_ENCODE_BUCKET, cap max_bucket) covering ceil(remaining / num_replicas).

        Bucketing bounds two costs at once: padding waste (always under one bucketed global
        batch per cadence — the old pad-to-full-global policy encoded a 100-stamp cadence
        as 10,240 rows) and the number of distinct traced shapes / cuDNN autotune events
        (at most len({16, 32, ..., max_bucket}) per run, however heterogeneous the catalog).
        """
        need = -(-remaining // num_replicas)  # ceil div
        bucket = _MIN_ENCODE_BUCKET
        while bucket < need and bucket < max_bucket:
            bucket *= 2
        return min(bucket, max_bucket)

    def _distributed_encode(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode `data` (n_samples, num_obs, time_bins, width) into pre-allocated
        (n_samples * num_observations, latent_dim) arrays of z_mean and z_log_var — the
        DETERMINISTIC posterior parameters (#282: pass 1 scores on z_mean-based features;
        pass-2 MC draws are reparameterized in numpy from these, seeded per cadence, so no
        stochastic z ever crosses the GPU boundary).

        Feeds the replicas directly from numpy slices via
        experimental_distribute_values_from_function (#298 I2 option (b)): no tf.data
        dataset, no per-cadence distribute/iter() rebuild (measured ~9.1 s per iter() on a
        5-GPU distributed dataset), no per-element generator crossings, and no full-array
        pad copy. Full steps run at per_replica_batch_size; the final partial step drops to
        a bucketed size (_encode_bucket) with its tail padded by cycling rows from the
        cadence start — padded outputs never leave this function (only real rows are
        written to the output arrays). Batch-shape changes can flip cuDNN algorithm
        selection, so kept-row latents may differ from the tf.data path in low bits —
        same science, gated on candidate-set + mc_mean/mc_std equality in the on-cluster
        A/B (#298). Per-replica results are gathered via experimental_local_results +
        np.concatenate (cheaper than an NCCL gather for the small latent payload).
        """
        n_samples = len(data)
        num_replicas = self.num_replicas
        max_bucket = self.per_replica_inf_batch_size

        # Pre-allocate output arrays
        # Use np.empty() instead of np.zeros() so problematic latent values don't fail silently
        z_mean_out = np.empty((n_samples * self.num_observations, self.latent_dim), np.float32)
        z_log_var_out = np.empty_like(z_mean_out)

        if self._encode_step is None:
            # Cache dimensions for tf.function
            time_bins = self.config.data.time_bins
            width_bin = self.config.data.width_bin // self.config.data.downsample_factor
            # The encode-only submodel when init_models derived one (#301: skips the
            # discarded sampling draw); a bare stub/full encoder still works — the first
            # two outputs are z_mean/z_log_var either way
            encode_model = getattr(self, "_encode_model", None) or self.encoder

            @tf.function
            def encode_step(batch_data):
                def encode_fn(data):
                    """Per-replica encoding step"""
                    # Reshape for encoder: (batch, 6, 16, 512) -> (batch * 6, 16, 512, 1)
                    reshaped = tf.reshape(data, [-1, time_bins, width_bin, 1])
                    outputs = encode_model(reshaped, training=False)
                    return outputs[0], outputs[1]  # z_mean, z_log_var

                # Run encoding on all replicas
                return self.strategy.run(encode_fn, args=(batch_data,))

            self._encode_step = encode_step

        encode_step = self._encode_step

        def _harvest(entry) -> None:
            # Extract results from each replica and concatenate (cheaper than an NCCL
            # gather for the small latent payload). The .numpy() calls block until the
            # step's kernels finish — which, in the pipelined loop below, overlaps the
            # NEXT step's device work instead of serializing against it.
            per_replica_mean, per_replica_log_var, out_rows, write_idx = entry
            batch_mean = np.concatenate(
                [r.numpy() for r in self.strategy.experimental_local_results(per_replica_mean)],
                axis=0,
            )
            batch_log_var = np.concatenate(
                [r.numpy() for r in self.strategy.experimental_local_results(per_replica_log_var)],
                axis=0,
            )
            # Only the real rows land in the outputs (obs-major: out_rows snippet rows)
            z_mean_out[write_idx : write_idx + out_rows] = batch_mean[:out_rows]
            z_log_var_out[write_idx : write_idx + out_rows] = batch_log_var[:out_rows]

        # One-step-deep software pipeline (#301): dispatch step k+1 (h2d + kernels are
        # enqueued asynchronously by eager TF) BEFORE harvesting step k, so the host-side
        # d2h + numpy writes of step k run while step k+1 computes on the GPUs. The old
        # loop harvested immediately, fully serializing convert -> compute -> sync per
        # step (~2% of benched encode throughput). Numerics are untouched: same tensors,
        # same step geometry, same write order — only the harvest timing moves.
        pending = None
        n_steps = 0
        out_idx = 0
        start = 0
        while start < n_samples:
            remaining = n_samples - start
            if remaining >= max_bucket * num_replicas:
                bucket = max_bucket
            else:
                bucket = self._encode_bucket(remaining, num_replicas, max_bucket)
            global_rows = bucket * num_replicas

            if remaining >= global_rows:
                step_slice = data[start : start + global_rows]
                real_rows = global_rows
            else:
                # Pad the final step with duplicate rows cycled from the cadence start
                # (deterministic, same semantics as the retired dataset padding); their
                # encoded outputs are sliced off in _harvest and never leave this method
                pad_indices = np.arange(global_rows - remaining) % n_samples
                step_slice = np.concatenate([data[start:], data[pad_indices]], axis=0)
                real_rows = remaining

            def value_fn(ctx, step_slice=step_slice, bucket=bucket):
                replica = ctx.replica_id_in_sync_group
                return tf.convert_to_tensor(step_slice[replica * bucket : (replica + 1) * bucket])

            per_replica_batch = self.strategy.experimental_distribute_values_from_function(value_fn)
            per_replica_mean, per_replica_log_var = encode_step(per_replica_batch)

            if pending is not None:
                _harvest(pending)

            out_rows = real_rows * self.num_observations
            pending = (per_replica_mean, per_replica_log_var, out_rows, out_idx)

            out_idx += out_rows
            start += real_rows
            n_steps += 1
            if n_steps % 10 == 0 or start >= n_samples:
                logger.info(f"Encoded {start}/{n_samples} snippets ({n_steps} steps)")

            # Refcount-managed numpy arrays and eager tensors: freed on del/rebind. The
            # full gc.collect() that used to run here EVERY STEP (~0.3 s each with TF
            # loaded, GIL held) reclaimed nothing cyclic (#298 I7). At most one extra
            # step of outputs stays live (the pipeline depth).
            del per_replica_mean, per_replica_log_var

        if pending is not None:
            _harvest(pending)

        return z_mean_out, z_log_var_out

    def _write_inference_results(
        self,
        npy_path: str,
        predictions: np.ndarray,
        confidence_scores: np.ndarray,
        latents: np.ndarray,
        screening_probas: np.ndarray | None = None,
        mc_mean: np.ndarray | None = None,
        mc_std: np.ndarray | None = None,
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

        def _optional_float(values, index):
            if values is None:
                return None
            value = float(values[index])
            return None if np.isnan(value) else value

        # NOTE: should we just store everything? benchmark storage requirements
        # Only store candidates above threshold (to reduce db size). Iterate the positives
        # directly (rider on #298 I8) — identical rows in identical order, without a Python
        # loop over every negative snippet of a 1e4-1e5-stamp cadence.
        positive_indices = np.nonzero(np.asarray(predictions) == 1)[0]
        n_candidates = len(positive_indices)
        for raw_idx in positive_indices:
            idx = int(raw_idx)
            confidence = float(confidence_scores[idx])
            prediction = int(predictions[idx])

            snippet_frequency_mhz = frequency_mhz
            if stamp_frequencies_mhz is not None:
                snippet_frequency_mhz = float(stamp_frequencies_mhz[idx])

            # One latent row per observation -> flatten the snippet's num_observations
            # rows into a single (num_observations * latent_dim,) vector
            latent_rows = latents[idx * self.num_observations : (idx + 1) * self.num_observations]

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
                screening_proba=_optional_float(screening_probas, idx),
                mc_mean=_optional_float(mc_mean, idx),
                mc_std=_optional_float(mc_std, idx),
            )

        logger.info(f"Wrote {n_candidates} candidates to database")
        return n_candidates

    def finalize_reference_cloud(self) -> str | None:
        """
        MC-score the reject reservoir and persist the reference cloud (#282): the
        (screening_proba, mc_mean, mc_std) triples for a seeded uniform subsample of the
        survey's pass-1 rejects, written to
        {output_path}/inference_reference_cloud_{tag}.npz together with the subsample
        size, total rejects seen, root seed, and draw count — so the candidate uncertainty
        plot can be regenerated later without re-running inference. Call once, after the
        last run_inference of the run. Returns the npz path (None when the cloud is
        disabled or no rejects were seen).
        """
        mean_flat, logvar_flat, screening = self._reference_reservoir.arrays()
        if len(screening) == 0:
            logger.info("Reference cloud: no pass-1 rejects collected — nothing to persist")
            return None

        logger.info(
            f"MC-scoring the reference cloud: {len(screening)} reject rows "
            f"(uniform reservoir over {self._reference_reservoir.seen} rejects) x "
            f"{self.mc_draws} draws"
        )
        variant = self.config.rf.latent_variant
        active_dims = self.config.rf.active_dims
        cloud_rng = derive_rng(self.config.reproducibility.seed, STREAM_REFERENCE_CLOUD, 1)
        draw_scores = _batched_mc_scores(
            self.rf_model,
            self.calibrator,
            variant,
            mean_flat,
            logvar_flat,
            self.num_observations,
            self.latent_dim,
            active_dims,
            self.mc_draws,
            cloud_rng,
        )

        tag = self.config.checkpoint.save_tag
        cloud_path = os.path.join(self.config.output_path, f"inference_reference_cloud_{tag}.npz")
        os.makedirs(os.path.dirname(cloud_path), exist_ok=True)
        np.savez_compressed(
            cloud_path,
            screening_proba=screening,
            mc_mean=draw_scores.mean(axis=0).astype(np.float32),
            mc_std=draw_scores.std(axis=0).astype(np.float32),
            subsample_size=np.int64(len(screening)),
            rejects_seen=np.int64(self._reference_reservoir.seen),
            mc_draws=np.int64(self.mc_draws),
            root_seed=np.int64(
                -1 if self.config.reproducibility.seed is None else self.config.reproducibility.seed
            ),
            # Explicit stream provenance so a reader can reproduce the cloud without
            # reverse-engineering the derivation: the reservoir consumes
            # derive_rng(root, STREAM_REFERENCE_CLOUD) and the MC scoring
            # derive_rng(root, STREAM_REFERENCE_CLOUD, 1)
            reservoir_stream_id=np.int64(STREAM_REFERENCE_CLOUD),
            mc_stream_key=np.array([STREAM_REFERENCE_CLOUD, 1], dtype=np.int64),
        )
        logger.info(f"Reference cloud persisted to {cloud_path}")
        return cloud_path

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
        # Reference cloud (#282), best-effort — the legacy single-shot path builds a fresh
        # pipeline per call, so the cloud covers this one cadence's rejects only
        try:
            pipeline.finalize_reference_cloud()
        except Exception as e:
            logger.error(f"Reference-cloud finalization failed ({e}); continuing")
    finally:
        # Force TensorFlow to release internal references to datasets/iterators. Runs once
        # per pipeline lifetime (after the loaded models are no longer needed) rather than
        # inside run_inference, which may be called repeatedly on live models.
        tf.keras.backend.clear_session()
        logger.info("Cleared TensorFlow session state")

    return results
