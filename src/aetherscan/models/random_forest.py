# TODO: refactor to expose public APIs for creating & destroying RandomForestModel instances
"""
Random Forest classifier implementation for Aetherscan Pipeline
Receives concatenated latents grouped by their original 6-observation cadence pattern
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

from aetherscan.config import get_config

logger = logging.getLogger(__name__)


def prepare_latent_features(
    latent_vectors: np.ndarray,
    num_observations: int = 6,
    dtype: np.typing.DTypeLike = np.float64,
) -> np.ndarray:
    """
    Reshape per-observation latent vectors of shape (num_cadences * num_observations, latent_dim)
    into per-cadence features of shape (num_cadences, num_observations * latent_dim), so each
    cadence's 6 latents are concatenated into a single feature row for the RF.

    Caller must keep row i..i+num_observations-1 grouped as cadence i. Raises ValueError if the
    row count isn't divisible by num_observations. `dtype` defaults to float64 (see the note at
    the return) — training callers MUST keep it; inference passes float32.
    """
    # Expected shape: (num_cadences * num_observations, latent_dim)
    num_latents = latent_vectors.shape[0]

    if num_latents % num_observations != 0:
        raise ValueError(
            f"Received {num_latents} latent vectors. Not divisible by num_observations ({num_observations})"
        )

    num_cadences = num_latents // num_observations
    latent_dim = latent_vectors.shape[1]

    # Target shape: (num_cadences, num_observations * latent_dim)
    # Where each element in the latent vector is treated as a feature by the Random Forest
    # We flatten the observations so all 6 latents in a cadence are grouped together.
    # A row-major reshape IS that flatten (row i = rows i*num_obs..(i+1)*num_obs-1
    # raveled) — the per-row Python loop this replaces cost ~0.1-0.5 s per RFI-dense
    # cadence at inference (#301). `dtype` defaults to float64 and MUST stay float64 for
    # the TRAINING callers: the Active-Units gate computes .var() on this output, whose
    # float32 accumulation differs in exactly the low bits that decide a hovering-at-
    # threshold dim (#288's margin lesson). Inference passes dtype=float32 — it never runs
    # that .var() gate (active_dims is loaded from the saved config, not recomputed), and
    # every inference consumer (build_variant_features, sample_z_flat, predict_proba) casts
    # to float32 anyway, so float32 here is byte-identical downstream (float32->float64->
    # float32 round-trips exactly) while skipping two full-matrix float64 widenings per
    # cadence on the GPU-idle RF stage.
    return np.asarray(latent_vectors, dtype=dtype).reshape(
        num_cadences, num_observations * latent_dim
    )


class RandomForestModel:
    """Random Forest classifier for SETI signal detection"""

    def __init__(self):
        self.config = get_config()
        if self.config is None:
            raise ValueError("get_config() returned None")

        self.model = RandomForestClassifier(
            n_estimators=self.config.rf.n_estimators,
            bootstrap=self.config.rf.bootstrap,
            max_features=self.config.rf.max_features,
            n_jobs=self.config.rf.n_jobs,
            # Derived from the pipeline root seed unless --rf-seed overrides (#279)
            random_state=self.config.resolved_rf_seed(),
        )

        self.is_trained = False

    def train(self, latent_vectors: np.ndarray, binary_labels: np.ndarray):
        """
        Fit the Random Forest on `latent_vectors` of shape
        (n_cadences * num_observations, latent_dim) and `binary_labels` of shape (n_cadences,)
        (0 = false, 1 = true signal). latent_vectors must be grouped so row
        i..i+num_observations-1 corresponds to cadence i — this is what
        prepare_latent_features expects.
        """
        # Prepare features
        features = prepare_latent_features(latent_vectors, self.config.data.num_observations)

        # Sanity check: make sure length of feature & label arrays are aligned
        if features.shape[0] != binary_labels.shape[0]:
            raise ValueError(
                f"Feature/label count mismatch: {features.shape[0]} vs {binary_labels.shape[0]}"
            )

        # Shuffle data
        features, binary_labels = shuffle(
            features, binary_labels, random_state=self.config.resolved_rf_seed()
        )
        logger.info(f"Prepared {features.shape[0]} training samples")

        # Start training
        logger.info("Training Random Forest classifier...")
        self.model.fit(features, binary_labels)
        self.is_trained = True

    def predict_proba(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Predict per-class probabilities for each cadence; returns shape (n_cadences, 2) with
        columns [P(class=0), P(class=1)]. latent_vectors follows the same grouped layout as
        train().
        """
        if not self.is_trained:
            logger.warning("Making predictions with untrained model")

        features = prepare_latent_features(latent_vectors, self.config.data.num_observations)
        return self.model.predict_proba(features)

    def predict(self, latent_vectors: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Binary class predictions: 1 where P(class=1) > threshold, else 0. Returns shape
        (n_cadences,) int array.
        """
        probas = self.predict_proba(latent_vectors)
        return (probas[:, 1] > threshold).astype(int)

    def predict_verbose(
        self, latent_vectors: np.ndarray, threshold: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Like predict(), but also returns per-cadence confidence (the probability of the predicted
        class — so high for confident negatives too, not just confident positives). Returns
        (predictions, confidences), each of shape (n_cadences,).
        """
        probas = self.predict_proba(latent_vectors)
        predictions = (probas[:, 1] > threshold).astype(int)

        # Confidence score = the probability of the predicted class
        confidences = np.where(predictions, probas[:, 1], probas[:, 0])

        return predictions, confidences

    def save(self, filepath: str):
        """Save RF model weights"""
        if not self.is_trained:
            logger.warning("Saving untrained model")

        joblib.dump(self.model, filepath)
        logger.info(f"Saved Random Forest model to {filepath}")

    def load(self, filepath: str):
        """Load RF model weights"""
        if self.is_trained:
            logger.warning("Overriding trained model")

        self.model = joblib.load(filepath)
        # Predict-time parallelism must come from the RUNTIME config, not the training
        # host's pickled value (#301): joblib.load replaces the estimator wholesale, so
        # the constructor's rf.n_jobs was silently dead on every loaded model. n_jobs is
        # a predict-time execution knob on a fitted forest — reassigning it cannot touch
        # the trees. Default (-1) matches the deployed artifacts, so behavior only
        # changes when an operator asks for it.
        self.model.n_jobs = self.config.rf.n_jobs
        self.is_trained = True
        logger.info(f"Loaded Random Forest model from {filepath}")
