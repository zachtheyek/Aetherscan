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


def prepare_latent_features(latent_vectors: np.ndarray, num_observations: int = 6) -> np.ndarray:
    """
    Prepare latent vectors for Random Forest input
    Recombines the latent vectors into their original 6-observation cadence pattern
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
    # We flatten the observations so all 6 latents in a cadence are grouped together
    features = np.zeros((num_cadences, num_observations * latent_dim))

    for i in range(num_cadences):
        # Flatten & concatenate the latent vectors according to the number of observations
        features[i, :] = latent_vectors[
            i * num_observations : (i + 1) * num_observations, :
        ].ravel()

    return features


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
            random_state=self.config.rf.seed,
        )

        self.is_trained = False

    def train(self, latent_vectors: np.ndarray, binary_labels: np.ndarray):
        """
        Train the Random Forest model

        Args:
            latent_vectors: Latent vectors shape (n_cadences * num_observations, latent_dim).
                Caller must ensure row i..i+num_observations-1 corresponds to cadence i.
            binary_labels: Binary labels shape (n_cadences,) with 0=false, 1=true signal.
        """
        # Prepare features
        features = prepare_latent_features(latent_vectors, self.config.data.num_observations)

        # Sanity check: make sure length of feature & label arrays are aligned
        if features.shape[0] != binary_labels.shape[0]:
            raise ValueError(
                f"Feature/label count mismatch: {features.shape[0]} vs {binary_labels.shape[0]}"
            )

        # Shuffle data
        features, binary_labels = shuffle(features, binary_labels, random_state=self.config.rf.seed)
        logger.info(f"Prepared {features.shape[0]} training samples")

        # Start training
        logger.info("Training Random Forest classifier...")
        self.model.fit(features, binary_labels)
        self.is_trained = True

        # NOTE: come back to this later
        # importances = self.model.feature_importances_
        # logger.info(
        #     f"Feature importance stats - Mean: {np.mean(importances):.4f}, "
        #     f"Std: {np.std(importances):.4f}"
        # )
        # logger.info(f"Feature importance: \n{importances}")

    def predict_proba(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Predict binary probabilities given some input latent cadences
        """
        if not self.is_trained:
            logger.warning("Making predictions with untrained model")

        features = prepare_latent_features(latent_vectors, self.config.data.num_observations)
        return self.model.predict_proba(features)

    def predict(self, latent_vectors: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary classes given some input latent cadences
        """
        probas = self.predict_proba(latent_vectors)
        return (probas[:, 1] > threshold).astype(int)

    def predict_verbose(
        self, latent_vectors: np.ndarray, threshold: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict binary classes given some input latent cadences
        Returns 1 if probability of true signal > threshold, else 0
        Also outputs confidence score (predicted probability of output class)
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
        self.is_trained = True
        logger.info(f"Loaded Random Forest model from {filepath}")
