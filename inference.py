# TODO:
"""
Inference orchestration for Aetherscan Pipeline
...
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tensorflow as tf
from numba import jit

from models.random_forest import RandomForestModel
from preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


@jit(nopython=True, parallel=True)
def extract_overlapping_snippets(
    data: np.ndarray, snippet_size: int, overlap: float = 0.5
) -> list[tuple[int, np.ndarray]]:
    """
    Extract overlapping snippets from continuous data

    Args:
        data: Continuous observation data
        snippet_size: Size of each snippet
        overlap: Overlap fraction

    Returns:
        List of (start_index, snippet) tuples
    """
    step_size = int(snippet_size * (1 - overlap))
    snippets = []

    for start in range(0, data.shape[-1] - snippet_size + 1, step_size):
        snippet = data[..., start : start + snippet_size]
        snippets.append((start, snippet))

    return snippets


class InferencePipeline:
    """Inference pipeline for SETI signal detection"""

    def __init__(self, config, vae_encoder_path: str, rf_model_path: str):
        """
        Initialize inference pipeline

        Args:
            config: Configuration object
            vae_encoder_path: Path to saved VAE encoder
            rf_model_path: Path to saved Random Forest model
        """
        self.config = config
        self.preprocessor = DataPreprocessor(config)

        # Load models
        logger.info("Loading models...")
        self.vae_encoder = tf.keras.models.load_model(vae_encoder_path)

        self.rf_model = RandomForestModel(config)
        self.rf_model.load(rf_model_path)

        # Results storage
        self.results = []

    def process_cadence(self, cadence_files: list[str]) -> dict[str, np.ndarray]:
        """
        Process a single cadence (6 observation files)

        Args:
            cadence_files: List of 6 file paths (3 ON, 3 OFF)

        Returns:
            Dictionary of results
        """
        if len(cadence_files) != 6:
            raise ValueError(f"Expected 6 files, got {len(cadence_files)}")

        # Load observations
        observations = []
        for filepath in cadence_files:
            # This would be replaced with actual file loading logic
            # For now, assuming numpy arrays are provided
            obs = np.load(filepath) if isinstance(filepath, str) else filepath
            observations.append(obs)

        # Preprocess cadence
        cadence_data = self.preprocessor.preprocess_cadence(observations)

        # Extract overlapping snippets
        snippets = self._extract_snippets_from_cadence(cadence_data)

        # Process snippets
        results = self._process_snippets(snippets)

        return results

    def _extract_snippets_from_cadence(self, cadence: np.ndarray) -> list[dict]:
        """
        Extract overlapping snippets from cadence data

        Args:
            cadence: Preprocessed cadence data

        Returns:
            List of snippet dictionaries
        """
        snippets = []
        snippet_size = self.config.data.width_bin // self.config.data.downsample_factor
        overlap = self.config.data.overlap_factor

        # Process each observation in the cadence
        for obs_idx in range(6):
            obs_data = cadence[:, obs_idx, :, :, :]

            # Extract snippets with overlap
            obs_snippets = extract_overlapping_snippets(
                obs_data[0, :, :, 0],  # Remove batch and channel dims
                snippet_size,
                overlap,
            )

            for start_idx, snippet in obs_snippets:
                snippets.append(
                    {
                        "observation": obs_idx,
                        "start_index": start_idx,
                        "data": snippet[np.newaxis, ..., np.newaxis],  # Add batch and channel dims
                    }
                )

        return snippets

    def _process_snippets(self, snippets: list[dict]) -> dict[str, np.ndarray]:
        """
        Process snippets through VAE and Random Forest

        Args:
            snippets: List of snippet dictionaries

        Returns:
            Results dictionary
        """
        n_snippets = len(snippets)
        batch_size = self.config.inference.batch_size

        all_predictions = []
        all_probabilities = []
        all_latents = []

        # Process in batches
        for i in range(0, n_snippets, batch_size):
            batch_snippets = snippets[i : i + batch_size]

            # Stack snippet data
            batch_data = np.concatenate([s["data"] for s in batch_snippets], axis=0)

            # Get latent representations
            _, _, latents = self.vae_encoder.predict(batch_data, batch_size=batch_size)

            # Random Forest predictions
            if len(latents) >= 6:  # Need full cadence for RF
                probas = self.rf_model.predict_proba(latents)
                preds = probas[:, 1] > self.config.inference.classification_threshold

                all_predictions.extend(preds)
                all_probabilities.extend(probas[:, 1])
                all_latents.extend(latents)

        return {
            "predictions": np.array(all_predictions),
            "probabilities": np.array(all_probabilities),
            "latents": np.array(all_latents),
            "snippet_info": snippets,
        }

    def analyze_frequency_band(
        self, observations: list[np.ndarray], frequency_range: tuple[float, float]
    ) -> pd.DataFrame:
        """
        Analyze a frequency band for signals

        Args:
            observations: List of observation arrays
            frequency_range: (start_freq, end_freq) in MHz

        Returns:
            DataFrame with detection results
        """
        logger.info(
            f"Analyzing frequency band {frequency_range[0]:.2f}-{frequency_range[1]:.2f} MHz"
        )

        start_time = time.time()

        # Process cadence
        results = self.process_cadence(observations)

        # Filter detections
        detections = self._filter_detections(results)

        # Create results DataFrame
        df_results = self._create_results_dataframe(detections, frequency_range)

        elapsed_time = time.time() - start_time
        logger.info(f"Processed band in {elapsed_time:.2f} seconds")

        return df_results

    def _filter_detections(self, results: dict) -> list[dict]:
        """
        Filter detections based on cadence pattern and confidence

        Args:
            results: Raw detection results

        Returns:
            Filtered list of detections
        """
        detections = []

        predictions = results["predictions"]
        probabilities = results["probabilities"]
        snippet_info = results["snippet_info"]

        # Group by frequency location
        frequency_groups = {}
        for i, (pred, prob, info) in enumerate(
            zip(predictions, probabilities, snippet_info, strict=False)
        ):
            if pred:  # Positive detection
                freq_key = info["start_index"]
                if freq_key not in frequency_groups:
                    frequency_groups[freq_key] = []
                frequency_groups[freq_key].append(
                    {"observation": info["observation"], "probability": prob, "index": i}
                )

        # Check cadence patterns
        for freq_key, group in frequency_groups.items():
            obs_pattern = [d["observation"] for d in group]

            # Check if signal appears in ON observations (0, 2, 4) but not OFF (1, 3, 5)
            on_obs = [0, 2, 4]
            off_obs = [1, 3, 5]

            on_detections = sum(1 for obs in obs_pattern if obs in on_obs)
            off_detections = sum(1 for obs in obs_pattern if obs in off_obs)

            if on_detections >= 2 and off_detections == 0:
                # Strong ETI candidate
                avg_prob = np.mean([d["probability"] for d in group])
                detections.append(
                    {
                        "frequency_index": freq_key,
                        "confidence": avg_prob,
                        "pattern": obs_pattern,
                        "classification": "ETI_candidate",
                    }
                )
            elif on_detections > 0 and off_detections > 0:
                # Possible RFI
                avg_prob = np.mean([d["probability"] for d in group])
                detections.append(
                    {
                        "frequency_index": freq_key,
                        "confidence": avg_prob,
                        "pattern": obs_pattern,
                        "classification": "RFI_likely",
                    }
                )

        return detections

    def _create_results_dataframe(
        self, detections: list[dict], frequency_range: tuple[float, float]
    ) -> pd.DataFrame:
        """
        Create DataFrame with detection results

        Args:
            detections: List of detection dictionaries
            frequency_range: Frequency range being analyzed

        Returns:
            Results DataFrame
        """
        if not detections:
            return pd.DataFrame()

        # Calculate actual frequencies
        freq_resolution = (
            self.config.data.freq_resolution * self.config.data.downsample_factor
        )  # 2.79 Hz -> 22.35 Hz after downsample
        start_freq = frequency_range[0] * 1e6  # MHz to Hz

        df_data = []
        for det in detections:
            freq_hz = start_freq + (det["frequency_index"] * freq_resolution)

            df_data.append(
                {
                    "frequency_MHz": freq_hz / 1e6,
                    "confidence": det["confidence"],
                    "classification": det["classification"],
                    "observation_pattern": str(det["pattern"]),
                    "frequency_index": det["frequency_index"],
                }
            )

        df = pd.DataFrame(df_data)
        df = df.sort_values("confidence", ascending=False)

        return df

    def parallel_process_bands(
        self, observation_files: list[list[str]], n_workers: int = 4
    ) -> pd.DataFrame:
        """
        Process multiple frequency bands in parallel

        Args:
            observation_files: List of observation file lists (one per band)
            n_workers: Number of parallel workers

        Returns:
            Combined results DataFrame
        """
        logger.info(f"Processing {len(observation_files)} bands with {n_workers} workers")

        all_results = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []

            for band_idx, obs_files in enumerate(observation_files):
                # Calculate frequency range for this band
                band_start = 1100 + (band_idx * 50)  # Example: 50 MHz bands
                band_end = band_start + 50

                future = executor.submit(
                    self.analyze_frequency_band, obs_files, (band_start, band_end)
                )
                futures.append(future)

            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    if not result.empty:
                        all_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing band: {e}")

        # Combine results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_df = combined_df.sort_values("confidence", ascending=False)
            return combined_df
        else:
            return pd.DataFrame()

    def save_results(self, results_df: pd.DataFrame, output_path: str):
        """Save detection results"""
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(results_df)} detections to {output_path}")

        # Also save high-confidence detections separately
        high_conf = results_df[results_df["confidence"] > 0.9]
        if not high_conf.empty:
            high_conf_path = output_path.replace(".csv", "_high_confidence.csv")
            high_conf.to_csv(high_conf_path, index=False)
            logger.info(f"Saved {len(high_conf)} high-confidence detections")


def run_inference(
    config,
    observation_files: list[list[str]],
    vae_encoder_path: str,
    rf_model_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Run inference on observation data

    Args:
        config: Configuration object
        observation_files: List of observation file lists
        vae_encoder_path: Path to VAE encoder
        rf_model_path: Path to Random Forest model
        output_path: Path to save results

    Returns:
        Detection results DataFrame
    """
    # Create pipeline
    pipeline = InferencePipeline(config, vae_encoder_path, rf_model_path)

    # Process data
    results = pipeline.parallel_process_bands(observation_files)

    # Save results
    pipeline.save_results(results, output_path)

    # Log summary
    if not results.empty:
        n_eti = len(results[results["classification"] == "ETI_candidate"])
        n_rfi = len(results[results["classification"] == "RFI_likely"])
        logger.info(f"Detection summary: {n_eti} ETI candidates, {n_rfi} likely RFI")
    else:
        logger.info("No detections found")

    return results
