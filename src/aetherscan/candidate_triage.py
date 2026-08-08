"""
Report-time candidate triage (#395, #397): frequency-exclusion partitioning for the
run tallies and Slack surfaces, and OOD triage scores for review ordering.

Everything here is strictly REPORT-time: no function in this module changes what a
snippet scores, which rows land in ``inference_results``, or which figures are rendered
and saved to disk — candidates in an excluded range are still detected, still persisted,
and still plotted; only the human-facing tallies and Slack uploads are filtered, so the
science record stays complete while the review surface stays clean. Accordingly,
``inference.report_exclude_frequency_ranges`` sits in ``run_state.py``'s fingerprint
denylist: changing it must never stale a resume row or rename a stamp-cache directory.

TF-free by design (mirrors ``candidate_figures``): imported by ``inference_viz`` (viz
surfaces) and ``main`` (summary tallies), and importable by standalone utils without
dragging in the model stack.
"""

from __future__ import annotations

import json
import logging
import math
import os

import numpy as np

logger = logging.getLogger(__name__)


def normalized_frequency_ranges(ranges) -> list[tuple[float, float]]:
    """
    Validate and normalize ``[[start_mhz, end_mhz], ...]`` into sorted float tuples.

    Raises ValueError on anything malformed — non-numeric entries, non-finite values
    (inf/NaN parse as valid floats on the CLI), non-positive frequencies, or
    start >= end. CLI validation applies the same rules up front; this is the defensive
    re-check for programmatically-set configs, and callers on exception-guarded paths
    catch the ValueError and fall back to no filtering.
    """
    if not ranges:
        return []
    normalized = []
    for pair in ranges:
        try:
            start, end = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError(
                f"malformed frequency range {pair!r}: expected [start_mhz, end_mhz]"
            ) from None
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"frequency range {pair!r} must be finite")
        if start <= 0 or end <= 0:
            raise ValueError(f"frequency range {pair!r} must be positive MHz values")
        if start >= end:
            raise ValueError(f"frequency range {pair!r} must have start < end")
        normalized.append((start, end))
    return sorted(normalized)


def frequency_excluded(frequency_mhz, ranges: list[tuple[float, float]]) -> bool:
    """True when the frequency falls inside any [start, end] MHz range (inclusive at both
    edges — a candidate exactly on an allocation boundary is the allocation's). A None
    frequency is never excluded: with no basis to filter, reporting is the safe side."""
    if frequency_mhz is None:
        return False
    frequency = float(frequency_mhz)
    return any(start <= frequency <= end for start, end in ranges)


def partition_candidates_by_frequency(
    rows: list[dict], ranges: list[tuple[float, float]]
) -> tuple[list[dict], list[dict]]:
    """Split candidate row dicts (``inference_results`` shape, keyed on 'frequency_mhz')
    into (reported, excluded), preserving order within each side."""
    if not ranges:
        return list(rows), []
    reported: list[dict] = []
    excluded: list[dict] = []
    for row in rows:
        side = excluded if frequency_excluded(row.get("frequency_mhz"), ranges) else reported
        side.append(row)
    return reported, excluded


def report_exclusion_ranges(config) -> list[tuple[float, float]]:
    """The configured exclusion ranges, normalized — or [] on malformed programmatic
    values (logged; report-time filtering must never fail a science run)."""
    try:
        return normalized_frequency_ranges(config.inference.report_exclude_frequency_ranges)
    except ValueError as e:
        logger.error(f"Ignoring malformed inference.report_exclude_frequency_ranges: {e}")
        return []


# ---------------------------------------------------------------------------
# OOD triage scores (#397)
#
# Both scores below are TRIAGE surfaces — review ordering and a CSV column — and must
# never gate candidacy: a genuine technosignature is itself out-of-distribution with
# respect to synthetic training data, so "unusual" ranks candidates for human eyes,
# it never drops them.
# ---------------------------------------------------------------------------


def mahalanobis_ood(candidates: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    (distances, percentiles) of each candidate row against the reference distribution:
    Mahalanobis distance under the reference's mean/covariance (ridge-stabilized pinv, so
    a degenerate or small reference never raises), and the percentile of each candidate's
    distance among the reference rows' own distances (100 = farther than every reference
    row, i.e. maximally unusual).
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    mean = reference.mean(axis=0)
    centered = reference - mean
    cov = centered.T @ centered / max(len(reference) - 1, 1)
    # Ridge scaled to the average variance: keeps pinv stable when the reference has
    # collapsed/duplicated dimensions without materially moving healthy distances
    cov[np.diag_indices_from(cov)] += 1e-6 * max(float(np.trace(cov)) / len(cov), 1e-12)
    inverse = np.linalg.pinv(cov)

    def _distances(rows: np.ndarray) -> np.ndarray:
        delta = rows - mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, inverse, delta), 0.0))

    reference_distances = np.sort(_distances(reference))
    candidate_distances = _distances(candidates)
    percentiles = (
        np.searchsorted(reference_distances, candidate_distances, side="right")
        / len(reference_distances)
        * 100.0
    )
    return candidate_distances, percentiles


def candidate_latent_matrix(rows: list[dict]) -> tuple[np.ndarray, list[int]]:
    """Parse the candidate rows' stored latent_vector JSON payloads into a (n, d) float
    matrix, returning it with the indices of the rows that had a parseable vector."""
    vectors: list[list[float]] = []
    row_indices: list[int] = []
    width: int | None = None
    for i, row in enumerate(rows):
        payload = row.get("latent_vector")
        if payload is None:
            continue
        try:
            vector = json.loads(payload) if isinstance(payload, str) else list(payload)
        except (TypeError, ValueError):
            continue
        if width is None:
            width = len(vector)
        if len(vector) != width or width == 0:
            continue
        vectors.append(vector)
        row_indices.append(i)
    if not vectors:
        return np.empty((0, 0), dtype=np.float64), []
    return np.asarray(vectors, dtype=np.float64), row_indices


def _ood_map(rows: list[dict], reference: np.ndarray) -> dict[tuple[str, int], tuple[float, float]]:
    """{(npy_path, snippet_index): (distance, percentile)} for rows with usable latents
    matching the reference dimensionality."""
    latents, row_indices = candidate_latent_matrix(rows)
    if latents.size == 0 or latents.shape[1] != reference.shape[1]:
        if latents.size and latents.shape[1] != reference.shape[1]:
            logger.info(
                f"OOD triage: candidate latent width {latents.shape[1]} != reference "
                f"width {reference.shape[1]}; skipping"
            )
        return {}
    distances, percentiles = mahalanobis_ood(latents, reference)
    return {
        (rows[i]["npy_path"], int(rows[i]["snippet_index"])): (
            float(distances[j]),
            float(percentiles[j]),
        )
        for j, i in enumerate(row_indices)
    }


def survey_ood_scores(
    rows: list[dict], cloud_path: str
) -> dict[tuple[str, int], tuple[float, float]]:
    """
    Candidate-vs-survey OOD (#397): distance of each candidate's latent to THIS run's
    reference cloud — the seeded uniform reservoir of pass-1 rejects, i.e. what the
    surveyed sky ordinarily looks like in latent space. High percentile = unlike the
    survey background. Requires the cloud NPZ to carry latent_mean rows (clouds written
    before #397 lack them — skipped with a log). Best-effort: {} on any failure.
    """
    try:
        if not os.path.exists(cloud_path):
            logger.info(f"OOD triage: no reference cloud at {cloud_path}; skipping survey OOD")
            return {}
        with np.load(cloud_path) as npz:
            if "latent_mean" not in npz.files:
                logger.info(
                    "OOD triage: reference cloud predates #397 (no latent_mean); "
                    "skipping survey OOD"
                )
                return {}
            reference = np.asarray(npz["latent_mean"], dtype=np.float64)
        if len(reference) < 2:
            return {}
        return _ood_map(rows, reference)
    except Exception as e:
        logger.error(f"Survey OOD scoring failed ({e}); skipping")
        return {}


def training_ood_scores(
    rows: list[dict], model_path: str, training_config_path: str | None
) -> dict[tuple[str, int], tuple[float, float]]:
    """
    Candidate-vs-training OOD (#397): distance of each candidate's latent to the training
    run's TRUE-class feature cloud from rf_eval_artifacts_{train_tag}.joblib — how unlike
    the synthetic signal manifold a candidate is. Only computed when the deployed variant
    is 'z_mean' (the one variant whose train_features live in the same space as the
    stored candidate z_mean latents); anything else, or a missing cluster-local artifact,
    skips with a log. Best-effort: {} on any failure.
    """
    import joblib  # noqa: PLC0415  # deferred: only this scorer needs it

    try:
        if not training_config_path or not os.path.exists(training_config_path):
            logger.info("OOD triage: no training config available; skipping training OOD")
            return {}
        with open(training_config_path) as f:
            train_tag = (json.load(f).get("checkpoint") or {}).get("save_tag")
        if not train_tag:
            logger.info("OOD triage: training config lacks checkpoint.save_tag; skipping")
            return {}
        artifact_path = os.path.join(model_path, f"rf_eval_artifacts_{train_tag}.joblib")
        if not os.path.exists(artifact_path):
            logger.info(
                f"OOD triage: no rf_eval_artifacts at {artifact_path} (cluster-local "
                f"artifact); skipping training OOD"
            )
            return {}
        artifacts = joblib.load(artifact_path)
        if artifacts.get("latent_variant") != "z_mean":
            logger.info(
                f"OOD triage: deployed variant {artifacts.get('latent_variant')!r} features "
                f"are not directly comparable to stored z_mean latents; skipping training OOD"
            )
            return {}
        labels = np.asarray(artifacts["train_binary_labels"])
        reference = np.asarray(artifacts["train_features"])[labels == 1]
        if len(reference) < 2:
            return {}
        return _ood_map(rows, reference)
    except Exception as e:
        logger.error(f"Training OOD scoring failed ({e}); skipping")
        return {}


def triage_sort_rows(
    rows: list[dict], survey_ood: dict[tuple[str, int], tuple[float, float]]
) -> list[dict]:
    """
    Review ordering (#397): confidence descending (unchanged science-first primary),
    tie-broken by survey-OOD distance descending — within a saturated P=1.000 tie the
    candidate LEAST like the survey background reviews first — then MC spread ascending
    (stable scores first; missing spread last). Pure function so the ordering contract
    is unit-testable apart from the gallery.
    """

    def _key(row: dict):
        distance, _ = survey_ood.get(
            (row.get("npy_path"), int(row.get("snippet_index") or 0)), (0.0, 0.0)
        )
        mc_std = row.get("mc_std")
        return (
            -(row.get("confidence") or 0.0),
            -distance,
            mc_std if mc_std is not None else float("inf"),
        )

    return sorted(rows, key=_key)
