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

import logging
import math

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
