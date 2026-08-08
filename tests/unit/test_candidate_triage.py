"""Unit tests for aetherscan.candidate_triage (#395/#397): frequency-exclusion
partitioning semantics — report-time only, so correctness here is about tallies and
Slack surfaces, never about what a snippet scores."""

from __future__ import annotations

import pytest

from aetherscan.candidate_triage import (
    frequency_excluded,
    normalized_frequency_ranges,
    partition_candidates_by_frequency,
)


class TestNormalizedFrequencyRanges:
    def test_none_and_empty_normalize_to_no_ranges(self):
        assert normalized_frequency_ranges(None) == []
        assert normalized_frequency_ranges([]) == []

    def test_pairs_normalize_sorted_as_float_tuples(self):
        ranges = normalized_frequency_ranges([[1616, 1626.5], [1575.0, 1576.0]])
        assert ranges == [(1575.0, 1576.0), (1616.0, 1626.5)]

    @pytest.mark.parametrize(
        "bad",
        [
            [[1626.5, 1616]],  # start >= end
            [[1616, 1616]],  # zero-width
            [[-5, 10]],  # non-positive start
            [[0, 10]],  # zero start
            [[float("inf"), 2000]],  # inf parses as a float on the CLI
            [[1616, float("nan")]],  # so does nan
            [["iridium", 1626.5]],  # non-numeric
            [[1616]],  # missing end
        ],
    )
    def test_malformed_ranges_raise(self, bad):
        with pytest.raises(ValueError):
            normalized_frequency_ranges(bad)


class TestFrequencyExcluded:
    RANGES = [(1575.0, 1576.0), (1616.0, 1626.5)]

    @pytest.mark.parametrize("freq", [1575.0, 1576.0, 1575.42, 1616.0, 1626.5, 1620.0])
    def test_inside_and_boundary_excluded(self, freq):
        # Inclusive at both edges: a candidate exactly on an allocation boundary is the
        # allocation's
        assert frequency_excluded(freq, self.RANGES) is True

    @pytest.mark.parametrize("freq", [1574.999, 1576.001, 1615.9, 1626.6, 8438.0])
    def test_outside_reported(self, freq):
        assert frequency_excluded(freq, self.RANGES) is False

    def test_none_frequency_never_excluded(self):
        # No basis to filter -> reporting is the safe side
        assert frequency_excluded(None, self.RANGES) is False


class TestPartitionCandidatesByFrequency:
    def test_no_ranges_reports_everything(self):
        rows = [{"frequency_mhz": 1575.42}, {"frequency_mhz": 8438.0}]
        reported, excluded = partition_candidates_by_frequency(rows, [])
        assert reported == rows
        assert excluded == []

    def test_partition_preserves_order_within_sides(self):
        rows = [
            {"frequency_mhz": 1575.42, "id": "gps"},
            {"frequency_mhz": 8438.0, "id": "clean-1"},
            {"frequency_mhz": 1626.0, "id": "iridium"},
            {"frequency_mhz": None, "id": "no-freq"},
            {"frequency_mhz": 4091.0, "id": "clean-2"},
        ]
        ranges = [(1575.0, 1576.0), (1616.0, 1626.5)]
        reported, excluded = partition_candidates_by_frequency(rows, ranges)
        assert [r["id"] for r in reported] == ["clean-1", "no-freq", "clean-2"]
        assert [r["id"] for r in excluded] == ["gps", "iridium"]
