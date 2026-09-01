"""Unit tests for utils/probe_candidate_location.py — the cascade location probe.

Covers the pure, TF-free helpers where index arithmetic could silently drift from
production: frequency->bin resolution under both foff signs, stamp-bound clamping,
the ED window-start-in-stamp predicate (which must match preprocessing's hit
indexing), catalog-key normalization, CLI validation, and the CSV row contract.
Stdlib + numpy only — every heavy import in the tool is deferred past parsing."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# utils/ is not a package — load the tool straight from its file (same pattern as
# tests/unit/test_perband_report.py). sys.modules registration precedes exec_module for
# PEP 563 dataclass annotation resolution.
_TOOL_PATH = Path(__file__).resolve().parents[2] / "utils" / "probe_candidate_location.py"
_spec = importlib.util.spec_from_file_location("probe_candidate_location", _TOOL_PATH)
probe = importlib.util.module_from_spec(_spec)
sys.modules["probe_candidate_location"] = probe
_spec.loader.exec_module(probe)


def _axis(fch1: float, foff: float, nchans: int):
    return probe.FrequencyAxis(fch1=fch1, foff=foff, nchans=nchans, shape=(16, 1, nchans))


class TestFrequencyBin:
    def test_negative_foff_round_trip(self):
        axis = _axis(fch1=4126.0, foff=-1e-3, nchans=1000)
        absolute_bin, resolved, offset_hz = probe._frequency_bin(axis, 4125.5)
        assert absolute_bin == 500
        assert resolved == pytest.approx(4125.5)
        assert offset_hz == pytest.approx(0.0, abs=1e-6)

    def test_positive_foff_round_trip(self):
        axis = _axis(fch1=1100.0, foff=1e-3, nchans=1000)
        absolute_bin, resolved, _ = probe._frequency_bin(axis, 1100.25)
        assert absolute_bin == 250
        assert resolved == pytest.approx(1100.25)

    def test_rounds_to_nearest_bin_center(self):
        axis = _axis(fch1=1000.0, foff=1e-3, nchans=100)
        # 1000.0014 sits 1.4 bins up -> nearest center is bin 1, i.e. 400 Hz below request
        absolute_bin, resolved, offset_hz = probe._frequency_bin(axis, 1000.0014)
        assert absolute_bin == 1
        assert resolved == pytest.approx(1000.001)
        assert offset_hz == pytest.approx(-400.0, abs=1e-3)

    def test_out_of_range_raises_with_bin_center_span(self):
        axis = _axis(fch1=4126.0, foff=-1e-3, nchans=100)
        with pytest.raises(ValueError, match="outside the h5 bin-center range"):
            probe._frequency_bin(axis, 4200.0)


class TestStampBounds:
    def test_centered_when_room(self):
        start, end, clamped = probe._stamp_bounds(5000, 4096, 100_000)
        assert (start, end) == (5000 - 2048, 5000 - 2048 + 4096)
        assert not clamped

    def test_clamps_at_low_edge(self):
        start, end, clamped = probe._stamp_bounds(10, 4096, 100_000)
        assert (start, end) == (0, 4096)
        assert clamped

    def test_clamps_at_high_edge(self):
        start, end, clamped = probe._stamp_bounds(99_999, 4096, 100_000)
        assert (start, end) == (100_000 - 4096, 100_000)
        assert clamped

    def test_nchans_narrower_than_stamp_raises(self):
        with pytest.raises(ValueError, match="smaller than configured stamp_width"):
            probe._stamp_bounds(100, 4096, 1000)


class TestMaxK2InStamp:
    def test_window_start_predicate_matches_hit_indexing(self):
        # Coarse channel 2, width 1000, step 100: window j starts at 2000 + 100*j —
        # the same arithmetic preprocessing uses to place hits. Stamp [2150, 2350)
        # covers window starts 2200 and 2300 only.
        k2 = np.array([1.0, 2.0, 9.0, 4.0, 5.0], dtype=np.float64)
        value = probe._max_k2_in_stamp(k2, 2, 1000, 100, 2150, 2350)
        assert value == 9.0

    def test_stamp_boundary_is_half_open(self):
        k2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        # Stamp end exactly at a window start excludes it.
        value = probe._max_k2_in_stamp(k2, 0, 1000, 100, 0, 100)
        assert value == 1.0

    def test_no_finite_windows_returns_nan(self):
        k2 = np.array([np.nan, np.nan], dtype=np.float64)
        assert math.isnan(probe._max_k2_in_stamp(k2, 0, 1000, 100, 0, 1000))


class TestValidateArgs:
    def _parse(self, argv: list[str]):
        parser = probe.build_parser()
        args = parser.parse_args(argv)
        probe._validate_args(parser, args)
        return args

    _BASE = [
        "--frequency-mhz",
        "1400.0",
        "--encoder-path",
        "e.keras",
        "--rf-path",
        "rf.joblib",
        "--config-path",
        "c.json",
    ]

    def test_catalog_requires_target_and_band(self):
        with pytest.raises(SystemExit):
            self._parse(["--catalog", "cat.csv", *self._BASE])

    def test_h5_files_reject_catalog_filters(self):
        with pytest.raises(SystemExit):
            self._parse(["--h5-files", *(["x.h5"] * 6), "--target", "T", *self._BASE])

    def test_negative_seed_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--h5-files", *(["x.h5"] * 6), "--seed", "-1", *self._BASE])

    def test_valid_h5_invocation_parses(self):
        args = self._parse(
            ["--h5-files", *(["x.h5"] * 6), "--frequency-mhz", "1500.5", *self._BASE]
        )
        # action="extend" accumulates repeated flags in argv order.
        assert args.frequency_mhz == [1500.5, 1400.0]
        assert args.cadence_seed_key == 0


class TestCatalogKeyMap:
    class _Group:
        def __init__(self, key):
            self.key = key

    def test_normalizes_column_names(self):
        key_map = probe._catalog_key_map(
            self._Group(("NGC1172", "AGBT21B_999_01", "C")),
            ["Target", " Session", "BAND"],
        )
        assert key_map == {
            "target": "NGC1172",
            "session": "AGBT21B_999_01",
            "band": "C",
        }

    def test_column_key_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            probe._catalog_key_map(self._Group(("a", "b")), ["Target"])


class TestCsvRowContract:
    def test_error_row_serializes_without_scores(self):
        result = probe.ProbeResult(
            requested_frequency_mhz=7499.0,
            resolved_frequency_mhz=float("nan"),
            frequency_offset_hz=float("nan"),
            absolute_bin=-1,
            coarse_channel=-1,
            bin_in_coarse=-1,
            stamp_start_bin=-1,
            stamp_end_bin=-1,
            stamp_clamped=False,
            on_max_k2=[float("nan")] * 3,
            ed_max_k2=float("nan"),
            ed_would_propose=False,
            normalized_stamp=None,
            status="error",
            error="boom",
        )

        row = probe._csv_row(result, self._context())
        assert row["status"] == "error"
        assert row["error"] == "boom"
        assert row["requested_frequency_mhz"] == 7499.0
        assert math.isnan(row["mc_mean"])
        # Sentinel rows (no bins computed) blank every boolean and the scoring mode —
        # a default False must never be aggregatable as a real verdict.
        for key in ("stamp_clamped", "ed_would_propose", "screen_pass", "mc_pass"):
            assert row[key] == ""
        assert row["mc_scoring_mode"] == ""

    def test_partial_row_keeps_computed_ed_fields(self):
        result = probe.ProbeResult(
            requested_frequency_mhz=1400.0,
            resolved_frequency_mhz=1400.0,
            frequency_offset_hz=0.0,
            absolute_bin=1234,
            coarse_channel=1,
            bin_in_coarse=234,
            stamp_start_bin=0,
            stamp_end_bin=4096,
            stamp_clamped=True,
            on_max_k2=[3000.0, 2500.0, 2900.0],
            ed_max_k2=3000.0,
            ed_would_propose=True,
            normalized_stamp=None,
            status="error",
            error="stamp failed the production validity filter",
        )
        row = probe._csv_row(result, self._context())
        # Stamp-stage failures keep the diagnostics that DID run...
        assert row["ed_would_propose"] is True
        assert row["stamp_clamped"] is True
        assert row["ed_max_k2"] == 3000.0
        # ...and blank only the gates that never ran.
        assert row["screen_pass"] == ""
        assert row["mc_pass"] == ""

    def test_ok_row_keeps_real_booleans(self):
        result = probe.ProbeResult(
            requested_frequency_mhz=1400.0,
            resolved_frequency_mhz=1400.0,
            frequency_offset_hz=0.0,
            absolute_bin=1234,
            coarse_channel=1,
            bin_in_coarse=234,
            stamp_start_bin=0,
            stamp_end_bin=4096,
            stamp_clamped=False,
            on_max_k2=[3000.0, 2500.0, 2900.0],
            ed_max_k2=3000.0,
            ed_would_propose=True,
            normalized_stamp=None,
            screen_pass=True,
            mc_pass=False,
        )
        row = probe._csv_row(result, self._context())
        assert row["ed_would_propose"] is True
        assert row["screen_pass"] is True
        assert row["mc_pass"] is False
        assert row["mc_scoring_mode"] == "production pass-2"

    @staticmethod
    def _context():
        return SimpleNamespace(
            config=SimpleNamespace(
                inference=SimpleNamespace(
                    stat_threshold=2048.0,
                    screening_threshold=0.5,
                    classification_threshold=0.99,
                    mc_draws=32,
                ),
                rf=SimpleNamespace(latent_variant="z_mean"),
            ),
            bandpass_method="pfb",
        )
