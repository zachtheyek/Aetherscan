"""Unit tests for aetherscan.candidate_figures (#298 I9): the TF-free per-candidate figure
renderer — single-figure output, index-ordered pool results, and per-figure failure
containment (a bad row degrades to None, never aborts the batch)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from aetherscan.candidate_figures import (
    candidate_frequency_range_mhz,
    render_candidate_figure,
    render_candidate_figures,
    stamp_frequency_range_mhz,
)


@pytest.fixture
def candidate_rows(tmp_path):
    """A tiny stamp .npy plus DB-row-shaped dicts pointing at its snippets."""
    rng = np.random.default_rng(13)
    stamps = rng.chisquare(df=4, size=(5, 6, 4, 16)).astype(np.float32)
    npy_path = tmp_path / "cadence.npy"
    np.save(npy_path, stamps)

    def _row(snippet_index, **extra):
        row = {
            "npy_path": str(npy_path),
            "snippet_index": snippet_index,
            "confidence": 0.99 - 0.01 * snippet_index,
            "frequency_mhz": 1420.0 + snippet_index,
            "target": "HIP0000",
            "latent_vector": "[0.1, -0.2, 0.3, 0.4, -0.5, 0.6]",
        }
        row.update(extra)
        return row

    return _row


class TestStampFrequencyRange:
    """#298 follow-up: cadence-snippet plots label their x-axis with the stamp's frequency
    span, computed from the metadata sidecar's header + stamp geometry."""

    METADATA = {
        "header": {"fch1": 2300.0, "foff": -1e-6},  # MHz per fine bin, descending
        "stamp_starts": [0, 1000],
        "stamp_width": 4096,
    }

    def test_exact_range_bin_order(self):
        low, high = stamp_frequency_range_mhz(self.METADATA, 1)
        assert low == 2300.0 - 1e-6 * 1000  # bin 0 of the stamp
        assert high == 2300.0 - 1e-6 * (1000 + 4095)  # last raw bin
        assert low > high  # negative foff: descending in bin order

    @pytest.mark.parametrize(
        "broken",
        [
            {},
            {"header": {"fch1": 2300.0}},  # no foff
            {"header": {"fch1": 2300.0, "foff": -1e-6}, "stamp_starts": []},  # index missing
            {"header": {"fch1": 2300.0, "foff": -1e-6}, "stamp_starts": [0]},  # no stamp_width
        ],
    )
    def test_missing_fields_return_none(self, broken):
        assert stamp_frequency_range_mhz(broken, 0) is None

    def test_candidate_range_reads_sidecar(self, tmp_path):
        import json  # noqa: PLC0415

        npy_path = tmp_path / "cad.npy"
        with open(tmp_path / "cad.json", "w") as f:
            json.dump(self.METADATA, f)
        row = {"npy_path": str(npy_path), "snippet_index": 0}
        assert candidate_frequency_range_mhz(row) == (2300.0, 2300.0 - 1e-6 * 4095)

    def test_candidate_range_missing_sidecar_is_none(self, tmp_path):
        row = {"npy_path": str(tmp_path / "nope.npy"), "snippet_index": 0}
        assert candidate_frequency_range_mhz(row) is None


class TestRenderCandidateFigure:
    def test_saves_png(self, tmp_path, candidate_rows):
        path = render_candidate_figure(candidate_rows(0), 0, "test_v1", str(tmp_path))
        assert path == os.path.join(str(tmp_path), "candidate_0_test_v1.png")
        assert os.path.getsize(path) > 0

    def test_no_latent_vector_still_renders(self, tmp_path, candidate_rows):
        path = render_candidate_figure(
            candidate_rows(1, latent_vector=None), 1, "test_v1", str(tmp_path)
        )
        assert os.path.getsize(path) > 0


class TestRenderCandidateFigures:
    def test_serial_path_preserves_index_order(self, tmp_path, candidate_rows):
        rows = [candidate_rows(i) for i in range(3)]  # below the pool floor -> in-process
        results = render_candidate_figures(rows, "test_v1", str(tmp_path))
        assert [index for index, _ in results] == [0, 1, 2]
        assert all(path and os.path.exists(path) for _, path in results)

    def test_failed_row_returns_none_without_aborting(self, tmp_path, candidate_rows):
        rows = [candidate_rows(0), candidate_rows(1, npy_path="/nonexistent/stamps.npy")]
        results = render_candidate_figures(rows, "test_v1", str(tmp_path))
        assert results[0][1] is not None
        assert results[1][1] is None

    def test_pool_path_matches_serial_output(self, tmp_path, candidate_rows):
        rows = [candidate_rows(i) for i in range(5)]  # above the pool floor
        results = render_candidate_figures(rows, "test_v1", str(tmp_path), n_workers=2)
        assert [index for index, _ in results] == [0, 1, 2, 3, 4]
        for index, path in results:
            assert path == os.path.join(str(tmp_path), f"candidate_{index}_test_v1.png")
            assert os.path.getsize(path) > 0

    def test_empty_rows(self, tmp_path):
        assert render_candidate_figures([], "test_v1", str(tmp_path)) == []
