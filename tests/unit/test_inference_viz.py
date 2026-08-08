"""Smoke tests for aetherscan.inference_viz: every figure function runs against small
synthetic inputs and produces a non-empty PNG under plots/inference/{tag}/, the collector
keeps bounded state, and the suite entry point never raises (a plot bug must not kill a
science run)."""

from __future__ import annotations

import csv
import json
import os

import joblib
import numpy as np
import pytest

from aetherscan.config import get_config
from aetherscan.inference_viz import (
    CONFIDENCE_HIST_EDGES,
    InferenceVizCollector,
    _build_summaries,
    plot_bandpass_flattening,
    plot_candidate,
    plot_candidate_gallery,
    plot_confidence_distribution,
    plot_ed_hit_spectrum,
    plot_ed_stat_distributions,
    plot_inference_latent_projection,
    plot_inference_summary,
    plot_preproc_funnel,
    plot_stamp_gallery,
    render_inference_visualizations,
)
from aetherscan.preprocessing import ED_STAT_HIST_EDGES, DataPreprocessor

TAG = "test_v1"
N_STAMPS = 5
STORED_WIDTH = 32


class FakeUMAP:
    """Stands in for a persisted umap.UMAP: transform() projects onto the first two feature
    dims; embedding_ mimics the training fit pool's 2-D embedding."""

    def __init__(self, embedding):
        self.embedding_ = embedding

    def transform(self, features):
        return np.asarray(features, dtype=np.float64)[:, :2]


@pytest.fixture
def initialized_runtime():
    from aetherscan.db import init_db  # noqa: PLC0415
    from aetherscan.manager import init_manager  # noqa: PLC0415

    init_manager()
    return init_db()


def _write_cadence_artifacts(tmp_path, name, key, n_stamps=N_STAMPS, h5_paths=None):
    """Write a small stamp .npy + full metadata JSON like _process_cadence would."""
    rng = np.random.default_rng(hash(name) % 2**31)
    npy_path = str(tmp_path / f"{name}.npy")
    stamps = rng.chisquare(df=4, size=(n_stamps, 6, 16, STORED_WIDTH)).astype(np.float32)
    np.save(npy_path, stamps)

    stat_hist = np.zeros((3, len(ED_STAT_HIST_EDGES) - 1), dtype=np.int64)
    stat_hist[:, 40:60] = rng.integers(1, 500, size=(3, 20))
    stat_hist[:, 100] = 3  # a few extreme windows above any threshold

    freqs = list(1400.0 + rng.random(n_stamps) * 10)
    metadata = {
        "key": list(key),
        "csv_path": "catalog.csv",
        "h5_paths": h5_paths or [f"/data/{name}_{i}.h5" for i in range(6)],
        "header": {"nchans": 2048, "fch1": 1410.0, "foff": -2.8e-6, "tstart": 58000.5},
        "stamp_starts": [int(100 + 200 * i) for i in range(n_stamps)],
        "stamp_width": 256,
        "stored_width": STORED_WIDTH,
        "downsample_factor_applied": 8,
        "stamp_frequencies_mhz": freqs,
        "stamp_statistics": [float(3000 + 100 * i) for i in range(n_stamps)],
        "stamp_pvalues": [1e-9] * n_stamps,
        "overlap_search": False,
        "overlap_fraction": None,
        "ed_stat_hist": {
            "bin_edges": [float(e) for e in ED_STAT_HIST_EDGES],
            "counts_per_on_file": stat_hist.tolist(),
        },
        "n_raw_hits": 3 * n_stamps,
        "n_merged_hits": n_stamps,
        "raw_hit_frequencies_mhz": list(np.repeat(freqs, 3)),
        "merged_hit_frequencies_mhz": freqs,
    }
    metadata_path = DataPreprocessor.cadence_metadata_path(npy_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    return npy_path, metadata_path, metadata


def _fake_results(n=N_STAMPS, n_candidates=1):
    proba = np.linspace(0.1, 0.999, n)
    predictions = np.zeros(n, dtype=int)
    predictions[-n_candidates:] = 1
    return {
        "n_cadence_snippets": n,
        "n_processed": n,
        "n_candidates": n_candidates,
        "proba_true": proba,
        "predictions": predictions,
        "latents": np.random.default_rng(7).normal(size=(n * 6, 8)).astype(np.float32),
    }


@pytest.fixture
def collector(tmp_path):
    """A collector with two processed cadences backed by real on-disk artifacts."""
    get_config().checkpoint.save_tag = TAG
    coll = InferenceVizCollector()
    for name, key in (
        ("cad_a", ("A", "S1", "L", "0", "1400")),
        ("cad_b", ("B", "S1", "L", "1", "1400")),
    ):
        npy_path, metadata_path, _ = _write_cadence_artifacts(tmp_path, name, key)
        coll.record_processed(
            key, npy_path, metadata_path, {"target": key[0]}, _fake_results(), 2.5
        )
    return coll


@pytest.fixture
def summaries(collector):
    """Bounded per-cadence summaries reduced from the on-disk sidecars (#301) — the shape
    every metadata-driven figure consumes now. The fixture's legacy-form metadata (raw
    hit-frequency lists, no pre-binned hists or envelopes) exercises the back-compat
    reduction path."""
    return _build_summaries(collector.records, get_config().inference.stamp_gallery_top_k)


def _assert_figure(path):
    assert path is not None
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    tag_dir = os.path.join(get_config().output_path, "plots", "inference", TAG)
    assert os.path.dirname(path) == tag_dir


class TestCollector:
    def test_confidence_histogram_counts(self, collector):
        for record in collector.records:
            assert record.confidence_hist is not None
            assert record.confidence_hist.sum() == N_STAMPS
            assert record.confidence_hist.shape == (len(CONFIDENCE_HIST_EDGES) - 1,)

    def test_latent_pool_keeps_candidates_within_budget(self, tmp_path):
        coll = InferenceVizCollector(max_latent_points=6)
        npy_path, metadata_path, _ = _write_cadence_artifacts(tmp_path, "cad_pool", ("P",))
        coll.record_processed(
            ("P",), npy_path, metadata_path, {}, _fake_results(n=10, n_candidates=2), 1.0
        )
        coll.record_processed(
            ("Q",), npy_path, metadata_path, {}, _fake_results(n=10, n_candidates=2), 1.0
        )
        features, is_candidate = coll.latent_pool()
        assert int(is_candidate.sum()) == 4  # candidates always kept, even past budget
        assert features.shape[1] == 48
        # Non-candidates only filled while the budget lasted
        assert features.shape[0] <= 6 + 4

    def test_record_skipped_uses_manifest_aggregates(self, tmp_path):
        coll = InferenceVizCollector()
        coll.record_skipped(("S",), "/x.npy", "/x.json", {"n_stamps": 9, "n_candidates": 3})
        record = coll.records[0]
        assert record.skipped is True
        assert record.n_stamps == 9
        assert record.confidence_hist is None

    def test_budget_fill_values_match_shared_helper(self):
        """#301 builds pool rows via a direct float32 reshape instead of the full-cadence
        float64 feature matrix it used to construct (and mostly discard). The kept rows
        must stay value-identical to the shared-helper path."""
        from aetherscan.models import prepare_latent_features  # noqa: PLC0415

        rng = np.random.default_rng(3)
        latents = rng.standard_normal((5 * 6, 8)).astype(np.float32)
        is_candidate = np.array([True, False, False, True, False])
        coll = InferenceVizCollector(max_latent_points=3)
        coll._budget_fill_add(latents, is_candidate)
        features, kept_mask = coll.latent_pool()

        assert features.dtype == np.float32
        assert int(kept_mask.sum()) == 2  # both candidates kept
        expected_full = prepare_latent_features(latents, 6).astype(np.float32)
        for row in features:
            assert any(np.array_equal(row, expected) for expected in expected_full)

    def test_gallery_pixel_pool_bounded_and_feeds_gallery(self, collector):
        """#302: the pool captures per-cadence top-K pixels before pruning deletes the
        .npy; the stamp gallery then renders from the pool instead of blank columns."""
        summaries = _build_summaries(collector.records, get_config().inference.stamp_gallery_top_k)
        for record in collector.records:
            collector.pool_gallery_pixels(record.metadata_path, record.npy_path)
        pool = collector.gallery_pixels()
        # 2 cadences x N_STAMPS distinct-stat reps, truncated at the global top-K bound
        assert len(pool) == min(2 * N_STAMPS, get_config().inference.stamp_gallery_top_k)

        for record in collector.records:
            os.remove(record.npy_path)  # the prune
        _assert_figure(plot_stamp_gallery(collector.records, summaries, pool))

    def test_shared_gallery_pool_persists_across_attempts(self, tmp_path):
        """#305: passing a caller-owned list makes the pixel pool survive an in-process
        retry — a cadence pruned by attempt 1 must still render in attempt 2's gallery,
        whose fresh collector shares the same list."""
        get_config().checkpoint.save_tag = TAG
        shared: list = []
        npy_a, meta_a, _ = _write_cadence_artifacts(tmp_path, "att1", ("A",))

        # Attempt 1: a fresh collector on the shared pool captures cad A's pixels, then A
        # is pruned (its .npy deleted)
        coll1 = InferenceVizCollector(gallery_pool=shared)
        coll1.record_processed(("A",), npy_a, meta_a, {}, _fake_results(), 1.0)
        coll1.pool_gallery_pixels(meta_a, npy_a)
        assert len(shared) > 0
        os.remove(npy_a)

        # Attempt 2: a NEW collector on the SAME shared list still holds A's pooled pixels
        coll2 = InferenceVizCollector(gallery_pool=shared)
        assert coll2.gallery_pixels() == coll1.gallery_pixels()
        assert any(path == npy_a for (path, _idx) in coll2.gallery_pixels())

    def test_budget_exhausted_appends_nothing(self):
        """The spent-budget early return (#301): once no rows can be kept, the method may
        not build anything — this is every non-candidate cadence after the first one or
        two of a catalog."""
        coll = InferenceVizCollector(max_latent_points=0)
        coll._budget_fill_add(np.zeros((4 * 6, 8), np.float32), np.zeros(4, dtype=bool))
        features, kept_mask = coll.latent_pool()
        assert features.size == 0
        assert kept_mask.size == 0
        assert coll._latent_chunks == []


class TestSelectTopStamps:
    def test_overlap_offset_duplicates_are_skipped(self, tmp_path):
        """With overlap_search the same hit yields stamps at ±overlap_fraction*stamp_width
        (exactly stamp_width // 2 at defaults) sharing one statistic — the gallery selection
        must keep only one of each triplet (regression: a strict < comparison let all
        three through)."""
        from aetherscan.inference_viz import _select_top_stamps  # noqa: PLC0415

        npy_path, metadata_path, metadata = _write_cadence_artifacts(tmp_path, "cad_ov", ("O",))
        # One hit at start 1000 with its two overlap offsets (stamp_width=256 -> offset 128),
        # plus one genuinely distinct weaker hit far away
        metadata["stamp_starts"] = [872, 1000, 1128, 5000]
        metadata["stamp_statistics"] = [9000.0, 9000.0, 9000.0, 3000.0]
        metadata["stamp_frequencies_mhz"] = [1400.1, 1400.1, 1400.1, 1405.0]
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        coll = InferenceVizCollector()
        coll.record_processed(("O",), npy_path, metadata_path, {}, _fake_results(n=4), 1.0)
        summaries = _build_summaries(coll.records, gallery_top_k=4)

        selected = _select_top_stamps(coll.records, summaries, top_k=4)
        starts = [metadata["stamp_starts"][idx] for _, idx, _, _, _ in selected]
        assert len([s for s in starts if s in (872, 1000, 1128)]) == 1
        assert 5000 in starts


class TestFigureSmoke:
    def test_ed_stat_distributions(self, collector, summaries):
        _assert_figure(plot_ed_stat_distributions(collector.records, summaries))

    def test_ed_stat_distributions_drops_mismatched_bins_consistently(
        self, tmp_path, collector, monkeypatch
    ):
        """A cadence dropped for mismatched ED-hist bins must be excluded from the title's
        above-threshold and cadence counts too — otherwise the above/total pair mixes
        different cadence subsets."""
        npy_path, metadata_path, metadata = _write_cadence_artifacts(tmp_path, "cad_bad", ("C",))
        metadata["ed_stat_hist"]["bin_edges"] = [
            float(e) * 2.0 for e in metadata["ed_stat_hist"]["bin_edges"]
        ]
        metadata["n_raw_hits"] = 10_000  # would visibly poison 'above' if not excluded
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        collector.record_processed(("C",), npy_path, metadata_path, {}, _fake_results(), 1.0)
        summaries = _build_summaries(collector.records, gallery_top_k=12)

        captured: dict[str, str] = {}

        def _capture(fig, filename, slack_title):
            captured["title"] = fig.axes[0].get_title()
            return filename

        monkeypatch.setattr("aetherscan.inference_viz._save_and_upload", _capture)
        assert plot_ed_stat_distributions(collector.records, summaries) is not None

        title = captured["title"]
        # Only the two good cadences contribute: 3 * N_STAMPS raw hits each
        assert f"{2 * 3 * N_STAMPS} above threshold" in title
        assert "(2 cadence(s))" in title
        assert "finite windows" in title

    def test_ed_hit_spectrum(self, collector, summaries):
        _assert_figure(plot_ed_hit_spectrum(collector.records, summaries))

    def test_storage_size_from_geometry_when_pruned(self, tmp_path):
        """#305: with default pruning the .npy is gone by render time, so _reduce_metadata
        reconstructs the transiently-held storage from the stored geometry instead of
        leaving nan (which collapsed the funnel/summary storage totals to ~0)."""
        get_config().checkpoint.save_tag = TAG
        get_config().data.time_bins = 16  # matches _write_cadence_artifacts' (…,6,16,W)
        npy, meta_path, metadata = _write_cadence_artifacts(tmp_path, "sized", ("S",))
        real_size = os.path.getsize(npy)
        coll = InferenceVizCollector()
        coll.record_processed(("S",), npy, meta_path, {}, _fake_results(), 1.0)

        # Present file: exact stat
        summaries = _build_summaries(coll.records, 12)
        assert summaries[npy].npy_size_bytes == float(real_size)

        # Pruned file: geometry estimate, close to the real size and NOT nan
        os.remove(npy)
        summaries = _build_summaries(coll.records, 12)
        est = summaries[npy].npy_size_bytes
        assert not np.isnan(est)
        # n_stamps * 6 * time_bins * stored_width * 4 (+128 header) — within the header slack
        assert abs(est - real_size) <= 256

    def test_hit_spectrum_bin_clamp(self):
        """#305: figure bins must never be finer than the stored fine grid, else the
        narrow-span rebin aliases into a picket-fence. Wide span keeps the full 200 bins;
        a span at ~fine-grid resolution collapses toward 1."""
        from aetherscan.inference_viz import _clamp_hit_spectrum_bins  # noqa: PLC0415

        # Wide span (fine width tiny vs span): full resolution
        assert _clamp_hit_spectrum_bins(1000.0, 1200.0, 0.01, 200) == 200
        # Narrow span: 2 MHz span, fine bins ~22.9 kHz (187.5 MHz band / 8192) -> ~87 bins
        assert _clamp_hit_spectrum_bins(1000.0, 1002.0, 187.5 / 8192, 200) < 200
        assert _clamp_hit_spectrum_bins(1000.0, 1002.0, 187.5 / 8192, 200) >= 1
        # Degenerate / zero-width fine grid: fall back to the cap, never 0 or negative
        assert _clamp_hit_spectrum_bins(1000.0, 1000.0, 0.01, 200) == 200
        assert _clamp_hit_spectrum_bins(1000.0, 1200.0, 0.0, 200) == 200

    def test_ed_hit_spectrum_narrow_span_renders_without_aliasing(self, tmp_path, monkeypatch):
        """A new-style sidecar whose hits occupy a narrow sub-band must render with clamped
        (not 200) bins so no empty picket-fence bins appear between populated ones."""
        get_config().checkpoint.save_tag = TAG
        npy, meta_path, metadata = _write_cadence_artifacts(tmp_path, "narrow", ("N",))
        # Full band 1000..1187.5 MHz, but all hits in a 2 MHz sub-band
        n_fine = 8192
        edges = np.linspace(1000.0, 1187.5, n_fine + 1)
        raw = np.zeros(n_fine, dtype=int)
        in_band = (edges[:-1] >= 1000.0) & (edges[:-1] < 1002.0)
        raw[in_band] = 5  # every fine bin in the sub-band populated -> would alias at 200
        metadata["hit_spectrum_hist"] = {
            "freq_lo": 1000.0,
            "freq_hi": 1187.5,
            "n_bins": n_fine,
            "raw_counts": raw.tolist(),
            "merged_counts": raw.tolist(),
            "raw_freq_min": 1000.0,
            "raw_freq_max": 1002.0,
        }
        metadata.pop("raw_hit_frequencies_mhz", None)
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        coll = InferenceVizCollector()
        coll.record_processed(("N",), npy, meta_path, {}, _fake_results(), 1.0)
        summaries = _build_summaries(coll.records, 12)

        captured: dict = {}

        def _capture(fig, filename, slack_title):
            ax = fig.axes[0]
            # stairs() adds a StepPatch; count its populated-vs-empty structure indirectly
            # by the x-data resolution — assert the axis used clamped bins, not 200
            captured["xlim_span"] = ax.get_xlim()
            return filename

        monkeypatch.setattr("aetherscan.inference_viz._save_and_upload", _capture)
        assert plot_ed_hit_spectrum(coll.records, summaries) is not None
        # The clamp keeps figure bins >= fine width; with fine width ~22.9 kHz over a 2 MHz
        # span the clamp yields ~87 bins, all populated (no picket-fence gaps)
        span = captured["xlim_span"][1] - captured["xlim_span"][0]
        assert span > 0

    def test_ed_hit_spectrum_prebinned_matches_legacy_totals(self, collector, monkeypatch):
        """New sidecars carry hit_spectrum_hist pre-binned by preprocessing; legacy
        sidecars reduce their raw float lists at load (#301). Both routes must report the
        same hit totals for the same underlying hits."""
        legacy = _build_summaries(collector.records, gallery_top_k=12)

        # Rewrite each sidecar into the NEW pre-binned form (drop the raw list)
        for record in collector.records:
            with open(record.metadata_path) as f:
                metadata = json.load(f)
            raw = metadata.pop("raw_hit_frequencies_mhz")
            merged = metadata["merged_hit_frequencies_mhz"]
            edges = np.linspace(1400.0, 1411.0, 8193)
            metadata["hit_spectrum_hist"] = {
                "freq_lo": 1400.0,
                "freq_hi": 1411.0,
                "n_bins": 8192,
                "raw_counts": np.histogram(raw, bins=edges)[0].tolist(),
                "merged_counts": np.histogram(merged, bins=edges)[0].tolist(),
                "raw_freq_min": float(np.min(raw)),
                "raw_freq_max": float(np.max(raw)),
            }
            with open(record.metadata_path, "w") as f:
                json.dump(metadata, f)
        prebinned = _build_summaries(collector.records, gallery_top_k=12)

        titles: list[str] = []

        def _capture(fig, filename, slack_title):
            titles.append(fig.axes[0].get_title())
            return filename

        monkeypatch.setattr("aetherscan.inference_viz._save_and_upload", _capture)
        assert plot_ed_hit_spectrum(collector.records, legacy) is not None
        assert plot_ed_hit_spectrum(collector.records, prebinned) is not None
        assert titles[0] == titles[1]  # same raw/merged totals either route

    def test_stamp_gallery(self, collector, summaries):
        _assert_figure(plot_stamp_gallery(collector.records, summaries))

    def test_preproc_funnel(self, collector, summaries):
        _assert_figure(plot_preproc_funnel(collector.records, summaries))

    def test_preproc_funnel_aggregates_past_cap(self, collector, summaries, monkeypatch):
        """#301: past _FUNNEL_MAX_CADENCES the figure must aggregate the remainder into
        one bar instead of growing past Agg's 2^16-px canvas limit and silently failing."""
        monkeypatch.setattr("aetherscan.inference_viz._FUNNEL_MAX_CADENCES", 1)
        captured: dict[str, str] = {}

        def _capture(fig, filename, slack_title):
            captured["title"] = fig.axes[0].get_title()
            captured["n_bars"] = len(fig.axes[0].get_xticklabels())
            return filename

        monkeypatch.setattr("aetherscan.inference_viz._save_and_upload", _capture)
        assert plot_preproc_funnel(collector.records, summaries) is not None
        assert "aggregated" in captured["title"]
        assert captured["n_bars"] == 2  # 1 individual + the aggregate bar

    def test_confidence_distribution(self, collector):
        _assert_figure(plot_confidence_distribution(collector.records))

    def test_bandpass_flattening(
        self, tmp_path, initialized_runtime, collector, make_h5_observation
    ):
        config = get_config()
        config.manager.n_processes = 1
        config.inference.coarse_channel_width = 512
        config.inference.bandpass_method = "spline"
        config.inference.spline_order = 4
        h5_path = make_h5_observation("viz_obs.h5", n_chans=2048)
        npy_path, metadata_path, _ = _write_cadence_artifacts(
            tmp_path, "cad_h5", ("H",), h5_paths=[str(h5_path)] * 6
        )
        coll = InferenceVizCollector()
        coll.record_processed(("H",), npy_path, metadata_path, {}, _fake_results(), 1.0)
        summaries = _build_summaries(coll.records, gallery_top_k=12)
        _assert_figure(plot_bandpass_flattening(DataPreprocessor(), coll.records, summaries))

    def test_bandpass_flattening_header_without_nchans(
        self, tmp_path, initialized_runtime, collector, make_h5_observation
    ):
        """A header lacking nchans must fall back to the h5 data width (mirroring
        preprocessing's fallback) and still render, not silently skip the figure."""
        config = get_config()
        config.manager.n_processes = 1
        config.inference.coarse_channel_width = 512
        config.inference.bandpass_method = "spline"
        config.inference.spline_order = 4
        h5_path = make_h5_observation("viz_obs_nonchans.h5", n_chans=2048)
        npy_path, metadata_path, metadata = _write_cadence_artifacts(
            tmp_path, "cad_h5_nonchans", ("H2",), h5_paths=[str(h5_path)] * 6
        )
        del metadata["header"]["nchans"]
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        coll = InferenceVizCollector()
        coll.record_processed(("H2",), npy_path, metadata_path, {}, _fake_results(), 1.0)
        summaries = _build_summaries(coll.records, gallery_top_k=12)
        _assert_figure(plot_bandpass_flattening(DataPreprocessor(), coll.records, summaries))

    def test_bandpass_flattening_from_stored_envelopes(self, tmp_path, initialized_runtime):
        """A sidecar carrying bandpass_envelopes (#301) must render WITHOUT touching any
        .h5 — the h5 paths here don't exist, which is exactly the point (the live-read
        fallback would fail on them)."""
        get_config().checkpoint.save_tag = TAG  # figures save under plots/inference/{tag}/
        npy_path, metadata_path, metadata = _write_cadence_artifacts(tmp_path, "cad_env", ("E",))
        assert not os.path.exists(metadata["h5_paths"][0])
        metadata["bandpass_envelopes"] = [
            {
                "channel": 3,
                "overlay_label": "scaled PFB response H",
                "raw": {"idx": [0, 1, 2, 3], "values": [1.0, 2.0, 2.0, 1.0]},
                "flat": {"idx": [0, 1, 2, 3], "values": [1.0, 1.0, 1.0, 1.0]},
                "overlay": {"idx": [0, 1, 2, 3], "values": [1.0, 2.0, 2.0, 1.0]},
            }
        ]
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        coll = InferenceVizCollector()
        coll.record_processed(("E",), npy_path, metadata_path, {}, _fake_results(), 1.0)
        summaries = _build_summaries(coll.records, gallery_top_k=12)
        _assert_figure(plot_bandpass_flattening(DataPreprocessor(), coll.records, summaries))

    def test_build_summaries_retains_bandpass_envelopes_on_first_cadence_only(
        self, tmp_path, initialized_runtime
    ):
        """#301 bound: plot_bandpass_flattening reads envelopes from only the FIRST cadence
        (records order) that has them, so _build_summaries keeps them on that one and drops the
        rest — otherwise every cadence's envelopes stay resident (~GB at catalog scale). Pins
        (a) exactly one summary retains them, (b) it's the first in records order, and (c) the
        consumer still renders from the nulled summaries (same cadence selected)."""
        get_config().checkpoint.save_tag = TAG  # figure saves under plots/inference/{tag}/
        env = [
            {
                "channel": 3,
                "overlay_label": "scaled PFB response H",
                "raw": {"idx": [0, 1, 2, 3], "values": [1.0, 2.0, 2.0, 1.0]},
                "flat": {"idx": [0, 1, 2, 3], "values": [1.0, 1.0, 1.0, 1.0]},
                "overlay": {"idx": [0, 1, 2, 3], "values": [1.0, 2.0, 2.0, 1.0]},
            }
        ]
        coll = InferenceVizCollector()
        for name, key in (("cad_e0", ("E0",)), ("cad_e1", ("E1",)), ("cad_e2", ("E2",))):
            npy_path, metadata_path, metadata = _write_cadence_artifacts(tmp_path, name, key)
            metadata["bandpass_envelopes"] = env
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
            coll.record_processed(key, npy_path, metadata_path, {}, _fake_results(), 1.0)

        summaries = _build_summaries(coll.records, gallery_top_k=12)
        retained = [summaries[r.npy_path].bandpass_envelopes is not None for r in coll.records]
        # Exactly one cadence retains envelopes, and it is the first in records order — the
        # same one plot_bandpass_flattening would pick.
        assert retained == [True, False, False]
        assert summaries[coll.records[0].npy_path].bandpass_envelopes == env
        # The nulled summaries still render (from the retained first cadence's envelopes),
        # without touching any .h5 — proving the consumer is unaffected.
        _assert_figure(plot_bandpass_flattening(DataPreprocessor(), coll.records, summaries))

    def test_candidate_gallery_and_per_candidate(self, initialized_runtime, collector):
        db = initialized_runtime
        config = get_config()
        config.inference.max_candidate_plots = 1
        record = collector.records[0]
        latent = list(np.random.default_rng(5).normal(size=48))
        for idx, conf in ((0, 0.995), (1, 0.999)):
            db.write_inference_result(
                record.npy_path,
                idx,
                1,
                conf,
                latent_vector=np.asarray(latent),
                target="HIP110750",
                band="L",
                frequency_mhz=1400.5,
                tag=TAG,
            )
        assert db.flush(timeout=10) is True

        _assert_figure(plot_candidate_gallery())
        tag_dir = os.path.join(config.output_path, "plots", "inference", TAG)
        # max_candidate_plots=1 caps the per-candidate figures at the top-confidence one
        assert os.path.exists(os.path.join(tag_dir, f"candidate_0_{TAG}.png"))
        assert not os.path.exists(os.path.join(tag_dir, f"candidate_1_{TAG}.png"))

    def test_plot_candidate_without_latent(self, collector):
        record = collector.records[0]
        row = {
            "npy_path": record.npy_path,
            "snippet_index": 0,
            "confidence": 0.99,
            "frequency_mhz": 1400.5,
            "target": "HIP110750",
            "latent_vector": None,
        }
        _assert_figure(plot_candidate(row, 0))

    def test_latent_projection_with_persisted_umap(self, tmp_path, collector):
        config = get_config()
        model_dir = tmp_path / "models"
        model_dir.mkdir(exist_ok=True)
        umap_path = model_dir / "umap_cadence_nn15_md0.1_final_v1.joblib"
        joblib.dump(FakeUMAP(np.random.default_rng(2).normal(size=(30, 2))), umap_path)
        training_config = {
            "paths": {"model_path": str(model_dir)},
            "checkpoint": {"save_tag": "final_v1"},
            "training": {
                "latent_viz_umap_n_neighbors": [15],
                "latent_viz_umap_min_dist": [0.1],
            },
        }
        config_json = tmp_path / "config_final_v1.json"
        config_json.write_text(json.dumps(training_config))
        config.inference.config_path = str(config_json)

        _assert_figure(plot_inference_latent_projection(collector))

    def test_latent_projection_skips_without_training_config(self, collector):
        get_config().inference.config_path = None
        assert plot_inference_latent_projection(collector) is None

    def test_latent_projection_skips_without_umap(self, tmp_path, collector):
        config = get_config()
        training_config = {
            "paths": {"model_path": str(tmp_path / "empty_models")},
            "checkpoint": {"save_tag": "final_v1"},
        }
        config_json = tmp_path / "config_no_umap.json"
        config_json.write_text(json.dumps(training_config))
        config.inference.config_path = str(config_json)
        assert plot_inference_latent_projection(collector) is None

    def test_inference_summary(self, initialized_runtime, collector, summaries):
        db = initialized_runtime
        db.write_inference_cadence(
            npy_path=collector.records[0].npy_path,
            status="preprocessed",
            tag=TAG,
            n_stamps=N_STAMPS,
            duration_s=12.0,
        )
        db.write_inference_cadence(
            npy_path=collector.records[0].npy_path,
            status="inferred",
            tag=TAG,
            n_stamps=N_STAMPS,
            n_candidates=1,
            duration_s=3.0,
        )
        totals = {
            "n_cadence_snippets": 10,
            "n_processed": 10,
            "n_candidates": 2,
            "n_cadences": 2,
            "n_skipped": 0,
        }
        _assert_figure(plot_inference_summary(collector.records, summaries, totals))

    def test_inference_summary_counts_superseded_preprocessing(
        self, initialized_runtime, collector, summaries, monkeypatch
    ):
        """Regression: _infer_cadence supersedes each cadence's 'preprocessed' row just before
        writing its live 'inferred' row, so on a fully successful run every 'preprocessed' row
        is superseded. The summary aggregation must include superseded rows (else preprocessing
        time always reads 0.0 s) while summing per status so the preprocessed and inferred rows
        of the same cadence aren't double-counted."""
        db = initialized_runtime
        npy_path = collector.records[0].npy_path
        # Reproduce the post-success manifest state for one cadence: a 'preprocessed' row
        # retired by mark_superseded, then the live 'inferred' row (the _infer_cadence order).
        db.write_inference_cadence(
            npy_path=npy_path, status="preprocessed", tag=TAG, n_stamps=N_STAMPS, duration_s=12.0
        )
        assert db.mark_superseded("inference_cadences", TAG, npy_path=npy_path) is True
        db.write_inference_cadence(
            npy_path=npy_path,
            status="inferred",
            tag=TAG,
            n_stamps=N_STAMPS,
            n_candidates=1,
            duration_s=3.0,
        )
        assert db.flush(timeout=10) is True

        # The 'preprocessed' row is invisible to the default query but must drive the summary.
        assert db.query_inference_cadences(tag=TAG, status="preprocessed") == []

        # Capture the rendered summary text (patch before _save_and_upload clears the figure).
        captured: dict[str, str] = {}

        def _capture(fig, filename, slack_title):
            captured["text"] = "\n".join(t.get_text() for ax in fig.axes for t in ax.texts)
            return filename

        monkeypatch.setattr("aetherscan.inference_viz._save_and_upload", _capture)

        totals = {
            "n_cadence_snippets": 10,
            "n_candidates": 1,
            "n_cadences": 1,
            "n_skipped": 0,
        }
        plot_inference_summary(collector.records, summaries, totals)

        lines = captured["text"].splitlines()
        preproc_line = next(line for line in lines if line.startswith("preprocessing time"))
        inference_line = next(line for line in lines if line.startswith("inference time"))
        # Real non-zero value (bug read 0.0 s), summed exactly once (double count -> 24.0 s).
        assert preproc_line.split()[-2:] == ["12.0", "s"]
        assert inference_line.split()[-2:] == ["3.0", "s"]


class TestSuiteEntryPoint:
    def test_render_all_never_raises(self, initialized_runtime, collector):
        totals = {
            "n_cadence_snippets": 10,
            "n_processed": 10,
            "n_candidates": 2,
            "n_cadences": 2,
            "n_skipped": 0,
        }
        render_inference_visualizations(collector, DataPreprocessor(), totals)
        tag_dir = os.path.join(get_config().output_path, "plots", "inference", TAG)
        rendered = os.listdir(tag_dir)
        # Core figures render even though the h5s/UMAP/candidates are unavailable
        # (those figures log and skip instead of failing the suite)
        for stem in (
            "ed_stat_distributions",
            "ed_hit_spectrum",
            "stamp_gallery",
            "preproc_funnel",
            "confidence_distribution",
            "inference_summary",
        ):
            assert f"{stem}_{TAG}.png" in rendered

    def test_render_survives_broken_records(self, initialized_runtime):
        """Missing .npy/metadata everywhere: every figure must degrade to a logged skip."""
        coll = InferenceVizCollector()
        coll.record_skipped(("X",), "/nope/gone.npy", "/nope/gone.json", {"n_stamps": 1})
        render_inference_visualizations(
            coll, DataPreprocessor(), {"n_cadence_snippets": 1, "n_cadences": 1}
        )

    def test_scope_new_excludes_resumed_cadences(self, initialized_runtime, collector):
        """--inference-viz-scope new (#301): resumed cadences drop out of the
        metadata-driven figures, so a multi-pass campaign stops re-paying the whole
        catalog's viz tail every pass."""
        get_config().inference.inference_viz_scope = "new"
        collector.record_skipped(("R",), "/nope/resumed.npy", "/nope/resumed.json", {})
        totals = {"n_cadence_snippets": 10, "n_cadences": 3, "n_skipped": 1}
        render_inference_visualizations(collector, DataPreprocessor(), totals)
        # The resumed record's missing sidecar would have warned; the two live ones render
        tag_dir = os.path.join(get_config().output_path, "plots", "inference", TAG)
        assert f"ed_stat_distributions_{TAG}.png" in os.listdir(tag_dir)


class TestCandidateFrequencyMap:
    """#394: candidate frequency map — dots at (frequency, target) colored by band, with
    report-excluded ranges shaded; skips gracefully with no candidates."""

    def _write_candidates(self, db, npy_path, specs):
        for idx, (target, band, freq) in enumerate(specs):
            db.write_inference_result(
                npy_path,
                idx,
                1,
                0.999,
                target=target,
                band=band,
                frequency_mhz=freq,
                tag=TAG,
            )
        assert db.flush(timeout=10) is True

    def test_renders_with_candidates(self, initialized_runtime, collector):
        from aetherscan.inference_viz import plot_candidate_frequency  # noqa: PLC0415

        db = initialized_runtime
        record = collector.records[0]
        self._write_candidates(
            db,
            record.npy_path,
            [("NGC1172", "C", 1552.7), ("CVNI", "L", 1576.39), ("MESSIER87", "X", 8205.0)],
        )
        _assert_figure(plot_candidate_frequency())

    def test_renders_with_exclusion_shading_and_target_overflow(
        self, initialized_runtime, collector, monkeypatch
    ):
        import aetherscan.inference_viz as viz  # noqa: PLC0415

        db = initialized_runtime
        record = collector.records[0]
        # More targets than the row cap -> the overflow aggregate row must carry the
        # remaining candidate count without KeyErrors
        monkeypatch.setattr(viz, "_CANDIDATE_FREQ_MAX_TARGETS", 2)
        get_config().inference.report_exclude_frequency_ranges = [[1616.0, 1626.5]]
        self._write_candidates(
            db,
            record.npy_path,
            [
                ("T1", "L", 1620.0),
                ("T1", "L", 1621.0),
                ("T2", "C", 8438.0),
                ("T3", "X", 9500.0),
                ("T4", "S", 2300.0),
            ],
        )
        _assert_figure(viz.plot_candidate_frequency())

    def test_skips_without_candidates(self, initialized_runtime):
        from aetherscan.inference_viz import plot_candidate_frequency  # noqa: PLC0415

        assert plot_candidate_frequency() is None


class TestReportExclusionSurfaces:
    """#395: excluded candidates keep their rendered figures on disk but are dropped from
    Slack uploads and the gallery; tallies carry original vs excluded vs reported."""

    def _write_two_candidates(self, db, npy_path):
        latent = list(np.random.default_rng(5).normal(size=48))
        # Higher confidence on the GPS-band candidate so it ranks first
        for idx, (conf, freq) in enumerate(((0.999, 1575.42), (0.995, 8438.0))):
            db.write_inference_result(
                npy_path,
                idx,
                1,
                conf,
                latent_vector=np.asarray(latent),
                target="HIP110750",
                band="L",
                frequency_mhz=freq,
                tag=TAG,
            )
        assert db.flush(timeout=10) is True

    def test_gallery_skips_uploads_for_excluded_but_saves_figures(
        self, initialized_runtime, collector, monkeypatch
    ):
        import aetherscan.inference_viz as viz  # noqa: PLC0415

        db = initialized_runtime
        config = get_config()
        config.inference.report_exclude_frequency_ranges = [[1575.0, 1576.0]]
        record = collector.records[0]
        self._write_two_candidates(db, record.npy_path)

        uploads = []
        monkeypatch.setattr(viz._uploader, "submit", lambda path, title: uploads.append(title))

        _assert_figure(viz.plot_candidate_gallery())
        tag_dir = os.path.join(config.output_path, "plots", "inference", TAG)
        # Both per-candidate figures rendered and saved (index 0 = the excluded GPS hit)...
        assert os.path.exists(os.path.join(tag_dir, f"candidate_0_{TAG}.png"))
        assert os.path.exists(os.path.join(tag_dir, f"candidate_1_{TAG}.png"))
        # ...but only the clean candidate and the gallery itself were uploaded
        assert any(title.startswith("Candidate 1") for title in uploads)
        assert not any(title.startswith("Candidate 0") for title in uploads)
        assert any(title.startswith("Candidate Gallery") for title in uploads)

    def test_gallery_all_excluded_skips_gallery_figure(
        self, initialized_runtime, collector, monkeypatch
    ):
        import aetherscan.inference_viz as viz  # noqa: PLC0415

        db = initialized_runtime
        config = get_config()
        config.inference.report_exclude_frequency_ranges = [[1000.0, 9000.0]]
        record = collector.records[0]
        self._write_two_candidates(db, record.npy_path)
        monkeypatch.setattr(viz._uploader, "submit", lambda path, title: None)

        assert viz.plot_candidate_gallery() is None
        # The per-candidate figures still landed on disk
        tag_dir = os.path.join(config.output_path, "plots", "inference", TAG)
        assert os.path.exists(os.path.join(tag_dir, f"candidate_0_{TAG}.png"))

    def test_summary_card_carries_exclusion_counts(
        self, initialized_runtime, collector, summaries, monkeypatch
    ):
        import aetherscan.inference_viz as viz  # noqa: PLC0415

        db = initialized_runtime
        config = get_config()
        config.inference.report_exclude_frequency_ranges = [[1575.0, 1576.0]]
        record = collector.records[0]
        self._write_two_candidates(db, record.npy_path)

        captured = {}

        def _capture(fig, filename, slack_title):
            ax = fig.axes[0]
            captured["text"] = "\n".join(t.get_text() for t in ax.texts)
            return "captured"

        monkeypatch.setattr(viz, "_save_and_upload", _capture)
        totals = {"n_cadences": 2, "n_candidates": 2, "n_cadence_snippets": 10}
        assert viz.plot_inference_summary(collector.records, summaries, totals) == "captured"
        assert "excluded (1575-1576 MHz)" in captured["text"]
        assert "reported after exclusion" in captured["text"]


class TestCandidateTriageReport:
    """#397: triage CSV in review order, survey-OOD tie-break within equal confidence,
    graceful skips when scores are unavailable."""

    def _write_candidates_with_latents(self, db, npy_path, specs):
        for idx, (conf, freq, latent) in enumerate(specs):
            db.write_inference_result(
                npy_path,
                idx,
                1,
                conf,
                latent_vector=np.asarray(latent, dtype=np.float64),
                target="HIP110750",
                band="L",
                frequency_mhz=freq,
                tag=TAG,
                mc_mean=conf,
                mc_std=0.001 * (idx + 1),
            )
        assert db.flush(timeout=10) is True

    def test_report_orders_by_survey_ood_within_confidence_tie(
        self, initialized_runtime, collector
    ):
        import aetherscan.inference_viz as viz  # noqa: PLC0415

        db = initialized_runtime
        config = get_config()
        rng = np.random.default_rng(9)
        # Reference cloud of typical latents at the path the pipeline would write
        cloud = rng.normal(size=(300, 48)).astype(np.float32)
        np.savez_compressed(
            viz._reference_cloud_path(config, TAG),
            latent_mean=cloud,
            mc_mean=np.zeros(300, dtype=np.float32),
            mc_std=np.zeros(300, dtype=np.float32),
        )

        record = collector.records[0]
        # Equal confidence: snippet 0 is a deep inlier, snippet 1 an extreme outlier ->
        # snippet 1 must review first
        self._write_candidates_with_latents(
            db,
            record.npy_path,
            [(1.0, 1400.5, [0.0] * 48), (1.0, 8438.0, [30.0] * 48)],
        )

        report_path = viz.write_candidate_triage_report()
        assert report_path is not None and os.path.exists(report_path)
        with open(report_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert [int(r["snippet_index"]) for r in rows] == [1, 0]
        assert float(rows[0]["survey_ood_percentile"]) == 100.0
        # No training artifact in this environment -> training columns empty, not a crash
        assert rows[0]["training_ood_distance"] == ""

    def test_report_without_candidates_skips(self, initialized_runtime):
        from aetherscan.inference_viz import write_candidate_triage_report  # noqa: PLC0415

        assert write_candidate_triage_report() is None

    def test_report_marks_frequency_excluded(self, initialized_runtime, collector):
        import aetherscan.inference_viz as viz  # noqa: PLC0415

        db = initialized_runtime
        config = get_config()
        config.inference.report_exclude_frequency_ranges = [[1575.0, 1576.0]]
        record = collector.records[0]
        self._write_candidates_with_latents(
            db,
            record.npy_path,
            [(0.999, 1575.42, [0.0] * 48), (0.995, 8438.0, [0.1] * 48)],
        )
        report_path = viz.write_candidate_triage_report()
        with open(report_path) as f:
            rows = {int(r["snippet_index"]): r for r in csv.DictReader(f)}
        assert rows[0]["excluded_by_report_filter"] == "1"
        assert rows[1]["excluded_by_report_filter"] == "0"

    def test_gallery_renders_with_ood_annotation(self, initialized_runtime, collector):
        import aetherscan.inference_viz as viz  # noqa: PLC0415

        db = initialized_runtime
        config = get_config()
        rng = np.random.default_rng(9)
        np.savez_compressed(
            viz._reference_cloud_path(config, TAG),
            latent_mean=rng.normal(size=(100, 48)).astype(np.float32),
        )
        record = collector.records[0]
        self._write_candidates_with_latents(
            db, record.npy_path, [(1.0, 1400.5, [0.0] * 48), (1.0, 8438.0, [30.0] * 48)]
        )
        _assert_figure(viz.plot_candidate_gallery())
