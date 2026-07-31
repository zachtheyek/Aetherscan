"""Smoke tests for aetherscan.inference_viz: every figure function runs against small
synthetic inputs and produces a non-empty PNG under plots/inference/{tag}/, the collector
keeps bounded state, and the suite entry point never raises (a plot bug must not kill a
science run)."""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pytest

from aetherscan.config import get_config
from aetherscan.inference_viz import (
    CONFIDENCE_HIST_EDGES,
    InferenceVizCollector,
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
def metadatas(collector):
    result = {}
    for record in collector.records:
        with open(record.metadata_path) as f:
            result[record.npy_path] = json.load(f)
    return result


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
        metadatas = {npy_path: metadata}

        selected = _select_top_stamps(coll.records, metadatas, top_k=4)
        starts = [metadata["stamp_starts"][idx] for _, idx, _, _ in selected]
        assert len([s for s in starts if s in (872, 1000, 1128)]) == 1
        assert 5000 in starts


class TestFigureSmoke:
    def test_ed_stat_distributions(self, collector, metadatas):
        _assert_figure(plot_ed_stat_distributions(collector.records, metadatas))

    def test_ed_stat_distributions_drops_mismatched_bins_consistently(
        self, tmp_path, collector, metadatas, monkeypatch
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
        metadatas[npy_path] = metadata

        captured: dict[str, str] = {}

        def _capture(fig, filename, slack_title):
            captured["title"] = fig.axes[0].get_title()
            return filename

        monkeypatch.setattr("aetherscan.inference_viz._save_and_upload", _capture)
        assert plot_ed_stat_distributions(collector.records, metadatas) is not None

        title = captured["title"]
        # Only the two good cadences contribute: 3 * N_STAMPS raw hits each
        assert f"{2 * 3 * N_STAMPS} above threshold" in title
        assert "(2 cadence(s))" in title
        assert "finite windows" in title

    def test_ed_hit_spectrum(self, collector, metadatas):
        _assert_figure(plot_ed_hit_spectrum(collector.records, metadatas))

    def test_stamp_gallery(self, collector, metadatas):
        _assert_figure(plot_stamp_gallery(collector.records, metadatas))

    def test_preproc_funnel(self, collector, metadatas):
        _assert_figure(plot_preproc_funnel(collector.records, metadatas))

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
        with open(metadata_path) as f:
            metas = {npy_path: json.load(f)}
        _assert_figure(plot_bandpass_flattening(DataPreprocessor(), coll.records, metas))

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
        metas = {npy_path: metadata}
        _assert_figure(plot_bandpass_flattening(DataPreprocessor(), coll.records, metas))

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

    def test_inference_summary(self, initialized_runtime, collector, metadatas):
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
        _assert_figure(plot_inference_summary(collector.records, metadatas, totals))

    def test_inference_summary_counts_superseded_preprocessing(
        self, initialized_runtime, collector, metadatas, monkeypatch
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
        plot_inference_summary(collector.records, metadatas, totals)

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
