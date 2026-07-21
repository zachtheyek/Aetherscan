"""
Inference visualization suite for Aetherscan Pipeline

Renders the end-of-run figures for a streaming CSV inference run: energy-detection statistic
distributions, hit spectrum (pre/post dedup), bandpass-flattening overlay, top-K stamp
gallery, preprocessing funnel, confidence distribution, candidate gallery + per-candidate
plots, latent projection through the persisted training UMAP, and a run summary card.

Data flows in three ways:
- transient per-cadence state (confidence histograms, subsampled latent features) collected
  by InferenceVizCollector during the streaming loop in main._run_streaming_csv_inference;
- durable per-cadence artifacts (the metadata JSON written by preprocessing next to each
  stamp .npy, and the .npy itself) — these also cover cadences the stage-aware resume
  skipped this pass;
- the database (inference_results candidates, inference_cadences manifest aggregates).

Every figure is saved under {output_path}/plots/inference/{save_tag}/ and uploaded to Slack,
mirroring train.py's plot pattern. render_inference_visualizations wraps each figure in
_viz_safe: a plot bug logs an error and moves on — it can never kill a science run.

Matplotlib usage is strictly the object-oriented Figure API (never pyplot): the pipeline runs
background threads, and pyplot's global figure registry is not thread-safe.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import h5py
import numpy as np
from matplotlib.figure import Figure

from aetherscan.config import get_config
from aetherscan.data_generation import log_norm
from aetherscan.db import get_db
from aetherscan.logger import get_logger
from aetherscan.models import prepare_latent_features
from aetherscan.pfb import gen_coarse_channel_response

logger = logging.getLogger(__name__)

# Fixed linear bins over [0, 1] for the per-cadence P(true) histograms accumulated by the
# collector (identical bins everywhere, so per-cadence counts combine by addition).
CONFIDENCE_HIST_EDGES = np.linspace(0.0, 1.0, 101)

# Cap on latent feature rows kept for the latent-projection figure. Candidates are always
# kept; non-candidates fill the remaining budget first-come (earlier cadences are slightly
# over-represented on huge catalogs, which is fine for a qualitative projection).
_MAX_LATENT_POINTS = 20_000

# Frequency-axis bins for the hit spectrum figure.
_HIT_SPECTRUM_BINS = 200

# Candidate gallery shows at most this many top-confidence candidates (per-candidate figures
# are governed separately by config.inference.max_candidate_plots).
_CANDIDATE_GALLERY_MAX = 12

# ON/OFF row labels for 6-observation ABACAD cadence strips.
_OBS_ROW_LABELS = ("ON", "OFF", "ON", "OFF", "ON", "OFF")


@dataclass
class CadenceVizRecord:
    """Per-cadence state the viz suite keeps after the cadence's arrays are dropped."""

    key: tuple
    npy_path: str
    metadata_path: str
    skipped: bool  # True when the stage-aware resume skipped the cadence this pass
    n_stamps: int
    n_candidates: int
    provenance: dict = field(default_factory=dict)
    confidence_hist: np.ndarray | None = None  # counts over CONFIDENCE_HIST_EDGES
    inference_duration_s: float | None = None


class InferenceVizCollector:
    """
    Accumulates bounded per-cadence state during the streaming inference loop.

    Memory stays O(#cadences + _MAX_LATENT_POINTS) regardless of catalog size: confidence
    vectors are folded into fixed-bin histograms immediately, and latent features are
    subsampled against a global budget (candidates always kept).
    """

    def __init__(self, max_latent_points: int = _MAX_LATENT_POINTS):
        self.records: list[CadenceVizRecord] = []
        self._max_latent_points = max_latent_points
        self._latent_chunks: list[np.ndarray] = []  # (k, num_obs * latent_dim) each
        self._candidate_chunks: list[np.ndarray] = []  # bool masks aligned with chunks
        self._latent_count = 0
        self._rng = np.random.default_rng(11)

    def record_skipped(
        self, key: tuple, npy_path: str, metadata_path: str, manifest_row: dict
    ) -> None:
        """Record a cadence the stage-aware resume skipped (aggregates from its manifest row)."""
        self.records.append(
            CadenceVizRecord(
                key=key,
                npy_path=npy_path,
                metadata_path=metadata_path,
                skipped=True,
                n_stamps=int(manifest_row.get("n_stamps") or 0),
                n_candidates=int(manifest_row.get("n_candidates") or 0),
                inference_duration_s=manifest_row.get("duration_s"),
            )
        )

    def record_processed(
        self,
        key: tuple,
        npy_path: str,
        metadata_path: str,
        provenance: dict,
        results: dict,
        duration_s: float,
    ) -> None:
        """Record a cadence inferred this pass. `results` is run_inference's dict; its
        proba_true / predictions / latents arrays are reduced here and not retained."""
        proba_true = np.asarray(results["proba_true"])
        predictions = np.asarray(results["predictions"])
        confidence_hist, _ = np.histogram(np.clip(proba_true, 0.0, 1.0), bins=CONFIDENCE_HIST_EDGES)

        self.records.append(
            CadenceVizRecord(
                key=key,
                npy_path=npy_path,
                metadata_path=metadata_path,
                skipped=False,
                n_stamps=int(results["n_cadence_snippets"]),
                n_candidates=int(results["n_candidates"]),
                provenance=provenance,
                confidence_hist=confidence_hist,
                inference_duration_s=duration_s,
            )
        )

        self._budget_fill_add(results["latents"], predictions.astype(bool))

    def _budget_fill_add(self, latents: np.ndarray, is_candidate: np.ndarray) -> None:
        """Fold one cadence's latents into the bounded projection pool by first-come greedy
        budget-fill (not reservoir sampling): cadence-level features (obs latents
        concatenated), candidates always kept; non-candidates fill whatever global budget
        remains, uniformly subsampled WITHIN this cadence when they'd overflow it (later
        cadences get nothing once the budget is spent)."""
        config = get_config()
        features = prepare_latent_features(
            np.asarray(latents), config.data.num_observations
        ).astype(np.float32)

        keep = np.nonzero(is_candidate)[0]
        budget = self._max_latent_points - self._latent_count - keep.size
        non_candidates = np.nonzero(~is_candidate)[0]
        if budget > 0 and non_candidates.size > 0:
            if non_candidates.size > budget:
                non_candidates = self._rng.choice(non_candidates, size=budget, replace=False)
            keep = np.concatenate([keep, non_candidates])

        if keep.size == 0:
            return
        self._latent_chunks.append(features[keep])
        self._candidate_chunks.append(is_candidate[keep])
        self._latent_count += keep.size

    def latent_pool(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (features, is_candidate) stacked over everything collected; empty arrays
        when nothing was processed this pass."""
        if not self._latent_chunks:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=bool)
        return np.concatenate(self._latent_chunks), np.concatenate(self._candidate_chunks)


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def _viz_safe(name: str, fn: Callable, *args, **kwargs):
    """Run one figure function, log-and-swallow any exception: a plot bug must never kill a
    science run (mirrors train.py's _safe_call, with viz-specific logging)."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.error(f"Inference viz '{name}' failed (run continues without it): {e}")
        return None


def _plots_dir(tag: str) -> str:
    path = os.path.join(get_config().output_path, "plots", "inference", tag)
    os.makedirs(path, exist_ok=True)
    return path


def _save_and_upload(fig: Figure, filename: str, slack_title: str) -> str:
    """Save a figure under plots/inference/{tag}/ and upload it to Slack (train.py's plot
    tail, minus pyplot). Returns the saved path."""
    tag = get_config().checkpoint.save_tag
    save_path = os.path.join(_plots_dir(tag), filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    # Eagerly release the figure's artists/render buffers. OO-API Figures aren't tracked by
    # a global registry, so this isn't a leak fix — it just frees the backing memory now
    # instead of at garbage collection (relevant for dense stamp/candidate galleries). The
    # Slack upload below reads the saved PNG, not the figure.
    fig.clear()
    logger.info(f"Inference viz saved: {save_path}")

    logger_instance = get_logger()
    if logger_instance:
        logger_instance.upload_image_to_slack(
            save_path, title=f"{slack_title} - ({tag}, {socket.gethostname()})"
        )
    return save_path


def _load_metadata(record: CadenceVizRecord) -> dict | None:
    """Best-effort read of a cadence's metadata JSON (durable ED provenance)."""
    try:
        with open(record.metadata_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Viz: could not read metadata {record.metadata_path} ({e}); skipping")
        return None


def _load_display_cadence(npy_path: str, snippet_index: int) -> np.ndarray:
    """Load one snippet's (num_obs, time_bins, width) stamp from its .npy and log-normalize
    it for display (the same transform the model input path applies)."""
    stamps = np.load(npy_path, mmap_mode="r")
    snippet = np.array(stamps[snippet_index], dtype=np.float32)
    del stamps
    return log_norm(snippet)


def _draw_cadence_strip(axes_column, snippet: np.ndarray, label_rows: bool) -> None:
    """Draw one snippet's 6 observation waterfalls down a column of axes."""
    for obs_idx, ax in enumerate(axes_column):
        ax.imshow(
            snippet[obs_idx],
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if label_rows:
            ax.set_ylabel(_OBS_ROW_LABELS[obs_idx], fontsize=8, rotation=0, ha="right", va="center")


def _key_label(key: tuple, max_len: int = 28) -> str:
    label = "/".join(str(part) for part in key)
    return label if len(label) <= max_len else label[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# Per-run figures: energy detection / preprocessing
# ---------------------------------------------------------------------------


def plot_ed_stat_distributions(
    records: list[CadenceVizRecord], metadatas: dict[str, dict]
) -> str | None:
    """Histogram of the D'Agostino-Pearson k2 statistic over all finite windows (not just
    hits — the ED workers histogram only finite k2 values), log-log, per-ON-file overlay +
    total, with the detection threshold marked. Sourced from the fixed-bin histograms the
    ED workers accumulate into each cadence's metadata."""
    config = get_config()
    tag = config.checkpoint.save_tag
    stat_threshold = config.inference.stat_threshold

    edges: np.ndarray | None = None
    per_on_totals: np.ndarray | None = None
    contributing: set[str] = set()  # npy_paths whose histograms actually landed in the totals
    for record in records:
        metadata = metadatas.get(record.npy_path)
        ed_hist = (metadata or {}).get("ed_stat_hist")
        if not ed_hist:
            continue
        cadence_edges = np.asarray(ed_hist["bin_edges"], dtype=np.float64)
        counts = np.asarray(ed_hist["counts_per_on_file"], dtype=np.int64)
        if edges is None:
            edges = cadence_edges
            per_on_totals = np.zeros_like(counts)
        elif cadence_edges.shape != edges.shape or not np.allclose(cadence_edges, edges):
            logger.warning(f"Viz: {record.npy_path} has mismatched ED hist bins; skipping it")
            continue
        if counts.shape != per_on_totals.shape:
            logger.warning(f"Viz: {record.npy_path} has unexpected ED hist shape; skipping it")
            continue
        per_on_totals += counts
        contributing.add(record.npy_path)

    if edges is None or per_on_totals is None or per_on_totals.sum() == 0:
        logger.info("Viz: no ED statistic histograms available; skipping ed_stat_distributions")
        return None

    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric bin centers (log-spaced bins)

    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    for on_idx in range(per_on_totals.shape[0]):
        ax.step(
            centers,
            per_on_totals[on_idx],
            where="mid",
            lw=1.0,
            label=f"ON file {on_idx + 1}",
        )
    ax.step(
        centers, per_on_totals.sum(axis=0), where="mid", lw=1.6, color="black", label="all ON files"
    )
    ax.axvline(
        stat_threshold, color="red", ls="--", lw=1.2, label=f"threshold ({stat_threshold:g})"
    )
    # Exact above-threshold count from the workers' hit lists (the histogram bins are fixed,
    # so the threshold generally falls inside a bin — summing bins would be approximate).
    # Summed over the SAME cadence subset that built the histogram, so the above/total pair
    # stays consistent when a mismatched-bins/shape cadence was dropped above.
    above = sum(int(metadatas[p].get("n_raw_hits") or 0) for p in contributing)
    total = int(per_on_totals.sum())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D'Agostino-Pearson $k^2$ statistic")
    ax.set_ylabel("window count")
    ax.set_title(
        f"Energy-detection statistic distribution ({tag})\n"
        f"{total:,} finite windows, {above:,} above threshold "
        f"({len(contributing)} cadence(s))"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)

    return _save_and_upload(fig, f"ed_stat_distributions_{tag}.png", "ED Statistic Distribution")


def plot_ed_hit_spectrum(records: list[CadenceVizRecord], metadatas: dict[str, dict]) -> str | None:
    """Hit density vs frequency (MHz) across the band, pre- vs post-deduplication — RFI comb
    structure shows up immediately as spikes/picket-fences."""
    tag = get_config().checkpoint.save_tag

    # NOTE: unlike the pre-binned ED stat histograms, hit frequencies are accumulated raw
    # across every cadence before the histogram call — an asymmetry that is bounded at
    # current catalog scale (~1e5-1e6 floats) but would warrant pre-binning in the
    # metadata (fixed frequency grid, like ed_stat_hist) if catalogs grow to hundreds of
    # RFI-dense cadences.
    raw_freqs: list[float] = []
    merged_freqs: list[float] = []
    for record in records:
        metadata = metadatas.get(record.npy_path) or {}
        raw_freqs.extend(metadata.get("raw_hit_frequencies_mhz") or [])
        merged_freqs.extend(metadata.get("merged_hit_frequencies_mhz") or [])

    if not raw_freqs:
        logger.info("Viz: no hit frequencies available; skipping ed_hit_spectrum")
        return None

    lo, hi = float(np.min(raw_freqs)), float(np.max(raw_freqs))
    if lo == hi:  # single-frequency degenerate range: give the histogram some width
        lo, hi = lo - 0.5, hi + 0.5
    bins = np.linspace(lo, hi, _HIT_SPECTRUM_BINS + 1)

    fig = Figure(figsize=(12, 5))
    ax = fig.subplots()
    ax.hist(raw_freqs, bins=bins, histtype="stepfilled", alpha=0.35, label="raw hits (pre-dedup)")
    ax.hist(merged_freqs, bins=bins, histtype="step", lw=1.4, color="crimson", label="merged hits")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("hit count")
    ax.set_title(
        f"Energy-detection hit spectrum ({tag})\n"
        f"{len(raw_freqs):,} raw → {len(merged_freqs):,} merged hits"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    return _save_and_upload(fig, f"ed_hit_spectrum_{tag}.png", "ED Hit Spectrum")


def plot_bandpass_flattening(
    preprocessor, records: list[CadenceVizRecord], metadatas: dict[str, dict]
) -> str | None:
    """Integrated spectrum raw vs flattened for a few coarse channels sampled across the
    band of the first cadence's primary ON file, with the removed model (scaled PFB response
    H or spline fit) overlaid — formalizes PR-07's opt-in debug artifact as a standard
    per-run figure."""
    # NOTE: reaches into DataPreprocessor's private helpers on purpose — they are the
    # single source of truth for how a channel is read/despiked/flattened, and duplicating
    # that here would let the figure drift from what detection actually does.
    from aetherscan.preprocessing import (  # noqa: PLC0415
        _decimate_for_plot,
        _fit_channel_bandpass,
        _pfb_flatten_bandpass,
    )

    config = get_config()
    tag = config.checkpoint.save_tag
    width = config.inference.coarse_channel_width

    h5_path = None
    for record in records:
        metadata = metadatas.get(record.npy_path) or {}
        h5_paths = metadata.get("h5_paths") or []
        if h5_paths and os.path.exists(h5_paths[0]):
            h5_path = h5_paths[0]
            n_chans = int((metadata.get("header") or {}).get("nchans", 0))
            break
    if h5_path is None:
        logger.info("Viz: no readable ON-source .h5 available; skipping bandpass_flattening")
        return None

    if n_chans <= 0:
        # Header lacks nchans: fall back to the data width, mirroring preprocessing's
        # int(header.get("nchans", data_shape[-1])) — detection processed such a cadence
        # fine, so the figure must not silently lose it. Shape access reads h5 metadata
        # only (no decompression), so plain h5py suffices.
        with h5py.File(h5_path, "r") as hf:
            n_chans = int(hf["data"].shape[-1])

    n_coarse_total = n_chans // width
    if n_coarse_total == 0:
        logger.info(
            f"Viz: file width ({n_chans} fine channels) is narrower than one coarse channel "
            f"({width}); skipping bandpass_flattening"
        )
        return None

    bandpass_flatten = preprocessor._get_bandpass_flattener(n_coarse_total)
    pfb_active = bandpass_flatten.func is _pfb_flatten_bandpass
    sampled = preprocessor._sample_channel_indices(n_coarse_total)

    fig = Figure(figsize=(14, 3.2 * len(sampled)))
    axes = fig.subplots(len(sampled), 2, squeeze=False)
    for row, ch in enumerate(sampled):
        channel = preprocessor._read_despiked_channel(h5_path, ch)
        raw = channel.mean(axis=0)
        flat = np.asarray(bandpass_flatten(channel)).mean(axis=0)
        if pfb_active:
            response = gen_coarse_channel_response(
                width, n_coarse_total, config.inference.pfb_taps_per_channel
            )
            # Least-squares scale so the unit-peak response overlays the raw spectrum
            overlay = response * (float(raw @ response) / float(response @ response))
            overlay_label = "scaled PFB response H"
        else:
            overlay = _fit_channel_bandpass(raw, width, config.inference.spline_order)
            overlay_label = "spline fit"

        ax_raw, ax_flat = axes[row]
        # Decimated to a min/max envelope: full-resolution lines are ~1M points each at
        # GBT scale, which makes rendering slow and memory-heavy for no visual gain.
        ax_raw.plot(
            *_decimate_for_plot(raw), lw=0.6, color="tab:blue", label="raw integrated spectrum"
        )
        ax_raw.plot(
            *_decimate_for_plot(overlay),
            lw=1.2,
            ls="--",
            color="tab:orange",
            label=overlay_label,
        )
        ax_raw.set_ylabel(f"coarse channel {ch}\nintegrated power")
        ax_flat.plot(
            *_decimate_for_plot(flat),
            lw=0.6,
            color="tab:green",
            label="flattened integrated spectrum",
        )
        if row == 0:
            ax_raw.legend(loc="upper right", fontsize=8)
            ax_flat.legend(loc="upper right", fontsize=8)
        if row == len(sampled) - 1:
            ax_raw.set_xlabel("fine channel (within coarse channel)")
            ax_flat.set_xlabel("fine channel (within coarse channel)")

    method = "pfb" if pfb_active else "spline"
    fig.suptitle(f"Bandpass flattening ({method}, {tag}): {os.path.basename(h5_path)}")
    fig.tight_layout()

    return _save_and_upload(fig, f"bandpass_flattening_{tag}.png", "Bandpass Flattening")


def _select_top_stamps(
    records: list[CadenceVizRecord], metadatas: dict[str, dict], top_k: int
) -> list[tuple[CadenceVizRecord, int, float, float]]:
    """Pick the top_k stamps by detection statistic across all cadences, collapsing
    overlap-search offset copies first: with overlap_search each hit yields up to three
    stamps (at -offset/0/+offset) that all carry the hit's statistic, so a plain top-K
    would show the same hit up to three times. Copies are grouped by their (exact) shared
    statistic per cadence and represented by the median-start stamp — the offset-0 center
    for a full triplet. Returns (record, snippet_index, statistic, frequency_mhz) tuples,
    strongest first."""
    representatives: list[tuple[float, CadenceVizRecord, int, float]] = []
    for record in records:
        metadata = metadatas.get(record.npy_path) or {}
        stats_list = metadata.get("stamp_statistics") or []
        freqs = metadata.get("stamp_frequencies_mhz") or []
        starts = metadata.get("stamp_starts") or []

        # Group this cadence's stamps by exact statistic value (offset copies of one hit
        # share the same float64 statistic; distinct hits colliding on the exact value is
        # vanishingly unlikely, and for a gallery a collision merely hides a duplicate look)
        by_stat: dict[float, list[tuple[int, int]]] = {}
        for idx, stat in enumerate(stats_list):
            start = int(starts[idx]) if idx < len(starts) else 0
            by_stat.setdefault(float(stat), []).append((start, idx))

        for stat, members in by_stat.items():
            members.sort()  # by start; median = offset-0 center for a full triplet
            _, idx = members[len(members) // 2]
            freq = float(freqs[idx]) if idx < len(freqs) else float("nan")
            representatives.append((stat, record, idx, freq))

    representatives.sort(key=lambda c: c[0], reverse=True)
    return [(record, idx, stat, freq) for stat, record, idx, freq in representatives[:top_k]]


def plot_stamp_gallery(records: list[CadenceVizRecord], metadatas: dict[str, dict]) -> str | None:
    """Top-K stamps by detection statistic, each rendered as the 6-observation cadence
    waterfall grid scientists actually inspect (ON/OFF rows, one stamp per column)."""
    config = get_config()
    tag = config.checkpoint.save_tag
    top_k = config.inference.stamp_gallery_top_k

    selected = _select_top_stamps(records, metadatas, top_k)
    if not selected:
        logger.info("Viz: no stamps available; skipping stamp_gallery")
        return None

    n_cols = len(selected)
    n_rows = len(_OBS_ROW_LABELS)
    fig = Figure(figsize=(1.9 * n_cols + 1.2, 1.1 * n_rows + 1.6))
    axes = fig.subplots(n_rows, n_cols, squeeze=False)

    for col, (record, idx, stat, freq) in enumerate(selected):
        try:
            snippet = _load_display_cadence(record.npy_path, idx)
        except Exception as e:
            logger.warning(f"Viz: failed to load stamp {idx} from {record.npy_path}: {e}")
            for row in range(n_rows):
                axes[row][col].set_axis_off()
            continue
        _draw_cadence_strip([axes[row][col] for row in range(n_rows)], snippet, label_rows=col == 0)
        axes[0][col].set_title(
            f"$k^2$={stat:.3g}\n{freq:.4f} MHz\n{_key_label(record.key, 20)}", fontsize=7
        )

    fig.suptitle(f"Top-{n_cols} energy-detection stamps by statistic ({tag})")

    return _save_and_upload(fig, f"stamp_gallery_{tag}.png", "Stamp Gallery")


def plot_preproc_funnel(records: list[CadenceVizRecord], metadatas: dict[str, dict]) -> str | None:
    """Per-cadence preprocessing funnel: raw hits → merged hits → stamps (incl. overlap
    offsets) → snippets inferred, plus per-cadence stamp storage annotated on top."""
    tag = get_config().checkpoint.save_tag

    labels: list[str] = []
    stage_counts: list[tuple[int, int, int, int]] = []
    storage_gb: list[float] = []
    for record in records:
        metadata = metadatas.get(record.npy_path) or {}
        n_raw = int(metadata.get("n_raw_hits") or 0)
        n_merged = int(metadata.get("n_merged_hits") or 0)
        n_stamps = len(metadata.get("stamp_starts") or []) or record.n_stamps
        labels.append(_key_label(record.key))
        stage_counts.append((n_raw, n_merged, n_stamps, record.n_stamps))
        try:
            storage_gb.append(os.path.getsize(record.npy_path) / 1e9)
        except OSError:
            storage_gb.append(float("nan"))

    if not stage_counts:
        logger.info("Viz: no cadences recorded; skipping preproc_funnel")
        return None

    stages = ("raw hits", "merged hits", "stamps (+overlap)", "snippets inferred")
    counts = np.asarray(stage_counts, dtype=np.float64)  # (n_cadences, 4)
    n_cadences = counts.shape[0]
    x = np.arange(n_cadences)
    bar_width = 0.8 / len(stages)

    fig = Figure(figsize=(max(8.0, 1.8 * n_cadences), 6))
    ax = fig.subplots()
    for stage_idx, stage in enumerate(stages):
        ax.bar(
            x + (stage_idx - (len(stages) - 1) / 2) * bar_width,
            counts[:, stage_idx],
            width=bar_width,
            label=stage,
        )
    for i in range(n_cadences):
        top = np.nanmax(counts[i]) if np.any(np.isfinite(counts[i])) else 0
        ax.annotate(
            f"{storage_gb[i]:.2f} GB",
            xy=(x[i], top),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("count")
    ax.set_title(f"Preprocessing funnel per cadence ({tag}) — stamp storage annotated")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)

    return _save_and_upload(fig, f"preproc_funnel_{tag}.png", "Preprocessing Funnel")


# ---------------------------------------------------------------------------
# Per-run figures: inference / anomaly detection
# ---------------------------------------------------------------------------


def plot_confidence_distribution(records: list[CadenceVizRecord]) -> str | None:
    """P(true) histogram over all snippets inferred this pass (log-y), classification
    threshold marked, per-cadence overlay when the pass covered ≤ 10 cadences. Cadences
    skipped by the resume are excluded (their confidence vectors were transient)."""
    config = get_config()
    tag = config.checkpoint.save_tag
    threshold = config.inference.classification_threshold

    with_hist = [r for r in records if r.confidence_hist is not None]
    if not with_hist:
        logger.info("Viz: no confidence histograms collected; skipping confidence_distribution")
        return None

    centers = (CONFIDENCE_HIST_EDGES[:-1] + CONFIDENCE_HIST_EDGES[1:]) / 2
    total = np.sum([r.confidence_hist for r in with_hist], axis=0)

    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    ax.step(centers, total, where="mid", lw=1.8, color="black", label="all snippets")
    if len(with_hist) <= 10:
        for record in with_hist:
            ax.step(
                centers,
                record.confidence_hist,
                where="mid",
                lw=0.9,
                alpha=0.7,
                label=_key_label(record.key, 22),
            )
    ax.axvline(threshold, color="red", ls="--", lw=1.2, label=f"threshold ({threshold:g})")
    ax.set_yscale("log")
    ax.set_xlabel("P(true) — Random Forest")
    ax.set_ylabel("snippet count")
    n_skipped = len(records) - len(with_hist)
    subtitle = f"{int(total.sum()):,} snippets over {len(with_hist)} cadence(s)"
    if n_skipped:
        subtitle += f" ({n_skipped} cadence(s) resumed earlier, not shown)"
    ax.set_title(f"Snippet confidence distribution ({tag})\n{subtitle}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    return _save_and_upload(fig, f"confidence_distribution_{tag}.png", "Confidence Distribution")


def _candidate_annotation(row: dict) -> str:
    lines = [f"confidence: {row.get('confidence', float('nan')):.4f}"]
    if row.get("frequency_mhz") is not None:
        lines.append(f"frequency: {row['frequency_mhz']:.6f} MHz")
    for label in ("target", "session", "band", "cadence_id"):
        if row.get(label) is not None:
            lines.append(f"{label}: {row[label]}")
    if row.get("timestamp_observed") is not None:
        lines.append(f"tstart (MJD): {row['timestamp_observed']:.5f}")
    if row.get("h5_path"):
        lines.append(f"h5: {os.path.basename(str(row['h5_path']))}")
    lines.append(f"npy: {os.path.basename(str(row.get('npy_path', '')))}")
    lines.append(f"snippet: {row.get('snippet_index')}")
    return "\n".join(lines)


def plot_candidate(row: dict, index: int) -> str | None:
    """One candidate's full picture (implements the long-standing inference.py stub):
    6-panel cadence waterfall of its stamp, annotated with confidence / frequency /
    target / session / band, plus the 48-dim latent vector as a bar chart."""
    tag = get_config().checkpoint.save_tag

    snippet = _load_display_cadence(str(row["npy_path"]), int(row["snippet_index"]))
    n_obs = snippet.shape[0]

    fig = Figure(figsize=(11, 7))
    grid = fig.add_gridspec(n_obs, 2, width_ratios=(2.2, 1.6), hspace=0.15, wspace=0.25, right=0.97)
    waterfall_axes = [fig.add_subplot(grid[i, 0]) for i in range(n_obs)]
    _draw_cadence_strip(waterfall_axes, snippet, label_rows=True)
    waterfall_axes[-1].set_xlabel("frequency bin")

    # Right column: latent bar chart on top, provenance text below. A nested gridspec gives
    # the two panels a dedicated vertical gap so the bar chart's x-axis label can never
    # collide with the first metadata line; the text panel takes the taller share so all of
    # the provenance lines stay clear of the axis label regardless of how many there are.
    right_grid = grid[:, 1].subgridspec(2, 1, height_ratios=(1.0, 1.3), hspace=0.55)
    latent_ax = fig.add_subplot(right_grid[0])
    latent_json = row.get("latent_vector")
    if latent_json:
        latent = np.asarray(json.loads(latent_json), dtype=np.float64).ravel()
        latent_dim = latent.size // n_obs if latent.size % n_obs == 0 else latent.size
        colors = [f"C{(i // latent_dim) % 10}" for i in range(latent.size)]
        latent_ax.bar(np.arange(latent.size), latent, color=colors)
        latent_ax.set_xlabel("latent dimension (colored per observation)", fontsize=8)
        latent_ax.set_ylabel("z", fontsize=8)
        latent_ax.tick_params(labelsize=7)
        latent_ax.grid(True, axis="y", alpha=0.2)
    else:
        latent_ax.set_axis_off()
        latent_ax.text(0.5, 0.5, "no latent vector stored", ha="center", va="center")

    text_ax = fig.add_subplot(right_grid[1])
    text_ax.set_axis_off()
    text_ax.text(
        0.0, 1.0, _candidate_annotation(row), va="top", ha="left", fontsize=9, family="monospace"
    )

    fig.suptitle(f"Candidate {index} ({tag}) — P(true) = {row.get('confidence', 0):.4f}")

    return _save_and_upload(fig, f"candidate_{index}_{tag}.png", f"Candidate {index}")


def plot_candidate_gallery() -> str | None:
    """Gallery of the top candidates by confidence (6-obs waterfall strips) plus capped
    per-candidate figures, sourced from the inference_results table so it also covers
    cadences the resume skipped this pass."""
    config = get_config()
    tag = config.checkpoint.save_tag
    max_candidate_plots = config.inference.max_candidate_plots

    db = get_db()
    if db is None:
        logger.info("Viz: no database instance; skipping candidate plots")
        return None
    db.flush()
    rows = db.query_inference_result(tag=tag, prediction=1)
    if not rows:
        logger.info("Viz: no candidates recorded; skipping candidate plots")
        return None
    rows.sort(key=lambda r: r.get("confidence") or 0.0, reverse=True)

    # Per-candidate figures (highest confidence first, capped)
    for index, row in enumerate(rows[:max_candidate_plots]):
        _viz_safe(f"candidate_{index}", plot_candidate, row, index)
    if len(rows) > max_candidate_plots:
        logger.info(
            f"Viz: rendered {max_candidate_plots} of {len(rows)} candidate figures "
            f"(--max-candidate-plots cap)"
        )

    gallery_rows = rows[:_CANDIDATE_GALLERY_MAX]
    n_cols = len(gallery_rows)
    n_rows = len(_OBS_ROW_LABELS)
    fig = Figure(figsize=(1.9 * n_cols + 1.2, 1.1 * n_rows + 1.6))
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    for col, row in enumerate(gallery_rows):
        try:
            snippet = _load_display_cadence(str(row["npy_path"]), int(row["snippet_index"]))
        except Exception as e:
            logger.warning(f"Viz: failed to load candidate stamp ({e})")
            for r in range(n_rows):
                axes[r][col].set_axis_off()
            continue
        _draw_cadence_strip([axes[r][col] for r in range(n_rows)], snippet, label_rows=col == 0)
        freq = row.get("frequency_mhz")
        freq_label = f"{freq:.4f} MHz" if freq is not None else "freq n/a"
        axes[0][col].set_title(
            f"P={row.get('confidence', 0):.3f}\n{freq_label}\n{row.get('target') or ''}",
            fontsize=7,
        )
    fig.suptitle(f"Candidate gallery ({tag}): top {n_cols} of {len(rows)} by confidence")

    return _save_and_upload(fig, f"candidate_gallery_{tag}.png", "Candidate Gallery")


def plot_inference_latent_projection(collector: InferenceVizCollector) -> str | None:
    """Project this run's cadence-level latent features through the persisted cadence-level
    UMAP from the training run (located via the training config JSON's model_path +
    save_tag), over a faint background of the UMAP's training embedding. Answers "where does
    real data live relative to the synthetic training classes". Skips gracefully (with a
    log) when the training config or a persisted UMAP is unavailable."""
    import joblib  # noqa: PLC0415  # deferred: only this figure needs it

    config = get_config()
    tag = config.checkpoint.save_tag

    features, is_candidate = collector.latent_pool()
    if features.size == 0:
        logger.info("Viz: no latent features collected this pass; skipping latent projection")
        return None

    training_config_path = config.inference.config_path
    if not training_config_path or not os.path.exists(training_config_path):
        logger.info(
            f"Viz: training config JSON not available ({training_config_path}); "
            f"skipping latent projection"
        )
        return None
    try:
        with open(training_config_path) as f:
            training_config = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.info(f"Viz: could not read training config ({e}); skipping latent projection")
        return None

    model_path = (training_config.get("paths") or {}).get("model_path")
    train_tag = (training_config.get("checkpoint") or {}).get("save_tag")
    training_section = training_config.get("training") or {}
    nn_values = training_section.get("latent_viz_umap_n_neighbors") or [15]
    md_values = training_section.get("latent_viz_umap_min_dist") or [0.1]
    if not model_path or not train_tag:
        logger.info("Viz: training config lacks model_path/save_tag; skipping latent projection")
        return None

    umap_path = None
    nn = md = None
    for candidate_nn in nn_values:
        for candidate_md in md_values:
            path = os.path.join(
                model_path, f"umap_cadence_nn{candidate_nn}_md{candidate_md}_{train_tag}.joblib"
            )
            if os.path.exists(path):
                umap_path, nn, md = path, candidate_nn, candidate_md
                break
        if umap_path:
            break
    if umap_path is None:
        logger.info(
            f"Viz: no persisted cadence-level UMAP found under {model_path} for tag "
            f"{train_tag}; skipping latent projection"
        )
        return None

    logger.info(f"Viz: projecting {features.shape[0]} latent features through {umap_path}")
    umap_model = joblib.load(umap_path)
    embedding = umap_model.transform(features)

    fig = Figure(figsize=(9, 8))
    ax = fig.subplots()
    # The training fit pool's class labels are not persisted, so the training embedding is
    # rendered as an unlabeled backdrop — it still shows where the synthetic classes live.
    background = getattr(umap_model, "embedding_", None)
    if background is not None and len(background):
        ax.scatter(
            background[:, 0],
            background[:, 1],
            s=3,
            c="lightgray",
            alpha=0.35,
            linewidths=0,
            label=f"training embedding ({train_tag})",
            rasterized=True,
        )
    non_candidates = ~is_candidate
    if non_candidates.any():
        ax.scatter(
            embedding[non_candidates, 0],
            embedding[non_candidates, 1],
            s=6,
            c="tab:blue",
            alpha=0.5,
            linewidths=0,
            label=f"snippets ({int(non_candidates.sum()):,})",
            rasterized=True,
        )
    if is_candidate.any():
        ax.scatter(
            embedding[is_candidate, 0],
            embedding[is_candidate, 1],
            s=42,
            c="red",
            marker="*",
            edgecolors="black",
            linewidths=0.4,
            label=f"candidates ({int(is_candidate.sum())})",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        f"Inference latents through training cadence-level UMAP ({tag})\n"
        f"nn={nn}, md={md}, training tag {train_tag}"
    )
    ax.legend(fontsize=8, loc="best")

    return _save_and_upload(
        fig, f"inference_latent_projection_{tag}.png", "Inference Latent Projection"
    )


def plot_inference_summary(
    records: list[CadenceVizRecord], metadatas: dict[str, dict], totals: dict
) -> str | None:
    """Table-style run summary card: cadence/snippet/candidate counts, per-stage durations
    and throughput from the inference_cadences manifest, and per-target/band candidate
    counts from inference_results."""
    config = get_config()
    tag = config.checkpoint.save_tag

    db = get_db()
    preproc_duration = inference_duration = 0.0
    if db is not None:
        db.flush()
        # NOTE: include_superseded=True is required here: _infer_cadence supersedes a
        # cadence's 'preprocessed' row before writing its live 'inferred' row, so on a fully
        # successful run every 'preprocessed' row is superseded and the default query would
        # hide it (preprocessing time would always read 0.0 s). Summing per status keeps each
        # metric on its own row — no double-counting between the 'preprocessed' and 'inferred'
        # rows of the same cadence.
        for row in db.query_inference_cadences(tag=tag, include_superseded=True):
            duration = row.get("duration_s") or 0.0
            if row.get("status") == "preprocessed":
                preproc_duration += duration
            elif row.get("status") == "inferred":
                inference_duration += duration

    n_snippets = int(totals.get("n_cadence_snippets", 0))
    n_raw_hits = sum(int((m or {}).get("n_raw_hits") or 0) for m in metadatas.values())
    n_merged_hits = sum(int((m or {}).get("n_merged_hits") or 0) for m in metadatas.values())
    storage_gb = 0.0
    for record in records:
        with contextlib.suppress(OSError):
            storage_gb += os.path.getsize(record.npy_path) / 1e9

    summary_rows = [
        ("run tag", tag),
        ("rendered", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("cadences", f"{totals.get('n_cadences', len(records))}"),
        ("  resumed (skipped this pass)", f"{totals.get('n_skipped', 0)}"),
        ("raw ED hits", f"{n_raw_hits:,}"),
        ("merged hits", f"{n_merged_hits:,}"),
        ("snippets inferred", f"{n_snippets:,}"),
        ("candidates", f"{totals.get('n_candidates', 0):,}"),
        ("stamp storage", f"{storage_gb:.2f} GB"),
        ("preprocessing time", f"{preproc_duration:,.1f} s"),
        ("inference time", f"{inference_duration:,.1f} s"),
        (
            "throughput",
            f"{n_snippets / inference_duration:,.1f} snippets/s"
            if inference_duration > 0
            else "n/a",
        ),
    ]

    per_group: Counter = Counter()
    if db is not None:
        candidate_rows = db.query_inference_result(
            tag=tag, prediction=1, columns=["target", "band"]
        )
        for row in candidate_rows:
            per_group[(row.get("target") or "?", row.get("band") or "?")] += 1

    fig = Figure(figsize=(8.5, 0.42 * (len(summary_rows) + min(len(per_group), 10)) + 2.5))
    ax = fig.subplots()
    ax.set_axis_off()

    lines = [f"{name:<32} {value}" for name, value in summary_rows]
    if per_group:
        lines.append("")
        lines.append("candidates per target/band:")
        for (target, band), count in per_group.most_common(10):
            lines.append(f"  {target} [{band}]: {count}")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        transform=ax.transAxes,
    )
    ax.set_title(f"Inference run summary ({tag})", fontsize=13, pad=14)

    return _save_and_upload(fig, f"inference_summary_{tag}.png", "Inference Summary")


# ---------------------------------------------------------------------------
# Suite entry point
# ---------------------------------------------------------------------------


def render_inference_visualizations(
    collector: InferenceVizCollector, preprocessor, totals: dict
) -> None:
    """Render every figure of the suite, each individually exception-guarded. Called by
    main._run_streaming_csv_inference after a fully successful pass (and gated on
    config.inference.inference_viz_enabled by the caller)."""
    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")
    tag = config.checkpoint.save_tag
    logger.info(f"Rendering inference visualization suite under plots/inference/{tag}/")

    records = collector.records
    metadatas: dict[str, dict] = {}
    for record in records:
        metadata = _load_metadata(record)
        if metadata is not None:
            metadatas[record.npy_path] = metadata

    _viz_safe("ed_stat_distributions", plot_ed_stat_distributions, records, metadatas)
    _viz_safe("ed_hit_spectrum", plot_ed_hit_spectrum, records, metadatas)
    _viz_safe("bandpass_flattening", plot_bandpass_flattening, preprocessor, records, metadatas)
    _viz_safe("stamp_gallery", plot_stamp_gallery, records, metadatas)
    _viz_safe("preproc_funnel", plot_preproc_funnel, records, metadatas)
    _viz_safe("confidence_distribution", plot_confidence_distribution, records)
    _viz_safe("candidate_gallery", plot_candidate_gallery)
    _viz_safe("inference_latent_projection", plot_inference_latent_projection, collector)
    _viz_safe("inference_summary", plot_inference_summary, records, metadatas, totals)

    logger.info("Inference visualization suite complete")
