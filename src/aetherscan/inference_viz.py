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

import json
import logging
import os
import queue
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import h5py
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from aetherscan.benchmark import stage_timer
from aetherscan.candidate_figures import (
    OBS_ROW_LABELS,
    candidate_frequency_range_mhz,
    draw_cadence_strip,
    load_display_cadence,
    render_candidate_figure,
    render_candidate_figures,
    stamp_frequency_range_mhz,
)
from aetherscan.candidate_triage import (
    frequency_excluded,
    partition_candidates_by_frequency,
    report_exclusion_ranges,
    survey_ood_scores,
    training_ood_scores,
    triage_sort_rows,
)
from aetherscan.config import get_config
from aetherscan.data_generation import log_norm
from aetherscan.db import get_db, get_machine_name
from aetherscan.display_tag import display_tag
from aetherscan.logger import get_logger
from aetherscan.pfb import gen_coarse_channel_response
from aetherscan.seeding import STREAM_INFERENCE_VIZ, derive_rng

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

# Fine-grid bin count used when a LEGACY sidecar's raw hit-frequency lists are reduced to
# a bounded per-cadence histogram at metadata-load time (#301) — new sidecars arrive
# pre-binned by preprocessing (_HIT_HIST_BINS, same value).
_LEGACY_HIT_HIST_BINS = 8192

# Cap on individually-drawn cadences in the preprocessing funnel (#301): at 270 px per
# cadence (1.8 in x 150 dpi) the figure exceeds Agg's hard 2^16-px canvas limit past 242
# cadences — the render then raises and _viz_safe swallowed it, so every catalog-scale
# pass silently LOST the funnel. Beyond the cap, the highest-raw-hit cadences are drawn
# individually and the remainder is aggregated into one summary bar.
_FUNNEL_MAX_CADENCES = 120

# Candidate gallery shows at most this many top-confidence candidates (per-candidate figures
# are governed separately by config.inference.max_candidate_plots).
_CANDIDATE_GALLERY_MAX = 12

# ON/OFF row labels for 6-observation ABACAD cadence strips — canonical definition lives in
# the TF-free candidate_figures module (#298 I9); re-exported here for the suite's figures.
_OBS_ROW_LABELS = OBS_ROW_LABELS

# Bounded wait for the async Slack uploader's drain at suite end: uploads are best-effort
# (same contract as _viz_safe), so a stuck Slack API must not hold the run open forever.
_UPLOAD_DRAIN_TIMEOUT_S = 180.0


class _AsyncUploader:
    """
    Single-worker FIFO Slack uploader (#298 I9): fig.savefig stays on the caller; the 3-4
    HTTP round trips per figure — plus any retry-backoff sleeps on a flaky Slack — leave
    the render critical path. ONE worker, deliberately: it preserves the suite's
    Slack-thread figure ordering and keeps the effective API rate identical to the old
    synchronous path. drain() must run before logger teardown;
    render_inference_visualizations guarantees it in a finally.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def submit(self, save_path: str, title: str) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="viz_slack_upload", daemon=True)
            self._thread.start()
        self._queue.put((save_path, title))

    def _run(self) -> None:
        # Capture THIS thread's stop event: drain() re-arms self._stop for any successor
        # thread, and a timed-out predecessor must keep honoring its own signal.
        stop = self._stop
        while True:
            item = self._queue.get()
            if item is None:
                return
            if stop.is_set():
                # Timed-out drain: discard the remaining backlog and spin to the sentinel
                # so the thread exits promptly after its in-flight upload (#298 review
                # note — no abandoned worker plodding through a dead queue)
                continue
            save_path, title = item
            try:
                logger_instance = get_logger()
                if logger_instance:
                    logger_instance.upload_image_to_slack(save_path, title=title)
            except Exception as e:
                logger.error(f"Async Slack upload failed for {save_path}: {e}")

    def drain(self) -> None:
        """Flush the queue and stop the worker (bounded — uploads are best-effort). On
        timeout the worker is signalled to discard its backlog and exit after the in-flight
        upload, rather than being abandoned mid-queue."""
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=_UPLOAD_DRAIN_TIMEOUT_S)
        if self._thread.is_alive():
            self._stop.set()
            logger.warning(
                f"Async Slack uploader did not drain within {_UPLOAD_DRAIN_TIMEOUT_S:.0f}s; "
                f"remaining uploads discarded once the in-flight call returns (figures are "
                f"on disk; the daemon thread cannot block shutdown)"
            )
        self._thread = None
        self._stop = threading.Event()


_uploader = _AsyncUploader()


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

    def __init__(
        self,
        max_latent_points: int = _MAX_LATENT_POINTS,
        gallery_pool: list | None = None,
    ):
        self.records: list[CadenceVizRecord] = []
        self._max_latent_points = max_latent_points
        self._latent_chunks: list[np.ndarray] = []  # (k, num_obs * latent_dim) each
        self._candidate_chunks: list[np.ndarray] = []  # bool masks aligned with chunks
        self._latent_count = 0
        config = get_config()
        root_seed = config.reproducibility.seed if config is not None else None
        self._rng = derive_rng(root_seed, STREAM_INFERENCE_VIZ)
        # Bounded top-K stamp-pixel pool (#302): raw pixels of the strongest stamps seen
        # so far, captured just before pruning deletes a cadence's .npy, so the stamp
        # gallery stays whole (~196 KB x top_k ≈ 2.4 MB at defaults). A caller-supplied
        # list persists it ACROSS the in-process retry attempts (#305 fix): each attempt
        # builds a fresh collector, but a cadence pruned in an earlier attempt is
        # resume-skipped (never re-pooled) in later ones, so a per-attempt pool would blank
        # its gallery column on the retry render. Cross-PROCESS relaunch still can't recover
        # those pixels (documented) — but a full-catalog run rarely relaunches.
        self._gallery_top_k = config.inference.stamp_gallery_top_k if config is not None else 12
        self._gallery_pixel_pool: list[tuple[float, str, int, np.ndarray]] = (
            gallery_pool if gallery_pool is not None else []
        )

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
        keep = np.nonzero(is_candidate)[0]
        budget = self._max_latent_points - self._latent_count - keep.size
        non_candidates = np.nonzero(~is_candidate)[0]
        if budget > 0 and non_candidates.size > 0:
            if non_candidates.size > budget:
                non_candidates = self._rng.choice(non_candidates, size=budget, replace=False)
            keep = np.concatenate([keep, non_candidates])

        if keep.size == 0:
            # Once the global budget is spent (a cadence or two into the catalog), every
            # non-candidate cadence lands here — so nothing may be built before this
            # return. The full-cadence feature matrix this method used to construct first
            # (~190 MB float64 for a 330k-stamp cadence) was thrown away each time (#301).
            return
        # Cadence-level rows for the kept indices only. Value-identical to
        # prepare_latent_features(latents)[keep].astype(np.float32): the helper's
        # float32 -> float64 -> float32 round trip is exact, and the row-major
        # reshape is the helper's own documented layout (fancy indexing yields a
        # fresh array, so the chunk owns its memory).
        features = np.asarray(latents, dtype=np.float32).reshape(len(is_candidate), -1)[keep]
        self._latent_chunks.append(features)
        self._candidate_chunks.append(is_candidate[keep])
        self._latent_count += keep.size

    def latent_pool(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (features, is_candidate) stacked over everything collected; empty arrays
        when nothing was processed this pass."""
        if not self._latent_chunks:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=bool)
        return np.concatenate(self._latent_chunks), np.concatenate(self._candidate_chunks)

    def pool_gallery_pixels(self, metadata_path: str, npy_path: str) -> None:
        """Capture this cadence's top-K stamp pixels into the bounded global pool (#302):
        called by the pruning step just before it deletes the stamp .npy. The global
        top-K is a subset of the per-cadence top-Ks, so pooling per-cadence top-K and
        truncating globally preserves exactly the stamps plot_stamp_gallery will select.
        Best-effort — a failure degrades the gallery, never the run."""
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            representatives = _cadence_top_stamps(metadata, self._gallery_top_k)
            if not representatives:
                return
            stamps = np.load(npy_path, mmap_mode="r")
            for stat, idx, _freq, _freq_range in representatives:
                self._gallery_pixel_pool.append(
                    (stat, npy_path, int(idx), np.array(stamps[int(idx)], dtype=np.float32))
                )
            del stamps
            self._gallery_pixel_pool.sort(key=lambda entry: entry[0], reverse=True)
            del self._gallery_pixel_pool[self._gallery_top_k :]
        except Exception as e:
            logger.warning(
                f"Gallery pixel pooling failed for {npy_path} ({e}); the stamp gallery "
                f"may show blank columns for this cadence"
            )

    def gallery_pixels(self) -> dict[tuple[str, int], np.ndarray]:
        """(npy_path, snippet_index) -> raw stamp pixels for the pooled top-K stamps."""
        return {(path, idx): pixels for _, path, idx, pixels in self._gallery_pixel_pool}


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def _viz_safe(name: str, fn: Callable, *args, **kwargs):
    """Run one figure function, log-and-swallow any exception: a plot bug must never kill a
    science run (mirrors train.py's _safe_call, with viz-specific logging). Each figure
    records its own pipeline_stages sub-span (#301) — the suite used to be one opaque
    'inference.viz' span covering 73% of a cached rerun's wall."""
    try:
        with stage_timer(name):
            return fn(*args, **kwargs)
    except Exception as e:
        logger.error(f"Inference viz '{name}' failed (run continues without it): {e}")
        return None


def _reference_cloud_path(config, tag: str) -> str:
    """The run's reference-cloud NPZ path (written by finalize_reference_cloud)."""
    return os.path.join(
        config.output_path,
        f"inference_reference_cloud_{display_tag(tag, get_machine_name())}.npz",
    )


def _plots_dir(tag: str) -> str:
    path = os.path.join(
        get_config().output_path, "plots", "inference", display_tag(tag, get_machine_name())
    )
    os.makedirs(path, exist_ok=True)
    return path


def _save_and_upload(fig: Figure, filename: str, slack_title: str) -> str:
    """Save a figure under plots/inference/{tag}/ and queue its Slack upload (#298 I9 —
    the upload's HTTP round trips run on the async FIFO uploader, drained before teardown
    by render_inference_visualizations). Returns the saved path."""
    tag = get_config().checkpoint.save_tag
    save_path = os.path.join(_plots_dir(tag), filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    # Eagerly release the figure's artists/render buffers. OO-API Figures aren't tracked by
    # a global registry, so this isn't a leak fix — it just frees the backing memory now
    # instead of at garbage collection (relevant for dense stamp/candidate galleries). The
    # Slack upload reads the saved PNG, not the figure.
    fig.clear()
    logger.info(f"Inference viz saved: {save_path}")

    _uploader.submit(save_path, f"{slack_title} - ({display_tag(tag, get_machine_name())})")
    return save_path


def _load_metadata(record: CadenceVizRecord) -> dict | None:
    """Best-effort read of a cadence's metadata JSON (durable ED provenance)."""
    try:
        with open(record.metadata_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Viz: could not read metadata {record.metadata_path} ({e}); skipping")
        return None


@dataclass
class CadenceVizSummary:
    """Bounded per-cadence reduction of the metadata sidecar (#301). The suite used to
    hold every cadence's fully parsed JSON (up to ~19 MB each on RFI-dense cadences —
    an OOM-class hundreds of GB at 6,093-cadence catalog scale) for the whole render;
    each figure now reads these ~KB summaries and the parsed dict is dropped as soon as
    its cadence is reduced."""

    n_raw_hits: int = 0
    n_merged_hits: int = 0
    n_stamp_rows: int = 0
    npy_size_bytes: float = float("nan")
    ed_hist_edges: np.ndarray | None = None
    ed_hist_counts: np.ndarray | None = None  # (n_on_files, n_bins) int64
    # {raw_counts, merged_counts, freq_lo, freq_hi, raw_freq_min, raw_freq_max} — fine
    # per-cadence histograms rebinned onto the figure's global axis at render time
    hit_hist: dict | None = None
    h5_path: str | None = None
    nchans: int = 0
    bandpass_envelopes: list | None = None
    # Per-cadence top-K stamp representatives: (statistic, snippet_idx, freq, freq_range)
    gallery_reps: list = field(default_factory=list)


def _reduce_hit_hist(metadata: dict) -> dict | None:
    """Normalize a cadence's hit-frequency data to a bounded fine histogram: new sidecars
    carry hit_spectrum_hist pre-binned by preprocessing; legacy sidecars' raw float lists
    are binned HERE, once, on their own span — so RAM stays bounded for old and new
    catalogs alike (#301)."""
    hist = metadata.get("hit_spectrum_hist")
    if hist:
        return {
            "raw_counts": np.asarray(hist["raw_counts"], dtype=np.int64),
            "merged_counts": np.asarray(hist["merged_counts"], dtype=np.int64),
            "freq_lo": float(hist["freq_lo"]),
            "freq_hi": float(hist["freq_hi"]),
            "raw_freq_min": float(hist["raw_freq_min"]),
            "raw_freq_max": float(hist["raw_freq_max"]),
        }
    raw = metadata.get("raw_hit_frequencies_mhz") or []
    if not raw:
        return None
    merged = metadata.get("merged_hit_frequencies_mhz") or []
    lo, hi = float(np.min(raw)), float(np.max(raw))
    span_lo, span_hi = (lo, hi) if lo < hi else (lo - 0.5, hi + 0.5)
    edges = np.linspace(span_lo, span_hi, _LEGACY_HIT_HIST_BINS + 1)
    return {
        "raw_counts": np.histogram(raw, bins=edges)[0],
        "merged_counts": np.histogram(merged, bins=edges)[0],
        "freq_lo": span_lo,
        "freq_hi": span_hi,
        "raw_freq_min": lo,
        "raw_freq_max": hi,
    }


def _cadence_top_stamps(metadata: dict, top_k: int) -> list[tuple[float, int, float, tuple | None]]:
    """This cadence's top-K stamp representatives by detection statistic, overlap-offset
    copies collapsed first (the per-cadence half of the old _select_top_stamps): stamps
    sharing one exact statistic are one hit's offset copies — represented by the
    median-start stamp (the offset-0 center for a full triplet). Keeping only the
    per-cadence top-K preserves the global top-K exactly (each cadence can contribute at
    most K entries to it) while bounding the reduction (#301)."""
    stats_list = metadata.get("stamp_statistics") or []
    freqs = metadata.get("stamp_frequencies_mhz") or []
    starts = metadata.get("stamp_starts") or []

    by_stat: dict[float, list[tuple[int, int]]] = {}
    for idx, stat in enumerate(stats_list):
        start = int(starts[idx]) if idx < len(starts) else 0
        by_stat.setdefault(float(stat), []).append((start, idx))

    representatives: list[tuple[float, int, float, tuple | None]] = []
    for stat, members in by_stat.items():
        members.sort()  # by start; median = offset-0 center for a full triplet
        _, idx = members[len(members) // 2]
        freq = float(freqs[idx]) if idx < len(freqs) else float("nan")
        representatives.append((stat, idx, freq, stamp_frequency_range_mhz(metadata, idx)))

    # Distinct stats per cadence (offset copies collapsed), so this sort has no ties and
    # cross-cadence tie order stays the stable record order the old global sort had
    representatives.sort(key=lambda r: r[0], reverse=True)
    return representatives[:top_k]


def _reduce_metadata(
    record: CadenceVizRecord, metadata: dict, gallery_top_k: int
) -> CadenceVizSummary:
    """Reduce one cadence's parsed metadata to the bounded summary the figures consume."""
    summary = CadenceVizSummary()
    summary.n_raw_hits = int(metadata.get("n_raw_hits") or 0)
    summary.n_merged_hits = int(metadata.get("n_merged_hits") or 0)
    summary.n_stamp_rows = len(metadata.get("stamp_starts") or []) or record.n_stamps
    try:
        summary.npy_size_bytes = float(os.path.getsize(record.npy_path))
    except OSError:
        # The .npy was pruned after scoring (#302 default) — reconstruct the size the run
        # transiently held from the stored geometry so the funnel/summary storage figures
        # don't silently collapse to nan/0 on every default run. The .npy shape is
        # (n_stamps, n_obs, time_bins, stored_width) float32 + a ~128-byte header.
        config = get_config()
        stored_width = metadata.get("stored_width")
        time_bins = config.data.time_bins if config is not None else None
        n_obs = len(metadata.get("h5_paths") or []) or 6
        if stored_width and time_bins:
            summary.npy_size_bytes = float(
                summary.n_stamp_rows * n_obs * int(time_bins) * int(stored_width) * 4 + 128
            )
    ed_hist = metadata.get("ed_stat_hist")
    if ed_hist:
        summary.ed_hist_edges = np.asarray(ed_hist["bin_edges"], dtype=np.float64)
        summary.ed_hist_counts = np.asarray(ed_hist["counts_per_on_file"], dtype=np.int64)
    summary.hit_hist = _reduce_hit_hist(metadata)
    h5_paths = metadata.get("h5_paths") or []
    if h5_paths:
        summary.h5_path = h5_paths[0]
        summary.nchans = int((metadata.get("header") or {}).get("nchans", 0) or 0)
    summary.bandpass_envelopes = metadata.get("bandpass_envelopes") or None
    summary.gallery_reps = _cadence_top_stamps(metadata, gallery_top_k)
    return summary


def _build_summaries(
    records: list[CadenceVizRecord], gallery_top_k: int
) -> dict[str, CadenceVizSummary]:
    """One pass over the records' metadata sidecars: parse, reduce, drop — the only place
    the suite touches the raw JSONs (#301)."""
    summaries: dict[str, CadenceVizSummary] = {}
    envelopes_captured = False
    for record in records:
        metadata = _load_metadata(record)
        if metadata is not None:
            summary = _reduce_metadata(record, metadata, gallery_top_k)
            # Only plot_bandpass_flattening reads bandpass_envelopes, and it uses the FIRST
            # cadence (in records order) that carries them, then returns — so retaining every
            # cadence's envelopes (each ~hundreds of KB of parsed floats) for the whole render
            # is ~GB of dead RAM at catalog scale, undoing the O(1)-per-cadence bound the
            # summaries were introduced (#301) to guarantee. Keep them on the first summary
            # that has them and drop the rest: _build_summaries iterates records in the same
            # order as the consumer, so the identical cadence is selected — byte-identical figure.
            if summary.bandpass_envelopes is not None:
                if envelopes_captured:
                    summary.bandpass_envelopes = None
                else:
                    envelopes_captured = True
            summaries[record.npy_path] = summary
        del metadata
    return summaries


# _load_display_cadence and _draw_cadence_strip moved to the TF-free candidate_figures
# module (#298 I9) so forkserver render workers can import them without pulling TF;
# the suite uses them via the load_display_cadence / draw_cadence_strip imports above.


def _key_label(key: tuple, max_len: int = 28) -> str:
    label = "/".join(str(part) for part in key)
    return label if len(label) <= max_len else label[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# Per-run figures: energy detection / preprocessing
# ---------------------------------------------------------------------------


def plot_ed_stat_distributions(
    records: list[CadenceVizRecord], summaries: dict[str, CadenceVizSummary]
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
        summary = summaries.get(record.npy_path)
        if summary is None or summary.ed_hist_edges is None:
            continue
        cadence_edges = summary.ed_hist_edges
        counts = summary.ed_hist_counts
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
    above = sum(summaries[p].n_raw_hits for p in contributing)
    total = int(per_on_totals.sum())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D'Agostino-Pearson $k^2$ statistic")
    ax.set_ylabel("window count")
    ax.set_title(
        f"Energy-detection statistic distribution ({display_tag(tag, get_machine_name())})\n"
        f"{total:,} finite windows, {above:,} above threshold "
        f"({len(contributing)} cadence(s))"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)

    return _save_and_upload(
        fig,
        f"ed_stat_distributions_{display_tag(tag, get_machine_name())}.png",
        "ED Statistic Distribution",
    )


def _clamp_hit_spectrum_bins(
    lo: float, hi: float, coarsest_fine_width: float, max_bins: int
) -> int:
    """Number of hit-spectrum figure bins over [lo, hi] that keeps each figure bin no finer
    than the coarsest stored fine grid (>= one fine bin per figure bin), so the rebin can't
    alias into a picket-fence (#305). Returns max_bins when the span is wide (the common
    case) and fewer as the span narrows toward the fine-grid resolution."""
    if coarsest_fine_width <= 0 or hi <= lo:
        return max_bins
    return max(1, min(max_bins, int((hi - lo) / coarsest_fine_width)))


def plot_ed_hit_spectrum(
    records: list[CadenceVizRecord], summaries: dict[str, CadenceVizSummary]
) -> str | None:
    """Hit density vs frequency (MHz) across the band, pre- vs post-deduplication — RFI comb
    structure shows up immediately as spikes/picket-fences. Rendered by rebinning each
    cadence's bounded fine histogram (pre-binned by preprocessing, or reduced from a legacy
    sidecar's raw lists at load — #301) onto one global axis whose bin count is clamped so
    the figure bins are never finer than the stored fine grid — so the result is visually
    faithful to histogramming the raw floats (identical for the wide-span common case)
    while never holding the catalog's hit lists in RAM."""
    tag = get_config().checkpoint.save_tag

    with_hits = [
        summaries[record.npy_path]
        for record in records
        if summaries.get(record.npy_path) and summaries[record.npy_path].hit_hist
    ]
    if not with_hits:
        logger.info("Viz: no hit frequencies available; skipping ed_hit_spectrum")
        return None

    lo = min(s.hit_hist["raw_freq_min"] for s in with_hits)
    hi = max(s.hit_hist["raw_freq_max"] for s in with_hits)
    if lo == hi:  # single-frequency degenerate range: give the histogram some width
        lo, hi = lo - 0.5, hi + 0.5

    # Clamp the figure's bin count so its bins are never FINER than the stored fine grid
    # over this span (#305 review): new sidecars pre-bin over the whole file band, so when
    # the catalog-wide hit span is narrow (< ~band/41) the 200-bin figure axis would be
    # finer than the fine bins and the center-weighted rebin below would draw a spurious
    # picket-fence (empty bins between populated ones) not present in the data.
    coarsest_fine_width = max(
        (s.hit_hist["freq_hi"] - s.hit_hist["freq_lo"]) / max(1, len(s.hit_hist["raw_counts"]))
        for s in with_hits
    )
    n_fig_bins = _clamp_hit_spectrum_bins(lo, hi, coarsest_fine_width, _HIT_SPECTRUM_BINS)
    bins = np.linspace(lo, hi, n_fig_bins + 1)

    raw_total = np.zeros(n_fig_bins, dtype=np.float64)
    merged_total = np.zeros(n_fig_bins, dtype=np.float64)
    for s in with_hits:
        h = s.hit_hist
        fine_edges = np.linspace(h["freq_lo"], h["freq_hi"], len(h["raw_counts"]) + 1)
        # Clip centers into the global range: a boundary hit's fine bin can center just
        # outside [lo, hi] — its count belongs in the edge bin, exactly where the raw
        # float would have landed
        centers = np.clip((fine_edges[:-1] + fine_edges[1:]) / 2, lo, hi)
        raw_total += np.histogram(centers, bins=bins, weights=h["raw_counts"])[0]
        merged_total += np.histogram(centers, bins=bins, weights=h["merged_counts"])[0]

    fig = Figure(figsize=(12, 5))
    ax = fig.subplots()
    ax.stairs(raw_total, bins, fill=True, alpha=0.35, label="raw hits (pre-dedup)")
    ax.stairs(merged_total, bins, lw=1.4, color="crimson", label="merged hits")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("hit count")
    ax.set_title(
        f"Energy-detection hit spectrum ({display_tag(tag, get_machine_name())})\n"
        f"{int(raw_total.sum()):,} raw → {int(merged_total.sum()):,} merged hits"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    return _save_and_upload(
        fig, f"ed_hit_spectrum_{display_tag(tag, get_machine_name())}.png", "ED Hit Spectrum"
    )


def _plot_bandpass_from_envelopes(
    envelopes: list[dict], h5_path: str | None, tag: str
) -> str | None:
    """Render the bandpass-flattening figure from the decimated envelope lines persisted
    in a cadence's metadata sidecar (#301): the same three lines per sampled channel the
    live path draws, computed by preprocessing while the channels were resident — no h5
    reads at viz time (they measured 114 s of cold /datag reads = 74% of a cached rerun's
    viz span)."""
    fig = Figure(figsize=(14, 3.2 * len(envelopes)))
    axes = fig.subplots(len(envelopes), 2, squeeze=False)
    overlay_label = "removed model"
    for row, entry in enumerate(envelopes):
        ax_raw, ax_flat = axes[row]
        overlay_label = entry.get("overlay_label") or overlay_label
        ax_raw.plot(
            entry["raw"]["idx"],
            entry["raw"]["values"],
            lw=0.6,
            color="tab:blue",
            label="raw integrated spectrum",
        )
        ax_raw.plot(
            entry["overlay"]["idx"],
            entry["overlay"]["values"],
            lw=1.2,
            ls="--",
            color="tab:orange",
            label=overlay_label,
        )
        ax_raw.set_ylabel(f"coarse channel {entry.get('channel', '?')}\nintegrated power")
        ax_flat.plot(
            entry["flat"]["idx"],
            entry["flat"]["values"],
            lw=0.6,
            color="tab:green",
            label="flattened integrated spectrum",
        )
        if row == 0:
            ax_raw.legend(loc="upper right", fontsize=8)
            ax_flat.legend(loc="upper right", fontsize=8)
        if row == len(envelopes) - 1:
            ax_raw.set_xlabel("fine channel (within coarse channel)")
            ax_flat.set_xlabel("fine channel (within coarse channel)")

    method = "pfb" if "PFB" in overlay_label else "spline"
    source = os.path.basename(h5_path) if h5_path else "stored envelopes"
    fig.suptitle(
        f"Bandpass flattening ({method}, {display_tag(tag, get_machine_name())}): {source}"
    )
    fig.tight_layout()
    return _save_and_upload(
        fig,
        f"bandpass_flattening_{display_tag(tag, get_machine_name())}.png",
        "Bandpass Flattening",
    )


def plot_bandpass_flattening(
    preprocessor, records: list[CadenceVizRecord], summaries: dict[str, CadenceVizSummary]
) -> str | None:
    """Integrated spectrum raw vs flattened for a few coarse channels sampled across the
    band of the first cadence's primary ON file, with the removed model (scaled PFB response
    H or spline fit) overlaid — formalizes PR-07's opt-in debug artifact as a standard
    per-run figure. Rendered from the envelopes persisted at preprocess time when a sidecar
    carries them (#301); legacy sidecars keep the historical live-read path below."""
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

    for record in records:
        summary = summaries.get(record.npy_path)
        if summary is not None and summary.bandpass_envelopes:
            return _plot_bandpass_from_envelopes(summary.bandpass_envelopes, summary.h5_path, tag)

    h5_path = None
    n_chans = 0
    for record in records:
        summary = summaries.get(record.npy_path)
        if summary is not None and summary.h5_path and os.path.exists(summary.h5_path):
            h5_path = summary.h5_path
            n_chans = summary.nchans
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
    fig.suptitle(
        f"Bandpass flattening ({method}, {display_tag(tag, get_machine_name())}): {os.path.basename(h5_path)}"
    )
    fig.tight_layout()

    return _save_and_upload(
        fig,
        f"bandpass_flattening_{display_tag(tag, get_machine_name())}.png",
        "Bandpass Flattening",
    )


def _select_top_stamps(
    records: list[CadenceVizRecord], summaries: dict[str, CadenceVizSummary], top_k: int
) -> list[tuple[CadenceVizRecord, int, float, float, tuple | None]]:
    """Pick the top_k stamps by detection statistic across all cadences, from the
    per-cadence top-K representatives the reduce pass computed (_cadence_top_stamps —
    overlap-offset copies already collapsed there). Selecting the global top-K from the
    per-cadence top-Ks is exact: each cadence can contribute at most K entries. Returns
    (record, snippet_index, statistic, frequency_mhz, freq_range) tuples, strongest
    first."""
    representatives: list[tuple[float, CadenceVizRecord, int, float, tuple | None]] = []
    for record in records:
        summary = summaries.get(record.npy_path)
        if summary is None:
            continue
        for stat, idx, freq, freq_range in summary.gallery_reps:
            representatives.append((stat, record, idx, freq, freq_range))

    representatives.sort(key=lambda c: c[0], reverse=True)
    return [
        (record, idx, stat, freq, freq_range)
        for stat, record, idx, freq, freq_range in representatives[:top_k]
    ]


def plot_stamp_gallery(
    records: list[CadenceVizRecord],
    summaries: dict[str, CadenceVizSummary],
    pixel_pool: dict[tuple[str, int], np.ndarray] | None = None,
) -> str | None:
    """Top-K stamps by detection statistic, each rendered as the 6-observation cadence
    waterfall grid scientists actually inspect (ON/OFF rows, one stamp per column).
    pixel_pool carries raw pixels the collector captured before pruning deleted their
    .npy (#302) — the fallback when the direct load fails."""
    config = get_config()
    tag = config.checkpoint.save_tag
    top_k = config.inference.stamp_gallery_top_k

    selected = _select_top_stamps(records, summaries, top_k)
    if not selected:
        logger.info("Viz: no stamps available; skipping stamp_gallery")
        return None

    n_cols = len(selected)
    n_rows = len(_OBS_ROW_LABELS)
    fig = Figure(figsize=(1.9 * n_cols + 1.2, 1.1 * n_rows + 1.6))
    axes = fig.subplots(n_rows, n_cols, squeeze=False)

    for col, (record, idx, stat, freq, freq_range) in enumerate(selected):
        try:
            snippet = load_display_cadence(record.npy_path, idx)
        except Exception as e:
            pooled = (pixel_pool or {}).get((record.npy_path, idx))
            if pooled is None:
                logger.warning(f"Viz: failed to load stamp {idx} from {record.npy_path}: {e}")
                for row in range(n_rows):
                    axes[row][col].set_axis_off()
                continue
            # Same display transform load_display_cadence applies to the raw stamp
            snippet = log_norm(np.array(pooled, dtype=np.float32))
        draw_cadence_strip(
            [axes[row][col] for row in range(n_rows)],
            snippet,
            label_rows=col == 0,
            freq_range_mhz=freq_range,
        )
        axes[0][col].set_title(
            f"$k^2$={stat:.3g}\n{freq:.4f} MHz\n{_key_label(record.key, 20)}", fontsize=7
        )

    fig.suptitle(
        f"Top-{n_cols} energy-detection stamps by statistic ({display_tag(tag, get_machine_name())})"
    )

    return _save_and_upload(
        fig, f"stamp_gallery_{display_tag(tag, get_machine_name())}.png", "Stamp Gallery"
    )


def plot_preproc_funnel(
    records: list[CadenceVizRecord], summaries: dict[str, CadenceVizSummary]
) -> str | None:
    """Per-cadence preprocessing funnel: raw hits → merged hits → stamps (incl. overlap
    offsets) → snippets inferred, plus per-cadence stamp storage annotated on top. Past
    _FUNNEL_MAX_CADENCES the strongest cadences (by raw hits) keep individual bars and
    the rest aggregate into one summary bar (#301 — at 270 px/cadence the unbounded
    figure exceeded Agg's 2^16-px canvas limit past 242 cadences, so every catalog-scale
    render burned the artist work and then silently lost the figure)."""
    tag = get_config().checkpoint.save_tag

    labels: list[str] = []
    stage_counts: list[tuple[int, int, int, int]] = []
    storage_gb: list[float] = []
    for record in records:
        summary = summaries.get(record.npy_path)
        n_raw = summary.n_raw_hits if summary else 0
        n_merged = summary.n_merged_hits if summary else 0
        n_stamps = (summary.n_stamp_rows if summary else 0) or record.n_stamps
        labels.append(_key_label(record.key))
        stage_counts.append((n_raw, n_merged, n_stamps, record.n_stamps))
        size = summary.npy_size_bytes if summary else float("nan")
        storage_gb.append(size / 1e9 if np.isfinite(size) else float("nan"))

    if not stage_counts:
        logger.info("Viz: no cadences recorded; skipping preproc_funnel")
        return None

    aggregated_note = ""
    if len(stage_counts) > _FUNNEL_MAX_CADENCES:
        order = np.argsort([c[0] for c in stage_counts])[::-1]
        keep = set(order[:_FUNNEL_MAX_CADENCES].tolist())
        rest = [i for i in range(len(stage_counts)) if i not in keep]
        agg_counts = tuple(int(sum(stage_counts[i][j] for i in rest)) for j in range(4))
        agg_storage = float(np.nansum([storage_gb[i] for i in rest]))
        # Kept cadences stay in catalog order; the aggregate bar closes the figure
        kept = sorted(keep)
        labels = [labels[i] for i in kept] + [f"+{len(rest)} more (aggregated)"]
        stage_counts = [stage_counts[i] for i in kept] + [agg_counts]
        storage_gb = [storage_gb[i] for i in kept] + [agg_storage]
        aggregated_note = f" — top {_FUNNEL_MAX_CADENCES} by raw hits, {len(rest)} aggregated"

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
    ax.set_title(
        f"Preprocessing funnel per cadence ({display_tag(tag, get_machine_name())}) — stamp storage annotated{aggregated_note}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)

    return _save_and_upload(
        fig, f"preproc_funnel_{display_tag(tag, get_machine_name())}.png", "Preprocessing Funnel"
    )


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
    ax.set_title(
        f"Snippet confidence distribution ({display_tag(tag, get_machine_name())})\n{subtitle}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    return _save_and_upload(
        fig,
        f"confidence_distribution_{display_tag(tag, get_machine_name())}.png",
        "Confidence Distribution",
    )


_CANDIDATE_FREQ_MAX_TARGETS = 40
_BAND_COLORS = {"L": "tab:blue", "S": "tab:green", "C": "tab:red", "X": "tab:purple"}


def plot_candidate_frequency() -> str | None:
    """
    Candidate frequency map (#394): every candidate as a dot at (frequency, target row),
    colored by band — targets sorted by candidate count, report-excluded frequency ranges
    (#395) shaded. The load-bearing read is VERTICAL alignment: the same frequency lighting
    up across many targets is the signature of a terrestrial transmitter (multi-target
    coincidence), whereas a genuine technosignature is a dot with no vertical company. In
    inf_blpc3_20260807_011509 two target/bands held 49% of all candidates and several top
    candidates sat in GPS/Iridium allocations — structure no other figure showed.
    """
    config = get_config()
    tag = config.checkpoint.save_tag

    db = get_db()
    if db is None:
        logger.info("Viz: no database instance; skipping candidate frequency map")
        return None
    db.flush()
    rows = db.query_inference_result(
        tag=tag, prediction=1, columns=["target", "band", "frequency_mhz"]
    )
    rows = [r for r in rows if r.get("frequency_mhz") is not None]
    if not rows:
        logger.info("Viz: no candidates with frequencies recorded; skipping frequency map")
        return None

    counts = Counter(r.get("target") or "?" for r in rows)
    ordered_targets = [target for target, _ in counts.most_common(_CANDIDATE_FREQ_MAX_TARGETS)]
    overflow = len(counts) - len(ordered_targets)
    overflow_label = f"(+{overflow} more targets)"
    if overflow:
        # The aggregate row's count is its candidate total, mirroring the per-target rows
        counts[overflow_label] = len(rows) - sum(counts[t] for t in ordered_targets)
        ordered_targets.append(overflow_label)
    # Row 0 at the top: most candidate-heavy target first
    y_index = {target: i for i, target in enumerate(ordered_targets)}

    fig = Figure(figsize=(12, 0.28 * len(ordered_targets) + 2.8))
    ax = fig.subplots()

    seen_bands = []
    for row in rows:
        target = row.get("target") or "?"
        y = y_index.get(target, y_index.get(overflow_label))
        band = row.get("band") or "?"
        if band not in seen_bands:
            seen_bands.append(band)
        ax.plot(
            float(row["frequency_mhz"]),
            y,
            marker="o",
            markersize=4,
            alpha=0.55,
            color=_BAND_COLORS.get(band, "tab:gray"),
            linestyle="none",
        )

    exclusion_ranges = report_exclusion_ranges(config)
    for start, end in exclusion_ranges:
        ax.axvspan(start, end, color="red", alpha=0.12, zorder=0)
    if exclusion_ranges:
        range_label = ", ".join(f"{start:g}-{end:g}" for start, end in exclusion_ranges)
        ax.set_title(
            f"Candidate frequency map ({display_tag(tag, get_machine_name())}) — "
            f"{len(rows):,} candidates; shaded: report-excluded {range_label} MHz"
        )
    else:
        ax.set_title(
            f"Candidate frequency map ({display_tag(tag, get_machine_name())}) — "
            f"{len(rows):,} candidates"
        )

    ax.set_yticks(range(len(ordered_targets)))
    ax.set_yticklabels([f"{target} ({counts[target]})" for target in ordered_targets], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("frequency (MHz)")
    ax.grid(True, axis="x", alpha=0.2)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=_BAND_COLORS.get(band, "tab:gray"),
            label=band,
        )
        for band in sorted(seen_bands)
    ]
    ax.legend(handles=handles, title="band", fontsize=8, loc="upper right")

    return _save_and_upload(
        fig,
        f"candidate_frequency_map_{display_tag(tag, get_machine_name())}.png",
        "Candidate Frequency Map",
    )


def plot_candidate(row: dict, index: int) -> str | None:
    """One candidate's full picture (implements the long-standing inference.py stub):
    6-panel cadence waterfall of its stamp, annotated with confidence / frequency /
    target / session / band, plus the 48-dim latent vector as a bar chart. Thin wrapper
    over the TF-free candidate_figures.render_candidate_figure (#298 I9 — the gallery path
    renders these across a forkserver pool; this in-process form serves direct callers)."""
    tag = get_config().checkpoint.save_tag
    display_tag_value = display_tag(tag, get_machine_name())
    save_path = render_candidate_figure(
        row, index, display_tag_value, _plots_dir(tag), candidate_frequency_range_mhz(row)
    )
    logger.info(f"Inference viz saved: {save_path}")
    _uploader.submit(save_path, f"Candidate {index} - ({display_tag_value})")
    return save_path


def plot_candidate_gallery() -> str | None:
    """Gallery of the top candidates (6-obs waterfall strips) plus capped per-candidate
    figures, sourced from the inference_results table so it also covers cadences the
    resume skipped this pass. Ordering (#397): confidence descending, tie-broken by
    survey-OOD distance descending then MC spread ascending — within a saturated P=1.000
    tie the candidate least like the survey background reviews first."""
    config = get_config()
    tag = config.checkpoint.save_tag
    display_tag_value = display_tag(tag, get_machine_name())
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
    survey_ood = survey_ood_scores(rows, _reference_cloud_path(config, tag))
    rows = triage_sort_rows(rows, survey_ood)

    # Report-time frequency exclusion (#395): excluded candidates keep their DB rows and
    # their rendered figures on disk; only the Slack uploads and the gallery membership
    # below are filtered.
    exclusion_ranges = report_exclusion_ranges(config)

    # Per-candidate figures (highest confidence first, capped): rendered across the
    # forkserver pool (#298 I9 — independent row dict + memmap read + PNG each; per-figure
    # failures return None and degrade the suite exactly like _viz_safe), then uploaded in
    # index order through the async FIFO uploader.
    top_rows = rows[:max_candidate_plots]
    rendered = render_candidate_figures(top_rows, display_tag_value, _plots_dir(tag))
    n_upload_excluded = 0
    for index, save_path in rendered:
        if save_path is None:
            continue
        logger.info(f"Inference viz saved: {save_path}")
        if exclusion_ranges and frequency_excluded(
            top_rows[index].get("frequency_mhz"), exclusion_ranges
        ):
            n_upload_excluded += 1
            continue
        _uploader.submit(save_path, f"Candidate {index} - ({display_tag_value})")
    if n_upload_excluded:
        logger.info(
            f"Viz: {n_upload_excluded} candidate figure(s) rendered and saved but not "
            f"uploaded (--report-exclude-frequency-range)"
        )
    if len(rows) > max_candidate_plots:
        logger.info(
            f"Viz: rendered {max_candidate_plots} of {len(rows)} candidate figures "
            f"(--max-candidate-plots cap)"
        )

    reported_rows, excluded_rows = partition_candidates_by_frequency(rows, exclusion_ranges)
    if not reported_rows:
        logger.info(
            f"Viz: all {len(rows)} candidate(s) fall in --report-exclude-frequency-range "
            f"ranges; skipping the gallery figure (per-candidate figures are still on disk)"
        )
        return None
    gallery_rows = reported_rows[:_CANDIDATE_GALLERY_MAX]
    n_cols = len(gallery_rows)
    n_rows = len(_OBS_ROW_LABELS)
    fig = Figure(figsize=(1.9 * n_cols + 1.2, 1.1 * n_rows + 1.6))
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    for col, row in enumerate(gallery_rows):
        try:
            snippet = load_display_cadence(str(row["npy_path"]), int(row["snippet_index"]))
        except Exception as e:
            logger.warning(f"Viz: failed to load candidate stamp ({e})")
            for r in range(n_rows):
                axes[r][col].set_axis_off()
            continue
        draw_cadence_strip(
            [axes[r][col] for r in range(n_rows)],
            snippet,
            label_rows=col == 0,
            freq_range_mhz=candidate_frequency_range_mhz(row),
        )
        freq = row.get("frequency_mhz")
        freq_label = f"{freq:.4f} MHz" if freq is not None else "freq n/a"
        title = f"P={row.get('confidence', 0):.3f}\n{freq_label}\n{row.get('target') or ''}"
        ood = survey_ood.get((row.get("npy_path"), int(row.get("snippet_index") or 0)))
        if ood is not None:
            # Survey-OOD percentile (#397): how unlike the run's reference cloud this
            # candidate's latent is (triage annotation, never a gate)
            title += f"\nOOD p{ood[1]:.1f}"
        axes[0][col].set_title(title, fontsize=7)
    exclusion_note = (
        f" ({len(excluded_rows)} excluded by frequency filter)" if excluded_rows else ""
    )
    fig.suptitle(
        f"Candidate gallery ({display_tag(tag, get_machine_name())}): "
        f"top {n_cols} of {len(reported_rows)} by confidence{exclusion_note}"
    )

    return _save_and_upload(
        fig, f"candidate_gallery_{display_tag(tag, get_machine_name())}.png", "Candidate Gallery"
    )


def write_candidate_triage_report() -> str | None:
    """
    Candidate triage CSV (#397): one row per candidate in review order (triage_sort_rows),
    carrying the scores a reviewer triages by — confidence, MC mean/spread, survey-OOD
    distance/percentile (vs this run's reference cloud), training-OOD distance/percentile
    (vs the training run's true-class features, when the cluster-local eval artifact is
    available), and the report-time frequency-exclusion flag (#395). Written beside the
    figures; logged, not uploaded (Slack surfaces stay image-only). Scores are triage
    ranking only — nothing here gates candidacy: a genuine technosignature is itself OOD
    with respect to synthetic training data.
    """
    import csv  # noqa: PLC0415  # stdlib; only this report needs it

    config = get_config()
    tag = config.checkpoint.save_tag

    db = get_db()
    if db is None:
        logger.info("Viz: no database instance; skipping candidate triage report")
        return None
    db.flush()
    rows = db.query_inference_result(tag=tag, prediction=1)
    if not rows:
        logger.info("Viz: no candidates recorded; skipping candidate triage report")
        return None

    survey_ood = survey_ood_scores(rows, _reference_cloud_path(config, tag))
    training_ood = training_ood_scores(rows, config.model_path, config.inference.config_path)
    exclusion_ranges = report_exclusion_ranges(config)
    ordered = triage_sort_rows(rows, survey_ood)

    report_path = os.path.join(
        _plots_dir(tag), f"candidate_triage_{display_tag(tag, get_machine_name())}.csv"
    )
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "target",
                "band",
                "frequency_mhz",
                "confidence",
                "mc_mean",
                "mc_std",
                "survey_ood_distance",
                "survey_ood_percentile",
                "training_ood_distance",
                "training_ood_percentile",
                "excluded_by_report_filter",
                "npy_path",
                "snippet_index",
            ]
        )
        for rank, row in enumerate(ordered):
            key = (row.get("npy_path"), int(row.get("snippet_index") or 0))
            s_dist, s_pct = survey_ood.get(key, (None, None))
            t_dist, t_pct = training_ood.get(key, (None, None))
            writer.writerow(
                [
                    rank,
                    row.get("target"),
                    row.get("band"),
                    row.get("frequency_mhz"),
                    row.get("confidence"),
                    row.get("mc_mean"),
                    row.get("mc_std"),
                    s_dist,
                    s_pct,
                    t_dist,
                    t_pct,
                    int(
                        bool(exclusion_ranges)
                        and frequency_excluded(row.get("frequency_mhz"), exclusion_ranges)
                    ),
                    row.get("npy_path"),
                    row.get("snippet_index"),
                ]
            )

    n_survey = len(survey_ood)
    n_training = len(training_ood)
    logger.info(
        f"Candidate triage report saved: {report_path} ({len(ordered)} candidate(s); "
        f"survey OOD for {n_survey}, training OOD for {n_training})"
    )
    return report_path


def plot_candidate_uncertainty() -> str | None:
    """
    Candidate uncertainty view (#282): x = final RF probability (MC mean), y = MC spread,
    each candidate bold red over a hexbin density background of the survey's reference
    cloud (the seeded uniform subsample of pass-1 rejects persisted by
    finalize_reference_cloud). Population context is the whole point: "p=0.97,
    spread=0.05" is only interpretable against where the survey sits — and the dangerous
    quadrant (high p, high spread: a mean that looks confident while draws swing) is
    exactly what p alone cannot flag. The science threshold is drawn as a vertical line.
    """
    config = get_config()
    tag = config.checkpoint.save_tag

    db = get_db()
    if db is None:
        logger.info("Viz: no database instance; skipping candidate uncertainty plot")
        return None
    db.flush()
    rows = db.query_inference_result(
        tag=tag, prediction=1, columns=["mc_mean", "mc_std", "target", "frequency_mhz"]
    )
    rows = [r for r in rows if r.get("mc_mean") is not None and r.get("mc_std") is not None]

    cloud_path = _reference_cloud_path(config, tag)
    cloud = None
    if os.path.exists(cloud_path):
        # Materialize the arrays and close the archive immediately — an NpzFile keeps its
        # file handle open until GC, which would leak on this function's early returns
        with np.load(cloud_path) as cloud_npz:
            cloud = {key: cloud_npz[key] for key in cloud_npz.files}
    elif config.inference.reference_cloud_size > 0:
        logger.warning(
            f"Viz: reference cloud {cloud_path} not found — the candidate uncertainty plot "
            "will lack the survey background (candidates would only be compared against "
            "each other)"
        )

    if not rows and cloud is None:
        logger.info("Viz: no MC-scored candidates and no reference cloud; skipping plot")
        return None

    fig = Figure(figsize=(9, 7))
    ax = fig.subplots(1, 1)

    if cloud is not None and len(cloud["mc_mean"]) > 0:
        hb = ax.hexbin(
            cloud["mc_mean"],
            cloud["mc_std"],
            gridsize=60,
            bins="log",
            cmap="Greys",
            mincnt=1,
        )
        fig.colorbar(hb, ax=ax, label="survey reference density (log count)")

    if rows:
        ax.scatter(
            [r["mc_mean"] for r in rows],
            [r["mc_std"] for r in rows],
            c="red",
            s=60,
            marker="*",
            zorder=5,
            label=f"candidates ({len(rows)})",
        )

    threshold = config.inference.classification_threshold
    ax.axvline(threshold, color="tab:blue", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(
        threshold,
        ax.get_ylim()[1],
        f"  science threshold = {threshold}",
        color="tab:blue",
        fontsize=8,
        va="top",
    )

    ax.set_xlabel(
        "RF probability (MC mean"
        + (", calibrated)" if config.rf.calibration_active else ", uncalibrated)")
    )
    ax.set_ylabel("MC spread (std of draw probabilities)")
    cloud_note = ""
    if cloud is not None:
        cloud_note = (
            f" — cloud: {int(cloud['subsample_size'])} of {int(cloud['rejects_seen'])} "
            f"rejects, {int(cloud['mc_draws'])} draws"
        )
    ax.set_title(
        f"Candidate uncertainty vs survey population ({display_tag(tag, get_machine_name())}){cloud_note}",
        fontsize=11,
    )
    if rows:
        ax.legend(loc="upper left", fontsize=8)

    return _save_and_upload(
        fig,
        f"candidate_uncertainty_{display_tag(tag, get_machine_name())}.png",
        "Candidate Uncertainty vs Survey",
    )


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
        f"Inference latents through training cadence-level UMAP ({display_tag(tag, get_machine_name())})\n"
        f"nn={nn}, md={md}, training tag {train_tag}"
    )
    ax.legend(fontsize=8, loc="best")

    return _save_and_upload(
        fig,
        f"inference_latent_projection_{display_tag(tag, get_machine_name())}.png",
        "Inference Latent Projection",
    )


def plot_inference_summary(
    records: list[CadenceVizRecord], summaries: dict[str, CadenceVizSummary], totals: dict
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
    n_raw_hits = sum(s.n_raw_hits for s in summaries.values())
    n_merged_hits = sum(s.n_merged_hits for s in summaries.values())
    # File sizes were stat()ed once by the reduce pass — no second walk over the catalog
    storage_gb = float(
        np.nansum([s.npy_size_bytes for s in summaries.values()]) / 1e9 if summaries else 0.0
    )

    # Report-time frequency exclusion (#395): the card carries all three numbers so the
    # science record (original) and the review surface (reported) stay distinguishable
    exclusion_ranges = report_exclusion_ranges(config)
    exclusion_rows: list[tuple[str, str]] = []
    if exclusion_ranges and db is not None:
        freq_rows = db.query_inference_result(tag=tag, prediction=1, columns=["frequency_mhz"])
        reported, excluded = partition_candidates_by_frequency(freq_rows, exclusion_ranges)
        range_label = ", ".join(f"{start:g}-{end:g}" for start, end in exclusion_ranges)
        exclusion_rows = [
            (f"  excluded ({range_label} MHz)", f"{len(excluded):,}"),
            ("  reported after exclusion", f"{len(reported):,}"),
        ]

    summary_rows = [
        ("run tag", tag),
        ("rendered", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("cadences", f"{totals.get('n_cadences', len(records))}"),
        ("  resumed (skipped this pass)", f"{totals.get('n_skipped', 0)}"),
        ("raw ED hits", f"{n_raw_hits:,}"),
        ("merged hits", f"{n_merged_hits:,}"),
        ("snippets inferred", f"{n_snippets:,}"),
        ("candidates", f"{totals.get('n_candidates', 0):,}"),
        *exclusion_rows,
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
    ax.set_title(
        f"Inference run summary ({display_tag(tag, get_machine_name())})", fontsize=13, pad=14
    )

    return _save_and_upload(
        fig, f"inference_summary_{display_tag(tag, get_machine_name())}.png", "Inference Summary"
    )


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
    logger.info(
        f"Rendering inference visualization suite under plots/inference/{display_tag(tag, get_machine_name())}/"
    )

    records = collector.records
    if config.inference.inference_viz_scope == "new":
        # Scope 'new' (#301): resumed passes on an accumulating tag re-rendered the FULL
        # catalog's figures every pass. This renders only cadences inferred THIS pass;
        # the DB-sourced candidate figures below still cover the whole tag, and the final
        # pass can render everything with the 'full' default.
        n_before = len(records)
        records = [record for record in records if not record.skipped]
        logger.info(
            f"Viz scope 'new': rendering {len(records)} of {n_before} recorded cadence(s) "
            f"(resumed cadences excluded; candidate figures still cover the full tag)"
        )

    with stage_timer("load_metadata"):
        summaries = _build_summaries(records, config.inference.stamp_gallery_top_k)

    try:
        _viz_safe("ed_stat_distributions", plot_ed_stat_distributions, records, summaries)
        _viz_safe("ed_hit_spectrum", plot_ed_hit_spectrum, records, summaries)
        _viz_safe("bandpass_flattening", plot_bandpass_flattening, preprocessor, records, summaries)
        _viz_safe(
            "stamp_gallery", plot_stamp_gallery, records, summaries, collector.gallery_pixels()
        )
        _viz_safe("preproc_funnel", plot_preproc_funnel, records, summaries)
        _viz_safe("confidence_distribution", plot_confidence_distribution, records)
        _viz_safe("candidate_frequency_map", plot_candidate_frequency)
        _viz_safe("candidate_gallery", plot_candidate_gallery)
        _viz_safe("candidate_triage", write_candidate_triage_report)
        _viz_safe("candidate_uncertainty", plot_candidate_uncertainty)
        _viz_safe("inference_latent_projection", plot_inference_latent_projection, collector)
        _viz_safe("inference_summary", plot_inference_summary, records, summaries, totals)
    finally:
        # The async uploader must be empty before the caller reaches logger teardown
        # (#298 I9) — uploads queued by any figure above are flushed here even when a
        # figure raised through _viz_safe's own guard
        with stage_timer("upload_drain"):
            _uploader.drain()

    logger.info("Inference visualization suite complete")
