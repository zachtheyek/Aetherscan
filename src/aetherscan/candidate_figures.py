"""
Process-parallel per-candidate figure rendering for the inference viz suite (#298 I9).

The ≤ max_candidate_plots per-candidate figures are independent (one DB row dict + one
memmap stamp read + one PNG each) but were rendered strictly serially on the main thread at
finalize time. This module farms them across a small process pool, mirroring
shap_parallel.py / latent_gif.py's isolation pattern: **forkserver context with an empty
preload**, so workers never re-import the parent's ``__main__`` (``aetherscan.main`` → TF →
the whole stack) and never touch the CUDA-initialized parent state — ``inference_viz``
itself imports TF transitively via ``aetherscan.models``, which is exactly why the
figure-building code lives HERE, deliberately TF-free, and ``inference_viz`` imports it
(never the reverse).

Figures are built on the object-oriented matplotlib API (``Figure`` + implicit Agg canvas —
pyplot is never imported, so no GUI backend can be selected in a worker). Per-figure
containment mirrors ``_viz_safe``: a failed render returns None for that candidate and the
suite continues. Slack uploads stay with the PARENT (inference_viz's async uploader), fed in
index order from the returned paths.
"""

from __future__ import annotations

import functools
import json
import logging
import multiprocessing as mp
import os
import uuid
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from matplotlib.figure import Figure

from aetherscan.data_generation import log_norm

logger = logging.getLogger(__name__)

# ABACAD cadence row labels (ON at positions 0/2/4) — single source for every figure that
# draws observation strips (the suite imports it from here).
OBS_ROW_LABELS = ("ON", "OFF", "ON", "OFF", "ON", "OFF")

# Pool sizing: candidate figures are ~1 s of matplotlib each with a hard cap of
# max_candidate_plots (50), so a handful of workers saturates the win; below the floor the
# pool spin-up would cost more than it saves and rendering stays in-process.
_MAX_RENDER_WORKERS = 8
_MIN_ROWS_FOR_POOL = 4


def candidate_sidecar_path(npy_path: str) -> str:
    """Sibling .candidates.npz path for a cadence's pruned candidate snippets (#302)."""
    return os.path.splitext(npy_path)[0] + ".candidates.npz"


def write_candidate_snippet_sidecar(npy_path: str, snippet_indices) -> str:
    """Snapshot the candidate snippets (~196 KB each) of a cadence into an atomic
    .candidates.npz sidecar next to its metadata (#302): stamp-cache pruning deletes the
    multi-GB stamp .npy after scoring, and the candidate figures re-read their snippets
    from this sidecar instead. Raw (pre-log-norm) values, exactly as stored in the .npy,
    so load_display_cadence's fallback applies the identical display transform."""
    stamps = np.load(npy_path, mmap_mode="r")
    indices = np.asarray(sorted({int(i) for i in snippet_indices}), dtype=np.int64)
    rows = np.array(stamps[indices], dtype=np.float32)
    del stamps
    path = candidate_sidecar_path(npy_path)
    tmp_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    with open(tmp_path, "wb") as f:
        np.savez(f, snippet_indices=indices, stamps=rows)
    os.replace(tmp_path, path)
    return path


def _load_snippet_from_sidecar(npy_path: str, snippet_index: int) -> np.ndarray:
    """Read one candidate snippet from the pruned cadence's sidecar; raises KeyError when
    the snippet was not a candidate (non-candidate pixels are gone by design, #302)."""
    with np.load(candidate_sidecar_path(npy_path)) as sidecar:
        match = np.nonzero(sidecar["snippet_indices"] == int(snippet_index))[0]
        if not len(match):
            raise KeyError(
                f"snippet {snippet_index} is not in the candidate sidecar for {npy_path} "
                f"(its stamp .npy was pruned and only candidate snippets were kept)"
            )
        return np.array(sidecar["stamps"][int(match[0])], dtype=np.float32)


def load_display_cadence(npy_path: str, snippet_index: int) -> np.ndarray:
    """Load one snippet's (num_obs, time_bins, width) stamp from its .npy — falling back
    to the candidate snippet sidecar when the .npy was pruned (#302) — and log-normalize
    it for display (the same transform the model input path applies)."""
    try:
        stamps = np.load(npy_path, mmap_mode="r")
    except OSError:
        snippet = _load_snippet_from_sidecar(npy_path, snippet_index)
    else:
        snippet = np.array(stamps[snippet_index], dtype=np.float32)
        del stamps
    return log_norm(snippet)


def cadence_metadata_path(npy_path: str) -> str:
    """Sibling .json path for a cadence's metadata — the same naming rule as
    preprocessing.DataPreprocessor.cadence_metadata_path, duplicated here (one line) so
    render workers never import the preprocessing module (manager/db singletons, setigen)."""
    return os.path.splitext(npy_path)[0] + ".json"


def stamp_frequency_range_mhz(metadata: dict, snippet_index: int) -> tuple[float, float] | None:
    """
    Frequency range (MHz) spanned by one stamp's frequency axis — bin 0 through the last
    RAW bin — from the cadence metadata sidecar: header fch1/foff plus the stamp's start
    index and stamp_width (#298 follow-up: cadence-snippet plots label their x-axis with
    the frequency span). Returned in bin order, so a negative foff yields a descending
    (high → low) pair — callers print it as-is. None when any field is missing or malformed
    (legacy sidecars): callers keep the unlabeled axis.
    """
    try:
        header = metadata.get("header") or {}
        fch1 = float(header["fch1"])
        foff = float(header["foff"])
        start = int((metadata.get("stamp_starts") or [])[snippet_index])
        width = int(metadata["stamp_width"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return fch1 + foff * start, fch1 + foff * (start + width - 1)


def draw_cadence_strip(
    axes_column,
    snippet: np.ndarray,
    label_rows: bool,
    freq_range_mhz: tuple[float, float] | None = None,
) -> None:
    """Draw one snippet's 6 observation waterfalls down a column of axes. With
    freq_range_mhz, the bottom panel's x-axis carries a "Frequency (MHz)" title and two
    tick labels — the frequency at the left and right borders (bin order, so a descending
    pair reflects a negative foff)."""
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
            ax.set_ylabel(OBS_ROW_LABELS[obs_idx], fontsize=8, rotation=0, ha="right", va="center")
    if freq_range_mhz is not None:
        # Annotate the frequency axis at its left/right borders under a "Frequency (MHz)"
        # title (bin 0 → last bin, so a negative foff prints the borders high → low).
        left_mhz, right_mhz = freq_range_mhz
        bottom = axes_column[-1]
        n_freq_bins = snippet.shape[-1]
        bottom.set_xticks([0, n_freq_bins - 1])
        bottom.set_xticklabels([f"{left_mhz:.4f}", f"{right_mhz:.4f}"])
        bottom.tick_params(axis="x", length=2, pad=1, labelsize=6)
        bottom.set_xlabel("Frequency (MHz)", fontsize=7)


def candidate_annotation(row: dict) -> str:
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


@functools.lru_cache(maxsize=4)
def _load_sidecar_cached(metadata_path: str) -> dict | None:
    """Small parent-side LRU over parsed metadata sidecars (#301): candidates cluster on
    few cadences, and the gallery/task-build path parsed the SAME multi-MB JSON once per
    candidate row (~15-20 s serial before the render pool even started). Sidecars are
    immutable once published (atomic tmp -> os.replace), so a cached parse can never go
    stale; maxsize bounds the held dicts.

    CONTRACT: the returned dict is the CACHED object, aliased across every caller —
    treat it as immutable (a mutation would silently poison all later readers of the
    same sidecar; a copy here would defeat the memory bound)."""
    try:
        with open(metadata_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def candidate_frequency_range_mhz(row: dict) -> tuple[float, float] | None:
    """Best-effort frequency span for one candidate row: read its cadence's metadata
    sidecar (derived from npy_path, cached across rows) and look up the stamp's range.
    None on any failure — the figure keeps its generic axis label."""
    try:
        metadata = _load_sidecar_cached(cadence_metadata_path(str(row["npy_path"])))
    except KeyError:
        return None
    if metadata is None:
        return None
    return stamp_frequency_range_mhz(metadata, int(row["snippet_index"]))


def render_candidate_figure(
    row: dict,
    index: int,
    tag: str,
    plots_dir: str,
    freq_range_mhz: tuple[float, float] | None = None,
) -> str:
    """
    Build and save one candidate's figure (the long-standing inference.py stub): 6-panel
    cadence waterfall of its stamp, annotated with confidence / frequency / target /
    session / band, plus the latent vector as a bar chart. Returns the saved PNG path.
    Pure function of its arguments — no config/db/logger singletons — so it runs
    identically in-process and in a forkserver worker. freq_range_mhz labels the waterfall
    x-axis with the stamp's frequency span (callers precompute it from the metadata
    sidecar; the axis stays generic when None).
    """
    snippet = load_display_cadence(str(row["npy_path"]), int(row["snippet_index"]))
    n_obs = snippet.shape[0]

    fig = Figure(figsize=(11, 7))
    grid = fig.add_gridspec(n_obs, 2, width_ratios=(2.2, 1.6), hspace=0.15, wspace=0.25, right=0.97)
    waterfall_axes = [fig.add_subplot(grid[i, 0]) for i in range(n_obs)]
    draw_cadence_strip(waterfall_axes, snippet, label_rows=True, freq_range_mhz=freq_range_mhz)

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
        0.0, 1.0, candidate_annotation(row), va="top", ha="left", fontsize=9, family="monospace"
    )

    fig.suptitle(f"Candidate {index} ({tag}) — P(true) = {row.get('confidence', 0):.4f}")

    save_path = os.path.join(plots_dir, f"candidate_{index}_{tag}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    fig.clear()
    return save_path


def _pin_worker_threads() -> None:
    # Must run before numpy imports its native thread pools; matplotlib rendering is
    # single-threaded, so this only tames incidental numpy/BLAS threads (mirrors
    # shap_parallel._pin_worker_threads). setdefault so an explicitly-set env is respected.
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _render_task(args: tuple) -> tuple[int, str | None]:
    """Worker body: render one candidate with _viz_safe-style containment (a failed figure
    must degrade the suite, never abort the pool pass)."""
    row, index, tag, plots_dir, freq_range_mhz = args
    try:
        return index, render_candidate_figure(row, index, tag, plots_dir, freq_range_mhz)
    except Exception as e:
        logger.error(f"Candidate figure {index} failed (suite continues without it): {e}")
        return index, None


def render_candidate_figures(
    rows: list[dict], tag: str, plots_dir: str, n_workers: int | None = None
) -> list[tuple[int, str | None]]:
    """
    Render one figure per (row, index) pair — in index order in the returned list — across
    a forkserver process pool, falling back to in-process rendering for small candidate
    counts or any pool-level failure. Rows must be plain dicts of primitives (the
    inference_results row shape); paths are None for candidates whose render failed.
    Frequency spans are resolved in the PARENT (one sidecar read per row, best-effort) so
    workers stay pure.
    """
    tasks = [
        (row, index, tag, plots_dir, candidate_frequency_range_mhz(row))
        for index, row in enumerate(rows)
    ]
    if not tasks:
        return []

    n_workers = min(_MAX_RENDER_WORKERS, len(tasks), n_workers or (os.cpu_count() or 1))
    if n_workers <= 1 or len(tasks) < _MIN_ROWS_FOR_POOL:
        return [_render_task(task) for task in tasks]

    try:
        # forkserver with an EMPTY preload: the fork server is spawned clean and does NOT
        # re-import the parent's __main__ (aetherscan.main -> TF -> CUDA) — a bare fork of
        # the CUDA-initialized 5-GPU parent at finalize time is exactly what this avoids.
        ctx = mp.get_context("forkserver")
        ctx.set_forkserver_preload([])
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_pin_worker_threads,
        ) as pool:
            return list(pool.map(_render_task, tasks))
    except Exception as e:
        logger.error(
            f"Candidate-figure pool failed ({e}); rendering the remaining figures in-process"
        )
        return [_render_task(task) for task in tasks]
