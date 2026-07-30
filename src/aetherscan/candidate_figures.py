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

import json
import logging
import multiprocessing as mp
import os
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


def load_display_cadence(npy_path: str, snippet_index: int) -> np.ndarray:
    """Load one snippet's (num_obs, time_bins, width) stamp from its .npy and log-normalize
    it for display (the same transform the model input path applies)."""
    stamps = np.load(npy_path, mmap_mode="r")
    snippet = np.array(stamps[snippet_index], dtype=np.float32)
    del stamps
    return log_norm(snippet)


def draw_cadence_strip(axes_column, snippet: np.ndarray, label_rows: bool) -> None:
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
            ax.set_ylabel(OBS_ROW_LABELS[obs_idx], fontsize=8, rotation=0, ha="right", va="center")


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


def render_candidate_figure(row: dict, index: int, tag: str, plots_dir: str) -> str:
    """
    Build and save one candidate's figure (the long-standing inference.py stub): 6-panel
    cadence waterfall of its stamp, annotated with confidence / frequency / target /
    session / band, plus the latent vector as a bar chart. Returns the saved PNG path.
    Pure function of (row, index, tag, plots_dir) — no config/db/logger singletons — so it
    runs identically in-process and in a forkserver worker.
    """
    snippet = load_display_cadence(str(row["npy_path"]), int(row["snippet_index"]))
    n_obs = snippet.shape[0]

    fig = Figure(figsize=(11, 7))
    grid = fig.add_gridspec(n_obs, 2, width_ratios=(2.2, 1.6), hspace=0.15, wspace=0.25, right=0.97)
    waterfall_axes = [fig.add_subplot(grid[i, 0]) for i in range(n_obs)]
    draw_cadence_strip(waterfall_axes, snippet, label_rows=True)
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
    row, index, tag, plots_dir = args
    try:
        return index, render_candidate_figure(row, index, tag, plots_dir)
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
    """
    tasks = [(row, index, tag, plots_dir) for index, row in enumerate(rows)]
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
