"""
Process-parallel latent-space GIF frame rendering (#278).

The latent-GIF stage's cost is almost entirely matplotlib: the old implementation rebuilt a
figure, re-scattered every category, and saved a PNG per frame on one thread — ~6.6 s/frame,
~55 min per GIF, ~24-29 h across the 24-GIF sweep. Frames are fully independent (each needs
only its 2D coords, labels, the shared axis limits and palette), so this module renders them
across a process pool, mirroring shap_parallel.py's isolation pattern: forkserver context with
an EMPTY preload so workers never re-import the parent __main__ (aetherscan.main -> TF -> the
training stack), and this module itself stays off the TF import graph (train.py imports it,
never the reverse). Unlike SHAP, per-worker memory is tiny (one chunk of frames), so the pool
is core-bound — size it by cores, not RAM.

Within each worker the figure and the per-category scatter artists are created ONCE and updated
per frame via set_offsets / set_title (the standard animation idiom); the legend is rebuilt per
frame from the categories actually present, matching the old output exactly. Rendering is
deterministic: the same frames produce byte-identical PNGs regardless of worker count.

GIF assembly (imageio) and Slack upload stay in the caller — only PNG rendering parallelizes.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

# Worker-global render state, set up once per worker by _worker_init (module globals survive
# across _worker calls within one worker process, so the figure is built once, not per chunk)
_STYLE: dict | None = None
_FIG = None
_AX = None
_ARTISTS: list | None = None


@dataclass
class FrameCategory:
    """One scatter category: its mask keys (signal_type, optional ON/OFF) and its style."""

    signal_type: str
    onoff: str | None  # None for cadence-level (labels-only masking)
    color: str
    marker: str
    display_name: str


def batched_umap_transform(model, coords_list: list[np.ndarray]) -> list[np.ndarray]:
    """
    Project every snapshot's coords through `model` in ONE stacked .transform() call instead
    of len(coords_list) serial calls. Returns the per-snapshot list split back out, order
    preserved.

    NOT used by the pipeline: the #278 benchmark measured it ~9% faster but NOT
    output-identical to per-snapshot transforms (UMAP consumes its random_state stream
    differently for the joint batch), which violates #278's outputs-unchanged constraint.
    Kept as benchmark-support so bench_latent_gif.py can keep re-measuring the trade.
    """
    if not coords_list:
        return []
    stacked = np.vstack(coords_list)
    transformed = model.transform(stacked)
    split_points = np.cumsum([len(coords) for coords in coords_list])[:-1]
    return [np.ascontiguousarray(part) for part in np.split(transformed, split_points)]


def _worker_init(style: dict) -> None:
    """Pin the Agg backend and stash the shared style; the figure is built lazily on the
    first frame so matplotlib initializes inside the worker, never in the forkserver."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    global _STYLE
    _STYLE = style


def _ensure_figure():
    """Build the reusable figure + one (initially empty) scatter artist per category."""
    global _FIG, _AX, _ARTISTS
    if _FIG is not None:
        return
    import matplotlib.pyplot as plt  # noqa: PLC0415

    style = _STYLE
    _FIG, _AX = plt.subplots(1, 1, figsize=tuple(style["figsize"]))
    _ARTISTS = []
    for category in style["categories"]:
        artist = _AX.scatter(
            [],
            [],
            c=category.color,
            marker=category.marker,
            s=10,
            alpha=0.75,
            label=category.display_name,
            rasterized=True,
        )
        _ARTISTS.append(artist)
    _AX.set_xlim(tuple(style["xlim"]))
    _AX.set_ylim(tuple(style["ylim"]))
    # Fixed limits + a single-line title => the geometry is frame-independent, so one
    # tight_layout call up front matches the old per-frame call's result
    _AX.set_title(" ", fontsize=11)
    _FIG.tight_layout()


def _render_chunk(task: tuple) -> list[str]:
    """Render one chunk of frames to PNGs; returns the frame paths in order."""
    frames, frame_indices = task
    _ensure_figure()
    style = _STYLE
    categories = style["categories"]
    paths: list[str] = []

    for frame, frame_idx in zip(frames, frame_indices, strict=True):
        coords_2d = frame["coords"]
        labels_arr = np.asarray(frame["labels"])
        onoff_arr = np.asarray(frame["onoff"]) if frame.get("onoff") is not None else None

        visible = []
        for category, artist in zip(categories, _ARTISTS, strict=True):
            mask = labels_arr == category.signal_type
            if category.onoff is not None and onoff_arr is not None:
                mask &= onoff_arr == category.onoff
            if mask.any():
                artist.set_offsets(coords_2d[mask])
                visible.append(artist)
            else:
                artist.set_offsets(np.empty((0, 2)))
        # Legend rebuilt from the categories present in THIS frame — same entries the old
        # per-frame implementation produced (it only scattered non-empty categories)
        _AX.legend(handles=visible, **style["legend_kwargs"])
        _AX.set_title(frame["title"], fontsize=11)

        frame_path = os.path.join(
            style["out_dir"], f"{style['method_name']}_frame_{frame_idx:05d}.png"
        )
        _FIG.savefig(frame_path, dpi=style["dpi"])
        paths.append(frame_path)

    return paths


def render_latent_gif_frames(
    frames: list[dict],
    categories: list[FrameCategory],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    legend_kwargs: dict,
    out_dir: str,
    method_name: str,
    n_workers: int,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 100,
) -> list[str]:
    """
    Render one PNG per frame across a process pool; returns the frame paths in frame order.

    Each frame dict carries: "coords" (N, 2 float array), "labels" (N str array), optional
    "onoff" (N str array, obs-level only), and "title" (str). n_workers <= 1 renders fully
    in-process (no pool) — byte-identical output either way.
    """
    if not frames:
        return []

    os.makedirs(out_dir, exist_ok=True)
    style = {
        "categories": categories,
        "xlim": tuple(xlim),
        "ylim": tuple(ylim),
        "legend_kwargs": legend_kwargs,
        "out_dir": out_dir,
        "method_name": method_name,
        "figsize": figsize,
        "dpi": dpi,
    }

    n_workers = max(1, min(n_workers, len(frames)))
    if n_workers == 1:
        # In-process render — reset worker globals afterwards so repeated calls (the 24-GIF
        # sweep runs in one process) rebuild against the new style
        global _FIG, _AX, _ARTISTS, _STYLE
        try:
            _worker_init(style)
            all_indices = list(range(len(frames)))
            return _render_chunk((frames, all_indices))
        finally:
            if _FIG is not None:
                import matplotlib.pyplot as plt  # noqa: PLC0415

                plt.close(_FIG)
            _FIG = _AX = _ARTISTS = _STYLE = None

    # Forkserver with an empty preload list, exactly like shap_parallel.py: workers get a
    # clean interpreter that imports only this light module, never the parent's TF stack
    ctx = mp.get_context("forkserver")
    ctx.set_forkserver_preload([])

    chunk_bounds = np.array_split(np.arange(len(frames)), n_workers)
    tasks = [
        ([frames[i] for i in chunk], [int(i) for i in chunk])
        for chunk in chunk_bounds
        if len(chunk)
    ]

    with ProcessPoolExecutor(
        max_workers=len(tasks), mp_context=ctx, initializer=_worker_init, initargs=(style,)
    ) as pool:
        chunk_results = list(pool.map(_render_chunk, tasks))

    return [path for chunk_paths in chunk_results for path in chunk_paths]
