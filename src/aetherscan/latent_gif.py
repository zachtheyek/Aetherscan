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
        # In-process render — reset worker globals afterwards so repeated calls (the multi-GIF
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


# ---------------------------------------------------------------------------
# Whole-combo parallelism (the #278 follow-up): one (level, nn, md) UMAP combo —
# fit + persist + per-snapshot transforms + frame render + GIF assembly — per worker.
#
# The 24-combo sweep was strictly serial in train.plot_latent_space_gif at ~95% single-core
# (~1.7-1.9 h per run) even after frame rendering parallelized: every combo is an independent
# UMAP fit with its own derived random_state, reads the same shared inputs read-only, and
# writes distinct files. Farming WHOLE combos to forkserver workers therefore cannot change
# any output byte — unlike the rejected within-fit ideas (batched_umap_transform above,
# precomputed-knn reuse), nothing changes how any single fit consumes its RNG stream.

# Palettes moved verbatim from train.plot_latent_space_gif so combo workers build their
# categories without importing train (obs/cadence palettes deliberately distinct — the
# cadence colors are shared with downstream RF plots).
OBS_COLORS = {
    ("false_no_signal", "ON"): "#1565C0",
    ("false_no_signal", "OFF"): "#64B5F6",
    ("false_with_rfi", "ON"): "#F9A825",
    ("false_with_rfi", "OFF"): "#FFF176",
    ("true_only_eti", "ON"): "#2E7D32",
    ("true_only_eti", "OFF"): "#81C784",
    ("true_eti_rfi", "ON"): "#C62828",
    ("true_eti_rfi", "OFF"): "#EF5350",
}
OBS_MARKERS = {"ON": "^", "OFF": "x"}
OBS_DISPLAY_NAMES = {
    ("false_no_signal", "ON"): "No Signal (ON)",
    ("false_no_signal", "OFF"): "No Signal (OFF)",
    ("false_with_rfi", "ON"): "RFI Only (ON)",
    ("false_with_rfi", "OFF"): "RFI Only (OFF)",
    ("true_only_eti", "ON"): "ETI Only (ON)",
    ("true_only_eti", "OFF"): "ETI Only (OFF)",
    ("true_eti_rfi", "ON"): "ETI+RFI (ON)",
    ("true_eti_rfi", "OFF"): "ETI+RFI (OFF)",
}
CADENCE_COLORS = {
    "false_no_signal": "tab:blue",
    "false_with_rfi": "tab:green",
    "true_only_eti": "tab:red",
    "true_eti_rfi": "tab:orange",
}
CADENCE_DISPLAY_NAMES = {
    "false_no_signal": "No Signal",
    "false_with_rfi": "RFI Only",
    "true_only_eti": "ETI Only",
    "true_eti_rfi": "ETI+RFI",
}

# Worker-global cache of the shared input bundle: one ProcessPoolExecutor worker handles
# several combos, and they all read the same bundle file
_SWEEP_BUNDLE: dict | None = None
_SWEEP_BUNDLE_PATH: str | None = None


def categories_for_mode(mode: str) -> tuple[list[FrameCategory], dict]:
    """The scatter categories + legend kwargs for one GIF level (moved verbatim from
    train.plot_latent_space_gif's closure)."""
    if mode == "obs":
        categories = [
            FrameCategory(
                signal_type=stype,
                onoff=status,
                color=color,
                marker=OBS_MARKERS[status],
                display_name=OBS_DISPLAY_NAMES[(stype, status)],
            )
            for (stype, status), color in OBS_COLORS.items()
        ]
        legend_kwargs = {
            "loc": "upper right",
            "fontsize": 8,
            "markerscale": 2,
            "ncol": 2,
            "framealpha": 0.8,
        }
    elif mode == "cadence":
        categories = [
            FrameCategory(
                signal_type=stype,
                onoff=None,
                color=color,
                marker="o",
                display_name=CADENCE_DISPLAY_NAMES[stype],
            )
            for stype, color in CADENCE_COLORS.items()
        ]
        legend_kwargs = {
            "loc": "upper right",
            "fontsize": 8,
            "markerscale": 2,
            "framealpha": 0.8,
        }
    else:
        raise ValueError(f"mode must be 'obs' or 'cadence', got {mode!r}")
    return categories, legend_kwargs


def compute_frame_limits(
    transformed_list: list[np.ndarray],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Shared axis limits over every frame's 2D projection, padded 5% (moved verbatim)."""
    x_min = min(t[:, 0].min() for t in transformed_list)
    x_max = max(t[:, 0].max() for t in transformed_list)
    y_min = min(t[:, 1].min() for t in transformed_list)
    y_max = max(t[:, 1].max() for t in transformed_list)
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def build_snapshot_frames(
    transformed_list: list[np.ndarray],
    snapshot_metadata: list[dict],
    labels_list: list,
    onoff_list: list | None,
    display_method: str,
) -> list[dict]:
    """Assemble render_latent_gif_frames' frame dicts (coords/labels/onoff/title) for one
    combo — the title logic moved verbatim from train.plot_latent_space_gif's closure."""
    frames = []
    for frame_idx, meta in enumerate(snapshot_metadata):
        meta_snr_base = meta["snr_base"]
        meta_snr_range = meta["snr_range"]
        meta_snr_floor = meta_snr_base if meta_snr_base is not None else "?"
        meta_snr_ceil = (
            meta_snr_base + meta_snr_range
            if meta_snr_base is not None and meta_snr_range is not None
            else "?"
        )
        frames.append(
            {
                "coords": transformed_list[frame_idx],
                "labels": labels_list[frame_idx],
                "onoff": onoff_list[frame_idx] if onoff_list is not None else None,
                "title": (
                    f"Beta-VAE Latent Space: {display_method} — "
                    f"Round {meta['round_number']}, "
                    f"Epoch {meta['epoch_number']}, "
                    f"Step {meta['step_number']} "
                    f"(SNR: {meta_snr_floor}–{meta_snr_ceil})"
                ),
            }
        )
    return frames


def _sweep_worker_init() -> None:
    """Pin BLAS-family thread pools before numpy/umap initialize in the fresh forkserver
    interpreter (the shap_parallel pattern) — but deliberately NOT numba's.

    UMAP calls numba.set_num_threads(cpu_count) on its large-N paths, and numba hard-errors
    when asked to grow past the pool it launched ("Cannot set NUMBA_NUM_THREADS to a
    different value once the threads have been launched") — so capping NUMBA_NUM_THREADS at
    1 breaks production-scale fits outright (found by the first full-scale run; the small-N
    unit fits take the brute-force path and never touch the thread API). OMP_NUM_THREADS is
    left unpinned for the same reason: numba's threading layer may be OpenMP-backed, in
    which case the OMP cap is the same trap under a different name. Shrinking is always
    legal, so an unpinned launch matches the pre-parallel serial behavior exactly (the old
    in-process sweep also ran with an unpinned numba pool); the seeded fits force their
    single-threaded deterministic paths regardless."""
    for var in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _load_sweep_bundle(path: str) -> dict:
    global _SWEEP_BUNDLE, _SWEEP_BUNDLE_PATH
    if path != _SWEEP_BUNDLE_PATH:
        import joblib  # noqa: PLC0415  (deferred so _sweep_worker_init pins threads first)

        _SWEEP_BUNDLE = joblib.load(path)
        _SWEEP_BUNDLE_PATH = path
    return _SWEEP_BUNDLE


def run_umap_gif_combo(task: dict) -> dict:
    """
    One (mode, nn, md) combo end-to-end: fit the UMAP on the mode's fit pool, persist it
    (warn-not-fail, matching the serial path), transform every snapshot serially (the
    output-identical form — see batched_umap_transform's rejection note), render the frame
    PNGs in-process, and assemble the GIF. Runs in a forkserver worker (or inline when the
    sweep is serial); consumes only the combo's own derived random_state.

    Returns {method_name, display_method, gif_path, umap_path, n_frames, warnings} —
    `warnings` carries messages for the parent to log (forkserver workers have no handler),
    and gif assembly/Slack stay split: assembly here, upload in the caller.
    """
    import imageio.v3 as iio  # noqa: PLC0415  (deferred so _sweep_worker_init pins threads first)
    import joblib  # noqa: PLC0415
    import umap  # noqa: PLC0415

    bundle = _load_sweep_bundle(task["bundle_path"])
    mode = task["mode"]
    warnings: list[str] = []

    if mode == "obs":
        fit_pool = bundle["fit_pool_obs"]
        coords_list = bundle["coords_obs"]
        labels_list = bundle["labels_obs"]
        onoff_list = bundle["onoff_obs"]
    else:
        fit_pool = bundle["fit_pool_cadence"]
        coords_list = bundle["coords_cadence"]
        labels_list = bundle["labels_cadence"]
        onoff_list = None

    model = umap.UMAP(
        n_components=2,
        random_state=task["seed"],
        n_neighbors=task["nn"],
        min_dist=task["md"],
    ).fit(fit_pool)

    try:
        os.makedirs(os.path.dirname(task["umap_path"]), exist_ok=True)
        joblib.dump(model, task["umap_path"])
    except Exception as exc:
        warnings.append(f"Failed to persist {mode}-level UMAP model ({task['umap_path']}): {exc}")

    transformed = [model.transform(coords) for coords in coords_list]
    del model

    xlim, ylim = compute_frame_limits(transformed)
    categories, legend_kwargs = categories_for_mode(mode)
    frames = build_snapshot_frames(
        transformed, bundle["snapshot_metadata"], labels_list, onoff_list, task["display_method"]
    )

    frame_paths = render_latent_gif_frames(
        frames,
        categories=categories,
        xlim=xlim,
        ylim=ylim,
        legend_kwargs=legend_kwargs,
        out_dir=task["frames_dir"],
        method_name=task["method_name"],
        n_workers=1,  # combos are the parallel unit; frames render inline per combo
    )

    n_frames = len(frame_paths)
    if n_frames > 0:
        # Assemble the GIF by streaming one frame at a time (moved verbatim from train)
        with iio.imopen(task["gif_path"], "w", plugin="pillow") as gif_writer:
            for frame_path in frame_paths:
                frame = iio.imread(frame_path)
                gif_writer.write(
                    frame,
                    duration=task["duration_ms"],
                    loop=0,
                    is_batch=False,
                )
                del frame

    return {
        "method_name": task["method_name"],
        "display_method": task["display_method"],
        "gif_path": task["gif_path"],
        "umap_path": task["umap_path"],
        "n_frames": n_frames,
        "warnings": warnings,
    }


def run_umap_gif_sweep(tasks: list[dict], n_workers: int) -> list[dict]:
    """
    Run every (mode, nn, md) combo across a forkserver pool (empty preload, threads pinned —
    the shap_parallel isolation pattern), or serially in-process when n_workers <= 1. Results
    come back in task order. Outputs are byte-identical either way: each combo's fit consumes
    only its own derived random_state, and process isolation cannot change that stream.
    """
    if not tasks:
        return []

    n_workers = max(1, min(n_workers, len(tasks)))
    if n_workers == 1:
        # In-process combos cache the shared bundle in THIS process's module globals — clear
        # them afterwards so the caller's pre-sweep memory cleanup (train.py dels its copies
        # of the bundled inputs) isn't quietly undone by a lingering multi-hundred-MB cache.
        # The pooled arm needs no equivalent: its cache lives in short-lived workers.
        global _SWEEP_BUNDLE, _SWEEP_BUNDLE_PATH
        try:
            return [run_umap_gif_combo(task) for task in tasks]
        finally:
            _SWEEP_BUNDLE = None
            _SWEEP_BUNDLE_PATH = None

    ctx = mp.get_context("forkserver")
    ctx.set_forkserver_preload([])
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx, initializer=_sweep_worker_init
    ) as pool:
        return list(pool.map(run_umap_gif_combo, tasks))
