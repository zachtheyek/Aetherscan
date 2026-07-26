#!/usr/bin/env python3
"""
Latent-GIF stage benchmark (#278): confirm where the ~24-29 h vae_plots GIF tail goes, then
measure each candidate optimization against the baseline — WITH output-equality checks, since
issue #278 requires the produced GIFs to be unchanged.

Phases (all timings per (n_neighbors, min_dist) combo unless noted):

    fit        UMAP.fit on a production-sized stratified pool, plus the precomputed-knn
               variant (kNN graph shared across min_dist values). Reports whether the
               precomputed-knn embedding is IDENTICAL to the direct fit — if not, the reuse
               changes the GIFs and must be rejected (UMAP consumes its random_state stream
               differently when the kNN step is skipped).
    transform  per-frame serial .transform() calls (the shipped behavior) vs one batched
               .transform() on the stacked frames. Reports the max abs divergence — UMAP's
               transform optimizes the query batch jointly, so batching may not be
               output-identical either.
    render     the pre-#278 per-frame figure/scatter/savefig loop (inline replica) vs
               aetherscan.latent_gif.render_latent_gif_frames at 1 worker and N workers.
               Verifies N-worker PNGs are byte-identical to 1-worker PNGs.
    assemble   imageio GIF assembly from the rendered frames.
    extras     rough scale timings for the two other UMAP sites #278 flags as unmeasured:
               a decision-boundary-scale transform + inverse_transform grid, and a
               SHAP-clustering-scale UMAP fit + KMeans.

Run inside the container on a cluster (umap/matplotlib/imageio come from the image; no TF
or GPU needed — aetherscan.latent_gif is deliberately off the TF import graph):

    ./utils/run_container.sh python benchmarks/bench_latent_gif.py --mode all

Writes a JSON result to benchmarks/results/ (or --output), like the other benchmarks.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time

import numpy as np
from _common import machine_info, write_result

_SIGNAL_TYPES = ("false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi")
_OBS_COLORS = {
    ("false_no_signal", "ON"): "#1565C0",
    ("false_no_signal", "OFF"): "#64B5F6",
    ("false_with_rfi", "ON"): "#F9A825",
    ("false_with_rfi", "OFF"): "#FFF176",
    ("true_only_eti", "ON"): "#2E7D32",
    ("true_only_eti", "OFF"): "#81C784",
    ("true_eti_rfi", "ON"): "#C62828",
    ("true_eti_rfi", "OFF"): "#EF5350",
}
_OBS_MARKERS = {"ON": "^", "OFF": "x"}


def _synthesize_snapshots(n_frames: int, points_per_frame: int, latent_dim: int):
    """Drifting Gaussian-mixture latents: per-frame (N, latent_dim) coords + 8-class strata."""
    rng = np.random.default_rng(0)
    n_classes = len(_SIGNAL_TYPES) * 2
    centers = rng.normal(0, 3, size=(n_classes, latent_dim)).astype(np.float32)
    drift = rng.normal(0, 0.05, size=(n_classes, latent_dim)).astype(np.float32)

    per_class = points_per_frame // n_classes
    labels, onoff = [], []
    for stype in _SIGNAL_TYPES:
        for status in ("ON", "OFF"):
            labels += [stype] * per_class
            onoff += [status] * per_class
    labels = np.array(labels, dtype="U20")
    onoff = np.array(onoff, dtype="U3")

    frames = []
    for frame_idx in range(n_frames):
        coords = np.concatenate(
            [
                rng.normal(0, 1, size=(per_class, latent_dim)).astype(np.float32)
                + centers[class_idx]
                + frame_idx * drift[class_idx]
                for class_idx in range(n_classes)
            ]
        )
        frames.append(coords)
    return frames, labels, onoff


def _legacy_render(transformed, labels, onoff, xlim, ylim, out_dir):
    """Verbatim replica of the pre-#278 per-frame render loop (fresh figure + scatter +
    savefig + close per frame)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    paths = []
    for frame_idx, coords_2d in enumerate(transformed):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        for (stype, status), color in _OBS_COLORS.items():
            mask = (labels == stype) & (onoff == status)
            if mask.any():
                ax.scatter(
                    coords_2d[mask, 0],
                    coords_2d[mask, 1],
                    c=color,
                    marker=_OBS_MARKERS[status],
                    s=10,
                    alpha=0.75,
                    label=f"{stype} ({status})",
                    rasterized=True,
                )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"Bench frame {frame_idx}", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, markerscale=2, ncol=2, framealpha=0.8)
        plt.tight_layout()
        path = os.path.join(out_dir, f"legacy_frame_{frame_idx:05d}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        paths.append(path)
    return paths


def _limits(transformed):
    x_min = min(t[:, 0].min() for t in transformed)
    x_max = max(t[:, 0].max() for t in transformed)
    y_min = min(t[:, 1].min() for t in transformed)
    y_max = max(t[:, 1].max() for t in transformed)
    return (
        (x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min)),
        (y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["fit", "transform", "render", "extras", "all"],
        default="all",
    )
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--points-per-frame", type=int, default=23_040)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--fit-pool", type=int, default=100_000)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import umap  # noqa: PLC0415
    from umap.umap_ import nearest_neighbors  # noqa: PLC0415

    from aetherscan.latent_gif import (  # noqa: PLC0415
        FrameCategory,
        batched_umap_transform,
        render_latent_gif_frames,
    )

    results: dict = {}
    frames, labels, onoff = _synthesize_snapshots(
        args.frames, args.points_per_frame, args.latent_dim
    )
    pooled = np.concatenate(frames)
    fit_pool = pooled[
        np.random.default_rng(1).choice(
            len(pooled), size=min(args.fit_pool, len(pooled)), replace=False
        )
    ]
    print(f"frames={args.frames} points/frame={args.points_per_frame} fit_pool={len(fit_pool)}")

    # --- fit phase: direct vs precomputed-knn (shared across min_dist) --------------------
    reducer = None
    if args.mode in ("fit", "transform", "render", "assemble", "all"):
        start = time.perf_counter()
        reducer = umap.UMAP(
            n_components=2, random_state=11, n_neighbors=args.n_neighbors, min_dist=0.1
        ).fit(fit_pool)
        results["fit_direct_s"] = time.perf_counter() - start
        print(f"fit_direct_s: {results['fit_direct_s']:.1f}")

    if args.mode in ("fit", "all"):
        start = time.perf_counter()
        knn = nearest_neighbors(
            fit_pool,
            n_neighbors=args.n_neighbors,
            metric="euclidean",
            metric_kwds=None,
            angular=False,
            random_state=11,
            n_jobs=1,
        )
        results["knn_graph_s"] = time.perf_counter() - start
        start = time.perf_counter()
        reducer_knn = umap.UMAP(
            n_components=2,
            random_state=11,
            n_neighbors=args.n_neighbors,
            min_dist=0.1,
            precomputed_knn=knn,
        ).fit(fit_pool)
        results["fit_with_precomputed_knn_s"] = time.perf_counter() - start
        diff = float(np.abs(reducer.embedding_ - reducer_knn.embedding_).max())
        results["precomputed_knn_max_abs_diff"] = diff
        results["precomputed_knn_output_identical"] = bool(diff == 0.0)
        print(
            f"knn_graph_s: {results['knn_graph_s']:.1f}  "
            f"fit_with_precomputed_knn_s: {results['fit_with_precomputed_knn_s']:.1f}  "
            f"max_abs_diff: {diff:.6g}"
        )
        del reducer_knn, knn

    # --- transform phase: serial per-frame vs one batched call ----------------------------
    transformed = None
    if args.mode in ("transform", "render", "assemble", "all"):
        start = time.perf_counter()
        transformed = [reducer.transform(c) for c in frames]
        results["transform_serial_s"] = time.perf_counter() - start
        print(f"transform_serial_s: {results['transform_serial_s']:.1f}")

    if args.mode in ("transform", "all"):
        start = time.perf_counter()
        batched = batched_umap_transform(reducer, frames)
        results["transform_batched_s"] = time.perf_counter() - start
        diff = float(max(np.abs(a - b).max() for a, b in zip(transformed, batched, strict=True)))
        results["transform_batched_max_abs_diff"] = diff
        results["transform_batched_output_identical"] = bool(diff == 0.0)
        print(
            f"transform_batched_s: {results['transform_batched_s']:.1f}  max_abs_diff: {diff:.6g}"
        )
        del batched

    # --- render phase: legacy loop vs pool (1 worker and N workers) -----------------------
    if args.mode in ("render", "all"):
        xlim, ylim = _limits(transformed)
        categories = [
            FrameCategory(
                signal_type=stype,
                onoff=status,
                color=color,
                marker=_OBS_MARKERS[status],
                display_name=f"{stype} ({status})",
            )
            for (stype, status), color in _OBS_COLORS.items()
        ]
        legend_kwargs = {
            "loc": "upper right",
            "fontsize": 8,
            "markerscale": 2,
            "ncol": 2,
            "framealpha": 0.8,
        }
        frame_dicts = [
            {
                "coords": coords,
                "labels": labels,
                "onoff": onoff,
                "title": f"Bench frame {frame_idx}",
            }
            for frame_idx, coords in enumerate(transformed)
        ]

        work_dir = tempfile.mkdtemp(prefix="bench_latent_gif_")
        try:
            legacy_dir = os.path.join(work_dir, "legacy")
            os.makedirs(legacy_dir)
            start = time.perf_counter()
            _legacy_render(transformed, labels, onoff, xlim, ylim, legacy_dir)
            results["render_legacy_s"] = time.perf_counter() - start
            results["render_legacy_s_per_frame"] = results["render_legacy_s"] / args.frames
            print(
                f"render_legacy_s: {results['render_legacy_s']:.1f} "
                f"({results['render_legacy_s_per_frame']:.2f}/frame)"
            )

            pool1_dir = os.path.join(work_dir, "pool1")
            start = time.perf_counter()
            paths_1 = render_latent_gif_frames(
                frame_dicts,
                categories=categories,
                xlim=xlim,
                ylim=ylim,
                legend_kwargs=legend_kwargs,
                out_dir=pool1_dir,
                method_name="bench",
                n_workers=1,
            )
            results["render_pool_1worker_s"] = time.perf_counter() - start
            print(f"render_pool_1worker_s: {results['render_pool_1worker_s']:.1f}")

            pooln_dir = os.path.join(work_dir, "pooln")
            start = time.perf_counter()
            paths_n = render_latent_gif_frames(
                frame_dicts,
                categories=categories,
                xlim=xlim,
                ylim=ylim,
                legend_kwargs=legend_kwargs,
                out_dir=pooln_dir,
                method_name="bench",
                n_workers=args.workers,
            )
            results["render_pool_nworkers_s"] = time.perf_counter() - start
            results["render_pool_workers"] = args.workers
            print(f"render_pool_nworkers_s: {results['render_pool_nworkers_s']:.1f}")

            def _same_bytes(path_a: str, path_b: str) -> bool:
                with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
                    return fa.read() == fb.read()

            identical = all(_same_bytes(a, b) for a, b in zip(paths_1, paths_n, strict=True))
            results["render_pool_output_identical"] = identical
            print(f"render_pool_output_identical: {identical}")

            # --- assemble phase ------------------------------------------------------------
            import imageio.v3 as iio  # noqa: PLC0415

            gif_path = os.path.join(work_dir, "bench.gif")
            start = time.perf_counter()
            with iio.imopen(gif_path, "w", plugin="pillow") as gif_writer:
                for path in paths_n:
                    gif_writer.write(iio.imread(path), duration=200, loop=0, is_batch=False)
            results["assemble_gif_s"] = time.perf_counter() - start
            print(f"assemble_gif_s: {results['assemble_gif_s']:.1f}")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    # --- extras: the two other UMAP sites flagged unmeasured in #278 ----------------------
    if args.mode in ("extras", "all"):
        from sklearn.cluster import KMeans  # noqa: PLC0415

        cadence_pool = np.random.default_rng(2).normal(size=(10_000, 48)).astype(np.float32)
        start = time.perf_counter()
        boundary_umap = umap.UMAP(
            n_components=2, random_state=11, n_neighbors=args.n_neighbors, min_dist=0.1
        ).fit(cadence_pool)
        results["extras_boundary_fit_s"] = time.perf_counter() - start
        start = time.perf_counter()
        grid_2d = np.stack(
            np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100)), axis=-1
        ).reshape(-1, 2)
        try:
            boundary_umap.inverse_transform(grid_2d.astype(np.float32))
            results["extras_boundary_inverse_transform_s"] = time.perf_counter() - start
        except Exception as exc:  # inverse_transform can fail off-manifold
            results["extras_boundary_inverse_transform_s"] = None
            print(f"inverse_transform failed (also possible in production): {exc}")

        shap_pool = np.random.default_rng(3).normal(size=(20_000, 48)).astype(np.float32)
        start = time.perf_counter()
        umap.UMAP(n_components=2, random_state=11, n_neighbors=15, min_dist=0.1).fit(shap_pool)
        results["extras_shap_umap_fit_s"] = time.perf_counter() - start
        start = time.perf_counter()
        KMeans(n_clusters=4, random_state=11, n_init=10).fit(shap_pool)
        results["extras_shap_kmeans_s"] = time.perf_counter() - start
        print(
            f"extras: boundary_fit={results['extras_boundary_fit_s']:.1f}s "
            f"shap_umap={results['extras_shap_umap_fit_s']:.1f}s "
            f"kmeans={results['extras_shap_kmeans_s']:.1f}s"
        )

    path = write_result(
        "bench_latent_gif",
        {
            "mode": args.mode,
            "frames": args.frames,
            "points_per_frame": args.points_per_frame,
            "latent_dim": args.latent_dim,
            "fit_pool": len(fit_pool),
            "n_neighbors": args.n_neighbors,
            "workers": args.workers,
            "host": machine_info()["hostname"],
        },
        results,
        args.output,
    )
    print(f"Result written to {path}")


if __name__ == "__main__":
    main()
