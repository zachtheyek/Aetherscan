"""Unit tests for aetherscan.latent_gif (#278): the process-parallel GIF frame renderer must
be byte-identical across worker counts, and the batched-transform helper must preserve
per-snapshot splits. TF-free by design — the module mirrors shap_parallel's isolation."""

from __future__ import annotations

import numpy as np

from aetherscan.latent_gif import (
    FrameCategory,
    batched_umap_transform,
    render_latent_gif_frames,
)

_CATEGORIES = [
    FrameCategory("false_no_signal", "ON", "#1565C0", "^", "No Signal (ON)"),
    FrameCategory("false_no_signal", "OFF", "#64B5F6", "x", "No Signal (OFF)"),
    FrameCategory("true_only_eti", "ON", "#2E7D32", "^", "ETI Only (ON)"),
    FrameCategory("true_only_eti", "OFF", "#81C784", "x", "ETI Only (OFF)"),
]
_LEGEND = {"loc": "upper right", "fontsize": 8, "markerscale": 2, "framealpha": 0.8}


def _make_frames(n_frames: int, n_points: int = 40) -> list[dict]:
    rng = np.random.default_rng(7)
    labels = np.array(["false_no_signal", "true_only_eti"] * (n_points // 2), dtype="U20")
    onoff = np.array(["ON", "OFF"] * (n_points // 2), dtype="U3")
    return [
        {
            "coords": rng.normal(size=(n_points, 2)).astype(np.float32) + i,
            "labels": labels,
            "onoff": onoff,
            "title": f"frame {i}",
        }
        for i in range(n_frames)
    ]


class TestRenderLatentGifFrames:
    def test_parallel_output_is_byte_identical_to_single_worker(self, tmp_path):
        frames = _make_frames(6)
        paths_1 = render_latent_gif_frames(
            frames,
            categories=_CATEGORIES,
            xlim=(-3, 9),
            ylim=(-3, 9),
            legend_kwargs=_LEGEND,
            out_dir=str(tmp_path / "one"),
            method_name="bench",
            n_workers=1,
        )
        paths_n = render_latent_gif_frames(
            frames,
            categories=_CATEGORIES,
            xlim=(-3, 9),
            ylim=(-3, 9),
            legend_kwargs=_LEGEND,
            out_dir=str(tmp_path / "many"),
            method_name="bench",
            n_workers=3,
        )
        assert len(paths_1) == len(paths_n) == 6
        for path_1, path_n in zip(paths_1, paths_n, strict=True):
            with open(path_1, "rb") as f1, open(path_n, "rb") as f2:
                assert f1.read() == f2.read()

    def test_frame_paths_are_ordered_and_named_by_index(self, tmp_path):
        frames = _make_frames(4)
        paths = render_latent_gif_frames(
            frames,
            categories=_CATEGORIES,
            xlim=(-3, 6),
            ylim=(-3, 6),
            legend_kwargs=_LEGEND,
            out_dir=str(tmp_path),
            method_name="gifx",
            n_workers=2,
        )
        assert [p.rsplit("/", 1)[-1] for p in paths] == [
            f"gifx_frame_{i:05d}.png" for i in range(4)
        ]

    def test_worker_count_clamped_to_frames(self, tmp_path):
        frames = _make_frames(2)
        paths = render_latent_gif_frames(
            frames,
            categories=_CATEGORIES,
            xlim=(-3, 5),
            ylim=(-3, 5),
            legend_kwargs=_LEGEND,
            out_dir=str(tmp_path),
            method_name="clamp",
            n_workers=16,
        )
        assert len(paths) == 2

    def test_empty_frames_returns_empty(self, tmp_path):
        assert (
            render_latent_gif_frames(
                [],
                categories=_CATEGORIES,
                xlim=(0, 1),
                ylim=(0, 1),
                legend_kwargs=_LEGEND,
                out_dir=str(tmp_path),
                method_name="none",
                n_workers=4,
            )
            == []
        )

    def test_cadence_mode_without_onoff(self, tmp_path):
        # onoff=None (cadence-level): masking is by signal_type only, and categories with
        # no points in a frame simply render empty
        categories = [
            FrameCategory("false_no_signal", None, "tab:blue", "o", "No Signal"),
            FrameCategory("true_eti_rfi", None, "tab:orange", "o", "ETI+RFI"),
        ]
        frames = [
            {
                "coords": np.zeros((4, 2), dtype=np.float32),
                "labels": np.array(["false_no_signal"] * 4, dtype="U20"),
                "onoff": None,
                "title": "cadence frame",
            }
        ]
        paths = render_latent_gif_frames(
            frames,
            categories=categories,
            xlim=(-1, 1),
            ylim=(-1, 1),
            legend_kwargs=_LEGEND,
            out_dir=str(tmp_path),
            method_name="cad",
            n_workers=1,
        )
        assert len(paths) == 1


class TestBatchedUmapTransform:
    class _FakeModel:
        """Deterministic stand-in: transform = first two dims (order-preserving)."""

        def transform(self, coords):
            return np.asarray(coords)[:, :2] * 2.0

    def test_split_matches_serial_transforms(self):
        model = self._FakeModel()
        rng = np.random.default_rng(3)
        coords_list = [rng.normal(size=(n, 5)).astype(np.float32) for n in (4, 7, 1)]
        batched = batched_umap_transform(model, coords_list)
        serial = [model.transform(c) for c in coords_list]
        assert len(batched) == 3
        for got, expected in zip(batched, serial, strict=True):
            np.testing.assert_array_equal(got, expected)

    def test_empty_list(self):
        assert batched_umap_transform(self._FakeModel(), []) == []
