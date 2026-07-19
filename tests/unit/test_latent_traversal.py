"""Unit tests for the latent-traversal math in aetherscan.train: the latent grid builder,
the stub-decodable panel computation, the display un-preprocessing (log-norm inversion +
frequency upsampling), and plot_latent_traversal's degenerate-input guards."""

from __future__ import annotations

import os

import numpy as np
import pytest

from aetherscan.config import get_config
from aetherscan.data_generation import log_norm
from aetherscan.train import (
    TrainingPipeline,
    build_traversal_latents,
    compute_traversal_panels,
    unpreprocess_traversal_panels,
)

_LATENT_DIM = 8
_NUM_STEPS = 7
_MAX_SIGMA = 3.0


@pytest.fixture
def z_base():
    return np.linspace(-1.0, 1.0, _LATENT_DIM).astype(np.float32)


@pytest.fixture
def sigmas():
    return (0.1 + np.arange(_LATENT_DIM, dtype=np.float32) * 0.05).astype(np.float32)


class TestBuildTraversalLatents:
    def test_shapes_and_steps(self, z_base, sigmas):
        latents, steps = build_traversal_latents(z_base, sigmas, _NUM_STEPS, _MAX_SIGMA)
        assert latents.shape == (_LATENT_DIM * _NUM_STEPS, _LATENT_DIM)
        assert latents.dtype == np.float32
        assert steps.shape == (_NUM_STEPS,)
        assert steps[0] == -_MAX_SIGMA
        assert steps[-1] == _MAX_SIGMA
        # Odd step count -> the center step is exactly the unperturbed decode.
        assert steps[_NUM_STEPS // 2] == 0.0
        np.testing.assert_allclose(steps, -steps[::-1])  # symmetric grid

    def test_center_rows_are_unperturbed(self, z_base, sigmas):
        latents, _ = build_traversal_latents(z_base, sigmas, _NUM_STEPS, _MAX_SIGMA)
        for d in range(_LATENT_DIM):
            center_row = latents[d * _NUM_STEPS + _NUM_STEPS // 2]
            np.testing.assert_array_equal(center_row, z_base)

    def test_only_target_dim_is_perturbed_by_step_times_sigma(self, z_base, sigmas):
        latents, steps = build_traversal_latents(z_base, sigmas, _NUM_STEPS, _MAX_SIGMA)
        for d in range(_LATENT_DIM):
            for s in range(_NUM_STEPS):
                row = latents[d * _NUM_STEPS + s]
                delta = row - z_base
                expected = np.zeros(_LATENT_DIM, dtype=np.float32)
                expected[d] = steps[s] * sigmas[d]
                np.testing.assert_allclose(delta, expected, atol=1e-6)

    @pytest.mark.parametrize("max_sigma", [2.5, 3.0, 0.7])
    def test_center_step_is_exactly_zero_for_any_max_sigma(self, z_base, sigmas, max_sigma):
        # linspace can leave ~1e-16 residue at the center for non-integral ranges; the
        # builder must snap it so the center column is the exact base decode.
        _, steps = build_traversal_latents(z_base, sigmas, _NUM_STEPS, max_sigma)
        assert steps[_NUM_STEPS // 2] == 0.0

    def test_mismatched_shapes_raise(self, z_base):
        with pytest.raises(ValueError, match="matching 1-D"):
            build_traversal_latents(z_base, np.ones(_LATENT_DIM + 1), _NUM_STEPS, _MAX_SIGMA)
        with pytest.raises(ValueError, match="matching 1-D"):
            build_traversal_latents(np.ones((2, _LATENT_DIM)), np.ones(_LATENT_DIM), 3, 1.0)


class TestComputeTraversalPanels:
    """Stub-decoder tests: an identity-ish decoder that paints each latent vector across the
    panel's frequency axis, so panel content directly reveals which latent was decoded."""

    @staticmethod
    def _identity_decoder(latents):
        # (n, latent_dim) -> (n, time, latent_dim): every time row equals the latent vector.
        return np.repeat(latents[:, None, :], 4, axis=1)

    def test_panels_are_row_major_over_dims_then_steps(self, z_base, sigmas):
        panels, steps = compute_traversal_panels(
            z_base, sigmas, _NUM_STEPS, _MAX_SIGMA, self._identity_decoder
        )
        assert panels.shape == (_LATENT_DIM, _NUM_STEPS, 4, _LATENT_DIM)
        for d in range(_LATENT_DIM):
            for s in range(_NUM_STEPS):
                expected = z_base.copy()
                expected[d] += steps[s] * sigmas[d]
                # Every time row of the identity decode equals the traversal latent.
                np.testing.assert_allclose(panels[d, s], np.tile(expected, (4, 1)), atol=1e-6)

    def test_center_column_is_base_decode(self, z_base, sigmas):
        panels, _ = compute_traversal_panels(
            z_base, sigmas, _NUM_STEPS, _MAX_SIGMA, self._identity_decoder
        )
        base_decode = self._identity_decoder(z_base[None, :])[0]
        for d in range(_LATENT_DIM):
            np.testing.assert_array_equal(panels[d, _NUM_STEPS // 2], base_decode)

    def test_trailing_channel_axis_is_squeezed(self, z_base, sigmas):
        def channel_decoder(latents):
            return self._identity_decoder(latents)[..., None]  # (n, t, f, 1)

        panels, _ = compute_traversal_panels(
            z_base, sigmas, _NUM_STEPS, _MAX_SIGMA, channel_decoder
        )
        assert panels.shape == (_LATENT_DIM, _NUM_STEPS, 4, _LATENT_DIM)

    def test_bad_decoder_output_raises(self, z_base, sigmas):
        with pytest.raises(ValueError, match="decode_fn"):
            compute_traversal_panels(
                z_base,
                sigmas,
                _NUM_STEPS,
                _MAX_SIGMA,
                lambda latents: latents,  # 2-D output
            )
        with pytest.raises(ValueError, match="decode_fn"):
            compute_traversal_panels(
                z_base,
                sigmas,
                _NUM_STEPS,
                _MAX_SIGMA,
                lambda latents: np.zeros((1, 4, 4)),  # wrong leading dim
            )


class TestUnpreprocessTraversalPanels:
    def test_lognorm_inversion_math(self):
        rng = np.random.default_rng(0)
        panels = rng.random((2, 3, 4, 8))
        min_log, range_log = -2.0, 3.0
        out, inverted = unpreprocess_traversal_panels(panels, (min_log, range_log), 1)
        assert inverted is True
        np.testing.assert_allclose(out, np.exp(panels * range_log + min_log))

    def test_no_params_stays_in_normalized_space(self):
        panels = np.random.default_rng(1).random((2, 3, 4, 8))
        out, inverted = unpreprocess_traversal_panels(panels, None, 1)
        assert inverted is False
        np.testing.assert_array_equal(out, panels)

    def test_degenerate_range_skips_inversion(self):
        panels = np.random.default_rng(2).random((2, 3, 4, 8))
        out, inverted = unpreprocess_traversal_panels(panels, (0.5, 0.0), 1)
        assert inverted is False
        np.testing.assert_array_equal(out, panels)

    def test_downsample_undone_by_nearest_neighbor_repeat(self):
        panels = np.arange(2 * 3 * 4 * 8, dtype=float).reshape(2, 3, 4, 8)
        out, _ = unpreprocess_traversal_panels(panels, None, 8)
        assert out.shape == (2, 3, 4, 64)
        np.testing.assert_array_equal(out, np.repeat(panels, 8, axis=-1))
        # Nearest-neighbor: each source bin becomes a constant run of 8.
        assert np.all(out[..., 0:8] == panels[..., 0:1])

    def test_roundtrip_against_log_norm(self):
        # Un-preprocessing a log_norm'ed observation with its own recorded params recovers
        # the (epsilon-shifted) linear intensities.
        rng = np.random.default_rng(3)
        data = rng.chisquare(df=4, size=(16, 32))
        normalized, params = log_norm(data, return_params=True)
        out, inverted = unpreprocess_traversal_panels(normalized[None, None], params, 1)
        assert inverted is True
        np.testing.assert_allclose(out[0, 0], data + 1e-10, rtol=1e-6)


class TestPlotLatentTraversalGuards:
    """plot_latent_traversal's degenerate-input guards, driven through a duck-typed pipeline
    (object.__new__ skips the heavyweight __init__) with stub encoder/decoder."""

    def _stub_pipeline(self, batch, labels):
        config = get_config()
        latent_dim = config.beta_vae.latent_dim

        class _CollapsedEncoder:
            def __call__(self, x, training=False):
                z = np.zeros((np.asarray(x).shape[0], latent_dim), dtype=np.float32)
                return z, z, z

        class _ExplodingDecoder:
            def __call__(self, z, training=False):
                raise AssertionError("decoder must not be called for a degenerate traversal")

        class _StubVAE:
            encoder = _CollapsedEncoder()
            decoder = _ExplodingDecoder()

        pipeline = object.__new__(TrainingPipeline)
        pipeline.config = config
        pipeline.vae = _StubVAE()
        pipeline._latent_viz_batch = batch
        pipeline._latent_viz_labels = labels
        pipeline._latent_viz_lognorm_params = None
        return pipeline

    def _no_traversal_figures(self, config):
        plots_dir = os.path.join(config.output_path, "plots")
        return not os.path.isdir(plots_dir) or not any(
            "latent_traversal" in name for name in os.listdir(plots_dir)
        )

    def test_collapsed_latents_skip_rendering(self):
        # An encoder whose latents all collapse to one point yields all-zero per-dim sigmas:
        # the plot must skip (before touching the decoder) instead of rendering blank grids.
        config = get_config()
        time_bins = config.data.time_bins
        width = config.data.width_bin // config.data.downsample_factor
        batch = np.random.default_rng(0).random((2, 6, time_bins, width)).astype(np.float32)
        labels = np.array(["true_only_eti", "false_no_signal"], dtype="U20")

        pipeline = self._stub_pipeline(batch, labels)
        pipeline.plot_latent_traversal()

        assert self._no_traversal_figures(config)

    def test_missing_viz_batch_skips_rendering(self):
        # A resumed run whose beta-VAE rounds were already complete never builds the viz
        # batch — the plot must warn and return, not raise.
        pipeline = self._stub_pipeline(None, None)
        pipeline.plot_latent_traversal()

        assert self._no_traversal_figures(get_config())
