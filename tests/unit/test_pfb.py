"""Unit tests for aetherscan.pfb: windowed-sinc filter design (cross-checked against a
transcription of the bliss C++ sinc approximation), the folded coarse-channel response's
shape properties (symmetry, mid-band peak, edge rolloff), equalization flatness on synthetic
PFB-shaped noise, and the edge/mid power-ratio validation statistic."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aetherscan.pfb import (
    edge_mid_power_ratio,
    equalize_passband,
    firdes,
    gen_coarse_channel_response,
)

# Small but representative geometry: 4 coarse channels of 512 fine channels, GBT-default taps
_FINE = 512
_NUM_COARSE = 4
_TAPS = 12


def _cpp_sinc(t: float) -> float:
    """Transcription of the bliss C++ sinc helper (passband_static_equalize.cpp): normalized
    sinc evaluated as sin(pi t)/(pi t), with a truncated cosine-product approximation
    cos(x/2)*cos(x/4)*cos(x/8) for |pi t| < 0.01. Transcribed here (test-only, no bliss
    import) to pin our exact np.sinc-based filter against the reference implementation."""
    x = math.pi * t
    if abs(x) < 0.01:
        return math.cos(x / 2.0) * math.cos(x / 4.0) * math.cos(x / 8.0)
    return math.sin(x) / x


def _cpp_firdes(num_taps: int, fc: float) -> np.ndarray:
    """Reference filter design: C++ sinc transcription x explicit Hamming window."""
    out = np.empty(num_taps, dtype=np.float64)
    center = (num_taps - 1) / 2.0
    for n in range(num_taps):
        window = 0.54 - 0.46 * math.cos(2.0 * math.pi * n / (num_taps - 1))
        out[n] = _cpp_sinc(fc * (n - center)) * window
    return out


class TestFirdes:
    @pytest.mark.parametrize(
        ("num_taps", "fc"),
        [(48, 1.0 / 4), (768, 1.0 / 64), (12 * 512, 1.0 / 512)],
    )
    def test_matches_cpp_reference_filter(self, num_taps, fc):
        ours = firdes(num_taps, fc)
        reference = _cpp_firdes(num_taps, fc)
        # np.sinc is exact where the C++ approximates; agreement must still be ~1e-6 or better
        np.testing.assert_allclose(ours, reference, atol=1e-6)

    def test_linear_phase_symmetry(self):
        # Windowed-sinc design is symmetric about the center tap (linear phase)
        h = firdes(_TAPS * _NUM_COARSE, 1.0 / _NUM_COARSE)
        np.testing.assert_allclose(h, h[::-1], atol=1e-15)


class TestGenCoarseChannelResponse:
    def test_shape_dtype_and_readonly(self):
        response = gen_coarse_channel_response(_FINE, _NUM_COARSE, _TAPS)
        assert response.shape == (_FINE,)
        assert response.dtype == np.float64
        # Cached array is shared across callers; it must be immutable
        assert not response.flags.writeable
        with pytest.raises(ValueError):
            response[0] = 0.0

    def test_symmetric_about_channel_center(self):
        # The response is even-symmetric excluding bin 0 (the unpaired Nyquist-edge bin of
        # the even-length spectrum)
        response = gen_coarse_channel_response(_FINE, _NUM_COARSE, _TAPS)
        np.testing.assert_allclose(response[1:], response[1:][::-1], atol=1e-12)

    def test_peaks_near_one_mid_band(self):
        response = gen_coarse_channel_response(_FINE, _NUM_COARSE, _TAPS)
        assert response.max() == pytest.approx(1.0)
        assert response[_FINE // 2] > 0.98
        # The peak sits in the passband, not at the rolled-off edges
        assert _FINE // 4 < int(response.argmax()) < 3 * _FINE // 4

    def test_rolls_off_at_edges(self):
        response = gen_coarse_channel_response(_FINE, _NUM_COARSE, _TAPS)
        # Classic PFB half-power rolloff at the channel boundary
        assert response[0] < 0.6
        assert response[-1] < 0.6
        assert response.min() >= response[0] - 1e-12

    def test_lru_cache_returns_same_object(self):
        a = gen_coarse_channel_response(_FINE, _NUM_COARSE, _TAPS)
        b = gen_coarse_channel_response(_FINE, _NUM_COARSE, _TAPS)
        assert a is b
        c = gen_coarse_channel_response(_FINE, _NUM_COARSE, _TAPS + 1)
        assert c is not a

    @pytest.mark.parametrize(
        ("fine", "num_coarse", "taps"),
        [(1, 4, 12), (512, 1, 12), (512, 4, 0)],
    )
    def test_invalid_arguments_raise(self, fine, num_coarse, taps):
        with pytest.raises(ValueError):
            gen_coarse_channel_response(fine, num_coarse, taps)


class TestEqualizePassband:
    def test_deterministic_shaped_input_becomes_exactly_flat(self):
        response = gen_coarse_channel_response(_FINE, _NUM_COARSE, _TAPS)
        shaped = np.ones((16, 1)) * response  # noise-free PFB-shaped "spectrum"
        equalized = equalize_passband(shaped, response)
        np.testing.assert_allclose(equalized, 1.0, rtol=1e-12)

    def test_synthetic_pfb_noise_integrates_flat(self):
        # Chi-squared-ish noise with mean 1 imprinted with the PFB shape: after
        # equalization the time-integrated spectrum must be flat to within a few percent,
        # while the un-equalized spectrum retains the full ~50% bandpass ripple.
        rng = np.random.default_rng(3)
        response = gen_coarse_channel_response(_FINE, 8, _TAPS)
        noise = rng.chisquare(df=64, size=(4096, _FINE)) / 64.0
        shaped = noise * response

        integrated_raw = shaped.mean(axis=0)
        raw_ripple = (integrated_raw.max() - integrated_raw.min()) / integrated_raw.mean()
        assert raw_ripple > 0.3

        integrated_eq = equalize_passband(shaped, response).mean(axis=0)
        eq_ripple = (integrated_eq.max() - integrated_eq.min()) / integrated_eq.mean()
        assert eq_ripple < 0.03

    def test_inputs_not_modified(self):
        response = gen_coarse_channel_response(_FINE, _NUM_COARSE, _TAPS)
        channel = np.full((4, _FINE), 2.0)
        original = channel.copy()
        equalize_passband(channel, response)
        np.testing.assert_array_equal(channel, original)

    def test_width_mismatch_raises(self):
        response = gen_coarse_channel_response(_FINE, _NUM_COARSE, _TAPS)
        with pytest.raises(ValueError, match="does not match"):
            equalize_passband(np.ones((4, _FINE + 1)), response)


class TestEdgeMidPowerRatio:
    def test_flat_spectrum_gives_one(self):
        assert edge_mid_power_ratio(np.ones(_FINE)) == pytest.approx(1.0)

    def test_pfb_response_ratio_reflects_rolloff(self):
        response = gen_coarse_channel_response(_FINE, _NUM_COARSE, _TAPS)
        ratio = edge_mid_power_ratio(response)
        # Edge bands sit on the rolloff, mid band on the passband peak
        assert 0.3 < ratio < 0.8

    def test_flat_vs_response_disagree_beyond_tolerance(self):
        # The pair the mismatch warning distinguishes: a recording without the PFB shape
        # (ratio ~1) vs the theoretical response — far more than 5% apart.
        response = gen_coarse_channel_response(_FINE, _NUM_COARSE, _TAPS)
        expected = edge_mid_power_ratio(response)
        measured = edge_mid_power_ratio(np.ones(_FINE))
        assert abs(measured - expected) / expected > 0.05
