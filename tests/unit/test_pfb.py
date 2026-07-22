"""Unit tests for aetherscan.pfb: windowed-sinc filter design (cross-checked against a
transcription of the bliss C++ sinc approximation and against the CASPER GBT512 reference
from PFBPassband.jl, including the hardware's bug=true half-step-offset quirk), the folded
coarse-channel response's shape properties (symmetry, mid-band peak, edge rolloff),
equalization flatness on synthetic PFB-shaped noise, and the edge/mid power-ratio
validation statistic."""

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


def _casper_firdes(taps_per_channel: int, nchan: int, bug: bool) -> np.ndarray:
    """Transcription of PFBPassband.jl's coefficient formula (src/PFBPassband.jl, coefs!):

        window(n) .* lpf.(width .* (((0:n-1) .+ 0.5*(1-bug)) ./ nchan .- ntaps/2))

    with the CASPER GBT512 preset window=hamming, lpf=sinc, width=1.0 (src/casperpfbs.jl).
    bug=True reproduces the CASPER coefficient-generation bug ("exclude half-step offset
    when pfb.bug is true"): the half-sample offset is omitted from the sinc argument,
    mis-centering the sinc by half a sample relative to the (unchanged) Hamming window.
    Transcribed here (test-only, no Julia dependency) to pin aetherscan's exact-sinc design
    against the hardware reference — see issue #180."""
    width = 1.0  # GBT512 preset
    n = taps_per_channel * nchan
    k = np.arange(n, dtype=np.float64)
    # PFBPassband's hamming (0.54 + 0.46*cospi(range(-1, 1, n))) == np.hamming(n)
    window = 0.54 - 0.46 * np.cos(2.0 * np.pi * k / (n - 1))
    offset = 0.5 * (1.0 - bug)
    return window * np.sinc(width * ((k + offset) / nchan - taps_per_channel / 2.0))


def _fold_response(prototype: np.ndarray, fine_per_coarse: int, num_coarse: int) -> np.ndarray:
    """Fold an arbitrary prototype filter into the per-coarse-channel power response,
    replicating gen_coarse_channel_response's algorithm (zero-pad -> |fftshift(FFT)|^2 ->
    slice half a coarse channel off each end -> sum spans -> normalize to peak 1.0)."""
    full_len = num_coarse * fine_per_coarse
    padded = np.zeros(full_len, dtype=np.float64)
    padded[: prototype.size] = prototype
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(padded))) ** 2
    half_fine = fine_per_coarse // 2
    sliced = spectrum[half_fine : full_len - half_fine]
    response = sliced.reshape(num_coarse - 1, fine_per_coarse).sum(axis=0)
    return response / response.max()


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


class TestCasperGbt512:
    """Codifies issue #180: aetherscan's firdes is exactly the bug-free CASPER GBT512 design,
    and the real hardware's bug=true response differs from the modeled one negligibly."""

    @pytest.mark.parametrize("nchan", [4, 64, 512, 1024])
    def test_firdes_equals_casper_bug_free_design(self, nchan):
        # bug=False must reproduce firdes to machine precision (measured ~1e-16): aetherscan
        # models the bug-free CASPER filter, at 12 taps/channel across the GBT geometry family
        # up to the true GBT512 nchan=1024.
        ours = firdes(_TAPS * nchan, 1.0 / nchan)
        reference = _casper_firdes(_TAPS, nchan, bug=False)
        np.testing.assert_allclose(ours, reference, rtol=0.0, atol=1e-12)

    def test_bug_true_response_delta_negligible(self):
        # GBT-like geometry: 64 coarse channels of 512 fine channels, 12 taps.
        fine, num_coarse = 512, 64
        response_exact = gen_coarse_channel_response(fine, num_coarse, _TAPS)

        # Sanity-pin the test-side fold against the production fold given the same prototype,
        # so the assertion below measures the prototype difference and nothing else.
        refolded = _fold_response(firdes(_TAPS * num_coarse, 1.0 / num_coarse), fine, num_coarse)
        np.testing.assert_allclose(refolded, response_exact, rtol=0.0, atol=1e-12)

        response_bug = _fold_response(_casper_firdes(_TAPS, num_coarse, bug=True), fine, num_coarse)
        rel_delta = np.abs(response_bug - response_exact) / response_exact
        # Measured ~9.4e-6 at this geometry (shrinks as taps*nchan grows; ~2.5e-3 only for a
        # degenerate 4-coarse-channel fold): 3-4 orders of magnitude below the 5% mismatch
        # warning tolerance, so keeping the exact bug-free design over the hardware's bug=true
        # form is a negligible modeling deviation.
        assert float(rel_delta.max()) < 1e-4


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
