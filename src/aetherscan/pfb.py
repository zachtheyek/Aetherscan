"""
Polyphase filterbank (PFB) static passband equalization for Aetherscan's energy detection.

Radio-telescope backends (e.g. the GBT/Breakthrough Listen digital backend) channelize the
band with a polyphase filterbank whose prototype filter imprints the same scalloped passband
shape onto every coarse channel. Because that shape is a static property of the instrument's
filter design — not of the data — it can be computed once from the filter parameters and
divided out, instead of fitting a spline to every coarse channel of every file (the historical
default, and a real per-channel cost inside energy detection).

This is a native NumPy port of the response-generation logic in bliss
(github.com/n-west/bliss, bliss/preprocess/passband_static_equalize.cpp); it introduces no
bliss (or TensorFlow) dependency. One intentional deviation: the C++ approximates sinc near
zero with a truncated cosine product (cos(x/2)*cos(x/4)*cos(x/8) for |x| < 0.01) purely as a
numerical shortcut, while np.sinc evaluates the same normalized sinc exactly — the unit tests
pin the two filter designs against each other.

A separate, unrelated quirk lives in the instrument itself: the CASPER GBT512 configuration
(PFBPassband.jl) carries bug=true — the hardware's coefficient generation omits the
half-sample offset from the sinc argument (sinc evaluated at (n - N/2) / nchan instead of the
bug-free (n + 0.5 - N/2) / nchan, N = taps * nchan), mis-centering the sinc by half a sample
relative to the Hamming window. firdes below is exactly the bug-free design; the two folded
power responses agree to ~1e-5 (relative) at a typical GBT geometry (64 coarse channels,
12 taps — bounded < 1e-4 by the unit tests, shrinking as N grows), orders of magnitude below
the 5% mismatch-warning tolerance, so the exact bug-free form is kept (issue #180).

The response depends on (fine_per_coarse, num_coarse_channels, taps_per_channel) only, so
gen_coarse_channel_response is lru_cached: each process pays the one-time FFT (seconds at
full GBT resolution), after which equalizing a channel is a single vectorized divide.
"""

from __future__ import annotations

import functools
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Fraction of a coarse channel treated as the "edge" band (each side) and the "mid" band by
# edge_mid_power_ratio — the validation statistic used to detect a static-response mismatch.
_RATIO_BAND_FRACTION = 16


def firdes(num_taps: int, fc: float) -> np.ndarray:
    """
    Design a windowed-sinc (Hamming) lowpass FIR prototype filter.

    h[n] = sinc(fc * (n - (num_taps - 1) / 2)) * w[n], where w is the Hamming window
    w[n] = 0.54 - 0.46 * cos(2*pi*n / (num_taps - 1)) (== np.hamming) and np.sinc is the
    normalized sin(pi*t)/(pi*t). fc is the cutoff as a fraction of the sample rate
    (1/num_coarse_channels for a critically-sampled PFB). Returns float64 of shape (num_taps,).
    """
    n = np.arange(num_taps, dtype=np.float64)
    return np.sinc(fc * (n - (num_taps - 1) / 2.0)) * np.hamming(num_taps)


# maxsize=4 bounds memory if a long-lived process sees several parameter combinations
# (e.g. a test session); a production run only ever uses one.
@functools.lru_cache(maxsize=4)
def gen_coarse_channel_response(
    fine_per_coarse: int, num_coarse_channels: int, taps_per_channel: int
) -> np.ndarray:
    """
    Compute the static per-coarse-channel passband response H of a critically-sampled PFB.

    Steps (mirroring the bliss reference): design the prototype lowpass
    (taps_per_channel * num_coarse_channels taps, cutoff 1/num_coarse_channels); zero-pad to
    num_coarse_channels * fine_per_coarse points; take |fftshift(fft(.))|^2; drop half a
    coarse channel from each end so the remaining band is an integer number of coarse-channel
    spans centered on channel boundaries; fold those (num_coarse_channels - 1) spans on top of
    each other (summing in adjacent-channel leakage); normalize to a peak of 1.0.

    Returns H of shape (fine_per_coarse,), float64, read-only (the array is cached and shared
    across callers — copy before mutating). Cached per argument tuple, so each process pays
    the FFT cost once per distinct (file resolution, coarse count, taps) combination.
    """
    if fine_per_coarse < 2:
        raise ValueError(f"fine_per_coarse must be >= 2, got {fine_per_coarse}")
    if fine_per_coarse % 2 != 0:
        # Half a coarse channel (fine_per_coarse // 2) is sliced off each end below, so the
        # remaining band is an exact multiple of fine_per_coarse only when it is even.
        raise ValueError(f"fine_per_coarse must be even, got {fine_per_coarse}")
    if num_coarse_channels < 2:
        # The fold needs at least one full coarse-channel span after edge-slicing.
        raise ValueError(f"num_coarse_channels must be >= 2, got {num_coarse_channels}")
    if taps_per_channel < 1:
        raise ValueError(f"taps_per_channel must be >= 1, got {taps_per_channel}")

    h = firdes(taps_per_channel * num_coarse_channels, 1.0 / num_coarse_channels)
    full_len = num_coarse_channels * fine_per_coarse
    padded = np.zeros(full_len, dtype=np.float64)
    padded[: h.size] = h
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(padded))) ** 2

    half_fine = fine_per_coarse // 2
    sliced = spectrum[half_fine : full_len - half_fine]
    response = sliced.reshape(num_coarse_channels - 1, fine_per_coarse).sum(axis=0)
    response /= response.max()
    response.setflags(write=False)

    logger.debug(
        f"Generated PFB coarse-channel response: fine_per_coarse={fine_per_coarse}, "
        f"num_coarse_channels={num_coarse_channels}, taps_per_channel={taps_per_channel}"
    )
    return response


def equalize_passband(channel: np.ndarray, response: np.ndarray) -> np.ndarray:
    """
    Divide one coarse channel by the static passband response, broadcasting over time.

    channel has shape (time_bins, fine_per_coarse) and response (fine_per_coarse,) — e.g. the
    output of gen_coarse_channel_response. Returns the equalized channel (float64 when
    response is float64); the inputs are not modified.
    """
    if channel.shape[-1] != response.shape[0]:
        raise ValueError(
            f"channel width {channel.shape[-1]} does not match response length {response.shape[0]}"
        )
    # Defensive floor: the 12-tap GBT design bottoms out around 0.25-0.5 at channel edges,
    # but a very high taps_per_channel (sharper rolloff) could push edge bins toward zero
    # and turn the divide into inf. The floor sits far below any physical response value,
    # so realistic responses pass through untouched.
    return channel / np.maximum(response, 1e-10)


def edge_mid_power_ratio(spectrum: np.ndarray) -> float:
    """
    Mean power in the coarse-channel edge bands relative to the mid band.

    spectrum is a 1-D per-bin power array over one coarse channel (either the theoretical
    response H or a time-integrated data channel). Both the edge band (outermost n // 16 bins
    on each side) and the mid band (a central band of the same n // 16 width) use
    _RATIO_BAND_FRACTION. Comparing this ratio between H and real data is the cheap sanity
    check (after the bliss `validate` flag) for whether the configured static response
    actually matches the recording.
    """
    n = spectrum.shape[0]
    band = max(1, n // _RATIO_BAND_FRACTION)
    edge = 0.5 * (float(spectrum[:band].mean()) + float(spectrum[-band:].mean()))
    mid = float(spectrum[n // 2 - band // 2 : n // 2 + band // 2 + 1].mean())
    # Defensive floor (mirroring equalize_passband's): an all-zero — e.g. fully masked —
    # channel would otherwise raise ZeroDivisionError on the Python-float divide.
    return edge / max(mid, 1e-30)
