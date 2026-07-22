# TODO: add logging to background loading & downsampling
# TODO: update docstring once preprocessing.py complete
"""
Data preprocessing for Aetherscan Pipeline
Handles data loading, downsampling, log-normalization for both training & inference
Uses multiprocessing and shared memory to process data in parallel
"""

from __future__ import annotations

import contextlib
import csv
import functools
import gc
import json
import logging
import math
import os
import re
import signal
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from multiprocessing.shared_memory import SharedMemory

import h5py

# NOTE: come back to this later (why noqa: F401? what's the difference between h5py & hdf5plugin?)
import hdf5plugin  # noqa: F401  # registers bitshuffle codec with h5py at import time
import numpy as np
from scipy import interpolate, stats
from skimage.transform import downscale_local_mean

from aetherscan.benchmark import stage_timer
from aetherscan.config import get_config
from aetherscan.data_generation import log_norm
from aetherscan.db import get_db
from aetherscan.logger import init_worker_logging
from aetherscan.manager import get_manager
from aetherscan.pfb import edge_mid_band_slices, equalize_passband, gen_coarse_channel_response

logger = logging.getLogger(__name__)

# NOTE: find a way to avoid using global refs (store under manager.py maybe?)
# NOTE: is there any room to use asyncio & load all chunks simultaneously?
# Global variable to store chunk data for multiprocessing workers
# This avoids serialization overhead when passing data between workers
_GLOBAL_SHM = None
_GLOBAL_CHUNK_DATA = None
_GLOBAL_SHAPE = None
_GLOBAL_DTYPE = None

# PFB static-response sanity check: warn when the median flattened (divided-by-H) edge/mid
# power ratio deviates from 1.0 by more than this. ~0.10 is a provisional threshold — the
# residual still carries the analog-frontend tilt, whose legitimate baseline is only fixed by
# the deferred pfb_taps-vs-backend characterization — so the warning stays informational
# (#180 bounds the response-model error itself at ~1e-5, so H is not the confounder).
_PFB_RESIDUAL_FLATNESS_TOL = 0.10
# Fixed log-spaced bin edges for the energy-detection statistic summary histograms
# accumulated by _energy_detect_channel_worker. The edges are identical for every channel /
# file / cadence, so per-channel counts combine by plain addition; the range spans sub-noise
# k2 values (~1e-3) through extreme RFI statistics (~1e9) at 10 bins per decade. Consumed by
# inference_viz.plot_ed_stat_distributions via each cadence's metadata JSON.
ED_STAT_HIST_EDGES = np.logspace(-3.0, 9.0, 121)
# Coarse channels sampled (evenly across the band) by the opt-in bandpass overlay debug plot.
_BANDPASS_SAMPLE_CHANNELS = 3
# The PFB static-response sanity check samples more channels and takes a median (not a mean),
# so a single RFI-heavy channel doesn't skew the statistic.
_PFB_MISMATCH_SAMPLE_CHANNELS = 9
# Target bin count for _decimate_for_plot: spectrum lines in the bandpass debug/viz figures
# are reduced to at most 2 * this many points (a min/max pair per bin) before plotting.
_PLOT_MAX_POINTS_PER_LINE = 4096


def _init_worker(shm_name, shape, dtype):
    """
    Worker pool initializer: attach to the named shared-memory block and set up logging.

    Passing the shm_name/shape/dtype through the pool initializer (rather than per-task args)
    avoids re-serializing the whole array on every map() call. The worker installs a SIGTERM
    handler that closes its shared-memory file descriptor before letting the signal kill the
    process; the main process is responsible for unlinking the shared memory afterwards (handled
    by ResourceManager).
    """
    global _GLOBAL_SHM, _GLOBAL_CHUNK_DATA, _GLOBAL_SHAPE, _GLOBAL_DTYPE

    # Initialize worker logging
    init_worker_logging()

    # Attach to existing shared memory block
    _GLOBAL_SHM = SharedMemory(name=shm_name)

    # Ignore SIGINT (Ctrl+C) in workers - let manager from parent handle cleanup coordination
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Setup custom SIGTERM handler for additional cleanup before termination
    # Note, manager will escalate SIGTERM to SIGKILL after pool_terminate_timeout seconds (see config.py)
    # This may interrupt the worker's cleanup process
    # Consider increasing pool_terminate_timeout if you're experiencing such issues
    def cleanup_on_sigterm(signum, frame):
        """SIGTERM handler that closes the worker's shared-memory fd before re-raising the signal."""
        # Note, a race condition may occur if a worker receives more than 1 SIGTERM delivery
        # at a time, triggering re-entry of the same cleanup handler
        # It suffices to guard against this by simply suppressing exceptions, since subsequent
        # close() calls will just raise an error; no state corruption or kernel-level hazards exist
        # Also, there are no cross-worker race conditions, since each worker's close() operates on
        # per-process resources, even though they all refer to the same underlying POSIX shm object
        with contextlib.suppress(Exception):
            if _GLOBAL_SHM is not None:
                # WARN: DO NOT LOG ANYTHING ON CLEANUP!
                # Any calls to logger will attempt to put a message onto QueueHandler, whose feeder
                # thread needs the GIL to transfer data to the underlying pipe. However, the main
                # process may be holding the GIL (e.g. TF's prefetch threads during training)
                # This causes deadlocks, preventing workers from completing their SIGTERM handlers
                # logger.info(f"Closing shared memory file descriptor in worker PID {os.getpid()}")
                _GLOBAL_SHM.close()

        # Restore default handler and re-raise SIGTERM to resume termination
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    # Register SIGTERM handler for graceful cleanup on pool.terminate()
    signal.signal(signal.SIGTERM, cleanup_on_sigterm)

    # Create numpy array view of shared memory (no copy!)
    _GLOBAL_CHUNK_DATA = np.ndarray(shape, dtype=dtype, buffer=_GLOBAL_SHM.buf)
    _GLOBAL_SHAPE = shape
    _GLOBAL_DTYPE = dtype


def _init_plain_worker():
    """
    Worker pool initializer for pools that don't attach shared memory (energy detection and
    stamp extraction): set up queue-based logging and ignore SIGINT so the parent's
    ResourceManager coordinates shutdown. No SIGTERM handler is installed — these workers hold
    no shared-memory file descriptors, so the default termination behavior is safe.
    """
    init_worker_logging()
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# NOTE: come back to this later
def _downsample_worker(args):
    """
    Downsample one cadence's 6 observations in parallel.

    args is a (cadence_idx, downsample_factor, final_width) tuple — the cadence itself is
    pulled from _GLOBAL_CHUNK_DATA to avoid pickling it across the pool boundary. Returns the
    downsampled cadence of shape (6, 16, final_width), or None if the source cadence contained
    NaN/Inf or had non-positive max (treated as invalid).
    """
    cadence_idx, downsample_factor, final_width = args

    # Get cadence from global chunk data
    if _GLOBAL_CHUNK_DATA is not None:
        cadence = _GLOBAL_CHUNK_DATA[cadence_idx]

        # Skip invalid cadences
        if np.any(np.isnan(cadence)) or np.any(np.isinf(cadence)) or np.max(cadence) <= 0:
            return None

        # Downsample each observation separately
        downsampled_cadence = np.zeros((6, 16, final_width), dtype=np.float32)

        for obs_idx in range(6):
            downsampled_cadence[obs_idx] = downscale_local_mean(
                cadence[obs_idx], (1, downsample_factor)
            ).astype(np.float32)

        return downsampled_cadence

    else:
        logger.warning("No global chunk data available")
        return None


def _lognorm_worker(args):
    """
    Log-normalize one already-downsampled cadence in parallel.

    Counterpart to _downsample_worker for stamps that were downsampled at extraction time
    (see _extract_stamps_worker): the stored .npy is already at final width, so loading only
    needs the per-cadence log-norm. args is a (cadence_idx,) tuple — the cadence itself is
    pulled from _GLOBAL_CHUNK_DATA to avoid pickling it across the pool boundary. Returns the
    log-normalized cadence of shape (6, time_bins, final_width) as float32, or None if the
    source cadence contained NaN/Inf or had non-positive max (treated as invalid, matching
    _downsample_worker).
    """
    (cadence_idx,) = args

    if _GLOBAL_CHUNK_DATA is None:
        logger.warning("No global chunk data available")
        return None

    cadence = _GLOBAL_CHUNK_DATA[cadence_idx]

    # Skip invalid cadences (same validity rule as _downsample_worker)
    if np.any(np.isnan(cadence)) or np.any(np.isinf(cadence)) or np.max(cadence) <= 0:
        return None

    return log_norm(cadence.astype(np.float32))


# NOTE: come back to this later (mirrors preprocess_fine.py:72-75 from the reference implementation)
def _remove_dc_spike(
    block_data: np.ndarray, coarse_channel_width: int, n_coarse_channels: int
) -> None:
    """
    Interpolate over the 2-bin DC spike at the center of each coarse channel, in place.

    block_data has shape (time_bins, n_coarse_channels * coarse_channel_width); the channel
    layout is contiguous so the spike sits at i * coarse_channel_width + coarse_channel_width//2
    for each i.
    """
    half_chan = coarse_channel_width // 2
    for i in range(n_coarse_channels):
        dc_ind = i * coarse_channel_width + half_chan
        # The two replacements are asymmetric on purpose: dc_ind pulls from
        # (+1, -3) and dc_ind-1 from (+2, -2). This matches the reference
        # implementation byte-for-byte (FX196/SETI-Energy-Detection
        # preprocess_fine.py:72-75) — the bins immediately around the spike
        # are themselves contaminated, so the offsets reach across the spike
        # to clean neighbors on both sides.
        block_data[:, dc_ind] = (block_data[:, dc_ind + 1] + block_data[:, dc_ind - 3]) / 2
        block_data[:, dc_ind - 1] = (block_data[:, dc_ind + 2] + block_data[:, dc_ind - 2]) / 2


# NOTE: come back to this later (mirrors utils.py:17-22 from the reference implementation.)
def _fit_channel_bandpass(
    integrated_channel: np.ndarray, channel_width: int, spl_order: int
) -> np.ndarray:
    """
    Fit a spline bandpass to a time-integrated coarse channel and evaluate it at every bin.

    integrated_channel is a 1-D array of shape (channel_width,). spl_order controls the number
    of interior knots (higher = finer fit). Returns the per-bin bandpass fit of the same shape.
    """
    x = np.arange(channel_width)
    # Interior knots must lie strictly inside (x[0], x[-1]); knots[1:] drops the
    # leading 0, which satisfies that constraint for the default config
    # (channel_width=1048576, spl_order=16). A pathological config with very
    # small channel_width relative to spl_order could produce a knots array
    # that violates the constraint — splrep would raise then. The defaults
    # used in InferenceConfig are safe.
    knots = np.arange(0, channel_width, channel_width // spl_order + 1)
    spl = interpolate.splrep(x, integrated_channel, t=knots[1:])
    return interpolate.splev(x, spl)


def _spline_flatten_bandpass(channel: np.ndarray, spl_order: int) -> np.ndarray:
    """
    Spline bandpass flattener (--bandpass-method spline): fit a spline to the time-integrated
    coarse channel and subtract it from every time row. channel has shape
    (time_bins, coarse_channel_width); returns the float64 residuals of the same shape.

    Bandpass flattening is a pluggable stage: _process_cadence obtains a picklable callable via
    DataPreprocessor._get_bandpass_flattener() and ships it to the pool workers, so a flattener
    only needs to be a module-level function with the same (channel) -> flattened contract
    (see _pfb_flatten_bandpass for the other implementation).
    """
    integrated_channel = np.mean(channel, axis=0)
    fit = _fit_channel_bandpass(integrated_channel, channel.shape[1], spl_order)
    return channel - fit


# maxsize=4 bounds memory if a long-lived worker sees several response paths (unlikely in
# production — one per run — but plausible across a long test session).
@functools.lru_cache(maxsize=4)
def _load_pfb_response(response_path: str) -> np.ndarray:
    """
    Load (and per-process cache) a PFB passband response written by
    DataPreprocessor._ensure_pfb_response_file. The array is marked read-only because the
    cache shares it across every call in the process.
    """
    response = np.load(response_path)
    response.setflags(write=False)
    return response


def _pfb_flatten_bandpass(channel: np.ndarray, response_path: str) -> np.ndarray:
    """
    PFB static-equalization bandpass flattener (--bandpass-method pfb, the default): divide the
    channel by the instrument's precomputed polyphase-filterbank passband response instead of
    fitting a spline per channel per file. channel has shape (time_bins, coarse_channel_width);
    returns the equalized float64 channel of the same shape.

    The response is computed ONCE, in the parent process, and shipped to pool workers as the
    path of a small sidecar .npy (see _ensure_pfb_response_file) rather than as generation
    parameters: at GBT scale, generating the response is an ~n_chans-point FFT with a
    tens-of-GB transient, and every pool worker running that concurrently on its first task
    would exhaust the node's memory. Workers just read the ~8 MB file, cached per process by
    _load_pfb_response.

    # NOTE: the spline path *subtracts* its fit while this path *divides* by H, so equalized
    # channels keep their DC offset and a bin-dependent scale. That asymmetry is fine for
    # detection: the D'Agostino-Pearson normality statistic is built from skewness and
    # kurtosis, which are central moments (location cancels) normalized by powers of the
    # variance (scale cancels), so flattening the bandpass *shape* is all that matters.
    # The spline-vs-PFB detection comparison in the PR body is the empirical arbiter.
    """
    return equalize_passband(channel, _load_pfb_response(response_path))


def _decimate_for_plot(
    y: np.ndarray, max_points: int = _PLOT_MAX_POINTS_PER_LINE
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce a long 1-D line to at most 2 * max_points (x, y) points for plotting, via a per-bin
    min/max envelope: each of max_points equal bins contributes its extremes at their true
    indices, so narrowband features (RFI spikes, hits) survive where a plain stride would
    usually miss them. At GBT scale this turns each ~1M-point matplotlib line in the bandpass
    figures into ~8k points. Returns (x_indices, values); inputs already at or below the
    budget pass through unchanged. Unpack into ax.plot(*_decimate_for_plot(y), ...).
    """
    y = np.asarray(y)
    n = y.shape[0]
    if n <= 2 * max_points:
        return np.arange(n), y
    bin_size = -(-n // max_points)  # ceil division
    # Edge-pad the tail bin; padded positions duplicate y[-1], and any argmin/argmax landing
    # there is clipped back to n - 1 below, so no synthetic value can be selected.
    padded = np.pad(y, (0, bin_size * max_points - n), mode="edge")
    binned = padded.reshape(max_points, bin_size)
    offsets = np.arange(max_points) * bin_size
    pairs = np.stack([offsets + binned.argmin(axis=1), offsets + binned.argmax(axis=1)], axis=1)
    # np.unique dedupes + keeps ascending order: a constant-valued bin has argmin==argmax, which
    # would otherwise emit the same (idx, y) point twice; deduping halves those bins for free.
    idx = np.unique(np.minimum(np.sort(pairs, axis=1).reshape(-1), n - 1))
    return idx, y[idx]


def _sliding_normality_k2(channel: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """
    D'Agostino-Pearson normality statistic (k2) for every sliding window across one coarse
    channel, computed in closed form from per-block power sums instead of a per-window Python
    loop over scipy.stats.normaltest (which was the #1 cost of inference preprocessing).

    channel has shape (time_bins, width). Window j covers columns
    [j*step_size, j*step_size + window_size) — the same windows as the historical loop
    `range(0, width - window_size, step_size)` — and each window's sample is the flattened
    (time_bins, window_size) slice, so n = time_bins * window_size is constant across windows.
    Returns a float64 array of shape (n_windows,) whose entries match
    `scipy.stats.normaltest(window.flatten()).statistic` (unit tests pin the equivalence to
    rtol=1e-9); windows with zero variance yield NaN, matching scipy's behavior of returning
    NaN rather than a spurious statistic.

    Derivation: normaltest = skewtest.Z**2 + kurtosistest.Z**2 where both Z transforms are
    elementwise closed forms in (n, m2, m3, m4). The central moments come from per-block raw
    power sums S1..S4 accumulated in float64: blocks are sized so that every window is an
    exact run of adjacent blocks (block = step_size when step divides window — the fast
    path — else gcd(window, step)), window sums are length-(window//block) moving sums over
    the block sums, and m2/m3/m4 follow from the standard raw-to-central-moment identities.
    Raw-moment differencing loses precision when |mean| >> std, so the channel mean is
    subtracted up front (central moments are shift-invariant); bandpass-subtracted residuals
    are near-zero-mean anyway, and the rtol=1e-9 unit-test gate is the arbiter.
    """
    time_bins, width = channel.shape
    n = time_bins * window_size  # samples per window (scalar across all windows)
    if n < 8:
        # Mirror scipy.stats.skewtest's minimum-sample requirement
        raise ValueError(f"normality test requires >= 8 samples per window, got {n}")
    n_windows = len(range(0, width - window_size, step_size))
    if n_windows <= 0:
        return np.empty(0, dtype=np.float64)

    # astype always copies, so the in-place shift below can't mutate the caller's array.
    # Shift-invariance guard: subtracting the channel mean leaves every central moment
    # mathematically unchanged while keeping the S2/n - mean**2 style differencing below
    # well conditioned even when the residuals carry a DC offset.
    data = channel.astype(np.float64)
    data -= data.mean()

    # Per-column power sums over the time axis, accumulated row by row so temporaries stay at
    # (width,) rather than (time_bins, width) x 3.
    # NOTE: the Python-level loop over time_bins rows is deliberate — it caps peak memory at a
    # few (width,) float64 vectors per worker. With the default time_bins=16 the loop overhead
    # is negligible; if time_bins ever grows large, vectorize via (channel**k).sum(axis=0).
    p1 = np.zeros(width, dtype=np.float64)
    p2 = np.zeros(width, dtype=np.float64)
    p3 = np.zeros(width, dtype=np.float64)
    p4 = np.zeros(width, dtype=np.float64)
    for t in range(time_bins):
        row = data[t]
        row2 = row * row
        p1 += row
        p2 += row2
        p3 += row2 * row
        p4 += row2 * row2

    # Aggregate columns into block sums. The fast path (step divides window — the default
    # config: window 256, step 128) uses step-sized blocks; the general path falls back to
    # gcd-sized blocks so windows are still exact runs of adjacent blocks.
    block = step_size if window_size % step_size == 0 else math.gcd(window_size, step_size)
    edges = np.arange(0, width, block)
    blocks_per_window = window_size // block
    blocks_per_step = step_size // block
    # When block does not divide width (non-power-of-2 geometries), np.add.reduceat emits a
    # short trailing block spanning the ragged tail [edges[-1], width). No in-range window ever
    # reaches it — every window ends at a multiple of block that is < width, hence <= edges[-1]
    # — but drop it explicitly so a partial sum can never leak into a window and silently
    # corrupt k2. width // block == number of full blocks (== len(edges) when block divides
    # width, so this is a no-op there); slicing to it keeps every window sum exact.
    n_full_blocks = width // block

    def _window_sums(col_sums: np.ndarray) -> np.ndarray:
        block_sums = np.add.reduceat(col_sums, edges)[:n_full_blocks]
        # Moving sum of blocks_per_window adjacent blocks, sampled every blocks_per_step
        # blocks — one entry per window, no long cumulative accumulation (precision).
        view = np.lib.stride_tricks.sliding_window_view(block_sums, blocks_per_window)
        return view[::blocks_per_step].sum(axis=-1)[:n_windows]

    s1 = _window_sums(p1)
    s2 = _window_sums(p2)
    s3 = _window_sums(p3)
    s4 = _window_sums(p4)

    # Raw power sums -> central moments (float64 throughout)
    mean = s1 / n
    m2 = s2 / n - mean**2
    m3 = s3 / n - 3.0 * mean * (s2 / n) + 2.0 * mean**3
    m4 = s4 / n - 4.0 * mean * (s3 / n) + 6.0 * mean**2 * (s2 / n) - 3.0 * mean**4

    # Z transforms transcribed from scipy.stats._stats_py::skewtest / kurtosistest with
    # scalar n, so every n-dependent constant is computed once. Variable names follow scipy.
    with np.errstate(all="ignore"):
        # Zero-variance windows can't support either test; scipy returns NaN there (and a
        # tiny negative m2 from float cancellation would otherwise fabricate a huge k2).
        degenerate = m2 <= 0.0
        m2 = np.where(degenerate, np.nan, m2)

        # --- skewtest: Z1 from g1 = m3 / m2**1.5
        b2 = m3 / m2**1.5
        y = b2 * math.sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)))
        beta2 = (
            3.0
            * (n**2 + 27 * n - 70)
            * (n + 1)
            * (n + 3)
            / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9))
        )
        w2 = -1 + math.sqrt(2 * (beta2 - 1))
        delta = 1 / math.sqrt(0.5 * math.log(w2))
        alpha = math.sqrt(2.0 / (w2 - 1))
        y = np.where(y == 0, 1.0, y)
        z1 = delta * np.log(y / alpha + np.sqrt((y / alpha) ** 2 + 1))

        # --- kurtosistest: Z2 from b2 = m4 / m2**2
        b2k = m4 / m2**2
        e = 3.0 * (n - 1) / (n + 1)
        varb2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1.0) * (n + 3) * (n + 5))
        x = (b2k - e) / math.sqrt(varb2)
        sqrtbeta1 = (
            6.0
            * (n * n - 5 * n + 2)
            / ((n + 7) * (n + 9))
            * math.sqrt((6.0 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)))
        )
        a = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + math.sqrt(1 + 4.0 / (sqrtbeta1**2)))
        term1 = 1 - 2 / (9.0 * a)
        denom = 1 + x * math.sqrt(2 / (a - 4.0))
        term2 = np.sign(denom) * np.where(
            denom == 0.0, np.nan, np.power((1 - 2.0 / a) / np.abs(denom), 1 / 3.0)
        )
        z2 = (term1 - term2) / math.sqrt(2 / (9.0 * a))

        k2 = z1 * z1 + z2 * z2

    return k2


def _energy_detect_channel_worker(args: tuple) -> tuple[list[tuple], np.ndarray]:
    """
    Fused worker: run the complete energy-detection chain for one coarse channel — read the
    channel's h5 slice, remove the DC spike, flatten the bandpass, and threshold the vectorized
    normality statistic. Returns (hits, stat_hist): hits is a small list of
    (absolute_fine_channel_index, statistic, pvalue) tuples; stat_hist is the histogram of
    *all* finite window statistics (not just hits) over the fixed ED_STAT_HIST_EDGES bins, so
    the parent can cheaply accumulate the full per-file statistic distribution for the
    visualization suite. The bulky (time_bins, coarse_channel_width) intermediates never leave
    the worker, so no shared memory or per-block parent arrays are needed.

    args is (h5_path, channel_index, coarse_channel_width, time_bins, bandpass_flatten,
    window_size, step_size, stat_threshold). bandpass_flatten is a picklable callable
    (channel) -> residuals (see _spline_flatten_bandpass). Each worker opens its own h5py.File
    since h5py file handles are fork-unsafe to share.
    """
    (
        h5_path,
        channel_index,
        coarse_channel_width,
        time_bins,
        bandpass_flatten,
        window_size,
        step_size,
        stat_threshold,
    ) = args

    start = channel_index * coarse_channel_width
    end = (channel_index + 1) * coarse_channel_width
    with h5py.File(h5_path, "r") as hf:
        channel = hf["data"][:time_bins, 0, start:end]

    # In-place DC spike removal. The spike sits at the channel center and the interpolation
    # offsets reach at most ±3 bins around it (see _remove_dc_spike), so per-channel
    # processing touches exactly the same bins as the historical block-based path — the
    # offsets can never cross a coarse-channel boundary.
    _remove_dc_spike(channel, coarse_channel_width, 1)

    residuals = bandpass_flatten(channel)

    k2 = _sliding_normality_k2(residuals, window_size, step_size)

    # Summary histogram over all finite window statistics — a handful of numpy ops on ~1e4
    # values, negligible next to the k2 computation itself. Out-of-range values are clipped
    # onto the fixed edges so counts are never silently dropped.
    finite = k2[np.isfinite(k2)]
    stat_hist, _ = np.histogram(
        np.clip(finite, ED_STAT_HIST_EDGES[0], ED_STAT_HIST_EDGES[-1]), bins=ED_STAT_HIST_EDGES
    )

    hit_windows = np.nonzero(k2 > stat_threshold)[0]
    if hit_windows.size == 0:
        return [], stat_hist

    # p-values are only stored in metadata, so chi2.sf runs on the (small) hit subset only
    pvalues = stats.chi2.sf(k2[hit_windows], 2)
    hits = [
        (start + int(j) * step_size, float(k2[j]), float(p))
        for j, p in zip(hit_windows, pvalues, strict=True)
    ]
    return hits, stat_hist


def _extract_stamps_worker(args: tuple) -> None:
    """Fill one (obs_file, stamp-range) slice of the memmap-backed cadence .npy.

    Each worker opens its own hdf5 handle and its own r+ view of the shared .npy, then
    copies a stamp_width-wide window (over the first `time_bins` rows, polarization 0) around
    each hit. When downsample_factor > 1 the stamp is downsampled along frequency with
    downscale_local_mean before writing, so the memmap stores width
    stamp_width // downsample_factor — this is the same operation load_inference_data used to
    apply after the fact, moved to extraction time to cut storage by the same factor. Tasks
    address disjoint output regions — distinct obs_idx and/or non-overlapping stamp indices —
    so concurrent writes from the pool never collide. `stamp_starts` is the contiguous,
    start-sorted slice for this task; `base_idx` is its offset into the full stamp list so the
    worker writes to the correct absolute rows.
    """
    (
        npy_path,
        obs_idx,
        obs_h5,
        stamp_starts,
        base_idx,
        time_bins,
        stamp_width,
        downsample_factor,
    ) = args
    out = np.lib.format.open_memmap(npy_path, mode="r+")
    try:
        with h5py.File(obs_h5, "r") as hf:
            dset = hf["data"]
            for local_i, start in enumerate(stamp_starts):
                stamp = dset[:time_bins, 0, start : start + stamp_width]
                if downsample_factor > 1:
                    stamp = downscale_local_mean(stamp, (1, downsample_factor)).astype(np.float32)
                out[base_idx + local_i, obs_idx] = stamp
        out.flush()
    finally:
        del out


# NOTE: come back to this later
# def _drop_side_channels(
#     block_data: np.ndarray, side_channel_count: int, coarse_channel_width: int
# ) -> None:
#     """Zero out the leading/trailing side_channel_count coarse channels of a block.
#     Reserved for future use — leave as-is until the energy-profile criterion is defined."""
#     pass


@dataclass
class CadenceGroup:
    """One cadence's worth of observations grouped from a CSV."""

    key: tuple  # The group-by column values
    h5_paths: list[str]  # Observation .h5 paths, in row order
    csv_path: str  # Source CSV
    expected_obs: int
    is_valid: bool  # True iff len(h5_paths) == expected_obs


# NOTE: come back to this later (what does metadata_path store exactly?)
@dataclass
class CadenceResult:
    """Output of processing one cadence."""

    npy_path: str
    h5_paths: list[str]  # Same as in CadenceGroup
    key: tuple  # Same as in CadenceGroup
    # Number of stamp rows in the .npy (post-dedup, incl. overlap offsets) — the same
    # quantity the inference_cadences manifest stores as n_stamps. Historical name: this
    # is NOT the raw energy-detection hit count (metadata's n_raw_hits carries that).
    n_hits: int
    metadata_path: str  # Sibling .json with hit details


# NOTE: come back to this later
@dataclass
class CadenceHit:
    """A single energy detection hit on an ON-source observation."""

    fine_channel: int  # absolute fine-channel index into the full spectrum
    statistic: float  # D'Agostino-Pearson statistic
    pvalue: float
    frequency_mhz: float = field(default=float("nan"))


@dataclass
class PendingCadence:
    """One unit of preprocessing work: a valid cadence group plus its target .npy path."""

    group: CadenceGroup
    npy_path: str
    # 1-based position in plan_cadences() output (CSV order, so stable across resumed
    # attempts) — the short label used in this cadence's pipeline_stages span names
    index: int = 0


def derive_cadence_provenance(key: tuple, group_by_cols: list[str], metadata: dict) -> dict:
    """
    Map one cadence's group-by key and stamp metadata JSON onto the observational-provenance
    fields of the inference_results table.

    key and group_by_cols come from the CadenceGroup (values zipped positionally onto column
    names, matched case-insensitively so 'Target'/'target' both resolve); metadata is the
    per-cadence JSON written by _process_cadence. Returns a dict with keys target, session,
    band, cadence_id (int when parseable, else None), timestamp_observed (the header's tstart,
    when present), h5_path (first observation of the cadence), and stamp_frequencies_mhz (the
    per-stamp center frequencies, one per snippet row in the .npy).
    """
    # strict=False: a malformed key/cols pairing degrades to sparse provenance rather than
    # aborting the cadence
    key_map = {str(col).strip().lower(): val for col, val in zip(group_by_cols, key, strict=False)}

    cadence_id: int | None = None
    raw_cadence_id = key_map.get("cadence id")
    if raw_cadence_id is not None:
        try:
            cadence_id = int(raw_cadence_id)
        except (TypeError, ValueError):
            logger.warning(f"Could not parse cadence id {raw_cadence_id!r} as int; storing None")

    header = metadata.get("header") or {}
    timestamp_observed: float | None = None
    if "tstart" in header:
        try:
            timestamp_observed = float(header["tstart"])
        except (TypeError, ValueError):
            logger.warning(f"Could not parse header tstart {header['tstart']!r} as float")

    h5_paths = metadata.get("h5_paths") or []

    return {
        "target": key_map.get("target"),
        "session": key_map.get("session"),
        "band": key_map.get("band"),
        "cadence_id": cadence_id,
        "timestamp_observed": timestamp_observed,
        "h5_path": h5_paths[0] if h5_paths else None,
        "stamp_frequencies_mhz": metadata.get("stamp_frequencies_mhz"),
    }


# NOTE: come back to this later (add sorting functionality to sort rows in csv after grouping, e.g. via timestamp metadata from filenames? edge case where multiple 6-cadence observations of the same target & with the same grouping params, but differed by time, e.g. t=X to t=X+ε, then t=Y to t=Y+σ, for some small numbers ε and σ, and where X and Y are far apart from each other. add a way to distinguish these cases from problematic cases where we actually want to invalidate a cadence with a weird number of grouped observations, e.g. if multiple of 6 and enough of a gap between X and Y, then count as separate cadences?)
def group_observations_from_csv(
    csv_path: str,
    group_by_cols: list[str],
    h5_path_col: str,
    expected_obs: int = 6,
) -> tuple[list[CadenceGroup], list[CadenceGroup]]:
    """
    Group rows of a CSV into cadences and return (valid_groups, flagged_groups).

    Rows are grouped by the joint value of group_by_cols and assumed to be already ordered
    correctly within each group in the source CSV. expected_obs (typically 6) is the required
    number of observations per cadence; groups with the wrong count are returned in
    flagged_groups rather than valid_groups. The function is column-agnostic — it never assumes
    specific column names beyond what the caller provides.

    Raises FileNotFoundError if csv_path doesn't exist, and KeyError if any column in
    group_by_cols + [h5_path_col] is missing from the CSV header.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    required_cols = list(group_by_cols) + [h5_path_col]

    groups: OrderedDict[tuple, list[str]] = OrderedDict()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        available = reader.fieldnames or []

        # NOTE: come back to this later (does this fail for all if one csv has missing columns, or does it skip the invalid csvs and continue processing the valid ones? is proper downstream checkpointing implemented in the latter case?)
        missing = [c for c in required_cols if c not in available]
        if missing:
            raise KeyError(
                f"CSV {csv_path} is missing required column(s): {missing}. "
                f"Available columns: {available}"
            )

        for row_idx, row in enumerate(reader, start=1):
            key = tuple(row[c] for c in group_by_cols)
            h5_path = row[h5_path_col]
            # A ragged row (fewer fields than the header) makes csv.DictReader fill the path
            # with None; an empty/whitespace-only cell is just as unusable. Skip such rows
            # loudly instead of grouping them under a None/empty path, which would otherwise
            # only surface much later as a cryptic failure when the .h5 is opened.
            if h5_path is None or not str(h5_path).strip():
                logger.warning(
                    f"CSV {csv_path}: data row {row_idx} (key={key}) has a missing/empty "
                    f"'{h5_path_col}' value ({h5_path!r}); skipping row"
                )
                continue
            groups.setdefault(key, []).append(h5_path)

    valid: list[CadenceGroup] = []
    flagged: list[CadenceGroup] = []
    for key, h5_paths in groups.items():
        is_valid = len(h5_paths) == expected_obs
        cg = CadenceGroup(
            key=key,
            h5_paths=h5_paths,
            csv_path=csv_path,
            expected_obs=expected_obs,
            is_valid=is_valid,
        )
        (valid if is_valid else flagged).append(cg)

    if flagged:
        sample = [(g.key, len(g.h5_paths)) for g in flagged[:5]]
        logger.warning(
            f"Found {len(flagged)} cadence group(s) in {csv_path} with obs count != "
            f"{expected_obs}; flagged and skipped. First {len(sample)}: {sample}"
        )

    logger.info(
        f"Grouped {csv_path}: {len(valid)} valid cadence(s), {len(flagged)} flagged "
        f"(expected_obs={expected_obs})"
    )

    return valid, flagged


class DataPreprocessor:
    """Data preprocessor"""

    def __init__(self):
        """
        Initialize preprocessor
        """
        self.config = get_config()
        if self.config is None:
            raise ValueError("get_config() returned None")

        # TODO: add db writes for inference
        self.db = get_db()
        if self.db is None:
            raise ValueError("get_db() returned None")

        self.manager = get_manager()
        if self.manager is None:
            raise ValueError("get_manager() returned None")

        # Persistent worker pool for energy detection + stamp extraction, shared across all
        # cadences of a run (see start/stop_energy_detection_pool)
        self._ed_pool: Pool | None = None

    # NOTE: come back to this later
    def close(self):
        """Explicitly close the multiprocessing pool and shared memory"""
        # if hasattr(self, "pool") and self.pool is not None:
        #     self.manager.close_pool(self.pool)
        #     self.pool = None
        #
        # if hasattr(self, "shm") and self.shm is not None:
        #     self.manager.close_shared_memory(self.shm)
        #     self.shm = None

        # Defense-in-depth: ensure the persistent energy-detection pool is torn down even if a
        # caller forgot to call stop_energy_detection_pool(). No-op when no pool was started.
        self.stop_energy_detection_pool()

        logger.info("DataPreprocessor closed")

    # NOTE: shared resources currently created & destroyed within function itself. think about abstractions once preprocessing.py is complete
    def load_train_data(self) -> np.ndarray:
        """
        Load and preprocess training data into an array of shape (n, 6, 16, width_bin_downsampled).

        Uses a multiprocessing pool over shared memory to downsample cadences in parallel.
        Log-normalization is deferred to data_generation.py (training-side log-norm runs
        per-sample after injection), unlike load_inference_data which applies it here.
        """
        logger.info(f"Loading backgrounds from {self.config.data_path} for training")

        downsample_factor = self.config.data.downsample_factor
        final_width = self.config.data.width_bin // downsample_factor

        num_target_backgrounds = self.config.data.num_target_backgrounds
        chunk_size = self.config.data.background_load_chunk_size
        max_chunks = self.config.data.max_chunks_per_file
        n_processes = self.config.manager.n_processes
        chunks_per_worker = self.config.manager.chunks_per_worker

        logger.info(f"Target backgrounds: {num_target_backgrounds}")
        logger.info(f"Processing chunks of: {chunk_size}")
        logger.info(f"Final resolution: {final_width}")

        all_backgrounds = []  # NOTE: preallocate this as empty ndarray?

        for filename in self.config.data.train_files:
            filepath = self.config.get_training_file_path(filename)

            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                continue

            logger.info(f"Processing {filename}...")

            try:
                # Use read-only memory mapping to avoid loading full file into memory
                # That is, insted of loading the whole file from disk to memory synchronously
                # The OS' virtual memory manager establishes a virtual address space pointer
                # from the file's location on disk to the virtual memory of the Python process
                # This allows us to lazy load the data on-demand page-by-page using page fault
                # Benefits of this approach include: reduced startup latency,
                # efficient memory usage (since the memory allocated for the mapped array does not
                # count towards the Python process' heap memory usage, allowing us to raise the
                # ceiling up to our OS' virtual memory limits, which is typically constrained by
                # free disk space and our system's address space, rather than physical RAM),
                # and optimized access patterns (spatial locality since data is loaded in pages,
                # and shared memory for multiprocess/multithreaded programs)
                raw_data = np.load(filepath, mmap_mode="r")

            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                continue

            # Apply subset parameters if specified in config
            start, end = self.config.get_file_subset(filename)
            if start is not None or end is not None:
                raw_data = raw_data[start:end]

            logger.info(f"  Raw data shape: {raw_data.shape}")

            # Divide background into equal chunks, then cutoff if exceeds max_chunks
            n_backgrounds_total = raw_data.shape[0]
            n_chunks = min(max_chunks, (n_backgrounds_total + chunk_size - 1) // chunk_size)

            for chunk_idx in range(n_chunks):
                logger.info(f"Processing {filename}: chunk {chunk_idx + 1}/{n_chunks}")

                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, n_backgrounds_total)

                # Load chunk into memory
                chunk_data = np.array(raw_data[chunk_start:chunk_end])

                # NOTE: is this access pattern the most efficient (least pickling)? see _downsample_worker()'s docstring on pulling the cadence from the shared-memory global to avoid per-cadence pickling
                # NOTE: currently, loading the backgrounds takes WAY more time than processing the backgrounds
                # Prepare arguments for downsampling (just indices, not data - data is in global state)
                n_cadences = min(chunk_data.shape[0], num_target_backgrounds - len(all_backgrounds))
                args_list = [
                    (
                        i,
                        downsample_factor,
                        final_width,
                    )  # Just pass the chunk index, not the full cadence data
                    for i in range(n_cadences)
                ]

                # NOTE: do we need to create & destroy the pool every chunk? or just the shared memory & pass new references in? is there a differenc?
                if n_processes > 1:
                    # Create shared memory block for chunk data
                    chunk_shm = self.manager.create_shared_memory(
                        size=chunk_data.nbytes,
                        name=f"DataPreproc_{filename}_chunk_{chunk_idx}",  # NOTE: come back to this later
                    )

                    # Copy chunk data into shared memory
                    shared_chunk = np.ndarray(
                        chunk_data.shape,
                        dtype=chunk_data.dtype,
                        buffer=chunk_shm.buf,  # NOTE: what is self.shm.buf?
                    )
                    shared_chunk[:] = chunk_data[:]

                    # Create pool using shared memory reference
                    chunk_pool = self.manager.create_pool(
                        n_processes=n_processes,
                        name=f"DataPreproc_{filename}_chunk_{chunk_idx}",  # NOTE: come back to this later
                        initializer=_init_worker,
                        initargs=(chunk_shm.name, chunk_data.shape, chunk_data.dtype),
                    )

                    # Calculate optimal chunksize for load balancing
                    try:
                        n_workers = chunk_pool._processes
                    except AttributeError:
                        n_workers = n_processes
                    # NOTE: should we use separate chunks_per_worker? how to benchmark?
                    chunksize = max(1, n_cadences // (n_workers * chunks_per_worker))
                    # TEST: does return order matter?
                    results = chunk_pool.map(_downsample_worker, args_list, chunksize=chunksize)
                    # results = chunk_pool.imap(_downsample_worker, args_list, chunksize=chunksize)
                    # results = chunk_pool.imap_unordered(
                    #     _downsample_worker, args_list, chunksize=chunksize
                    # )

                else:
                    # Sequential processing
                    logger.info("DataPreprocessor running in sequential mode (n_processes=1)")

                    chunk_shm = None
                    chunk_pool = None

                    # Set global variable manually since no initializer ran
                    global _GLOBAL_CHUNK_DATA
                    shared_chunk = chunk_data
                    _GLOBAL_CHUNK_DATA = shared_chunk

                    results = [_downsample_worker(args) for args in args_list]

                # NOTE: is there a more efficient/elegant way to do this (e.g. with list comprehension/slicing)?
                # Collect valid results (filter out None from invalid cadences)
                for result in results:
                    if result is not None:
                        all_backgrounds.append(result)
                        if len(all_backgrounds) >= num_target_backgrounds:
                            break

                # Clear chunk data & shared resources
                del chunk_data, shared_chunk
                if chunk_shm:
                    self.manager.close_shared_memory(chunk_shm)
                    chunk_shm = None
                if chunk_pool:
                    self.manager.close_pool(chunk_pool)
                    chunk_pool = None
                del chunk_shm, chunk_pool
                gc.collect()

            # Clear raw_data reference
            del raw_data
            gc.collect()

        if len(all_backgrounds) == 0:
            raise ValueError("No data loaded successfully")

        # Stack all_backgrounds together
        background_array = np.array(all_backgrounds, dtype=np.float32)

        # Clear all_backgrounds reference
        del all_backgrounds
        gc.collect()

        # Sanity check: print descriptive stats
        min_val = np.min(background_array)
        max_val = np.max(background_array)
        mean_val = np.mean(background_array)

        logger.info(f"Total backgrounds loaded: {background_array.shape[0]}")
        logger.info(f"Background array shape: {background_array.shape}")
        logger.info(f"Background value range: [{min_val:.6f}, {max_val:.6f}]")
        logger.info(f"Background mean: {mean_val:.6f}")
        logger.info(f"Memory usage: {background_array.nbytes / 1e9:.2f} GB")
        logger.info(f"Background data ready at {background_array.shape[3]} resolution")

        return background_array

    # NOTE: shared resources currently created & destroyed within function itself. think about abstractions once preprocessing.py is complete
    # NOTE: calculate intensity statistics to overlay with training distributions (C' vs C)?
    def load_inference_data(
        self, override_filepaths: list[str] | None = None, parallel: bool = True
    ) -> np.ndarray:
        """
        Load and preprocess inference data into an array of shape
        (n, 6, 16, width_bin_downsampled). Uses a multiprocessing pool over shared memory to
        process cadences in parallel.

        Each file's stored width decides its path: files already at the downsampled width
        (written by _extract_stamps_worker with store_downsampled_stamps enabled) only need
        per-cadence log-normalization, which runs vectorized in the pool workers; legacy
        full-width files (width_bin, e.g. stamps preprocessed before downsample-at-extraction
        landed) keep the historical downsample-then-log-norm behavior.

        override_filepaths, when given, supplies absolute paths to iterate directly instead of
        resolving config.data.test_files via get_test_file_path — used by the inference command
        to chain per-cadence .npy outputs from preprocessing into inference without
        monkey-patching paths.

        parallel=False routes every chunk through the sequential in-process branch (no chunk
        pool, no shared memory) regardless of manager.n_processes. The streaming per-cadence
        path (main._infer_cadence) uses this: it loads exactly one already-downsampled cadence
        .npy whose per-cadence work is a cheap vectorized log-norm, while the prefetch thread
        is already driving the persistent energy-detection pool at full n_processes width —
        forking a second n_processes pool for that would double-subscribe the CPU.
        """
        logger.info(f"Loading backgrounds from {self.config.data_path} for inference")

        downsample_factor = self.config.data.downsample_factor
        width_bin = self.config.data.width_bin
        final_width = width_bin // downsample_factor

        chunk_size = self.config.data.inference_background_load_chunk_size
        n_processes = self.config.manager.n_processes
        chunks_per_worker = self.config.manager.chunks_per_worker

        logger.info(f"Processing chunks of: {chunk_size}")
        logger.info(f"Final resolution: {final_width}")

        all_cadences = []

        if override_filepaths is not None:
            # Iterate absolute paths directly (e.g. per-cadence .npy outputs from find_hits)
            file_iter = [(os.path.basename(p), p) for p in override_filepaths]
        else:
            file_iter = [
                (filename, self.config.get_test_file_path(filename))
                for filename in self.config.data.test_files
            ]

        for filename, filepath in file_iter:
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                continue

            logger.info(f"Processing {filename}...")

            try:
                # Use read-only memory mapping to avoid loading full file into memory
                # That is, insted of loading the whole file from disk to memory synchronously
                # The OS' virtual memory manager establishes a virtual address space pointer
                # from the file's location on disk to the virtual memory of the Python process
                # This allows us to lazy load the data on-demand page-by-page using page fault
                # Benefits of this approach include: reduced startup latency,
                # efficient memory usage (since the memory allocated for the mapped array does not
                # count towards the Python process' heap memory usage, allowing us to raise the
                # ceiling up to our OS' virtual memory limits, which is typically constrained by
                # free disk space and our system's address space, rather than physical RAM),
                # and optimized access patterns (spatial locality since data is loaded in pages,
                # and shared memory for multiprocess/multithreaded programs)
                raw_data = np.load(filepath, mmap_mode="r")

            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                continue

            # Apply subset parameters if specified in config
            start, end = self.config.get_file_subset(filename)
            if start is not None or end is not None:
                raw_data = raw_data[start:end]

            logger.info(f"  Raw data shape: {raw_data.shape}")

            # A cadence plate must be 4-D (n, 6, time_bins, width); only the trailing width was
            # validated historically, so a wrong-rank array would flow into the pool workers and
            # either fail cryptically or silently mis-index the 6 observations. Skip-and-warn,
            # matching the unsupported-width skip below, so one malformed file can't sink the run.
            if raw_data.ndim != 4:
                logger.error(
                    f"  {filename} has ndim {raw_data.ndim} (shape {raw_data.shape}), "
                    f"expected 4 (n, 6, time_bins, width); skipping file"
                )
                del raw_data
                continue

            # Everything downstream casts to float32; a non-float input (e.g. integer power
            # counts) was previously coerced silently. Surface the coercion loudly — the values
            # are unchanged, but the dtype change is worth naming so a mis-typed catalog is
            # visible rather than hidden.
            if not np.issubdtype(raw_data.dtype, np.floating):
                logger.warning(
                    f"  {filename} has non-float dtype {raw_data.dtype}; coercing to float32"
                )

            # Branch on the stored width: already-downsampled stamps (written by
            # _extract_stamps_worker) only need log-norm; legacy full-width files keep the
            # historical downsample-then-log-norm path.
            stored_width = raw_data.shape[-1]
            if stored_width == final_width:
                already_downsampled = True
                logger.info(f"  {filename} stored at final width {final_width}: log-norm only")
            elif stored_width == width_bin:
                already_downsampled = False
                logger.info(f"  {filename} stored at full width {width_bin}: downsample + log-norm")
            else:
                logger.error(
                    f"  {filename} width {stored_width} matches neither width_bin "
                    f"({width_bin}) nor width_bin // downsample_factor ({final_width}); "
                    f"skipping file"
                )
                del raw_data
                continue

            # Divide background into equal chunks, then cutoff if exceeds max_chunks
            n_cadences_total = raw_data.shape[0]
            n_chunks = (n_cadences_total + chunk_size - 1) // chunk_size

            for chunk_idx in range(n_chunks):
                logger.info(f"Processing {filename}: chunk {chunk_idx + 1}/{n_chunks}")

                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, n_cadences_total)

                # Load chunk into memory
                chunk_data = np.array(raw_data[chunk_start:chunk_end])

                # NOTE: is this access pattern the most efficient (least pickling)? see _downsample_worker()'s docstring on pulling the cadence from the shared-memory global to avoid per-cadence pickling
                # NOTE: currently, loading the backgrounds takes WAY more time than processing the backgrounds
                # Prepare arguments (just indices, not data - data is in global state).
                # Downsampled files fold log-norm into the workers (resolving the old
                # "downsample & log-norm simultaneously" TODO); legacy files downsample in
                # the workers and log-norm in-process below, exactly as before.
                n_cadences = chunk_data.shape[0]
                if already_downsampled:
                    worker_fn = _lognorm_worker
                    args_list = [(i,) for i in range(n_cadences)]
                else:
                    worker_fn = _downsample_worker
                    args_list = [(i, downsample_factor, final_width) for i in range(n_cadences)]

                # NOTE: do we need to create & destroy the pool every chunk? or just the shared memory & pass new references in? is there a differenc?
                if parallel and n_processes > 1:
                    # Create shared memory block for chunk data
                    chunk_shm = self.manager.create_shared_memory(
                        size=chunk_data.nbytes,
                        name=f"DataPreproc_{filename}_chunk_{chunk_idx}",  # NOTE: come back to this later
                    )

                    # Copy chunk data into shared memory
                    shared_chunk = np.ndarray(
                        chunk_data.shape,
                        dtype=chunk_data.dtype,
                        buffer=chunk_shm.buf,  # NOTE: what is self.shm.buf?
                    )
                    shared_chunk[:] = chunk_data[:]

                    # Create pool using shared memory reference
                    chunk_pool = self.manager.create_pool(
                        n_processes=n_processes,
                        name=f"DataPreproc_{filename}_chunk_{chunk_idx}",  # NOTE: come back to this later
                        initializer=_init_worker,
                        initargs=(chunk_shm.name, chunk_data.shape, chunk_data.dtype),
                    )

                    # Calculate optimal chunksize for load balancing
                    try:
                        n_workers = chunk_pool._processes
                    except AttributeError:
                        n_workers = n_processes
                    # NOTE: should we use separate chunks_per_worker? how to benchmark?
                    chunksize = max(1, n_cadences // (n_workers * chunks_per_worker))
                    # TEST: does return order matter?
                    results = chunk_pool.map(worker_fn, args_list, chunksize=chunksize)

                else:
                    # Sequential processing
                    logger.info(
                        f"DataPreprocessor running in sequential mode "
                        f"(parallel={parallel}, n_processes={n_processes})"
                    )

                    chunk_shm = None
                    chunk_pool = None

                    # Set global variable manually since no initializer ran
                    global _GLOBAL_CHUNK_DATA
                    shared_chunk = chunk_data
                    _GLOBAL_CHUNK_DATA = shared_chunk

                    results = [worker_fn(args) for args in args_list]

                # NOTE: is there a more efficient/elegant way to do this (e.g. with list comprehension/slicing)?
                # Collect valid results (filter out None from invalid cadences)
                for result in results:
                    if result is not None:
                        if already_downsampled:
                            all_cadences.append(result)
                        else:
                            # Legacy path: per-cadence log-norm in-process, as before
                            all_cadences.append(log_norm(result))

                # Clear chunk data & shared resources
                del chunk_data, shared_chunk
                if chunk_shm:
                    self.manager.close_shared_memory(chunk_shm)
                    chunk_shm = None
                if chunk_pool:
                    self.manager.close_pool(chunk_pool)
                    chunk_pool = None
                del chunk_shm, chunk_pool
                gc.collect()

            # Clear raw_data reference
            del raw_data
            gc.collect()

        if len(all_cadences) == 0:
            raise ValueError("No data loaded successfully")

        # Stack all_cadences together (every cadence is downsampled + log-normalized by now)
        cadence_array = np.array(all_cadences, dtype=np.float32)

        # Clear all_cadences reference
        del all_cadences
        gc.collect()

        # Sanity check: print descriptive stats
        min_val = np.min(cadence_array)
        max_val = np.max(cadence_array)
        mean_val = np.mean(cadence_array)

        if max_val > 1.0:
            logger.error(f"Cadence array values too large! Max: {max_val}")
            raise ValueError("Preprocessing normalization check failed")
        elif min_val < 0.0:
            logger.error(f"Cadence array values too small! Min: {min_val}")
            raise ValueError("Preprocessing normalization check failed")
        elif np.isnan(cadence_array).any():
            logger.error("Cadence array contains NaN values!")
            raise ValueError("Preprocessing normalization check failed")
        elif np.isinf(cadence_array).any():
            logger.error("Cadence array contains Inf values!")
            raise ValueError("Preprocessing normalization check failed")
        else:
            logger.info("Cadence array properly normalized")
            logger.info(f"Total cadences loaded: {cadence_array.shape[0]}")
            logger.info(f"Cadence array shape: {cadence_array.shape}")
            logger.info(f"Cadence value range: [{min_val:.6f}, {max_val:.6f}]")
            logger.info(f"Cadence mean: {mean_val:.6f}")
            logger.info(f"Memory usage: {cadence_array.nbytes / 1e9:.2f} GB")
            logger.info(f"Cadence data ready at {cadence_array.shape[3]} resolution")

        return cadence_array

    def start_energy_detection_pool(self) -> None:
        """
        Create the persistent worker pool used for energy detection and stamp extraction.

        One pool serves every cadence of the run (no per-block or per-cadence pool churn).
        Call from the main thread before any cadence processing starts — forking from the main
        thread before background threads spin up avoids inheriting mid-operation locks — and
        pair with stop_energy_detection_pool() when the run is done. No-op when a pool already
        exists or n_processes == 1 (sequential mode).
        """
        if self._ed_pool is not None:
            return
        n_processes = self.config.manager.n_processes
        if n_processes > 1:
            self._ed_pool = self.manager.create_pool(
                n_processes=n_processes,
                name="DataPreproc_energy_detection",
                initializer=_init_plain_worker,
            )

    def stop_energy_detection_pool(self) -> None:
        """Close the persistent energy-detection pool (no-op if never started)."""
        if self._ed_pool is not None:
            self.manager.close_pool(self._ed_pool)
            self._ed_pool = None

    def plan_cadences(self) -> list[PendingCadence]:
        """
        Group every CSV in config.data.inference_files into per-cadence work units without
        processing anything. Returns one PendingCadence (valid group + target .npy path) per
        valid cadence, in CSV order — the unit list the streaming inference loop iterates.
        Raises ValueError when two inference_files entries share a basename stem (their
        output dir and stamp filenames would collide).
        """
        inference_files = self.config.data.inference_files
        if not inference_files:
            logger.warning("plan_cadences() called with no inference_files configured")
            return []

        # Fail fast on duplicate CSV basename stems: both the tag-scoped default output dir
        # and the per-cadence stamp .npy names are keyed on the stem, so two entries like
        # runA/x.csv and runB/x.csv would silently share a directory (and/or stamp filenames)
        # and cross-resume each other's cadences.
        stem_sources: dict[str, str] = {}
        for csv_filename in inference_files:
            stem = os.path.splitext(os.path.basename(csv_filename))[0]
            if stem in stem_sources:
                raise ValueError(
                    f"Duplicate inference CSV basename stem '{stem}' "
                    f"('{stem_sources[stem]}' vs '{csv_filename}'): output directories and "
                    f"stamp filenames are keyed on the basename, so these entries would "
                    f"overwrite/resume each other. Rename one so basenames are unique."
                )
            stem_sources[stem] = csv_filename

        # Output directory resolution: an explicit --preprocess-output-dir is used as-is
        # (shared across CSVs); otherwise each CSV gets its own tag-scoped default,
        # {data_path}/inference/preprocessed/<csv_stem>_<save_tag>/. Tag scoping trades
        # cross-run caching for isolation: a retry under the same tag still resumes via the
        # existing-.npy skip, while a fresh run (new datetime tag) starts from a clean
        # directory that stale stamps from an older failed attempt can't leak into. To reuse
        # an old run's preprocessing, pass its directory explicitly.
        explicit_output_dir = self.config.inference.preprocess_output_dir
        save_tag = self.config.checkpoint.save_tag

        group_by_cols = self.config.inference.cadence_group_by_cols
        h5_path_col = self.config.inference.cadence_h5_path_col
        expected_obs = self.config.inference.cadence_expected_obs

        units: list[PendingCadence] = []

        for csv_filename in inference_files:
            csv_path = self.config.get_inference_file_path(csv_filename)
            logger.info(f"Processing inference CSV: {csv_path}")

            try:
                valid_groups, flagged_groups = group_observations_from_csv(
                    csv_path=csv_path,
                    group_by_cols=group_by_cols,
                    h5_path_col=h5_path_col,
                    expected_obs=expected_obs,
                )
            except (FileNotFoundError, KeyError) as e:
                logger.error(f"Failed to group CSV {csv_path}: {e}")
                continue

            logger.info(
                f"CSV {csv_filename}: {len(valid_groups)} valid, {len(flagged_groups)} flagged"
            )

            csv_stem = os.path.splitext(os.path.basename(csv_path))[0]

            output_dir = explicit_output_dir or self.config.get_inference_file_path(
                os.path.join("preprocessed", f"{csv_stem}_{save_tag}")
            )
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Preprocessing output directory for {csv_filename}: {output_dir}")

            for group in valid_groups:
                npy_filename = self._cadence_npy_filename(csv_stem, group.key)
                units.append(
                    PendingCadence(
                        group=group,
                        npy_path=os.path.join(output_dir, npy_filename),
                        index=len(units) + 1,
                    )
                )

        logger.info(
            f"Planned {len(units)} cadence work unit(s) across {len(inference_files)} CSV(s)"
        )
        return units

    def process_pending_cadence(self, unit: PendingCadence) -> CadenceResult | None:
        """
        Resume-or-process one cadence work unit. Returns a CadenceResult when a stamp .npy is
        available (freshly written or already on disk), or None when the cadence produced no
        hits or failed — single-cadence failures are logged and swallowed so one bad cadence
        can't abort the whole catalog (the retry loop at the inference_command level handles
        broader recovery).
        """
        group, npy_path = unit.group, unit.npy_path

        if os.path.exists(npy_path):
            # Resume path: rebuild a minimal CadenceResult from the existing file
            metadata_path = self.cadence_metadata_path(npy_path)
            try:
                existing = np.load(npy_path, mmap_mode="r")
                n_hits = existing.shape[0]
                del existing
            except Exception as e:
                logger.warning(
                    f"Existing .npy at {npy_path} could not be inspected ({e}); "
                    f"reprocessing cadence"
                )
                # Fall through (no return) to the _process_cadence call below so a
                # corrupted .npy gets regenerated rather than silently skipped.
            else:
                logger.info(
                    f"Skipping cadence {group.key}: {npy_path} already exists ({n_hits} hits)"
                )
                return CadenceResult(
                    npy_path=npy_path,
                    h5_paths=group.h5_paths,
                    key=group.key,
                    n_hits=n_hits,
                    metadata_path=metadata_path,
                )

        try:
            # Umbrella stage span for this cadence's preprocessing phase — the per-ON-file
            # read_ed / dedup / extract sub-stages inside _process_cadence nest under it
            # via thread-local naming. The resume path above records nothing (no work done)
            with stage_timer(f"inference.preprocess_cadence_{unit.index:03d}"):
                return self._process_cadence(group, npy_path)
        except Exception as e:
            logger.error(f"Failed to process cadence {group.key}: {e}")
            return None

    # NOTE: come back to this later (based on docstring, we're processing cadences sequentially. if so, any way to parallelize?)
    def find_hits(self) -> list[CadenceResult]:
        """
        Convert raw .h5 cadence observations into (n_hits, 6, 16, stored_width) .npy snippets,
        returning one CadenceResult per successfully processed (or already cached) cadence.

        Driven by CSVs in config.data.inference_files. Each CSV is grouped into cadences via
        plan_cadences() and processed sequentially over one persistent worker pool. Within each
        cadence, energy detection runs on ON-source files (positions 0, 2, 4 in ABACAD); stamps
        are then extracted from all 6 observations at the hit frequencies. Each cadence is
        checkpointed to disk as soon as its .npy is ready, and on retry, cadences whose output
        already exists are skipped.

        The streaming inference path in main.py drives plan_cadences() /
        process_pending_cadence() directly instead (so preprocessing of cadence i+1 can overlap
        inference of cadence i); this wrapper preserves the batch contract for callers that
        want every cadence preprocessed up front.
        """
        units = self.plan_cadences()
        if not units:
            return []

        results: list[CadenceResult] = []
        self.start_energy_detection_pool()
        try:
            for unit in units:
                cadence_result = self.process_pending_cadence(unit)
                if cadence_result is not None:
                    results.append(cadence_result)
        finally:
            self.stop_energy_detection_pool()

        logger.info(f"find_hits completed: {len(results)} cadence .npy files available")
        return results

    # NOTE: come back to this later (ensure filenames are structured similarly as in train_files & test_files)
    @staticmethod
    def _cadence_npy_filename(csv_stem: str, key: tuple) -> str:
        """Build a deterministic filename for a cadence group's .npy output."""
        # Sanitize each key component via an allowlist: only word chars, dash, and
        # dot survive — everything else collapses to underscore. This is broader
        # than just stripping path separators / whitespace: CSV column values can
        # carry quotes, control chars, shell metacharacters, or path-traversal
        # sequences (e.g. "..") that would otherwise leak through.
        safe_parts = [re.sub(r"[^\w\-.]+", "_", str(part)) for part in key]
        return f"{csv_stem}_{'_'.join(safe_parts)}.npy"

    @staticmethod
    def cadence_metadata_path(npy_path: str) -> str:
        """Return the sibling .json path for a cadence's metadata. Public: the streaming
        loop (main.py) and the viz suite derive metadata paths for resume-skipped cadences
        whose CadenceResult was never rebuilt."""
        return os.path.splitext(npy_path)[0] + ".json"

    @staticmethod
    def _to_json_safe(obj):
        """
        Coerce h5py / numpy values into JSON-native types.

        h5py attributes can be bytes, numpy scalars, or numpy arrays — none of
        which json.dump handles by default. Walk the structure once and
        convert leaf nodes; everything else passes through.
        """
        if isinstance(obj, dict):
            return {str(k): DataPreprocessor._to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [DataPreprocessor._to_json_safe(v) for v in obj]
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if isinstance(obj, np.ndarray):
            return DataPreprocessor._to_json_safe(obj.tolist())
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        return obj

    # NOTE: come back to this later (does a cadence snippet get created if only 1 of the ON observations crosses the threshold, or all 3? should probably be all 3 as of now since models are trained on signals that don't yet drift out of frame?)
    def _process_cadence(self, group: CadenceGroup, npy_path: str) -> CadenceResult | None:
        """
        Run energy detection on one cadence and write its stamp .npy at the given absolute
        npy_path. Returns a CadenceResult on success, or None if no hits survived.

        Energy detection runs only on ON-source observations (positions 0, 2, 4 in ABACAD order);
        the hits found there define the frequency slices that get extracted from all 6
        observations. group is assumed to be a validated CadenceGroup with len(h5_paths) ==
        expected_obs. Each coarse channel is one fused task (read -> DC spike -> bandpass
        flatten -> vectorized threshold) on the persistent pool from
        start_energy_detection_pool(); only the small per-channel hit lists cross the process
        boundary, so no blocks, block-sized shared memory, or per-stage pools exist anymore.
        """
        coarse_channel_width = self.config.inference.coarse_channel_width
        # coarse_channel_log_interval is a progress-logging chunk size only (None -> n_processes);
        # actual parallelism is the persistent pool's worker count.
        log_interval = self.config.inference.coarse_channel_log_interval
        window_size = self.config.inference.detection_window_size
        step_size = self.config.inference.detection_step_size
        stat_threshold = self.config.inference.stat_threshold
        stamp_width = self.config.inference.stamp_width
        overlap_search = self.config.inference.overlap_search
        overlap_fraction = self.config.inference.overlap_fraction
        store_downsampled = self.config.inference.store_downsampled_stamps
        downsample_factor = self.config.data.downsample_factor if store_downsampled else 1
        time_bins = self.config.data.time_bins
        n_processes = self.config.manager.n_processes
        progress_chunk = max(1, log_interval if log_interval is not None else n_processes)

        # Read header / metadata from the first ON-source file
        on_source_paths = [group.h5_paths[i] for i in (0, 2, 4)]
        primary_h5 = on_source_paths[0]

        with h5py.File(primary_h5, "r") as hf:
            header = {k: hf["data"].attrs[k] for k in hf["data"].attrs}
            data_shape = hf["data"].shape

        # Up-front geometry validation of ALL 6 observation files, not just the primary.
        # Energy detection and stamp extraction index every dataset as [time, polarization,
        # frequency] and read exactly the first time_bins rows (see
        # _energy_detect_channel_worker / _extract_stamps_worker — extra rows beyond
        # time_bins are simply ignored), so a malformed file anywhere in the cadence would
        # otherwise fail deep inside a worker: a wrong-rank dataset mis-slices cryptically,
        # and a short row count in ANY of the 6 files (ON or OFF) raises a broadcast
        # ValueError mid-extraction when the short stamp is assigned into its fixed
        # (n_stamps, 6, time_bins, stored_width) memmap slot — a short ON file additionally
        # degrades the k2 statistic silently first, since _sliding_normality_k2 derives its
        # sample count from the rows it receives. The whole cadence is skipped rather than
        # just the offending file: the 6-observation stamp tensor and the num_observations
        # contract downstream (encoder reshape, RF features, ABACAD viz) have no
        # representation for a 5-observation cadence.
        rank_problems: list[str] = []
        short_problems: list[str] = []
        for idx, obs_h5 in enumerate(group.h5_paths):
            if idx == 0:
                # group.h5_paths[0] == primary_h5, already opened above for header/data_shape —
                # reuse it instead of a redundant second h5py.File open.
                obs_shape = data_shape
            else:
                with h5py.File(obs_h5, "r") as hf:
                    obs_shape = hf["data"].shape
            if len(obs_shape) != 3:
                rank_problems.append(
                    f"{obs_h5} has 'data' of rank {len(obs_shape)} (shape {obs_shape})"
                )
            elif int(obs_shape[0]) < time_bins:
                short_problems.append(f"{obs_h5} has only {int(obs_shape[0])} time bins")
        if rank_problems:
            # The caller (process_pending_cadence) turns this ValueError into a logged skip,
            # so the skip-and-continue policy across a large catalog is preserved.
            raise ValueError(
                f"Cadence {group.key}: expected rank 3 (time, polarization, frequency) "
                f"'data' in every observation file; offending file(s): "
                f"{'; '.join(rank_problems)}"
            )
        if short_problems:
            logger.warning(
                f"Cadence {group.key}: every observation file needs >= {time_bins} time "
                f"bins; skipping whole cadence — offending file(s): "
                f"{'; '.join(short_problems)}"
            )
            return None

        n_chans = int(header.get("nchans", data_shape[-1]))
        foff = float(header["foff"])
        fch1 = float(header["fch1"])

        # NOTE: every complete coarse channel is processed. The historical block-based path
        # floored to a multiple of the old parallel_coarse_chans knob, silently dropping up to
        # that many - 1 trailing coarse channels when n_chans wasn't an exact multiple of a
        # block. (That knob is now coarse_channel_log_interval and only affects log cadence.)
        n_coarse_total = n_chans // coarse_channel_width
        if n_coarse_total == 0:
            logger.warning(
                f"Cadence {group.key}: n_chans={n_chans} is smaller than one coarse channel "
                f"({coarse_channel_width}); skipping"
            )
            return None

        logger.info(
            f"Cadence {group.key}: n_chans={n_chans}, coarse channels={n_coarse_total}, "
            f"ON-source files={len(on_source_paths)}"
        )

        # The flattener depends on the file's actual coarse count (PFB folds
        # adjacent-channel leakage), so it can only be built once n_coarse_total is known.
        bandpass_flatten = self._get_bandpass_flattener(n_coarse_total)
        pfb_active = bandpass_flatten.func is _pfb_flatten_bandpass

        if self.config.inference.bandpass_debug_plot:
            try:
                self._plot_bandpass_overlay(primary_h5, n_coarse_total, bandpass_flatten, npy_path)
            except Exception as e:
                logger.error(f"Cadence {group.key}: bandpass overlay plot failed: {e}")

        cadence_start_time = time.time()

        # Aggregate hits across all ON-source files, plus the per-ON-file summary histogram
        # of every window statistic (for the ed_stat_distributions figure)
        all_hits: list[tuple] = []  # (abs_idx, stat, p)
        stat_hists = np.zeros((len(on_source_paths), len(ED_STAT_HIST_EDGES) - 1), dtype=np.int64)

        for on_source_idx, on_h5 in enumerate(on_source_paths):
            logger.info(
                f"Cadence {group.key}: running energy detection on ON-source "
                f"{on_source_idx + 1}/{len(on_source_paths)}: {on_h5}"
            )

            # One stage span per ON file (read + DC spike + bandpass flatten + threshold)
            with stage_timer(f"read_ed_on{on_source_idx + 1}"):
                if pfb_active and on_source_idx == 0:
                    # Cheap residual-flatness sanity check — primary ON file only: the static
                    # response is a property of the backend, shared by every file of the cadence
                    self._warn_on_pfb_response_mismatch(on_h5, n_coarse_total)

                tasks = [
                    (
                        on_h5,
                        ch,
                        coarse_channel_width,
                        time_bins,
                        bandpass_flatten,
                        window_size,
                        step_size,
                        stat_threshold,
                    )
                    for ch in range(n_coarse_total)
                ]

                if self._ed_pool is not None:
                    # imap (ordered, chunksize 1) keeps every worker busy across the whole
                    # file while results stream back for progress logging
                    channel_hits_iter = self._ed_pool.imap(_energy_detect_channel_worker, tasks)
                else:
                    if n_processes > 1:
                        logger.info(
                            "Energy detection running sequentially: no persistent pool "
                            "started (call start_energy_detection_pool() to parallelize)"
                        )
                    channel_hits_iter = map(_energy_detect_channel_worker, tasks)

                for done, (channel_hits, channel_hist) in enumerate(channel_hits_iter, start=1):
                    all_hits.extend(channel_hits)
                    stat_hists[on_source_idx] += channel_hist
                    if done % progress_chunk == 0 or done == n_coarse_total:
                        logger.info(
                            f"  Coarse channel {done}/{n_coarse_total} of ON-source "
                            f"{on_source_idx + 1}/{len(on_source_paths)}"
                        )

        logger.info(f"Cadence {group.key}: {len(all_hits)} raw hits across ON-source files")

        # NOTE: come back to this later (what's the trade-off for doing dedup vs not? e.g. lower storage & compute, but higher FNR or lower DR sensitivity?)
        # Deduplicate: greedy merge of any pair within stamp_width // 2
        with stage_timer("dedup"):
            merged_hits = self._deduplicate_hits(all_hits, stamp_width)
        logger.info(
            f"Cadence {group.key}: {len(merged_hits)} hits after deduplication "
            f"(stamp_width={stamp_width})"
        )

        if not merged_hits:
            logger.info(f"Cadence {group.key}: no hits survived; skipping")
            return None

        # Build the stamp centers (optionally including overlap offsets)
        stamp_centers: list[tuple[int, float, float]] = []
        offsets = [0]
        if overlap_search:
            offset_mag = int(overlap_fraction * stamp_width)
            offsets = [-offset_mag, 0, offset_mag]

        half = stamp_width // 2
        for hit in merged_hits:
            abs_idx, stat, pval = hit
            for offset in offsets:
                center = abs_idx + offset
                start = center - half
                end = center + half
                if start < 0 or end > n_chans:
                    continue
                stamp_centers.append((start, stat, pval))

        if not stamp_centers:
            logger.info(f"Cadence {group.key}: no valid in-bounds stamps; skipping")
            return None

        # Sort stamps by start index before extraction. With overlap_search the raw
        # order is hit-interleaved (hit1-off, hit1, hit1+off, hit2-off, ...), so
        # neighboring hits can produce out-of-order starts. Reads against
        # bitshuffle-compressed .h5 files are dominated by chunk decompression cost;
        # sequential start order gives the OS / hdf5 chunk cache a chance to reuse a
        # decompressed chunk across adjacent stamps instead of redecompressing it.
        stamp_centers.sort(key=lambda s: s[0])

        # Extract stamps into a memmap-backed .npy so worker processes can fill disjoint
        # (obs_file, stamp-range) slices in parallel. The previous sequential per-file loop
        # over all 6 observations (single-threaded reads + bitshuffle chunk decompression)
        # was the dominant, GPU-idle cost of CSV inference.
        #
        # Stamps are downsampled along frequency at extraction time (downsample_factor > 1,
        # the store_downsampled_stamps default), so the stored width is stamp_width //
        # downsample_factor — an ~8x storage cut at defaults that also removes the separate
        # downsample pass from load_inference_data.
        #
        # Atomicity is unchanged: the memmap is written to a .tmp sibling and we os.replace()
        # it onto the canonical name only after every worker finishes, so the resume path
        # (which treats npy_path's existence as proof of a complete write) still holds.
        n_stamps = len(stamp_centers)
        stored_width = stamp_width // downsample_factor
        tmp_npy_path = os.path.splitext(npy_path)[0] + ".tmp.npy"
        # A leftover .tmp.npy means a previous attempt died (e.g. SIGKILL) between memmap
        # creation and os.replace; it was never promoted to npy_path, so it's safe to drop
        # (open_memmap would truncate it anyway — the warning is the point)
        if os.path.exists(tmp_npy_path):
            logger.warning(
                f"Cadence {group.key}: removing stale partial output {tmp_npy_path} "
                f"from an interrupted previous run"
            )
            os.remove(tmp_npy_path)
        # NOTE: np.lib.format.open_memmap is a semi-public numpy API (stable across 1.x and
        # documented via np.lib.format); revisit if a future numpy bump moves it
        memmap = np.lib.format.open_memmap(
            tmp_npy_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_stamps, len(group.h5_paths), time_bins, stored_width),
        )
        memmap.flush()
        del memmap  # header + full-size file are on disk; workers reopen it in r+ mode

        stamp_starts = [start for start, _, _ in stamp_centers]
        # Split each obs file's (already start-sorted) stamps into contiguous chunks so more
        # than len(h5_paths) workers can run, while keeping each worker's reads sequential to
        # preserve the hdf5 chunk-cache reuse that the sort above buys us.
        chunks_per_file = max(1, -(-n_processes // len(group.h5_paths)))  # ceil div
        chunk_size = max(1, -(-n_stamps // chunks_per_file))  # ceil div
        tasks = [
            (
                tmp_npy_path,
                obs_idx,
                obs_h5,
                stamp_starts[base : base + chunk_size],
                base,
                time_bins,
                stamp_width,
                downsample_factor,
            )
            for obs_idx, obs_h5 in enumerate(group.h5_paths)
            for base in range(0, n_stamps, chunk_size)
        ]

        with stage_timer("extract"):
            if self._ed_pool is not None and len(tasks) > 1:
                # Reuse the persistent energy-detection pool — extraction workers are plain
                # (no shared memory), so the same pool serves both stages without churn
                self._ed_pool.map(_extract_stamps_worker, tasks)
            else:
                for task in tasks:
                    _extract_stamps_worker(task)

            os.replace(tmp_npy_path, npy_path)

        metadata_path = self.cadence_metadata_path(npy_path)
        # Per-stamp frequency = center bin's frequency, computed from header's fch1/foff
        stamp_freqs_mhz = [float(fch1 + foff * (start + half)) for start, _, _ in stamp_centers]
        stamp_stats = [float(s) for _, s, _ in stamp_centers]
        stamp_pvals = [float(p) for _, _, p in stamp_centers]

        metadata = {
            "key": group.key,
            "csv_path": group.csv_path,
            "h5_paths": group.h5_paths,
            "header": header,
            "stamp_starts": [int(start) for start, _, _ in stamp_centers],
            "stamp_width": stamp_width,
            "stored_width": stored_width,
            "downsample_factor_applied": downsample_factor,
            "stamp_frequencies_mhz": stamp_freqs_mhz,
            "stamp_statistics": stamp_stats,
            "stamp_pvalues": stamp_pvals,
            "overlap_search": overlap_search,
            "overlap_fraction": overlap_fraction if overlap_search else None,
            # Energy-detection provenance for the visualization suite: the all-window
            # statistic histograms (per ON file, fixed log-spaced bins) and the hit
            # frequencies before/after deduplication (hit spectrum + funnel figures).
            # NOTE: the frequency lists are stored raw (unlike the pre-binned stat
            # histograms — an asymmetry): tens of thousands of floats per RFI-dense
            # cadence, bounded at current catalog scale. If metadata JSONs grow unwieldy,
            # pre-bin these onto a fixed frequency grid the way ed_stat_hist does.
            "ed_stat_hist": {
                "bin_edges": [float(e) for e in ED_STAT_HIST_EDGES],
                "counts_per_on_file": stat_hists.tolist(),
            },
            "n_raw_hits": len(all_hits),
            "n_merged_hits": len(merged_hits),
            "raw_hit_frequencies_mhz": [float(fch1 + foff * idx) for idx, _, _ in all_hits],
            "merged_hit_frequencies_mhz": [float(fch1 + foff * idx) for idx, _, _ in merged_hits],
        }
        tmp_metadata_path = metadata_path + ".tmp"
        with open(tmp_metadata_path, "w") as f:
            json.dump(self._to_json_safe(metadata), f, indent=2)
        os.replace(tmp_metadata_path, metadata_path)

        gc.collect()

        # Record the stage transition in the inference_cadences run manifest. Written only
        # when preprocessing actually ran (the resume path never re-writes it); a crash
        # between the os.replace above and this write just loses an informational row —
        # resume keys off the .npy's existence, and the 'inferred' row keys off inference.
        self.db.write_inference_cadence(
            npy_path=npy_path,
            status="preprocessed",
            tag=self.config.checkpoint.save_tag,
            csv_path=group.csv_path,
            cadence_key=group.key,
            n_stamps=n_stamps,
            duration_s=time.time() - cadence_start_time,
        )

        logger.info(
            f"Cadence {group.key}: wrote {n_stamps} stamps -> "
            f"{npy_path} (metadata: {metadata_path})"
        )

        return CadenceResult(
            npy_path=npy_path,
            h5_paths=group.h5_paths,
            key=group.key,
            n_hits=n_stamps,
            metadata_path=metadata_path,
        )

    def _get_bandpass_flattener(
        self, num_coarse_channels: int
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return the configured bandpass-flattening callable for energy detection.

        The callable takes one coarse channel of shape (time_bins, coarse_channel_width) and
        returns the flattened channel; it must be picklable (a functools.partial over a
        module-level function) so pool workers can receive it in their task args.

        num_coarse_channels is the *file's* actual coarse-channel count
        (n_chans // coarse_channel_width) — the PFB response folds adjacent-channel leakage,
        so it depends on how many coarse channels the recording actually has. Files with a
        single coarse channel can't support the fold and fall back to the spline flattener
        with a warning.
        """
        method = self.config.inference.bandpass_method
        if method == "pfb":
            if num_coarse_channels >= 2:
                response_path = self._ensure_pfb_response_file(
                    self.config.inference.coarse_channel_width,
                    num_coarse_channels,
                    self.config.inference.pfb_taps_per_channel,
                )
                return functools.partial(_pfb_flatten_bandpass, response_path=response_path)
            logger.warning(
                "bandpass_method='pfb' requires >= 2 coarse channels to fold adjacent-channel "
                f"leakage, but this file has {num_coarse_channels}; falling back to the "
                "spline flattener for this cadence"
            )
        elif method != "spline":
            raise ValueError(f"Unknown bandpass_method {method!r}; expected 'pfb' or 'spline'")
        return functools.partial(
            _spline_flatten_bandpass, spl_order=self.config.inference.spline_order
        )

    def _ensure_pfb_response_file(
        self, fine_per_coarse: int, num_coarse_channels: int, taps_per_channel: int
    ) -> str:
        """
        Compute the PFB passband response in the parent process and persist it to a
        deterministic sidecar .npy under {output_path}/pfb_cache/, returning its path.

        The heavy work (an ~n_chans-point FFT) runs exactly once per parameter combination in
        the parent — gen_coarse_channel_response is process-cached and the file is reused when
        its content matches — while pool workers receive only the path and read the ~8 MB
        array (see _pfb_flatten_bandpass). The file is content-addressed by its parameters, so
        stale-run leftovers are impossible; a corrupt or mismatched file is rewritten. Writes
        are atomic (tmp + os.replace), matching the stamp-extraction pattern.
        """
        response = gen_coarse_channel_response(
            fine_per_coarse, num_coarse_channels, taps_per_channel
        )

        cache_dir = os.path.join(self.config.output_path, "pfb_cache")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(
            cache_dir,
            f"pfb_response_w{fine_per_coarse}_c{num_coarse_channels}_t{taps_per_channel}.npy",
        )

        if os.path.exists(path):
            try:
                existing = np.load(path)
                if np.array_equal(existing, response):
                    return path
                logger.warning(f"PFB response cache {path} does not match; rewriting")
            except Exception as e:
                logger.warning(f"PFB response cache {path} unreadable ({e}); rewriting")

        # Per-writer tmp name (pid + uuid) so two runs sharing this output_path don't clobber
        # each other's in-progress write on a single shared "{path}.tmp"; os.replace stays
        # atomic and the content is deterministic, so whichever writer lands last is harmless.
        tmp_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with open(tmp_path, "wb") as f:
            np.save(f, response)
        os.replace(tmp_path, path)
        logger.info(f"Wrote PFB response cache: {path}")
        return path

    @staticmethod
    def _sample_channel_indices(
        n_coarse_total: int, num: int = _BANDPASS_SAMPLE_CHANNELS
    ) -> list[int]:
        """Pick up to `num` coarse-channel indices evenly across the band."""
        num = min(num, n_coarse_total)
        return sorted({int(i) for i in np.linspace(0, n_coarse_total - 1, num=num)})

    def _read_despiked_channel(self, h5_path: str, channel_index: int) -> np.ndarray:
        """Read one coarse channel as float64 with the DC spike interpolated away — the same
        preparation the energy-detection workers apply before bandpass flattening."""
        width = self.config.inference.coarse_channel_width
        time_bins = self.config.data.time_bins
        start = channel_index * width
        with h5py.File(h5_path, "r") as hf:
            channel = hf["data"][:time_bins, 0, start : start + width].astype(np.float64)
        _remove_dc_spike(channel, width, 1)
        return channel

    def _flattened_edge_mid_ratio(
        self, h5_path: str, channel_index: int, response: np.ndarray
    ) -> float:
        """
        Edge/mid power ratio of one coarse channel AFTER flattening by the static response H.

        Reads only the bands edge_mid_power_ratio actually evaluates (edge_mid_band_slices:
        the outermost width // 16 bins each side plus a central band of the same width) as
        native float32 and time-integrates each immediately — ~10x less I/O than the
        full-channel float64 read of _read_despiked_channel, and no fp64 blowup, which is what
        keeps the per-file sanity check cheap at GBT scale. The DC spike (2 bins at
        width // 2) sits inside the mid band; its interpolation (_remove_dc_spike) is linear,
        so it commutes with the time mean and is replicated on the integrated slice. Matches
        edge_mid_power_ratio(equalize_passband(channel, response).mean(axis=0)) on the
        despiked full read, up to float32-read rounding (pinned by a unit test).
        """
        width = self.config.inference.coarse_channel_width
        time_bins = self.config.data.time_bins
        start = channel_index * width
        left, mid, right = edge_mid_band_slices(width)
        # Widen the mid slice so the DC-spike interpolation sources (up to 3 bins around
        # width // 2, see _remove_dc_spike) are always present.
        lo = max(0, min(mid.start, width // 2 - 3))
        hi = max(mid.stop, width // 2 + 3)
        with h5py.File(h5_path, "r") as hf:
            data = hf["data"]
            left_int = data[:time_bins, 0, start + left.start : start + left.stop].mean(axis=0)
            right_int = data[:time_bins, 0, start + right.start : start + right.stop].mean(axis=0)
            mid_int = data[:time_bins, 0, start + lo : start + hi].mean(axis=0)
        dc = width // 2 - lo
        mid_int[dc] = (mid_int[dc + 1] + mid_int[dc - 3]) / 2
        mid_int[dc - 1] = (mid_int[dc + 2] + mid_int[dc - 2]) / 2
        mid_int = mid_int[mid.start - lo : mid.stop - lo]

        # Same divide-by-H (with equalize_passband's defensive floor) as the real flattener;
        # per-bin division commutes with the time mean, so integrating first is equivalent.
        floor = np.maximum(response, 1e-10)
        edge = 0.5 * (
            float((left_int / floor[left]).mean()) + float((right_int / floor[right]).mean())
        )
        return edge / float((mid_int / floor[mid]).mean())

    def _warn_on_pfb_response_mismatch(self, h5_path: str, n_coarse_total: int) -> None:
        """
        Residual-flatness sanity check of the static PFB response against the recording (after
        the bliss `validate` flag): flatten several coarse channels — sampled evenly across the
        band — with the active response H, and log an INFORMATIONAL warning once per file
        (never per channel) when the median flattened edge/mid power ratio deviates from 1.0
        by more than _PFB_RESIDUAL_FLATNESS_TOL.

        This directly measures the operational question — does dividing by H actually flatten
        the data? — so, unlike the raw-vs-pure-ratio comparison it replaced, the
        analog-frontend passband tilt contributes only a small residual baseline rather than
        the entire statistic (and #180's characterization bounds the response-model error at
        ~1e-5, so H itself is not a confounder). The residual still includes that frontend
        tilt and any edge RFI, so a moderate deviation can be benign and the threshold stays
        provisional/informational until the deferred pfb_taps-vs-backend characterization
        fixes the legitimate baseline; a LARGE or CONSISTENT deviation (across many files) is
        the actionable signal of a wrong --pfb-taps-per-channel. Only the edge and mid bands
        of each sampled channel are read (see _flattened_edge_mid_ratio), keeping the check
        cheap; it runs on the cadence's primary ON file only, since the response is a static
        property of the backend shared by every file.
        """
        width = self.config.inference.coarse_channel_width
        taps = self.config.inference.pfb_taps_per_channel
        try:
            response = gen_coarse_channel_response(width, n_coarse_total, taps)
            # Median (not mean) so a single RFI-heavy sampled channel doesn't skew the statistic.
            ratios = [
                self._flattened_edge_mid_ratio(h5_path, ch, response)
                for ch in self._sample_channel_indices(
                    n_coarse_total, _PFB_MISMATCH_SAMPLE_CHANNELS
                )
            ]
            median = float(np.median(ratios))
        except Exception as e:
            logger.warning(f"PFB static-response sanity check failed for {h5_path}: {e}")
            return

        deviation = abs(median - 1.0)
        if deviation > _PFB_RESIDUAL_FLATNESS_TOL:
            logger.warning(
                f"{h5_path}: median flattened edge/mid power ratio {median:.3f} deviates from "
                f"1.0 by {deviation:.1%} after dividing by the static PFB response "
                f"(residual-flatness sanity check — informational). The residual also reflects "
                f"analog-frontend tilt and edge RFI the response does not model, so a moderate "
                f"deviation can be benign. Investigate a large or consistent deviation across "
                f"many files: it may mean --pfb-taps-per-channel (currently {taps}) is wrong "
                f"for this backend, in which case --bandpass-method spline is the data-driven "
                f"fallback."
            )

    def _plot_bandpass_overlay(
        self,
        h5_path: str,
        n_coarse_total: int,
        bandpass_flatten: Callable[[np.ndarray], np.ndarray],
        npy_path: str,
    ) -> None:
        """
        Opt-in debug artifact (--bandpass-debug-plot): for a few coarse channels sampled evenly
        across the band of the cadence's primary ON-source file, plot the time-integrated
        spectrum raw vs flattened, overlaying the model being removed (the scaled PFB response
        H, or the spline fit). Saved under {output_path}/plots/inference/. Deliberately
        minimal — PR-08's inference visualization suite formalizes this figure.

        Uses matplotlib's object-oriented Figure API rather than pyplot: _process_cadence runs
        on the streaming-inference prefetch thread, and pyplot's global figure registry is not
        thread-safe.
        """
        from matplotlib.figure import Figure  # noqa: PLC0415

        width = self.config.inference.coarse_channel_width
        pfb_active = bandpass_flatten.func is _pfb_flatten_bandpass
        sampled = self._sample_channel_indices(n_coarse_total)

        fig = Figure(figsize=(14, 3.2 * len(sampled)))
        axes = fig.subplots(len(sampled), 2, squeeze=False)
        for row, ch in enumerate(sampled):
            channel = self._read_despiked_channel(h5_path, ch)
            raw = channel.mean(axis=0)
            flat = np.asarray(bandpass_flatten(channel)).mean(axis=0)
            if pfb_active:
                response = gen_coarse_channel_response(
                    width, n_coarse_total, self.config.inference.pfb_taps_per_channel
                )
                # Least-squares scale so the unit-peak response overlays the raw spectrum
                overlay = response * (float(raw @ response) / float(response @ response))
                overlay_label = "scaled PFB response H"
            else:
                overlay = _fit_channel_bandpass(raw, width, self.config.inference.spline_order)
                overlay_label = "spline fit"

            ax_raw, ax_flat = axes[row]
            # Decimated to a min/max envelope: full-resolution lines are ~1M points each at
            # GBT scale, which makes rendering slow and memory-heavy for no visual gain.
            ax_raw.plot(
                *_decimate_for_plot(raw), lw=0.6, color="tab:blue", label="raw integrated spectrum"
            )
            ax_raw.plot(
                *_decimate_for_plot(overlay),
                lw=1.2,
                ls="--",
                color="tab:orange",
                label=overlay_label,
            )
            ax_raw.set_ylabel(f"coarse channel {ch}\nintegrated power")
            ax_flat.plot(
                *_decimate_for_plot(flat),
                lw=0.6,
                color="tab:green",
                label="flattened integrated spectrum",
            )
            if row == 0:
                ax_raw.legend(loc="upper right", fontsize=8)
                ax_flat.legend(loc="upper right", fontsize=8)
            if row == len(sampled) - 1:
                ax_raw.set_xlabel("fine channel (within coarse channel)")
                ax_flat.set_xlabel("fine channel (within coarse channel)")

        method = "pfb" if pfb_active else "spline"
        fig.suptitle(f"Bandpass flattening overlay ({method}): {os.path.basename(h5_path)}")
        fig.tight_layout()

        save_dir = os.path.join(self.config.output_path, "plots", "inference")
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(npy_path))[0]
        tag = self.config.checkpoint.save_tag
        out_path = os.path.join(save_dir, f"bandpass_overlay_{stem}_{tag}.png")
        # No close/registry bookkeeping needed: an OO-API Figure is garbage-collected
        fig.savefig(out_path, dpi=120)
        logger.info(f"Saved bandpass overlay debug plot: {out_path}")

    # NOTE: come back to this later (what's the trade-off for doing dedup vs not? e.g. lower storage & compute, but higher FNR or lower DR sensitivity?)
    @staticmethod
    def _deduplicate_hits(hits: list[tuple], stamp_width: int) -> list[tuple]:
        """
        Greedy merge of hits whose centers are within stamp_width // 2,
        keeping the one with the higher statistic.
        """
        if not hits:
            return []
        sorted_hits = sorted(hits, key=lambda h: h[0])
        half = stamp_width // 2
        merged: list[tuple] = [sorted_hits[0]]
        for h in sorted_hits[1:]:
            prev = merged[-1]
            if h[0] - prev[0] < half:
                if h[1] > prev[1]:
                    merged[-1] = h
            else:
                merged.append(h)
        return merged
