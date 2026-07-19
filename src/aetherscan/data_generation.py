"""
Synthetic data generation for Aetherscan Pipeline
Handles signal injection & log-normalization for training
Uses multiprocessing and shared memory to process data in parallel

Generated rounds are written straight into disk-backed .npy memmaps (see
aetherscan.round_data): each pool task generates a batch of cadences and writes them
into its slice of the round's memmap in-place, returning only the small per-sample
stats dicts — no more per-sample IPC pickling or ~294 GB in-RAM round arrays.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import random
import shutil
import signal
import time
from dataclasses import dataclass
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import setigen as stg
from astropy import units as u
from scipy import stats as scipy_stats

from aetherscan.config import get_config
from aetherscan.db import get_db
from aetherscan.logger import init_worker_logging
from aetherscan.manager import get_manager
from aetherscan.round_data import RoundDataPaths, build_manifest, write_done_manifest

logger = logging.getLogger(__name__)

# NOTE: find a way to avoid using global refs (store under manager.py maybe?)
# Global variables to store background data for multiprocessing workers
# This avoids serialization overhead when passing data between workers
_GLOBAL_SHM = None
_GLOBAL_BACKGROUNDS = None
_GLOBAL_SHAPE = None
_GLOBAL_DTYPE = None


def _init_worker(shm_name, shape, dtype, log_queue=None):
    """
    Worker pool initializer: attach to the named shared-memory block holding the background
    plates, seed numpy/random from the worker PID so each process gets a distinct RNG state, and
    set up logging.

    Passing shm_name/shape/dtype through the pool initializer (rather than per-task args) avoids
    re-serializing the array on every map() call. `log_queue` must be passed explicitly for
    pools whose parent process was spawn-started (the RoundDataProducer's — no inherited Logger
    singleton there); fork-started pools omit it and inherit the singleton's queue. The worker
    installs a SIGTERM handler that closes its shared-memory file descriptor before letting the
    signal kill the process; the main process is responsible for unlinking shared memory
    afterwards (handled by ResourceManager).
    """
    global _GLOBAL_SHM, _GLOBAL_BACKGROUNDS, _GLOBAL_SHAPE, _GLOBAL_DTYPE

    # Initialize worker logging
    init_worker_logging(log_queue)

    # Seed processes with process IDs so each worker gets a different random state
    random.seed(os.getpid())
    np.random.seed(os.getpid())

    # Attach to existing shared memory block
    _GLOBAL_SHM = SharedMemory(name=shm_name)

    # Create numpy array view of shared memory (no copy!)
    _GLOBAL_BACKGROUNDS = np.ndarray(shape, dtype=dtype, buffer=_GLOBAL_SHM.buf)
    _GLOBAL_SHAPE = shape
    _GLOBAL_DTYPE = dtype

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


def log_norm(data: np.ndarray) -> np.ndarray:
    """Log-transform `data` (with a 1e-10 epsilon), then shift and rescale into [0, 1]."""
    # Add small epsilon to avoid log(0)
    data = data + 1e-10

    # Transform data into log-space
    data = np.log(data)
    # Shift data to be >= 0
    data = data - data.min()
    # Normalize data to [0, 1]
    if data.max() > 0:
        data = data / data.max()

    return data


# NOTE: not 100% sure how this function works. ported from Peter's code. comments added by Claude. assuming it works as intended?
def new_cadence(
    data: np.ndarray, snr: float, width_bin: int, freq_resolution: float, time_resolution: float
) -> tuple[np.ndarray, dict[str, float], bool]:
    """
    Inject a single drifting narrowband signal into a stacked cadence array.

    Returns (modified_data, signal_info, slope_was_clamped). signal_info carries the realized
    parameters {snr, drift_rate, signal_width, starting_bin, slope_pixel, y_intercept};
    slope_was_clamped is True if the drift slope was forced away from ~0 to avoid a degenerate
    near-vertical injection (downstream queries can filter on this).
    """
    # NOTE: should noise = 3 parametrized?
    # Set noise parameter (for simulating randomness in drift rate calculation)
    noise = 3

    # Randomly select a starting frequency bin (channel) to start the signal injection
    # Avoids edges (bin 0)
    starting_bin = int(random.random() * (width_bin - 1)) + 1

    # Get the total number of time samples in stacked array (typically 96 for 6 obs x 16 time bins)
    total_time = data.shape[0]

    # Randomly select a positive or negative drift direction
    if np.random.choice([-1, 1]) > 0:
        # Positive drift
        slope_pixel = total_time / starting_bin  # Signal drifts upward in frequency
        # Convert from pixel space to physical units by multiplying by time_resolution / freq_resolution ratio
        # Then add random noise to make drift rates more realistic
        slope_physical = (slope_pixel) * (
            time_resolution / freq_resolution
        ) + random.random() * noise
    else:
        # Negative drift
        slope_pixel = total_time / (starting_bin - width_bin)  # Signal drifts downward in frequency
        # Convert from pixel space to physical units by multiplying by time_resolution / freq_resolution ratio
        # Then add random noise to make drift rates more realistic
        slope_physical = (slope_pixel) * (
            time_resolution / freq_resolution
        ) - random.random() * noise

    # Clamp slopes that are too small to prevent divide-by-zero errors
    # While this may alter the physics slightly, a near-zero slope is an edge-case representing a
    # nearly horizontal signal trajectory
    # Note that we still preserve the drift direction; merely the magnitude is clamped
    # NOTE: should MIN_SLOPE_PHYSICAL = 1e-6 be parametrized in config.py instead?
    # NOTE: does slope_pixel need to be changed before db write if slope_physical is clamped?
    MIN_SLOPE_PHYSICAL = 1e-6  # noqa: N806
    slope_was_clamped = False
    if abs(slope_physical) < MIN_SLOPE_PHYSICAL:
        slope_was_clamped = True
        logger.warning(f"new_cadence: slope_physical ({slope_physical}) near zero, clamping")
        slope_physical = (
            np.sign(slope_physical) * MIN_SLOPE_PHYSICAL
            if slope_physical != 0
            else MIN_SLOPE_PHYSICAL
        )

    # Convert slope to drift rate
    drift_rate = -1 * (1 / slope_physical)

    # NOTE: should 0-50 randomness be parametrized?
    # NOTE: where does 18.0 come from? hard-coded time resolution?
    # Calculate signal width (in Hz)
    # Base random component: 0-50 Hz
    # Add component proportional to drift rate magnitude to keep signal coherent
    signal_width = random.random() * 50 + abs(drift_rate) * 18.0 / 1

    # Calculate y-intercept for linear signal trajectory
    y_intercept = total_time - slope_pixel * (starting_bin)

    # Create setigen Frame
    frame = stg.Frame.from_data(
        df=freq_resolution * u.Hz,
        dt=time_resolution * u.s,
        fch1=0 * u.MHz,  # Set reference frequency (center frequency offset)
        data=data,
        ascending=True,  # Frequency increases with channel index
    )

    # Inject signal
    signal = frame.add_signal(
        # Use linear drift trajectory starting at starting_bin & with the calculated drift rate
        stg.constant_path(
            f_start=frame.get_frequency(index=starting_bin), drift_rate=drift_rate * u.Hz / u.s
        ),
        # Constant intensity over time, calibrated to achieve target snr
        stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
        # Gaussian shape in frequency domain with calculated signal width
        stg.gaussian_f_profile(width=signal_width * u.Hz),
        # Constant bandpass profile (no frequency-dependent scaling)
        stg.constant_bp_profile(level=1),
    )

    # Extract the modified data (with signal injection) from the setigen Frame
    modified_data = frame.data.copy()

    # Cleanup intermediate data
    del frame, signal
    gc.collect()

    # Build signal info dictionary
    signal_info = {
        "snr": float(snr),
        "drift_rate": float(drift_rate),
        "signal_width": float(signal_width),
        "starting_bin": float(starting_bin),
        "slope_pixel": float(slope_pixel),
        "y_intercept": float(y_intercept),
    }

    # Return the modified data array, signal info, and clamping flag
    return modified_data, signal_info, slope_was_clamped


def check_valid_intersection(slope_1, slope_2, intercept_1, intercept_2):
    """
    Check if 2 drifting signals intersect in the ON regions
    """
    if slope_1 == slope_2:
        return True  # Parallel lines never intersect. Avoids division by 0

    x_intersect = (intercept_2 - intercept_1) / (slope_1 - slope_2)
    y_intersect = slope_1 * x_intersect + intercept_1

    on_y_coords = [(0, 16), (32, 48), (64, 80)]
    return all(not y_lower <= y_intersect <= y_upper for y_lower, y_upper in on_y_coords)


# TODO: add more sophisticated statistics for data leakage analysis
def _compute_intensity_stats(data: np.ndarray) -> dict[str, float]:
    """
    Compute global intensity statistics on a spectrogram array, flattening to 1-D first and
    promoting to float64 to avoid overflow in higher-order moments (especially pre-normalization).

    Returns {global_mean, global_median, global_std, global_mad, global_skew, global_kurtosis}.
    Empty input returns NaN for every key — write_injection_stat() will record those with
    is_finite=0 so they can be filtered out at query time.
    """
    # Return NaN for empty arrays
    # write_injection_stat() will set is_finite=0
    if data.size == 0:
        logger.warning("_compute_intensity_stats received empty array")
        return dict.fromkeys(
            [
                "global_mean",
                "global_median",
                "global_std",
                "global_mad",
                "global_skew",
                "global_kurtosis",
            ],
            float("nan"),
        )

    # Temporarily promote to float64 to prevent overflow in higher-order moments (especially for pre-normalization stats)
    flat = data.ravel().astype(np.float64)
    median_val = np.median(flat)

    stats = {
        "global_mean": float(np.mean(flat)),
        "global_median": float(median_val),
        "global_std": float(np.std(flat)),
        "global_mad": float(np.median(np.abs(flat - median_val))),
        "global_skew": float(scipy_stats.skew(flat)),
        "global_kurtosis": float(scipy_stats.kurtosis(flat)),
    }

    return stats


def create_false(
    plate: np.ndarray,
    snr_base: float,
    snr_range: float,
    width_bin: int,
    freq_resolution: float,
    time_resolution: float,
    inject: bool = True,
    dynamic_range: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Create a false-class cadence and return (final, sample_info). final has shape
    (6, 16, width_bin); sample_info carries background_index, per-stage intensity_stats (A/B/C),
    signal_info (rfi_* keys when inject=True, empty otherwise), and slope_was_clamped.

    When inject=True, a drifting RFI signal is injected into all 6 observations; when False, the
    background is log-normalized and returned as-is (Stage B = Stage A, signal_info empty).
    """
    # Select random background from plate
    background_index = int(plate.shape[0] * random.random())
    base = plate[background_index, :, :, :]

    # Initialize empty output array
    n_obs = plate.shape[1]
    n_time = plate.shape[2]
    final = np.zeros((n_obs, n_time, width_bin))

    # STAGE A: Pre-injection, pre-normalization (raw background)
    stats_a = _compute_intensity_stats(base)

    # Initialize signal info (will be populated if inject=True)
    signal_info = {}

    # Inject RFI into all 6 observations
    slope_was_clamped = False
    if inject:
        # Prepare data for signal injection by stacking all 6 observations vertically
        # (6, 16, 512) -> (96, 512)
        # Obs 0: rows 0-15, Obs 1: rows 16-31, Obs 2: rows 32-47, ...
        data = np.zeros((n_obs * n_time, width_bin))
        for i in range(n_obs):
            data[i * n_time : (i + 1) * n_time, :] = base[i, :, :]

        # Select a random SNR from the given range & inject RFI into all 6 observations
        snr = random.random() * snr_range + snr_base
        cadence, rfi_signal_info, slope_was_clamped = new_cadence(
            data, snr, width_bin, freq_resolution, time_resolution
        )

        # Prefix signal characteristics with rfi_ (this is RFI injection)
        signal_info = {f"rfi_{k}": v for k, v in rfi_signal_info.items()}

        # STAGE B: Post-injection, pre-normalization
        stats_b = _compute_intensity_stats(cadence)

        # Reshape stacked data back into original shape & log-normalize after signal injection
        for i in range(n_obs):
            final[i, :, :] = log_norm(cadence[i * n_time : (i + 1) * n_time, :])

    # Just return background. No signal injection
    else:
        # No injection: Stage B = Stage A, no signal_info
        stats_b = stats_a.copy()
        # Log-normalize base background
        for i in range(n_obs):
            final[i, :, :] = log_norm(base[i, :, :])

    # STAGE C: Post-injection, post-normalization
    stats_c = _compute_intensity_stats(final)

    sample_info = {
        "background_index": background_index,
        "intensity_stats": {"A": stats_a, "B": stats_b, "C": stats_c},
        # NOTE: how do we handle db writes & plotting when signal_info is empty?
        "signal_info": signal_info,  # Empty if no injection, rfi_* if injected
        "slope_was_clamped": slope_was_clamped,
    }

    return final, sample_info


def create_true_single(
    plate: np.ndarray,
    snr_base: float,
    snr_range: float,
    width_bin: int,
    freq_resolution: float,
    time_resolution: float,
    inject: bool | None = None,
    dynamic_range: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Create a true-single-class cadence (ETI injected into ON observations only) and return
    (final, sample_info). final has shape (6, 16, width_bin); sample_info carries
    background_index, per-stage intensity_stats (A/B/C), eti_*-prefixed signal_info, and
    slope_was_clamped.
    """
    # Select random background from plate
    background_index = int(plate.shape[0] * random.random())
    base = plate[background_index, :, :, :]

    # Initialize empty output array
    n_obs = plate.shape[1]
    n_time = plate.shape[2]
    final = np.zeros((n_obs, n_time, width_bin))

    # STAGE A: Pre-injection, pre-normalization (raw background)
    stats_a = _compute_intensity_stats(base)

    # Prepare data for signal injection by stacking all 6 observations vertically
    # (6, 16, 512) -> (96, 512)
    # Obs 0: rows 0-15, Obs 1: rows 16-31, Obs 2: rows 32-47, ...
    data = np.zeros((n_obs * n_time, width_bin))
    for i in range(n_obs):
        data[i * n_time : (i + 1) * n_time, :] = base[i, :, :]

    # Select a random SNR from the given range & inject ETI
    snr = random.random() * snr_range + snr_base
    cadence, eti_signal_info, slope_was_clamped = new_cadence(
        data, snr, width_bin, freq_resolution, time_resolution
    )

    # Prefix signal characteristics with eti_ (this is ETI injection)
    signal_info = {f"eti_{k}": v for k, v in eti_signal_info.items()}

    # STAGE B: Post-injection, pre-normalization
    stats_b = _compute_intensity_stats(cadence)

    # Reshape stacked data back into original shape & log-normalize after signal injection
    for i in range(n_obs):
        if i % 2 == 0:
            # ONs: injected signal
            final[i, :, :] = log_norm(cadence[i * n_time : (i + 1) * n_time, :])
        else:
            # OFFs: original background
            final[i, :, :] = log_norm(data[i * n_time : (i + 1) * n_time, :])

    # STAGE C: Post-injection, post-normalization
    stats_c = _compute_intensity_stats(final)

    sample_info = {
        "background_index": background_index,
        "intensity_stats": {"A": stats_a, "B": stats_b, "C": stats_c},
        "signal_info": signal_info,  # eti_* signal characteristics
        "slope_was_clamped": slope_was_clamped,
    }

    return final, sample_info


def create_true_double(
    plate: np.ndarray,
    snr_base: float,
    snr_range: float,
    width_bin: int,
    freq_resolution: float,
    time_resolution: float,
    inject: bool | None = None,
    dynamic_range: float = 1,
) -> tuple[np.ndarray, dict]:
    """
    Create a true-double-class cadence (non-intersecting ETI and RFI signals injected into
    ON-only and ON-OFF respectively) and return (final, sample_info). final has shape
    (6, 16, width_bin); sample_info carries background_index, per-stage intensity_stats (A/B/C),
    signal_info with both eti_* and rfi_* keys, and slope_was_clamped.
    """
    # Select random background from plate
    background_index = int(plate.shape[0] * random.random())
    base = plate[background_index, :, :, :]

    # Initialize empty output array
    n_obs = plate.shape[1]
    n_time = plate.shape[2]
    final = np.zeros((n_obs, n_time, width_bin))

    # STAGE A: Pre-injection, pre-normalization (raw background)
    stats_a = _compute_intensity_stats(base)

    # Prepare data for signal injection by stacking all 6 observations vertically
    # (6, 16, 512) -> (96, 512)
    # Obs 0: rows 0-15, Obs 1: rows 16-31, Obs 2: rows 32-47, ...
    data = np.zeros((n_obs * n_time, width_bin))
    for i in range(n_obs):
        data[i * n_time : (i + 1) * n_time, :] = base[i, :, :]

    # Select a random SNR from the given range
    snr = random.random() * snr_range + snr_base

    # Note, small but nonzero probability for "infinite" (long-running) loops
    # NOTE: quantified in #118 — acceptance is i.i.d. geometric with p~=0.42, so the worst
    # sample over a full 499200-round is ~25 retries (~3s); a >100-retry sample is effectively
    # impossible (P~1e-24). This loop is therefore NOT the ~10-min single-worker stall seen in
    # the #117 smoke (that is gc/IO/scheduling, see #118). #118 tracks a defensive retry cap.
    # Retry signal injection until we get valid non-intersecting signals
    while True:
        # Inject RFI
        cadence_1, rfi_signal_info, rfi_slope_clamped = new_cadence(
            data, snr, width_bin, freq_resolution, time_resolution
        )
        # Inject ETI
        cadence_2, eti_signal_info, eti_slope_clamped = new_cadence(
            cadence_1, snr * dynamic_range, width_bin, freq_resolution, time_resolution
        )

        # Extract slope and intercept for intersection check
        slope_1 = rfi_signal_info["slope_pixel"]
        intercept_1 = rfi_signal_info["y_intercept"]
        slope_2 = eti_signal_info["slope_pixel"]
        intercept_2 = eti_signal_info["y_intercept"]

        if slope_1 != slope_2 and check_valid_intersection(
            slope_1, slope_2, intercept_1, intercept_2
        ):
            break

    # Track if any slope was clamped (either RFI or ETI)
    slope_was_clamped = rfi_slope_clamped or eti_slope_clamped

    # Combine both signal infos with appropriate prefixes
    signal_info = {
        **{f"rfi_{k}": v for k, v in rfi_signal_info.items()},
        **{f"eti_{k}": v for k, v in eti_signal_info.items()},
    }

    # STAGE B: Post-injection, pre-normalization (after both injections)
    stats_b = _compute_intensity_stats(cadence_2)

    # Reshape stacked data back into original shape & log-normalize after signal injection
    for i in range(n_obs):
        if i % 2 == 0:
            # ONs: 2 injected signals (ETI + RFI)
            final[i, :, :] = log_norm(cadence_2[i * n_time : (i + 1) * n_time, :])
        else:
            # OFFs: 1 injected signal (RFI only)
            final[i, :, :] = log_norm(cadence_1[i * n_time : (i + 1) * n_time, :])

    # STAGE C: Post-injection, post-normalization
    stats_c = _compute_intensity_stats(final)

    sample_info = {
        "background_index": background_index,
        "intensity_stats": {"A": stats_a, "B": stats_b, "C": stats_c},
        "signal_info": signal_info,  # Both eti_* and rfi_* signal characteristics
        "slope_was_clamped": slope_was_clamped,
    }

    return final, sample_info


# Cadence generator functions addressable by name, so batched tasks stay picklable-cheap
# (workers resolve the callable locally instead of unpickling a function reference per task)
_CREATE_FUNCTIONS = {
    "create_false": create_false,
    "create_true_single": create_true_single,
    "create_true_double": create_true_double,
}


@dataclass(frozen=True)
class ChunkSegment:
    """
    One contiguous class-segment of a chunk: `count` cadences of a single signal type written
    at rows [start_idx, start_idx + count) of the round's `array_name` memmap. Each chunk is
    made of 8 segments (4 quarters for main, 2 halves each for false/true) — the same
    contiguous per-chunk layout the in-RAM generator used, so the labels array preserves the
    row -> signal-type mapping and the stratified split downstream keeps working unchanged.
    """

    array_name: str  # "main" | "false" | "true"
    signal_class: str  # "main" | "false" | "true" (DB signal_class)
    signal_type: str  # e.g. "false_no_signal"
    create_fn_name: str  # key into _CREATE_FUNCTIONS
    inject: bool | None
    dynamic_range: float | None
    start_idx: int  # absolute row offset into the round array
    count: int


# The 4 signal types in main-array (and labels) order within each chunk
_SIGNAL_TYPES = ("false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi")


def build_chunk_segments(chunk_start: int, chunk_size: int) -> list[ChunkSegment]:
    """
    Lay out the 8 class-segments for one chunk starting at absolute row `chunk_start`.

    Requires chunk_size % 4 == 0 (validated for signal_injection_chunk_size in cli.py), so
    quarter = chunk_size // 4 and half = chunk_size // 2 are exact — no oversample/subsample
    dance and no post-hoc stats filtering.
    """
    if chunk_size % 4 != 0:
        raise ValueError(f"chunk_size must be divisible by 4, got {chunk_size}")

    quarter = chunk_size // 4
    half = chunk_size // 2

    def seg(array_name, signal_class, signal_type, fn, inject, dyn, offset, count):
        return ChunkSegment(
            array_name=array_name,
            signal_class=signal_class,
            signal_type=signal_type,
            create_fn_name=fn,
            inject=inject,
            dynamic_range=dyn,
            start_idx=chunk_start + offset,
            count=count,
        )

    return [
        # main: 4-way balanced quarters (order matches _SIGNAL_TYPES / the labels array)
        seg("main", "main", "false_no_signal", "create_false", False, None, 0, quarter),
        seg("main", "main", "false_with_rfi", "create_false", True, None, quarter, quarter),
        seg(
            "main", "main", "true_only_eti", "create_true_single", None, None, 2 * quarter, quarter
        ),
        seg("main", "main", "true_eti_rfi", "create_true_double", None, 1.0, 3 * quarter, quarter),
        # false: 2-way balanced halves
        seg("false", "false", "false_no_signal", "create_false", False, None, 0, half),
        seg("false", "false", "false_with_rfi", "create_false", True, None, half, half),
        # true: 2-way balanced halves
        seg("true", "true", "true_only_eti", "create_true_single", None, None, 0, half),
        seg("true", "true", "true_eti_rfi", "create_true_double", None, 1.0, half, half),
    ]


def build_segment_tasks(
    segment: ChunkSegment,
    array_path: str,
    task_size: int,
    snr_base: float,
    snr_range: float,
    width_bin: int,
    freq_resolution: float,
    time_resolution: float,
    seed_rng: np.random.Generator,
) -> list[tuple]:
    """
    Partition one segment into batched worker tasks of at most `task_size` cadences each.
    Together the tasks cover rows [segment.start_idx, segment.start_idx + segment.count)
    exactly once. Each task carries a fresh RNG seed drawn from `seed_rng` so results don't
    depend on which persistent worker picks the task up.
    """
    if task_size < 1:
        raise ValueError(f"task_size must be >= 1, got {task_size}")

    tasks = []
    for offset in range(0, segment.count, task_size):
        count = min(task_size, segment.count - offset)
        seed = int(seed_rng.integers(0, 2**31 - 1))
        tasks.append(
            (
                array_path,
                segment.start_idx + offset,
                count,
                segment.create_fn_name,
                snr_base,
                snr_range,
                width_bin,
                freq_resolution,
                time_resolution,
                segment.inject,
                segment.dynamic_range,
                seed,
            )
        )
    return tasks


def _run_memmap_task(args: tuple, backgrounds: np.ndarray) -> tuple[float, list[dict]]:
    """
    Execute one batched generation task against `backgrounds`: open the target .npy r+ (like
    preprocessing._extract_stamps_worker), generate `count` cadences with the chosen create_*
    function, and write each result row straight into the memmap. Tasks address disjoint row
    ranges, so concurrent pool writes never collide. Returns (elapsed_seconds,
    [sample_info, ...]) — the only data that crosses the IPC boundary.
    """
    (
        array_path,
        start_idx,
        count,
        create_fn_name,
        snr_base,
        snr_range,
        width_bin,
        freq_resolution,
        time_resolution,
        inject,
        dynamic_range,
        seed,
    ) = args

    task_start = time.time()

    # Per-task seeding keeps results independent of worker scheduling (workers are
    # persistent, so PID-only seeding from _init_worker would tie the stream to which
    # worker happens to pick the task up).
    # NOTE: this seeds the LEGACY global RNGs because new_cadence/create_* draw from
    # random.random() and the np.random.* module API. The determinism therefore holds only
    # as long as nothing else mutates the global RNG state between cadences within a task;
    # threading an explicit np.random.Generator through the create_* functions would remove
    # that coupling if it ever matters. In practice the coupling is not exposed: pool workers
    # are single-threaded processes, and the only in-process (pool=None) generation is the RF
    # dataset, which runs after the training datasets/iterators are torn down (train_round's
    # finally: holder.clear() -> del datasets -> clear_session -> gc), so no tf.data generator
    # thread is alive mutating global RNG state during it. Note per-task seeds come from OS
    # entropy (round_data seed_rng) and are not persisted, so runs are not reproducible anyway;
    # what this guarantees is independence of results from worker scheduling within a run.
    random.seed(seed)
    np.random.seed(seed % (2**32))

    create_fn = _CREATE_FUNCTIONS[create_fn_name]
    all_sample_info = []

    out = np.lib.format.open_memmap(array_path, mode="r+")
    try:
        for i in range(count):
            cadence, sample_info = create_fn(
                backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=width_bin,
                freq_resolution=freq_resolution,
                time_resolution=time_resolution,
                inject=inject,
                dynamic_range=dynamic_range,
            )
            out[start_idx + i] = cadence
            all_sample_info.append(sample_info)
    finally:
        # Flush in the finally so rows written before a mid-task exception still reach disk
        # deterministically (a failed round has no .done manifest and reads as garbage either
        # way, but not all filesystems are guaranteed to write back dirty pages on munmap)
        out.flush()
        del out

    return time.time() - task_start, all_sample_info


def _memmap_task_worker(args: tuple) -> tuple[float, list[dict]]:
    """Pool entry point for batched memmap tasks: run against the shared-memory backgrounds."""
    return _run_memmap_task(args, _GLOBAL_BACKGROUNDS)


def write_segment_stats(db, tag: str, segment: dict) -> None:
    """
    Write one class-segment's injection stats to the DB (main-process only — the DB queue is
    a thread queue.Queue, not process-safe). `segment` is the dict emitted by
    generate_round_to_memmap's stats_cb.

    Per-sample writes: 6 intensity stats x 3 stages = 18 rows, plus 0-12 signal-characteristic
    rows depending on signal_type. Segment-level writes: 4 metadata rows.
    """
    if db is None:
        raise RuntimeError("No database instance detected - cannot write injection stats")

    round_number = segment["round_number"]
    chunk_number = segment["chunk_number"]
    signal_class = segment["signal_class"]
    signal_type = segment["signal_type"]
    timestamp = segment["timestamp"]

    for sample_idx, sample_info in enumerate(segment["stats_list"]):
        background_index = sample_info["background_index"]
        intensity_stats = sample_info["intensity_stats"]
        signal_info = sample_info["signal_info"]
        slope_was_clamped = sample_info.get("slope_was_clamped", False)

        # Intensity stats for each stage
        for stage in ["A", "B", "C"]:
            stage_stats = intensity_stats[stage]

            for stat_name in [
                "global_mean",
                "global_median",
                "global_std",
                "global_mad",
                "global_skew",
                "global_kurtosis",
            ]:
                db.write_injection_stat(
                    stat_name=stat_name,
                    value=stage_stats[stat_name],
                    round_number=round_number,
                    chunk_number=chunk_number,
                    sample_index=sample_idx,
                    background_index=background_index,
                    signal_class=signal_class,
                    signal_type=signal_type,
                    injection_stage=stage,
                    slope_clamped=slope_was_clamped,
                    tag=tag,
                    timestamp=timestamp,
                )

        # TODO: handle or verify behavior when signal_info is empty for false_no_rfi signal types
        # Signal characteristics (eti_snr, rfi_drift_rate, etc.)
        # injection_stage=None since these describe the injection itself
        for stat_name, value in signal_info.items():
            db.write_injection_stat(
                stat_name=stat_name,
                value=float(value),
                round_number=round_number,
                chunk_number=chunk_number,
                sample_index=sample_idx,
                background_index=background_index,
                signal_class=signal_class,
                signal_type=signal_type,
                injection_stage=None,
                slope_clamped=slope_was_clamped,
                tag=tag,
                timestamp=timestamp,
            )

    # Segment-level metadata stats (once per segment, not per sample)
    metadata_stats = [
        ("snr_range_floor", segment["snr_range_floor"]),
        ("snr_range_ceil", segment["snr_range_ceil"]),
        ("num_samples", float(segment["num_samples"])),
        ("inject_duration", segment["inject_duration"]),
    ]

    for stat_name, value in metadata_stats:
        db.write_injection_stat(
            stat_name=stat_name,
            value=value,
            round_number=round_number,
            chunk_number=chunk_number,
            sample_index=None,
            background_index=None,
            signal_class=signal_class,
            signal_type=signal_type,
            injection_stage=None,
            tag=tag,
            timestamp=timestamp,
        )


def generate_round_to_memmap(
    paths: RoundDataPaths,
    n_samples: int,
    snr_base: float,
    snr_range: float,
    *,
    width_bin: int,
    num_observations: int,
    time_bins: int,
    chunk_size: int,
    task_size: int,
    freq_resolution: float,
    time_resolution: float,
    pool: Pool | None = None,
    backgrounds: np.ndarray | None = None,
    round_num: int | None = None,
    stats_cb=None,
    progress_cb=None,
) -> dict:
    """
    Generate one round's triplet dataset straight into disk-backed .npy memmaps.

    main: collapsed cadences, 1/4 balanced across the 4 signal types (labels array tracks the
    per-row type); false: 1/2 false_no_signal + 1/2 false_with_rfi; true: 1/2 true_only_eti +
    1/2 true_eti_rfi — each of shape (n_samples, num_observations, time_bins, width_bin)
    float32, laid out contiguously per chunk (see build_chunk_segments).

    Work is dispatched as one unified batched task list per chunk through a single pool.map
    barrier (instead of the old 8 sequential per-class barriers); workers write rows in-place
    and return only stats dicts. With `pool` None, tasks run sequentially in-process against
    `backgrounds`. On success the labels array is saved and an atomic .done manifest is
    written (a crash mid-generation leaves no manifest, so the dir reads as garbage).

    stats_cb(segment_dict) fires once per class-segment per chunk; progress_cb(chunk,
    n_chunks) once per chunk. Returns the manifest dict.
    """
    if n_samples % 4 != 0:
        raise ValueError(f"n_samples must be divisible by 4, got {n_samples}")
    if chunk_size % 4 != 0:
        raise ValueError(f"chunk_size must be divisible by 4, got {chunk_size}")
    if pool is None and backgrounds is None:
        raise ValueError("backgrounds must be provided when no pool is given")

    wall_start = time.time()

    # Start from a clean slate: a previous partial generation (no .done manifest) may have
    # left stale arrays behind.
    shutil.rmtree(paths.round_dir, ignore_errors=True)
    os.makedirs(paths.round_dir, exist_ok=True)

    # Pre-create the three destination memmaps; workers reopen them r+ per task
    shape = (n_samples, num_observations, time_bins, width_bin)
    array_paths = paths.array_paths
    for path in array_paths.values():
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        del mm  # Close immediately: creation only reserves the file; workers do the writing

    labels = np.empty(n_samples, dtype="U20")

    n_chunks = max(1, (n_samples + chunk_size - 1) // chunk_size)
    seed_rng = np.random.default_rng()  # OS entropy; per-task seeds are drawn from this

    logger.info(
        f"Generating {n_samples} samples into {paths.round_dir} "
        f"({n_chunks} chunks of max {chunk_size}, task size {task_size})"
    )

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        this_chunk_size = min(chunk_size, n_samples - chunk_start)
        if this_chunk_size <= 0:
            break

        # Capture single timestamp for all stats in this chunk
        chunk_timestamp = time.time()

        segments = build_chunk_segments(chunk_start, this_chunk_size)

        # One unified task list across all 8 class-segments -> one pool.map barrier per chunk
        tasks: list[tuple] = []
        task_owners: list[int] = []  # task index -> segment index (map preserves order)
        for segment_idx, segment in enumerate(segments):
            segment_tasks = build_segment_tasks(
                segment,
                array_paths[segment.array_name],
                task_size,
                snr_base,
                snr_range,
                width_bin,
                freq_resolution,
                time_resolution,
                seed_rng,
            )
            tasks.extend(segment_tasks)
            task_owners.extend([segment_idx] * len(segment_tasks))

        # Labels mirror the main array's contiguous per-chunk layout
        quarter = this_chunk_size // 4
        chunk_labels: list[str] = []
        for signal_type in _SIGNAL_TYPES:
            chunk_labels.extend([signal_type] * quarter)
        labels[chunk_start : chunk_start + this_chunk_size] = chunk_labels

        logger.info(
            f"Generating chunk {chunk_idx + 1}/{n_chunks} "
            f"({this_chunk_size} samples, {len(tasks)} tasks)"
        )

        if pool is not None:
            # chunksize=1: each task is already a coarse unit of work (task_size cadences),
            # so per-task scheduling overhead is negligible and load balance is best
            results = pool.map(_memmap_task_worker, tasks, chunksize=1)
        else:
            results = [_run_memmap_task(task, backgrounds) for task in tasks]

        # Group returned stats by segment. inject_duration is the sum of the segment's
        # per-task wall times (aggregate worker time, not the barrier wall time — tasks from
        # all segments run interleaved across the pool, so per-segment barrier walls no
        # longer exist).
        segment_stats: list[list[dict]] = [[] for _ in segments]
        segment_durations = [0.0] * len(segments)
        for owner, (task_duration, task_stats) in zip(task_owners, results, strict=True):
            segment_stats[owner].extend(task_stats)
            segment_durations[owner] += task_duration

        if stats_cb is not None:
            for segment, stats_list, duration in zip(
                segments, segment_stats, segment_durations, strict=True
            ):
                stats_cb(
                    {
                        "round_number": round_num,
                        "chunk_number": chunk_idx + 1,
                        "signal_class": segment.signal_class,
                        "signal_type": segment.signal_type,
                        "snr_range_floor": snr_base,
                        "snr_range_ceil": snr_base + snr_range,
                        "num_samples": len(stats_list),
                        "inject_duration": duration,
                        "timestamp": chunk_timestamp,
                        "stats_list": stats_list,
                    }
                )

        del results, segment_stats
        gc.collect()

        if progress_cb is not None:
            progress_cb(chunk_idx + 1, n_chunks)
        logger.info(f"Chunk {chunk_idx + 1}/{n_chunks} complete")

    np.save(paths.labels_path, labels)

    # The .done manifest is only written after every chunk has finished — the same atomicity
    # idea as preprocessing's .tmp -> os.replace stamp extraction
    manifest = build_manifest(
        paths,
        n_samples=n_samples,
        snr_base=snr_base,
        snr_range=snr_range,
        wall_time_s=time.time() - wall_start,
        chunk_count=n_chunks,
    )
    write_done_manifest(paths, manifest)

    logger.info(
        f"Round data generation complete: {paths.round_dir} ({manifest['wall_time_s']:.1f}s wall)"
    )
    return manifest


class DataGenerator:
    """Synthetic data generator"""

    def __init__(
        self,
        background_plates: np.ndarray,
    ):
        """
        Initialize the generator. background_plates is an array of preprocessed background
        observations with shape (n_backgrounds, 6, 16, 512). Plates are copied into shared memory
        for worker access; the worker pool itself is created lazily on the first in-process
        generate_round() call (when overlap_data_generation is on, the RoundDataProducer process
        owns its own pool against the same shared memory, and this one is never needed for the
        beta-VAE rounds).
        """
        self.config = get_config()
        if self.config is None:
            raise ValueError("get_config() returned None")

        self.db = get_db()
        if self.db is None:
            raise ValueError("get_db() returned None")

        self.manager = get_manager()
        if self.manager is None:
            raise ValueError("get_manager() returned None")

        # Pool is created on demand by _ensure_pool()
        self.pool = None

        # Load background plates into shared memory
        self._load_backgrounds(background_plates)

    def _load_backgrounds(self, background_plates: np.ndarray):
        """Load background plates into shared memory"""
        # Sanity check: verify no NaN or Inf values in background plates
        if np.isnan(background_plates).any():
            raise ValueError("background_plates contains NaN values")
        if np.isinf(background_plates).any():
            raise ValueError("background_plates contains Inf values")

        self.n_backgrounds = len(background_plates)
        self._background_shape = background_plates.shape
        self._background_dtype = background_plates.dtype

        # Sanity check: verify downsampling working as expected
        width_bin_downsampled = self.config.data.width_bin // self.config.data.downsample_factor
        if self._background_shape[3] != width_bin_downsampled:
            raise ValueError(
                f"Expected {width_bin_downsampled} channels. Got {self._background_shape[3]} instead"
            )

        self.width_bin = width_bin_downsampled
        self.freq_resolution = self.config.data.freq_resolution
        self.time_resolution = self.config.data.time_resolution

        # Get multiprocessing params from config
        self.n_processes = self.config.manager.n_processes

        # Setup shared memory to avoid duplicating background data across workers
        if self.n_processes > 1:
            # Create shared memory block for background data
            nbytes = background_plates.nbytes
            self.shm = self.manager.create_shared_memory(
                size=nbytes,
                name=f"DataGen_backgrounds_{id(self)}",  # NOTE: come back to this later
            )

            # Copy background data into shared memory
            shared_array = np.ndarray(
                self._background_shape,
                dtype=self._background_dtype,
                buffer=self.shm.buf,  # NOTE: what is self.shm.buf?
            )
            shared_array[:] = background_plates[:]
            self.backgrounds = shared_array
        else:
            self.shm = None
            self.backgrounds = background_plates

        logger.info(f"DataGenerator initialized with {self.n_backgrounds} background plates")
        logger.info(f"  Background shape: {self._background_shape}")
        logger.info(f"  Background dtype: {self._background_dtype}")

    def _ensure_pool(self):
        """
        Lazily stand up the persistent ResourceManager-owned multiprocessing pool whose workers
        attach to the shared-memory background block at init time, so per-task dispatches don't
        have to re-serialize the plates. No-op in sequential mode (n_processes == 1, no shared
        memory) — generate_round() then runs tasks in-process.

        The pool must be released via _free_managed_pool() or close() — the ResourceManager
        won't reap it automatically.
        """
        # NOTE: should we explicitly guarantee only 1 shm & 1 pool can exist at a time?
        if self.pool is not None or not self.shm:
            return
        self.pool = self.manager.create_pool(
            n_processes=self.n_processes,
            name=f"DataGen_pool_{id(self)}",  # TODO: use a more descriptive pool name
            initializer=_init_worker,
            initargs=(self.shm.name, self._background_shape, self._background_dtype),
        )

    def _free_managed_pool(self):
        """Close multiprocessing pool"""
        if hasattr(self, "pool") and self.pool is not None:
            self.manager.close_pool(self.pool)
            self.pool = None

    def reset_managed_pool(self):
        """
        Reset multiprocessing pool

        Should be called between training rounds, since workers can accumulate memory through
        memory fragmentation in long-lived processes, python's reference counter leaking in workers,
        and caches / global state accumulating in workers. The pool is re-created lazily by
        _ensure_pool() on the next generate_round() call.
        """
        if hasattr(self, "pool") and self.pool is not None:
            try:
                self._free_managed_pool()
                gc.collect()  # Garbage collect between resets
            except Exception as e:
                logger.warning(f"Error resetting DataGenerator pool: {e}")

    def _free_managed_shared_memory(self):
        """Close shared memory"""
        if hasattr(self, "shm") and self.shm is not None:
            self.manager.close_shared_memory(self.shm)
            self.shm = None

    def close(self):
        """Free managed resources & close DataGenerator"""
        self._free_managed_pool()
        self._free_managed_shared_memory()
        logger.info("DataGenerator closed")

    def generate_round(
        self,
        paths,
        n_samples: int,
        snr_base: float,
        snr_range: float,
        round_num: int | None = None,
    ) -> dict:
        """
        Generate one round's triplet dataset into the disk-backed memmaps at `paths`
        (a round_data.RoundDataPaths), writing injection stats to the DB as chunks complete.
        Used for the sequential (non-overlapped) generation path — the RF dataset, and beta-VAE
        rounds when overlap_data_generation is disabled. Returns the .done manifest dict.
        """
        self._ensure_pool()

        tag = self.config.checkpoint.save_tag

        def _stats_cb(segment: dict) -> None:
            write_segment_stats(self.db, tag, segment)

        return generate_round_to_memmap(
            paths=paths,
            n_samples=n_samples,
            snr_base=snr_base,
            snr_range=snr_range,
            width_bin=self.width_bin,
            num_observations=self._background_shape[1],
            time_bins=self._background_shape[2],
            chunk_size=self.config.training.signal_injection_chunk_size,
            task_size=self.config.training.data_gen_task_size,
            freq_resolution=self.freq_resolution,
            time_resolution=self.time_resolution,
            pool=self.pool,
            backgrounds=self.backgrounds if self.pool is None else None,
            round_num=round_num,
            stats_cb=_stats_cb,
        )
