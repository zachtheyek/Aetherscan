"""
Synthetic data generation for Aetherscan Pipeline
Handles signal injection & log-normalization
Uses multiprocessing and shared memory to process backgrounds in parallel
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import random
import signal
import time
from multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import setigen as stg
from astropy import units as u
from scipy import stats as scipy_stats

from aetherscan.config import get_config
from aetherscan.db import get_db
from aetherscan.logger import init_worker_logging
from aetherscan.manager import get_manager

logger = logging.getLogger(__name__)

# NOTE: find a way to avoid using global refs (store under manager.py maybe?)
# NOTE: is there any room to use asyncio & load all chunks simultaneously?
# Global variables to store background data for multiprocessing workers
# This avoids serialization overhead when passing data between workers
_GLOBAL_SHM = None
_GLOBAL_BACKGROUNDS = None
_GLOBAL_SHAPE = None
_GLOBAL_DTYPE = None

# NOTE: come back to this later
# # NEW: Output shared memory for zero-copy results
# _GLOBAL_OUTPUT_SHM = None
# _GLOBAL_OUTPUT_ARRAY = None
# _GLOBAL_OUTPUT_SHAPE = None


def _init_worker(shm_name, shape, dtype):
    """
    Initialize worker process with shared memory reference and queue-based logging
    This avoids serialization overhead between workers

    Args:
        shm_name: Name of the shared memory block
        shape: Shape of the background array
        dtype: Data type of the background array

    Note:
        Worker cleanup uses a custom SIGTERM handler to properly close shared memory
        file descriptors before termination. When pool.terminate() is called by the
        main process, workers intercept SIGTERM, close their shared memory handles,
        then re-raise the signal to complete termination.

        The main process is responsible for unlinking shared memory (handled by ResourceManager).
    """
    # NOTE: come back to this later
    # def _init_worker(shm_name, shape, dtype, output_shm_name=None, output_shape=None):
    #     """
    #     Initialize worker process with shared memory references.
    #
    #     Args:
    #         shm_name: Name of the shared memory block containing background plates
    #         shape: Shape of the background array
    #         dtype: Data type of the background array
    #         output_shm_name: Name of shared memory for output results (optional)
    #         output_shape: Shape of the output array (optional)
    #
    #     Why this change:
    #         Adding output_shm_name and output_shape parameters allows workers to write
    #         results directly to shared memory instead of returning them through IPC.
    #         This eliminates ~23GB of pickle serialization per training round.
    #     """
    global _GLOBAL_SHM, _GLOBAL_BACKGROUNDS, _GLOBAL_SHAPE, _GLOBAL_DTYPE
    # global _GLOBAL_OUTPUT_SHM, _GLOBAL_OUTPUT_ARRAY, _GLOBAL_OUTPUT_SHAPE

    # Initialize worker logging
    init_worker_logging()

    # Seed processes with process IDs so each worker gets a different random state
    random.seed(os.getpid())
    np.random.seed(os.getpid())

    # Attach to existing shared memory block
    _GLOBAL_SHM = SharedMemory(name=shm_name)

    # Create numpy array view of shared memory (no copy!)
    _GLOBAL_BACKGROUNDS = np.ndarray(shape, dtype=dtype, buffer=_GLOBAL_SHM.buf)
    _GLOBAL_SHAPE = shape
    _GLOBAL_DTYPE = dtype

    # NOTE: come back to this later
    # # NEW: Attach to output shared memory if provided
    # if output_shm_name is not None and output_shape is not None:
    #     _GLOBAL_OUTPUT_SHM = SharedMemory(name=output_shm_name)
    #     _GLOBAL_OUTPUT_ARRAY = np.ndarray(
    #         output_shape, dtype=np.float32, buffer=_GLOBAL_OUTPUT_SHM.buf
    #     )
    #     _GLOBAL_OUTPUT_SHAPE = output_shape

    # Ignore SIGINT (Ctrl+C) in workers - let manager from parent handle cleanup coordination
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Setup custom SIGTERM handler for additional cleanup before termination
    # Note, manager will escalate SIGTERM to SIGKILL after pool_terminate_timeout seconds (see config.py)
    # This may interrupt the worker's cleanup process
    # Consider increasing pool_terminate_timeout if you're experiencing such issues
    def cleanup_on_sigterm(signum, frame):
        """
        Cleanup handler called when pool.terminate() sends SIGTERM
        Closes shared memory file descriptor before process termination
        """
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
    """
    Apply log normalization to data
    """
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
    Inject a single drifting narrowband signal into a stacked cadence array

    Returns:
        modified_data: Array with injected signal
        signal_info: Dict with keys: snr, drift_rate, signal_width, starting_bin, slope_pixel, y_intercept
        slope_was_clamped: True if slope was clamped due to being near zero
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
    Compute global intensity statistics on a spectrogram array.

    Args:
        data: Array of any shape (will be flattened for global stats)

    Returns:
        Dict with keys: global_mean, global_median, global_std,
                        global_mad, global_skew, global_kurtosis
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
    Create false signal class
    If specified, RFI is injected into all 6 observations. Otherwise, no RFI is injected.

    Returns:
        final: Output array of shape (6, 16, width_bin)
        sample_info: Dict with background_index, intensity_stats, signal_info
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
    Create true-single signal class
    ETI signal is injected into the ON observations only.

    Returns:
        final: Output array of shape (6, 16, width_bin)
        sample_info: Dict with background_index, intensity_stats, signal_info
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
    Create true-double signal class
    Non-intersecting ETI & RFI signals are injected into ON-only & ON-OFF, respectively.

    Returns:
        final: Output array of shape (6, 16, width_bin)
        sample_info: Dict with background_index, intensity_stats, signal_info
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

    # NOTE: small but nonzero probability for "infinite" (long-running) loops
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


def _single_cadence_wrapper(args):
    """
    Wrapper function for multiprocessing that unpacks arguments and generates a single cadence
    Uses global background plates to avoid serialization overhead

    Args:
        args: Tuple of (function, snr_base, snr_range, width_bin, freq_resolution, time_resolution, inject, dynamic_range)

    Returns:
        Single cadence array of shape (6, 16, width_bin)
    """
    (
        function,
        snr_base,
        snr_range,
        width_bin,
        freq_resolution,
        time_resolution,
        inject,
        dynamic_range,
    ) = args
    return function(
        # NOTE:
        # is _GLOBAL_BACKGROUNDS a reference to the shared memory, or the entire data array itself?
        # is this method of doing things slower due to pickling costs?
        # should we move the index selection into _single_cadence_wrapper so only one background plate is sent at a time?
        # how can we benchmark this?
        # # # Select random background from plate
        # # background_index = int(plate.shape[0] * random.random())
        # # base = plate[background_index, :, :, :]
        _GLOBAL_BACKGROUNDS,
        snr_base=snr_base,
        snr_range=snr_range,
        width_bin=width_bin,
        freq_resolution=freq_resolution,
        time_resolution=time_resolution,
        inject=inject,
        dynamic_range=dynamic_range,
    )


# NOTE: come back to this later
# def _batch_cadence_worker(args):
#     """
#     Process a batch of cadences and write results directly to shared memory.
#
#     Args:
#         args: Tuple of (start_idx, end_idx, function, snr_base, snr_range,
#                        width_bin, freq_resolution, time_resolution, inject, dynamic_range)
#
#     Returns:
#         None - results written directly to _GLOBAL_OUTPUT_ARRAY
#
#     Why this approach:
#         Instead of returning 192KB per cadence through IPC (which requires pickle
#         serialization), workers write directly to pre-allocated shared memory.
#         This eliminates the dominant bottleneck: ~23GB of IPC data transfer per round.
#
#         Processing cadences in batches (e.g., 500-1000 per task) also reduces
#         task scheduling overhead from 120,000 dispatches to ~120-240 dispatches.
#     """
#     (
#         start_idx,
#         end_idx,
#         function,
#         snr_base,
#         snr_range,
#         width_bin,
#         freq_resolution,
#         time_resolution,
#         inject,
#         dynamic_range,
#     ) = args
#
#     # Process each cadence in this batch
#     for i in range(start_idx, end_idx):
#         result = function(
#             _GLOBAL_BACKGROUNDS,
#             snr_base=snr_base,
#             snr_range=snr_range,
#             width_bin=width_bin,
#             freq_resolution=freq_resolution,
#             time_resolution=time_resolution,
#             inject=inject,
#             dynamic_range=dynamic_range,
#         )
#
#         # Write directly to shared memory output array - NO IPC!
#         _GLOBAL_OUTPUT_ARRAY[i, :, :, :] = result
#
#     # Return None - no data through IPC


def batch_create_cadence(
    function,
    samples: int,
    plate: np.ndarray,
    snr_base: int = 10,
    snr_range: float = 40,
    width_bin: int = 512,
    freq_resolution: float = 2.7939677238464355,
    time_resolution: float = 18.25361108,
    inject: bool | None = None,
    dynamic_range: float | None = None,
    pool: Pool | None = None,
    n_processes: int | None = cpu_count(),
    chunks_per_worker: int | None = 4,
) -> tuple[np.ndarray, list[dict]]:
    """
    Batch wrapper for creating multiple cadences using multiprocessing

    Args:
        function: Cadence generation function (create_false, create_true_single, create_true_double)
        samples: Number of cadences to generate
        plate: Background plate array (only used if pool is None)
        snr_base: Base SNR value
        snr_range: SNR range for randomization
        width_bin: Number of frequency bins
        freq_resolution: Frequency resolution in Hz
        time_resolution: Time resolution in seconds
        inject: Whether to inject signals (for create_false)
        dynamic_range: Dynamic range for signal injection (for create_true_double)
        pool: Pre-initialized multiprocessing Pool (if None, runs sequentially)
        n_processes: Number of processes in multiprocessing Pool (1 if running sequentially)
        chunks_per_worker: Used to calculate optimal chunksize for load balancing

    Returns:
        cadence: Array of shape (samples, 6, 16, width_bin) containing generated cadences
        all_sample_info: List of sample_info dicts (one per sample)
    """
    # Pre-allocate output array
    cadence = np.zeros((samples, 6, 16, width_bin))
    all_sample_info = []

    if pool:
        # Parallel execution using provided pool
        # Prepare arguments for each parallel task (no plate - uses global)
        args_list = [
            (
                function,
                snr_base,
                snr_range,
                width_bin,
                freq_resolution,
                time_resolution,
                inject,
                dynamic_range,
            )
            for _ in range(samples)
        ]

        # Calculate optimal chunksize for load balancing
        try:
            n_workers = pool._processes
        except AttributeError:
            n_workers = n_processes
        # NOTE: should we use separate chunks_per_worker? how to benchmark?
        chunksize = max(1, samples // (n_workers * chunks_per_worker))

        # Use pool to generate cadences in parallel
        for i, (result, sample_info) in enumerate(
            # TEST: does return order matter?
            pool.map(_single_cadence_wrapper, args_list, chunksize=chunksize)
            # pool.imap(_single_cadence_wrapper, args_list, chunksize=chunksize)
            # pool.imap_unordered(_single_cadence_wrapper, args_list, chunksize=chunksize)
        ):
            cadence[i, :, :, :] = result
            all_sample_info.append(sample_info)
    else:
        # Fallback to sequential execution
        for i in range(samples):
            result, sample_info = function(
                plate,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=width_bin,
                freq_resolution=freq_resolution,
                time_resolution=time_resolution,
                inject=inject,
                dynamic_range=dynamic_range,
            )
            cadence[i, :, :, :] = result
            all_sample_info.append(sample_info)

    return cadence, all_sample_info


# NOTE: come back to this later
# def batch_create_cadence(
#     function,
#     samples: int,
#     plate: np.ndarray,
#     snr_base: int = 10,
#     snr_range: float = 40,
#     width_bin: int = 512,
#     freq_resolution: float = 2.7939677238464355,
#     time_resolution: float = 18.25361108,
#     inject: bool | None = None,
#     dynamic_range: float | None = None,
#     pool: Pool | None = None,
#     output_shm: SharedMemory | None = None,
#     output_array: np.ndarray | None = None,
#     output_offset: int = 0,
#     n_processes: int | None = cpu_count(),
#     chunks_per_worker: int | None = 4,
#     # NOTE: parametrize this in config
#     batch_size: int = 500,  # cadences per task
# ) -> np.ndarray:
#     """
#     Batch wrapper for creating multiple cadences using multiprocessing.
#
#     Key changes from original:
#     1. Workers write directly to shared memory (output_array) instead of returning results
#     2. Tasks are batched (batch_size cadences per task) to reduce scheduling overhead
#     3. output_offset allows writing to a specific slice of the output array
#
#     Args:
#         function: Cadence generation function (create_false, create_true_single, create_true_double)
#         samples: Number of cadences to generate
#         plate: Background plate array (only used if pool is None)
#         snr_base: Base SNR value
#         snr_range: SNR range for randomization
#         width_bin: Number of frequency bins
#         freq_resolution: Frequency resolution in Hz
#         time_resolution: Time resolution in seconds
#         inject: Whether to inject signals (for create_false)
#         dynamic_range: Dynamic range for signal injection (for create_true_double)
#         pool: Pre-initialized multiprocessing Pool with shared memory
#         output_shm: SharedMemory object for output (used for cleanup tracking)
#         output_array: numpy array view of output shared memory
#         output_offset: Starting index in output_array for this batch
#         n_processes: Number of processes in multiprocessing Pool
#         chunks_per_worker: Used to calculate optimal chunksize for load balancing
#         batch_size: Number of cadences per worker task (default 500)
#
#     Returns:
#         Array of shape (samples, 6, 16, width_bin) containing generated cadences
#         (either from shared memory or newly allocated)
#     """
#     if pool and output_array is not None:
#         # OPTIMIZED PATH: Use shared memory output
#
#         # Create batched task arguments
#         # Instead of 1 task per cadence, create 1 task per batch_size cadences
#         tasks = []
#         for batch_start in range(0, samples, batch_size):
#             batch_end = min(batch_start + batch_size, samples)
#             # Indices are relative to output_offset in the shared memory
#             tasks.append(
#                 (
#                     output_offset + batch_start,  # start_idx in output array
#                     output_offset + batch_end,  # end_idx in output array
#                     function,
#                     snr_base,
#                     snr_range,
#                     width_bin,
#                     freq_resolution,
#                     time_resolution,
#                     inject,
#                     dynamic_range,
#                 )
#             )
#
#         # Calculate chunksize for pool.map
#         # With batched tasks, we have far fewer tasks, so chunksize can be smaller
#         n_tasks = len(tasks)
#         try:
#             n_workers = pool._processes
#         except AttributeError:
#             n_workers = n_processes
#
#         # Aim for ~4 chunks per worker for load balancing
#         chunksize = max(1, n_tasks // (n_workers * 4))
#
#         logger.debug(
#             f"Dispatching {n_tasks} batched tasks (batch_size={batch_size}, chunksize={chunksize})"
#         )
#
#         # Execute - results are written directly to shared memory
#         # We use pool.map which blocks until complete, but returns None for each task
#         list(pool.map(_batch_cadence_worker, tasks, chunksize=chunksize))
#
#         # Return view of the shared memory output for this batch
#         return output_array[output_offset : output_offset + samples]
#
#     elif pool:
#         # LEGACY PATH: Pool exists but no shared memory output
#         # Fall back to original IPC-based approach (for backward compatibility)
#         cadence = np.zeros((samples, 6, 16, width_bin), dtype=np.float32)
#
#         args_list = [
#             (
#                 function,
#                 snr_base,
#                 snr_range,
#                 width_bin,
#                 freq_resolution,
#                 time_resolution,
#                 inject,
#                 dynamic_range,
#             )
#             for _ in range(samples)
#         ]
#
#         try:
#             n_workers = pool._processes
#         except AttributeError:
#             n_workers = n_processes
#
#         # FIX: Use reasonable chunksize instead of always 1
#         chunksize = max(50, samples // (n_workers * 4))
#
#         for i, result in enumerate(
#             pool.imap(_single_cadence_wrapper, args_list, chunksize=chunksize)
#         ):
#             cadence[i, :, :, :] = result
#
#         return cadence
#
#     else:
#         # Sequential execution (no pool)
#         cadence = np.zeros((samples, 6, 16, width_bin), dtype=np.float32)
#
#         for i in range(samples):
#             cadence[i, :, :, :] = function(
#                 plate,
#                 snr_base=snr_base,
#                 snr_range=snr_range,
#                 width_bin=width_bin,
#                 freq_resolution=freq_resolution,
#                 time_resolution=time_resolution,
#                 inject=inject,
#                 dynamic_range=dynamic_range,
#             )
#
#         return cadence


class DataGenerator:
    """Synthetic data generator"""

    def __init__(
        self,
        background_plates: np.ndarray,
    ):
        """
        Initialize generator

        Args:
            background_plates: Array of background observations
                               Shape: (n_backgrounds, 6, 16, 512) after preprocessing
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

        # Load background plates into shared memory
        self._load_backgrounds(background_plates)

        # Setup persistent process pool for efficient parallel execution
        self._setup_managed_pool()

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
        self.chunks_per_worker = self.config.manager.chunks_per_worker

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

    def _setup_managed_pool(self):
        """
        Setup managed multiprocessing pool with shared memory

        Creates a persistent worker pool that shares access to background data via
        shared memory, avoiding costly data serialization for each worker process.

        Note:
            The pool is managed by the ResourceManager and should be closed via
            _free_managed_pool() or close() to properly release resources.
        """
        # NOTE: should we explicitly guarantee only 1 shm & 1 pool can exist at a time?
        # If shared memory exists, then create pool using shared memory reference
        if self.shm:
            self.pool = self.manager.create_pool(
                n_processes=self.n_processes,
                name=f"DataGen_pool_{id(self)}",  # NOTE: come back to this later
                initializer=_init_worker,
                initargs=(self.shm.name, self._background_shape, self._background_dtype),
            )
        # Else run in sequential mode (no pool)
        else:
            self.pool = None
            logger.info("DataGenerator running in sequential mode (n_processes=1)")

    # NOTE: come back to this later
    # def _setup_managed_pool(self):
    #     """
    #     Setup managed multiprocessing pool with shared memory.
    #
    #     Why we don't create output shared memory here:
    #         Output shared memory size depends on the batch size requested in generate_batch(),
    #         which varies between calls. We create output shared memory on-demand in generate_batch()
    #         and pass references to workers via the task arguments.
    #
    #         The pool is initialized with background shared memory only. Workers will attach to
    #         output shared memory when provided via _batch_cadence_worker args.
    #     """
    #     if self.shm:
    #         self.pool = self.manager.create_pool(
    #             n_processes=self.n_processes,
    #             name=f"DataGen_pool_{id(self)}",
    #             initializer=_init_worker,
    #             initargs=(self.shm.name, self._background_shape, self._background_dtype),
    #         )
    #     else:
    #         self.pool = None
    #         logger.info("DataGenerator running in sequential mode (n_processes=1)")

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
        and caches / global state accumulating in workers.
        """
        if hasattr(self, "pool") and self.pool is not None:
            try:
                self._free_managed_pool()
                gc.collect()  # Garbage collect between resets
                self._setup_managed_pool()
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

    # NOTE: update docstring values when more stats are added
    def _write_batch_stats(
        self,
        stats_list: list[dict],
        round_number: int | None,
        chunk_number: int,
        signal_class: str,
        signal_type: str,
        snr_range_floor: float,
        snr_range_ceil: float,
        num_samples: int,
        inject_duration: float,
        timestamp: float,
    ):
        """
        Write per-sample stats to DB.

        Per-sample writes:
        - Intensity stats: 6 stats × 3 stages = 18 writes per sample
        - Signal characteristics: 0-12 writes depending on signal_type
        - background_index is included with each write

        Batch-level writes:
        - 4 metadata stats written once per batch
        """
        tag = self.config.checkpoint.save_tag

        if self.db is None:
            raise RuntimeError(
                "No database instance detected - cannot generate training progress plot"
            )

        # Write per-sample stats
        for sample_idx, sample_info in enumerate(stats_list):
            background_index = sample_info["background_index"]
            intensity_stats = sample_info["intensity_stats"]
            signal_info = sample_info["signal_info"]
            slope_was_clamped = sample_info.get("slope_was_clamped", False)

            # Write intensity stats for each stage
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
                    self.db.write_injection_stat(
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

            # NOTE: what happens when signal_info is empty for false_no_rfi signal types?
            # Write signal characteristics (eti_snr, rfi_drift_rate, etc.)
            # injection_stage=None since these describe the injection itself
            for stat_name, value in signal_info.items():
                self.db.write_injection_stat(
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

        # Write batch-level metadata stats (once per batch, not per sample)
        metadata_stats = [
            ("snr_range_floor", snr_range_floor),
            ("snr_range_ceil", snr_range_ceil),
            ("num_samples", float(num_samples)),
            ("inject_duration", inject_duration),
        ]

        for stat_name, value in metadata_stats:
            self.db.write_injection_stat(
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

    # TODO:
    # separate generate_batch() into generate_train_batch() & generate_test_batch()
    # since test doesn't require (main, false, true), just (false, true)
    # verify this is correct with train_random_forest() vs train_round()
    # benchmark compute time / memory saved with this change
    def generate_batch(
        self, n_samples: int, snr_base: int, snr_range: int, round_num: int | None = None
    ) -> dict[str, np.ndarray]:
        """
        Generate batch using chunking & multiprocessing

        main: collapsed cadences
          - total: n_samples
          - split: 1/4 balanced between false-no-signal, false-with-rfi, true-single, true-double
        false: non-collapsed false cadences
          - total: n_samples
          - split: 1/2 balanced between false-no-signal, false-with-rfi
        true: non-collapsed true cadences
          - total: n_samples
          - split: 1/2 balanced between true-single, true-double
        """
        max_chunk_size = self.config.training.signal_injection_chunk_size
        n_chunks = max(1, (n_samples + max_chunk_size - 1) // max_chunk_size)

        logger.info(f"Generating {n_samples} samples in {n_chunks} chunks of max {max_chunk_size}")

        # Pre-allocate output arrays
        all_main = np.empty((n_samples, 6, 16, self.width_bin), dtype=np.float32)
        all_false = np.empty((n_samples, 6, 16, self.width_bin), dtype=np.float32)
        all_true = np.empty((n_samples, 6, 16, self.width_bin), dtype=np.float32)

        for chunk_idx in range(n_chunks):
            chunk_size = min(max_chunk_size, n_samples - chunk_idx * max_chunk_size)
            if chunk_size <= 0:
                break

            start_idx = chunk_idx * max_chunk_size
            end_idx = start_idx + chunk_size

            logger.info(f"Generating chunk {chunk_idx + 1}/{n_chunks} with {chunk_size} samples")

            # Capture single timestamp for all stats in this chunk
            chunk_timestamp = time.time()

            # Split chunk into equal partitions (for balanced classes)
            quarter = max(1, chunk_size // 4)
            half = max(1, chunk_size // 2)

            # Pure background (main)
            batch_start = time.time()
            quarter_false_no_signal, stats_main_false_no_signal = batch_create_cadence(
                create_false,
                quarter,
                self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                inject=False,
                pool=self.pool,
                n_processes=self.n_processes,
                chunks_per_worker=self.chunks_per_worker,
            )
            self._write_batch_stats(
                stats_list=stats_main_false_no_signal,
                round_number=round_num,
                chunk_number=chunk_idx + 1,
                signal_class="main",
                signal_type="false_no_signal",
                snr_range_floor=snr_base,
                snr_range_ceil=snr_base + snr_range,
                num_samples=quarter,
                inject_duration=time.time() - batch_start,
                timestamp=chunk_timestamp,
            )

            # RFI only (main)
            batch_start = time.time()
            quarter_false_with_rfi, stats_main_false_with_rfi = batch_create_cadence(
                create_false,
                quarter,
                self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                inject=True,
                pool=self.pool,
                n_processes=self.n_processes,
                chunks_per_worker=self.chunks_per_worker,
            )
            self._write_batch_stats(
                stats_list=stats_main_false_with_rfi,
                round_number=round_num,
                chunk_number=chunk_idx + 1,
                signal_class="main",
                signal_type="false_with_rfi",
                snr_range_floor=snr_base,
                snr_range_ceil=snr_base + snr_range,
                num_samples=quarter,
                inject_duration=time.time() - batch_start,
                timestamp=chunk_timestamp,
            )

            # ETI only (main)
            batch_start = time.time()
            quarter_true_single, stats_main_true_only_eti = batch_create_cadence(
                create_true_single,
                quarter,
                self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                pool=self.pool,
                n_processes=self.n_processes,
                chunks_per_worker=self.chunks_per_worker,
            )
            self._write_batch_stats(
                stats_list=stats_main_true_only_eti,
                round_number=round_num,
                chunk_number=chunk_idx + 1,
                signal_class="main",
                signal_type="true_only_eti",
                snr_range_floor=snr_base,
                snr_range_ceil=snr_base + snr_range,
                num_samples=quarter,
                inject_duration=time.time() - batch_start,
                timestamp=chunk_timestamp,
            )

            # ETI + RFI (main)
            batch_start = time.time()
            quarter_true_double, stats_main_true_eti_rfi = batch_create_cadence(
                create_true_double,
                quarter,
                self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                dynamic_range=1,
                pool=self.pool,
                n_processes=self.n_processes,
                chunks_per_worker=self.chunks_per_worker,
            )
            self._write_batch_stats(
                stats_list=stats_main_true_eti_rfi,
                round_number=round_num,
                chunk_number=chunk_idx + 1,
                signal_class="main",
                signal_type="true_eti_rfi",
                snr_range_floor=snr_base,
                snr_range_ceil=snr_base + snr_range,
                num_samples=quarter,
                inject_duration=time.time() - batch_start,
                timestamp=chunk_timestamp,
            )

            # Concatenate for main training data (collapsed cadences)
            chunk_main = np.concatenate(
                [
                    quarter_false_no_signal,
                    quarter_false_with_rfi,
                    quarter_true_single,
                    quarter_true_double,
                ],
                axis=0,
            )

            # Generate separate true/false non-collapsed cadences for training set diversity
            # Used to calculate clustering loss & train RF

            # Pure background (false)
            batch_start = time.time()
            half_false_no_signal, stats_false_no_signal = batch_create_cadence(
                create_false,
                half,
                self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                inject=False,
                pool=self.pool,
                n_processes=self.n_processes,
                chunks_per_worker=self.chunks_per_worker,
            )
            self._write_batch_stats(
                stats_list=stats_false_no_signal,
                round_number=round_num,
                chunk_number=chunk_idx + 1,
                signal_class="false",
                signal_type="false_no_signal",
                snr_range_floor=snr_base,
                snr_range_ceil=snr_base + snr_range,
                num_samples=half,
                inject_duration=time.time() - batch_start,
                timestamp=chunk_timestamp,
            )

            # RFI only (false)
            batch_start = time.time()
            half_false_with_rfi, stats_false_with_rfi = batch_create_cadence(
                create_false,
                half,
                self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                inject=True,
                pool=self.pool,
                n_processes=self.n_processes,
                chunks_per_worker=self.chunks_per_worker,
            )
            self._write_batch_stats(
                stats_list=stats_false_with_rfi,
                round_number=round_num,
                chunk_number=chunk_idx + 1,
                signal_class="false",
                signal_type="false_with_rfi",
                snr_range_floor=snr_base,
                snr_range_ceil=snr_base + snr_range,
                num_samples=half,
                inject_duration=time.time() - batch_start,
                timestamp=chunk_timestamp,
            )

            # ETI only (true)
            batch_start = time.time()
            half_true_single, stats_true_only_eti = batch_create_cadence(
                create_true_single,
                half,
                self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                pool=self.pool,
                n_processes=self.n_processes,
                chunks_per_worker=self.chunks_per_worker,
            )
            self._write_batch_stats(
                stats_list=stats_true_only_eti,
                round_number=round_num,
                chunk_number=chunk_idx + 1,
                signal_class="true",
                signal_type="true_only_eti",
                snr_range_floor=snr_base,
                snr_range_ceil=snr_base + snr_range,
                num_samples=half,
                inject_duration=time.time() - batch_start,
                timestamp=chunk_timestamp,
            )

            # ETI + RFI (true)
            batch_start = time.time()
            half_true_double, stats_true_eti_rfi = batch_create_cadence(
                create_true_double,
                half,
                self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                dynamic_range=1,
                pool=self.pool,
                n_processes=self.n_processes,
                chunks_per_worker=self.chunks_per_worker,
            )
            self._write_batch_stats(
                stats_list=stats_true_eti_rfi,
                round_number=round_num,
                chunk_number=chunk_idx + 1,
                signal_class="true",
                signal_type="true_eti_rfi",
                snr_range_floor=snr_base,
                snr_range_ceil=snr_base + snr_range,
                num_samples=half,
                inject_duration=time.time() - batch_start,
                timestamp=chunk_timestamp,
            )

            chunk_false = np.concatenate([half_false_no_signal, half_false_with_rfi], axis=0)

            chunk_true = np.concatenate([half_true_single, half_true_double], axis=0)

            # Store chunks directly into output array
            all_main[start_idx:end_idx] = chunk_main
            all_false[start_idx:end_idx] = chunk_false
            all_true[start_idx:end_idx] = chunk_true

            # Clean up chunk data immediately
            del (
                quarter_false_no_signal,
                quarter_false_with_rfi,
                quarter_true_single,
                quarter_true_double,
            )
            del half_false_no_signal, half_false_with_rfi, half_true_single, half_true_double
            del chunk_main, chunk_false, chunk_true
            del stats_main_false_no_signal, stats_main_false_with_rfi
            del stats_main_true_only_eti, stats_main_true_eti_rfi
            del stats_false_no_signal, stats_false_with_rfi
            del stats_true_only_eti, stats_true_eti_rfi
            gc.collect()

            logger.info(f"Chunk {chunk_idx + 1} complete, memory cleared")

        # Create result dictionary with references to pre-allocated arrays
        result = {"concatenated": all_main, "false": all_false, "true": all_true}

        # NOTE: is there a more efficient way to do this? these checks currently take a few minutes to complete. should we comment this portion out?
        # Sanity check: verify post-injection data normalization
        for key in ["concatenated", "false", "true"]:
            min_val = np.min(result[key])
            max_val = np.max(result[key])
            mean_val = np.mean(result[key])
            logger.info(
                f"Post-injection {key} stats: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}"
            )
            if max_val > 1.0:
                logger.error(f"Post-injection {key} values too large! Max: {max_val}")
                raise ValueError(f"Post-injection {key} normalization check failed")
            elif min_val < 0.0:
                logger.error(f"Post-injection {key} values too small! Min: {min_val}")
                raise ValueError(f"Post-injection {key} normalization check failed")
            elif np.isnan(result[key]).any():
                logger.error(f"Post-injection {key} contains NaN values!")
                raise ValueError(f"Post-injection {key} normalization check failed")
            elif np.isinf(result[key]).any():
                logger.error(f"Post-injection {key} contains Inf values!")
                raise ValueError(f"Post-injection {key} normalization check failed")
            else:
                logger.info(f"Post-injection {key} data properly normalized")

        return result

    # NOTE: come back to this later
    # def generate_batch(
    #     self, n_samples: int, snr_base: int, snr_range: int
    # ) -> dict[str, np.ndarray]:
    #     """
    #     Generate batch using unified task submission and shared memory outputs.
    #
    #     Key optimizations:
    #     1. Single shared memory allocation for all outputs
    #     2. All 8 batch types submitted as unified task queue
    #     3. Workers write directly to shared memory (no IPC returns)
    #     4. Single synchronization point instead of 8
    #
    #     Output structure:
    #         main: collapsed cadences (n_samples total)
    #           - 1/4 false-no-signal, 1/4 false-with-rfi, 1/4 true-single, 1/4 true-double
    #         false: non-collapsed false cadences (n_samples total)
    #           - 1/2 false-no-signal, 1/2 false-with-rfi
    #         true: non-collapsed true cadences (n_samples total)
    #           - 1/2 true-single, 1/2 true-double
    #     """
    #     max_chunk_size = self.config.training.signal_injection_chunk_size
    #     n_chunks = max(1, (n_samples + max_chunk_size - 1) // max_chunk_size)
    #
    #     logger.info(f"Generating {n_samples} samples in {n_chunks} chunks of max {max_chunk_size}")
    #
    #     # Pre-allocate output arrays
    #     all_main = np.empty((n_samples, 6, 16, self.width_bin), dtype=np.float32)
    #     all_false = np.empty((n_samples, 6, 16, self.width_bin), dtype=np.float32)
    #     all_true = np.empty((n_samples, 6, 16, self.width_bin), dtype=np.float32)
    #
    #     for chunk_idx in range(n_chunks):
    #         chunk_size = min(max_chunk_size, n_samples - chunk_idx * max_chunk_size)
    #         if chunk_size <= 0:
    #             break
    #
    #         start_idx = chunk_idx * max_chunk_size
    #         end_idx = start_idx + chunk_size
    #
    #         logger.info(f"Generating chunk {chunk_idx + 1}/{n_chunks} with {chunk_size} samples")
    #
    #         # Calculate sample counts for this chunk
    #         quarter = max(1, chunk_size // 4)
    #         half = max(1, chunk_size // 2)
    #
    #         # Total outputs needed for this chunk:
    #         # - main: 4 * quarter = chunk_size
    #         # - false: 2 * half = chunk_size
    #         # - true: 2 * half = chunk_size
    #         # Total: 3 * chunk_size
    #         total_outputs = 3 * chunk_size
    #
    #         # Create output shared memory for this chunk
    #         output_shape = (total_outputs, 6, 16, self.width_bin)
    #         output_nbytes = int(np.prod(output_shape) * np.float32().nbytes)
    #
    #         output_shm = self.manager.create_shared_memory(
    #             size=output_nbytes,
    #             name=f"DataGen_output_chunk_{chunk_idx}_{id(self)}",
    #         )
    #
    #         # Create numpy view of output shared memory
    #         output_array = np.ndarray(output_shape, dtype=np.float32, buffer=output_shm.buf)
    #
    #         # Reinitialize pool with output shared memory reference
    #         # Workers need to attach to the new output shared memory
    #         if self.pool is not None:
    #             self._free_managed_pool()
    #
    #         self.pool = self.manager.create_pool(
    #             n_processes=self.n_processes,
    #             name=f"DataGen_pool_chunk_{chunk_idx}_{id(self)}",
    #             initializer=_init_worker,
    #             initargs=(
    #                 self.shm.name,
    #                 self._background_shape,
    #                 self._background_dtype,
    #                 output_shm.name,  # NEW: output shared memory
    #                 output_shape,  # NEW: output shape
    #             ),
    #         )
    #
    #         # Define output layout in shared memory:
    #         # [0:quarter]                          -> main: false_no_signal
    #         # [quarter:2*quarter]                  -> main: false_with_rfi
    #         # [2*quarter:3*quarter]                -> main: true_single
    #         # [3*quarter:4*quarter]                -> main: true_double
    #         # [chunk_size:chunk_size+half]         -> false: no_signal
    #         # [chunk_size+half:chunk_size+2*half]  -> false: with_rfi
    #         # [2*chunk_size:2*chunk_size+half]     -> true: single
    #         # [2*chunk_size+half:3*chunk_size]     -> true: double
    #
    #         # Build unified task list for all 8 batch types
    #         batch_size = 500  # Cadences per task
    #         all_tasks = []
    #
    #         # Helper to create batched tasks for a given output range
    #         def add_tasks(
    #             output_start,
    #             count,
    #             function,
    #             inject=None,
    #             dynamic_range=None,
    #             _batch_size=batch_size,
    #             _all_tasks=all_tasks,
    #         ):
    #             for batch_start in range(0, count, _batch_size):
    #                 batch_end = min(batch_start + _batch_size, count)
    #                 _all_tasks.append(
    #                     (
    #                         output_start + batch_start,
    #                         output_start + batch_end,
    #                         function,
    #                         snr_base,
    #                         snr_range,
    #                         self.width_bin,
    #                         self.freq_resolution,
    #                         self.time_resolution,
    #                         inject,
    #                         dynamic_range,
    #                     )
    #                 )
    #
    #         # Main outputs (quarters)
    #         add_tasks(0, quarter, create_false, inject=False)
    #         add_tasks(quarter, quarter, create_false, inject=True)
    #         add_tasks(2 * quarter, quarter, create_true_single)
    #         add_tasks(3 * quarter, quarter, create_true_double, dynamic_range=1)
    #
    #         # False outputs (halves)
    #         add_tasks(chunk_size, half, create_false, inject=False)
    #         add_tasks(chunk_size + half, half, create_false, inject=True)
    #
    #         # True outputs (halves)
    #         add_tasks(2 * chunk_size, half, create_true_single)
    #         add_tasks(2 * chunk_size + half, half, create_true_double, dynamic_range=1)
    #
    #         logger.info(f"Submitting {len(all_tasks)} unified tasks to pool")
    #
    #         # Execute ALL tasks in single pool.map call
    #         # This is the key optimization: one sync barrier instead of 8
    #         n_workers = self.n_processes
    #         chunksize = max(1, len(all_tasks) // (n_workers * 4))
    #
    #         list(self.pool.map(_batch_cadence_worker, all_tasks, chunksize=chunksize))
    #
    #         logger.info("All tasks complete, extracting results from shared memory")
    #
    #         # Extract results from shared memory into output arrays
    #         # Main outputs
    #         chunk_main = np.concatenate(
    #             [
    #                 output_array[0:quarter],
    #                 output_array[quarter : 2 * quarter],
    #                 output_array[2 * quarter : 3 * quarter],
    #                 output_array[3 * quarter : 4 * quarter],
    #             ],
    #             axis=0,
    #         )
    #
    #         # False outputs
    #         chunk_false = np.concatenate(
    #             [
    #                 output_array[chunk_size : chunk_size + half],
    #                 output_array[chunk_size + half : chunk_size + 2 * half],
    #             ],
    #             axis=0,
    #         )
    #
    #         # True outputs
    #         chunk_true = np.concatenate(
    #             [
    #                 output_array[2 * chunk_size : 2 * chunk_size + half],
    #                 output_array[2 * chunk_size + half : 3 * chunk_size],
    #             ],
    #             axis=0,
    #         )
    #
    #         # Store chunks in pre-allocated arrays
    #         all_main[start_idx:end_idx] = chunk_main
    #         all_false[start_idx:end_idx] = chunk_false
    #         all_true[start_idx:end_idx] = chunk_true
    #
    #         # Cleanup chunk resources
    #         del chunk_main, chunk_false, chunk_true
    #         del output_array
    #
    #         # Close pool before closing shared memory (workers have references)
    #         self._free_managed_pool()
    #         self.manager.close_shared_memory(output_shm)
    #
    #         gc.collect()
    #
    #         logger.info(f"Chunk {chunk_idx + 1} complete, memory cleared")
    #
    #     # Recreate pool for next generate_batch call
    #     self._setup_managed_pool()
    #
    #     result = {"concatenated": all_main, "false": all_false, "true": all_true}
    #
    #     return result
