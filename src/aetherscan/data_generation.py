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
from multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import setigen as stg
from astropy import units as u

from aetherscan.config import get_config
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
    global _GLOBAL_SHM, _GLOBAL_BACKGROUNDS, _GLOBAL_SHAPE, _GLOBAL_DTYPE

    # Initialize worker logging
    init_worker_logging()

    # Seed processes with process IDs so each worker gets a different random state
    random.seed(os.getpid())
    np.random.seed(os.getpid())

    # Attach to existing shared memory block
    _GLOBAL_SHM = SharedMemory(name=shm_name)

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
                # TEST:
                # logger.info(f"Closing shared memory file descriptor in worker PID {os.getpid()}")
                _GLOBAL_SHM.close()

        # Restore default handler and re-raise SIGTERM to resume termination
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    # Register SIGTERM handler for graceful cleanup on pool.terminate()
    signal.signal(signal.SIGTERM, cleanup_on_sigterm)

    # Create numpy array view of shared memory (no copy!)
    _GLOBAL_BACKGROUNDS = np.ndarray(shape, dtype=dtype, buffer=_GLOBAL_SHM.buf)
    _GLOBAL_SHAPE = shape
    _GLOBAL_DTYPE = dtype


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
# NOTE: verify that we're randomly drawing a combo of snr, drift_rate, and signal_width for each injection?
def new_cadence(
    data: np.ndarray, snr: float, width_bin: int, freq_resolution: float, time_resolution: float
) -> tuple[np.ndarray, float, float]:
    """
    Inject a single drifting narrowband signal into a stacked cadence array
    """
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

    # Convert slope to drift rate
    drift_rate = -1 * (1 / slope_physical)

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

    # Return the modified data array, slope (in pixel coordinates), and y-intercept
    return modified_data, slope_pixel, y_intercept


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


def create_false(
    plate: np.ndarray,
    snr_base: float,
    snr_range: float,
    width_bin: int,
    freq_resolution: float,
    time_resolution: float,
    inject: bool = True,
    dynamic_range: float | None = None,
) -> np.ndarray:
    """
    Create false signal class
    If specified, RFI is injected into all 6 observations. Otherwise, no RFI is injected
    """
    # Select random background from plate
    background_index = int(plate.shape[0] * random.random())
    base = plate[background_index, :, :, :]

    # Initialize empty output array
    n_obs = plate.shape[1]
    n_time = plate.shape[2]
    final = np.zeros((n_obs, n_time, width_bin))

    # Inject RFI into all 6 observations
    if inject:
        # Prepare data for signal injection by stacking all 6 observations vertically
        # (6, 16, 512) -> (96, 512)
        # Obs 0: rows 0-15, Obs 1: rows 16-31, Obs 2: rows 32-47, ...
        data = np.zeros((n_obs * n_time, width_bin))
        for i in range(n_obs):
            data[i * n_time : (i + 1) * n_time, :] = base[i, :, :]

        # Select a random SNR from the given range & inject RFI into all 6 observations
        snr = random.random() * snr_range + snr_base
        cadence, _, _ = new_cadence(data, snr, width_bin, freq_resolution, time_resolution)

        # Reshape stacked data back into original shape & log-normalize after signal injection
        for i in range(n_obs):
            final[i, :, :] = log_norm(cadence[i * n_time : (i + 1) * n_time, :])

    # Just return background. No signal injection
    else:
        # Log-normalize base background
        for i in range(n_obs):
            final[i, :, :] = log_norm(base[i, :, :])

    return final


def create_true_single(
    plate: np.ndarray,
    snr_base: float,
    snr_range: float,
    width_bin: int,
    freq_resolution: float,
    time_resolution: float,
    inject: bool | None = None,
    dynamic_range: float | None = None,
) -> np.ndarray:
    """
    Create true-single signal class
    ETI signal is injected into the ON observations only
    """
    # Select random background from plate
    background_index = int(plate.shape[0] * random.random())
    base = plate[background_index, :, :, :]

    # Initialize empty output array
    n_obs = plate.shape[1]
    n_time = plate.shape[2]
    final = np.zeros((n_obs, n_time, width_bin))

    # Prepare data for signal injection by stacking all 6 observations vertically
    # (6, 16, 512) -> (96, 512)
    # Obs 0: rows 0-15, Obs 1: rows 16-31, Obs 2: rows 32-47, ...
    data = np.zeros((n_obs * n_time, width_bin))
    for i in range(n_obs):
        data[i * n_time : (i + 1) * n_time, :] = base[i, :, :]

    # Select a random SNR from the given range & inject RFI
    snr = random.random() * snr_range + snr_base
    cadence, _, _ = new_cadence(data, snr, width_bin, freq_resolution, time_resolution)

    # Reshape stacked data back into original shape & log-normalize after signal injection
    for i in range(n_obs):
        if i % 2 == 0:
            # ONs: injected signal
            final[i, :, :] = log_norm(cadence[i * n_time : (i + 1) * n_time, :])
        else:
            # OFFs: original background
            final[i, :, :] = log_norm(data[i * n_time : (i + 1) * n_time, :])

    return final


def create_true_double(
    plate: np.ndarray,
    snr_base: float,
    snr_range: float,
    width_bin: int,
    freq_resolution: float,
    time_resolution: float,
    inject: bool | None = None,
    dynamic_range: float = 1,
) -> np.ndarray:
    """
    Create true-double signal class
    Non-intersecting ETI & RFI signals are injected into ON-only & ON-OFF, respectively
    """
    # Select random background from plate
    background_index = int(plate.shape[0] * random.random())
    base = plate[background_index, :, :, :]

    # Initialize empty output array
    n_obs = plate.shape[1]
    n_time = plate.shape[2]
    final = np.zeros((n_obs, n_time, width_bin))

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
        cadence_1, slope_1, intercept_1 = new_cadence(
            data, snr, width_bin, freq_resolution, time_resolution
        )
        # Inject ETI
        cadence_2, slope_2, intercept_2 = new_cadence(
            cadence_1, snr * dynamic_range, width_bin, freq_resolution, time_resolution
        )

        if slope_1 != slope_2 and check_valid_intersection(
            slope_1, slope_2, intercept_1, intercept_2
        ):
            break

    # Reshape stacked data back into original shape & log-normalize after signal injection
    for i in range(n_obs):
        if i % 2 == 0:
            # ONs: 2 injected signals (ETI + RFI)
            final[i, :, :] = log_norm(cadence_2[i * n_time : (i + 1) * n_time, :])
        else:
            # OFFs: 1 injected signal (RFI only)
            final[i, :, :] = log_norm(cadence_1[i * n_time : (i + 1) * n_time, :])

    return final


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
) -> np.ndarray:
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
        Array of shape (samples, 6, 16, width_bin) containing generated cadences
    """
    # Pre-allocate output array
    cadence = np.zeros((samples, 6, 16, width_bin))

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
        for i, result in enumerate(
            # NOTE: does return order matter?
            pool.imap(_single_cadence_wrapper, args_list, chunksize=chunksize)
            # pool.imap_unordered(_single_cadence_wrapper, args_list, chunksize=chunksize)
        ):
            cadence[i, :, :, :] = result
    else:
        # Fallback to sequential execution
        for i in range(samples):
            cadence[i, :, :, :] = function(
                plate,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=width_bin,
                freq_resolution=freq_resolution,
                time_resolution=time_resolution,
                inject=inject,
                dynamic_range=dynamic_range,
            )

    return cadence


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

        self.manager = get_manager()

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

    # TODO:
    # separate generate_batch() into generate_train_batch() & generate_test_batch()
    # since test doesn't require (main, false, true), just (false, true)
    # verify this is correct with train_random_forest() vs train_round()
    # benchmark compute time / memory saved with this change
    def generate_batch(
        self, n_samples: int, snr_base: int, snr_range: int
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

            # Split chunk into equal partitions (for balanced classes)
            quarter = max(1, chunk_size // 4)
            half = max(1, chunk_size // 2)

            # Pure background
            quarter_false_no_signal = batch_create_cadence(
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

            # RFI only
            quarter_false_with_rfi = batch_create_cadence(
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

            # ETI only
            quarter_true_single = batch_create_cadence(
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

            # ETI + RFI
            quarter_true_double = batch_create_cadence(
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
            half_false_no_signal = batch_create_cadence(
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

            half_false_with_rfi = batch_create_cadence(
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

            half_true_single = batch_create_cadence(
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

            half_true_double = batch_create_cadence(
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
            gc.collect()

            logger.info(f"Chunk {chunk_idx + 1} complete, memory cleared")

        # Create result dictionary with references to pre-allocated arrays
        result = {"concatenated": all_main, "false": all_false, "true": all_true}

        # NOTE: is there a more efficient way to do this?
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
