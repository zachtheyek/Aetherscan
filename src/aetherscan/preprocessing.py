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
import gc
import logging
import os
import signal
from collections import OrderedDict
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory

import h5py
import hdf5plugin  # noqa: F401  # registers bitshuffle codec with h5py at import time
import numpy as np
from scipy import interpolate, stats
from skimage.transform import downscale_local_mean

from aetherscan.config import get_config
from aetherscan.data_generation import log_norm
from aetherscan.db import get_db
from aetherscan.logger import init_worker_logging
from aetherscan.manager import get_manager

logger = logging.getLogger(__name__)

# NOTE: find a way to avoid using global refs (store under manager.py maybe?)
# NOTE: is there any room to use asyncio & load all chunks simultaneously?
# Global variable to store chunk data for multiprocessing workers
# This avoids serialization overhead when passing data between workers
_GLOBAL_SHM = None
_GLOBAL_CHUNK_DATA = None
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

    # Create numpy array view of shared memory (no copy!)
    _GLOBAL_CHUNK_DATA = np.ndarray(shape, dtype=dtype, buffer=_GLOBAL_SHM.buf)
    _GLOBAL_SHAPE = shape
    _GLOBAL_DTYPE = dtype


# NOTE: come back to this later
def _downsample_worker(args):
    """
    Worker function to downsample a single cadence in parallel
    Uses global chunk data to avoid serialization overhead

    Args:
        args: Tuple of (cadence_idx, downsample_factor, final_width)

    Returns:
        Downsampled cadence of shape (6, 16, final_width) or None if invalid
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


# --- Energy detection helpers & workers ---


def _remove_dc_spike(
    block_data: np.ndarray, coarse_channel_width: int, n_coarse_channels: int
) -> None:
    """
    Interpolate over the 2-bin DC spike at the center of each coarse channel, in place.

    Mirrors preprocess_fine.py:72-75 from the reference implementation.

    Args:
        block_data: shape (time_bins, n_coarse_channels * coarse_channel_width)
        coarse_channel_width: fine channels per coarse channel
        n_coarse_channels: number of coarse channels in block_data
    """
    half_chan = coarse_channel_width // 2
    for i in range(n_coarse_channels):
        dc_ind = i * coarse_channel_width + half_chan
        block_data[:, dc_ind] = (block_data[:, dc_ind + 1] + block_data[:, dc_ind - 3]) / 2
        block_data[:, dc_ind - 1] = (block_data[:, dc_ind + 2] + block_data[:, dc_ind - 2]) / 2


def _fit_channel_bandpass(
    integrated_channel: np.ndarray, channel_width: int, spl_order: int
) -> np.ndarray:
    """
    Fit a spline bandpass to a time-integrated coarse channel.

    Mirrors utils.py:17-22 from the reference implementation.

    Args:
        integrated_channel: 1-D array of shape (channel_width,), time-integrated
        channel_width: fine channels per coarse channel
        spl_order: spline order (higher = more knots = finer fit)

    Returns:
        1-D bandpass fit of shape (channel_width,)
    """
    x = np.arange(channel_width)
    knots = np.arange(0, channel_width, channel_width // spl_order + 1)
    spl = interpolate.splrep(x, integrated_channel, t=knots[1:])
    return interpolate.splev(x, spl)


def _read_coarse_channel_worker(args: tuple) -> np.ndarray:
    """
    Worker: read one coarse channel from an .h5 file.

    Workers open their own h5py.File (h5py file handles are fork-unsafe to share).

    Args:
        args: (h5_path, channel_index, coarse_channel_width)

    Returns:
        ndarray of shape (time_bins, coarse_channel_width)
    """
    h5_path, channel_index, coarse_channel_width = args
    start = channel_index * coarse_channel_width
    end = (channel_index + 1) * coarse_channel_width
    with h5py.File(h5_path, "r") as hf:
        return hf["data"][:, 0, start:end]


def _remove_bandpass_worker(args: tuple) -> np.ndarray:
    """
    Worker: subtract the per-coarse-channel spline bandpass from one coarse channel.

    Reads its slice from _GLOBAL_CHUNK_DATA (a (time_bins, n_coarse*width) view of
    the current block's shared memory).

    Args:
        args: (channel_index, coarse_channel_width, spl_order)

    Returns:
        Bandpass-cleaned slice of shape (time_bins, coarse_channel_width)
    """
    channel_index, coarse_channel_width, spl_order = args

    if _GLOBAL_CHUNK_DATA is None:
        logger.warning("No global chunk data available for bandpass removal")
        return np.zeros((0, coarse_channel_width))

    start = channel_index * coarse_channel_width
    end = (channel_index + 1) * coarse_channel_width
    channel = _GLOBAL_CHUNK_DATA[:, start:end]
    integrated_channel = np.mean(channel, axis=0)
    fit = _fit_channel_bandpass(integrated_channel, coarse_channel_width, spl_order)
    return channel - fit


def _threshold_hits_worker(args: tuple) -> list[tuple]:
    """
    Worker: slide a window across one coarse channel and emit hits above threshold.

    The window is run through scipy.stats.normaltest (D'Agostino-Pearson).
    Reads its slice from _GLOBAL_CHUNK_DATA (cleaned residuals for this block).

    Args:
        args: (channel_index, coarse_channel_width, window_size, step_size,
               stat_threshold, block_offset)
            block_offset is added to the returned absolute index so callers see
            indices in the full spectrum rather than block-relative.

    Returns:
        List of (absolute_fine_channel_index, statistic, pvalue) tuples.
    """
    (
        channel_index,
        coarse_channel_width,
        window_size,
        step_size,
        stat_threshold,
        block_offset,
    ) = args

    if _GLOBAL_CHUNK_DATA is None:
        logger.warning("No global chunk data available for hit thresholding")
        return []

    start = channel_index * coarse_channel_width
    end = (channel_index + 1) * coarse_channel_width
    channel = _GLOBAL_CHUNK_DATA[:, start:end]

    hits: list[tuple] = []
    for i in range(0, coarse_channel_width - window_size, step_size):
        window = channel[:, i : i + window_size]
        s, p = stats.normaltest(window.flatten())
        if s > stat_threshold:
            abs_idx = block_offset + channel_index * coarse_channel_width + i
            hits.append((abs_idx, float(s), float(p)))

    return hits


# def _drop_side_channels(
#     block_data: np.ndarray, side_channel_count: int, coarse_channel_width: int
# ) -> None:
#     """Zero out the leading/trailing side_channel_count coarse channels of a block.
#     Reserved for future use — leave as-is until the energy-profile criterion is defined."""
#     pass


@dataclass
class CadenceGroup:
    """One cadence's worth of observations grouped from a CSV."""

    key: tuple  # the group-by column values
    h5_paths: list[str]  # observation .h5 paths, in row order
    csv_path: str  # source CSV
    expected_obs: int
    is_valid: bool  # True iff len(h5_paths) == expected_obs


@dataclass
class CadenceResult:
    """Output of processing one cadence."""

    npy_path: str
    h5_paths: list[str]
    key: tuple
    n_hits: int
    metadata_path: str  # sibling .pkl with hit details


@dataclass
class CadenceHit:
    """A single energy detection hit on an ON-source observation."""

    fine_channel: int  # absolute fine-channel index into the full spectrum
    statistic: float  # D'Agostino-Pearson statistic
    pvalue: float
    frequency_mhz: float = field(default=float("nan"))


def group_observations_from_csv(
    csv_path: str,
    group_by_cols: list[str],
    h5_path_col: str,
    expected_obs: int = 6,
) -> tuple[list[CadenceGroup], list[CadenceGroup]]:
    """
    Group rows of a CSV into cadences.

    Rows are grouped by the joint value of `group_by_cols` (rows are assumed
    already ordered correctly within each group in the source CSV). The function
    is column-agnostic: it never assumes specific column names beyond what the
    caller provides.

    Args:
        csv_path: path to CSV
        group_by_cols: columns whose joint value defines cadence membership
        h5_path_col: column containing the .h5 file path for that observation
        expected_obs: required number of observations per cadence (typically 6)

    Returns:
        (valid_groups, flagged_groups) — flagged groups have wrong obs count.

    Raises:
        FileNotFoundError: if csv_path doesn't exist.
        KeyError: if any column in group_by_cols + [h5_path_col] is missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    required_cols = list(group_by_cols) + [h5_path_col]

    groups: OrderedDict[tuple, list[str]] = OrderedDict()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        available = reader.fieldnames or []

        missing = [c for c in required_cols if c not in available]
        if missing:
            raise KeyError(
                f"CSV {csv_path} is missing required column(s): {missing}. "
                f"Available columns: {available}"
            )

        for row in reader:
            key = tuple(row[c] for c in group_by_cols)
            groups.setdefault(key, []).append(row[h5_path_col])

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

        logger.info("DataPreprocessor closed")

    # NOTE: shared resources currently created & destroyed within function itself. think about abstractions once preprocessing.py is complete
    def load_train_data(self) -> np.ndarray:
        """
        Load & preprocess data for training
        Uses parallel processing to load and downsample the data (log-normalization is deferred to data_generation.py)

        Returns:
            Array of preprocessed cadences with shape (n, 6, 16, width_bin_downsampled)
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

                # NOTE: is this access pattern the most efficient (least pickling)? see comments in _single_cadence_wrapper() from data_generation.py for more details
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
    def load_inference_data(self) -> np.ndarray:
        """
        Load & preprocess data for inference
        Uses parallel processing to load, downsample, and log-normalize the data

        Returns:
            Array of preprocessed cadences with shape (n, 6, 16, width_bin_downsampled)
        """
        logger.info(f"Loading backgrounds from {self.config.data_path} for inference")

        downsample_factor = self.config.data.downsample_factor
        final_width = self.config.data.width_bin // downsample_factor

        chunk_size = self.config.data.inference_background_load_chunk_size
        n_processes = self.config.manager.n_processes
        chunks_per_worker = self.config.manager.chunks_per_worker

        logger.info(f"Processing chunks of: {chunk_size}")
        logger.info(f"Final resolution: {final_width}")

        all_cadences = []

        for filename in self.config.data.test_files:
            filepath = self.config.get_test_file_path(filename)

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
            n_cadences_total = raw_data.shape[0]
            n_chunks = (n_cadences_total + chunk_size - 1) // chunk_size

            for chunk_idx in range(n_chunks):
                logger.info(f"Processing {filename}: chunk {chunk_idx + 1}/{n_chunks}")

                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, n_cadences_total)

                # Load chunk into memory
                chunk_data = np.array(raw_data[chunk_start:chunk_end])

                # NOTE: is this access pattern the most efficient (least pickling)? see comments in _single_cadence_wrapper() from data_generation.py for more details
                # NOTE: currently, loading the backgrounds takes WAY more time than processing the backgrounds
                # Prepare arguments for downsampling (just indices, not data - data is in global state)
                n_cadences = chunk_data.shape[0]
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
                    results = chunk_pool.map(
                        _downsample_worker,  # TODO: create separate function that performs downsampling & log-norm simultaneously
                        args_list,
                        chunksize=chunksize,
                    )
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

                    # TODO: create separate function that performs downsampling & log-norm simultaneously
                    results = [_downsample_worker(args) for args in args_list]

                # NOTE: is there a more efficient/elegant way to do this (e.g. with list comprehension/slicing)?
                # Collect valid results (filter out None from invalid cadences)
                for result in results:
                    if result is not None:
                        all_cadences.append(result)

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

        # Stack all_cadences together
        cadence_array = np.array(all_cadences, dtype=np.float32)

        # Clear all_cadences reference
        del all_cadences
        gc.collect()

        logger.info(f"Total cadences loaded: {cadence_array.shape[0]}")
        logger.info(f"Cadence array shape before log norm: {cadence_array.shape}")

        # TODO: create separate function that performs downsampling & log-norm simultaneously
        # Apply log normalization to each cadence
        logger.info("Applying log normalization")
        for i in range(cadence_array.shape[0]):
            cadence_array[i] = log_norm(cadence_array[i])

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

    # TODO: complete find_hits function to perform DC spike removal, bandpass filtering, and energy detection. should take .h5 filepaths as input, then write hits into (n, 6, 16, 4096) shaped .npy files (to data_path? output_path?). have boolean flag overlap_search as arg. if true, write additional cadence snippets +/- 50% in frequency from the actual hit to the .npy file. parametrize overlap amount to InferenceConfig().
    def find_hits(self):
        pass
