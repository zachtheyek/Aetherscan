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
import json
import logging
import os
import re
import signal
from collections import OrderedDict
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory

import h5py

# NOTE: come back to this later (why noqa: F401? what's the difference between h5py & hdf5plugin?)
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


# NOTE: come back to this later (mirrors preprocess_fine.py:72-75 from the reference implementation)
def _remove_dc_spike(
    block_data: np.ndarray, coarse_channel_width: int, n_coarse_channels: int
) -> None:
    """
    Interpolate over the 2-bin DC spike at the center of each coarse channel, in place.

    Args:
        block_data: shape (time_bins, n_coarse_channels * coarse_channel_width)
        coarse_channel_width: fine channels per coarse channel
        n_coarse_channels: number of coarse channels in block_data
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
    Fit a spline bandpass to a time-integrated coarse channel.

    Args:
        integrated_channel: 1-D array of shape (channel_width,), time-integrated
        channel_width: fine channels per coarse channel
        spl_order: spline order (higher = more knots = finer fit)

    Returns:
        1-D bandpass fit of shape (channel_width,)
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


# NOTE: come back to this later
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


# NOTE: come back to this later
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


# NOTE: come back to this later
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

    # TODO: profile on HPC and consider vectorizing. With default config this loop
    # runs ~8190 windows per coarse channel × parallel_chans × num_blocks × 3
    # ON-source files per cadence. scipy.stats.normaltest copies + flattens each
    # window and re-derives skew/kurtosis via separate scipy calls. A stride_tricks
    # view over the cleaned block followed by vectorized scipy.stats.skew /
    # kurtosis (with axis arg) would replace the Python loop with a few large
    # ndarray ops and could be substantially faster end-to-end.
    hits: list[tuple] = []
    for i in range(0, coarse_channel_width - window_size, step_size):
        window = channel[:, i : i + window_size]
        s, p = stats.normaltest(window.flatten())
        if s > stat_threshold:
            abs_idx = block_offset + channel_index * coarse_channel_width + i
            hits.append((abs_idx, float(s), float(p)))

    return hits


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


# NOTE: come back to this later (add sorting functionality to sort rows in csv after grouping, e.g. via timestamp metadata from filenames? edge case where multiple 6-cadence observations of the same target & with the same grouping params, but differed by time, e.g. t=X to t=X+ε, then t=Y to t=Y+σ, for some small numbers ε and σ, and where X and Y are far apart from each other. add a way to distinguish these cases from problematic cases where we actually want to invalidate a cadence with a weird number of grouped observations, e.g. if multiple of 6 and enough of a gap between X and Y, then count as separate cadences?)
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

        # NOTE: come back to this later (does this fail for all if one csv has missing columns, or does it skip the invalid csvs and continue processing the valid ones? is proper downstream checkpointing implemented in the latter case?)
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
    def load_inference_data(self, override_filepaths: list[str] | None = None) -> np.ndarray:
        """
        Load & preprocess data for inference
        Uses parallel processing to load, downsample, and log-normalize the data

        Args:
            override_filepaths: If provided, iterate these absolute paths directly
                instead of resolving config.data.test_files via get_test_file_path.
                Used by find_hits() to chain per-cadence .npy outputs into inference
                without monkey-patching paths.

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

    # NOTE: come back to this later (based on docstring, we're processing cadences sequentially. if so, any way to parallelize?)
    def find_hits(self) -> list[CadenceResult]:
        """
        Convert raw .h5 cadence observations into (n_hits, 6, 16, stamp_width) .npy snippets.

        Driven by CSVs in config.data.inference_files. Each CSV is grouped into
        cadences via group_observations_from_csv() and processed sequentially.
        Within each cadence, energy detection runs on ON-source files (positions
        0, 2, 4 in ABACAD); stamps are extracted from all 6 observations at hit
        frequencies.

        Each cadence produces one .npy file on disk as soon as it's ready
        (periodic checkpointing). On retry, cadences whose output already exists
        are skipped.

        Returns:
            List of CadenceResult, one per successfully processed (or already
            cached) cadence.
        """
        inference_files = self.config.data.inference_files
        if not inference_files:
            logger.warning("find_hits() called with no inference_files configured")
            return []

        # Resolve output directory
        output_dir = self.config.inference.preprocess_output_dir or os.path.join(
            self.config.output_path, "preprocessed"
        )
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Preprocessing output directory: {output_dir}")

        group_by_cols = self.config.inference.cadence_group_by_cols
        h5_path_col = self.config.inference.cadence_h5_path_col
        expected_obs = self.config.inference.cadence_expected_obs

        results: list[CadenceResult] = []

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

            for group in valid_groups:
                npy_filename = self._cadence_npy_filename(csv_stem, group.key)
                npy_path = os.path.join(output_dir, npy_filename)

                if os.path.exists(npy_path):
                    # Resume path: rebuild a minimal CadenceResult from the existing file
                    metadata_path = self._cadence_metadata_path(npy_path)
                    try:
                        existing = np.load(npy_path, mmap_mode="r")
                        n_hits = existing.shape[0]
                        del existing
                    except Exception as e:
                        logger.warning(
                            f"Existing .npy at {npy_path} could not be inspected ({e}); "
                            f"reprocessing cadence"
                        )
                        # Fall through (no continue) to the _process_cadence call
                        # below so a corrupted .npy gets regenerated rather than
                        # silently skipped.
                    else:
                        logger.info(
                            f"Skipping cadence {group.key}: {npy_path} already exists "
                            f"({n_hits} hits)"
                        )
                        results.append(
                            CadenceResult(
                                npy_path=npy_path,
                                h5_paths=group.h5_paths,
                                key=group.key,
                                n_hits=n_hits,
                                metadata_path=metadata_path,
                            )
                        )
                        continue

                try:
                    cadence_result = self._process_cadence(group, npy_path)
                except Exception as e:
                    # Single-cadence failures should not abort the whole CSV;
                    # the retry loop at the inference_command level handles broader recovery
                    logger.error(f"Failed to process cadence {group.key}: {e}")
                    continue

                if cadence_result is not None:
                    results.append(cadence_result)

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
    def _cadence_metadata_path(npy_path: str) -> str:
        """Return the sibling .json path for a cadence's metadata."""
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
        Run energy detection on one cadence and write its stamp .npy.

        Energy detection runs only on ON-source observations (positions 0, 2, 4
        in ABACAD order). Hits define the frequency slices extracted from all 6
        observations.

        Args:
            group: validated CadenceGroup (len(h5_paths) == expected_obs)
            npy_path: target output path (already absolute)

        Returns:
            CadenceResult on success, or None if no hits survived.
        """
        coarse_channel_width = self.config.inference.coarse_channel_width
        parallel_chans = self.config.inference.parallel_coarse_chans
        spl_order = self.config.inference.spline_order
        window_size = self.config.inference.detection_window_size
        step_size = self.config.inference.detection_step_size
        stat_threshold = self.config.inference.stat_threshold
        stamp_width = self.config.inference.stamp_width
        overlap_search = self.config.inference.overlap_search
        overlap_fraction = self.config.inference.overlap_fraction
        time_bins = self.config.data.time_bins
        n_processes = self.config.manager.n_processes

        # Read header / metadata from the first ON-source file
        on_source_paths = [group.h5_paths[i] for i in (0, 2, 4)]
        primary_h5 = on_source_paths[0]

        with h5py.File(primary_h5, "r") as hf:
            header = {k: hf["data"].attrs[k] for k in hf["data"].attrs}
            data_shape = hf["data"].shape

        n_chans = int(header.get("nchans", data_shape[-1]))
        foff = float(header["foff"])
        fch1 = float(header["fch1"])
        n_time_avail = int(data_shape[0])

        # NOTE: come back to this later (why do we only check if n_time_avail < time_bins? what happens if n_time_avail > time_bins?)
        if n_time_avail < time_bins:
            logger.warning(
                f"Cadence {group.key}: primary file has only {n_time_avail} time bins, "
                f"expected >= {time_bins}; skipping"
            )
            return None

        num_blocks = n_chans // (coarse_channel_width * parallel_chans)
        if num_blocks == 0:
            logger.warning(
                f"Cadence {group.key}: n_chans={n_chans} is smaller than one block "
                f"({coarse_channel_width * parallel_chans}); skipping"
            )
            return None

        block_width = coarse_channel_width * parallel_chans
        logger.info(
            f"Cadence {group.key}: n_chans={n_chans}, num_blocks={num_blocks}, "
            f"block_width={block_width}, ON-source files={len(on_source_paths)}"
        )

        # Aggregate hits across all ON-source files
        all_hits: list[tuple] = []  # (abs_idx, stat, p)

        for on_source_idx, on_h5 in enumerate(on_source_paths):
            logger.info(
                f"Cadence {group.key}: running energy detection on ON-source "
                f"{on_source_idx + 1}/{len(on_source_paths)}: {on_h5}"
            )

            for block_num in range(num_blocks):
                block_offset = block_num * block_width
                logger.info(
                    f"  Block {block_num + 1}/{num_blocks} "
                    f"(coarse {block_num * parallel_chans}..{(block_num + 1) * parallel_chans - 1})"
                )

                block_data = self._read_block(
                    on_h5, block_num, parallel_chans, coarse_channel_width, n_processes
                )

                # Slice to first time_bins
                block_data = block_data[:time_bins]

                # In-place DC spike removal
                _remove_dc_spike(block_data, coarse_channel_width, parallel_chans)

                # _drop_side_channels(block_data, side_channel_count, coarse_channel_width)

                cleaned_block = self._remove_block_bandpass(
                    block_data, parallel_chans, coarse_channel_width, spl_order, n_processes
                )

                # Free original block memory; we only need the cleaned residuals downstream
                del block_data

                block_hits = self._threshold_block_hits(
                    cleaned_block,
                    parallel_chans,
                    coarse_channel_width,
                    window_size,
                    step_size,
                    stat_threshold,
                    block_offset,
                    n_processes,
                )

                all_hits.extend(block_hits)

                del cleaned_block
                gc.collect()

        logger.info(f"Cadence {group.key}: {len(all_hits)} raw hits across ON-source files")

        # NOTE: come back to this later (what's the trade-off for doing dedup vs not? e.g. lower storage & compute, but higher FNR or lower DR sensitivity?)
        # Deduplicate: greedy merge of any pair within stamp_width // 2
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

        # Extract stamps for all 6 observations (sequential per file, no pool)
        cadence_stamps = np.zeros(
            (len(stamp_centers), len(group.h5_paths), time_bins, stamp_width), dtype=np.float32
        )

        for obs_idx, obs_h5 in enumerate(group.h5_paths):
            with h5py.File(obs_h5, "r") as hf:
                for stamp_idx, (start, _, _) in enumerate(stamp_centers):
                    end = start + stamp_width
                    cadence_stamps[stamp_idx, obs_idx] = hf["data"][:time_bins, 0, start:end]

        # Write the .npy and the sibling metadata atomically. A process kill
        # mid-write must not leave a corrupt file at the canonical path —
        # find_hits' resume path treats the existence of npy_path as proof of
        # a complete write. We achieve that by writing to a .tmp sibling first
        # then os.replace()-ing it onto the canonical name.
        tmp_npy_path = os.path.splitext(npy_path)[0] + ".tmp.npy"
        np.save(tmp_npy_path, cadence_stamps)
        os.replace(tmp_npy_path, npy_path)

        metadata_path = self._cadence_metadata_path(npy_path)
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
            "stamp_frequencies_mhz": stamp_freqs_mhz,
            "stamp_statistics": stamp_stats,
            "stamp_pvalues": stamp_pvals,
            "overlap_search": overlap_search,
            "overlap_fraction": overlap_fraction if overlap_search else None,
        }
        tmp_metadata_path = metadata_path + ".tmp"
        with open(tmp_metadata_path, "w") as f:
            json.dump(self._to_json_safe(metadata), f, indent=2)
        os.replace(tmp_metadata_path, metadata_path)

        gc.collect()

        logger.info(
            f"Cadence {group.key}: wrote {cadence_stamps.shape[0]} stamps -> "
            f"{npy_path} (metadata: {metadata_path})"
        )

        return CadenceResult(
            npy_path=npy_path,
            h5_paths=group.h5_paths,
            key=group.key,
            n_hits=cadence_stamps.shape[0],
            metadata_path=metadata_path,
        )

    def _read_block(
        self,
        h5_path: str,
        block_num: int,
        parallel_chans: int,
        coarse_channel_width: int,
        n_processes: int,
    ) -> np.ndarray:
        """Read parallel_chans coarse channels in parallel and concatenate."""
        args_list = [
            (h5_path, ch, coarse_channel_width)
            for ch in range(block_num * parallel_chans, (block_num + 1) * parallel_chans)
        ]

        # NOTE: come back to this later (is this correct?)
        # The read worker doesn't need shared memory; create a plain pool
        if n_processes > 1:
            pool = self.manager.create_pool(
                n_processes=min(parallel_chans, n_processes),
                name=f"DataPreproc_read_block_{block_num}",  # NOTE: come back to this later
            )
            try:
                # NOTE: come back to this later (is pool.map correct here?)
                results = pool.map(_read_coarse_channel_worker, args_list)
            finally:
                self.manager.close_pool(pool)
        else:
            results = [_read_coarse_channel_worker(a) for a in args_list]

        return np.concatenate(results, axis=1)

    # NOTE: come back to this later
    def _remove_block_bandpass(
        self,
        block_data: np.ndarray,
        parallel_chans: int,
        coarse_channel_width: int,
        spl_order: int,
        n_processes: int,
    ) -> np.ndarray:
        """Spline-fit + subtract bandpass per coarse channel, return cleaned block."""
        args_list = [(ch, coarse_channel_width, spl_order) for ch in range(parallel_chans)]

        if n_processes > 1:
            shm = self.manager.create_shared_memory(
                size=block_data.nbytes,
                name="DataPreproc_bandpass_block",  # NOTE: come back to this later
            )
            shared = np.ndarray(block_data.shape, dtype=block_data.dtype, buffer=shm.buf)
            shared[:] = block_data[:]

            pool = self.manager.create_pool(
                n_processes=min(parallel_chans, n_processes),
                name="DataPreproc_bandpass_block",  # NOTE: come back to this later
                initializer=_init_worker,
                initargs=(shm.name, block_data.shape, block_data.dtype),
            )
            try:
                results = pool.map(_remove_bandpass_worker, args_list)
            finally:
                del shared
                self.manager.close_shared_memory(shm)
                self.manager.close_pool(pool)
        else:
            global _GLOBAL_CHUNK_DATA
            _GLOBAL_CHUNK_DATA = block_data
            try:
                results = [_remove_bandpass_worker(a) for a in args_list]
            finally:
                # Always clear the global, even if a worker raised, so subsequent
                # calls in the same process don't see stale state
                _GLOBAL_CHUNK_DATA = None

        return np.concatenate(results, axis=1)

    # NOTE: come back to this later
    def _threshold_block_hits(
        self,
        cleaned_block: np.ndarray,
        parallel_chans: int,
        coarse_channel_width: int,
        window_size: int,
        step_size: int,
        stat_threshold: float,
        block_offset: int,
        n_processes: int,
    ) -> list[tuple]:
        """Sliding-window normality test across one cleaned block, return all hits."""
        args_list = [
            (ch, coarse_channel_width, window_size, step_size, stat_threshold, block_offset)
            for ch in range(parallel_chans)
        ]

        if n_processes > 1:
            shm = self.manager.create_shared_memory(
                size=cleaned_block.nbytes,
                name="DataPreproc_threshold_block",  # NOTE: come back to this later
            )
            shared = np.ndarray(cleaned_block.shape, dtype=cleaned_block.dtype, buffer=shm.buf)
            shared[:] = cleaned_block[:]

            pool = self.manager.create_pool(
                n_processes=min(parallel_chans, n_processes),
                name="DataPreproc_threshold_block",  # NOTE: come back to this later
                initializer=_init_worker,
                initargs=(shm.name, cleaned_block.shape, cleaned_block.dtype),
            )
            try:
                results = pool.map(_threshold_hits_worker, args_list)
            finally:
                del shared
                self.manager.close_shared_memory(shm)
                self.manager.close_pool(pool)
        else:
            global _GLOBAL_CHUNK_DATA
            _GLOBAL_CHUNK_DATA = cleaned_block
            try:
                results = [_threshold_hits_worker(a) for a in args_list]
            finally:
                # Always clear the global, even if a worker raised, so subsequent
                # calls in the same process don't see stale state
                _GLOBAL_CHUNK_DATA = None

        flat: list[tuple] = []
        for r in results:
            flat.extend(r)
        return flat

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
