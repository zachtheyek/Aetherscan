# TODO: add logging to background loading & downsampling
"""
Data preprocessing for Aetherscan Pipeline
Handles background data loading & downsampling  # TODO: update once preprocessing.py complete
Uses multiprocessing and shared memory to process data in parallel
"""

from __future__ import annotations

import gc
import logging
import os
import signal
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from skimage.transform import downscale_local_mean

from config import get_config
from logger import init_worker_logging
from manager import get_manager

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
        Worker cleanup is automatic - when the pool terminates, the OS reclaims
        all worker process resources including shared memory file descriptors.
        Cleanup is handled by the main process
    """
    global _GLOBAL_SHM, _GLOBAL_CHUNK_DATA, _GLOBAL_SHAPE, _GLOBAL_DTYPE

    # Ignore SIGINT (Ctrl+C) in workers - let parent handle cleanup coordination
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Restore default SIGTERM behavior so pool.terminate() from parent doesn't hang
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    # Initialize worker logging
    init_worker_logging()

    # Attach to existing shared memory block
    _GLOBAL_SHM = SharedMemory(name=shm_name)

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


class DataPreprocessor:
    """Data preprocessor"""

    def __init__(self):
        """
        Initialize preprocessor
        """
        self.config = get_config()
        if self.config is None:
            raise ValueError("get_config() returned None")

        self.manager = get_manager()

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
    def load_background_data(self) -> np.ndarray:
        """
        Load & downsample background plates using parallel processing

        Returns:
            Array of background plates with shape (n_backgrounds, 6, 16, width_bin_downsampled)
        """
        logger.info(f"Loading background data from {self.config.data_path}")

        # Use config values
        num_target_backgrounds = self.config.data.num_target_backgrounds
        downsample_factor = self.config.data.downsample_factor
        final_width = self.config.data.width_bin // downsample_factor

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
            n_chunks = min(max_chunks, (raw_data.shape[0] + chunk_size - 1) // chunk_size)

            for chunk_idx in range(n_chunks):
                logger.info(f"Processing {filename}: chunk {chunk_idx + 1}/{n_chunks}")

                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, raw_data.shape[0])

                # Load chunk into memory
                chunk_data = np.array(raw_data[chunk_start:chunk_end])

                # NOTE: is this access pattern the most efficient (least pickling)? see comments in _single_cadence_wrapper() from data_generation.py for more details
                # Prepare arguments (just indices, not data - data is in global state)
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
                    # NOTE: does return order matter?
                    # results = chunk_pool.map(_downsample_worker, args_list, chunksize=chunksize)
                    results = chunk_pool.imap_unordered(
                        _downsample_worker, args_list, chunksize=chunksize
                    )

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
            raise ValueError("No background data loaded successfully")

        # Stack all_backgrounds together
        background_array = np.array(all_backgrounds, dtype=np.float32)

        # Clear all_backgrounds reference
        del all_backgrounds
        gc.collect()

        # Sanity check: print descriptive stats
        min_val = np.min(background_array)
        max_val = np.max(background_array)
        mean_val = np.mean(background_array)

        logger.info(f"Total background cadences loaded: {background_array.shape[0]}")
        logger.info(f"Background array shape: {background_array.shape}")
        logger.info(f"Background value range: [{min_val:.6f}, {max_val:.6f}]")
        logger.info(f"Background mean: {mean_val:.6f}")
        logger.info(f"Memory usage: {background_array.nbytes / 1e9:.2f} GB")
        logger.info(f"Background data ready at {background_array.shape[3]} resolution")

        return background_array
