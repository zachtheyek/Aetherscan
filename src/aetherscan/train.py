# NOTE: are we properly clearing memory after db read for plotting functions? are there any db reads that can be grouped together to reduce plotting times?
"""
Training orchestration for Aetherscan Pipeline
Implements full workflow for both beta-VAE & RF classifier,
Supports curriculum learning, adaptive LR, distributed datasets & training,
gradient accumulation, and model checkpointing
"""

from __future__ import annotations

import gc
import glob
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime

import imageio.v3 as iio
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import umap
from tensorflow.keras.initializers import GlorotNormal, HeNormal
from tensorflow.keras.layers import Conv2D, Dense

from aetherscan.config import get_config
from aetherscan.data_generation import DataGenerator
from aetherscan.db import get_db, get_system_metadata
from aetherscan.logger import get_logger
from aetherscan.models import RandomForestModel, Sampling, create_beta_vae_model

logger = logging.getLogger(__name__)


# NOTE: Removing TensorBoard support
# archive_directory() includes (incomplete) functionality for setting up & handling
# TensorBoard directories. unless you're reviving TensorBoard support, simply leave target_dirs as
# None to use the function as "normal"
def archive_directory(base_dir: str, target_dirs: list[str] | None = None, round_num: int = 1):
    """
    Archive and clean up a directory

    Args:
        base_dir: Base directory to archive/clean
        target_dirs: List of subdirectory names to include in archiving (e.g., ['train', 'validation'])
                     If None, only files are considered (directories are ignored)
        round_num: Training round number (1 for fresh run, >1 for resume)
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Check if base_dir is empty
    is_empty = True

    if target_dirs is None:
        # Check for files only (ignore all directories)
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isfile(item_path):
                is_empty = False
                break
    else:
        # Check for files AND target directories
        has_files = False
        has_target_dirs = False

        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isfile(item_path):
                has_files = True
            elif os.path.isdir(item_path) and item in target_dirs:
                has_target_dirs = True

        is_empty = not (has_files or has_target_dirs)

    # If empty, do nothing & return
    if is_empty:
        logger.info(f"Directory {base_dir} is empty, nothing to archive")
        return

    # Otherwise, archive and clean up
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(base_dir, "archive", timestamp)
    os.makedirs(archive_dir, exist_ok=True)

    if round_num == 1:
        # Fresh run: move everything to archive
        logger.info(f"Archiving the following items from {base_dir}:")

        items_moved = 0
        for item in os.listdir(base_dir):
            if item == "archive":  # Don't move the archive directory itself
                continue

            item_path = os.path.join(base_dir, item)

            # Move all files
            if os.path.isfile(item_path):
                shutil.move(item_path, os.path.join(archive_dir, item))
                logger.info(f"  {item_path}")
                items_moved += 1
            # Move target directories if specified
            elif os.path.isdir(item_path) and (target_dirs is not None and item in target_dirs):
                shutil.move(item_path, os.path.join(archive_dir, item))
                logger.info(f"  {item_path}")
                items_moved += 1

                # Replace directory with empty one after moving
                os.makedirs(item_path)

        logger.info(f"Moved {items_moved} items to archive: {archive_dir}")

    else:
        # Resume: copy to archive, then delete files with round >= round_num
        logger.info(f"Backing up the following items from {base_dir}:")

        items_copied = 0
        for item in os.listdir(base_dir):
            if item == "archive":  # Don't copy the archive directory itself
                continue

            item_path = os.path.join(base_dir, item)

            # Copy all files
            if os.path.isfile(item_path):
                shutil.copy2(item_path, os.path.join(archive_dir, item))
                logger.info(f"  {item_path}")
                items_copied += 1
            # Copy target directories if specified
            elif os.path.isdir(item_path) and (target_dirs is not None and item in target_dirs):
                shutil.copytree(item_path, os.path.join(archive_dir, item))
                logger.info(f"  {item_path}")
                items_copied += 1

                # TODO: instead of deleting the whole event files, intelligently parse & filter out future steps, then write filtered events to new files
                # Replace directory with empty one after copying
                shutil.rmtree(item_path)
                os.makedirs(item_path, exist_ok=True)

        logger.info(f"Backed up {items_copied} items to archive: {archive_dir}")

        # Delete files matching "round_X" where X >= round_num
        logger.info(f"Deleting the following items from {base_dir}:")
        pattern = re.compile(r"round_(\d+)")
        deleted_files = []

        for item in os.listdir(base_dir):
            if item == "archive":  # Don't touch the archive directory
                continue

            item_path = os.path.join(base_dir, item)

            # Only process files, not directories
            if os.path.isfile(item_path):
                match = pattern.search(item)
                if match:
                    round_x = int(match.group(1))
                    if round_x >= round_num:
                        os.remove(item_path)
                        deleted_files.append(item)
                        logger.info(f"  {item_path}")

        if deleted_files:
            logger.info(f"Deleted {len(deleted_files)} files with round >= {round_num}")
        else:
            logger.info(f"No files with round >= {round_num} found to delete")


def get_latest_tag(checkpoints_dir: str) -> str:
    """
    Find the latest checkpoint tag from the checkpoints directory

    Returns:
        Latest checkpoint tag (e.g., "round_05")
    """
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Directory doesn't exist: {checkpoints_dir}")

    # Find all encoder files
    encoder_pattern = os.path.join(checkpoints_dir, "vae_encoder_*.keras")
    encoder_files = glob.glob(encoder_pattern)

    if not encoder_files:
        raise FileNotFoundError(f"No encoder files found in {checkpoints_dir}")

    # Extract tags and find complete pairs
    valid_tags = []
    for file in encoder_files:
        basename = os.path.basename(file)
        match = re.search(r"vae_encoder_(.+)\.keras", basename)
        if match:
            tag = match.group(1)
            # Verify decoder exists
            decoder_file = os.path.join(checkpoints_dir, f"vae_decoder_{tag}.keras")
            if os.path.exists(decoder_file):
                valid_tags.append(tag)

    if not valid_tags:
        raise FileNotFoundError(f"No valid model pairs found in {checkpoints_dir}")

    # Sort tags to find the latest
    def sort_key(tag_str):
        # Handle final_vX format with highest priority
        if tag_str.startswith("final_"):
            try:
                final_ver = int(tag_str.split("_v")[1])
                return (0, final_ver)
            except (ValueError, IndexError):
                return (1, tag_str)
        # Handle round_XX format with secondary priority
        elif tag_str.startswith("round_"):
            try:
                round_num = int(tag_str.split("_")[1])
                return (2, round_num)
            except (ValueError, IndexError):
                return (3, tag_str)
        # Handle timestamp format YYYYMMDD_HHMMSS tertiary priority
        elif re.match(r"\d{8}_\d{6}", tag_str):
            try:
                timestamp = datetime.strptime(tag_str, "%Y%m%d_%H%M%S")
                return (4, timestamp)
            except ValueError:
                return (5, tag_str)
        # Handle test_vX format with lowest priority
        if tag_str.startswith("test_"):
            try:
                test_ver = int(tag_str.split("_v")[1])
                return (6, test_ver)
            except (ValueError, IndexError):
                return (7, tag_str)
        # Fallback for all other formats
        else:
            return (99, tag_str)

    # Filter for the highest priority group
    priorities = [sort_key(t)[0] for t in valid_tags]
    highest_priority = min(priorities)  # smaller = higher priority

    if highest_priority == 99:
        raise FileNotFoundError(
            "No valid model tags found (e.g. final_vX, round_XX, YYYYMMDD_HHMMSS, test_vX)"
        )

    filtered_tags = [t for t in valid_tags if sort_key(t)[0] == highest_priority]

    # Select the latest tag within highest priority group
    filtered_tags.sort(key=sort_key)
    tag = filtered_tags[-1]  # Get the latest
    return tag


def compute_expected_std(layer):
    """Compute expected std based on initializer."""
    weights = layer.get_weights()
    if not weights:
        return None

    kernel = weights[0]

    if isinstance(layer.kernel_initializer, HeNormal):
        # HeNormal: std = sqrt(2 / fan_in)
        fan_in = np.prod(kernel.shape[:-1])
        expected_std = np.sqrt(2.0 / fan_in)
    elif isinstance(layer.kernel_initializer, GlorotNormal):
        # GlorotNormal: std = sqrt(2 / (fan_in + fan_out))
        fan_in = np.prod(kernel.shape[:-1])
        fan_out = kernel.shape[-1]
        expected_std = np.sqrt(2.0 / (fan_in + fan_out))
    else:
        return None

    return expected_std


def check_encoder_trained(encoder, threshold=0.2):
    """
    Checks all Conv2D and Dense layers in encoder for deviation from initializer.
    Returns True if at least one layer shows substantial deviation (i.e. the encoder was likely trained).
    """
    trained_layers = []
    for layer in encoder.layers:
        if isinstance(layer, Conv2D | Dense):
            weights = layer.get_weights()
            if not weights:
                continue  # skip layers without weights

            kernel = weights[0]
            actual_std = np.std(kernel)
            expected_std = compute_expected_std(layer)
            if expected_std is None:
                continue  # skip layers with no expected std

            relative_dev = abs(actual_std - expected_std) / expected_std

            logger.info(
                f"{layer.name}: actual std={actual_std:.5f}, expected std={expected_std:.5f}, deviation={relative_dev:.2%}"
            )

            if relative_dev > threshold:
                trained_layers.append(layer.name)

    if trained_layers:
        logger.info("Encoder appears trained (substantial deviation detected in layers):")
        logger.info(f"{trained_layers}")
        return True
    else:
        logger.info("Encoder appears untrained (all layers close to initializer).")
        return False


# Create data holder objects, to be paired with data generators, for TF's distributed datasets
# Allows for explicit dereferencing of large arrays using holder.clear(), which lets
# Python's garbage collector free up memory on-demand
# Note, holder.clear() is only useful at the end of an epoch, once indices have been exhausted,
# since the data generators' local caches maintain references to the data until then
# This is not an issue in our current implementation, where we only clear & reset resources at the
# end of a round. However, if you require early exit behavior, you may want to remove the _lock and
# use explicit _cleared() checks instead, which negates the need for local caches (see commit hash
# 2a404a4). The trade-off being that you're at risk of race conditions if multiple threads attempt
# to access/clear the holder simultaneously. While this is not the case in our current
# implementation, we opted for a more defensive approach rather than accomodating future design
# patterns. As well, the data should not be modified once the holder has been initialized to
# prevent corrupted state in the holder
# Note, there's a potential deadlock issue with holder lock contention
# Since the generators acquire locks at the start of every loop iteration, if TF's prefetch threads
# (.prefetch(tf.data.AUTOTUNE)) are blocked waiting on this lock while the main thread is trying to
# call holder.clear() (which also needs the lock), there could be contention.
# This has not been an issue so far, but if you encounter this in the future, pls update this
# comment with your findings
class TrainDataHolder:
    def __init__(self, concat, true, false):
        self._cleared = False
        self._lock = threading.Lock()
        self.concat = concat
        self.true = true
        self.false = false

    def clear(self):
        with self._lock:
            if self._cleared:
                return
            self._cleared = True
            self.concat = None
            self.true = None
            self.false = None


class InfDataHolder:
    def __init__(self, true, false):
        self._cleared = False
        self._lock = threading.Lock()
        self.true = true
        self.false = false

    def clear(self):
        with self._lock:
            if self._cleared:
                return
            self._cleared = True
            self.true = None
            self.false = None


class VizDataHolder:
    def __init__(self, concat):
        self._cleared = False
        self._lock = threading.Lock()
        self.concat = concat

    def clear(self):
        with self._lock:
            if self._cleared:
                return
            self._cleared = True
            self.concat = None


def prepare_distributed_train_dataset(
    data: dict,
    train_val_split: float,
    per_replica_batch_size: int,
    effective_batch_size: int,
    per_replica_val_batch_size: int,
    num_replicas: int,
    strategy: tf.distribute.Strategy,
    shuffle: bool = True,
) -> dict:
    """
    Prepare distributed datasets for training & validation
    Yields datasets with signature ((concat, true, false), concat)

    Args:
        data: Dictionary with keys 'concatenated', 'true', 'false' (numpy arrays)
        train_val_split: Split data into train/val sets
        per_replica_batch_size: Batch size per replica for training
        effective_batch_size: Effective batch size across all replicas for training
        per_replica_val_batch_size: Batch size per replica for validation
        num_replicas: Number of replicas in strategy
        strategy: TensorFlow distribution strategy
        shuffle: Whether to shuffle training data

    Returns: {train_dataset, val_dataset, n_train_trimmed, n_val_trimmed, train_steps,
              accumulation_steps, val_steps, _train_holder, val_indices}
             Train/val distributed datasets, number of samples in each, number of steps for each
              (including accumulation sub-steps), shared TrainDataHolder reference (both generators
              read from the original arrays via index subsets — no copies), and the stratified
              validation indices into the original data arrays
    """
    global_train_batch_size = per_replica_batch_size * num_replicas
    global_val_batch_size = per_replica_val_batch_size * num_replicas

    # Stratified train/val split to ensure both sets contain proportional representation
    # of all 4 signal types (false_no_signal, false_with_rfi, true_only_eti, true_eti_rfi).
    # This is necessary because generate_triplet_batch() arranges labels sequentially within
    # chunks, so a naive positional split would over-represent later signal types in val.
    labels = data["labels"]
    unique_labels = np.unique(labels)

    train_indices = []
    val_indices = []

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        n_label_train = int(len(label_indices) * train_val_split)
        train_indices.append(label_indices[:n_label_train])
        val_indices.append(label_indices[n_label_train:])

    train_indices = np.concatenate(train_indices)
    val_indices = np.concatenate(val_indices)

    n_train = len(train_indices)
    n_val = len(val_indices)

    # Trim datasets to fit train/val batch sizes (prevents uneven batches on final step)
    # Note, n_train should already be divisible by effective_batch_size and
    # per_replica_batch_size * num_replicas
    # As well, n_val should also already be divisible by per_replica_val_batch_size * num_replicas
    # Trimming here is just a defensive measure to doubly ensure divisibility before creating &
    # distributing our datasets
    # Alternatively, we could also pad the data instead of trimming
    n_train_trimmed = (n_train // effective_batch_size) * effective_batch_size
    n_val_trimmed = (n_val // global_val_batch_size) * global_val_batch_size

    # Randomly subsample to trimmed size (avoids positional bias from slicing the tail)
    if n_train_trimmed < n_train:
        train_indices = np.random.choice(train_indices, size=n_train_trimmed, replace=False)
    if n_val_trimmed < n_val:
        val_indices = np.random.choice(val_indices, size=n_val_trimmed, replace=False)

    logger.info(f"Data alignment: Train {n_train}→{n_train_trimmed}, Val {n_val}→{n_val_trimmed}")

    # Share the original arrays between train and val generators via a single data holder.
    # The stratified split requires non-contiguous indices, which would force numpy to create
    # full copies via fancy indexing (~2x peak memory). Instead, both generators read from the
    # same original arrays using their respective index subsets — zero extra copies.
    train_holder = TrainDataHolder(data["concatenated"], data["true"], data["false"])

    # Create generator functions for memory-efficient data loading
    def train_generator():
        while True:  # Make generators infinite to reset state between epochs
            # Acquire lock to check cleared status and capture data references
            # Local references keep data alive even if clear() is called mid-epoch
            with train_holder._lock:
                if train_holder._cleared:
                    return  # Exit if data already cleared
                # Cache references while holding lock
                concat = train_holder.concat
                true = train_holder.true
                false = train_holder.false

            # Work with local references (safe from clearing, no per-sample lock needed)
            # Copy train_indices because np.random.shuffle mutates in-place
            indices = train_indices.copy()
            if shuffle:
                # Perform global shuffle on each epoch so each pass through the data is unique
                np.random.shuffle(indices)
            for idx in indices:
                yield (concat[idx], true[idx], false[idx]), concat[idx]

            # Remove cache references to ensure garbage collection in future
            del concat, true, false

    def val_generator():
        while True:  # Make generators infinite to reset state between epochs
            # Acquire lock to check cleared status and capture data references
            # Local references keep data alive even if clear() is called mid-epoch
            with train_holder._lock:
                if train_holder._cleared:
                    return  # Exit if data already cleared
                # Cache references while holding lock
                concat = train_holder.concat
                true = train_holder.true
                false = train_holder.false

            # Maintain order on each epoch since shuffling provides no benefits (no gradients
            # are calculated during validation)
            for idx in val_indices:
                yield (concat[idx], true[idx], false[idx]), concat[idx]

            # Remove cache references to ensure garbage collection in future
            del concat, true, false

    # Determine dataset output signature
    sample_shape = data["concatenated"].shape[1:]
    output_signature = (
        (
            tf.TensorSpec(shape=sample_shape, dtype=tf.float32),
            tf.TensorSpec(shape=sample_shape, dtype=tf.float32),
            tf.TensorSpec(shape=sample_shape, dtype=tf.float32),
        ),
        tf.TensorSpec(shape=sample_shape, dtype=tf.float32),
    )

    # Create datasets using generators to reduce GPU memory pressure
    # Data is kept on CPU & transferred to GPU in batches on-demand
    # Note that the datasets yield data in batches before being sharded (distributed) across replicas
    # Hence, we use global batch sizes here to ensure per replica batch sizes match expectations
    logger.info(
        f"Creating infinite datasets from generators with global batch size - "
        f"Train: {global_train_batch_size}, Val: {global_val_batch_size}"
    )

    train_dataset = (
        tf.data.Dataset.from_generator(train_generator, output_signature=output_signature)
        .batch(global_train_batch_size, drop_remainder=True)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_generator(val_generator, output_signature=output_signature)
        .batch(global_val_batch_size, drop_remainder=True)
        # NOTE: do we need repeat for val dataset? run test without repeat & see if anything breaks?
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    # Distribute datasets across GPUs
    logger.info(f"Distributing datasets across {num_replicas} GPUs")

    train_dataset_distributed = strategy.experimental_distribute_dataset(train_dataset)
    val_dataset_distributed = strategy.experimental_distribute_dataset(val_dataset)

    # Calculate steps
    train_steps = n_train_trimmed // effective_batch_size
    accumulation_steps = effective_batch_size // (global_train_batch_size)
    val_steps = n_val_trimmed // global_val_batch_size

    # Sanity check: verify step sizes are valid before returning
    if train_steps < 1:
        raise ValueError(
            f"train_steps < 1: n_train_trimmed ({n_train_trimmed}) must be >= effective_batch_size ({effective_batch_size})"
        )
    if accumulation_steps < 1:
        raise ValueError(
            f"accumulation_steps < 1: effective_batch_size ({effective_batch_size}) must be >= per_replica_batch_size * num_replicas ({per_replica_batch_size} * {num_replicas})"
        )
    if val_steps < 1:
        raise ValueError(
            f"val_steps < 1: n_val_trimmed ({n_val_trimmed}) must be >= per_replica_val_batch_size * num_replicas ({per_replica_val_batch_size} * {num_replicas})"
        )

    return {
        "train_dataset": train_dataset_distributed,
        "val_dataset": val_dataset_distributed,
        "n_train_trimmed": n_train_trimmed,
        "n_val_trimmed": n_val_trimmed,
        "train_steps": train_steps,
        "accumulation_steps": accumulation_steps,
        "val_steps": val_steps,
        "_train_holder": train_holder,
        "val_indices": val_indices,  # For train_round() -> _prepare_latent_viz_batch()
    }


def prepare_distributed_inf_dataset(
    data: dict,
    per_replica_inf_batch_size: int,
    num_replicas: int,
    strategy: tf.distribute.Strategy,
) -> dict:
    """
    Prepare distributed datasets for inference
    Yields datasets with signature (true, false)

    Note, this function is meant for RF training
    It is different from aetherscan.inference.prepare_distributed_inf_dataset(),
    since we assume signal classes are known ahead of time

    Args:
        data: Dictionary with keys 'concatenated', 'true', 'false' (numpy arrays)
        per_replica_inf_batch_size: Batch size per replica for inference
        num_replicas: Number of replicas in strategy
        strategy: TensorFlow distribution strategy

    Returns: {inf_dataset, n_inf_trimmed, inf_steps, _inf_holder}
             Inference distributed dataset, number of samples, number of steps,
              and InfDataHolder reference
    """
    global_inf_batch_size = per_replica_inf_batch_size * num_replicas
    n_samples = data["true"].shape[0]

    # NOTE: does trimming/divisibility matter for inference?
    # Trim datasets to fit batch sizes (prevents uneven batches on final step)
    # Note, n_samples should already be divisible by effective_batch_size
    # Trimming here is just a defensive measure to doubly ensure divisibility before creating &
    # distributing our datasets
    # Alternatively, we could also pad the data instead of trimming
    n_inf_trimmed = (n_samples // global_inf_batch_size) * global_inf_batch_size

    logger.info(f"Data alignment: Inf {n_samples}→{n_inf_trimmed}")

    # Randomly subsample to trimmed size (avoids positional bias from slicing the tail)
    if n_inf_trimmed < n_samples:
        indices = np.random.choice(n_samples, size=n_inf_trimmed, replace=False)
        inf_true = data["true"][indices]
        inf_false = data["false"][indices]
    else:
        inf_true = data["true"][:n_inf_trimmed]
        inf_false = data["false"][:n_inf_trimmed]

    inf_holder = InfDataHolder(inf_true, inf_false)

    # Create generator function for memory-efficient data loading
    def inf_generator():
        while True:  # Make generator infinite to reset state between passes
            # Acquire lock to check cleared status and capture data references
            # Local references keep data alive even if clear() is called mid-epoch
            with inf_holder._lock:
                if inf_holder._cleared:
                    return  # Exit if data already cleared
                # Cache references while holding lock
                true = inf_holder.true
                false = inf_holder.false

            # Maintain order on each epoch since shuffling provides no benefits (no gradients
            # are calculated during inference)
            for idx in range(len(true)):
                yield true[idx], false[idx]

            # Remove cache references for future garbage collection
            del true, false

    # Determine dataset output signature
    sample_shape = inf_true.shape[1:]
    output_signature = (
        tf.TensorSpec(shape=sample_shape, dtype=tf.float32),
        tf.TensorSpec(shape=sample_shape, dtype=tf.float32),
    )

    # Create dataset using generator to reduce GPU memory pressure
    # Data is kept on CPU & transferred to GPU in batches on-demand
    # Note that the dataset yields data in batches before being sharded (distributed) across replicas
    # Hence, we use global batch sizes here to ensure per replica batch sizes match expectations
    logger.info(
        f"Creating infinite dataset from generator with global batch size: {global_inf_batch_size}"
    )

    inf_dataset = (
        tf.data.Dataset.from_generator(inf_generator, output_signature=output_signature)
        .batch(global_inf_batch_size, drop_remainder=True)
        # NOTE: do we need repeat for inf dataset? run test without repeat & see if anything breaks?
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    # Distribute dataset across GPUs
    logger.info(f"Distributing dataset across {num_replicas} GPUs")

    inf_dataset_distributed = strategy.experimental_distribute_dataset(inf_dataset)

    # Calculate steps
    inf_steps = n_inf_trimmed // global_inf_batch_size

    # Sanity check: verify step sizes are valid before returning
    if inf_steps < 1:
        raise ValueError(
            f"inf_steps < 1: n_inf_trimmed ({n_inf_trimmed}) must be >= per_replica_inf_batch_size * num_replicas ({per_replica_inf_batch_size} * {num_replicas})"
        )

    return {
        "inf_dataset": inf_dataset_distributed,
        "n_inf_trimmed": n_inf_trimmed,
        "inf_steps": inf_steps,
        "_inf_holder": inf_holder,
    }


def prepare_distributed_viz_dataset(
    concat_data: np.ndarray,
    per_replica_inf_batch_size: int,
    num_replicas: int,
    strategy: tf.distribute.Strategy,
) -> dict:
    """
    Prepare distributed datasets for latent space visualization
    Yields datasets with signature (concat)

    Args:
        concat_data: Concatenated cadences array, shape (n_cadences, 6, 16, width_bin)
        per_replica_inf_batch_size: Batch size per replica for inference
        num_replicas: Number of replicas in strategy
        strategy: TensorFlow distribution strategy

    Returns: {viz_dataset, n_padded, n_samples, viz_steps, _viz_holder}
             Viz distributed dataset, number of padded samples, number of real/unpadded samples,
              number of steps, and VizDataHolder reference
    """
    global_viz_batch_size = per_replica_inf_batch_size * num_replicas
    n_samples = concat_data.shape[0]

    # NOTE: does padding/divisibility matter for inference?
    # Pad datasets to fit batch sizes (prevents uneven batches on final step)
    # Note, n_samples should already be divisible by effective_batch_size
    # Padding here is just a defensive measure to doubly ensure divisibility before creating &
    # distributing our datasets
    # Alternatively, we could also trim the data instead of padding
    n_padded = int(np.ceil(n_samples / global_viz_batch_size)) * global_viz_batch_size

    if n_padded > n_samples:
        pad_count = n_padded - n_samples
        pad_indices = np.random.choice(n_samples, size=pad_count, replace=True)
        padded_data = np.concatenate([concat_data, concat_data[pad_indices]], axis=0)
        logger.info(f"Data alignment: Viz {n_samples}→{n_padded} (padded {pad_count})")
    else:
        padded_data = concat_data
        logger.info(f"Data alignment: Viz {n_samples} (no padding needed)")

    viz_holder = VizDataHolder(padded_data)

    # Create generator function for memory-efficient data loading
    def viz_generator():
        while True:  # Make generator infinite to reset state between passes
            # Acquire lock to check cleared status and capture data references
            # Local references keep data alive even if clear() is called mid-epoch
            with viz_holder._lock:
                if viz_holder._cleared:
                    return  # Exit if data already cleared
                # Cache references while holding lock
                concat = viz_holder.concat

            # WARN: DO NOT SHUFFLE viz_generator(), OR ELSE YOU'LL BREAK plot_latent_space_gif()
            # Maintain order on each epoch since shuffling provides no benefits (no gradients
            # are calculated during inference)
            for idx in range(len(concat)):
                yield concat[idx]

            # Remove cache references for future garbage collection
            del concat

    # Determine dataset output signature
    sample_shape = padded_data.shape[1:]
    output_signature = tf.TensorSpec(shape=sample_shape, dtype=tf.float32)

    # Create dataset using generator to reduce GPU memory pressure
    # Data is kept on CPU & transferred to GPU in batches on-demand
    # Note that the dataset yields data in batches before being sharded (distributed) across replicas
    # Hence, we use global batch sizes here to ensure per replica batch sizes match expectations
    logger.info(
        f"Creating infinite dataset from generator with global batch size: {global_viz_batch_size}"
    )

    viz_dataset = (
        tf.data.Dataset.from_generator(viz_generator, output_signature=output_signature)
        .batch(global_viz_batch_size, drop_remainder=True)
        # NOTE: do we need repeat for viz dataset? run test without repeat & see if anything breaks?
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    # Distribute dataset across GPUs
    logger.info(f"Distributing dataset across {num_replicas} GPUs")

    viz_dataset_distributed = strategy.experimental_distribute_dataset(viz_dataset)

    # Calculate steps
    viz_steps = n_padded // global_viz_batch_size

    # Sanity check: verify step sizes are valid before returning
    if viz_steps < 1:
        raise ValueError(
            f"viz_steps < 1: n_padded ({n_padded}) must be >= "
            f"per_replica_inf_batch_size * num_replicas ({per_replica_inf_batch_size} * {num_replicas})"
        )

    return {
        "viz_dataset": viz_dataset_distributed,
        "n_padded": n_padded,
        "n_samples": n_samples,
        "viz_steps": viz_steps,
        "_viz_holder": viz_holder,
    }


class TrainingPipeline:
    """Training pipeline"""

    def __init__(self, background_data, strategy: tf.distribute.Strategy = None):
        """
        Initialize training pipeline

        Args:
            background_data: Array of background observations
            strategy: TensorFlow distribution strategy
        """
        self.config = get_config()
        if self.config is None:
            raise ValueError("get_config() returned None")

        self.db = get_db()
        if self.db is None:
            raise ValueError("get_db() returned None")

        # Initialize data generator
        self.data_generator = DataGenerator(background_data)

        # Set distributed strategy
        self.strategy = strategy or tf.distribute.get_strategy()

        # Create VAE model & optimizer inside distributed context
        with self.strategy.scope():
            self.vae = create_beta_vae_model()
            self._build_optimizer()

        # NOTE: this approach doesn't play well with fault tolerance. rethink later
        # Latent viz data (prepared once on first round, persisted across rounds)
        self._latent_viz_batch = None
        self._latent_viz_labels = None
        self._latent_viz_dataset = None
        self._latent_viz_n_padded = None
        self._latent_viz_n_samples = None
        self._latent_viz_steps = None
        self._latent_viz_holder = None
        self._viz_encode_fn = None

        # Initialize RF model as None
        self.rf_model = None

        try:
            # Load models from checkpoints if provided
            if self.config.checkpoint.load_tag or self.config.checkpoint.load_dir:
                logger.info("Resuming from checkpoint")
                self.load_models(
                    tag=self.config.checkpoint.load_tag, dir=self.config.checkpoint.load_dir
                )

        except Exception as e:
            logger.error(f"Error loading models from checkpoint: {e}")
            logger.info("Resetting config.checkpoint to start training from scratch")
            self.config.checkpoint.load_dir = None
            self.config.checkpoint.load_tag = None
            self.config.checkpoint.start_round = 1
            raise  # Re-raise to propagate error

        # NOTE: similar to _setup_directories() & archive_directory(), perhaps we need a flag that gets toggled when fault tolerance is triggered, s.t. future reads from the db know to ignore the flagged rows as "archived" from a previous failed training run?
        finally:
            # Regardless whether checkpoints were loaded or not, we finish the directory setup
            # since fault tolerance expects a clean directory structure
            self._setup_directories()

            # COMMENTED OUT: Removing TensorBoard support
            # Setup TensorBoard logging
            # self._setup_tensorboard_logging()

    # COMMENTED OUT: Removing TensorBoard support
    # def __del__(self):
    #     """Cleanup TensorBoard writers and data generator"""
    #     if hasattr(self, "train_writer"):
    #         self.train_writer.close()
    #     if hasattr(self, "val_writer"):
    #         self.val_writer.close()

    def _build_optimizer(self):
        """
        Build optimizer state by performing a dummy forward/backward pass
        Must be called within strategy scope & before training to initialize optimizer variables
        """
        # Create a dummy batch to trigger optimizer build
        dummy_batch_size = 1
        num_observations = self.config.data.num_observations
        time_bins = self.config.data.time_bins
        width_bin = self.config.data.width_bin // self.config.data.downsample_factor

        dummy_data = tf.zeros(
            (dummy_batch_size, num_observations, time_bins, width_bin), dtype=tf.float32
        )

        # Perform one forward pass to build the model
        _ = self.vae(dummy_data, training=False)

        # Create dummy gradients to build optimizer state
        dummy_grads = [tf.zeros_like(var) for var in self.vae.trainable_variables]

        # Apply dummy gradients to build optimizer variables
        @tf.function
        def apply_dummy_grads():
            self.vae.optimizer.apply_gradients(
                zip(dummy_grads, self.vae.trainable_variables, strict=False)
            )

        self.strategy.run(apply_dummy_grads)

        logger.info("Optimizer built successfully within strategy scope")

    def _setup_directories(self):
        """Create necessary directories"""
        logger.info("Setting up directories")

        start_round = self.config.checkpoint.start_round

        model_checkpoints_dir = os.path.join(self.config.model_path, "checkpoints")
        archive_directory(model_checkpoints_dir, target_dirs=None, round_num=start_round)

        plot_checkpoints_dir = os.path.join(self.config.output_path, "plots", "checkpoints")
        archive_directory(plot_checkpoints_dir, target_dirs=None, round_num=start_round)

        logger.info("Setup directories complete")

    # COMMENTED OUT: Removing TensorBoard support
    # def _setup_tensorboard_logging(self):
    #     """Setup TensorBoard logging"""
    #     logger.info("Setting up TensorBoard logging")
    #
    #     start_round = self.config.checkpoint.start_round
    #
    #     logs_dir = os.path.join(self.config.output_path, "logs")
    #     archive_directory(logs_dir, target_dirs=["train", "validation"], round_num=start_round)
    #
    #     self.global_step = (start_round - 1) * self.config.training.epochs_per_round
    #     if start_round == 1:
    #         logger.info("Starting fresh TensorBoard logs")
    #     else:
    #         logger.info(
    #             f"Resuming TensorBoard logs from step {self.global_step} (round {start_round})"
    #         )
    #
    #     # Create TensorBoard writers
    #     train_log_dir = os.path.join(logs_dir, "train")
    #     val_log_dir = os.path.join(logs_dir, "validation")
    #
    #     self.train_writer = tf.summary.create_file_writer(train_log_dir)
    #     self.val_writer = tf.summary.create_file_writer(val_log_dir)
    #
    #     logger.info(f"TensorBoard logs directory: {logs_dir}")
    #     logger.info(f"Initial global_step: {self.global_step}")

    def train_beta_vae(self):
        """
        Train beta-VAE with curriculum learning & adaptive LR setup
        """
        n_rounds = self.config.training.num_training_rounds
        epochs = self.config.training.epochs_per_round
        start_round = self.config.checkpoint.start_round

        if start_round > n_rounds:
            return  # Return early if beta-VAE already trained (can occur from fault tolerance)
        elif start_round > 1:
            logger.info(f"Resuming training from round {start_round}/{n_rounds}")
        else:
            logger.info(f"Starting training for {n_rounds} rounds")

        # NOTE: this approach doesn't play well with fault tolerance. rethink later
        self.start_time = time.time()

        try:
            for round_idx in range(start_round - 1, n_rounds):
                snr_base, snr_range = self._calculate_curriculum_snr(round_idx)

                logger.info(f"{'=' * 50}")
                logger.info(f"ROUND {round_idx + 1}/{n_rounds}")
                logger.info(f"SNR range: {snr_base}-{snr_base + snr_range}")
                logger.info(f"{'=' * 50}")

                # Reset learning rate & adaptive state before new curriculum stage
                original_lr = self.config.training.base_learning_rate
                current_lr = self.vae.optimizer.learning_rate.numpy()
                self.vae.optimizer.learning_rate.assign(original_lr)

                if hasattr(self, "best_val_loss"):
                    delattr(self, "best_val_loss")
                if hasattr(self, "patience_counter"):
                    delattr(self, "patience_counter")

                logger.info(f"Curriculum LR reset: {current_lr:.2e} → {original_lr:.2e}")

                self.train_round(
                    round_idx=round_idx, epochs=epochs, snr_base=snr_base, snr_range=snr_range
                )
        finally:
            # NOTE: this approach doesn't play well with fault tolerance. rethink later
            # Free the latent viz batch once all rounds are complete (or on failure)
            del self._latent_viz_batch, self._latent_viz_labels
            self._latent_viz_batch = None
            self._latent_viz_labels = None
            gc.collect()

    def train_round(self, round_idx: int, epochs: int, snr_base: int, snr_range: int):
        """
        Perform a single training round
        """
        logger.info(
            f"Training round {round_idx + 1} - Epochs: {epochs}, SNR: {snr_base}-{snr_base + snr_range}"
        )

        # Generate training data
        train_data = self.data_generator.generate_triplet_batch(
            self.config.training.num_samples_beta_vae, snr_base, snr_range, round_idx + 1
        )

        # Extract labels before distributing (prepare_distributed_train_dataset keeps the
        # original arrays alive via a shared train_holder — no copies — so we can free the
        # dict reference immediately after)
        train_labels = train_data.get("labels")

        # Distribute training data
        data = prepare_distributed_train_dataset(
            data=train_data,
            train_val_split=self.config.training.train_val_split,
            per_replica_batch_size=self.config.training.per_replica_batch_size,
            effective_batch_size=self.config.training.effective_batch_size,
            per_replica_val_batch_size=self.config.training.per_replica_val_batch_size,
            num_replicas=self.strategy.num_replicas_in_sync,
            strategy=self.strategy,
            shuffle=True,
        )

        # Free the dict shell (original arrays stay alive via the shared train_holder)
        del train_data
        gc.collect()

        # Prepare latent viz batch (if it's the first round) using the validation partition.
        # Using held-out data ensures the latent space visualization captures generalization, while
        # persisting the same data across rounds eliminates the effects of distribution shift (from
        # the curriculum schedule)
        if self._latent_viz_batch is None:
            if train_labels is not None:
                # Use the data holder's original arrays directly with val_indices to avoid creating
                # an intermediate copy of the entire validation partition
                self._prepare_latent_viz_batch(
                    concat_data=data["_train_holder"].concat,
                    labels=train_labels,
                    candidate_indices=data["val_indices"],
                )
        # On subsequent rounds, the latent viz batch is persisted,
        # but the distributed dataset needs to be rebuilt
        elif self._latent_viz_batch is not None and self._latent_viz_dataset is None:
            self._build_latent_viz_dataset()

        del train_labels
        gc.collect()

        train_dataset = data["train_dataset"]
        val_dataset = data["val_dataset"]
        n_train_trimmed = data["n_train_trimmed"]
        n_val_trimmed = data["n_val_trimmed"]
        steps_per_epoch = data["train_steps"]
        accumulation_steps = data["accumulation_steps"]
        val_steps = data["val_steps"]
        train_holder = data["_train_holder"]

        del data
        gc.collect()

        logger.info(
            f"Initializing training loop with {steps_per_epoch} train steps, {val_steps} val steps"
        )
        logger.info(f"Gradients accumulated every {accumulation_steps} sub-steps")

        try:
            for epoch in range(epochs):
                # Training
                epoch_losses, epoch_gradient_norms, train_duration = self._train_epoch(
                    round_idx,
                    epoch,
                    snr_base,
                    snr_range,
                    train_dataset,
                    steps_per_epoch,
                    accumulation_steps,
                    time.time(),
                )

                # Validation
                val_losses, val_duration = self._validate_epoch(val_dataset, val_steps, time.time())

                # Queue db writes (non-blocking) & log results
                current_time = time.time()

                if self.db is None:
                    raise RuntimeError(
                        "No database instance detected - cannot generate loss curves plot"
                    )

                # Training losses
                for stat_name, key in [
                    ("total_loss", "total"),
                    ("reconstruction_loss", "reconstruction"),
                    ("kl_loss", "kl"),
                    ("true_loss", "true"),
                    ("false_loss", "false"),
                ]:
                    self.db.write_training_stat(
                        model_name="beta_vae",
                        stat_name=stat_name,
                        value=float(epoch_losses[key]),
                        round_number=round_idx + 1,
                        epoch_number=epoch + 1,
                        tag=self.config.checkpoint.save_tag,
                        timestamp=current_time,
                    )

                # Validation losses
                for stat_name, key in [
                    ("val_total_loss", "total"),
                    ("val_reconstruction_loss", "reconstruction"),
                    ("val_kl_loss", "kl"),
                    ("val_true_loss", "true"),
                    ("val_false_loss", "false"),
                ]:
                    self.db.write_training_stat(
                        model_name="beta_vae",
                        stat_name=stat_name,
                        value=float(val_losses[key]),
                        round_number=round_idx + 1,
                        epoch_number=epoch + 1,
                        tag=self.config.checkpoint.save_tag,
                        timestamp=current_time,
                    )

                # Gradient norm/clipping statistics
                gradient_norm_mean = np.mean(epoch_gradient_norms)
                gradient_norm_max = np.max(epoch_gradient_norms)
                gradient_norm_std = np.std(epoch_gradient_norms)
                clipping_rate = np.sum(np.array(epoch_gradient_norms) > 1.0) / steps_per_epoch

                for stat_name, stat_value in [
                    ("gradient_norm_mean", gradient_norm_mean),
                    ("gradient_norm_max", gradient_norm_max),
                    ("gradient_norm_std", gradient_norm_std),
                    ("clipping_rate", clipping_rate),
                ]:
                    self.db.write_training_stat(
                        model_name="beta_vae",
                        stat_name=stat_name,
                        value=float(stat_value),
                        round_number=round_idx + 1,
                        epoch_number=epoch + 1,
                        tag=self.config.checkpoint.save_tag,
                        timestamp=current_time,
                    )

                # Learning rate
                current_lr = float(self.vae.optimizer.learning_rate.numpy())
                self.db.write_training_stat(
                    model_name="beta_vae",
                    stat_name="learning_rate",
                    value=current_lr,
                    round_number=round_idx + 1,
                    epoch_number=epoch + 1,
                    tag=self.config.checkpoint.save_tag,
                    timestamp=current_time,
                )

                # Misc stats
                for stat_name, stat_value in [
                    ("train_duration", train_duration),
                    ("val_duration", val_duration),
                    ("snr_range_floor", snr_base),
                    ("snr_range_ceil", snr_base + snr_range),
                    ("num_steps", steps_per_epoch),
                    ("num_sub_steps", accumulation_steps),
                ]:
                    self.db.write_training_stat(
                        model_name="beta_vae",
                        stat_name=stat_name,
                        value=stat_value,
                        round_number=round_idx + 1,
                        epoch_number=epoch + 1,
                        tag=self.config.checkpoint.save_tag,
                        timestamp=current_time,
                    )

                # COMMENTED OUT: Removing TensorBoard support
                # TensorBoard logging
                # with self.train_writer.as_default():
                #     tf.summary.scalar("total_loss", epoch_losses["total"], step=self.global_step)
                #     tf.summary.scalar(
                #         "reconstruction_loss", epoch_losses["reconstruction"], step=self.global_step
                #     )
                #     tf.summary.scalar("kl_loss", epoch_losses["kl"], step=self.global_step)
                #     tf.summary.scalar("true_loss", epoch_losses["true"], step=self.global_step)
                #     tf.summary.scalar("false_loss", epoch_losses["false"], step=self.global_step)
                #     tf.summary.scalar(
                #         "learning_rate",
                #         self.vae.optimizer.learning_rate.numpy(),
                #         step=self.global_step,
                #     )
                #
                # with self.val_writer.as_default():
                #     tf.summary.scalar(
                #         "validation_total_loss", val_losses["total"], step=self.global_step
                #     )
                #     tf.summary.scalar(
                #         "validation_reconstruction_loss",
                #         val_losses["reconstruction"],
                #         step=self.global_step,
                #     )
                #     tf.summary.scalar("validation_kl_loss", val_losses["kl"], step=self.global_step)
                #     tf.summary.scalar(
                #         "validation_true_loss", val_losses["true"], step=self.global_step
                #     )
                #     tf.summary.scalar(
                #         "validation_false_loss", val_losses["false"], step=self.global_step
                #     )
                #
                # # Flush writers to ensure data is written
                # self.train_writer.flush()
                # self.val_writer.flush()
                #
                # # Increment global step
                # self.global_step += 1

                logger.info(f"Epoch {epoch + 1}")
                logger.info(
                    f"Train -- Total: {epoch_losses['total']:.4f}, "
                    f"Recon: {epoch_losses['reconstruction']:.4f}, "
                    f"KL: {epoch_losses['kl']:.4f}, "
                    f"True: {epoch_losses['true']:.4f}, "
                    f"False: {epoch_losses['false']:.4f}, "
                    f"Duration: {train_duration:.2f} "
                )
                logger.info(
                    f"Gradient norm -- Mean: {gradient_norm_mean:.4f}, "
                    f"Std: {gradient_norm_std:.4f}, "
                    f"Max: {gradient_norm_max:.4f}, "
                    f"Clipping rate: {clipping_rate:.4f} "
                )
                logger.info(
                    f"Val -- Total: {val_losses['total']:.4f}, "
                    f"Recon: {val_losses['reconstruction']:.4f}, "
                    f"KL: {val_losses['kl']:.4f}, "
                    f"True: {val_losses['true']:.4f}, "
                    f"False: {val_losses['false']:.4f}, "
                    f"Duration: {val_duration:.2f} "
                )

                # Adaptive learning rate
                self._update_learning_rate(val_losses)

            # NOTE: combine plot_beta_vae_loss_curves(), plot_beta_vae_training_stability(), and plot_latent_space_gif() into plot_training_progress()?
            # Plot loss curves
            self.plot_beta_vae_loss_curves(tag=f"round_{round_idx + 1:02d}", dir="checkpoints")

            # Plot clipping rate
            self.plot_beta_vae_training_stability(
                tag=f"round_{round_idx + 1:02d}", dir="checkpoints"
            )

            # Plot injection stats
            self.plot_injection_stats(
                tag=f"round_{round_idx + 1:02d}",
                dir="checkpoints",
            )

            # NOTE: commented out to save compute. a final latent space gif at the end of training should suffice
            # Generate latent space GIF
            # self.plot_latent_space_gif(tag=f"round_{round_idx + 1:02d}", dir="checkpoints")

            # Save checkpoint
            self.save_models(tag=f"round_{round_idx + 1:02d}", dir="checkpoints")

        except Exception as e:
            logger.error(f"Error in train_round(): {e}")
            raise  # Re-raise to propagate error

        # Run cleanup regardless if round finishes successfully or not
        finally:
            # NOTE: should check to make sure train_holder & datasets exist first
            # Clear intermediate data
            train_holder.clear()
            del train_dataset, val_dataset

            # Clear latent viz distributed dataset (rebuilt each round)
            if self._latent_viz_holder is not None:
                self._latent_viz_holder.clear()
            self._latent_viz_dataset = None
            self._latent_viz_n_padded = None
            self._latent_viz_n_samples = None
            self._latent_viz_steps = None
            self._latent_viz_holder = None
            self._viz_encode_fn = None

            # Force TensorFlow to release internal references to datasets/iterators
            # This prevents generator closures from accumulating in memory between rounds
            tf.keras.backend.clear_session()
            logger.info("Cleared TensorFlow session state")

            # Reset multiprocessing pools in DataGenerator after each round
            # to further avoid memory accumulation
            self.data_generator.reset_managed_pool()
            logger.info("Reset managed pools")

            gc.collect()

    def _train_epoch(
        self,
        round_idx,
        epoch_idx,
        snr_base,
        snr_range,
        dataset,
        steps_per_epoch,
        accumulation_steps=1,
        start_time=None,
    ):
        """
        Perform a single training epoch with gradient accumulation
        """
        if not start_time:
            start_time = time.time()

        epoch_losses = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0, "true": 0.0, "false": 0.0}
        epoch_gradient_norms = []
        iterator = iter(dataset)

        try:
            for step in range(steps_per_epoch):
                step_losses = {
                    "total": 0.0,
                    "reconstruction": 0.0,
                    "kl": 0.0,
                    "true": 0.0,
                    "false": 0.0,
                }

                # Initialize accumulated gradients
                accumulated_gradients = None
                successful_accumulations = 0

                # Process sub-steps for gradient accumulation
                for sub_step in range(accumulation_steps):
                    try:
                        micro_batch = next(iterator)

                        # Compute gradients & losses
                        micro_grads, micro_losses = self._distributed_train_step(micro_batch)

                        # NOTE: come back to this later (any or all?)
                        # Sanity check: verify gradients are valid before accumulating
                        if micro_grads is None or all(g is None for g in micro_grads):
                            logger.warning(
                                f"Step {step + 1}, sub-step {sub_step + 1}: "
                                f"All gradients are None, skipping this micro-batch"
                            )
                            continue

                        # Accumulate gradients over sub-steps
                        if accumulated_gradients is None:
                            accumulated_gradients = micro_grads
                        else:
                            accumulated_gradients = [
                                # NOTE: come back to this later (what if ag and g are both None)
                                ag + g if ag is not None and g is not None else ag or g
                                for ag, g in zip(accumulated_gradients, micro_grads, strict=False)
                            ]

                        successful_accumulations += 1

                        # Accumulate losses over sub-steps
                        for key in step_losses:
                            step_losses[key] += micro_losses[key]

                    except StopIteration:  # Empty dataset
                        logger.error(
                            f"Dataset exhausted at step {step + 1}, sub-step {sub_step + 1}"
                        )
                        raise  # Re-raise to propagate error

                    except Exception as e:
                        logger.error(
                            f"Error during gradient computation at step {step + 1}, sub-step {sub_step + 1}: {e}"
                        )
                        raise  # Re-raise to propagate error

                # Sanity check: verify that gradient accumulation was successful
                if accumulated_gradients is None or successful_accumulations == 0:
                    logger.error(f"Step {step + 1}: No valid gradients accumulated!")
                    raise RuntimeError(f"Failed to accumulate gradients at step {step + 1}")

                # Average accumulated gradients over sub-steps
                accumulated_gradients = [
                    g / successful_accumulations if g is not None else None
                    for g in accumulated_gradients
                ]

                # Sanity check: verify no NaN/Inf in gradients
                has_nan_or_inf = False
                for g in accumulated_gradients:
                    if g is not None and (
                        tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g))
                    ):
                        has_nan_or_inf = True
                        break

                if has_nan_or_inf:
                    logger.error(f"Step {step + 1}: NaN or Inf detected in gradients!")
                    raise RuntimeError(f"NaN/Inf gradients at step {step + 1}")

                # Apply accumulated gradients
                global_norm = self._apply_gradients(accumulated_gradients)
                epoch_gradient_norms.append(float(global_norm))

                # Capture latent snapshot every N steps, and on the final step
                is_interval_step = (step + 1) % self.config.training.latent_viz_step_interval == 0
                is_final_step = (step + 1) == steps_per_epoch
                if (is_interval_step or is_final_step) and self._latent_viz_dataset is not None:
                    self._capture_latent_snapshot(
                        round_idx,
                        epoch_idx,
                        step,
                        snr_base,
                        snr_range,
                    )

                for key, loss in step_losses.items():
                    # Average step losses over sub-steps
                    avg_loss = loss / successful_accumulations
                    step_losses[key] = avg_loss
                    # Accumulate epoch losses over training steps
                    epoch_losses[key] += avg_loss

            # Average epoch losses over training steps
            for key in epoch_losses:
                epoch_losses[key] /= steps_per_epoch

            # Calculate train epoch duration
            train_duration = time.time() - start_time

            return epoch_losses, epoch_gradient_norms, train_duration

        except Exception as e:
            logger.error(f"Error in _train_epoch(): {e}")
            raise  # Re-raise to propagate error

        # Run cleanup regardless if epoch finishes successfully or not
        finally:
            # NOTE: should check to make sure iterator exists first
            del iterator
            gc.collect()

    def _validate_epoch(self, dataset, steps, start_time=None):
        """
        Perform a single validation epoch
        """
        if not start_time:
            start_time = time.time()

        val_losses = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0, "true": 0.0, "false": 0.0}
        iterator = iter(dataset)

        try:
            for _step in range(steps):
                batch = next(iterator)

                # Compute losses
                step_losses = self._distributed_val_step(batch)

                # Accumulate validation losses over validation steps
                for key in val_losses:
                    val_losses[key] += step_losses[key]

            # Average validation losses over validation steps
            for key in val_losses:
                val_losses[key] /= steps

            # Calculate val epoch duration
            val_duration = time.time() - start_time

            return val_losses, val_duration

        except Exception as e:
            logger.error(f"Error in _validate_epoch(): {e}")
            raise  # Re-raise to propagate error

        # Run cleanup regardless if epoch finishes successfully or not
        finally:
            # NOTE: should check to make sure iterator exists first
            del iterator
            gc.collect()

    @tf.function
    def _distributed_train_step(self, batch_data):
        """
        Perform a single distributed training step
        Returns reduced gradients & losses
        """
        num_replicas = self.strategy.num_replicas_in_sync

        def step_fn(data):
            """Per-replica training step"""
            x, y = data
            main_data = x[0]
            true_data = x[1]
            false_data = x[2]

            with tf.GradientTape() as tape:
                # Compute losses
                losses = self.vae.compute_total_loss(
                    main_data, true_data, false_data, y, training=True
                )

            # Compute gradients
            gradients = tape.gradient(losses["total_loss"], self.vae.trainable_variables)

            return gradients, losses

        # Run training step on all replicas
        per_replica_grads, per_replica_losses = self.strategy.run(step_fn, args=(batch_data,))

        # Reduce gradients across replicas
        reduced_grads = []
        for grad in per_replica_grads:
            if grad is not None:
                reduced_grad = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, grad, axis=None)
                reduced_grads.append(reduced_grad)
            else:
                reduced_grads.append(None)

        # Reduce losses across replicas
        reduced_losses = {
            "total": self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses["total_loss"], axis=None
            ),
            "reconstruction": self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses["reconstruction_loss"], axis=None
            ),
            "kl": self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses["kl_loss"], axis=None
            ),
            "true": self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses["true_loss"], axis=None
            ),
            "false": self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses["false_loss"], axis=None
            ),
        }

        return reduced_grads, reduced_losses

    @tf.function
    def _apply_gradients(self, gradients):
        """
        Apply gradients after gradient clipping by global L2 norm
        """
        # TEST: is this still needed? currently every step seems to be getting clipped. what happens if we just don't?
        # Clip gradients for additional stability
        # Note, this step is optional but recommended given our beta-VAE architecture's
        # heterogeneous gradient scale (reconstruction + KL loss components)
        # Gradient clipping computes the global L2 norm across all gradient tensors, then rescales
        # them proportionally if that norm exceeds some clip_norm threshold. This preserves the
        # relative direction of the gradient vector in parameter space while simultaneously bounding
        # its magnitude, maintaining the optimization trajectory's direction, which is critical for
        # training stability
        # Alternatively, per-tensor clipping (with tf.clip_by_norm() on each gradient independently)
        # could also work, but may distort the gradient direction (parameters with smaller gradients
        # get disproportionately boosted relative to those with larger gradients). Only use if you
        # need layer-specific interventions (e.g. lower LR for encoder, or gradient clipping
        # per-component, etc.)
        # The 1.0 threshold was chosen to be aggressive enough to prevent exploding gradients, while
        # permissive enough to not overly dampen learning. Healthy training should progress with
        # global_norm consistently below clip_norm, with the occasional instability (e.g. due to bad
        # batches, or KL spikes) getting caught & dampened. If you notice global_norm consistently
        # exceeding clip_norm, even with adaptive LR in place, consider increasing clip_norm to
        # allow more of the true gradients to pass through. A general heuristic for clip_norm is to
        # have no more than 1-5% of steps trigger gradient clipping
        clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        # Apply gradients
        self.vae.optimizer.apply_gradients(
            zip(clipped_gradients, self.vae.trainable_variables, strict=False)
        )

        # Return pre-clipping global norm (for monitoring)
        return global_norm

    @tf.function
    def _distributed_val_step(self, batch_data):
        """
        Perform a single distributed validation step
        Returns reduced losses
        """

        def step_fn(data):
            """Per-replica validation step"""
            x, y = data
            main_data = x[0]
            true_data = x[1]
            false_data = x[2]

            # Compute losses
            losses = self.vae.compute_total_loss(
                main_data, true_data, false_data, y, training=False
            )

            return losses

        # Run validation step on all replicas
        per_replica_losses = self.strategy.run(step_fn, args=(batch_data,))

        # Reduce losses across replicas
        reduced_losses = {
            "total": self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses["total_loss"], axis=None
            ),
            "reconstruction": self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses["reconstruction_loss"], axis=None
            ),
            "kl": self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses["kl_loss"], axis=None
            ),
            "true": self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses["true_loss"], axis=None
            ),
            "false": self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses["false_loss"], axis=None
            ),
        }

        return reduced_losses

    def _calculate_curriculum_snr(self, round_idx: int) -> tuple[int, int]:
        """
        Calculate SNR parameters for curriculum learning

        Args:
            round_idx: Current training round (0-indexed)
            total_rounds: Total number of training rounds

        Returns:
            (snr_base, snr_range) tuple
        """
        total_rounds = self.config.training.num_training_rounds
        snr_base = self.config.training.snr_base
        initial_snr_range = self.config.training.initial_snr_range
        final_snr_range = self.config.training.final_snr_range
        schedule = self.config.training.curriculum_schedule

        # Edge case: use initial snr range if only training for 1 round
        if total_rounds == 1:
            return snr_base, initial_snr_range

        # Progress through curriculum: 0.0 (easy) -> 1.0 (hard)
        progress = round_idx / (total_rounds - 1)

        # Linear progression from wide to narrow SNR range
        if schedule == "linear":
            current_range = initial_snr_range - progress * (initial_snr_range - final_snr_range)
        # Exponential decay - start easy, then get hard quickly
        elif schedule == "exponential":
            decay_rate = self.config.training.exponential_decay_rate
            # Sanity check: validate decay_rate < 0 to avoid division by zero
            if decay_rate >= 0:
                raise ValueError(
                    f"exponential_decay_rate must be < 0 for exponential schedule, got {decay_rate}"
                )
            # Normalize exponential to ensure exact endpoints at progress=0 and progress=1
            decay_factor = (np.exp(decay_rate * progress) - np.exp(decay_rate)) / (
                1 - np.exp(decay_rate)
            )
            current_range = final_snr_range + (initial_snr_range - final_snr_range) * decay_factor
        # TODO: generalize this to receive a step schedule (as a list/dict?) validate that len(list/dict) is divisible by num_training_rounds
        # Step function - easy for first part, hard for second part
        elif schedule == "step":
            easy_rounds = self.config.training.step_easy_rounds
            hard_rounds = self.config.training.step_hard_rounds
            # Sanity check: validate easy_rounds + hard_rounds add up to total_rounds
            if easy_rounds + hard_rounds != total_rounds:
                raise ValueError(
                    f"easy_rounds ({easy_rounds}) + hard_rounds ({hard_rounds}) must equal total_rounds ({total_rounds}), got {easy_rounds + hard_rounds} instead"
                )
            if round_idx < easy_rounds:
                current_range = initial_snr_range
            elif round_idx - easy_rounds < hard_rounds:
                current_range = final_snr_range
            else:
                raise RuntimeError(
                    f"round_idx ({round_idx}) exceeded easy_rounds + hard_rounds ({easy_rounds} + {hard_rounds})"
                )
        else:
            raise ValueError(
                f"'{schedule} is invalid. Accepted values: 'linear', 'exponential', 'step'"
            )

        return snr_base, int(current_range)

    def _update_learning_rate(self, val_losses):
        """
        Robust adaptive learning rate with multiple safeguards

        Note the following heuristic:
        min_learning_rate - base_learning_rate * (1 - reduction_factor) ^ (epochs_per_round / patience_threshold)
          => LR can only reach min_learning_rate during round if above expression is > 0
          => else LR will reset at start of new round before reaching min_learning_rate
        """

        current_lr = self.vae.optimizer.learning_rate.numpy()
        if current_lr <= self.config.training.min_learning_rate:
            return current_lr

        # Use validation loss for better generalization
        if not hasattr(self, "best_val_loss"):
            self.best_val_loss = float("inf")
            self.patience_counter = 0

        current_val_loss = float(val_losses["total"])

        # Check if validation loss improved
        if current_val_loss < self.best_val_loss * (1 - self.config.training.min_pct_improvement):
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Reduce LR if no meaningful improvement for consecutive epochs
        if self.patience_counter >= self.config.training.patience_threshold:
            new_lr = max(
                current_lr * (1 - self.config.training.reduction_factor),
                self.config.training.min_learning_rate,
            )

            self.vae.optimizer.learning_rate.assign(new_lr)
            self.patience_counter = 0  # Reset counter

            logger.info(f"Reduced learning rate: {current_lr:.2e} -> {new_lr:.2e}")
            return new_lr

        return current_lr

    def _distributed_encode(
        self, dataset, n_steps, encode_fn, n_samples, latent_dim, logging=False
    ):
        """
        Run distributed encoding over a dataset using a provided @tf.function.

        The number of output arrays is inferred from encode_fn's return type on the first step:
        a single PerReplica tensor produces 1 output, a tuple of PerReplica tensors produces N.

        Args:
            dataset: Distributed dataset to iterate
            n_steps: Number of steps to iterate
            encode_fn: @tf.function that takes a batch and returns one or more per-replica tensors
            n_samples: Total number of output rows per array
            latent_dim: Latent dimension
            logging: Whether to log progress (default: False)

        Returns:
            List of np.ndarray. Each shape (n_samples, latent_dim).
        """
        # Process all batches
        iterator = iter(dataset)
        current_idx = 0
        outputs = None

        try:
            for _ in range(n_steps):
                batch = next(iterator)

                # Get per-replica latents for this batch
                results = encode_fn(batch)

                # Normalize to tuple: a single PerReplica tensor is not a tuple,
                # while multiple outputs are returned as a tuple of PerReplica tensors
                if not isinstance(results, tuple):
                    results = (results,)

                # Lazily allocate output arrays on first step (avoids needing n_outputs param)
                if outputs is None:
                    outputs = [
                        np.empty((n_samples, latent_dim), dtype=np.float32)
                        for _ in range(len(results))
                    ]

                # Extract results from each replica and concatenate across all replicas
                # This avoids the inefficient gather operation with NCCL
                for i, per_replica in enumerate(results):
                    local_results = self.strategy.experimental_local_results(per_replica)
                    batch_np = np.concatenate([r.numpy() for r in local_results], axis=0)
                    batch_size = batch_np.shape[0]
                    outputs[i][current_idx : current_idx + batch_size] = batch_np
                    del local_results, batch_np

                current_idx += batch_size

                del results
                gc.collect()

            # Log progress
            if logging:
                logger.info(f"Finished encoding {n_steps} steps")
        finally:
            # NOTE: should check to make sure iterator exists first
            del iterator
            gc.collect()

        return outputs

    # NOTE: write what to db? (e.g. accuracy, F1)
    # TODO: visualize classification accuracy (ROC-AUC, precision-recall) at different thresholds when training is complete
    def train_random_forest(self):
        """Train Random Forest"""
        logger.info("Training Random Forest classifier...")

        # Initialize RF model
        if self.rf_model is None:
            self.rf_model = RandomForestModel()

        elif self.rf_model.is_trained:
            logger.info("Random Forest classifier already trained. Exiting training loop.")
            return

        # # BUG:
        # # edge case where if save-tag was already used in the past, train_random_forest() will
        # # load previous model weights to train RF, since pipeline.save_models() happens after
        # # pipeline.train_random_forest()
        # # add a check to validate_args() to make sure save-tag is distinct?
        # # note, should only be a problem during testing, since load_models() only runs if
        # # check_encoder_trained() fails
        # # maybe we first look into checkpoints first, then only in the main models directory?
        # Load encoder weights if untrained
        logger.info("Checking if encoder weights appear trained")

        try:
            if not check_encoder_trained(self.vae.encoder, threshold=0.2):
                try:
                    logger.info("Loading latest pre-trained encoder weights")
                    self.load_models()

                except Exception as e:
                    logger.warning(f"Failed to load pre-trained encoder weights: {e}")

                    try:
                        logger.info("Loading latest checkpointed weights instead")
                        self.load_models(dir="checkpoints")

                    except Exception as e:
                        logger.warning(f"Failed to load latest checkpointed weights: {e}")
                        logger.warning("Proceeding with current encoder weights")

        except Exception as e:
            logger.warning(f"Could not verify encoder weights status: {e}")
            logger.warning("Proceeding with current encoder weights")

        n_samples = (
            self.config.training.num_samples_rf // 2
        )  # Divide by 2 to compensate for generate_triplet_batch internally creating n * 2 samples (n true, n false)
        snr_base = self.config.training.snr_base
        snr_range = (
            self.config.training.initial_snr_range
        )  # NOTE: should we use initial_snr_range or final_snr_range?

        latent_dim = self.config.beta_vae.latent_dim
        num_observations = self.config.data.num_observations
        time_bins = self.config.data.time_bins
        width_bin = self.config.data.width_bin // self.config.data.downsample_factor

        # Generate training data
        logger.info(f"Preparing training set with SNR: {snr_base}-{snr_base + snr_range}")

        rf_data = self.data_generator.generate_triplet_batch(n_samples, snr_base, snr_range)

        # Prepare distributed dataset for inference
        results = prepare_distributed_inf_dataset(
            data=rf_data,
            per_replica_inf_batch_size=self.config.training.per_replica_val_batch_size,
            num_replicas=self.strategy.num_replicas_in_sync,
            strategy=self.strategy,
        )

        del rf_data
        gc.collect()

        inf_dataset = results["inf_dataset"]
        n_inf_trimmed = results["n_inf_trimmed"]
        inf_steps = results["inf_steps"]
        inf_holder = results["_inf_holder"]

        del results
        gc.collect()

        logger.info(f"Generating latents for {n_inf_trimmed} samples using distributed inference")

        # Create distributed inference function
        @tf.function
        def rf_encode_fn(batch_data):
            """Encode batch data using distributed strategy"""

            def encode_fn(data):
                """Per-replica encoding step"""
                # Extract true & false components
                true_data, false_data = data

                # Reshape for encoder: (batch, 6, 16, 512) -> (batch * 6, 16, 512, 1)
                true_reshaped = tf.reshape(true_data, [-1, time_bins, width_bin, 1])
                false_reshaped = tf.reshape(false_data, [-1, time_bins, width_bin, 1])

                # Encode (returns z_mean, z_log_var, z)
                _, _, true_z = self.vae.encoder(true_reshaped, training=False)
                _, _, false_z = self.vae.encoder(false_reshaped, training=False)

                return true_z, false_z

            # Run encoding on all replicas
            per_replica_true, per_replica_false = self.strategy.run(encode_fn, args=(batch_data,))

            return per_replica_true, per_replica_false

        try:
            # TEST: make sure this works after refactor
            true_latents, false_latents = self._distributed_encode(
                dataset=inf_dataset,
                n_steps=inf_steps,
                encode_fn=rf_encode_fn,
                n_samples=n_inf_trimmed * num_observations,
                latent_dim=latent_dim,
                logging=True,
            )

            # Train Random Forest classifier
            self.rf_model.train(true_latents, false_latents)

            logger.info("Random Forest training complete")

        except Exception as e:
            logger.error(f"Error in train_random_forest(): {e}")
            raise  # Re-raise to propagate error

        finally:
            # NOTE: should check to make sure holder & dataset exist first
            # Clear intermediate data
            inf_holder.clear()
            del inf_dataset

            # Force TensorFlow to release internal references to datasets/iterators
            # This prevents generator closures from accumulating in memory between rounds
            tf.keras.backend.clear_session()
            logger.info("Cleared TensorFlow session state")

            # Reset multiprocessing pools in DataGenerator to further avoid memory accumulation
            self.data_generator.reset_managed_pool()
            logger.info("Reset managed pools")

            # NOTE: should check to make sure arrays exist first
            del true_latents, false_latents
            gc.collect()

    # TODO: reorder plot methods (def & call sites): train -> latent -> injection
    # NOTE: combine plot_beta_vae_loss_curves(), plot_beta_vae_training_stability(), and plot_latent_space_gif() into plot_training_progress()?
    def plot_beta_vae_loss_curves(self, tag: str | None = None, dir: str | None = None):
        """Plot beta-VAE training history"""
        if tag is None:
            tag = self.config.checkpoint.save_tag

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        current_time = time.time()

        if self.db is None:
            raise RuntimeError("No database instance detected - cannot generate loss curves plot")

        # Flush database to ensure all training stats are written before plotting
        logger.info("Flushing database before plotting...")
        if not self.db.flush():
            logger.warning(
                "Database flush failed. Plotting may encounter issues. Proceeding anyways..."
            )
        else:
            logger.info("Database flushed")

        # NOTE: how to handle retries under current implementation?
        # Query training stats from database
        all_stats = self.db.query_training_stat(
            model_name="beta_vae",
            start_round_number=1,  # NOTE: come back to this later (is this correct? what about fault tolerance?)
            tag=self.config.checkpoint.save_tag,  # The query tag is different from the input arg tag
            start_time=self.start_time,
            end_time=current_time,
        )

        if not all_stats:
            logger.warning("No training stats data to plot")
            return

        # TODO: potential memory optimization here with array pre-allocation? is there a way to just use the all_stats dict directly? is the potential improvement worth the effort?
        # Group query results by stat_name
        raw_history = {}
        for stat in all_stats:
            key = stat["stat_name"]
            if key not in raw_history:
                raw_history[key] = []
            # Store (round, epoch, value) tuple for later sorting
            raw_history[key].append((stat["round_number"], stat["epoch_number"], stat["value"]))

        del all_stats
        gc.collect()

        # Sort by (round, epoch) and extract just the values
        history = {}
        for key, values in raw_history.items():
            sorted_values = sorted(values, key=lambda x: (x[0], x[1]))  # Sort by round, epoch
            history[key] = [v[2] for v in sorted_values]  # Extract just the values

        del raw_history
        gc.collect()

        epochs = range(1, len(history.get("total_loss", [])) + 1)

        # Add SNR range background shading to all axes
        snr_by_round = self._get_snr_by_round(current_time)
        epochs_per_round = self.config.training.epochs_per_round

        # Scale figure width for many rounds
        num_rounds = len(snr_by_round)
        base_width = 25
        fig_width = base_width * (1 + max(0, num_rounds - 10) * 0.05)  # +5% width per round over 10

        # Create figure & setup gridspec
        fig = plt.figure(figsize=(fig_width, 12))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

        # Top subplot spanning full width - Total Loss
        ax_top = fig.add_subplot(gs[0, :])

        # Bottom subplots - Individual losses
        ax_recon = fig.add_subplot(gs[1, 0])
        ax_kl = fig.add_subplot(gs[1, 1])
        ax_true = fig.add_subplot(gs[1, 2])
        ax_false = fig.add_subplot(gs[1, 3])

        fig.suptitle(
            f"Beta-VAE Loss Curves ({tag}, {machine_name})", fontsize=18, fontweight="bold"
        )

        # Top subplot gets shading + text annotations, bottom subplots get shading only
        self._add_snr_range_shading(
            ax_top, snr_by_round, epochs_per_round, use_rounds=False, show_text_annotations=True
        )
        for ax in [ax_recon, ax_kl, ax_true, ax_false]:
            self._add_snr_range_shading(
                ax, snr_by_round, epochs_per_round, use_rounds=False, show_text_annotations=False
            )

        # Helper function to plot dual y-axis
        def plot_dual_axis(ax, title, train_key, val_key):
            # Create secondary y-axis for learning rate
            ax2 = ax.twinx()

            # Plot train and validation on left y-axis
            if train_key in history and history[train_key]:
                ax.plot(epochs, history[train_key], color="blue", label="train", linewidth=2)
            if val_key in history and history[val_key]:
                ax.plot(epochs, history[val_key], color="orange", label="val", linewidth=2)

            # Plot learning rate on right y-axis
            if "learning_rate" in history and history["learning_rate"]:
                ax2.plot(
                    epochs,
                    history["learning_rate"],
                    color="grey",
                    label="learning rate",
                    linewidth=1,
                    alpha=0.7,
                    linestyle="--",
                )

            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

            ax.tick_params(axis="both", labelsize=12)
            ax2.tick_params(axis="y", labelcolor="grey", labelsize=12)

        # Top subplot - Total Loss
        plot_dual_axis(ax_top, "Total Loss", "total_loss", "val_total_loss")

        # Bottom subplots
        plot_dual_axis(
            ax_recon, "Reconstruction Loss", "reconstruction_loss", "val_reconstruction_loss"
        )
        plot_dual_axis(ax_kl, "KL Divergence", "kl_loss", "val_kl_loss")
        plot_dual_axis(ax_true, "True Loss", "true_loss", "val_true_loss")
        plot_dual_axis(ax_false, "False Loss", "false_loss", "val_false_loss")

        # Create shared legend at top right of figure
        train_line = mlines.Line2D([], [], color="blue", linewidth=2, label="Train")
        val_line = mlines.Line2D([], [], color="orange", linewidth=2, label="Validation")
        lr_line = mlines.Line2D(
            [], [], color="grey", linewidth=1, linestyle="--", alpha=0.7, label="Learning Rate"
        )

        fig.legend(
            handles=[train_line, val_line, lr_line],
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        # WARN: not sure why the following warning occurs?
        # WARN: 2026-01-03 14:01:12,049 | py.warnings | WARNING | /home/zachy/src/Aetherscan/src/aetherscan/train.py:1606: UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
        # WARN: plt.tight_layout()
        plt.tight_layout()

        # Save plot
        if dir is not None:
            save_path = os.path.join(
                self.config.output_path, "plots", dir, f"beta_vae_loss_curves_{tag}.png"
            )
        else:
            save_path = os.path.join(
                self.config.output_path, "plots", f"beta_vae_loss_curves_{tag}.png"
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create dir if it doesn't exist

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()

        logger.info(f"Beta-VAE loss curves plot saved to: {save_path}")

        # Upload to Slack
        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"Beta-VAE Loss Curves - ({tag}, {machine_name})",
            )

        # NOTE:
        # del history and del epochs get flagged by ruff as error code F821
        # that is, ruff flags all references inside nested functions (plot_dual_axis) as undefined
        # despite plot_dual_axis being called before del
        # we could try a hacky solution of moving the del statements before plot_dual_axis is defined,
        # or by assigning history & epochs to local variables inside plot_dual_axis as default params,
        # capturing the values at definition time and allowing us to dereference the variables in the
        # outer scope after the function definition
        # but realistically, the variables will be garbage collected anyways when the frame exits
        # after the return statement, plus the training_stats arrays are much more manageable wrt
        # memory compared to injection_stats arrays, so we don't call del on history or epochs
        del snr_by_round
        gc.collect()

    # TODO: reorder plot methods (def & call sites): train -> latent -> injection
    # NOTE: combine plot_beta_vae_loss_curves(), plot_beta_vae_training_stability(), and plot_latent_space_gif() into plot_training_progress()?
    def plot_beta_vae_training_stability(self, tag: str | None = None, dir: str | None = None):
        """
        Plot gradient clipping rate and gradient norm statistics.

        Generates a 2x3 grid:
        - Top row: Clipping rate spanning full width
        - Bottom row: gradient_norm_mean, gradient_norm_std, gradient_norm_max
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        current_time = time.time()

        if self.db is None:
            raise RuntimeError(
                "No database instance detected - cannot generate training stability plot"
            )

        # Flush database to ensure all training stats are written before plotting
        logger.info("Flushing database before plotting...")
        if not self.db.flush():
            logger.warning(
                "Database flush failed. Plotting may encounter issues. Proceeding anyways..."
            )
        else:
            logger.info("Database flushed")

        # Query training stats from database
        all_stats = self.db.query_training_stat(
            model_name="beta_vae",
            start_round_number=1,
            tag=self.config.checkpoint.save_tag,
            start_time=self.start_time,
            end_time=current_time,
        )

        if not all_stats:
            logger.warning("No training stats data to plot")
            return

        # Group query results by stat_name
        raw_history = {}
        for stat in all_stats:
            key = stat["stat_name"]
            if key not in raw_history:
                raw_history[key] = []
            raw_history[key].append((stat["round_number"], stat["epoch_number"], stat["value"]))

        del all_stats
        gc.collect()

        # Sort by (round, epoch) and extract just the values
        history = {}
        for key, values in raw_history.items():
            sorted_values = sorted(values, key=lambda x: (x[0], x[1]))
            history[key] = [v[2] for v in sorted_values]

        del raw_history
        gc.collect()

        epochs = range(1, len(history.get("clipping_rate", [])) + 1)

        if not epochs:
            logger.warning("No clipping rate data to plot")
            return

        # Add SNR range background shading to all axes
        snr_by_round = self._get_snr_by_round(current_time)
        epochs_per_round = self.config.training.epochs_per_round

        # Scale figure width for many rounds
        num_rounds = len(snr_by_round)
        base_width = 25
        fig_width = base_width * (1 + max(0, num_rounds - 10) * 0.05)  # +5% width per round over 10

        # Create figure & setup gridspec (following plot_beta_vae_loss_curves pattern)
        fig = plt.figure(figsize=(fig_width, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

        # Top subplot spanning full width - Clipping Rate
        ax_top = fig.add_subplot(gs[0, :])

        # Bottom subplots - Gradient norm statistics
        ax_mean = fig.add_subplot(gs[1, 0])
        ax_std = fig.add_subplot(gs[1, 1])
        ax_max = fig.add_subplot(gs[1, 2])

        fig.suptitle(
            f"Beta-VAE Training Stability ({tag}, {machine_name})", fontsize=18, fontweight="bold"
        )

        # Top subplot gets shading + text annotations, bottom subplots get shading only
        self._add_snr_range_shading(
            ax_top, snr_by_round, epochs_per_round, use_rounds=False, show_text_annotations=True
        )
        for ax in [ax_mean, ax_std, ax_max]:
            self._add_snr_range_shading(
                ax, snr_by_round, epochs_per_round, use_rounds=False, show_text_annotations=False
            )

        # Top plot: Clipping Rate (blue)
        if "clipping_rate" in history and history["clipping_rate"]:
            ax_top.plot(epochs, history["clipping_rate"], color="blue", linewidth=2)

        # Add optimal clipping rate bounds (gray dashed)
        ax_top.axhline(y=0.01, color="gray", linestyle="--", linewidth=1.5, alpha=0.8)
        ax_top.axhline(y=0.05, color="gray", linestyle="--", linewidth=1.5, alpha=0.8)

        ax_top.set_title("Gradient Clipping Rate", fontsize=14, fontweight="bold")
        ax_top.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax_top.grid(True, alpha=0.3)

        # Bottom left: Gradient Norm Mean (orange)
        if "gradient_norm_mean" in history and history["gradient_norm_mean"]:
            ax_mean.plot(epochs, history["gradient_norm_mean"], color="orange", linewidth=2)
        ax_mean.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.8)
        ax_mean.set_title("Gradient Norm Mean", fontsize=14, fontweight="bold")
        ax_mean.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax_mean.grid(True, alpha=0.3)

        # Bottom center: Gradient Norm Std (orange)
        if "gradient_norm_std" in history and history["gradient_norm_std"]:
            ax_std.plot(epochs, history["gradient_norm_std"], color="orange", linewidth=2)
        ax_std.set_title("Gradient Norm Std", fontsize=14, fontweight="bold")
        ax_std.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax_std.grid(True, alpha=0.3)

        # Bottom right: Gradient Norm Max (orange)
        if "gradient_norm_max" in history and history["gradient_norm_max"]:
            ax_max.plot(epochs, history["gradient_norm_max"], color="orange", linewidth=2)
        ax_max.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.8)
        ax_max.set_title("Gradient Norm Max", fontsize=14, fontweight="bold")
        ax_max.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax_max.grid(True, alpha=0.3)

        # Helper to convert data y to figure y
        def data_to_fig_y(ax, y_data):
            ax_bbox = ax.get_position()
            ylim = ax.get_ylim()
            y_norm = (y_data - ylim[0]) / (ylim[1] - ylim[0])
            return ax_bbox.y0 + y_norm * ax_bbox.height

        # External annotations for threshold lines (placed outside subplots on right)
        ax_top_bbox = ax_top.get_position()
        fig.text(
            ax_top_bbox.x1 + 0.005,
            data_to_fig_y(ax_top, 0.01),
            "Min optimal clipping rate (1%)",
            fontsize=9,
            va="center",
            color="gray",
        )
        fig.text(
            ax_top_bbox.x1 + 0.005,
            data_to_fig_y(ax_top, 0.05),
            "Max optimal clipping rate (5%)",
            fontsize=9,
            va="center",
            color="gray",
        )

        ax_mean_bbox = ax_mean.get_position()
        fig.text(
            ax_mean_bbox.x1 + 0.005,
            data_to_fig_y(ax_mean, 1.0),
            "Clipping\nthreshold\n(1.0)",
            fontsize=9,
            va="center",
            ma="left",
            color="gray",
        )

        ax_max_bbox = ax_max.get_position()
        fig.text(
            ax_max_bbox.x1 + 0.005,
            data_to_fig_y(ax_max, 1.0),
            "Clipping\nthreshold\n(1.0)",
            fontsize=9,
            va="center",
            ma="left",
            color="gray",
        )

        # Unified figure legend (separate entries for Clipping Rate and Gradient Norm)
        legend_handles = [
            mlines.Line2D([], [], color="blue", linewidth=2, label="Clipping Rate"),
            mlines.Line2D([], [], color="orange", linewidth=2, label="Gradient Norm"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        plt.tight_layout(rect=[0, 0, 0.72, 1])  # More room on right for annotations

        # Save plot
        if dir is not None:
            save_path = os.path.join(
                self.config.output_path, "plots", dir, f"beta_vae_training_stability_{tag}.png"
            )
        else:
            save_path = os.path.join(
                self.config.output_path, "plots", f"beta_vae_training_stability_{tag}.png"
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Beta-VAE training stability plot saved to: {save_path}")

        # Upload to Slack
        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"Beta-VAE Training Stability - ({tag}, {machine_name})",
            )

        del history, snr_by_round, epochs
        gc.collect()

    def _get_snr_by_round(self, current_time: float) -> dict[int, dict[str, float]]:
        """
        Query SNR range data from training_stats and return per-round SNR info.

        Returns:
            Dict mapping round_number to {"floor": x, "ceil": y}
        """
        if self.db is None:
            return {}

        snr_by_round: dict[int, dict[str, float]] = {}

        # Query snr_range_floor
        floor_results = self.db.query_training_stat(
            model_name="beta_vae",
            stat_name="snr_range_floor",
            tag=self.config.checkpoint.save_tag,
            start_time=self.start_time,
            end_time=current_time,
        )

        for r in floor_results:
            round_num = r["round_number"]
            if round_num not in snr_by_round:
                snr_by_round[round_num] = {}
            snr_by_round[round_num]["floor"] = r["value"]

        del floor_results

        # Query snr_range_ceil
        ceil_results = self.db.query_training_stat(
            model_name="beta_vae",
            stat_name="snr_range_ceil",
            tag=self.config.checkpoint.save_tag,
            start_time=self.start_time,
            end_time=current_time,
        )

        for r in ceil_results:
            round_num = r["round_number"]
            if round_num not in snr_by_round:
                snr_by_round[round_num] = {}
            snr_by_round[round_num]["ceil"] = r["value"]

        del ceil_results

        return snr_by_round

    def _add_snr_range_shading(
        self,
        ax,
        snr_by_round: dict[int, dict[str, float]],
        epochs_per_round: int | None = None,
        use_rounds: bool = False,
        show_text_annotations: bool = True,
    ) -> None:
        """
        Add transparent background regions showing SNR range per round.

        Args:
            ax: Matplotlib axis to add shading to
            snr_by_round: Dict mapping round_number to {"floor": x, "ceil": y}
            epochs_per_round: Number of epochs per training round (required if use_rounds=False)
            use_rounds: If True, use round numbers for x-axis; if False, use epochs
            show_text_annotations: If True, show SNR range text annotations in subplot
        """
        if not snr_by_round:
            return

        if not use_rounds and epochs_per_round is None:
            raise ValueError("epochs_per_round is required when use_rounds=False")

        # Alternating colors for visual distinction
        colors = ["#e6f2ff", "#fff2e6"]  # Light blue, light orange
        hatches = ["//", None]  # Striped for odd idx, solid for even idx

        fontsize, rotation = 10, 0

        for idx, (round_num, snr_info) in enumerate(sorted(snr_by_round.items())):
            if "floor" not in snr_info or "ceil" not in snr_info:
                continue

            # Calculate x positions based on use_rounds flag
            if use_rounds:
                start_x = round_num - 0.5
                end_x = round_num + 0.5
                mid_x = round_num
            else:
                start_epoch = (round_num - 1) * epochs_per_round + 1
                end_epoch = round_num * epochs_per_round
                start_x = start_epoch - 0.5
                end_x = end_epoch + 0.5
                mid_x = (start_epoch + end_epoch) / 2

            # Add shaded region with alternating hatch patterns
            hatch = hatches[idx % 2]
            ax.axvspan(
                start_x,
                end_x,
                color=colors[idx % 2],
                alpha=0.5,
                zorder=0,
                hatch=hatch,
                edgecolor="gray" if hatch else None,
            )

            # Add SNR text annotation at top of region
            if show_text_annotations:
                snr_floor = int(snr_info["floor"])
                snr_ceil = int(snr_info["ceil"])
                ax.text(
                    mid_x,
                    0.98,
                    f"SNR: {snr_floor}-{snr_ceil}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=fontsize,
                    rotation=rotation,
                    alpha=0.7,
                )

    # TODO: reorder plot methods (def & call sites): train -> latent -> injection
    # TODO: move injection plots to data_generation.py & call at end of generate_triplet_batch() (instead of at the end of train_round() & run_training_pipeline())
    # NOTE: there's a ton of improvements we could make to this function (and subsequent _plot functions), but i just care that it works well enough for now
    def plot_injection_stats(self, tag: str | None = None, dir: str | None = None):
        """
        Plot injection statistics for bias/leakage analysis.

        Generates 8 figures:
        - 1 injected signal characteristics
        - 1 injection stability
        - 4 global intensity distributions (one per signal_type)
        - 1 A->B global intensity biases
        - 1 final global intensity biases

        Args:
            tag: Plot tag for filename (defaults to save_tag)
            dir: Subdirectory (e.g., "checkpoints" for per-round)
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        if dir is not None:
            save_dir = os.path.join(self.config.output_path, "plots", dir)
        else:
            save_dir = os.path.join(self.config.output_path, "plots")
        os.makedirs(save_dir, exist_ok=True)

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        current_time = time.time()

        if self.db is None:
            raise RuntimeError(
                "No database instance detected - cannot generate injection stats plot"
            )

        # Flush database to ensure all injection stats are written before plotting
        logger.info("Flushing database before plotting...")
        if not self.db.flush():
            logger.warning(
                "Database flush failed. Plotting may encounter issues. Proceeding anyways..."
            )
        else:
            logger.info("Database flushed")

        # Figure 1: Injected signal characteristics
        signal_stats = [
            "snr",
            "drift_rate",
            "signal_width",
            "starting_bin",
            "slope_pixel",
            "y_intercept",
        ]
        eti_stats = {}
        rfi_stats = {}

        for stat_name in signal_stats:
            # Query ETI stats (from true_only_eti and true_eti_rfi) in a single call
            results = self.db.query_injection_stat(
                stat_name=f"eti_{stat_name}",
                signal_type=["true_only_eti", "true_eti_rfi"],
                tag=self.config.checkpoint.save_tag,
                start_time=self.start_time,
                end_time=current_time,
                columns=["value"],
            )
            eti_stats[stat_name] = [r["value"] for r in results]
            del results

            # Query RFI stats (from false_with_rfi and true_eti_rfi) in a single call
            results = self.db.query_injection_stat(
                stat_name=f"rfi_{stat_name}",
                signal_type=["false_with_rfi", "true_eti_rfi"],
                tag=self.config.checkpoint.save_tag,
                start_time=self.start_time,
                end_time=current_time,
                columns=["value"],
            )
            rfi_stats[stat_name] = [r["value"] for r in results]
            del results

        # Query background_index values for ETI and RFI signal types in single calls
        results = self.db.query_injection_stat(
            stat_name="global_mean",  # Any stat works here. Select "mean" to reduce rows queried
            injection_stage="A",  # Any stage works here. Select "A" to reduce rows queried
            signal_type=["true_only_eti", "true_eti_rfi"],
            tag=self.config.checkpoint.save_tag,
            start_time=self.start_time,
            end_time=current_time,
            columns=["background_index"],
        )
        eti_background_indices = [
            r["background_index"] for r in results if r["background_index"] is not None
        ]
        del results

        results = self.db.query_injection_stat(
            stat_name="global_mean",  # Any stat works here. Select "mean" to reduce rows queried
            injection_stage="A",  # Any stage works here. Select "A" to reduce rows queried
            signal_type=["false_with_rfi", "true_eti_rfi"],
            tag=self.config.checkpoint.save_tag,
            start_time=self.start_time,
            end_time=current_time,
            columns=["background_index"],
        )
        rfi_background_indices = [
            r["background_index"] for r in results if r["background_index"] is not None
        ]
        del results

        save_path = os.path.join(save_dir, f"injected_signal_characteristics_{tag}.png")
        self._plot_injected_signal_characteristics(
            eti_stats,
            rfi_stats,
            eti_background_indices,
            rfi_background_indices,
            tag,
            machine_name,
            save_path,
        )

        del eti_stats, rfi_stats, eti_background_indices, rfi_background_indices
        gc.collect()

        # Figure 2: Injection stability metrics
        # Compute sanitization rates per stat using SQL-level aggregation
        intensity_stats = [
            "global_mean",
            "global_median",
            "global_std",
            "global_mad",
            "global_skew",
            "global_kurtosis",
        ]
        sanitization_rates_by_stat: dict[str, dict[int, float]] = {s: {} for s in intensity_stats}

        for stat_name in intensity_stats:
            agg_results = self.db.query_injection_stat_stability(
                stat_name=stat_name,
                injection_stage="A",  # NOTE: why are we only computing for A? only works if we assume sanitization happens evenly for all stages per cadence (is this always true?)
                tag=self.config.checkpoint.save_tag,
                start_time=self.start_time,
                end_time=current_time,
            )
            for row in agg_results:
                round_num = row["round_number"]
                total = row["total_count"]
                non_finite = row["non_finite_count"]
                if round_num is not None:
                    sanitization_rates_by_stat[stat_name][round_num] = (
                        non_finite / total if total > 0 else 0.0
                    )
            del agg_results

        # Compute clamping rate per round using SQL-level aggregation
        clamping_rates_by_round: dict[int, float] = {}
        clamping_results = self.db.query_injection_stat_stability(
            stat_name="global_mean",  # Slope is the same for all stats. Use "global_mean" to reduce rows queried
            injection_stage="A",  # Slope is the same for all stages. Use "A" to reduce rows queried
            tag=self.config.checkpoint.save_tag,
            start_time=self.start_time,
            end_time=current_time,
        )
        for row in clamping_results:
            round_num = row["round_number"]
            total = row["total_count"]
            clamped = row["clamped_count"]
            if round_num is not None:
                clamping_rates_by_round[round_num] = clamped / total if total > 0 else 0.0
        del clamping_results

        save_path = os.path.join(save_dir, f"injection_stability_{tag}.png")
        self._plot_injection_stability(
            sanitization_rates_by_stat,
            clamping_rates_by_round,
            tag,
            machine_name,
            save_path,
        )

        del sanitization_rates_by_stat, clamping_rates_by_round
        gc.collect()

        # Figure 3-6: Global intensity distribution (one per signal_type)
        signal_types = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]
        intensity_stats = [
            "global_mean",
            "global_median",
            "global_std",
            "global_mad",
            "global_skew",
            "global_kurtosis",
        ]
        stages = ["A", "B", "C"]

        for signal_type in signal_types:
            stats_by_stage = {stage: {} for stage in stages}

            for stat_name in intensity_stats:
                for stage in stages:
                    results = self.db.query_injection_stat(
                        stat_name=stat_name,
                        injection_stage=stage,
                        signal_type=signal_type,
                        tag=self.config.checkpoint.save_tag,
                        start_time=self.start_time,
                        end_time=current_time,
                        columns=["value"],
                    )
                    stats_by_stage[stage][stat_name] = [r["value"] for r in results]
                    del results

            # Generate plot for this signal_type
            save_path = os.path.join(
                save_dir, f"{signal_type}_global_intensity_distributions_{tag}.png"
            )
            self._plot_global_intensity_distributions(
                stats_by_stage, signal_type, tag, machine_name, save_path
            )

            del stats_by_stage
            gc.collect()

        # Figure 7: A->B global intensity biases
        transitions = {stat_name: {} for stat_name in intensity_stats}

        for stat_name in intensity_stats:
            for signal_type in signal_types:
                # Query stage A values
                results_a = self.db.query_injection_stat(
                    stat_name=stat_name,
                    injection_stage="A",
                    signal_type=signal_type,
                    tag=self.config.checkpoint.save_tag,
                    start_time=self.start_time,
                    end_time=current_time,
                    columns=["value"],
                )
                values_a = [r["value"] for r in results_a]
                del results_a

                # Query stage B values
                results_b = self.db.query_injection_stat(
                    stat_name=stat_name,
                    injection_stage="B",
                    signal_type=signal_type,
                    tag=self.config.checkpoint.save_tag,
                    start_time=self.start_time,
                    end_time=current_time,
                    columns=["value"],
                )
                values_b = [r["value"] for r in results_b]
                del results_b

                transitions[stat_name][signal_type] = (values_a, values_b)

        save_path = os.path.join(save_dir, f"a_b_global_intensity_biases_{tag}.png")
        self._plot_injection_intensity_biases(transitions, tag, machine_name, save_path)

        del transitions
        gc.collect()

        # Figure 8: Final global intensity biases
        stats_by_type = {signal_type: {} for signal_type in signal_types}

        for signal_type in signal_types:
            for stat_name in intensity_stats:
                results = self.db.query_injection_stat(
                    stat_name=stat_name,
                    injection_stage="C",
                    signal_type=signal_type,
                    tag=self.config.checkpoint.save_tag,
                    start_time=self.start_time,
                    end_time=current_time,
                    columns=["value"],
                )
                stats_by_type[signal_type][stat_name] = [r["value"] for r in results]
                del results

        save_path = os.path.join(save_dir, f"final_global_intensity_biases_{tag}.png")
        self._plot_final_intensity_biases(stats_by_type, tag, machine_name, save_path)

        del stats_by_type
        gc.collect()

        logger.info(f"Injection stats plots saved to: {save_dir}")

    def _plot_injected_signal_characteristics(
        self,
        eti_stats: dict[str, list[float]],
        rfi_stats: dict[str, list[float]],
        eti_background_indices: list[int],
        rfi_background_indices: list[int],
        tag: str,
        machine_name: str,
        save_path: str,
    ) -> None:
        """Generate signal characteristics grid with GridSpec layout."""
        signal_stats = [
            "snr",
            "drift_rate",
            "signal_width",
            "starting_bin",
            "slope_pixel",
            "y_intercept",
        ]
        # NOTE: are these units correct?
        stat_display_names = {
            "snr": "SNR",
            "drift_rate": "Drift Rate (Hz/s)",
            "signal_width": "Signal Width (Hz)",
            "starting_bin": "Starting Bin",
            "slope_pixel": "Slope (px)",
            "y_intercept": "Y-Intercept",
        }
        signal_colors = {"ETI": "blue", "RFI": "orange"}

        # Use GridSpec for flexible layout
        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

        fig.suptitle(
            f"Injected Signal Characteristics ({tag}, {machine_name})",
            fontsize=16,
            fontweight="bold",
        )

        # 6 signal stats in top 2 rows
        for idx, stat_name in enumerate(signal_stats):
            row, col = idx // 3, idx % 3
            ax = fig.add_subplot(gs[row, col])

            eti_data = eti_stats.get(stat_name, [])
            rfi_data = rfi_stats.get(stat_name, [])

            if eti_data:
                ax.hist(
                    eti_data,
                    bins=50,
                    alpha=0.2,
                    color=signal_colors["ETI"],
                    edgecolor=signal_colors["ETI"],
                    linewidth=2,
                    histtype="stepfilled",
                )
            if rfi_data:
                ax.hist(
                    rfi_data,
                    bins=50,
                    alpha=0.2,
                    color=signal_colors["RFI"],
                    edgecolor=signal_colors["RFI"],
                    linewidth=2,
                    histtype="stepfilled",
                )

            ax.set_title(
                stat_display_names.get(stat_name, stat_name), fontsize=12, fontweight="bold"
            )
            ax.set_ylabel("Count", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Background plates histogram spanning full bottom row
        ax_bg = fig.add_subplot(gs[2, :])  # Spans all 3 columns

        # Plot overlapping ETI and RFI background indices
        all_bg_indices = eti_background_indices + rfi_background_indices
        if all_bg_indices:
            max_bg = max(all_bg_indices)

            # Create bins in units of 1000 (indices 0-999 = bin 1, 1000-1999 = bin 2, etc.)
            bin_size = 1000
            num_bins = (max_bg // bin_size) + 1
            bins = [i * bin_size for i in range(num_bins + 1)]  # [0, 1000, 2000, ...]

            if eti_background_indices:
                ax_bg.hist(
                    eti_background_indices,
                    bins=bins,
                    alpha=0.2,
                    color=signal_colors["ETI"],
                    edgecolor=signal_colors["ETI"],
                    linewidth=2,
                    histtype="stepfilled",
                    label="ETI",
                )
            if rfi_background_indices:
                ax_bg.hist(
                    rfi_background_indices,
                    bins=bins,
                    alpha=0.2,
                    color=signal_colors["RFI"],
                    edgecolor=signal_colors["RFI"],
                    linewidth=2,
                    histtype="stepfilled",
                    label="RFI",
                )

        ax_bg.set_title("Background Plates", fontsize=12, fontweight="bold")
        ax_bg.set_ylabel("Count", fontsize=10)
        ax_bg.grid(True, alpha=0.3)

        # Create legend handles for figure-level legend
        legend_handles = [
            mlines.Line2D([], [], color=signal_colors["ETI"], linewidth=4, alpha=0.5, label="ETI"),
            mlines.Line2D([], [], color=signal_colors["RFI"], linewidth=4, alpha=0.5, label="RFI"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        plt.tight_layout(rect=[0, 0, 0.92, 1])  # Leave room for legend on right
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Injected signal characteristics plot saved: {save_path}")

        # Upload to Slack
        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"Injected signal characteristics - ({tag}, {machine_name})",
            )

    def _plot_injection_stability(
        self,
        sanitization_rates_by_stat: dict[str, dict[int, float]],
        clamping_rates_by_round: dict[int, float],
        tag: str,
        machine_name: str,
        save_path: str,
    ) -> None:
        """Plot injection stability metrics: sanitization rate and clamping rate."""
        current_time = time.time()

        intensity_stats = [
            "global_mean",
            "global_median",
            "global_std",
            "global_mad",
            "global_skew",
            "global_kurtosis",
        ]
        stat_colors = {
            "global_mean": "blue",
            "global_median": "orange",
            "global_std": "green",
            "global_mad": "red",
            "global_skew": "purple",
            "global_kurtosis": "pink",
        }
        stat_display_names = {
            "global_mean": "Mean",
            "global_median": "Median",
            "global_std": "Std Dev",
            "global_mad": "MAD",
            "global_skew": "Skewness",
            "global_kurtosis": "Kurtosis",
        }

        # Add SNR range shading
        snr_by_round = self._get_snr_by_round(current_time)

        # Scale figure width for many rounds
        num_rounds = len(snr_by_round)
        base_width = 15
        fig_width = base_width * (1 + max(0, num_rounds - 10) * 0.05)  # +5% width per round over 10

        # Create figure with GridSpec layout
        fig = plt.figure(figsize=(fig_width, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

        ax_sanitization = fig.add_subplot(gs[0])
        ax_clamping = fig.add_subplot(gs[1])

        fig.suptitle(
            f"Injection Stability ({tag}, {machine_name})",
            fontsize=16,
            fontweight="bold",
        )

        # Add SNR range shading to both axes
        for ax in [ax_sanitization, ax_clamping]:
            self._add_snr_range_shading(ax, snr_by_round, use_rounds=True)

        # Top plot: Sanitization rate per statistic (grouped bar chart)
        rounds = sorted(
            set().union(*[set(sanitization_rates_by_stat[s].keys()) for s in intensity_stats])
        )
        n_rounds = len(rounds)
        n_stats = len(intensity_stats)

        if n_rounds > 0:
            bar_width = 0.8 / n_stats
            x_positions = np.array(rounds)

            for stat_idx, stat_name in enumerate(intensity_stats):
                rates = sanitization_rates_by_stat[stat_name]
                values = [rates.get(r, 0) for r in rounds]
                offset = (stat_idx - (n_stats - 1) / 2) * bar_width

                ax_sanitization.bar(
                    x_positions + offset,
                    values,
                    width=bar_width,
                    color=stat_colors[stat_name],
                    edgecolor="black",
                    linewidth=0.5,
                )

            ax_sanitization.set_xticks(rounds)

        ax_sanitization.set_title("NaN/Inf Sanitization Rate", fontsize=14, fontweight="bold")
        # ax_sanitization.set_xlabel("Round", fontsize=12, fontweight="bold")
        ax_sanitization.grid(True, alpha=0.3, axis="y")
        ax_sanitization.set_ylim(bottom=0)

        # Bottom plot: Slope clamping rate (single bar chart)
        if clamping_rates_by_round:
            rounds_clamping = sorted(clamping_rates_by_round.keys())
            values = [clamping_rates_by_round[r] for r in rounds_clamping]

            ax_clamping.bar(
                rounds_clamping,
                values,
                width=0.6,
                color="blue",
                edgecolor="black",
                linewidth=0.5,
            )
            ax_clamping.set_xticks(rounds_clamping)

        ax_clamping.set_title("Slope Clamping Rate", fontsize=14, fontweight="bold")
        ax_clamping.set_xlabel("Round", fontsize=12, fontweight="bold")
        ax_clamping.grid(True, alpha=0.3, axis="y")
        ax_clamping.set_ylim(bottom=0)

        # Create unified legend using patches for bar charts
        legend_handles = [
            mpatches.Patch(
                facecolor=stat_colors[stat_name],
                edgecolor="black",
                linewidth=0.5,
                label=stat_display_names[stat_name],
            )
            for stat_name in intensity_stats
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Injection stability plot saved: {save_path}")

        # Upload to Slack
        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"Injection stability - ({tag}, {machine_name})",
            )

        del snr_by_round

    def _plot_global_intensity_distributions(
        self,
        stats_by_stage: dict[str, dict[str, list[float]]],
        signal_type: str,
        tag: str,
        machine_name: str,
        save_path: str,
    ) -> None:
        """Generate 2x3 intensity histogram grid for one signal_type."""
        intensity_stats = [
            "global_mean",
            "global_median",
            "global_std",
            "global_mad",
            "global_skew",
            "global_kurtosis",
        ]
        stat_display_names = {
            "global_mean": "Mean",
            "global_median": "Median",
            "global_std": "Std Dev",
            "global_mad": "MAD",
            "global_skew": "Skewness",
            "global_kurtosis": "Kurtosis",
        }
        stage_colors = {"A": "blue", "B": "orange", "C": "green"}

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"{signal_type} Global Intensities ({tag}, {machine_name})",
            fontsize=16,
            fontweight="bold",
        )

        for idx, stat_name in enumerate(intensity_stats):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]

            # Plot stages A and B on primary axis (pre-normalization scale)
            for stage in ["A", "B"]:
                data = stats_by_stage[stage].get(stat_name, [])
                if data:
                    ax.hist(
                        data,
                        bins=50,
                        alpha=0.2,
                        color=stage_colors[stage],
                        edgecolor=stage_colors[stage],
                        linewidth=2,
                        histtype="stepfilled",
                    )

            # Create twin x-axis for stage C (post-normalization scale [0,1])
            ax2 = ax.twiny()
            data_c = stats_by_stage["C"].get(stat_name, [])
            if data_c:
                ax2.hist(
                    data_c,
                    bins=50,
                    alpha=0.2,
                    color=stage_colors["C"],
                    edgecolor=stage_colors["C"],
                    linewidth=2,
                    histtype="stepfilled",
                )

            ax.set_title(
                stat_display_names.get(stat_name, stat_name), fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Pre-norm (A, B)", fontsize=9, color="darkblue")
            ax.set_ylabel("Count", fontsize=10)
            ax.tick_params(axis="x", colors="darkblue")
            ax2.set_xlabel("Post-norm (C)", fontsize=9, color="darkgreen")
            ax2.tick_params(axis="x", colors="darkgreen")
            ax.grid(True, alpha=0.3)

        # Create legend handles for figure-level legend
        legend_handles = [
            mlines.Line2D(
                [], [], color=stage_colors["A"], linewidth=4, alpha=0.5, label="Stage A (pre-inj)"
            ),
            mlines.Line2D(
                [], [], color=stage_colors["B"], linewidth=4, alpha=0.5, label="Stage B (post-inj)"
            ),
            mlines.Line2D(
                [], [], color=stage_colors["C"], linewidth=4, alpha=0.5, label="Stage C (post-norm)"
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        plt.tight_layout(rect=[0, 0, 0.88, 1])  # Leave room for legend on right
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"{signal_type} global intensity distributions plot saved: {save_path}")

        # Upload to Slack
        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"{signal_type} global intensity distributions - ({tag}, {machine_name})",
            )

    def _plot_injection_intensity_biases(
        self,
        transitions: dict[str, dict[str, tuple[list, list]]],
        tag: str,
        machine_name: str,
        save_path: str,
    ) -> None:
        """
        Generate 2x3 scatter plot grid showing A→B transitions.

        Uses subsampling by absolute distance (grouped by stat_name, signal_type) to avoid rendering
        all individual points via ax.scatter().
        Specifically, plots will include all points beyond outlier_pct. If num_points < max_points,
        we randomly sample without replacement from the remaining points to make up the difference
        """
        intensity_stats = [
            "global_mean",
            "global_median",
            "global_std",
            "global_mad",
            "global_skew",
            "global_kurtosis",
        ]
        stat_display_names = {
            "global_mean": "Mean",
            "global_median": "Median",
            "global_std": "Std Dev",
            "global_mad": "MAD",
            "global_skew": "Skewness",
            "global_kurtosis": "Kurtosis",
        }
        signal_types = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]
        type_colors = {
            "false_no_signal": "blue",
            "false_with_rfi": "orange",
            "true_only_eti": "green",
            "true_eti_rfi": "red",
        }
        type_display_names = {
            "false_no_signal": "No Signal",
            "false_with_rfi": "RFI Only",
            "true_only_eti": "ETI Only",
            "true_eti_rfi": "ETI + RFI",
        }

        max_points = self.config.training.plot_injection_subsampling_count
        outlier_pct = self.config.training.plot_injection_outlier_percentile

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"A→B Global Intensity Biases — Subsampled {max_points} pts, {outlier_pct} pct ({tag}, {machine_name})",
            fontsize=16,
            fontweight="bold",
        )

        for idx, stat_name in enumerate(intensity_stats):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]

            min_val, max_val = np.inf, -np.inf

            for signal_type in signal_types:
                values_a, values_b = transitions[stat_name].get(signal_type, ([], []))
                if values_a and values_b:
                    # Ensure equal length (take minimum)
                    min_len = min(len(values_a), len(values_b))
                    va = np.array(values_a[:min_len])
                    vb = np.array(values_b[:min_len])

                    # Subsample if exceeding max points per series
                    if len(va) > max_points:
                        distances = np.abs(va - vb)
                        threshold = np.percentile(distances, outlier_pct)
                        outlier_mask = distances >= threshold
                        normal_mask = ~outlier_mask

                        # Always keep outliers
                        outlier_indices = np.where(outlier_mask)[0]
                        normal_indices = np.where(normal_mask)[0]

                        # Subsample normal points to fill remaining budget
                        remaining = max(0, max_points - len(outlier_indices))
                        if remaining < len(normal_indices):
                            sampled_normal = np.random.choice(
                                normal_indices, size=remaining, replace=False
                            )
                            keep_indices = np.concatenate([outlier_indices, sampled_normal])
                        else:
                            keep_indices = np.arange(len(va))

                        va = va[keep_indices]
                        vb = vb[keep_indices]

                    min_val = min(min_val, va.min(), vb.min())
                    max_val = max(max_val, va.max(), vb.max())

                    ax.scatter(
                        va,
                        vb,
                        alpha=0.12,
                        facecolor=type_colors[signal_type],
                        edgecolor=type_colors[signal_type],
                        linewidth=0.3,
                        s=12,
                    )

            # Add diagonal reference line
            if min_val < np.inf and max_val > -np.inf:
                ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1)

            ax.set_title(
                stat_display_names.get(stat_name, stat_name), fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Stage A (pre-inj)", fontsize=10)
            ax.set_ylabel("Stage B (post-inj)", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Create legend handles for figure-level legend
        legend_handles = [
            mlines.Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor=type_colors[st],
                markersize=8,
                alpha=0.7,
                label=type_display_names[st],
            )
            for st in signal_types
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            fontsize=9,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        plt.tight_layout(rect=[0, 0, 0.88, 1])  # Leave room for legend on right
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"A→B global intensity biases plot saved: {save_path}")

        # Upload to Slack
        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"A→B global intensity biases - ({tag}, {machine_name})",
            )

    def _plot_final_intensity_biases(
        self,
        stats_by_type: dict[str, dict[str, list[float]]],
        tag: str,
        machine_name: str,
        save_path: str,
    ) -> None:
        """Generate 2x3 box plot grid comparing signal types at Stage C."""
        intensity_stats = [
            "global_mean",
            "global_median",
            "global_std",
            "global_mad",
            "global_skew",
            "global_kurtosis",
        ]
        stat_display_names = {
            "global_mean": "Mean",
            "global_median": "Median",
            "global_std": "Std Dev",
            "global_mad": "MAD",
            "global_skew": "Skewness",
            "global_kurtosis": "Kurtosis",
        }
        signal_types = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]
        type_colors = {
            "false_no_signal": "blue",
            "false_with_rfi": "orange",
            "true_only_eti": "green",
            "true_eti_rfi": "red",
        }
        type_display_names = {
            "false_no_signal": "No Signal",
            "false_with_rfi": "RFI Only",
            "true_only_eti": "ETI Only",
            "true_eti_rfi": "ETI + RFI",
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"Final Global Intensity Biases ({tag}, {machine_name})",
            fontsize=16,
            fontweight="bold",
        )

        for idx, stat_name in enumerate(intensity_stats):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]

            box_data = []
            for signal_type in signal_types:
                data = stats_by_type[signal_type].get(stat_name, [])
                box_data.append(data if data else [0])  # Use [0] if empty to avoid error

            # Horizontal box plots with colors
            bp = ax.boxplot(
                box_data,
                vert=False,
                patch_artist=True,
                tick_labels=[""] * len(signal_types),  # No y-axis labels, use legend instead
            )

            # Color each box
            for patch, signal_type in zip(bp["boxes"], signal_types, strict=True):
                patch.set_facecolor(type_colors[signal_type])
                patch.set_alpha(0.6)

            ax.set_title(
                stat_display_names.get(stat_name, stat_name), fontsize=12, fontweight="bold"
            )
            ax.grid(True, alpha=0.3, axis="x")

        # Create legend handles for figure-level legend
        legend_handles = [
            mpatches.Patch(facecolor=type_colors[st], alpha=0.6, label=type_display_names[st])
            for st in signal_types
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            fontsize=9,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        plt.tight_layout(rect=[0, 0, 0.88, 1])  # Leave room for legend on right
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Final global intensity biases plot saved: {save_path}")

        # Upload to Slack
        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"Final global intensity biases - ({tag}, {machine_name})",
            )

    # NOTE: come back to this later (verify what happens when self._latent_viz_batch and/or self._latent_viz_labels is None)
    # TODO: reorder plot methods (def & call sites): train -> latent -> injection
    # NOTE: combine plot_beta_vae_loss_curves(), plot_beta_vae_training_stability(), and plot_latent_space_gif() into plot_training_progress()?
    def plot_latent_space_gif(self, tag: str | None = None, dir: str | None = None):
        """
        Generate GIF showing how the latent space evolves during training using using UMAP with 8
        color categories (4 signal types * ON/OFF).

        Args:
            tag: Plot tag for filename (defaults to save_tag)
            dir: Subdirectory (e.g., "checkpoints" for per-round)
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        if dir is not None:
            save_dir = os.path.join(self.config.output_path, "plots", dir)
        else:
            save_dir = os.path.join(self.config.output_path, "plots")

        os.makedirs(save_dir, exist_ok=True)  # Create dir if it doesn't exist

        if self.db is None:
            raise RuntimeError("No database instance detected - cannot generate latent space GIF")

        # Flush database to ensure all latent snapshots are written before plotting
        logger.info("Flushing database before latent space GIF generation...")
        if not self.db.flush():
            logger.warning(
                "Database flush failed. GIF generation may encounter issues. Proceeding anyways..."
            )
        else:
            logger.info("Database flushed")

        # Query distinct snapshot keys (model, round, epoch, step, snr_base, snr_range)
        # Sorting already handled on the query side (i.e. in query_latent_snapshot_keys)
        snapshot_keys = self.db.query_latent_snapshot_keys(
            tag=self.config.checkpoint.save_tag,
            start_time=self.start_time,
        )

        if not snapshot_keys:
            logger.warning("No latent snapshots found in database — skipping GIF generation")
            return

        logger.info(f"Found {len(snapshot_keys)} unique snapshots for GIF generation")

        # Subsample to max_frames if too many snapshots
        # Uses log-spaced indices so earlier training steps (where the most dramatic
        # latent space changes occur) get higher frame density than later steps
        # Only works if snapshot_keys is sorted by progression
        max_frames = self.config.training.latent_viz_gif_max_frames
        if len(snapshot_keys) > max_frames:
            n = len(snapshot_keys)
            # Generate more candidates than needed, then deduplicate to hit max_frames
            # Log-spacing naturally clusters near index 0, so oversampling + dedup
            # preserves the early-training bias while filling the frame budget
            oversample = max_frames
            indices = set()
            indices.add(0)  # Always include first snapshot
            while len(indices) < max_frames:
                oversample *= 2  # NOTE: is 2 a good choice here?
                raw = np.logspace(0, np.log10(n - 1), num=oversample).astype(int)
                indices = {0} | set(raw)
                del raw
                if oversample > n * 10:
                    break  # Safety valve: can't exceed unique indices available
            # Sort and trim to exactly max_frames
            indices = sorted(indices)[:max_frames]
            snapshot_keys = [snapshot_keys[i] for i in indices]

            logger.info(
                f"Subsampled to {len(snapshot_keys)} frames (log-spaced, max {max_frames}, oversample {oversample})"
            )

            del indices
            gc.collect()

        # Load selected snapshots and pool latent data
        all_coords = []  # List of (N_obs, latent_dim) arrays per snapshot
        all_snapshot_labels = []  # List of (N_obs,) label arrays per snapshot
        all_snapshot_onoff = []  # List of (N_obs,) ON/OFF arrays per snapshot
        snapshot_metadata = []  # (model, round, epoch, step, snr_base, snr_range) per snapshot

        for key in snapshot_keys:
            rows = self.db.query_latent_snapshots(
                model_name=key["model_name"],
                round_number=key["round_number"],
                epoch_number=key["epoch_number"],
                step_number=key["step_number"],
                tag=self.config.checkpoint.save_tag,
                start_time=self.start_time,
                columns=["signal_type", "latent_vector"],
            )

            if not rows:
                continue

            # Parse latent vectors and build arrays
            snapshot_latents = []
            snapshot_labels = []
            snapshot_onoff = []

            for row in rows:
                latent_6x = json.loads(row["latent_vector"])  # (6, latent_dim)
                for obs_idx, vec in enumerate(latent_6x):
                    snapshot_latents.append(vec)
                    snapshot_labels.append(row["signal_type"])
                    snapshot_onoff.append("ON" if obs_idx % 2 == 0 else "OFF")

            all_coords.append(np.array(snapshot_latents, dtype=np.float32))
            all_snapshot_labels.append(snapshot_labels)
            all_snapshot_onoff.append(snapshot_onoff)
            snapshot_metadata.append(key)

            del rows, snapshot_latents, snapshot_labels, snapshot_onoff
            gc.collect()

        del snapshot_keys
        gc.collect()

        if not all_coords:
            logger.warning("No valid latent data loaded — skipping GIF generation")
            return

        # Pool all latents for global UMAP fit
        pooled = np.concatenate(all_coords, axis=0)
        logger.info(
            f"Pooled {pooled.shape[0]} latent vectors from {len(all_coords)} snapshots "
            f"for UMAP fitting"
        )

        # Subsample pooled vectors for UMAP fit (fitting on the full set of pooled vectors is slow;
        # the subsampled fit generalizes well and remaining vectors are projected via .transform())
        # Stratified by signal_type × ON/OFF (8 classes) for balanced representation
        umap_fit_max = self.config.training.latent_viz_umap_fit_max_samples
        if pooled.shape[0] > umap_fit_max:
            pooled_labels = np.concatenate(
                [np.array(lab, dtype="U") for lab in all_snapshot_labels]
            )
            pooled_onoff = np.concatenate([np.array(o, dtype="U") for o in all_snapshot_onoff])
            strata = np.char.add(np.char.add(pooled_labels, "|"), pooled_onoff)
            del pooled_labels, pooled_onoff

            # NOTE: use a global config seed instead of hard-coding
            rng = np.random.default_rng(11)
            unique_classes = np.unique(strata)
            per_class = umap_fit_max // len(unique_classes)
            fit_indices = []
            for cls in unique_classes:
                cls_idx = np.nonzero(strata == cls)[0]
                n_take = min(per_class, len(cls_idx))
                if n_take < per_class:
                    logger.warning(
                        f"Only {n_take} latents for {cls} "
                        f"(requested {per_class}), using all available"
                    )
                fit_indices.append(rng.choice(cls_idx, size=n_take, replace=False))
            fit_indices = np.concatenate(fit_indices)
            del strata

            fit_pool = pooled[fit_indices]
            logger.info(
                f"Stratified subsampled {fit_pool.shape[0]} / {pooled.shape[0]} "
                f"latent vectors for UMAP fit ({len(unique_classes)} classes, "
                f"~{per_class} per class)"
            )
            del fit_indices
        else:
            fit_pool = pooled

        del pooled

        # Compute consistent axis limits with 5% padding (streaming min/max to avoid concat)
        def _compute_limits(transformed_list):
            x_min = min(t[:, 0].min() for t in transformed_list)
            x_max = max(t[:, 0].max() for t in transformed_list)
            y_min = min(t[:, 1].min() for t in transformed_list)
            y_max = max(t[:, 1].max() for t in transformed_list)
            x_pad = (x_max - x_min) * 0.05
            y_pad = (y_max - y_min) * 0.05
            return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)

        # Generate frames and assemble GIF
        colors = {
            ("false_no_signal", "ON"): "#1565C0",
            ("false_no_signal", "OFF"): "#64B5F6",
            ("false_with_rfi", "ON"): "#F9A825",
            ("false_with_rfi", "OFF"): "#FFF176",
            ("true_only_eti", "ON"): "#2E7D32",
            ("true_only_eti", "OFF"): "#81C784",
            ("true_eti_rfi", "ON"): "#C62828",
            ("true_eti_rfi", "OFF"): "#EF5350",
        }
        markers = {"ON": "o", "OFF": "x"}
        display_names = {
            ("false_no_signal", "ON"): "No Signal (ON)",
            ("false_no_signal", "OFF"): "No Signal (OFF)",
            ("false_with_rfi", "ON"): "RFI Only (ON)",
            ("false_with_rfi", "OFF"): "RFI Only (OFF)",
            ("true_only_eti", "ON"): "ETI Only (ON)",
            ("true_only_eti", "OFF"): "ETI Only (OFF)",
            ("true_eti_rfi", "ON"): "ETI+RFI (ON)",
            ("true_eti_rfi", "OFF"): "ETI+RFI (OFF)",
        }

        # NOTE: instead of temp_dir, save frames in persistent dir. update dir archiving to handle
        temp_dir = tempfile.mkdtemp(prefix="latent_gif_")

        gif_paths = {}
        duration_ms = self.config.training.latent_viz_gif_duration_ms

        # NOTE: how do we store final UMAP model params for later use (e.g. during inference results viz)
        n_neighbors_values = self.config.training.latent_viz_umap_n_neighbors
        min_dist_values = self.config.training.latent_viz_umap_min_dist

        for nn in n_neighbors_values:
            for md in min_dist_values:
                logger.info(f"Fitting UMAP with n_neighbors={nn}, min_dist={md}")

                # NOTE: use a global config seed instead of hard-coding
                # Fit UMAP model
                # Note that by setting random_state, we get a deterministic UMAP fit, at the expense
                # of single-thread performance (n_jobs=1). This is a hard constraint of the UMAP
                # library. We compensate by fitting the UMAP model to a stratified subsample of the
                # pooled latents
                umap_model = umap.UMAP(
                    n_components=2,
                    random_state=11,
                    n_neighbors=nn,
                    min_dist=md,
                ).fit(fit_pool)

                # Transform each snapshot
                transformed = []
                for coords in all_coords:
                    transformed.append(umap_model.transform(coords))
                del umap_model
                gc.collect()

                # Compute global axis limits
                xlim, ylim = _compute_limits(transformed)

                method_name = f"umap_nn{nn}_md{md}"
                display_method = f"UMAP (n_neighbors={nn}, min_dist={md})"

                methods = [
                    (method_name, display_method, transformed, xlim, ylim),
                ]

                for method_name, display_method, transformed_list, xlim, ylim in methods:
                    frame_paths = []

                    for frame_idx, (coords_2d, labels, onoff, meta) in enumerate(
                        zip(
                            transformed_list,
                            all_snapshot_labels,
                            all_snapshot_onoff,
                            snapshot_metadata,
                            strict=True,
                        )
                    ):
                        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

                        # Plot each category
                        labels_arr = np.array(labels)
                        onoff_arr = np.array(onoff)

                        for (stype, status), color in colors.items():
                            mask = (labels_arr == stype) & (onoff_arr == status)
                            if mask.any():
                                ax.scatter(
                                    coords_2d[mask, 0],
                                    coords_2d[mask, 1],
                                    c=color,
                                    marker=markers[status],
                                    s=5,
                                    label=display_names[(stype, status)],
                                    rasterized=True,
                                )

                        del labels_arr, onoff_arr

                        ax.set_xlim(xlim)
                        ax.set_ylim(ylim)

                        meta_snr_base = meta["snr_base"]
                        meta_snr_range = meta["snr_range"]
                        meta_snr_floor = meta_snr_base if meta_snr_base is not None else "?"
                        meta_snr_ceil = (
                            meta_snr_base + meta_snr_range
                            if meta_snr_base is not None and meta_snr_range is not None
                            else "?"
                        )
                        ax.set_title(
                            f"Beta-VAE Latent Space ({display_method}) — "
                            f"Round {meta['round_number']}, "
                            f"Epoch {meta['epoch_number']}, "
                            f"Step {meta['step_number']} "
                            f"(SNR: {meta_snr_floor}–{meta_snr_ceil})",
                            fontsize=11,
                        )
                        ax.set_xlabel("UMAP 1")
                        ax.set_ylabel("UMAP 2")
                        ax.legend(
                            loc="upper right",
                            fontsize=7,
                            markerscale=3,
                            ncol=2,
                            framealpha=0.8,
                        )

                        plt.tight_layout()

                        frame_path = os.path.join(
                            temp_dir, f"{method_name}_frame_{frame_idx:05d}.png"
                        )
                        fig.savefig(frame_path, dpi=100)
                        plt.close(fig)
                        frame_paths.append(frame_path)

                    del transformed_list

                    # Assemble GIF by streaming one frame at a time via imageio (reduces memory pressure)
                    gif_filename = f"latent_space_{method_name}_{tag}.gif"
                    gif_path = os.path.join(save_dir, gif_filename)

                    n_frames = len(frame_paths)
                    if n_frames > 0:
                        with iio.imopen(gif_path, "w", plugin="pillow") as gif_writer:
                            for frame_path in frame_paths:
                                frame = iio.imread(frame_path)
                                gif_writer.write(
                                    frame,
                                    duration=duration_ms,
                                    loop=0,
                                    is_batch=False,
                                )
                                del frame

                        logger.info(
                            f"Latent space {method_name.upper()} GIF saved: "
                            f"{gif_path} ({n_frames} frames)"
                        )

                        # Upload to Slack
                        logger_instance = get_logger()
                        if logger_instance:
                            logger_instance.upload_image_to_slack(
                                gif_path,
                                title=f"Latent Space {display_method} - ({tag})",
                            )

                    del frame_paths
                    gc.collect()

                    gif_paths[method_name] = gif_path

                del transformed
                gc.collect()

        # Cleanup
        del fit_pool, all_coords
        # del umap_transformed
        del all_snapshot_labels, all_snapshot_onoff, snapshot_metadata
        # NOTE: temp_dir isn't cleaned on exception (should use try/finally or tempfile.TemporaryDirectory() with context manager)
        shutil.rmtree(temp_dir, ignore_errors=True)
        gc.collect()

    def _prepare_latent_viz_batch(self, concat_data, labels, candidate_indices=None):
        """
        Subsample cadences from concat_data for latent space visualization
        Will attempt to preserve an equal distribution of distinct values from labels if possible

        Called once on the first round's stratified validation partition and persisted across
        subsequent rounds. Using held-out data ensures the latent space visualization captures
        generalization, while persisting the same data across rounds eliminates the effects of
        distribution shift (from the curriculum schedule)

        Args:
            concat_data: Full cadences array, shape (n_total, 6, 16, width_bin)
            labels: Per-cadence signal type labels, shape (n_total,)
            candidate_indices: Optional indices into concat_data/labels restricting which
                samples are eligible (e.g. validation partition indices). If None, all
                samples are eligible.
        """
        n_per_type = self.config.training.latent_viz_num_cadences_per_type
        signal_types = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]

        # Restrict to candidate subset if provided (avoids copying the entire partition)
        if candidate_indices is not None:
            candidate_labels = labels[candidate_indices]
        else:
            candidate_indices = np.arange(len(labels))
            candidate_labels = labels

        selected_indices = []
        selected_labels = []

        for stype in signal_types:
            type_mask = candidate_labels == stype
            type_global_indices = candidate_indices[type_mask]
            if len(type_global_indices) == 0:
                logger.warning(f"No cadences for {stype} — skipping this type for viz batch")
                continue
            if len(type_global_indices) < n_per_type:
                logger.warning(
                    f"Only {len(type_global_indices)} cadences for {stype} "
                    f"(requested {n_per_type}), using all available"
                )
                sampled = type_global_indices
            else:
                sampled = np.random.choice(type_global_indices, size=n_per_type, replace=False)
            selected_indices.append(sampled)
            selected_labels.extend([stype] * len(sampled))

        if not selected_indices:
            logger.warning("No cadences found for any signal type — skipping viz batch")
            self._latent_viz_batch = None
            self._latent_viz_labels = None
            return

        # Fancy indexing already creates a new independent array (no .copy() needed)
        all_indices = np.concatenate(selected_indices)
        self._latent_viz_batch = concat_data[all_indices]
        self._latent_viz_labels = np.array(selected_labels, dtype="U20")

        if len(all_indices) < n_per_type * len(signal_types):
            logger.warning(
                f"Requested {n_per_type} per type * {len(signal_types)} types. "
                f"Only {len(all_indices)} cadences available in latent viz batch"
            )
        else:
            logger.info(
                f"Prepared latent viz batch: {len(all_indices)} cadences "
                f"({n_per_type} per type * {len(signal_types)} types)"
            )

        # Build distributed dataset
        self._build_latent_viz_dataset()

    def _build_latent_viz_dataset(self):
        """Build the distributed viz dataset from the persisted viz batch."""
        # Prepare distributed dataset for latent viz encoding
        viz_results = prepare_distributed_viz_dataset(
            concat_data=self._latent_viz_batch,
            per_replica_inf_batch_size=self.config.training.per_replica_val_batch_size,
            num_replicas=self.strategy.num_replicas_in_sync,
            strategy=self.strategy,
        )

        self._latent_viz_dataset = viz_results["viz_dataset"]
        self._latent_viz_n_padded = viz_results["n_padded"]
        self._latent_viz_n_samples = viz_results["n_samples"]
        self._latent_viz_steps = viz_results["viz_steps"]
        self._latent_viz_holder = viz_results["_viz_holder"]
        del viz_results

        time_bins = self.config.data.time_bins
        width_bin = self.config.data.width_bin // self.config.data.downsample_factor

        # Create distributed inference function
        @tf.function
        def viz_encode_fn(batch_data):
            """Encode batch data using distributed strategy"""

            def encode_fn(data):
                """Per-replica encoding step"""
                # Reshape for encoder: (batch, 6, 16, 512) -> (batch * 6, 16, 512, 1)
                reshaped = tf.reshape(data, [-1, time_bins, width_bin, 1])

                # Encode (returns z_mean, z_log_var, z)
                z_mean, _, _ = self.vae.encoder(reshaped, training=False)

                return z_mean

            # Run encoding on all replicas
            per_replica_z_mean = self.strategy.run(encode_fn, args=(batch_data,))

            return per_replica_z_mean

        self._viz_encode_fn = viz_encode_fn

    def _capture_latent_snapshot(self, round_idx, epoch, step, snr_base, snr_range):
        """
        Run distributed inference on viz batch and write latent vectors to DB.

        Uses the distributed viz dataset and shared _distributed_encode() method to encode
        cadences across all GPUs, then writes one row per cadence to the latent_snapshots table.
        """
        if self._latent_viz_dataset is None:
            return

        n_padded = self._latent_viz_n_padded
        n_samples = self._latent_viz_n_samples
        num_obs = self.config.data.num_observations
        latent_dim = self.config.beta_vae.latent_dim

        # TEST: make sure this works as expected
        # Encode all cadences using distributed inference
        [all_z_mean] = self._distributed_encode(
            dataset=self._latent_viz_dataset,
            n_steps=self._latent_viz_steps,
            encode_fn=self._viz_encode_fn,
            n_samples=n_padded * num_obs,
            latent_dim=latent_dim,
            logging=False,
        )

        # Truncate padding and reshape to per-cadence: (n_samples, 6, latent_dim)
        all_z_mean = all_z_mean[: n_samples * num_obs]
        z_mean_per_cadence = all_z_mean.reshape(n_samples, num_obs, latent_dim)
        del all_z_mean

        # Write to DB
        timestamp = time.time()
        tag = self.config.checkpoint.save_tag
        for cadence_idx in range(n_samples):
            # NOTE: 8 decimal precison for stored latents
            latent_vector_list = np.round(z_mean_per_cadence[cadence_idx], 8).tolist()
            self.db.write_latent_snapshot(
                model_name="beta_vae",
                round_number=round_idx + 1,
                epoch_number=epoch + 1,
                step_number=step + 1,
                cadence_index=cadence_idx,
                signal_type=str(self._latent_viz_labels[cadence_idx]),
                latent_vector=latent_vector_list,
                snr_base=snr_base,
                snr_range=snr_range,
                tag=tag,
                timestamp=timestamp,
            )

        del z_mean_per_cadence

    def save_models(self, tag: str | None = None, dir: str | None = None):
        """Save model weights"""
        if tag is None:
            tag = self.config.checkpoint.save_tag

        if dir is not None:
            encoder_path = os.path.join(self.config.model_path, dir, f"vae_encoder_{tag}.keras")
            decoder_path = os.path.join(self.config.model_path, dir, f"vae_decoder_{tag}.keras")
            rf_path = os.path.join(self.config.model_path, dir, f"random_forest_{tag}.joblib")
        else:
            encoder_path = os.path.join(self.config.model_path, f"vae_encoder_{tag}.keras")
            decoder_path = os.path.join(self.config.model_path, f"vae_decoder_{tag}.keras")
            rf_path = os.path.join(self.config.model_path, f"random_forest_{tag}.joblib")

        os.makedirs(
            os.path.dirname(encoder_path), exist_ok=True
        )  # Create dir if it doesn't exist (encoder_path, decoder_path, rf_path all share parent dir)

        # Save VAE encoder (main model for inference)
        self.vae.encoder.save(encoder_path)
        logger.info(f"Saved VAE encoder to {encoder_path}")

        # Save decoder
        self.vae.decoder.save(decoder_path)
        logger.info(f"Saved VAE decoder to {decoder_path}")

        # Save Random Forest
        if self.rf_model is not None:
            self.rf_model.save(rf_path)
            logger.info(f"Saved Random Forest to {rf_path}")

    def load_models(self, tag: str | None = None, dir: str | None = None):
        """Load model weights"""
        if tag is None:
            # NOTE: use a more sensible default
            logger.info("No tag specified. Defaulting to 'final'")
            tag = "final"
        original_tag = tag

        # Construct filepaths
        if dir is not None:
            base_dir = os.path.join(self.config.model_path, dir)
            encoder_path = os.path.join(base_dir, f"vae_encoder_{tag}.keras")
            decoder_path = os.path.join(base_dir, f"vae_decoder_{tag}.keras")
            rf_path = os.path.join(base_dir, f"random_forest_{tag}.joblib")
        else:
            base_dir = self.config.model_path
            encoder_path = os.path.join(base_dir, f"vae_encoder_{tag}.keras")
            decoder_path = os.path.join(base_dir, f"vae_decoder_{tag}.keras")
            rf_path = os.path.join(base_dir, f"random_forest_{tag}.joblib")

        if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
            # If the specified path doesn't exist, try to find the latest tag from base_dir
            logger.warning(f"No models tagged as '{original_tag}' in {base_dir}")
            logger.warning(f"Looking for latest tag in {base_dir} instead")

            tag = get_latest_tag(
                base_dir
            )  # get_latest_tag() will raise an error if no valid tags exist in base_dir
            logger.info(f"Tag '{original_tag}' not found. Loading latest model with tag: '{tag}'")

            # Reconstruct paths with new tag
            encoder_path = os.path.join(base_dir, f"vae_encoder_{tag}.keras")
            decoder_path = os.path.join(base_dir, f"vae_decoder_{tag}.keras")
            rf_path = os.path.join(base_dir, f"random_forest_{tag}.joblib")

            # Sanity check: if paths still don't exist, raise an error
            if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
                raise FileNotFoundError("Models not found")

        # Load the models
        try:
            logger.info(f"Loading models from {base_dir} with tag '{tag}'")

            # Load encoder & decoder
            checkpoint_encoder = tf.keras.models.load_model(
                encoder_path, custom_objects={"Sampling": Sampling}
            )
            checkpoint_decoder = tf.keras.models.load_model(
                decoder_path, custom_objects={"Sampling": Sampling}
            )

            # Transfer weights
            self.vae.encoder.set_weights(checkpoint_encoder.get_weights())
            self.vae.decoder.set_weights(checkpoint_decoder.get_weights())

            logger.info("VAE loaded successfully")

            # Load Random Forest if it exists
            if os.path.exists(rf_path):
                # Initialize RF model if it doesn't exist yet
                if self.rf_model is None:
                    self.rf_model = RandomForestModel()

                self.rf_model.load(rf_path)
                logger.info("Random Forest loaded successfully")
            else:
                logger.info(
                    f"Random Forest not found at {rf_path} - this is normal if RF hasn't been trained yet"
                )

            logger.info(f"Successfully loaded models from {base_dir} with tag '{tag}'")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise  # Re-raise to propagate error


def run_training_pipeline(
    background_data: np.ndarray, strategy: tf.distribute.Strategy = None
) -> TrainingPipeline:
    """
    Complete Aetherscan training pipeline run

    Args:
        background_data: Array of preprocessed backgrounds, shape (n, 6, 16, 512)
        strategy: TensorFlow distribution strategy
    """
    try:
        # Create pipeline (no cleanup needed on failure)
        pipeline = TrainingPipeline(background_data, strategy)
    except Exception as e:
        logger.error(f"Error creating TrainingPipeline: {e}")
        raise  # Re-raise to propagate error

    try:
        try:
            # Train beta-VAE
            pipeline.train_beta_vae()
        except Exception as e:
            logger.error(f"Error in train_beta_vae(): {e}")
            raise  # Re-raise to propagate error

        try:
            # NOTE: combine plot_beta_vae_loss_curves(), plot_beta_vae_training_stability(), and plot_latent_space_gif() into plot_training_progress()?
            # Plot loss curves
            pipeline.plot_beta_vae_loss_curves()

            # Plot clipping rate
            pipeline.plot_beta_vae_training_stability()

            # Plot injection stats
            pipeline.plot_injection_stats()

            # Plot latent space GIF
            pipeline.plot_latent_space_gif()
        except Exception as e:
            logger.error(f"Error in plotting: {e}")
            raise  # Re-raise to propagate error

        try:
            # Train Random Forest
            pipeline.train_random_forest()
        except Exception as e:
            logger.error(f"Error in train_random_forest(): {e}")
            # Attempt to save models on RF training failure
            pipeline.rf_model = None  # Avoid saving incomplete RF model state
            _safe_call(pipeline.save_models, "save_models")
            raise  # Re-raise to propagate error

        try:
            # Save final models
            pipeline.save_models()
        except Exception as e:
            logger.error(f"Error in save_models(): {e}")
            raise  # Re-raise to propagate error

        logger.info("Training complete!")

        return pipeline

    finally:
        # Free shared resources on exit
        pipeline.data_generator.close()


def _safe_call(func: Callable, name: str, args: tuple | None = None) -> None:
    """Best-effort execution during error cleanup."""
    try:
        func(*args) if args else func()
    except Exception as e:
        logger.warning(f"Failed to execute {name} during cleanup: {e}")
