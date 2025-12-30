"""
Training orchestration for Aetherscan Pipeline
Implements full workflow for both beta-VAE & RF classifier,
Supports curriculum learning, distributed GPU training, and model checkpointing
"""

from __future__ import annotations

import gc
import glob
import logging
import os
import re
import shutil
import threading
from datetime import datetime

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.gridspec import GridSpec
from tensorflow.keras.initializers import GlorotNormal, HeNormal
from tensorflow.keras.layers import Conv2D, Dense

from aetherscan.config import get_config
from aetherscan.data_generation import DataGenerator
from aetherscan.models import RandomForestModel, Sampling, create_beta_vae_model

logger = logging.getLogger(__name__)


# TODO: simplify this to exclude tensorboard setup & rename to archive_directories()
def handle_directory(base_dir: str, target_dirs: list[str] | None = None, round_num: int = 1):
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


# TEST: resources are properly released on normal exits (between rounds) & emergency shutdown
# Create data holder objects, to be paired with data generators, for TF's distributed datasets
# Allows for explicit dereferencing of large arrays using DataHolder.clear(), which lets
# Python's garbage collector free up memory on-demand
# Note, DataHolder.clear() is only useful at the end of an epoch, once indices have been exhausted,
# since the data generators' local caches maintain references to the data until then
# This is not an issue in our current implementation, where we only clear & reset resources at the
# end of a round. However, if you require early exit behavior, you may want to remove the _lock and
# use explicit _cleared() checks instead, which negates the need for local caches (see commit hash
# 2a404a4). The trade-off being that you're at risk of race conditions if multiple threads attempt
# to access/clear the DataHolder simultaneously. While this is not the case in our current
# implementation, we opted for a more defensive approach rather than accomodating future design
# patterns. As well, the data should not be modified once the DataHolder has been initialized to
# prevent corrupted state in the DataHolder
class DataHolder:
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


# TODO: split if-else branches into prepare_distributed_train_dataset & prepare_distributed_inference_dataset
def prepare_distributed_dataset(
    data: dict,
    n_samples: int,
    train_val_split: float | None,
    per_replica_batch_size: int,
    global_batch_size: int | None,
    per_replica_val_batch_size: int | None,
    num_replicas: int,
    strategy: tf.distribute.Strategy,
    shuffle: bool = True,
) -> dict:
    """
    Prepare distributed datasets for training or inference

    Args:
        data: Dictionary with keys 'concatenated', 'true', 'false' (numpy arrays)
        n_samples: Number of samples in data
        train_val_split: If provided, split data into train/val sets. If None, return single dataset for inference
        per_replica_batch_size: Batch size per replica for training (or inference if no split)
        global_batch_size: Required if train_val_split is provided. Effective batch size across all replicas for training
        per_replica_val_batch_size: Required if train_val_split is provided. Batch size per replica for validation
        num_replicas: Number of replicas in strategy
        strategy: TensorFlow distribution strategy
        shuffle: Whether to shuffle training data (default True, set False for inference)

    Returns:
        If train_val_split is None:
            {dataset, n_trimmed, steps}
            Single distributed dataset, number of samples in dataset, and number of steps
        If train_val_split is provided:
            {train_dataset, val_dataset, n_train_trimmed, n_val_trimmed, train_steps, accumulation_steps, val_steps}
            Train/val distributed datasets, number of samples in each, number of steps for each (including accumulation sub-steps)
    """
    if train_val_split is not None:
        # Training case: split into train and val
        if global_batch_size is None or per_replica_val_batch_size is None:
            raise ValueError(
                "global_batch_size and per_replica_val_batch_size are required when train_val_split is provided"
            )

        # Split & trim to fit train/val batch size
        n_train = int(n_samples * train_val_split)
        n_val = n_samples - n_train

        n_train_trimmed = (n_train // global_batch_size) * global_batch_size
        n_val_trimmed = (n_val // per_replica_val_batch_size) * per_replica_val_batch_size

        logger.info(
            f"Data alignment: Train {n_train}→{n_train_trimmed}, Val {n_val}→{n_val_trimmed}"
        )

        # Prepare data
        train_concat = data["concatenated"][:n_train_trimmed]
        train_true = data["true"][:n_train_trimmed]
        train_false = data["false"][:n_train_trimmed]
        train_holder = DataHolder(train_concat, train_true, train_false)

        val_start = n_train
        val_end = val_start + n_val_trimmed
        val_concat = data["concatenated"][val_start:val_end]
        val_true = data["true"][val_start:val_end]
        val_false = data["false"][val_start:val_end]
        val_holder = DataHolder(val_concat, val_true, val_false)

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
                indices = np.arange(len(concat))
                if shuffle:
                    # Perform global shuffle on each epoch so each pass through the data is unique
                    np.random.shuffle(indices)
                for idx in indices:
                    yield (concat[idx], true[idx], false[idx]), concat[idx]

                # Remove cache references for future garbage collection
                del concat, true, false

        def val_generator():
            while True:  # Make generators infinite to reset state between epochs
                # Acquire lock to check cleared status and capture data references
                # Local references keep data alive even if clear() is called mid-epoch
                with val_holder._lock:
                    if val_holder._cleared:
                        return  # Exit if data already cleared
                    # Cache references while holding lock
                    concat = val_holder.concat
                    true = val_holder.true
                    false = val_holder.false

                # Maintain order on each epoch since no gradients are calculated during validation
                for idx in range(len(concat)):
                    yield (concat[idx], true[idx], false[idx]), concat[idx]

                # Remove cache references for future garbage collection
                del concat, true, false

        # Determine dataset output signature
        sample_shape = train_concat.shape[1:]
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
        logger.info(
            f"Creating infinite datasets from generators with per replica batch size - "
            f"Train: {per_replica_batch_size}, Val: {per_replica_val_batch_size}"
        )

        train_dataset = (
            tf.data.Dataset.from_generator(train_generator, output_signature=output_signature)
            .batch(per_replica_batch_size)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )

        val_dataset = (
            tf.data.Dataset.from_generator(val_generator, output_signature=output_signature)
            .batch(per_replica_val_batch_size)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )

        # Distribute datasets across GPUs
        logger.info(f"Distributing datasets across {num_replicas} GPUs")

        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = strategy.experimental_distribute_dataset(val_dataset)

        # Calculate steps
        train_steps = n_train_trimmed // global_batch_size
        accumulation_steps = global_batch_size // (per_replica_batch_size * num_replicas)
        val_steps = n_val_trimmed // (per_replica_val_batch_size * num_replicas)

        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "n_train_trimmed": n_train_trimmed,
            "n_val_trimmed": n_val_trimmed,
            "train_steps": train_steps,
            "accumulation_steps": accumulation_steps,
            "val_steps": val_steps,
            "_train_holder": train_holder,
            "_val_holder": val_holder,
        }

    else:
        # Inference case: no train/val split
        batch_size = per_replica_batch_size * num_replicas
        n_trimmed = (n_samples // batch_size) * batch_size

        logger.info(f"Data alignment: {n_samples}→{n_trimmed}")

        # Prepare data
        concat = data["concatenated"][:n_trimmed]
        true = data["true"][:n_trimmed]
        false = data["false"][:n_trimmed]
        holder = DataHolder(concat, true, false)

        # Create generator function for memory-efficient data loading
        def data_generator():
            while True:  # Make generator infinite to reset state between passes
                # Acquire lock to check cleared status and capture data references
                # Local references keep data alive even if clear() is called mid-epoch
                with holder._lock:
                    if holder._cleared:
                        return  # Exit if data already cleared
                    # Cache references while holding lock
                    concat = holder.concat
                    true = holder.true
                    false = holder.false

                # Work with local references (safe from clearing, no per-sample lock needed)
                indices = np.arange(len(concat))
                if shuffle:
                    np.random.shuffle(indices)
                for idx in indices:
                    yield (concat[idx], true[idx], false[idx]), concat[idx]

                # Remove cache references for future garbage collection
                del concat, true, false

        # Determine dataset output signature
        sample_shape = concat.shape[1:]
        output_signature = (
            (
                tf.TensorSpec(shape=sample_shape, dtype=tf.float32),
                tf.TensorSpec(shape=sample_shape, dtype=tf.float32),
                tf.TensorSpec(shape=sample_shape, dtype=tf.float32),
            ),
            tf.TensorSpec(shape=sample_shape, dtype=tf.float32),
        )

        # Create dataset using generator to reduce GPU memory pressure
        # Data is kept on CPU & transferred to GPU in batches on-demand
        logger.info(
            f"Creating infinite dataset from generator with per replica batch size: {per_replica_batch_size}"
        )

        dataset = (
            tf.data.Dataset.from_generator(data_generator, output_signature=output_signature)
            .batch(per_replica_batch_size)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )

        # Distribute dataset across GPUs
        logger.info(f"Distributing dataset across {num_replicas} GPUs")

        dataset = strategy.experimental_distribute_dataset(dataset)

        # Calculate steps
        steps = n_trimmed // batch_size

        return {
            "dataset": dataset,
            "n_trimmed": n_trimmed,
            "steps": steps,
            "_holder": holder,
        }


class TrainingPipeline:
    """Training pipeline"""

    def __init__(self, background_data, strategy=None):
        """
        Initialize training pipeline

        Args:
            background_data: Array of background observations
            strategy: TensorFlow distribution strategy
        """
        self.config = get_config()
        if self.config is None:
            raise ValueError("get_config() returned None")

        # Initialize data generator
        self.data_generator = DataGenerator(background_data)

        # Set distributed strategy
        self.strategy = strategy or tf.distribute.get_strategy()

        # Create VAE model & optimizer inside distributed context
        with self.strategy.scope():
            self.vae = create_beta_vae_model()
            self._build_optimizer()

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

        finally:
            # Regardless whether checkpoints were loaded or not, we finish the directory setup
            # since fault tolerance expects a clean directory structure

            # TODO: replace this with db writes? (search for all self.history instances)
            # Training history
            self.history = {
                "loss": [],
                "reconstruction_loss": [],
                "kl_loss": [],
                "true_loss": [],
                "false_loss": [],
                "val_loss": [],
                "val_reconstruction_loss": [],
                "val_kl_loss": [],
                "val_true_loss": [],
                "val_false_loss": [],
                "learning_rate": [],
            }

            # Setup directories
            self._setup_directories()

            # TODO: replace tensorboard logging with db writes + checkpoint plots? (search for all _setup_tensorboard_logging() & self.global_step instances)
            # Setup TensorBoard logging
            self._setup_tensorboard_logging()

    # TODO: delete this method once tensorboard logging is removed
    def __del__(self):
        """Cleanup TensorBoard writers and data generator"""
        if hasattr(self, "train_writer"):
            self.train_writer.close()
        if hasattr(self, "val_writer"):
            self.val_writer.close()

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
        handle_directory(model_checkpoints_dir, target_dirs=None, round_num=start_round)

        plot_checkpoints_dir = os.path.join(self.config.output_path, "plots", "checkpoints")
        handle_directory(plot_checkpoints_dir, target_dirs=None, round_num=start_round)

        logger.info("Setup directories complete")

    def _setup_tensorboard_logging(self):
        """Setup TensorBoard logging"""
        logger.info("Setting up TensorBoard logging")

        start_round = self.config.checkpoint.start_round

        logs_dir = os.path.join(self.config.output_path, "logs")
        handle_directory(logs_dir, target_dirs=["train", "validation"], round_num=start_round)

        self.global_step = (start_round - 1) * self.config.training.epochs_per_round
        if start_round == 1:
            logger.info("Starting fresh TensorBoard logs")
        else:
            logger.info(
                f"Resuming TensorBoard logs from step {self.global_step} (round {start_round})"
            )

        # Create TensorBoard writers
        train_log_dir = os.path.join(logs_dir, "train")
        val_log_dir = os.path.join(logs_dir, "validation")

        self.train_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_writer = tf.summary.create_file_writer(val_log_dir)

        logger.info(f"TensorBoard logs directory: {logs_dir}")
        logger.info(f"Initial global_step: {self.global_step}")

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
        decay_rate = self.config.training.exponential_decay_rate
        easy_rounds = self.config.training.step_easy_rounds
        hard_rounds = self.config.training.step_hard_rounds

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
        # TODO: allow user to pass in a list of step changes (add validation that len(list) divisible by num_training_rounds)
        # Step function - easy for first part, hard for second part
        elif schedule == "step":
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

        Note the following soft constraint:
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

    def train_round(self, round_idx: int, epochs: int, snr_base: int, snr_range: int):
        """
        Perform a single training round
        """
        logger.info(
            f"Training round {round_idx + 1} - Epochs: {epochs}, SNR: {snr_base}-{snr_base + snr_range}"
        )

        # Generate training data
        train_data = self.data_generator.generate_batch(
            self.config.training.num_samples_beta_vae, snr_base, snr_range
        )

        # Distribute training data
        data = prepare_distributed_dataset(
            data=train_data,
            n_samples=self.config.training.num_samples_beta_vae,
            train_val_split=self.config.training.train_val_split,
            per_replica_batch_size=self.config.training.per_replica_batch_size,
            global_batch_size=self.config.training.global_batch_size,
            per_replica_val_batch_size=self.config.training.per_replica_val_batch_size,
            num_replicas=self.strategy.num_replicas_in_sync,
            strategy=self.strategy,
            shuffle=True,
        )

        del train_data
        gc.collect()

        train_dataset = data["train_dataset"]
        val_dataset = data["val_dataset"]
        n_train_trimmed = data["n_train_trimmed"]
        n_val_trimmed = data["n_val_trimmed"]
        steps_per_epoch = data["train_steps"]
        accumulation_steps = data["accumulation_steps"]
        val_steps = data["val_steps"]
        train_holder = data["_train_holder"]
        val_holder = data["_val_holder"]

        del data
        gc.collect()

        # Sanity check: verify step sizes are valid
        if accumulation_steps < 1:
            raise ValueError(
                f"Accumulation steps < 1: global_batch_size ({self.config.training.global_batch_size}) must be >= per_replica_batch_size * num_replicas ({self.config.training.per_replica_batch_size * self.strategy.num_replicas_in_sync})"
            )
        if steps_per_epoch < 1:
            raise ValueError(
                f"Steps per epoch < 1: n_train_trimmed ({n_train_trimmed}) must be >= global_batch_size ({self.config.training.global_batch_size})"
            )
        if val_steps < 1:
            raise ValueError(
                f"Validation steps < 1: n_val_trimmed ({n_val_trimmed}) must be >= per_replica_val_batch_size * num_replicas ({self.config.training.per_replica_val_batch_size * self.strategy.num_replicas_in_sync})"
            )

        logger.info(
            f"Initializing training loop with {steps_per_epoch} train steps, {val_steps} val steps"
        )
        logger.info(f"Gradients accumulated every {accumulation_steps} sub-steps")

        try:
            for epoch in range(epochs):
                # Log resources at start of epoch
                logger.info(f"{'-' * 30}")
                logger.info(f"Epoch {epoch + 1}/{epochs} Start")

                # Training
                epoch_losses = self._train_epoch(train_dataset, steps_per_epoch, accumulation_steps)

                # Validation
                val_losses = self._validate_epoch(val_dataset, val_steps)

                # Log results
                logger.info(f"Epoch {epoch + 1} Complete")
                logger.info(
                    f"Train -- Total: {epoch_losses['total']:.4f}, "
                    f"Recon: {epoch_losses['reconstruction']:.4f}, "
                    f"KL: {epoch_losses['kl']:.4f}, "
                    f"True: {epoch_losses['true']:.4f}, "
                    f"False: {epoch_losses['false']:.4f}, "
                )
                logger.info(
                    f"Val -- Total: {val_losses['total']:.4f}, "
                    f"Recon: {val_losses['reconstruction']:.4f}, "
                    f"KL: {val_losses['kl']:.4f}, "
                    f"True: {val_losses['true']:.4f}, "
                    f"False: {val_losses['false']:.4f}"
                )

                # Update history
                for key, train_key in [
                    ("loss", "total"),
                    ("reconstruction_loss", "reconstruction"),
                    ("kl_loss", "kl"),
                    ("true_loss", "true"),
                    ("false_loss", "false"),
                ]:
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(float(epoch_losses[train_key]))
                for key, val_key in [
                    ("val_loss", "total"),
                    ("val_reconstruction_loss", "reconstruction"),
                    ("val_kl_loss", "kl"),
                    ("val_true_loss", "true"),
                    ("val_false_loss", "false"),
                ]:
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(float(val_losses[val_key]))
                self.history["learning_rate"].append(
                    float(self.vae.optimizer.learning_rate.numpy())
                )

                # TensorBoard logging
                with self.train_writer.as_default():
                    tf.summary.scalar("total_loss", epoch_losses["total"], step=self.global_step)
                    tf.summary.scalar(
                        "reconstruction_loss", epoch_losses["reconstruction"], step=self.global_step
                    )
                    tf.summary.scalar("kl_loss", epoch_losses["kl"], step=self.global_step)
                    tf.summary.scalar("true_loss", epoch_losses["true"], step=self.global_step)
                    tf.summary.scalar("false_loss", epoch_losses["false"], step=self.global_step)
                    tf.summary.scalar(
                        "learning_rate",
                        self.vae.optimizer.learning_rate.numpy(),
                        step=self.global_step,
                    )

                with self.val_writer.as_default():
                    tf.summary.scalar(
                        "validation_total_loss", val_losses["total"], step=self.global_step
                    )
                    tf.summary.scalar(
                        "validation_reconstruction_loss",
                        val_losses["reconstruction"],
                        step=self.global_step,
                    )
                    tf.summary.scalar("validation_kl_loss", val_losses["kl"], step=self.global_step)
                    tf.summary.scalar(
                        "validation_true_loss", val_losses["true"], step=self.global_step
                    )
                    tf.summary.scalar(
                        "validation_false_loss", val_losses["false"], step=self.global_step
                    )

                # Flush writers to ensure data is written
                self.train_writer.flush()
                self.val_writer.flush()

                # Increment global step
                self.global_step += 1

                # Adaptive learning rate
                self._update_learning_rate(val_losses)

                # Log resources at end of epoch
                logger.info(f"Epoch {epoch + 1}/{epochs} End")

            # Save checkpoint
            self.save_models(tag=f"round_{round_idx + 1:02d}", dir="checkpoints")

            # Plot progress
            self.plot_beta_vae_training_progress(
                tag=f"round_{round_idx + 1:02d}", dir="checkpoints"
            )

        # Run cleanup regardless if round finishes successfully or not
        finally:
            # Clear intermediate data
            # TEST:
            del train_dataset, val_dataset
            gc.collect()

            import time  # noqa: PLC0415

            time.sleep(0.1)

            train_holder.clear()
            val_holder.clear()
            # del train_dataset, val_dataset

            # Force TensorFlow to release internal references to datasets/iterators
            # This prevents generator closures from accumulating in memory between rounds
            tf.keras.backend.clear_session()
            logger.info("Cleared TensorFlow session state")

            # Reset multiprocessing pools in DataGenerator after each round
            # to further avoid memory accumulation
            self.data_generator.reset_managed_pool()

            gc.collect()

    def _train_epoch(self, dataset, steps_per_epoch, accumulation_steps=1):
        """
        Perform a single training epoch with gradient accumulation
        """
        epoch_losses = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0, "true": 0.0, "false": 0.0}
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
                        raise

                    except Exception as e:
                        logger.error(
                            f"Error during gradient computation at step {step + 1}, sub-step {sub_step + 1}: {e}"
                        )
                        raise

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
                self._apply_gradients(accumulated_gradients)

                for key, loss in step_losses.items():
                    # Average step losses over sub-steps
                    avg_loss = loss / successful_accumulations
                    step_losses[key] = avg_loss
                    epoch_losses[key] += avg_loss

                # Log progress every 10 steps
                if (step + 1) % 10 == 0 or (step + 1) == steps_per_epoch:
                    logger.info(
                        f"Step {step + 1}/{steps_per_epoch}, "
                        f"Total: {step_losses['total']:.4f}, "
                        f"Recon: {step_losses['reconstruction']:.4f}, "
                        f"KL: {step_losses['kl']:.4f}, "
                        f"True: {step_losses['true']:.4f}, "
                        f"False: {step_losses['false']:.4f}"
                    )

            # Average epoch losses over training steps
            for key in epoch_losses:
                epoch_losses[key] /= steps_per_epoch

            return epoch_losses

        # Run cleanup regardless if epoch finishes successfully or not
        finally:
            del iterator
            gc.collect()

    def _validate_epoch(self, dataset, steps):
        """
        Perform a single validation epoch
        """
        val_losses = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0, "true": 0.0, "false": 0.0}
        iterator = iter(dataset)

        try:
            for _step in range(steps):
                batch = next(iterator)
                step_losses = self._distributed_val_step(batch)

                for key in val_losses:
                    val_losses[key] += step_losses[key]

            # Average validation losses over validation steps
            for key in val_losses:
                val_losses[key] /= steps

            return val_losses

        # Run cleanup regardless if epoch finishes successfully or not
        finally:
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

                # Scale loss by num_replicas for gradient averaging
                scaled_loss = losses["total_loss"] / tf.cast(num_replicas, tf.float32)

            # Compute gradients
            gradients = tape.gradient(scaled_loss, self.vae.trainable_variables)

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
        Clip & apply gradients
        """
        # Clip gradients for additional stability
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        # Apply gradients
        self.vae.optimizer.apply_gradients(
            zip(clipped_gradients, self.vae.trainable_variables, strict=False)
        )

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

    def train_random_forest(self):
        """Train Random Forest"""
        logger.info("Training Random Forest classifier...")

        # Initialize RF model
        if self.rf_model is None:
            self.rf_model = RandomForestModel()

        elif self.rf_model.is_trained:
            logger.info("Random Forest classifier already trained. Exiting training loop.")
            return

        # BUG: edge case where if save-tag was already used in the past, train_random_forest() will load previous model weights to train RF, since pipeline.save_models() happens after pipeline.train_random_forest(). add a check to validate_args() to make sure save-tag is distinct?
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

        n_samples = self.config.training.num_samples_rf
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

        rf_data = self.data_generator.generate_batch(n_samples, snr_base, snr_range)

        # Prepare distributed dataset for inference
        results = prepare_distributed_dataset(
            data=rf_data,
            n_samples=n_samples,
            train_val_split=None,
            per_replica_batch_size=self.config.training.per_replica_val_batch_size,
            global_batch_size=None,
            per_replica_val_batch_size=None,
            num_replicas=self.strategy.num_replicas_in_sync,
            strategy=self.strategy,
            shuffle=False,  # Deterministic latent generation
        )

        del rf_data
        gc.collect()

        dataset = results["dataset"]
        n_trimmed = results["n_trimmed"]
        steps = results["steps"]
        holder = results["_holder"]

        del results
        gc.collect()

        logger.info(f"Generating latents for {n_trimmed} samples using distributed inference")

        # Pre-allocate latent arrays
        true_latents = np.empty((n_trimmed * num_observations, latent_dim), dtype=np.float32)
        false_latents = np.empty((n_trimmed * num_observations, latent_dim), dtype=np.float32)

        # Create distributed inference function
        @tf.function
        def distributed_encode(batch_data):
            """Encode batch data using distributed strategy"""

            def encode_fn(data):
                """Per-replica encoding step"""
                x, _ = data
                true_data = x[1]  # Extract true component
                false_data = x[2]  # Extract false component

                # Reshape for encoder: (batch, 6, 16, 512) -> (batch * 6, 16, 512, 1)
                batch_size = tf.shape(true_data)[0]
                true_reshaped = tf.reshape(true_data, [-1, time_bins, width_bin, 1])
                false_reshaped = tf.reshape(false_data, [-1, time_bins, width_bin, 1])

                # Encode (returns mean, log_var, z)
                _, _, true_z = self.vae.encoder(true_reshaped, training=False)
                _, _, false_z = self.vae.encoder(false_reshaped, training=False)

                return true_z, false_z

            # Run encoding on all replicas
            per_replica_true, per_replica_false = self.strategy.run(encode_fn, args=(batch_data,))

            return per_replica_true, per_replica_false

        # Process all batches
        iterator = iter(dataset)
        current_idx = 0

        try:
            for step in range(steps):
                batch = next(iterator)

                # Get per-replica latents for this batch
                per_replica_true, per_replica_false = distributed_encode(batch)

                # Extract results from each replica and concatenate
                # This avoids the inefficient gather operation with NCCL
                true_results = self.strategy.experimental_local_results(per_replica_true)
                false_results = self.strategy.experimental_local_results(per_replica_false)

                # Concatenate results from all replicas
                true_batch_np = np.concatenate([t.numpy() for t in true_results], axis=0)
                false_batch_np = np.concatenate([f.numpy() for f in false_results], axis=0)

                batch_latent_size = true_batch_np.shape[0]
                true_latents[current_idx : current_idx + batch_latent_size] = true_batch_np
                false_latents[current_idx : current_idx + batch_latent_size] = false_batch_np

                current_idx += batch_latent_size

                # Log progress
                if (step + 1) % 10 == 0 or (step + 1) == steps:
                    logger.info(f"Generated latents for step {step + 1}/{steps}")

                del per_replica_true, true_results, true_batch_np
                del per_replica_false, false_results, false_batch_np
                gc.collect()

            # Train Random Forest classifier
            self.rf_model.train(true_latents, false_latents)

            logger.info("Random Forest training complete")

        finally:
            del iterator

            # Clear intermediate data
            holder.clear()
            del dataset

            # Force TensorFlow to release internal references to datasets/iterators
            # This prevents generator closures from accumulating in memory between rounds
            tf.keras.backend.clear_session()
            logger.info("Cleared TensorFlow session state")

            # Reset multiprocessing pools in DataGenerator to further avoid memory accumulation
            self.data_generator.reset_managed_pool()

            del true_latents, false_latents
            gc.collect()

    def plot_beta_vae_training_progress(self, tag: str | None = None, dir: str | None = None):
        """Plot beta-VAE training history"""
        fig = plt.figure(figsize=(25, 12))
        gs = GridSpec(2, 4, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

        # Top subplot spanning full width - Total Loss
        ax_top = fig.add_subplot(gs[0, :])

        # Bottom subplots - Individual losses
        ax_recon = fig.add_subplot(gs[1, 0])
        ax_kl = fig.add_subplot(gs[1, 1])
        ax_true = fig.add_subplot(gs[1, 2])
        ax_false = fig.add_subplot(gs[1, 3])

        fig.suptitle("Beta-VAE Training Progress", fontsize=16)

        epochs = range(1, len(self.history.get("loss", [])) + 1)

        # Helper function to plot dual y-axis
        def plot_dual_axis(ax, title, train_key, val_key):
            # Create secondary y-axis for learning rate
            ax2 = ax.twinx()

            # Plot train and validation on left y-axis
            if train_key in self.history and self.history[train_key]:
                ax.plot(epochs, self.history[train_key], color="blue", label="train", linewidth=2)
            if val_key in self.history and self.history[val_key]:
                ax.plot(epochs, self.history[val_key], color="orange", label="val", linewidth=2)

            # Plot learning rate on right y-axis
            if "learning_rate" in self.history and self.history["learning_rate"]:
                ax2.plot(
                    epochs,
                    self.history["learning_rate"],
                    color="grey",
                    label="learning rate",
                    linewidth=1,
                    alpha=0.7,
                    linestyle="--",
                )

            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.3)

            ax2.tick_params(axis="y", labelcolor="grey")

        # Top subplot - Total Loss
        plot_dual_axis(ax_top, "Total Loss", "loss", "val_loss")

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
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        plt.tight_layout()

        # Save plot
        if tag is None:
            tag = self.config.checkpoint.save_tag

        if dir is not None:
            save_path = os.path.join(
                self.config.output_path, "plots", dir, f"beta_vae_training_progress_{tag}.png"
            )
        else:
            save_path = os.path.join(
                self.config.output_path, "plots", f"beta_vae_training_progress_{tag}.png"
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create dir if it doesn't exist

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()

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
            raise


def train_full_pipeline(background_data: np.ndarray, strategy=None) -> TrainingPipeline:
    """
    Complete Aetherscan training pipeline run

    Args:
        background_data: Array of background observations
        strategy: TensorFlow distribution strategy

    Returns:
        Trained pipeline object
    """
    # Create pipeline
    pipeline = TrainingPipeline(background_data, strategy)

    try:
        # Train beta-VAE
        pipeline.train_beta_vae()

        # Train Random Forest
        pipeline.train_random_forest()

        # Save final models
        pipeline.save_models()

        # Final plot
        pipeline.plot_beta_vae_training_progress()

        logger.info("Training complete!")

        return pipeline

    finally:
        # Free shared resources before exiting
        pipeline.data_generator.close()
