# NOTE: are we properly clearing memory after db read for plotting functions? are there any db reads that can be grouped together to reduce plotting times?
"""
Training orchestration for Aetherscan Pipeline
Implements full workflow for both beta-VAE & RF classifier,
Supports curriculum learning, adaptive LR, distributed datasets & training,
gradient accumulation, and model checkpointing
"""

from __future__ import annotations

import contextlib
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
import joblib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow as tf
import umap
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.initializers import GlorotNormal, HeNormal
from tensorflow.keras.layers import Conv2D, Dense

from aetherscan.benchmark import round_stage_name, stage_timer
from aetherscan.config import get_config
from aetherscan.data_generation import DataGenerator
from aetherscan.db import get_db, get_system_metadata
from aetherscan.hf_hub import upload_run_to_hf
from aetherscan.logger import get_logger
from aetherscan.models import (
    RandomForestModel,
    create_beta_vae_model,
    prepare_latent_features,
)
from aetherscan.rf_metrics import compute_rf_eval_metrics
from aetherscan.round_data import (
    RoundDataPaths,
    RoundDataProducer,
    load_round_arrays,
    prepare_round_data_dir,
    validate_done_manifest,
)
from aetherscan.run_state import (
    STAGE_FINAL_SAVE,
    STAGE_HF_UPLOAD,
    STAGE_RF_PLOTS,
    STAGE_RF_TRAIN,
    STAGE_VAE_PLOTS,
    STAGE_VAE_ROUNDS,
    TrainingRunState,
    config_changed,
    config_fingerprint,
    load_run_state,
    run_state_path,
    save_run_state,
)
from aetherscan.seeding import STREAM_DATASET, STREAM_PLOT, STREAM_VIZ, derive_rng
from aetherscan.shap_parallel import parallel_shap

logger = logging.getLogger(__name__)


# NOTE: Removing TensorBoard support
# archive_directory() includes (incomplete) functionality for setting up & handling
# TensorBoard directories. unless you're reviving TensorBoard support, simply leave target_dirs as
# None to use the function as "normal"
def archive_directory(base_dir: str, target_dirs: list[str] | None = None, round_num: int = 1):
    """
    Move (round_num == 1) or copy (round_num > 1) the contents of base_dir into a timestamped
    archive/ subdirectory; on resume, also delete entries whose round number is >= round_num.

    target_dirs lists subdirectory names to include in archiving (e.g. ['train', 'validation']);
    when None, only loose files in base_dir are considered and any subdirectories are ignored.
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
    Find the latest checkpoint tag (e.g. "round_05") in checkpoints_dir, choosing the most recent
    tag whose encoder/decoder pair both exist on disk.

    Tag families are ranked by priority: final_vX > round_XX > YYYYMMDD_HHMMSS > test_vX. The
    highest-priority family present wins, then ties within that family are broken by version /
    round number / timestamp. Raises FileNotFoundError if no valid pair is found.
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


def _model_pair_exists(base_dir: str, tag: str) -> bool:
    """True when the encoder/decoder pair for `tag` both exist in base_dir."""
    return os.path.exists(os.path.join(base_dir, f"vae_encoder_{tag}.keras")) and os.path.exists(
        os.path.join(base_dir, f"vae_decoder_{tag}.keras")
    )


def _resolve_load_tag(base_dir: str, tag: str | None) -> str:
    """
    Resolve the tag load_models() should load from base_dir.

    An explicitly requested tag must exist: raising FileNotFoundError beats the old silent
    get_latest_tag() fallback, which could resume training from a stale, unrelated model while
    reporting success (issue #142). Only the tag=None default may fall back to the latest tag
    present in base_dir.
    """
    if tag is not None:
        if _model_pair_exists(base_dir, tag):
            return tag
        msg = (
            f"No models tagged '{tag}' in {base_dir} — refusing to fall back to the latest tag "
            f"for an explicitly requested tag."
        )
        # The per-round-checkpoint hint only helps for a round_XX tag; for any other explicit
        # tag (e.g. a typo'd final_v2) it's a red herring, so only append it for round tags.
        if re.fullmatch(r"round_\d+", tag):
            msg += (
                " If you meant to resume from a per-round checkpoint, pass --load-dir checkpoints"
            )
        raise FileNotFoundError(msg)

    # NOTE: use a more sensible default
    logger.info("No tag specified. Defaulting to 'final'")
    if _model_pair_exists(base_dir, "final"):
        return "final"

    logger.warning(f"No models tagged as 'final' in {base_dir}")
    logger.warning(f"Looking for latest tag in {base_dir} instead")

    latest = get_latest_tag(
        base_dir
    )  # get_latest_tag() will raise an error if no valid tags exist in base_dir
    logger.info(f"Tag 'final' not found. Loading latest model with tag: '{latest}'")

    # Sanity check: get_latest_tag() only returns tags whose pair existed at scan time
    if not _model_pair_exists(base_dir, latest):
        raise FileNotFoundError("Models not found")
    return latest


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
    Heuristic: return True if at least one Conv2D/Dense layer's weight std deviates from its
    initializer's expected std by more than `threshold` (default 20%) — taken as evidence that
    the encoder has been trained rather than freshly initialized.
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


def check_val_auc_floor(
    val_binary_labels: np.ndarray, val_probas: np.ndarray, min_val_auc: float, tag: str
) -> float | None:
    """
    Opt-in quality floor on the RF's validation ROC-AUC (issue #139 Gate 1). Returns None
    without computing anything when the gate is disabled (min_val_auc <= 0.0, the default) or
    when val_binary_labels is single-class (roc_auc_score is undefined there; guaranteed not
    to happen by the data pipeline, but guarded per the gate's warn-don't-abort philosophy —
    mirrors compute_rf_eval_metrics' identical guard in rf_metrics.py); otherwise returns the
    computed AUC. When the floor is set and unmet, emits a loud WARNING (which reaches the
    Slack summary) rather than failing the run — so a run that "completes" but learned nothing
    (bad data, collapsed latent, mislabeled classes) is flagged before its model is promoted.
    """
    if min_val_auc <= 0.0:
        return None

    if np.unique(val_binary_labels).size < 2:
        logger.warning(
            f"Model quality gate cannot be evaluated: validation labels for tag '{tag}' are "
            f"single-class, so ROC-AUC is undefined (training.min_val_auc={min_val_auc} was "
            f"configured). This should not happen — data generation guarantees both classes."
        )
        return None

    val_auc = float(roc_auc_score(val_binary_labels, val_probas))
    if val_auc < min_val_auc:
        logger.warning(
            f"MODEL QUALITY GATE UNMET: validation ROC-AUC {val_auc:.4f} is below the "
            f"configured floor training.min_val_auc={min_val_auc} for tag '{tag}'. The "
            f"trained model may be degenerate — inspect rf_eval_artifacts_{tag}.joblib "
            f"before promoting this model."
        )
    else:
        logger.info(
            f"Model quality gate passed: validation ROC-AUC {val_auc:.4f} >= "
            f"training.min_val_auc={min_val_auc}"
        )
    return val_auc


def build_traversal_latents(
    z_base: np.ndarray, sigmas: np.ndarray, num_steps: int, max_sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the latent grid for a traversal of every latent dimension around `z_base`.

    Row (d, s) (row-major over dims then steps) is z_base + steps[s] * sigmas[d] * e_d, where
    steps = linspace(-max_sigma, +max_sigma, num_steps) — with the validated odd num_steps the
    center step is exactly 0, i.e. the unperturbed base decode. Returns (latents of shape
    (latent_dim * num_steps, latent_dim) float32, steps of shape (num_steps,)).
    """
    z_base = np.asarray(z_base, dtype=np.float32)
    sigmas = np.asarray(sigmas, dtype=np.float32)
    if z_base.ndim != 1 or z_base.shape != sigmas.shape:
        raise ValueError(
            f"z_base and sigmas must be matching 1-D vectors, got {z_base.shape} and {sigmas.shape}"
        )

    latent_dim = z_base.shape[0]
    steps = np.linspace(-max_sigma, max_sigma, num_steps).astype(np.float32)
    if num_steps % 2 == 1:
        # Snap the center step to exactly 0 — linspace can leave ~1e-16 residue for
        # non-integral max_sigma, and the center column must be the exact base decode
        steps[num_steps // 2] = 0.0
    latents = np.tile(z_base, (latent_dim * num_steps, 1))
    for d in range(latent_dim):
        latents[d * num_steps : (d + 1) * num_steps, d] += steps * sigmas[d]
    return latents, steps


def compute_traversal_panels(
    z_base: np.ndarray,
    sigmas: np.ndarray,
    num_steps: int,
    max_sigma: float,
    decode_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode a full traversal grid into per-(dim, step) reconstruction panels.

    `decode_fn` maps a (n, latent_dim) latent batch to (n, time, freq) or (n, time, freq, 1)
    reconstructions (production passes the VAE decoder; unit tests pass a stub). Returns
    (panels of shape (latent_dim, num_steps, time, freq), steps of shape (num_steps,)).
    """
    latents, steps = build_traversal_latents(z_base, sigmas, num_steps, max_sigma)
    recon = np.asarray(decode_fn(latents))
    if recon.ndim == 4 and recon.shape[-1] == 1:
        recon = recon[..., 0]
    if recon.ndim != 3 or recon.shape[0] != latents.shape[0]:
        raise ValueError(
            f"decode_fn must return (n, time, freq[, 1]) reconstructions for n={latents.shape[0]} "
            f"latents, got shape {recon.shape}"
        )
    latent_dim = latents.shape[1]
    panels = recon.reshape(latent_dim, num_steps, recon.shape[1], recon.shape[2])
    return panels, steps


def unpreprocess_traversal_panels(
    panels: np.ndarray, lognorm_params: tuple[float, float] | None, downsample_factor: int
) -> tuple[np.ndarray, bool]:
    """
    Approximately undo generation-time preprocessing on decoded panels, for display only.

    Exact inversion is impossible: downsampling is lossy and the log-norm parameters are
    per-observation while a traversal decode blends many observations — so this is an honest
    approximation (stated on the figures). Where `lognorm_params` = (min_log, range_log) is
    available (and non-degenerate), the log-norm is inverted via exp(x * range_log + min_log);
    otherwise panels stay in normalized log space. Downsampling is undone by nearest-neighbor
    repetition along the frequency axis so the axis is in true (pre-downsample) bins.

    Returns (display_panels, inverted) — `inverted` says whether the log-norm inversion was
    applied (drives the figures' intensity-scale caption).
    """
    panels = np.asarray(panels)
    inverted = False
    if lognorm_params is not None:
        min_log, range_log = lognorm_params
        if range_log > 0:
            panels = np.exp(panels * range_log + min_log)
            inverted = True
    if downsample_factor > 1:
        panels = np.repeat(panels, downsample_factor, axis=-1)
    return panels, inverted


# Create data holder objects, to be paired with data generators, for TF's distributed datasets
# Allows for explicit dereferencing of the backing arrays using holder.clear(), which lets
# Python's garbage collector free up memory on-demand
# The holders now hold np.load(mmap_mode="r") memmap references rather than ~294 GB of in-RAM
# arrays, so clear() drops file mappings (letting the round's directory be deleted and its page
# cache reclaimed) rather than freeing huge heap allocations — but the semantics are unchanged:
# clear() is only fully effective at the end of an epoch, once the generators' local caches have
# been dropped, and the data must not be modified once the holder has been initialized
# Note, if you require early exit behavior, you may want to remove the _lock and use explicit
# _cleared() checks instead, which negates the need for local caches (see commit hash 2a404a4).
# The trade-off being that you're at risk of race conditions if multiple threads attempt to
# access/clear the holder simultaneously — we opted for the defensive approach
# Lock contention with TF's prefetch threads is a non-issue in the batched design: each generator
# acquires the lock exactly once per epoch pass (to snapshot the array references), then yields
# whole global batches without touching the lock, so a clear() on the main thread can no longer
# race hundreds of thousands of per-sample lock acquisitions (which the pre-memmap, per-sample
# generators were at least theoretically exposed to)
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
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Build distributed training and validation datasets from the `data` dict, returning a dict
    with the two tf.data datasets, sample/step counts, the shared TrainDataHolder, and the
    stratified train/val indices into the original arrays.

    `rng` supplies all randomness (stratified split, trim subsampling, per-epoch shuffles):
    callers pass a Generator derived from the pipeline root seed for reproducible runs (see
    aetherscan.seeding.derive_rng); None falls back to OS entropy (the historical behavior).

    `data` must contain 'concatenated', 'true', 'false', and 'labels' — typically the read-only
    memmaps returned by round_data.load_round_arrays(), though plain in-RAM ndarrays work too.
    The split is stratified across the 4 signal types (false_no_signal, false_with_rfi,
    true_only_eti, true_eti_rfi) — generation lays labels out sequentially within chunks, so a
    naive positional split would over-represent later signal types in val.

    The generators yield whole GLOBAL batches (leading batch dim in the output signature; no
    .batch() call downstream): each batch gathers sorted fancy indices from the memmaps, cutting
    the per-sample Python/tf.data boundary crossings by a factor of the global batch size — the
    per-sample yields were the main source of the 0-14 % GPU utilization. Randomness lives at
    the epoch level (train_indices are reshuffled each pass); sorting *within* a batch only
    improves memmap read locality and the model is order-invariant within a batch.

    Page-cache framing: gathering from the memmaps pulls pages through the OS page cache, so
    after the first epoch a round's ~294 GB (at full-scale defaults) is served at RAM speed from
    otherwise-free memory on the 503 GB training nodes — but under memory pressure the kernel
    evicts pages instead of OOM-killing the process, which is exactly the failure mode the old
    in-RAM arrays hit.

    Each generated batch has the signature ((concat, true, false), concat). Sample counts are
    trimmed to the global / effective batch size to keep all replicas evenly fed (so every epoch
    pass yields whole batches exactly); the holder is shared by both generators so neither pays
    a memory cost beyond index subsets.
    """
    global_train_batch_size = per_replica_batch_size * num_replicas
    global_val_batch_size = per_replica_val_batch_size * num_replicas

    # NOTE: runtime backstop for the cross-replica check cli.py's collect_validation_errors
    # skips when TF can't see the GPUs at validation time (#143). n_train is trimmed to a
    # multiple of effective_batch_size below, so an indivisible combination would make the
    # train generator emit a partial trailing global batch every epoch (split unevenly across
    # replicas) and silently degrade accumulation_steps. Message mirrors the validator's so
    # guidance is identical.
    if effective_batch_size % global_train_batch_size != 0:
        raise ValueError(
            f"--effective-batch-size ({effective_batch_size}) must be divisible by "
            f"per_replica_batch_size * num_replicas ({global_train_batch_size})"
        )

    # The train generator below closes over this rng, so epoch k's shuffle consumes the
    # stream's k-th state — deterministic given a seeded Generator, different every epoch
    if rng is None:
        rng = np.random.default_rng()

    # Stratified train/val split to ensure both sets contain proportional representation
    # of all 4 signal types (false_no_signal, false_with_rfi, true_only_eti, true_eti_rfi).
    # This is necessary because generation arranges labels sequentially within
    # chunks, so a naive positional split would over-represent later signal types in val.
    labels = np.asarray(data["labels"])
    unique_labels = np.unique(labels)

    train_indices = []
    val_indices = []

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
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
        train_indices = rng.choice(train_indices, size=n_train_trimmed, replace=False)
    if n_val_trimmed < n_val:
        val_indices = rng.choice(val_indices, size=n_val_trimmed, replace=False)

    # Sort both index sets ascending. For shuffle=False this pins the generators' yield order
    # to the returned train_indices/val_indices arrays (the alignment contract
    # train_random_forest depends on) while giving monotone memmap reads; for shuffle=True the
    # epoch-level reshuffle below supplies the randomness anyway. Stratification is a property
    # of index *membership*, not order, so sorting doesn't affect it.
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)

    logger.info(f"Data alignment: Train {n_train}→{n_train_trimmed}, Val {n_val}→{n_val_trimmed}")

    # Share the original arrays between train and val generators via a single data holder.
    # The stratified split requires non-contiguous indices, which would force numpy to create
    # full copies via fancy indexing (~2x peak memory). Instead, both generators gather
    # per-batch slices from the same original arrays using their respective index subsets —
    # only one global batch is materialized at a time.
    train_holder = TrainDataHolder(data["concatenated"], data["true"], data["false"])

    # Create generator functions yielding whole global batches gathered from the (memmap)
    # arrays — see the docstring for the batching/locality/page-cache rationale
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

            # Work with local references (safe from clearing, no per-batch lock needed)
            # Copy train_indices because rng.shuffle mutates in-place
            indices = train_indices.copy()
            if shuffle:
                # Perform global shuffle on each epoch so each pass through the data is unique
                rng.shuffle(indices)
            for start in range(0, len(indices), global_train_batch_size):
                batch_indices = indices[start : start + global_train_batch_size]
                if shuffle:
                    # Within-batch sorted order improves memmap read locality; random batch
                    # membership is already guaranteed by the epoch-level shuffle above
                    batch_indices = np.sort(batch_indices)
                concat_batch = concat[batch_indices]
                yield (concat_batch, true[batch_indices], false[batch_indices]), concat_batch

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

            # Maintain val_indices order on each epoch (already sorted above): no gradients are
            # calculated during validation, and train_random_forest relies on the i-th encoded
            # val cadence corresponding to val_indices[i]
            for start in range(0, len(val_indices), global_val_batch_size):
                batch_indices = val_indices[start : start + global_val_batch_size]
                concat_batch = concat[batch_indices]
                yield (concat_batch, true[batch_indices], false[batch_indices]), concat_batch

            # Remove cache references to ensure garbage collection in future
            del concat, true, false

    # Determine dataset output signature: the generators yield whole global batches, so the
    # specs carry a leading (None) batch dimension and no .batch() call is applied downstream
    sample_shape = data["concatenated"].shape[1:]
    batch_spec = tf.TensorSpec(shape=(None, *sample_shape), dtype=tf.float32)
    output_signature = ((batch_spec, batch_spec, batch_spec), batch_spec)

    # Create datasets using generators to reduce GPU memory pressure
    # Data is kept on CPU & transferred to GPU in batches on-demand
    # Note that the generators yield data in global batches before being sharded (distributed)
    # across replicas, ensuring per replica batch sizes match expectations
    logger.info(
        f"Creating infinite batched datasets from generators with global batch size - "
        f"Train: {global_train_batch_size}, Val: {global_val_batch_size}"
    )

    train_dataset = (
        tf.data.Dataset.from_generator(train_generator, output_signature=output_signature)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_generator(val_generator, output_signature=output_signature)
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
        "train_indices": train_indices,  # For train_random_forest() label alignment (not shuffled)
        "val_indices": val_indices,  # For train_round() -> _prepare_latent_viz_batch()
    }


def prepare_distributed_viz_dataset(
    concat_data: np.ndarray,
    per_replica_inf_batch_size: int,
    num_replicas: int,
    strategy: tf.distribute.Strategy,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Build a distributed dataset for latent-space visualization from `concat_data` (shape
    (n_cadences, 6, 16, width_bin)), returning a dict with the dataset, padded/real sample counts,
    step count, and the VizDataHolder.

    The dataset yields cadences in original order (no shuffle) — plot_latent_space_gif() depends
    on this ordering. If n_samples isn't divisible by the global batch size, the input is padded
    with random duplicates to keep all replicas evenly fed; downstream code can use n_samples vs.
    n_padded to drop the padded tail when needed. `rng` seeds the padding choice (a Generator
    derived from the pipeline root seed, or None for OS entropy).
    """
    global_viz_batch_size = per_replica_inf_batch_size * num_replicas
    n_samples = concat_data.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    # NOTE: does padding/divisibility matter for inference?
    # Pad datasets to fit batch sizes (prevents uneven batches on final step)
    # Note, n_samples should already be divisible by effective_batch_size
    # Padding here is just a defensive measure to doubly ensure divisibility before creating &
    # distributing our datasets
    # Alternatively, we could also trim the data instead of padding
    n_padded = int(np.ceil(n_samples / global_viz_batch_size)) * global_viz_batch_size

    if n_padded > n_samples:
        pad_count = n_padded - n_samples
        pad_indices = rng.choice(n_samples, size=pad_count, replace=True)
        padded_data = np.concatenate([concat_data, concat_data[pad_indices]], axis=0)
        logger.info(f"Data alignment: Viz {n_samples}→{n_padded} (padded {pad_count})")
    else:
        padded_data = concat_data
        logger.info(f"Data alignment: Viz {n_samples} (no padding needed)")

    viz_holder = VizDataHolder(padded_data)

    # Create generator function yielding whole global batches — this feeds
    # _capture_latent_snapshot every latent_viz_step_interval training steps, so per-sample
    # yields here used to tax every capture during the epoch loop
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
            # Contiguous in-order slices preserve the original cadence order on every pass
            # (n_padded is an exact multiple of the global batch size, so slices are whole)
            for start in range(0, len(concat), global_viz_batch_size):
                yield concat[start : start + global_viz_batch_size]

            # Remove cache references for future garbage collection
            del concat

    # Determine dataset output signature: the generator yields whole global batches, so the
    # spec carries a leading (None) batch dimension and no .batch() call is applied downstream
    sample_shape = padded_data.shape[1:]
    output_signature = tf.TensorSpec(shape=(None, *sample_shape), dtype=tf.float32)

    # Create dataset using generator to reduce GPU memory pressure
    # Data is kept on CPU & transferred to GPU in batches on-demand
    # Note that the generator yields data in global batches before being sharded (distributed)
    # across replicas, ensuring per replica batch sizes match expectations
    logger.info(
        f"Creating infinite batched dataset from generator with global batch size: {global_viz_batch_size}"
    )

    viz_dataset = (
        tf.data.Dataset.from_generator(viz_generator, output_signature=output_signature)
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


# _select_positive_class_shap now lives in aetherscan.shap_parallel (shared with the TF-free SHAP
# worker processes) and is imported at the top of this module under the same name.


@contextlib.contextmanager
def _silence_stderr():
    """
    Redirect stderr to /dev/null for the duration of the block.

    SHAP's TreeExplainer drives a tqdm progress bar that writes to stderr. The
    project-wide logger.py replaces sys.stderr with a StreamToLogger at ERROR
    level, so every progress-bar refresh becomes an ERROR record and is forwarded
    to Slack — flooding the channel and triggering SSL EOFs from the webhook.

    Python exceptions still propagate normally (they don't go through stderr
    text), so wrapping a block in this context manager only suppresses the
    tqdm output, not legitimate errors. Note however that any C-level warnings
    emitted by underlying libraries (sklearn tree internals, etc.) are also
    silenced; the debug-level enter/exit logs make this visible in the log
    record so an unexpectedly silent stretch can be traced back here.
    """
    logger.debug("Entering _silence_stderr() — stderr redirected to /dev/null")
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            yield
    finally:
        logger.debug("Exiting _silence_stderr() — stderr restored")


class TrainingPipeline:
    """Training pipeline"""

    def __init__(self, background_data, strategy: tf.distribute.Strategy = None):
        """
        Initialize the training pipeline with a background observation array and an optional
        tf.distribute strategy (defaults to the current strategy, i.e. no-op for single-device).
        """
        self.config = get_config()
        if self.config is None:
            raise ValueError("get_config() returned None")

        self.db = get_db()
        if self.db is None:
            raise ValueError("get_db() returned None")

        # Reproducibility: seed TF's global RNG before any model/variable creation so weight
        # initialization (HeNormal/GlorotNormal) and the VAE Sampling layer draw
        # deterministic streams. numpy/python randomness is NOT globally seeded here — each
        # consumer derives its own independent stream from the same root seed (see
        # aetherscan.seeding.derive_rng and its call sites). No-op when seed is None
        if self.config.training.seed is not None:
            tf.random.set_seed(self.config.training.seed)
            logger.info(f"Seeded TF global RNG from root seed {self.config.training.seed}")
        if self.config.training.tf_deterministic_ops:
            # Deterministic cuDNN/reduction kernels for bit-exact GPU reproducibility, at
            # some training-speed cost. Only useful alongside a root seed
            tf.config.experimental.enable_op_determinism()
            logger.info("TF op determinism enabled (deterministic GPU kernels)")
            if self.config.training.seed is None:
                # Without a seed the deterministic kernels cost speed but buy no
                # reproducibility (TF's global RNG stays unseeded) — warn so it isn't silent.
                logger.warning(
                    "tf_deterministic_ops is enabled but --seed is not set: deterministic "
                    "kernels incur a speed cost without making the run reproducible. Pass "
                    "--seed to seed the RNG streams."
                )

        # Load (or create) the persisted run manifest for this tag. This resolves
        # self.start_time (wall clock of attempt 1 — used by every DB query/plot, so retries
        # see the whole run), self._start_round (where the beta-VAE loop resumes), and marks
        # stale DB rows from a previously failed attempt as superseded.
        self._init_run_state()

        # Initialize data generator
        self.data_generator = DataGenerator(background_data)

        # Set distributed strategy
        self.strategy = strategy or tf.distribute.get_strategy()

        # Create VAE model & optimizer inside distributed context
        with self.strategy.scope():
            self.vae = create_beta_vae_model()
            self._build_optimizer()

        # Latent viz data (prepared once on first round, persisted across rounds within an
        # attempt; a resumed attempt rebuilds it from the resumed round's val split — the data
        # distribution matches by construction, and latent-GIF frames captured before the
        # failure stay valid in the DB)
        self._latent_viz_batch = None
        self._latent_viz_labels = None
        self._latent_viz_lognorm_params = None
        self._latent_viz_dataset = None
        self._latent_viz_n_padded = None
        self._latent_viz_n_samples = None
        self._latent_viz_steps = None
        self._latent_viz_holder = None
        self._viz_encode_fn = None

        # Initialize RF model as None
        self.rf_model = None

        # Tag an already-trained RF was loaded from (set by load_models); lets
        # train_random_forest name the stale source when it skips retraining (issue #142)
        self._rf_loaded_from_tag: str | None = None

        # Set when train_random_forest skips because a pre-loaded RF was already trained —
        # main.py annotates the terminal status instead of reporting unqualified success
        self.rf_training_skipped_from_tag: str | None = None

        # Background round-data producer (created in train_beta_vae when
        # overlap_data_generation is enabled; None otherwise)
        self._round_producer: RoundDataProducer | None = None

        # In-memory caches for RF eval artifacts and SHAP values, keyed by tag.
        # All ten RF plots consume the same eval-artifact joblib (and the five SHAP
        # plots additionally share a SHAP-values joblib); without these caches each
        # plot would deserialize the same large arrays from disk independently.
        # Cleared by _clear_rf_caches() at the end of run_training_pipeline().
        self._rf_artifacts_cache: dict[str, dict] = {}
        self._rf_shap_cache: dict[str, dict] = {}

        try:
            # Load models from checkpoints: explicit user flags win; otherwise a
            # manifest-driven resume reloads the last completed round's checkpoint
            if self.config.checkpoint.load_tag or self.config.checkpoint.load_dir:
                logger.info("Resuming from checkpoint")
                self.load_models(
                    tag=self.config.checkpoint.load_tag, dir=self.config.checkpoint.load_dir
                )
            elif self._start_round > 1:
                resume_tag = f"round_{self._start_round - 1:02d}"
                logger.info(f"Resuming from checkpoint {resume_tag} (per run manifest)")
                self.load_models(tag=resume_tag, dir="checkpoints")

        except Exception as e:
            logger.error(f"Error loading models from checkpoint: {e}")
            logger.info("Resetting config.checkpoint to start training from scratch")
            self.config.checkpoint.load_dir = None
            self.config.checkpoint.load_tag = None
            self.config.checkpoint.start_round = 1
            raise  # Re-raise to propagate error

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

    def _init_run_state(self):
        """
        Load or create the persisted run manifest (run_state_{save_tag}.json) and derive all
        resume state from it:

        - self.start_time: the manifest's run_start_time (wall clock of attempt 1). Setting it
          here — unconditionally, before any training/plotting can run — fixes the old bug
          where train_beta_vae()'s already-trained early return skipped the assignment and
          every subsequent plot raised AttributeError, masking the real failure.
        - self._start_round: 1-based round the beta-VAE loop starts from. Explicit user
          checkpoint flags (--load-tag/--load-dir [+ --start-round]) win as an escape hatch;
          otherwise the manifest's completed_rounds drive resume; a fresh run falls back to
          config.checkpoint.start_round (default 1).
        - Supersede marking: on resume of round k, rows for (tag, round >= k) in
          training_stats / injection_stats / latent_snapshots are stale output of the failed
          attempt and get flagged before the rerun re-writes them — otherwise duplicated
          epochs corrupt the loss-curve plots (which sort by (round, epoch)).
        """
        tag = self.config.checkpoint.save_tag
        self._run_state_path = run_state_path(self.config.output_path, tag)

        state = load_run_state(self._run_state_path)

        # Config-drift guard: if the manifest was written under a different result-affecting
        # config (reused --save-tag with changed hyperparameters, or a manifest that predates
        # fingerprinting), do NOT silently resume/skip and ship the stale model — warn loudly
        # and downgrade to a fresh run (the previous attempt's outputs get overwritten).
        current_fingerprint = config_fingerprint(self.config.to_dict())
        if config_changed(state, current_fingerprint):
            logger.warning("=" * 60)
            logger.warning(
                f"Run manifest at {self._run_state_path} was written under a DIFFERENT "
                f"training config (fingerprint mismatch) — starting a FRESH run and "
                f"overwriting the previous attempt's outputs for tag '{tag}'. Use a new "
                f"--save-tag to keep both, or restore the original config to resume."
            )
            logger.warning("=" * 60)
            state = None

        self._resumed = state is not None

        if state is None:
            state = TrainingRunState(
                tag=tag, run_start_time=time.time(), config_fingerprint=current_fingerprint
            )
            logger.info(f"Starting a fresh training run manifest at {self._run_state_path}")
        else:
            state.attempt += 1
            logger.info(
                f"Resuming training run from manifest {self._run_state_path} "
                f"(attempt {state.attempt}, completed rounds: {state.completed_rounds}, "
                f"stages done: {state.stages_done}, stages failed: {state.stages_failed})"
            )

        explicit_checkpoint = (
            self.config.checkpoint.load_tag is not None
            or self.config.checkpoint.load_dir is not None
        )
        if explicit_checkpoint:
            self._start_round = self.config.checkpoint.start_round
            if self._resumed:
                # User override re-runs rounds >= start_round: drop manifest records the
                # override invalidates (stages_done is cleared wholesale — downstream stages
                # depend on the re-run rounds and will simply re-execute)
                logger.info(
                    f"Explicit checkpoint flags override the manifest: restarting from round "
                    f"{self._start_round} and re-running all pipeline stages"
                )
                state.completed_rounds = [
                    r for r in state.completed_rounds if r < self._start_round
                ]
                state.stages_done = []
        elif state.completed_rounds:
            self._start_round = state.resume_round
        else:
            self._start_round = self.config.checkpoint.start_round

        self.start_time = state.run_start_time
        self.run_state = state

        # Flag the failed attempt's partial rows before this attempt re-writes them.
        # Rows from completed rounds (< _start_round) stay live — they're valid history
        # (including latent-GIF snapshot frames). No-op on a fresh run (nothing to mark)
        # and when resuming into post-VAE stages (_start_round > num_training_rounds)
        if self._resumed:
            for table in ("training_stats", "injection_stats", "latent_snapshots"):
                if not self.db.mark_superseded(table, tag, round_ge=self._start_round):
                    # Non-fatal (matches flush() semantics), but must be loud: unmarked
                    # stale rows would re-corrupt the loss-curve plots on this attempt
                    logger.warning(
                        f"Could not mark stale {table} rows superseded "
                        f"(tag={tag}, round_ge={self._start_round}); "
                        f"plots may include rows from the failed attempt"
                    )

        self._persist_run_state()

    def _persist_run_state(self):
        """Persist the run manifest (atomic tmp -> replace)."""
        save_run_state(self.run_state, self._run_state_path)

    def _mark_stage_done(self, stage: str):
        """Record a pipeline stage success in the manifest and persist it."""
        self.run_state.mark_stage_done(stage)
        self._persist_run_state()
        logger.info(f"Stage '{stage}' complete (recorded in run manifest)")

    def _record_stage_failure(self, stage: str):
        """Record a non-critical stage failure in the manifest and persist it. The stage
        stays out of stages_done, so a relaunch retries it; main.py exits nonzero at the very
        end if any recorded failure never recovers."""
        self.run_state.record_stage_failure(stage)
        self._persist_run_state()
        logger.error(f"Stage '{stage}' failed (recorded in run manifest)")

    def _clear_stage_failure(self, stage: str):
        """Drop a recorded stage failure from the manifest and persist it (used when the
        user opts out of an optional stage whose previous attempt failed — see
        TrainingRunState.clear_stage_failure)."""
        self.run_state.clear_stage_failure(stage)
        self._persist_run_state()

    def _mark_round_completed(self, round_number: int):
        """Record a fully-trained round (checkpoint saved) in the manifest and persist it."""
        self.run_state.mark_round_completed(round_number)
        self._persist_run_state()
        logger.info(f"Round {round_number} recorded as completed in run manifest")

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

        start_round = self._start_round

        model_checkpoints_dir = os.path.join(self.config.model_path, "checkpoints")
        archive_directory(model_checkpoints_dir, target_dirs=None, round_num=start_round)

        plot_checkpoints_dir = os.path.join(
            self.config.output_path,
            "plots",
            "training",
            self.config.checkpoint.save_tag,
            "checkpoints",
        )
        archive_directory(plot_checkpoints_dir, target_dirs=None, round_num=start_round)

        # Disk-backed round-data directory for this tag: delete round dirs >= start_round,
        # keep earlier ones only if their .done manifest validates (round-data mirror of the
        # checkpoint archiving above, minus the archiving — a round is ~295 GB)
        round_data_root = self.config.training.round_data_dir or self.config.get_training_file_path(
            "round_data"
        )
        self._round_data_base_dir = os.path.join(round_data_root, self.config.checkpoint.save_tag)
        prepare_round_data_dir(self._round_data_base_dir, start_round)

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
        start_round = self._start_round

        if start_round > n_rounds:
            # Every round already has a saved checkpoint (per the run manifest) — nothing to
            # train. Safe to return early: self.start_time is set in __init__ from the
            # manifest's run_start_time, so downstream plots still span the whole run.
            logger.info(f"All {n_rounds} rounds already trained — skipping beta-VAE training")
            return
        elif start_round > 1:
            logger.info(f"Resuming training from round {start_round}/{n_rounds}")
        else:
            logger.info(f"Starting training for {n_rounds} rounds")

        # Stand up the background round-data producer (a dedicated process owning its own
        # worker pool against the shared-memory background plates), so round k+1's data
        # generates while round k trains AND generation escapes this process's GIL — TF's
        # prefetch/callback threads made round 2+ generation far slower than round 1's.
        # Falls back to sequential in-process generation when disabled
        # (--no-overlap-data-generation) or when the DataGenerator has no shared memory for
        # producer workers to attach to (n_processes == 1).
        if self.config.training.overlap_data_generation:
            if self.data_generator.shm is not None:
                self._round_producer = RoundDataProducer(
                    base_dir=self._round_data_base_dir,
                    n_samples=self.config.training.num_samples_beta_vae,
                    shm_name=self.data_generator.shm.name,
                    background_shape=self.data_generator._background_shape,
                    background_dtype=str(self.data_generator._background_dtype),
                    n_processes=self.data_generator.n_processes,
                    width_bin=self.data_generator.width_bin,
                    num_observations=self.config.data.num_observations,
                    time_bins=self.config.data.time_bins,
                    chunk_size=self.config.training.signal_injection_chunk_size,
                    task_size=self.config.training.data_gen_task_size,
                    freq_resolution=self.data_generator.freq_resolution,
                    time_resolution=self.data_generator.time_resolution,
                    db=self.db,
                    tag=self.config.checkpoint.save_tag,
                    seed=self.config.training.seed,
                )
                self._round_producer.start()
                # Kick off the first round's data right away (nothing to overlap with yet —
                # the unavoidable serial start)
                first_snr_base, first_snr_range = self._calculate_curriculum_snr(start_round - 1)
                self._round_producer.request_generation(
                    start_round, first_snr_base, first_snr_range
                )
            else:
                logger.warning(
                    "overlap_data_generation is enabled but DataGenerator is in sequential "
                    "mode (n_processes=1, no shared memory) — falling back to in-process "
                    "round data generation"
                )

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

                # Umbrella stage span for the whole round (data wait + epochs + plots +
                # checkpoint save) — the sub-stages inside train_round nest under it
                with stage_timer(round_stage_name(round_idx + 1)):
                    self.train_round(
                        round_idx=round_idx, epochs=epochs, snr_base=snr_base, snr_range=snr_range
                    )
        finally:
            # Wind down the producer (graceful shutdown message, escalating to
            # terminate -> kill through the ResourceManager if it's mid-generation)
            if self._round_producer is not None:
                self._round_producer.shutdown()
                self._round_producer = None

            # NOTE: the latent viz batch intentionally survives this method — the vae_plots
            # stage's plot_latent_traversal re-encodes/decodes it. It is freed by
            # _clear_latent_viz_data(), called from the stage machine after vae_plots (and
            # from run_training_pipeline's cleanup on any earlier failure)

    def train_round(self, round_idx: int, epochs: int, snr_base: int, snr_range: int):
        """
        Perform a single training round
        """
        logger.info(
            f"Training round {round_idx + 1} - Epochs: {epochs}, SNR: {snr_base}-{snr_base + snr_range}"
        )

        round_number = round_idx + 1
        n_samples = self.config.training.num_samples_beta_vae
        paths = RoundDataPaths.for_round(self._round_data_base_dir, round_number)
        round_trained = False  # Set True once the round fully completes (drives dir deletion)

        # Obtain this round's disk-backed data: reuse a validated on-disk dataset if one
        # exists, otherwise wait on the background producer (which was asked to generate it
        # while the previous round trained) or generate in-process (overlap disabled)
        if validate_done_manifest(paths, expected_n_samples=n_samples) is not None:
            logger.info(f"Reusing validated round {round_number} data at {paths.round_dir}")
        elif self._round_producer is not None:
            logger.info(f"Waiting for round {round_number} data from the background producer")
            wait_start = time.time()
            self._round_producer.await_round(round_number)
            logger.info(f"Round {round_number} data ready (waited {time.time() - wait_start:.1f}s)")
        else:
            # In-process generation (overlap disabled). The producer path records the same
            # stage from its own timing message (see round_data._producer_main / the
            # drainer's "timing" handler) — the two paths never both run for one round
            with stage_timer("data_generation", metadata={"source": "in-process"}):
                self.data_generator.generate_round(
                    paths, n_samples, snr_base, snr_range, round_number
                )

        # Immediately queue generation of the next round's data so it runs in the producer
        # process while this round's epochs train (curriculum SNR for round k+1 is
        # deterministic, so it can be computed ahead of time)
        if (
            self._round_producer is not None
            and round_number < self.config.training.num_training_rounds
        ):
            next_snr_base, next_snr_range = self._calculate_curriculum_snr(round_idx + 1)
            self._round_producer.request_generation(round_number + 1, next_snr_base, next_snr_range)

        # Open the round's arrays as read-only memmaps (nothing is loaded into RAM here; the
        # batched generators gather from the OS page cache during training)
        train_data = load_round_arrays(paths)

        # Extract labels and the (tiny, eagerly-loaded) log-norm parameter array before
        # distributing (prepare_distributed_train_dataset keeps the original arrays alive via
        # a shared train_holder — no copies — so we can free the dict reference immediately
        # after)
        train_labels = train_data.get("labels")
        train_lognorm = train_data.get("lognorm")

        # Distribute training data (rng keyed on the round so each round's split/epoch
        # shuffles are an independent — but seed-reproducible — stream)
        data = prepare_distributed_train_dataset(
            data=train_data,
            train_val_split=self.config.training.train_val_split,
            per_replica_batch_size=self.config.training.per_replica_batch_size,
            effective_batch_size=self.config.training.effective_batch_size,
            per_replica_val_batch_size=self.config.training.per_replica_val_batch_size,
            num_replicas=self.strategy.num_replicas_in_sync,
            strategy=self.strategy,
            shuffle=True,
            rng=derive_rng(self.config.training.seed, STREAM_DATASET, round_number),
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
                    lognorm_params=train_lognorm,
                )
        # On subsequent rounds, the latent viz batch is persisted,
        # but the distributed dataset needs to be rebuilt
        elif self._latent_viz_batch is not None and self._latent_viz_dataset is None:
            self._build_latent_viz_dataset()

        del train_labels, train_lognorm
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
            # Round-level epoch span (per-epoch durations already live in training_stats).
            # stage_timer records the span even on a mid-loop exception (status=failed).
            with stage_timer("epochs"):
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
                    val_losses, val_duration = self._validate_epoch(
                        val_dataset, val_steps, time.time()
                    )

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
            with stage_timer("plots"):
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

                # Optional per-round latent traversal (config-gated; the canonical set
                # renders once at end of training in the vae_plots stage). Failures are
                # logged and swallowed — an optional plot mustn't fail the round and cost
                # a retry cycle including data regeneration
                if self.config.training.latent_traversal_every_round:
                    try:
                        self.plot_latent_traversal(
                            tag=f"round_{round_idx + 1:02d}", dir="checkpoints"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to execute plot_latent_traversal for round {round_idx + 1}: {e}"
                        )

                # NOTE: commented out to save compute. a final latent space gif at the end of training should suffice
                # Generate latent space GIF
                # self.plot_latent_space_gif(tag=f"round_{round_idx + 1:02d}", dir="checkpoints")

            # Save checkpoint
            with stage_timer("checkpoint_save"):
                self.save_models(tag=f"round_{round_idx + 1:02d}", dir="checkpoints")

            # Checkpoint is on disk — record the round in the run manifest so a retry (or a
            # relaunch of the identical command) resumes at the next round
            self._mark_round_completed(round_number)

            round_trained = True

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

            # Delete the round's on-disk data as soon as its training completed (keeps the
            # disk footprint at ~2 rounds max with overlap). A failed round leaves its data
            # in place; the retry's _setup_directories() -> prepare_round_data_dir() decides
            # what survives (dirs >= the resume round are regenerated). Deleting after the
            # holder.clear()/clear_session() above is safe even if stray memmap handles
            # linger — POSIX keeps the inodes alive until the mappings drop.
            if round_trained and not self.config.training.keep_round_data:
                shutil.rmtree(paths.round_dir, ignore_errors=True)
                logger.info(f"Deleted round {round_number} data directory: {paths.round_dir}")

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
        Compute the (snr_base, snr_range) for the round at 0-indexed `round_idx`, narrowing the
        range from initial_snr_range to final_snr_range across config.training.num_training_rounds.

        Three schedules are supported via config.training.curriculum_schedule: 'linear' (uniform
        narrowing), 'exponential' (fast then slow, normalized so progress=0 and progress=1 hit the
        exact endpoints), and 'step' (initial_snr_range for the first step_easy_rounds, then
        final_snr_range for step_hard_rounds; the two must sum to num_training_rounds).
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
        Adaptive LR: track validation loss; if it fails to improve by min_pct_improvement for
        patience_threshold consecutive epochs, scale current LR by (1 - reduction_factor), floored
        at min_learning_rate. Returns the (possibly unchanged) current LR.

        Heuristic to keep in mind:
            min_learning_rate - base_learning_rate * (1 - reduction_factor) ^ (epochs_per_round / patience_threshold)
        LR can only reach min_learning_rate within a round if the expression above is > 0;
        otherwise the LR resets at the start of the next round before bottoming out.
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
        Run a provided @tf.function (`encode_fn`) over `n_steps` of a distributed `dataset` and
        return a list of (n_samples, latent_dim) ndarrays — one per tensor that encode_fn yields.

        The number of output arrays is inferred from encode_fn's return on the first step: a bare
        PerReplica tensor yields 1 output, a tuple of PerReplica tensors yields N. Per-replica
        results are gathered via experimental_local_results + np.concatenate, which is faster than
        a strategy-level gather over NCCL for the small latent payload.
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

            # Drift guard: the train/val datasets returned by
            # prepare_distributed_train_dataset() are .repeat()-ed, so the iterator
            # never terminates on its own — the caller's n_steps is the only thing
            # that bounds the encode loop. If n_steps undercounts (e.g. the
            # accumulation-step convention in prepare_distributed_train_dataset
            # changes and train_random_forest's `train_steps * accumulation_steps`
            # math falls out of sync), we silently leave trailing rows of `outputs`
            # uninitialized. Fail loudly here so downstream plots don't consume
            # garbage. Overcounts are caught earlier by the slice-assignment.
            if current_idx != n_samples:
                raise RuntimeError(
                    f"_distributed_encode produced {current_idx} samples, expected "
                    f"{n_samples} (n_steps={n_steps}). Step-counting drift between "
                    f"caller and prepare_distributed_train_dataset()."
                )

            # Log progress
            if logging:
                logger.info(f"Finished encoding {n_steps} steps")
        finally:
            # NOTE: should check to make sure iterator exists first
            del iterator
            gc.collect()

        return outputs

    # NOTE: write what to db? (e.g. accuracy, F1). should we move all joblib read/writes to db read/writes?
    def train_random_forest(self):
        """Train Random Forest"""
        logger.info("Training Random Forest classifier...")

        # Resume path: a previous attempt of THIS run (same tag AND matching config
        # fingerprint) already trained and persisted the RF — load it back instead of
        # regenerating ~num_samples_rf cadences and retraining. self._resumed is False for a
        # fresh run, and for a reused tag whose config changed (the fingerprint guard in
        # _init_run_state downgraded it to a fresh run), so stale artifacts are never
        # silently reused.
        if self._resumed and self.try_load_rf_for_resume():
            logger.info(
                "Loaded existing Random Forest model + eval artifacts for this tag — "
                "skipping RF data generation and retraining"
            )
            return

        # Initialize RF model
        if self.rf_model is None:
            self.rf_model = RandomForestModel()

        elif self.rf_model.is_trained:
            # A pre-loaded, already-trained RF short-circuits the whole stage — make the skip
            # loud and record it, or a resume from the wrong tag ships a stale RF while the
            # run reports unqualified success (issue #142)
            source_tag = self._rf_loaded_from_tag or "unknown"
            logger.warning("=" * 60)
            logger.warning(
                f"RF training SKIPPED: an already-trained Random Forest (loaded from tag "
                f"'{source_tag}') is in memory — no new Random Forest will be trained for "
                f"save tag '{self.config.checkpoint.save_tag}'; the loaded model is reused "
                f"as-is"
            )
            logger.warning("=" * 60)
            self.rf_training_skipped_from_tag = source_tag
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

        n_samples = self.config.training.num_samples_rf
        snr_base = self.config.training.snr_base
        snr_range = (
            self.config.training.initial_snr_range
        )  # NOTE: should we use initial_snr_range or final_snr_range?

        latent_dim = self.config.beta_vae.latent_dim
        num_observations = self.config.data.num_observations
        time_bins = self.config.data.time_bins
        width_bin = self.config.data.width_bin // self.config.data.downsample_factor

        # Generate training data (concatenated is 4-way balanced; labels track per-sample
        # subtype) into a disk-backed dataset alongside the per-round dirs. Generation is
        # in-process (sequential with training) — there is nothing left to overlap with, the
        # beta-VAE producer has already been shut down by train_beta_vae()
        logger.info(f"Preparing training set with SNR: {snr_base}-{snr_base + snr_range}")
        rf_paths = RoundDataPaths(
            round_dir=os.path.join(self._round_data_base_dir, "rf"), round_idx=0
        )
        rf_trained = False  # Set True once RF training fully completes (drives dir deletion)
        if validate_done_manifest(rf_paths, expected_n_samples=n_samples) is not None:
            logger.info(f"Reusing validated RF dataset at {rf_paths.round_dir}")
        else:
            # RF data is always generated in-process (no background producer for the RF phase);
            # tag source to match the per-round data_generation spans (producer / in-process)
            #
            # round_num: a sentinel one past the last beta-VAE round, NOT the literal RF phase
            # (there is no "RF round" numbering). Passing it (instead of the default None) gives
            # every injection_stats row from this call a real round_number, so a stale row from a
            # crashed rf_train attempt is reachable by _init_run_state's round_ge supersede on the
            # next retry. That call marks (tag, round_number >= self._start_round); SQL NULLs never
            # satisfy ">=", so a NULL round_number (the previous default) permanently escaped it.
            # resume_round (run_state.py) is max(completed_rounds) + 1, and completed_rounds only
            # ever holds values <= num_training_rounds, so _start_round <= num_training_rounds + 1
            # always — and rf_train only (re)starts once vae_rounds is fully done, i.e. exactly
            # when _start_round == num_training_rounds + 1. round_number has no other consumer
            # (plot_injection_stats/query_injection_stat don't filter or group by it), so this is
            # safe to change without touching mark_superseded or the DB schema.
            with stage_timer("data_generation", metadata={"source": "in-process"}):
                self.data_generator.generate_round(
                    rf_paths,
                    n_samples,
                    snr_base,
                    snr_range,
                    round_num=self.config.training.num_training_rounds + 1,
                )
        rf_data = load_round_arrays(rf_paths)

        # Prepare distributed train+val datasets (stratified split). shuffle=False so the
        # train generator yields in train_indices order, letting us align encoded features
        # with data["labels"] deterministically
        # Note that we use the same prepare_distributed_train_dataset() call, with the same training
        # batch sizes as the beta-VAE, for convenience. We account for the gradient accumulation
        # baked into train_steps later in train_encode_steps
        results = prepare_distributed_train_dataset(
            data=rf_data,
            train_val_split=self.config.training.train_val_split,
            per_replica_batch_size=self.config.training.per_replica_batch_size,
            effective_batch_size=self.config.training.effective_batch_size,
            per_replica_val_batch_size=self.config.training.per_replica_val_batch_size,
            num_replicas=self.strategy.num_replicas_in_sync,
            strategy=self.strategy,
            shuffle=False,
            # RF dataset uses the round-0 stream key, mirroring its data generation
            # (beta-VAE rounds are 1-based, so no collision)
            rng=derive_rng(self.config.training.seed, STREAM_DATASET, 0),
        )

        # NOTE: come back to this later
        rf_subtypes_full = rf_data["labels"]

        # Free the dict shell (original arrays stay alive via the shared train_holder)
        del rf_data
        gc.collect()

        train_dataset = results["train_dataset"]
        val_dataset = results["val_dataset"]
        n_train_trimmed = results["n_train_trimmed"]
        n_val_trimmed = results["n_val_trimmed"]
        train_steps = results["train_steps"]
        accumulation_steps = results["accumulation_steps"]
        val_steps = results["val_steps"]
        train_holder = results["_train_holder"]
        train_indices = results["train_indices"]
        val_indices = results["val_indices"]

        del results
        gc.collect()

        logger.info(
            f"Generating latents for {n_train_trimmed} train + {n_val_trimmed} val samples "
            f"using distributed inference"
        )

        # Create distributed inference function
        @tf.function
        def rf_encode_fn(batch_data):
            """Encode batch data using distributed strategy"""

            def encode_fn(data):
                """Per-replica encoding step"""
                (concat_data, _, _), _ = data

                # Reshape for encoder: (batch, 6, 16, 512) -> (batch * 6, 16, 512, 1)
                concat_reshaped = tf.reshape(concat_data, [-1, time_bins, width_bin, 1])

                # Encode (returns z_mean, z_log_var, z)
                _, _, concat_z = self.vae.encoder(concat_reshaped, training=False)

                return concat_z

            per_replica_concat = self.strategy.run(encode_fn, args=(batch_data,))
            return per_replica_concat

        train_latents = None
        val_latents = None

        try:
            # train_steps accounts for gradient accumulation (each "step" = accumulation_steps
            # sub-batches), but _distributed_encode fetches one batch per step. Multiply by
            # accumulation_steps so we iterate over the full training set.
            # Note that this is a workaround resulting from using prepare_distributed_train_dataset(),
            # which was originally designed with only the beta-VAE in mind. _distributed_encode
            # asserts current_idx == n_samples after the loop, so any future drift in
            # prepare_distributed_train_dataset's step-counting convention will fail loudly
            # rather than silently encode the wrong number of cadences.
            #
            # Disjointness contract: the train and val passes below read from the same
            # TrainDataHolder backing arrays via train_indices / val_indices, which are
            # built by stratified per-label split + rng.choice subsampling — disjoint
            # by construction. The .repeat()-ed datasets cannot leak across splits because
            # each generator only yields rows from its own index subset.
            train_encode_steps = train_steps * accumulation_steps

            with stage_timer("encode"):
                [train_latents] = self._distributed_encode(
                    dataset=train_dataset,
                    n_steps=train_encode_steps,
                    encode_fn=rf_encode_fn,
                    n_samples=n_train_trimmed * num_observations,
                    latent_dim=latent_dim,
                    logging=True,
                )

                [val_latents] = self._distributed_encode(
                    dataset=val_dataset,
                    n_steps=val_steps,
                    encode_fn=rf_encode_fn,
                    n_samples=n_val_trimmed * num_observations,
                    latent_dim=latent_dim,
                    logging=True,
                )

            # Derive aligned binary & sub-type labels for train/val splits. With shuffle=False,
            # the i-th cadence in the encoded train/val array corresponds to train_indices[i] /
            # val_indices[i] in the original data
            train_subtype_labels = rf_subtypes_full[train_indices].astype("U20")
            val_subtype_labels = rf_subtypes_full[val_indices].astype("U20")
            train_binary_labels = np.array(
                [s.startswith("true_") for s in train_subtype_labels], dtype=np.int64
            )
            val_binary_labels = np.array(
                [s.startswith("true_") for s in val_subtype_labels], dtype=np.int64
            )

            # Train Random Forest classifier (passes latent_vectors; model flattens internally)
            with stage_timer("fit"):
                self.rf_model.train(train_latents, train_binary_labels)
            logger.info("Random Forest training complete")

            # NOTE: come back to this later (is it correct to call prepare_latent_features directly? this impacts the __init__.py in models/. what do we need features & probas for? why are we calling model.predict_proba directly? is it better to have a wrapper here? could we modify the return signature of predict_proba to return both features & probabilities?)
            # Compute flattened features + val probas/preds for downstream plotting.
            # val_preds is thresholded at inference.classification_threshold so the
            # confusion matrix, SHAP correct/incorrect markers, and decision-boundary
            # markers all reflect the model's deployment operating point — not
            # sklearn's argmax default (implicitly 0.5 for the binary RF), which
            # would understate FN / overstate TP relative to production behavior.
            train_features = prepare_latent_features(train_latents, num_observations)
            val_features = prepare_latent_features(val_latents, num_observations)
            val_probas = self.rf_model.model.predict_proba(val_features)[:, 1].astype(np.float32)
            classification_threshold = self.config.inference.classification_threshold
            val_preds = (val_probas >= classification_threshold).astype(np.int64)

            # NOTE: come back to this later (what is this artifact for? is it handled properly by archiving functions on startup?)
            # Persist a single eval-artifact joblib that every RF plot function consumes
            tag = self.config.checkpoint.save_tag
            artifacts = {
                "train_features": train_features,
                "train_binary_labels": train_binary_labels,
                "train_subtype_labels": train_subtype_labels,
                "val_features": val_features,
                "val_binary_labels": val_binary_labels,
                "val_subtype_labels": val_subtype_labels,
                "val_probas": val_probas,
                "val_preds": val_preds,
                "classification_threshold": classification_threshold,
                "snr_base": snr_base,
                "snr_range": snr_range,
                "tag": tag,
            }
            artifact_path = os.path.join(self.config.model_path, f"rf_eval_artifacts_{tag}.joblib")
            os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
            joblib.dump(artifacts, artifact_path)
            logger.info(f"Saved RF eval artifacts to {artifact_path}")

            # Opt-in model-quality gate on the val ROC-AUC (issue #139 Gate 1) — a loud
            # WARNING when unmet, not a failure
            check_val_auc_floor(
                val_binary_labels=val_binary_labels,
                val_probas=val_probas,
                min_val_auc=self.config.training.min_val_auc,
                tag=tag,
            )

            # Persist the trained RF immediately (final_save re-saves it later): a retry that
            # resumes into rf_plots/final_save can then reload the model without regenerating
            # data — see try_load_rf_for_resume()
            rf_model_path = os.path.join(self.config.model_path, f"random_forest_{tag}.joblib")
            self.rf_model.save(rf_model_path)
            logger.info(f"Saved Random Forest to {rf_model_path}")

            rf_trained = True

            # Persist scalar eval metrics to training_stats (model_name='rf') so the RF is
            # first-class on the live dashboard (issue #171). Written only after both
            # joblibs above landed: any later retry then resumes via try_load_rf_for_resume()
            # and skips retraining, so these rows are never duplicated (a reused tag's stale
            # rows are handled by the dashboard's last-write-wins read). Best-effort:
            # metric persistence must never fail the training run.
            try:
                rf_metrics = compute_rf_eval_metrics(
                    val_binary_labels=val_binary_labels,
                    val_subtype_labels=val_subtype_labels,
                    val_probas=val_probas,
                    val_preds=val_preds,
                )
                rf_metrics["classification_threshold"] = float(classification_threshold)
                metrics_timestamp = time.time()
                for stat_name, value in rf_metrics.items():
                    self.db.write_training_stat(
                        model_name="rf",
                        stat_name=stat_name,
                        value=value,
                        tag=tag,
                        timestamp=metrics_timestamp,
                    )
                logger.info(
                    f"Wrote {len(rf_metrics)} RF eval metrics to training_stats (tag={tag})"
                )
            except Exception as e:
                logger.warning(f"Failed to write RF eval metrics to db: {e}")

            # NOTE: come back to this later (are we dereferencing the correct things? can we instead write things to db instead of storing in memory?)
            del (
                artifacts,
                train_features,
                val_features,
                train_subtype_labels,
                val_subtype_labels,
                train_binary_labels,
                val_binary_labels,
                val_probas,
                val_preds,
            )
            gc.collect()

        except Exception as e:
            logger.error(f"Error in train_random_forest(): {e}")
            raise  # Re-raise to propagate error

        finally:
            # NOTE: should check to make sure holder & dataset exist first
            # Clear intermediate data
            train_holder.clear()
            del train_dataset, val_dataset
            # NOTE: come back to this later (why are these dereferenced in finally instead of main code block?)
            del rf_subtypes_full, train_indices, val_indices

            # Force TensorFlow to release internal references to datasets/iterators
            # This prevents generator closures from accumulating in memory between rounds
            tf.keras.backend.clear_session()
            logger.info("Cleared TensorFlow session state")

            # Reset multiprocessing pools in DataGenerator to further avoid memory accumulation
            self.data_generator.reset_managed_pool()
            logger.info("Reset managed pools")

            # Delete the RF dataset once RF training fully completed. On failure it stays on
            # disk for post-mortem; the next run's prepare_round_data_dir() treats it as
            # stale and regenerates (matching the pre-memmap per-attempt regeneration)
            if rf_trained and not self.config.training.keep_round_data:
                shutil.rmtree(rf_paths.round_dir, ignore_errors=True)
                logger.info(f"Deleted RF data directory: {rf_paths.round_dir}")

            # NOTE: is this the right way to check if arrays exist before dereferencing?
            if train_latents is not None:
                del train_latents
            if val_latents is not None:
                del val_latents
            gc.collect()

    def try_load_rf_for_resume(self) -> bool:
        """
        Attempt to restore the trained Random Forest persisted by a previous attempt of this
        run: both random_forest_{tag}.joblib and rf_eval_artifacts_{tag}.joblib must exist
        under model_path (the artifacts joblib is what every RF plot consumes, so resuming
        into rf_plots without it would be pointless — but it is loaded lazily by
        _load_rf_eval_artifacts(), not here). Returns True when the model is loaded
        and ready; False falls back to full RF training.
        """
        tag = self.config.checkpoint.save_tag
        rf_model_path = os.path.join(self.config.model_path, f"random_forest_{tag}.joblib")
        artifact_path = os.path.join(self.config.model_path, f"rf_eval_artifacts_{tag}.joblib")

        if not (os.path.exists(rf_model_path) and os.path.exists(artifact_path)):
            return False

        try:
            if self.rf_model is None:
                self.rf_model = RandomForestModel()
            self.rf_model.load(rf_model_path)
            return True
        except Exception as e:
            logger.warning(f"Failed to load persisted Random Forest from {rf_model_path}: {e}")
            return False

    def _load_rf_eval_artifacts(self, tag: str | None = None) -> dict:
        """
        Load the RF eval-artifacts joblib written by train_random_forest().

        Memoized on self by tag — a single pipeline run produces ten RF plots
        and we only want to deserialize the (large) features/probas/preds
        arrays once. _clear_rf_caches() drops the cache after all plots run.
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        if tag in self._rf_artifacts_cache:
            return self._rf_artifacts_cache[tag]

        artifact_path = os.path.join(self.config.model_path, f"rf_eval_artifacts_{tag}.joblib")
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"RF eval artifacts not found at {artifact_path}. "
                f"train_random_forest() must run before RF plots are generated."
            )
        logger.info(f"Loading RF eval artifacts from {artifact_path}")
        artifacts = joblib.load(artifact_path)
        self._rf_artifacts_cache[tag] = artifacts
        return artifacts

    def _compute_or_load_shap_values(self, artifacts: dict) -> dict:
        """
        Compute SHAP values for the trained RF (summary, interaction, log-loss decomposition)
        and cache to disk. Subsequent calls load the cache.

        Returns a dict with keys:
            shap_values_summary ->
                SHAP values for the positive class (true signal) on a sampled subset of val.
                Drives summary, dependence, and log-loss plots.
            summary_indices ->
                Which val rows the summary subset corresponds to.
                Needed to look up matching features/labels.
            shap_values_interaction ->
                Pairwise interaction values.
                Diagonal = main effect, off-diagonal = pure interaction between feature pairs.
            interaction_indices ->
                Which val rows the interaction subset corresponds to.
            shap_values_logloss ->
                SHAP values that decompose log loss instead of P(true).
                Computed with model_output="log_loss" plus a background dataset and per-sample y.
                Negative entries = feature reduced loss, positive = feature increased loss.
            expected_value
        """
        tag = artifacts["tag"]

        if tag in self._rf_shap_cache:
            return self._rf_shap_cache[tag]

        shap_path = os.path.join(self.config.model_path, f"rf_shap_values_{tag}.joblib")

        if os.path.exists(shap_path):
            logger.info(f"Loading cached SHAP values from {shap_path}")
            cached = joblib.load(shap_path)
            self._rf_shap_cache[tag] = cached
            return cached

        train_features = artifacts["train_features"]
        val_features = artifacts["val_features"]
        val_binary_labels = artifacts["val_binary_labels"]
        n_val = val_features.shape[0]

        n_summary = min(self.config.training.shap_max_samples_summary, n_val)
        n_interact = min(self.config.training.shap_max_samples_interaction, n_val)

        # NOTE: use a global config seed instead of hard-coding
        rng = np.random.default_rng(self.config.rf.seed)
        summary_indices = np.sort(rng.choice(n_val, size=n_summary, replace=False))
        interaction_indices = np.sort(rng.choice(n_val, size=n_interact, replace=False))

        n_workers = self.config.manager.n_processes
        logger.info(
            f"Computing SHAP across {n_workers} worker(s) — summary ({n_summary}), interaction "
            f"({n_interact}), log-loss ({n_summary}). See docs/TRAINING_PIPELINE.md for the passes."
        )

        # shap's TreeSHAP C extension is single-threaded, so we chunk the samples across processes
        # (aetherscan.shap_parallel), each rebuilding a stock TreeExplainer — byte-identical to the
        # serial result. Workers load the RF from its persisted joblib, so make sure it is on disk.
        rf_path = os.path.join(self.config.model_path, f"random_forest_{tag}.joblib")
        # Always (re)dump so the workers load exactly the in-process model — a stale on-disk RF under
        # a reused tag would otherwise silently diverge from the expected_value computed below.
        joblib.dump(self.rf_model.model, rf_path)

        # Expected value (positive-class base value): one lightweight in-process explainer.
        # _silence_stderr() drops SHAP's tqdm (which the logger would otherwise forward to Slack).
        with _silence_stderr():
            ev = shap.TreeExplainer(self.rf_model.model).expected_value
        if isinstance(ev, list | tuple | np.ndarray):
            ev_arr = np.asarray(ev)
            expected_value = float(ev_arr[1]) if ev_arr.size > 1 else float(ev_arr.flat[0])
        else:
            expected_value = float(ev)

        shap_values_summary = parallel_shap(
            rf_path, val_features[summary_indices], "summary", n_workers
        )
        shap_values_interaction = parallel_shap(
            rf_path, val_features[interaction_indices], "interaction", n_workers
        )

        # Log-loss (interventional) decomposition: needs a background subset + per-sample y.
        n_bg = min(1000, train_features.shape[0])
        bg_indices = rng.choice(train_features.shape[0], size=n_bg, replace=False)
        background = train_features[bg_indices]
        try:
            shap_values_logloss = parallel_shap(
                rf_path,
                val_features[summary_indices],
                "logloss",
                n_workers,
                background=background,
                y=val_binary_labels[summary_indices],
            )
        except Exception as e:
            logger.warning(
                f"SHAP log-loss decomposition failed ({e}); falling back to zeros — "
                f"loss-monitoring plot will be empty"
            )
            shap_values_logloss = np.zeros_like(shap_values_summary)

        del background, bg_indices

        # NOTE: come back to this later (what does any of this mean?)
        result = {
            "shap_values_summary": shap_values_summary.astype(np.float32),
            "summary_indices": summary_indices,
            "shap_values_interaction": shap_values_interaction.astype(np.float32),
            "interaction_indices": interaction_indices,
            "shap_values_logloss": shap_values_logloss.astype(np.float32),
            "expected_value": expected_value,
        }
        os.makedirs(os.path.dirname(shap_path), exist_ok=True)
        joblib.dump(result, shap_path)
        logger.info(f"Saved SHAP values to {shap_path}")
        self._rf_shap_cache[tag] = result
        return result

    def _clear_rf_caches(self) -> None:
        """Drop in-memory RF eval-artifact and SHAP caches to free memory."""
        if self._rf_artifacts_cache:
            logger.info(
                f"Clearing RF eval-artifact cache ({len(self._rf_artifacts_cache)} entries)"
            )
            self._rf_artifacts_cache.clear()
        if self._rf_shap_cache:
            logger.info(f"Clearing RF SHAP cache ({len(self._rf_shap_cache)} entries)")
            self._rf_shap_cache.clear()
        gc.collect()

    def _clear_latent_viz_data(self) -> None:
        """
        Free the withheld latent viz batch (plus its labels and log-norm parameters).

        Called from the stage machine once the vae_plots stage has run (the traversal plot is
        the batch's last consumer) and from run_training_pipeline's cleanup on any earlier
        failure; a resumed attempt rebuilds the batch from the resumed round's val split.
        Idempotent.
        """
        self._latent_viz_batch = None
        self._latent_viz_labels = None
        self._latent_viz_lognorm_params = None
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
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                dir,
                f"beta_vae_loss_curves_{tag}.png",
            )
        else:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                f"beta_vae_loss_curves_{tag}.png",
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
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                dir,
                f"beta_vae_training_stability_{tag}.png",
            )
        else:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                f"beta_vae_training_stability_{tag}.png",
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
        Query snr_range_floor and snr_range_ceil from training_stats for the current run and
        return a dict mapping round_number to {"floor": x, "ceil": y}. current_time bounds the
        query's upper time range.
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
        Overlay per-round SNR range shading on a matplotlib axis, with alternating light blue /
        striped light orange backgrounds and optional "SNR: floor-ceil" text annotations.

        snr_by_round is the dict produced by _get_snr_by_round. When use_rounds is False the
        x-axis is in epochs and epochs_per_round must be supplied so round boundaries can be
        located; when True the round number itself is the x-axis coordinate.
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
    # TODO: move injection plots to data_generation.py & call at end of generate_round_to_memmap() (instead of at the end of train_round() & run_training_pipeline())
    # NOTE: there's a ton of improvements we could make to this function (and subsequent _plot functions), but i just care that it works well enough for now
    def plot_injection_stats(self, tag: str | None = None, dir: str | None = None):
        """
        Generate 8 figures for bias/leakage analysis of the injection pipeline: 1 injected signal
        characteristics, 1 injection stability, 4 global intensity distributions (one per
        signal_type), 1 A->B global intensity biases, and 1 final global intensity biases.

        tag defaults to config.checkpoint.save_tag and is used in filenames; dir is an optional
        subdirectory under plots/ (e.g. "checkpoints" for per-round outputs).
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        if dir is not None:
            save_dir = os.path.join(
                self.config.output_path, "plots", "training", self.config.checkpoint.save_tag, dir
            )
        else:
            save_dir = os.path.join(
                self.config.output_path, "plots", "training", self.config.checkpoint.save_tag
            )
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
        rng = derive_rng(self.config.training.seed, STREAM_PLOT)

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
                            sampled_normal = rng.choice(
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
        Generate GIF(s) showing how the latent space evolves during training using UMAP.

        Fits and persists two parallel UMAP projections per (n_neighbors, min_dist) combo:
          - obs-level: each point is a single observation. The latent space is viewed as
            (N * num_observations, latent_dim), stratified into 8 classes
            (4 signal types × ON/OFF). Matches how the VAE sees the data.
          - cadence-level: each point is a full cadence with its num_observations latent
            vectors concatenated, shape (N, num_observations * latent_dim), stratified
            into 4 classes by signal type only. Matches how the RF sees the data.

        Both UMAP models are persisted to disk (joblib) under `{model_path}/umap_{obs,
        cadence}_nn{nn}_md{md}_{tag}.joblib` so downstream visualizations can reuse them
        (e.g. `plot_rf_latent_decision_boundary`). tag defaults to config.checkpoint.save_tag
        and is used in filenames; dir is an optional subdirectory under plots/ (e.g.
        "checkpoints" for per-round outputs).
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        if dir is not None:
            save_dir = os.path.join(
                self.config.output_path, "plots", "training", self.config.checkpoint.save_tag, dir
            )
        else:
            save_dir = os.path.join(
                self.config.output_path, "plots", "training", self.config.checkpoint.save_tag
            )

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

        # Load selected snapshots and build two parallel views of the latent data:
        #   - obs-level: each observation is its own point  → (N * num_observations, latent_dim)
        #   - cadence-level: all num_observations stacked per cadence → (N, num_observations * latent_dim)
        num_observations = self.config.data.num_observations
        all_coords_obs = []  # List of (N_obs, latent_dim) arrays per snapshot
        all_snapshot_labels_obs = []  # List of (N_obs,) label arrays per snapshot
        all_snapshot_onoff_obs = []  # List of (N_obs,) ON/OFF arrays per snapshot
        all_coords_cadence = []  # List of (N_cadences, num_obs*latent_dim) arrays per snapshot
        all_snapshot_labels_cadence = []  # List of (N_cadences,) label arrays per snapshot
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

            # Parse latent vectors and build both views
            obs_latents = []
            obs_labels = []
            obs_onoff = []
            cadence_latents = []
            cadence_labels = []

            for row in rows:
                latent_stack = json.loads(row["latent_vector"])  # (num_observations, latent_dim)
                if len(latent_stack) != num_observations:
                    logger.warning(
                        f"Row has {len(latent_stack)} observations, expected {num_observations}; "
                        f"skipping"
                    )
                    continue
                # Cadence-level: concatenate all observations into a single vector.
                # np.ravel() on a (num_observations, latent_dim) array matches the row-major
                # flatten used by prepare_latent_features (obs_0_dim_0..obs_0_dim_{d-1},
                # obs_1_dim_0..., ...), keeping the cadence-level UMAP fit consistent with
                # the RF feature ordering.
                cadence_vec = np.asarray(latent_stack, dtype=np.float32).ravel()
                cadence_latents.append(cadence_vec)
                cadence_labels.append(row["signal_type"])
                # Obs-level: one entry per observation, with ON/OFF position preserved
                for obs_idx, vec in enumerate(latent_stack):
                    obs_latents.append(vec)
                    obs_labels.append(row["signal_type"])
                    obs_onoff.append("ON" if obs_idx % 2 == 0 else "OFF")

            all_coords_obs.append(np.array(obs_latents, dtype=np.float32))
            all_snapshot_labels_obs.append(obs_labels)
            all_snapshot_onoff_obs.append(obs_onoff)
            all_coords_cadence.append(np.array(cadence_latents, dtype=np.float32))
            all_snapshot_labels_cadence.append(cadence_labels)
            snapshot_metadata.append(key)

            del rows, obs_latents, obs_labels, obs_onoff, cadence_latents, cadence_labels
            gc.collect()

        del snapshot_keys
        gc.collect()

        # NOTE: come back to this later (why are we only checking all_coords_obs and not all_coords_cadence?)
        if not all_coords_obs:
            logger.warning("No valid latent data loaded — skipping GIF generation")
            return

        umap_fit_max = self.config.training.latent_viz_umap_fit_max_samples

        def _build_stratified_fit_pool(all_coords, strata_list, mode_label):
            """
            Concatenate per-snapshot latent arrays and draw a stratified sample to feed UMAP.fit().

            Fitting UMAP on the full pool is slow; a stratified subsample generalizes well
            and the remaining vectors are projected through .transform() afterwards.
            """
            pooled = np.concatenate(all_coords, axis=0)
            logger.info(
                f"Pooled {pooled.shape[0]} {mode_label} latent vectors from "
                f"{len(all_coords)} snapshots for UMAP fitting"
            )

            if pooled.shape[0] <= umap_fit_max:
                return pooled

            strata = np.concatenate([np.array(s, dtype="U") for s in strata_list])

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
                        f"Only {n_take} {mode_label} latents for {cls} "
                        f"(requested {per_class}), using all available"
                    )
                fit_indices.append(rng.choice(cls_idx, size=n_take, replace=False))
            fit_indices = np.concatenate(fit_indices)
            del strata

            fit_pool = pooled[fit_indices]
            logger.info(
                f"Stratified subsampled {fit_pool.shape[0]} / {pooled.shape[0]} "
                f"{mode_label} latent vectors for UMAP fit ({len(unique_classes)} classes, "
                f"~{per_class} per class)"
            )
            del pooled, fit_indices
            return fit_pool

        # Obs-level strata: signal_type × ON/OFF (8 classes)
        obs_strata_list = [
            np.char.add(
                np.char.add(np.array(lab, dtype="U"), "|"),
                np.array(onoff, dtype="U"),
            )
            for lab, onoff in zip(all_snapshot_labels_obs, all_snapshot_onoff_obs, strict=True)
        ]
        fit_pool_obs = _build_stratified_fit_pool(all_coords_obs, obs_strata_list, "obs-level")
        del obs_strata_list

        # Cadence-level strata: signal_type only (4 classes)
        fit_pool_cadence = _build_stratified_fit_pool(
            all_coords_cadence, all_snapshot_labels_cadence, "cadence-level"
        )

        # Compute consistent axis limits with 5% padding (streaming min/max to avoid concat)
        def _compute_limits(transformed_list):
            x_min = min(t[:, 0].min() for t in transformed_list)
            x_max = max(t[:, 0].max() for t in transformed_list)
            y_min = min(t[:, 1].min() for t in transformed_list)
            y_max = max(t[:, 1].max() for t in transformed_list)
            x_pad = (x_max - x_min) * 0.05
            y_pad = (y_max - y_min) * 0.05
            return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)

        # NOTE: come back to this later
        # Obs-level palette: 8 categories (signal_type × ON/OFF) with ON/OFF as triangle/x
        obs_colors = {
            ("false_no_signal", "ON"): "#1565C0",
            ("false_no_signal", "OFF"): "#64B5F6",
            ("false_with_rfi", "ON"): "#F9A825",
            ("false_with_rfi", "OFF"): "#FFF176",
            ("true_only_eti", "ON"): "#2E7D32",
            ("true_only_eti", "OFF"): "#81C784",
            ("true_eti_rfi", "ON"): "#C62828",
            ("true_eti_rfi", "OFF"): "#EF5350",
        }
        obs_markers = {"ON": "^", "OFF": "x"}
        obs_display_names = {
            ("false_no_signal", "ON"): "No Signal (ON)",
            ("false_no_signal", "OFF"): "No Signal (OFF)",
            ("false_with_rfi", "ON"): "RFI Only (ON)",
            ("false_with_rfi", "OFF"): "RFI Only (OFF)",
            ("true_only_eti", "ON"): "ETI Only (ON)",
            ("true_only_eti", "OFF"): "ETI Only (OFF)",
            ("true_eti_rfi", "ON"): "ETI+RFI (ON)",
            ("true_eti_rfi", "OFF"): "ETI+RFI (OFF)",
        }

        # NOTE: come back to this later (change colors, add markers. potentially need to change _render_frames_and_write_gif: docstring & collapse mode arg?)
        # Cadence-level palette: 4 categories, kept distinct from the obs palette on
        # purpose so downstream RF plots can reuse the same colors
        cadence_colors = {
            "false_no_signal": "tab:blue",
            "false_with_rfi": "tab:green",
            "true_only_eti": "tab:red",
            "true_eti_rfi": "tab:orange",
        }
        cadence_display_names = {
            "false_no_signal": "No Signal",
            "false_with_rfi": "RFI Only",
            "true_only_eti": "ETI Only",
            "true_eti_rfi": "ETI+RFI",
        }

        # NOTE: instead of temp_dir, save frames in persistent dir. update dir archiving to handle
        temp_dir = tempfile.mkdtemp(prefix="latent_gif_")

        gif_paths = {}
        duration_ms = self.config.training.latent_viz_gif_duration_ms

        n_neighbors_values = self.config.training.latent_viz_umap_n_neighbors
        min_dist_values = self.config.training.latent_viz_umap_min_dist

        def _render_frames_and_write_gif(
            transformed_list,
            method_name,
            display_method,
            xlim,
            ylim,
            mode,
        ):
            """
            Render one scatter frame per snapshot and assemble them into a GIF.

            mode="obs"     → 8-category (signal_type × ON/OFF) scatter with triangle/x markers
            mode="cadence" → 4-category (signal_type) scatter with single-marker style
            """
            frame_paths = []
            for frame_idx, meta in enumerate(snapshot_metadata):
                coords_2d = transformed_list[frame_idx]
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))

                if mode == "obs":
                    labels_arr = np.array(all_snapshot_labels_obs[frame_idx])
                    onoff_arr = np.array(all_snapshot_onoff_obs[frame_idx])
                    for (stype, status), color in obs_colors.items():
                        mask = (labels_arr == stype) & (onoff_arr == status)
                        if mask.any():
                            ax.scatter(
                                coords_2d[mask, 0],
                                coords_2d[mask, 1],
                                c=color,
                                marker=obs_markers[status],
                                s=10,
                                alpha=0.75,
                                label=obs_display_names[(stype, status)],
                                rasterized=True,
                            )
                    del labels_arr, onoff_arr
                    legend_kwargs = {
                        "loc": "upper right",
                        "fontsize": 8,
                        "markerscale": 2,
                        "ncol": 2,
                        "framealpha": 0.8,
                    }
                else:  # cadence
                    labels_arr = np.array(all_snapshot_labels_cadence[frame_idx])
                    for stype, color in cadence_colors.items():
                        mask = labels_arr == stype
                        if mask.any():
                            ax.scatter(
                                coords_2d[mask, 0],
                                coords_2d[mask, 1],
                                c=color,
                                marker="o",
                                s=10,
                                alpha=0.75,
                                label=cadence_display_names[stype],
                                rasterized=True,
                            )
                    del labels_arr
                    legend_kwargs = {
                        "loc": "upper right",
                        "fontsize": 8,
                        "markerscale": 2,
                        "framealpha": 0.8,
                    }

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
                    f"Beta-VAE Latent Space: {display_method} — "
                    f"Round {meta['round_number']}, "
                    f"Epoch {meta['epoch_number']}, "
                    f"Step {meta['step_number']} "
                    f"(SNR: {meta_snr_floor}–{meta_snr_ceil})",
                    fontsize=11,
                )
                ax.legend(**legend_kwargs)

                plt.tight_layout()

                frame_path = os.path.join(temp_dir, f"{method_name}_frame_{frame_idx:05d}.png")
                fig.savefig(frame_path, dpi=100)
                plt.close(fig)
                frame_paths.append(frame_path)

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
                    f"Latent space {method_name.upper()} GIF saved: {gif_path} ({n_frames} frames)"
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

            return gif_path

        for nn in n_neighbors_values:
            for md in min_dist_values:
                # Obs-level UMAP
                logger.info(f"Fitting obs-level UMAP with n_neighbors={nn}, min_dist={md}")
                # NOTE: use a global config seed instead of hard-coding
                # Note that by setting random_state, we get a deterministic UMAP fit, at the
                # expense of single-thread performance (n_jobs=1). This is a hard constraint of
                # the UMAP library. We compensate by fitting UMAP on a stratified subsample.
                umap_obs = umap.UMAP(
                    n_components=2,
                    random_state=11,
                    n_neighbors=nn,
                    min_dist=md,
                ).fit(fit_pool_obs)

                obs_umap_path = os.path.join(
                    self.config.model_path, f"umap_obs_nn{nn}_md{md}_{tag}.joblib"
                )
                try:
                    os.makedirs(self.config.model_path, exist_ok=True)
                    joblib.dump(umap_obs, obs_umap_path)
                    logger.info(f"Saved obs-level UMAP model: {obs_umap_path}")
                except Exception as exc:
                    logger.warning(
                        f"Failed to persist obs-level UMAP model ({obs_umap_path}): {exc}"
                    )

                transformed_obs = [umap_obs.transform(c) for c in all_coords_obs]
                del umap_obs
                gc.collect()

                xlim_obs, ylim_obs = _compute_limits(transformed_obs)
                method_name_obs = f"obs_umap_nn{nn}_md{md}"
                display_method_obs = f"Obs-level UMAP (n_neighbors: {nn}, min_dist: {md})"

                gif_path_obs = _render_frames_and_write_gif(
                    transformed_obs,
                    method_name_obs,
                    display_method_obs,
                    xlim_obs,
                    ylim_obs,
                    mode="obs",
                )
                gif_paths[method_name_obs] = gif_path_obs

                del transformed_obs
                gc.collect()

                # Cadence-level UMAP
                logger.info(f"Fitting cadence-level UMAP with n_neighbors={nn}, min_dist={md}")
                # NOTE: use a global config seed instead of hard-coding
                # Note that by setting random_state, we get a deterministic UMAP fit, at the
                # expense of single-thread performance (n_jobs=1). This is a hard constraint of
                # the UMAP library. We compensate by fitting UMAP on a stratified subsample.
                umap_cadence = umap.UMAP(
                    n_components=2,
                    random_state=11,
                    n_neighbors=nn,
                    min_dist=md,
                ).fit(fit_pool_cadence)

                cadence_umap_path = os.path.join(
                    self.config.model_path, f"umap_cadence_nn{nn}_md{md}_{tag}.joblib"
                )
                try:
                    os.makedirs(self.config.model_path, exist_ok=True)
                    joblib.dump(umap_cadence, cadence_umap_path)
                    logger.info(f"Saved cadence-level UMAP model: {cadence_umap_path}")
                except Exception as exc:
                    logger.warning(
                        f"Failed to persist cadence-level UMAP model ({cadence_umap_path}): {exc}"
                    )

                transformed_cadence = [umap_cadence.transform(c) for c in all_coords_cadence]
                del umap_cadence
                gc.collect()

                xlim_cad, ylim_cad = _compute_limits(transformed_cadence)
                method_name_cad = f"cadence_umap_nn{nn}_md{md}"
                display_method_cad = f"Cadence-level UMAP (n_neighbors: {nn}, min_dist: {md})"

                gif_path_cad = _render_frames_and_write_gif(
                    transformed_cadence,
                    method_name_cad,
                    display_method_cad,
                    xlim_cad,
                    ylim_cad,
                    mode="cadence",
                )
                gif_paths[method_name_cad] = gif_path_cad

                del transformed_cadence
                gc.collect()

        # Cleanup — leave closure-captured variables for Python's scope teardown rather
        # than `del`ing them here (ruff F821 flags closure refs when the enclosing
        # scope explicitly deletes the name, even if the closure already ran).
        # NOTE: temp_dir isn't cleaned on exception (should use try/finally or tempfile.TemporaryDirectory() with context manager)
        shutil.rmtree(temp_dir, ignore_errors=True)
        gc.collect()

    def plot_latent_traversal(self, tag: str | None = None, dir: str | None = None):
        """
        Decoder-based interpretation of the latent dimensions: for each signal type, decode
        z_t + s·σ_d·e_d for every latent dim d and step s ∈ linspace(-max_sigma, +max_sigma,
        num_steps), where z_t is the mean encoder z_mean over that type's ON observations
        (indices 0/2/4) from the withheld latent viz batch and σ_d is the per-dim std of
        z_mean over the whole batch. Two figures per signal type:

        - latent_traversal_{signal_type}_{tag}.png: latent_dim × num_steps waterfall grid
          (shared per-row color scale); the center column is the unperturbed class-mean decode.
        - latent_traversal_spectra_{signal_type}_{tag}.png: per-dim time-integrated spectra,
          one line per step (step colormap) — brightness/drift/width shifts read off easily.

        Display un-preprocessing is an honest approximation, stated on each figure (see
        unpreprocess_traversal_panels). Encoding/decoding uses the plain non-distributed
        models (the viz batch is ≤ ~960 cadences — a trivial single pass, and keeping the
        plot independent of the distributed dataset plumbing). Requires the in-memory viz
        batch, which lives from the first trained round through the vae_plots stage; on a
        resumed run whose beta-VAE rounds were already complete there is nothing to encode
        and the plot is skipped with a warning.
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        if self._latent_viz_batch is None or self._latent_viz_labels is None:
            logger.warning(
                "plot_latent_traversal: no latent viz batch available (e.g. a resumed run "
                "whose beta-VAE rounds were already complete) — skipping traversal figures"
            )
            return

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        num_steps = self.config.training.latent_traversal_num_steps
        max_sigma = self.config.training.latent_traversal_max_sigma
        latent_dim = self.config.beta_vae.latent_dim
        num_obs = self.config.data.num_observations
        time_bins = self.config.data.time_bins
        downsample_factor = self.config.data.downsample_factor
        width_bin = self.config.data.width_bin // downsample_factor

        batch = np.asarray(self._latent_viz_batch, dtype=np.float32)
        labels = self._latent_viz_labels
        n_cadences = batch.shape[0]

        # Encode the whole viz batch in one simple non-distributed pass, chunked by cadence
        # (each chunk expands to chunk x num_obs observations at the encoder) so a large viz
        # batch can't spike device memory (the encoder was created inside strategy.scope()
        # but runs fine on the default device)
        chunk = max(1, self.config.training.per_replica_val_batch_size)
        z_parts = []
        for start in range(0, n_cadences, chunk):
            observations = batch[start : start + chunk].reshape(-1, time_bins, width_bin, 1)
            z_mean_part, _, _ = self.vae.encoder(tf.convert_to_tensor(observations), training=False)
            z_parts.append(np.asarray(z_mean_part))
        z_mean = np.concatenate(z_parts, axis=0).reshape(n_cadences, num_obs, latent_dim)
        del z_parts

        # Per-dim σ over the whole viz batch (all observations pooled) — sets each dim's
        # traversal scale in units the encoder actually uses
        sigmas = z_mean.reshape(-1, latent_dim).std(axis=0)
        if np.all(sigmas == 0):
            # Fully collapsed latents (e.g. a degenerate/untrained encoder): every traversal
            # step would decode to the same image — skip rather than render 8 blank grids
            logger.warning(
                "plot_latent_traversal: all per-dim sigmas are zero (collapsed latents) — "
                "skipping traversal figures"
            )
            return

        on_indices = np.arange(0, num_obs, 2)  # ON observations (ABACAD -> indices 0/2/4)

        def decode_fn(latents):
            # Single-shot decode: the batch is bounded by latent_dim * num_steps rows (56 at
            # defaults; even num_steps=99 is ~800 (16, 512, 1) reconstructions ≈ 26 MB) — no
            # chunking needed, unlike the cadence-count-scaled encoding loop above
            return np.asarray(self.vae.decoder(tf.convert_to_tensor(latents), training=False))

        signal_types = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]
        type_display_names = {
            "false_no_signal": "No Signal",
            "false_with_rfi": "RFI Only",
            "true_only_eti": "ETI Only",
            "true_eti_rfi": "ETI + RFI",
        }

        for signal_type in signal_types:
            mask = labels == signal_type
            if not mask.any():
                logger.warning(
                    f"plot_latent_traversal: no viz cadences for {signal_type} — skipping"
                )
                continue
            display_name = type_display_names[signal_type]

            # Base vector: mean z_mean over this type's ON observations
            z_t = z_mean[mask][:, on_indices, :].mean(axis=(0, 1))

            # Class-mean log-norm params over this type's ON observations, for the approximate
            # display inversion. Collapsing the per-observation (6, 2) params to one
            # (min_log, range_log) pair is itself part of the approximation — a traversal decode
            # blends observations — and combined with the lossy frequency downsampling the
            # inverted intensity is DISPLAY-ONLY, never exact. None -> plot in normalized log space.
            lognorm = None
            if self._latent_viz_lognorm_params is not None:
                on_params = self._latent_viz_lognorm_params[mask][:, on_indices, :]
                lognorm = (float(on_params[..., 0].mean()), float(on_params[..., 1].mean()))

            panels, steps = compute_traversal_panels(z_t, sigmas, num_steps, max_sigma, decode_fn)
            panels, inverted = unpreprocess_traversal_panels(panels, lognorm, downsample_factor)

            # State the approximation on the figure — exact inversion is impossible
            # (downsampling is lossy; log-norm params are per-observation, a decode blends many)
            if inverted:
                caption = (
                    "Approximate un-preprocessing: log-norm inverted with class-mean ON-obs "
                    f"params (exp(x·range_log + min_log)); frequency ×{downsample_factor} "
                    "nearest-neighbor upsampled. Exact inversion impossible (lossy downsample, "
                    "per-observation params)."
                )
            else:
                caption = (
                    "Intensity in normalized log space (log-norm params unavailable or "
                    f"degenerate); frequency ×{downsample_factor} nearest-neighbor upsampled. "
                    "Exact un-preprocessing impossible (lossy downsample)."
                )

            self._render_traversal_waterfalls(
                panels, steps, sigmas, display_name, signal_type, caption, tag, dir, machine_name
            )
            self._render_traversal_spectra(
                panels,
                steps,
                sigmas,
                inverted,
                display_name,
                signal_type,
                caption,
                tag,
                dir,
                machine_name,
            )

            del panels

        del batch, z_mean
        gc.collect()

    def _save_traversal_figure(self, fig, filename: str, dir: str | None, slack_title: str):
        """Standard plot tail shared by the two traversal renderers: save under the plots
        directory (optionally nested in `dir`), close the figure, and upload to Slack."""
        if dir is not None:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                dir,
                filename,
            )
        else:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                filename,
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Latent traversal plot saved to: {save_path}")

        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(save_path, title=slack_title)

    def _render_traversal_waterfalls(
        self, panels, steps, sigmas, display_name, signal_type, caption, tag, dir, machine_name
    ):
        """Primary traversal figure: latent_dim × num_steps grid of decoded waterfalls with a
        shared color scale per row (per dim), rows labeled dim/σ, columns labeled step
        multiples. panels has shape (latent_dim, num_steps, time, freq_display)."""
        latent_dim, num_steps = panels.shape[0], panels.shape[1]

        fig, axes = plt.subplots(
            latent_dim,
            num_steps,
            figsize=(1.9 * num_steps + 1.5, 1.5 * latent_dim + 1.8),
            squeeze=False,
        )
        fig.suptitle(
            f"Latent Traversal — {display_name} ({tag}, {machine_name})",
            fontsize=16,
            fontweight="bold",
        )
        fig.text(0.5, 0.955, caption, ha="center", fontsize=8, style="italic", wrap=True)

        for d in range(latent_dim):
            # Shared color scale per row so intensity changes read across steps
            vmin = float(panels[d].min())
            vmax = float(panels[d].max())
            if vmax <= vmin:
                vmax = vmin + 1e-12
            for s in range(num_steps):
                ax = axes[d][s]
                ax.imshow(
                    panels[d, s],
                    aspect="auto",
                    origin="lower",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    extent=(0, panels.shape[3], 0, panels.shape[2]),
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if d == 0:
                    # The center column (odd, validated step count) is the unperturbed decode
                    is_center = num_steps % 2 == 1 and s == num_steps // 2
                    ax.set_title("0σ (base)" if is_center else f"{steps[s]:+.3g}σ", fontsize=10)
                if s == 0:
                    ax.set_ylabel(f"dim {d}\n(σ={sigmas[d]:.3f})", fontsize=9)
        axes[-1][num_steps // 2].set_xlabel(
            "Frequency bin (full resolution) — each panel: time × frequency", fontsize=10
        )

        plt.tight_layout(rect=(0, 0, 1, 0.94))

        self._save_traversal_figure(
            fig,
            f"latent_traversal_{signal_type}_{tag}.png",
            dir,
            slack_title=f"Latent Traversal ({display_name}) - ({tag}, {machine_name})",
        )

    def _render_traversal_spectra(
        self,
        panels,
        steps,
        sigmas,
        inverted,
        display_name,
        signal_type,
        caption,
        tag,
        dir,
        machine_name,
    ):
        """Secondary traversal figure: one panel per latent dim showing the time-integrated
        spectrum of every traversal step (colormap over steps) — brightness/drift/width shifts
        read off more easily than in the waterfall grid."""
        latent_dim, num_steps = panels.shape[0], panels.shape[1]
        ncols = 4
        nrows = (latent_dim + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows + 1.0), squeeze=False
        )
        fig.suptitle(
            f"Latent Traversal Spectra — {display_name} ({tag}, {machine_name})",
            fontsize=16,
            fontweight="bold",
        )
        fig.text(0.5, 0.935, caption, ha="center", fontsize=8, style="italic", wrap=True)

        cmap = plt.cm.coolwarm
        norm = plt.Normalize(float(steps[0]), float(steps[-1]))
        freq_bins = np.arange(panels.shape[3])
        intensity_label = (
            "Mean intensity (approx. linear)" if inverted else "Mean intensity (normalized log)"
        )

        for d in range(latent_dim):
            ax = axes[d // ncols][d % ncols]
            for s in range(num_steps):
                ax.plot(
                    freq_bins,
                    panels[d, s].mean(axis=0),  # Time-integrated spectrum
                    color=cmap(norm(float(steps[s]))),
                    linewidth=1.2,
                )
            ax.set_title(f"dim {d} (σ={sigmas[d]:.3f})", fontsize=11)
            ax.grid(True, alpha=0.3)
            # Label the bottom-most panel in each column (no panel d+ncols below it), so a
            # partial last row (latent_dim not a multiple of ncols) still labels every column
            if d + ncols >= latent_dim:
                ax.set_xlabel("Frequency bin (full resolution)", fontsize=10)
            if d % ncols == 0:
                ax.set_ylabel(intensity_label, fontsize=10)

        # Hide any unused grid slots (latent_dim not divisible by ncols)
        for idx in range(latent_dim, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        # Step colorbar (skip tight_layout — it doesn't cooperate with a stolen-axes colorbar)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        fig.colorbar(
            mappable,
            ax=axes.ravel().tolist(),
            label="Traversal step (× per-dim σ)",
            shrink=0.85,
        )

        self._save_traversal_figure(
            fig,
            f"latent_traversal_spectra_{signal_type}_{tag}.png",
            dir,
            slack_title=f"Latent Traversal Spectra ({display_name}) - ({tag}, {machine_name})",
        )

    # TODO: implement plot_rf_snr_sensitivity_curve()
    def plot_rf_confusion_matrices(self, tag: str | None = None, dir: str | None = None):
        """
        Confusion matrices for the RF — binary (true vs false) and 4-way sub-type.
        1×2 grid: left is the 2×2 binary matrix, right is the 4×2 sub-type matrix.
        Both with counts + rates.
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        artifacts = self._load_rf_eval_artifacts(tag)
        val_binary = artifacts["val_binary_labels"]
        val_subtype = artifacts["val_subtype_labels"]
        val_probas = artifacts["val_probas"]
        val_preds = artifacts["val_preds"]
        classification_threshold = artifacts["classification_threshold"]

        # Binary confusion matrix
        cm_binary = confusion_matrix(val_binary, val_preds, labels=[0, 1])
        row_sums = cm_binary.sum(axis=1, keepdims=True)
        cm_binary_norm = np.divide(
            cm_binary,
            row_sums,
            out=np.zeros_like(cm_binary, dtype=np.float64),
            where=row_sums > 0,
        )

        # Sub-type confusion matrix (rows=4 sub-types, cols=binary prediction)
        signal_types = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]
        display_names = {
            "false_no_signal": "No Signal",
            "false_with_rfi": "RFI Only",
            "true_only_eti": "ETI Only",
            "true_eti_rfi": "ETI + RFI",
        }
        cm_subtype = np.zeros((4, 2), dtype=np.int64)
        for row_idx, stype in enumerate(signal_types):
            mask = val_subtype == stype
            if not mask.any():
                continue
            preds_for_type = val_preds[mask]
            cm_subtype[row_idx, 0] = int(np.sum(preds_for_type == 0))
            cm_subtype[row_idx, 1] = int(np.sum(preds_for_type == 1))
        subtype_row_sums = cm_subtype.sum(axis=1, keepdims=True)
        cm_subtype_norm = np.divide(
            cm_subtype,
            subtype_row_sums,
            out=np.zeros_like(cm_subtype, dtype=np.float64),
            where=subtype_row_sums > 0,
        )

        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.25)
        ax_binary = fig.add_subplot(gs[0, 0])
        ax_subtype = fig.add_subplot(gs[0, 1])

        fig.suptitle(
            f"Random Forest Confusion Matrices (t={classification_threshold:.2f}, {tag}, {machine_name})",
            fontsize=15,
            fontweight="bold",
        )

        # Binary heatmap
        ax_binary.imshow(cm_binary_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        ax_binary.set_title("Binary", fontsize=12, fontweight="bold")
        ax_binary.set_xticks([0, 1])
        ax_binary.set_xticklabels(["Pred False", "Pred True"], fontsize=10)
        ax_binary.set_yticks([0, 1])
        ax_binary.set_yticklabels(["Actual False", "Actual True"], fontsize=10)
        for i in range(2):
            for j in range(2):
                txt_color = "white" if cm_binary_norm[i, j] > 0.5 else "black"
                ax_binary.text(
                    j,
                    i,
                    f"{cm_binary[i, j]}\n({cm_binary_norm[i, j] * 100:.1f}%)",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=11,
                    fontweight="bold",
                )

        # Sub-type heatmap
        ax_subtype.imshow(cm_subtype_norm, cmap="Oranges", vmin=0.0, vmax=1.0, aspect="auto")
        ax_subtype.set_title("Sub-Type", fontsize=12, fontweight="bold")
        ax_subtype.set_xticks([0, 1])
        ax_subtype.set_xticklabels(["Pred False", "Pred True"], fontsize=10)
        ax_subtype.set_yticks(range(4))
        ax_subtype.set_yticklabels([display_names[s] for s in signal_types], fontsize=10)
        for i in range(4):
            for j in range(2):
                txt_color = "white" if cm_subtype_norm[i, j] > 0.5 else "black"
                ax_subtype.text(
                    j,
                    i,
                    f"{cm_subtype[i, j]}\n({cm_subtype_norm[i, j] * 100:.1f}%)",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=10,
                    fontweight="bold",
                )

        plt.tight_layout()

        if dir is not None:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                dir,
                f"rf_confusion_matrices_{tag}.png",
            )
        else:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                f"rf_confusion_matrices_{tag}.png",
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"RF confusion matrices plot saved to: {save_path}")

        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"RF Confusion Matrices - ({tag}, {machine_name})",
            )

        del artifacts, cm_binary, cm_binary_norm, cm_subtype, cm_subtype_norm
        del val_binary, val_subtype, val_probas, val_preds
        gc.collect()

    def plot_rf_classification_curves(self, tag: str | None = None, dir: str | None = None):
        """
        2x2 grid: ROC + AUC, PR + AP, confidence histogram overall, confidence histogram
        per sub-type.
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        artifacts = self._load_rf_eval_artifacts(tag)
        val_binary = artifacts["val_binary_labels"]
        val_subtype = artifacts["val_subtype_labels"]
        val_probas = artifacts["val_probas"]

        fpr, tpr, roc_thresholds = roc_curve(val_binary, val_probas)
        roc_auc = auc(fpr, tpr)
        precision, recall, pr_thresholds = precision_recall_curve(val_binary, val_probas)
        ap = average_precision_score(val_binary, val_probas)

        signal_types = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]
        subtype_colors = {
            "false_no_signal": "blue",
            "false_with_rfi": "green",
            "true_only_eti": "red",
            "true_eti_rfi": "orange",
        }
        display_names = {
            "false_no_signal": "No Signal",
            "false_with_rfi": "RFI Only",
            "true_only_eti": "ETI Only",
            "true_eti_rfi": "ETI + RFI",
        }

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(
            f"Random Forest Classification Curves ({tag}, {machine_name})",
            fontsize=15,
            fontweight="bold",
        )

        ax_roc, ax_pr = axes[0, 0], axes[0, 1]
        ax_conf_all, ax_conf_sub = axes[1, 0], axes[1, 1]

        # NOTE: come back to this later (what do the annotations mean?)
        # ROC
        ax_roc.plot(fpr, tpr, color="tab:red", linewidth=2, label=f"AUC = {roc_auc:.4f}")
        ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        for thr in [0.3, 0.5, 0.7, 0.9]:
            # Find the operating point closest to this threshold
            idx = int(np.argmin(np.abs(roc_thresholds - thr)))
            if idx < len(fpr):
                ax_roc.scatter(fpr[idx], tpr[idx], color="black", s=40, zorder=5)
                ax_roc.annotate(
                    f"t={thr:.1f}",
                    (fpr[idx], tpr[idx]),
                    textcoords="offset points",
                    xytext=(5, -10),
                    fontsize=8,
                )
        # Operating points for high recall
        for target_recall in [0.95, 0.99]:
            valid = tpr >= target_recall
            if valid.any():
                first_idx = int(np.argmax(valid))
                ax_roc.scatter(
                    fpr[first_idx], tpr[first_idx], color="tab:green", s=60, marker="*", zorder=6
                )
                ax_roc.annotate(
                    f"recall={target_recall:.2f}",
                    (fpr[first_idx], tpr[first_idx]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    color="tab:green",
                )
        ax_roc.set_xlabel("False Positive Rate", fontsize=11)
        ax_roc.set_ylabel("True Positive Rate", fontsize=11)
        ax_roc.set_title("ROC Curve", fontsize=13, fontweight="bold")
        ax_roc.grid(True, alpha=0.3)
        ax_roc.legend(loc="lower right", fontsize=10)
        ax_roc.set_xlim(-0.01, 1.01)
        ax_roc.set_ylim(-0.01, 1.01)

        # PR curve
        ax_pr.plot(recall, precision, color="tab:blue", linewidth=2, label=f"AP = {ap:.4f}")
        for thr in [0.3, 0.5, 0.7, 0.9]:
            idx = int(np.argmin(np.abs(pr_thresholds - thr))) if len(pr_thresholds) else 0
            if idx < len(precision) - 1:
                ax_pr.scatter(recall[idx], precision[idx], color="black", s=40, zorder=5)
                ax_pr.annotate(
                    f"t={thr:.1f}",
                    (recall[idx], precision[idx]),
                    textcoords="offset points",
                    xytext=(5, -10),
                    fontsize=8,
                )
        ax_pr.set_xlabel("Recall", fontsize=11)
        ax_pr.set_ylabel("Precision", fontsize=11)
        ax_pr.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
        ax_pr.grid(True, alpha=0.3)
        ax_pr.legend(loc="lower right", fontsize=10)
        ax_pr.set_xlim(-0.01, 1.01)
        ax_pr.set_ylim(-0.01, 1.01)

        # Overall confidence histogram
        bins = np.linspace(0.0, 1.0, 40)
        true_mask = val_binary == 1
        false_mask = val_binary == 0
        ax_conf_all.hist(
            val_probas[true_mask],
            bins=bins,
            alpha=0.55,
            color="tab:red",
            label="True",
            edgecolor="darkred",
        )
        ax_conf_all.hist(
            val_probas[false_mask],
            bins=bins,
            alpha=0.55,
            color="tab:blue",
            label="False",
            edgecolor="darkblue",
        )
        ax_conf_all.axvline(x=0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax_conf_all.set_xlabel("Predicted P(true)", fontsize=11)
        ax_conf_all.set_ylabel("Count", fontsize=11)
        ax_conf_all.set_title("Confidence Distribution (Binary)", fontsize=13, fontweight="bold")
        ax_conf_all.grid(True, alpha=0.3)
        ax_conf_all.legend(loc="upper center", fontsize=10)

        # Per-subtype confidence histogram
        for stype in signal_types:
            mask = val_subtype == stype
            if not mask.any():
                continue
            ax_conf_sub.hist(
                val_probas[mask],
                bins=bins,
                alpha=0.45,
                color=subtype_colors[stype],
                label=display_names[stype],
                edgecolor=subtype_colors[stype],
            )
        ax_conf_sub.axvline(x=0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax_conf_sub.set_xlabel("Predicted P(true)", fontsize=11)
        ax_conf_sub.set_ylabel("Count", fontsize=11)
        ax_conf_sub.set_title("Confidence Distribution (Sub-Type)", fontsize=13, fontweight="bold")
        ax_conf_sub.grid(True, alpha=0.3)
        ax_conf_sub.legend(loc="upper center", fontsize=9)

        plt.tight_layout()

        if dir is not None:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                dir,
                f"rf_classification_curves_{tag}.png",
            )
        else:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                f"rf_classification_curves_{tag}.png",
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"RF classification curves plot saved to: {save_path}")

        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"RF Classification Curves - ({tag}, {machine_name})",
            )

        del artifacts, fpr, tpr, roc_thresholds, precision, recall, pr_thresholds
        del val_binary, val_subtype, val_probas
        gc.collect()

    def _rf_feature_names(self) -> list[str]:
        """Human-readable names for the 48 flattened latent features.

        Naming follows the cadence convention from data_generation.py: even-indexed
        observations are ON, odd-indexed are OFF. Each ON/OFF pair is numbered 1..3.
        Example for 6 obs / 8 dims: ON-1_dim-0 ... ON-1_dim-7, OFF-1_dim-0 ... OFF-3_dim-7.
        """
        num_obs = self.config.data.num_observations
        latent_dim = self.config.beta_vae.latent_dim
        names = []
        for o in range(num_obs):
            kind = "ON" if o % 2 == 0 else "OFF"
            pair_idx = o // 2 + 1
            for d in range(latent_dim):
                names.append(f"{kind}-{pair_idx}_dim-{d}")
        return names

    def plot_rf_shap_summary(self, tag: str | None = None, dir: str | None = None):
        """
        SHAP beeswarm summary of the top features driving P(true) predictions.
        Reveals which flattened latents (obs×dim) matter most, their directionality, and the
        sample-level spread of their contribution.
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        artifacts = self._load_rf_eval_artifacts(tag)
        shap_data = self._compute_or_load_shap_values(artifacts)

        shap_values = shap_data["shap_values_summary"]
        summary_indices = shap_data["summary_indices"]
        features_sub = artifacts["val_features"][summary_indices]
        feature_names = self._rf_feature_names()

        fig = plt.figure(figsize=(10, 14))
        shap.summary_plot(
            shap_values,
            features_sub,
            feature_names=feature_names,
            show=False,
            plot_size=None,
            max_display=len(feature_names),
        )
        fig = plt.gcf()
        fig.suptitle(
            f"Random Forest SHAP Summary ({tag}, {machine_name})",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        if dir is not None:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                dir,
                f"rf_shap_summary_{tag}.png",
            )
        else:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                f"rf_shap_summary_{tag}.png",
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"RF SHAP summary plot saved to: {save_path}")

        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"RF SHAP Summary - ({tag}, {machine_name})",
            )

        del artifacts, shap_data, shap_values, summary_indices, features_sub
        gc.collect()

    def plot_rf_shap_dependence(self, tag: str | None = None, dir: str | None = None):
        """
        Grid of SHAP dependence plots for the top-K features by mean |SHAP|.
        Each panel plots (feature value) vs (SHAP for that feature) colored by
        the strongest-interacting feature (auto-detected).
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        artifacts = self._load_rf_eval_artifacts(tag)
        shap_data = self._compute_or_load_shap_values(artifacts)

        shap_values = shap_data["shap_values_summary"]
        summary_indices = shap_data["summary_indices"]
        features_sub = artifacts["val_features"][summary_indices]
        feature_names = self._rf_feature_names()

        k = self.config.training.shap_top_k_features_dependence
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        top_k_idx = np.argsort(mean_abs)[::-1][:k]

        # Use a wider grid (8 cols) when plotting many features so panels stay readable
        n_cols = 8 if k > 16 else 4
        n_rows = int(np.ceil(k / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes_flat = axes.flatten() if n_rows * n_cols > 1 else [axes]

        fig.suptitle(
            f"Random Forest SHAP Dependence ({tag}, {machine_name})",
            fontsize=15,
            fontweight="bold",
        )

        for panel_idx, feat_idx in enumerate(top_k_idx):
            ax = axes_flat[panel_idx]
            try:
                shap.dependence_plot(
                    int(feat_idx),
                    shap_values,
                    features_sub,
                    feature_names=feature_names,
                    ax=ax,
                    show=False,
                )
            except Exception as e:
                logger.warning(f"Dependence plot failed for feature {feature_names[feat_idx]}: {e}")
                ax.text(
                    0.5,
                    0.5,
                    f"failed: {feature_names[feat_idx]}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_title(
                f"{feature_names[feat_idx]}  |SHAP|={mean_abs[feat_idx]:.4f}",
                fontsize=11,
                fontweight="bold",
            )

        # Turn off unused subplots
        for panel_idx in range(len(top_k_idx), len(axes_flat)):
            axes_flat[panel_idx].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if dir is not None:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                dir,
                f"rf_shap_dependence_{tag}.png",
            )
        else:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                f"rf_shap_dependence_{tag}.png",
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"RF SHAP dependence plot saved to: {save_path}")

        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"RF SHAP Dependence - ({tag}, {machine_name})",
            )

        del artifacts, shap_data, shap_values, summary_indices, features_sub
        del mean_abs, top_k_idx
        gc.collect()

    def plot_rf_shap_interactions(self, tag: str | None = None, dir: str | None = None):
        """
        Compact summary of SHAP pairwise interaction values across n×n feature pairs.
        The diagonal is the main effect (same as SHAP summary). Off-diagonals are pure interaction
        effects. Strong off-diagonal entries imply the RF exploits cross-observation structure
        within a cadence, whereas weak off-diagonal entries imply the RF is relying on
        per-observation features alone.
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        artifacts = self._load_rf_eval_artifacts(tag)
        shap_data = self._compute_or_load_shap_values(artifacts)

        interaction_values = shap_data["shap_values_interaction"]
        interaction_indices = shap_data["interaction_indices"]
        features_sub = artifacts["val_features"][interaction_indices]
        feature_names = self._rf_feature_names()

        fig = plt.figure(figsize=(12, 16))
        try:
            shap.summary_plot(
                interaction_values,
                features_sub,
                feature_names=feature_names,
                plot_type="compact_dot",
                show=False,
                plot_size=None,
                max_display=len(feature_names),
            )
        except Exception as e:
            logger.warning(f"SHAP interaction summary_plot failed: {e}; falling back to heatmap")
            plt.close(fig)
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            mean_abs_interaction = np.mean(np.abs(interaction_values), axis=0)
            im = ax.imshow(mean_abs_interaction, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(feature_names)))
            ax.set_yticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
            ax.set_yticklabels(feature_names, fontsize=6)
            plt.colorbar(im, ax=ax, label="mean |SHAP interaction|")

        fig = plt.gcf()
        fig.suptitle(
            f"Random Forest SHAP Interactions ({tag}, {machine_name})",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        if dir is not None:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                dir,
                f"rf_shap_interactions_{tag}.png",
            )
        else:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                f"rf_shap_interactions_{tag}.png",
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"RF SHAP interactions plot saved to: {save_path}")

        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"RF SHAP Interactions - ({tag}, {machine_name})",
            )

        del artifacts, shap_data, interaction_values, interaction_indices, features_sub
        gc.collect()

    # NOTE: come back to this later (why is this plot missing? update docstring)
    def plot_rf_shap_loss_monitoring(self, tag: str | None = None, dir: str | None = None):
        """
        Model-monitoring view built on log-loss SHAP decomposition.
        Left: histogram of total per-sample log loss on val, colored by class —
        identifies the long tail of high-loss samples worth inspecting.
        Right: mean log-loss-SHAP per feature split into loss-decreasing (negative)
        vs loss-increasing (positive) — which features the model uses well vs poorly.
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        artifacts = self._load_rf_eval_artifacts(tag)
        shap_data = self._compute_or_load_shap_values(artifacts)

        shap_logloss = shap_data["shap_values_logloss"]
        summary_indices = shap_data["summary_indices"]
        val_binary = artifacts["val_binary_labels"][summary_indices]
        val_probas = artifacts["val_probas"][summary_indices]
        feature_names = self._rf_feature_names()

        eps = 1e-12
        per_sample_logloss = -(
            val_binary * np.log(val_probas + eps) + (1 - val_binary) * np.log(1 - val_probas + eps)
        )
        mean_logloss_shap = np.mean(shap_logloss, axis=0)
        loss_increasing_mask = mean_logloss_shap > 0

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f"Random Forest SHAP Loss Monitoring ({tag}, {machine_name})",
            fontsize=15,
            fontweight="bold",
        )

        ax_hist, ax_bar = axes[0], axes[1]

        bins = np.linspace(0.0, max(float(per_sample_logloss.max()), 1e-3), 40)
        ax_hist.hist(
            per_sample_logloss[val_binary == 1],
            bins=bins,
            color="tab:red",
            alpha=0.55,
            label="True samples",
            edgecolor="darkred",
        )
        ax_hist.hist(
            per_sample_logloss[val_binary == 0],
            bins=bins,
            color="tab:blue",
            alpha=0.55,
            label="False samples",
            edgecolor="darkblue",
        )
        ax_hist.set_xlabel("Per-sample log loss", fontsize=11)
        ax_hist.set_ylabel("Count", fontsize=11)
        ax_hist.set_title("Per-sample Log Loss", fontsize=13, fontweight="bold")
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend(loc="upper right", fontsize=10)

        order = np.argsort(np.abs(mean_logloss_shap))[::-1]
        # matplotlib's barh() rejects ndarrays of color strings; pass a Python list instead
        colors_bar = ["tab:red" if m else "tab:green" for m in loss_increasing_mask[order]]
        y_pos = np.arange(len(order))
        ax_bar.barh(
            y_pos,
            mean_logloss_shap[order],
            color=colors_bar,
            alpha=0.7,
            edgecolor="black",
        )
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([feature_names[i] for i in order], fontsize=6)
        ax_bar.axvline(x=0.0, color="black", linewidth=0.8)
        ax_bar.set_xlabel("Mean SHAP contribution to log loss", fontsize=11)
        ax_bar.set_title(
            "Loss-Decreasing (green) vs Loss-Increasing (red)", fontsize=13, fontweight="bold"
        )
        ax_bar.invert_yaxis()
        ax_bar.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if dir is not None:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                dir,
                f"rf_shap_loss_monitoring_{tag}.png",
            )
        else:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                f"rf_shap_loss_monitoring_{tag}.png",
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"RF SHAP loss monitoring plot saved to: {save_path}")

        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"RF SHAP Loss Monitoring - ({tag}, {machine_name})",
            )

        del artifacts, shap_data, shap_logloss, per_sample_logloss, mean_logloss_shap
        gc.collect()

    def plot_rf_shap_explanation_clustering(self, tag: str | None = None, dir: str | None = None):
        """
        Supervised clustering on SHAP explanation vectors.
        UMAP-reduces the (n_summary × 48) SHAP matrix to 2D, then scatters val samples
        colored by sub-type with marker shape indicating correct/incorrect binary prediction.
        Cadences cluster together when the model reasons about them similarly, even if their
        latents look different.
        Optional k-means overlay (k=4) to compare model-reasoning clusters against sub-types.
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        artifacts = self._load_rf_eval_artifacts(tag)
        shap_data = self._compute_or_load_shap_values(artifacts)

        shap_values = shap_data["shap_values_summary"]
        summary_indices = shap_data["summary_indices"]
        val_subtype = artifacts["val_subtype_labels"][summary_indices]
        val_binary = artifacts["val_binary_labels"][summary_indices]
        val_preds = artifacts["val_preds"][summary_indices]
        classification_threshold = artifacts["classification_threshold"]
        correct = val_preds == val_binary

        # SHAP-space UMAP + KMeans persisted alongside the SHAP cache so re-runs
        # of just this plot (without retraining) reproduce the same cluster
        # layout. Unlike the cadence-level UMAP that plot_latent_space_gif fits
        # on raw latents, this projection is over the (n_summary × 48) SHAP
        # matrix, so it lives in its own joblib.
        clustering_path = os.path.join(self.config.model_path, f"rf_shap_clustering_{tag}.joblib")
        if os.path.exists(clustering_path):
            logger.info(f"Loading cached SHAP clustering from {clustering_path}")
            cached = joblib.load(clustering_path)
            embedding = cached["embedding"]
            cluster_labels = cached["cluster_labels"]
        else:
            logger.info("Fitting SHAP-space UMAP + KMeans on summary SHAP values")
            # NOTE: come back to this later (are these values of n_neighbors & min_dist appropriate always?)
            # NOTE: use a global config seed instead of hard-coding
            umap_model = umap.UMAP(
                n_components=2, random_state=11, n_neighbors=15, min_dist=0.1
            ).fit(shap_values)
            embedding = umap_model.transform(shap_values)
            # NOTE: come back to this later (are these values of n_clusters & n_init appropriate always?)
            # NOTE: use a global config seed instead of hard-coding
            kmeans = KMeans(n_clusters=4, random_state=11, n_init=10)
            cluster_labels = kmeans.fit_predict(shap_values)
            os.makedirs(os.path.dirname(clustering_path), exist_ok=True)
            joblib.dump(
                {"embedding": embedding, "cluster_labels": cluster_labels},
                clustering_path,
            )
            logger.info(f"Saved SHAP clustering to {clustering_path}")
            del umap_model, kmeans

        signal_types = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]
        subtype_colors = {
            "false_no_signal": "blue",
            "false_with_rfi": "green",
            "true_only_eti": "red",
            "true_eti_rfi": "orange",
        }
        display_names = {
            "false_no_signal": "No Signal",
            "false_with_rfi": "RFI Only",
            "true_only_eti": "ETI Only",
            "true_eti_rfi": "ETI + RFI",
        }

        fig, ax = plt.subplots(1, 1, figsize=(11, 9))
        fig.suptitle(
            f"Random Forest SHAP Explanation Clustering (t={classification_threshold:.2f}, {tag}, {machine_name})",
            fontsize=14,
            fontweight="bold",
        )

        for stype in signal_types:
            mask_type = val_subtype == stype
            if not mask_type.any():
                continue
            correct_type_mask = mask_type & correct
            wrong_type_mask = mask_type & (~correct)
            if correct_type_mask.any():
                ax.scatter(
                    embedding[correct_type_mask, 0],
                    embedding[correct_type_mask, 1],
                    c=subtype_colors[stype],
                    marker="o",
                    s=18,
                    alpha=0.6,
                    edgecolor="none",
                    label=f"{display_names[stype]} (✓)",
                )
            if wrong_type_mask.any():
                ax.scatter(
                    embedding[wrong_type_mask, 0],
                    embedding[wrong_type_mask, 1],
                    c=subtype_colors[stype],
                    marker="x",
                    s=28,
                    alpha=0.8,
                    label=f"{display_names[stype]} (✗)",
                )

        # Overlay k-means cluster centroids (in SHAP space, then projected via
        # nearest-sample lookup to avoid a separate UMAP inverse)
        for cluster_id in range(4):
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.any():
                cx = np.mean(embedding[cluster_mask, 0])
                cy = np.mean(embedding[cluster_mask, 1])
                ax.scatter(
                    cx,
                    cy,
                    s=250,
                    marker="P",
                    facecolor="none",
                    edgecolor="black",
                    linewidth=2.0,
                    zorder=5,
                )
                ax.text(
                    cx,
                    cy,
                    # NOTE: come back to this later (is this enough spacing?)
                    f"    k{cluster_id}",
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                )

        ax.set_xlabel("UMAP (SHAP) 1", fontsize=11)
        ax.set_ylabel("UMAP (SHAP) 2", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.85)

        plt.tight_layout()

        if dir is not None:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                dir,
                f"rf_shap_explanation_clustering_{tag}.png",
            )
        else:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                f"rf_shap_explanation_clustering_{tag}.png",
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"RF SHAP explanation clustering plot saved to: {save_path}")

        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"RF SHAP Explanation Clustering - ({tag}, {machine_name})",
            )

        del artifacts, shap_data, shap_values, embedding, cluster_labels
        gc.collect()

    def plot_rf_calibration_curve(self, tag: str | None = None, dir: str | None = None):
        """
        Two stacked subplots:
        - Top: reliability diagram (quantile-binned). Annotated with Brier score and ECE.
        - Bottom: histogram of val predicted probabilities (binning denominator).
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        artifacts = self._load_rf_eval_artifacts(tag)
        val_binary = artifacts["val_binary_labels"]
        val_probas = artifacts["val_probas"]

        # NOTE: come back to this later (what does brier score & ECE mean exactly?)
        frac_pos, mean_pred = calibration_curve(
            val_binary, val_probas, n_bins=10, strategy="quantile"
        )
        brier = brier_score_loss(val_binary, val_probas)

        # Expected Calibration Error (ECE) = sum over bins of weight * |conf - acc|
        bin_edges = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        n_total = len(val_probas)
        for i in range(10):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == 9:
                in_bin = (val_probas >= lo) & (val_probas <= hi)
            else:
                in_bin = (val_probas >= lo) & (val_probas < hi)
            if in_bin.any():
                bin_conf = float(np.mean(val_probas[in_bin]))
                bin_acc = float(np.mean(val_binary[in_bin]))
                ece += (float(in_bin.sum()) / n_total) * abs(bin_conf - bin_acc)

        fig, axes = plt.subplots(2, 1, figsize=(9, 11), gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle(
            f"Random Forest Calibration Curve ({tag}, {machine_name})",
            fontsize=15,
            fontweight="bold",
        )

        ax_rel, ax_hist = axes[0], axes[1]

        ax_rel.plot(
            [0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Perfect calibration"
        )
        ax_rel.plot(
            mean_pred,
            frac_pos,
            color="tab:red",
            marker="o",
            linewidth=2,
            markersize=7,
            label="RF model",
        )
        ax_rel.set_xlabel("Mean predicted probability (bin)", fontsize=11)
        ax_rel.set_ylabel("Fraction of positives (bin)", fontsize=11)
        ax_rel.set_title("Reliability Diagram", fontsize=13, fontweight="bold")
        ax_rel.grid(True, alpha=0.3)
        ax_rel.legend(loc="upper left", fontsize=10)
        ax_rel.set_xlim(-0.01, 1.01)
        ax_rel.set_ylim(-0.01, 1.01)
        ax_rel.text(
            0.98,
            0.02,
            f"Brier = {brier:.4f}\nECE = {ece:.4f}",
            transform=ax_rel.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

        ax_hist.hist(
            val_probas,
            bins=np.linspace(0.0, 1.0, 40),
            color="tab:blue",
            edgecolor="darkblue",
            alpha=0.7,
        )
        ax_hist.set_xlabel("Predicted P(true)", fontsize=11)
        ax_hist.set_ylabel("Count", fontsize=11)
        ax_hist.set_title("Predicted Probability Distribution", fontsize=13, fontweight="bold")
        ax_hist.grid(True, alpha=0.3)
        ax_hist.set_xlim(-0.01, 1.01)

        plt.tight_layout()

        if dir is not None:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                dir,
                f"rf_calibration_curve_{tag}.png",
            )
        else:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                f"rf_calibration_curve_{tag}.png",
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"RF calibration curve plot saved to: {save_path}")

        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"RF Calibration Curve - ({tag}, {machine_name})",
            )

        del artifacts, val_binary, val_probas, frac_pos, mean_pred
        gc.collect()

    def plot_rf_ensemble_accuracy_curve(self, tag: str | None = None, dir: str | None = None):
        """
        Cumulative ensemble accuracy as a function of tree count, computed on the
        held-out val set (and a subsample of the train set as a sanity baseline).
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        artifacts = self._load_rf_eval_artifacts(tag)
        train_features = artifacts["train_features"]
        train_binary = artifacts["train_binary_labels"]
        val_features = artifacts["val_features"]
        val_binary = artifacts["val_binary_labels"]

        # NOTE: use a global config seed instead of hard-coding
        rng = np.random.default_rng(self.config.rf.seed)
        # NOTE: come back to this later (parametrize 10% hold-out in config.py)
        n_train_sample = max(1, int(0.1 * train_features.shape[0]))
        train_sample_idx = rng.choice(train_features.shape[0], size=n_train_sample, replace=False)
        train_features_sub = train_features[train_sample_idx]
        train_binary_sub = train_binary[train_sample_idx]

        estimators = self.rf_model.model.estimators_
        n_trees = len(estimators)

        # NOTE: come back to this later (use flexible threshold, not hard-coded 0.5? in fact, wouldn't it be better to store the preds from rf.py itself rather than storing the probas and trying to figure it out after the fact?)
        # Accumulate per-tree predict_proba for positive class.
        # Running mean after each tree is the ensemble probability; threshold at 0.5.
        cum_val = np.zeros(val_features.shape[0], dtype=np.float64)
        cum_train = np.zeros(train_features_sub.shape[0], dtype=np.float64)
        val_acc = np.empty(n_trees, dtype=np.float64)
        train_acc = np.empty(n_trees, dtype=np.float64)

        logger.info(
            f"Computing cumulative RF ensemble accuracy across {n_trees} trees "
            f"on {val_features.shape[0]} val samples and {n_train_sample} train-sample baseline"
        )

        for t, tree in enumerate(estimators):
            tree_val = tree.predict_proba(val_features)[:, 1]
            tree_train = tree.predict_proba(train_features_sub)[:, 1]
            cum_val += tree_val
            cum_train += tree_train
            # NOTE: come back to this later (use flexible threshold, not hard-coded 0.5? in fact, wouldn't it be better to store the preds from rf.py itself rather than storing the probas and trying to figure it out after the fact?)
            val_pred = (cum_val / (t + 1) >= 0.5).astype(np.int64)
            train_pred = (cum_train / (t + 1) >= 0.5).astype(np.int64)
            val_acc[t] = float(np.mean(val_pred == val_binary))
            train_acc[t] = float(np.mean(train_pred == train_binary_sub))

        # Find elbow: first tree after which val_acc stays within 1% of its final value
        final_val_acc = val_acc[-1]
        near_final = np.where(val_acc >= final_val_acc - 0.01)[0]
        elbow_idx = int(near_final[0]) if near_final.size else n_trees - 1

        # Persist the per-tree series to training_stats (model_name='rf', epoch_number =
        # tree count) for the dashboard's live ensemble curve (issue #171). Written here —
        # not in train_random_forest() — to reuse the O(n_trees) per-tree predict_proba
        # loop above, so these rows only land when the rf_plots stage runs and this plot
        # succeeds; an rf_plots retry re-writes them (the dashboard reads last-write-wins).
        # NOTE: this series thresholds the running ensemble mean at the hard-coded 0.5
        # above, unlike the scalar val_accuracy stat (deployment classification_threshold).
        series_timestamp = time.time()
        for t in range(n_trees):
            self.db.write_training_stat(
                model_name="rf",
                stat_name="ensemble_val_accuracy",
                value=float(val_acc[t]),
                epoch_number=t + 1,
                tag=tag,
                timestamp=series_timestamp,
            )

        fig, ax = plt.subplots(1, 1, figsize=(11, 6))
        fig.suptitle(
            f"Random Forest Ensemble Accuracy vs Tree Count ({tag}, {machine_name})",
            fontsize=14,
            fontweight="bold",
        )
        x = np.arange(1, n_trees + 1)
        ax.plot(x, val_acc, color="tab:red", linewidth=2, label="Val accuracy")
        ax.plot(
            x,
            train_acc,
            color="tab:blue",
            linewidth=1.6,
            linestyle="--",
            label="Train-subsample accuracy (baseline)",
        )
        ax.axvline(
            x=elbow_idx + 1,
            color="tab:green",
            linestyle=":",
            linewidth=1.5,
            label=f"Elbow: {elbow_idx + 1} trees (val acc ≈ {val_acc[elbow_idx]:.4f})",
        )
        ax.set_xlabel("Number of trees", fontsize=11)
        ax.set_ylabel("Cumulative accuracy", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)

        plt.tight_layout()

        if dir is not None:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                dir,
                f"rf_oob_accuracy_curve_{tag}.png",
            )
        else:
            save_path = os.path.join(
                self.config.output_path,
                "plots",
                "training",
                self.config.checkpoint.save_tag,
                f"rf_oob_accuracy_curve_{tag}.png",
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"RF ensemble accuracy curve saved to: {save_path}")

        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path,
                title=f"RF Ensemble Accuracy vs Trees - ({tag}, {machine_name})",
            )

        del artifacts, val_features, val_binary, train_features, train_binary
        del train_features_sub, train_binary_sub, cum_val, cum_train, val_acc, train_acc
        gc.collect()

    def plot_rf_latent_decision_boundary(self, tag: str | None = None, dir: str | None = None):
        """
        For each persisted cadence-level UMAP (one per (n_neighbors, min_dist) combo from
        plot_latent_space_gif), render a decision-boundary figure:
          - background: filled contour of RF P(true) on a 2D grid (inverse_transformed to 48D)
          - foreground: val samples colored by sub-type with correct/✗ markers
          - overlay: explicit contour at 0.5 for the decision boundary
        One PNG per (nn, md) combo.
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        artifacts = self._load_rf_eval_artifacts(tag)
        val_features = artifacts["val_features"]
        val_binary = artifacts["val_binary_labels"]
        val_subtype = artifacts["val_subtype_labels"]
        val_preds = artifacts["val_preds"]
        classification_threshold = artifacts["classification_threshold"]
        correct = val_preds == val_binary

        max_points = self.config.training.rf_decision_boundary_max_points
        grid_size = self.config.training.rf_decision_boundary_grid_size

        if val_features.shape[0] > max_points:
            # NOTE: use a global config seed instead of hard-coding
            rng = np.random.default_rng(self.config.rf.seed)
            point_idx = rng.choice(val_features.shape[0], size=max_points, replace=False)
            pts_features = val_features[point_idx]
            pts_subtype = val_subtype[point_idx]
            pts_correct = correct[point_idx]
        else:
            pts_features = val_features
            pts_subtype = val_subtype
            pts_correct = correct

        signal_types = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]
        subtype_colors = {
            "false_no_signal": "blue",
            "false_with_rfi": "green",
            "true_only_eti": "red",
            "true_eti_rfi": "orange",
        }
        display_names = {
            "false_no_signal": "No Signal",
            "false_with_rfi": "RFI Only",
            "true_only_eti": "ETI Only",
            "true_eti_rfi": "ETI + RFI",
        }

        n_neighbors_values = self.config.training.latent_viz_umap_n_neighbors
        min_dist_values = self.config.training.latent_viz_umap_min_dist

        n_generated = 0
        for nn in n_neighbors_values:
            for md in min_dist_values:
                umap_path = os.path.join(
                    self.config.model_path, f"umap_cadence_nn{nn}_md{md}_{tag}.joblib"
                )
                if not os.path.exists(umap_path):
                    logger.warning(
                        f"Cadence UMAP model not found: {umap_path} — skipping this combo"
                    )
                    continue

                logger.info(f"Rendering decision boundary for (nn={nn}, md={md})")
                try:
                    umap_model = joblib.load(umap_path)
                    embedding = umap_model.transform(pts_features)
                except Exception as e:
                    logger.warning(f"Failed to load/transform UMAP at {umap_path}: {e}")
                    continue

                x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
                y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
                x_pad = (x_max - x_min) * 0.05
                y_pad = (y_max - y_min) * 0.05
                x_min, x_max = x_min - x_pad, x_max + x_pad
                y_min, y_max = y_min - y_pad, y_max + y_pad

                xs = np.linspace(x_min, x_max, grid_size)
                ys = np.linspace(y_min, y_max, grid_size)
                xx, yy = np.meshgrid(xs, ys)
                grid_2d = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

                try:
                    grid_48d = umap_model.inverse_transform(grid_2d)
                except Exception as e:
                    logger.warning(
                        f"UMAP inverse_transform failed for (nn={nn}, md={md}): {e} — "
                        f"skipping decision boundary for this combo"
                    )
                    continue

                grid_probas = self.rf_model.model.predict_proba(grid_48d)[:, 1]
                proba_grid = grid_probas.reshape(xx.shape)

                fig, ax = plt.subplots(1, 1, figsize=(11, 9))
                fig.suptitle(
                    f"Random Forest Decision Boundary (nn={nn}, md={md}, "
                    f"t={classification_threshold:.2f}) — ({tag}, {machine_name})",
                    fontsize=13,
                    fontweight="bold",
                )

                contourf = ax.contourf(
                    xx,
                    yy,
                    proba_grid,
                    levels=np.linspace(0.0, 1.0, 21),
                    cmap="RdBu_r",
                    alpha=0.4,
                )
                fig.colorbar(contourf, ax=ax, label="P(true)")
                # Decision-boundary contour drawn at the deployment threshold (not the
                # symmetric 0.5 midpoint) so the black line on the figure matches the
                # rule the model will actually be deployed with — and is consistent
                # with the correct/wrong markers below, which are also derived at this
                # threshold.
                ax.contour(
                    xx,
                    yy,
                    proba_grid,
                    levels=[classification_threshold],
                    colors="black",
                    linewidths=1.6,
                )

                for stype in signal_types:
                    mask_type = pts_subtype == stype
                    if not mask_type.any():
                        continue
                    correct_mask = mask_type & pts_correct
                    wrong_mask = mask_type & (~pts_correct)
                    if correct_mask.any():
                        ax.scatter(
                            embedding[correct_mask, 0],
                            embedding[correct_mask, 1],
                            c=subtype_colors[stype],
                            marker="o",
                            s=12,
                            alpha=0.6,
                            edgecolor="none",
                            label=f"{display_names[stype]} (✓)",
                        )
                    if wrong_mask.any():
                        ax.scatter(
                            embedding[wrong_mask, 0],
                            embedding[wrong_mask, 1],
                            c=subtype_colors[stype],
                            marker="x",
                            s=22,
                            alpha=0.9,
                            label=f"{display_names[stype]} (✗)",
                        )

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel("UMAP 1", fontsize=11)
                ax.set_ylabel("UMAP 2", fontsize=11)
                ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.85)

                plt.tight_layout()

                filename = f"rf_latent_decision_boundary_nn{nn}_md{md}_{tag}.png"
                if dir is not None:
                    save_path = os.path.join(
                        self.config.output_path,
                        "plots",
                        "training",
                        self.config.checkpoint.save_tag,
                        dir,
                        filename,
                    )
                else:
                    save_path = os.path.join(
                        self.config.output_path,
                        "plots",
                        "training",
                        self.config.checkpoint.save_tag,
                        filename,
                    )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"RF decision boundary plot saved to: {save_path}")

                logger_instance = get_logger()
                if logger_instance:
                    logger_instance.upload_image_to_slack(
                        save_path,
                        title=f"RF Decision Boundary (nn={nn}, md={md}) - ({tag}, {machine_name})",
                    )

                n_generated += 1

        if n_generated == 0:
            logger.warning(
                "plot_rf_latent_decision_boundary: no figures generated — check that "
                "plot_latent_space_gif has run and persisted cadence-level UMAP models"
            )

        del artifacts, val_features, val_subtype, val_binary, val_preds, correct
        del pts_features, pts_subtype, pts_correct
        gc.collect()

    def _prepare_latent_viz_batch(
        self, concat_data, labels, candidate_indices=None, lognorm_params=None
    ):
        """
        Subsample cadences from concat_data for latent-space visualization, attempting an equal
        distribution across the 4 signal_type values in `labels` (n_per_type per type).

        Called once on the first round's stratified validation partition and persisted across
        subsequent rounds — using held-out data ensures the visualization captures generalization,
        and reusing the same data across rounds removes distribution-shift artifacts from the
        curriculum schedule. concat_data is shape (n_total, 6, 16, width_bin); labels is shape
        (n_total,); candidate_indices, when given, restricts eligible samples (e.g. to validation
        partition indices) without copying the full partition. lognorm_params, when given, is the
        (n_total, 6, 2) per-observation log-norm parameter array recorded at generation time —
        the selected cadences' rows are kept for plot_latent_traversal's display inversion.
        """
        n_per_type = self.config.training.latent_viz_num_cadences_per_type
        signal_types = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]
        rng = derive_rng(self.config.training.seed, STREAM_VIZ, 0)

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
                sampled = rng.choice(type_global_indices, size=n_per_type, replace=False)
            selected_indices.append(sampled)
            selected_labels.extend([stype] * len(sampled))

        if not selected_indices:
            logger.warning("No cadences found for any signal type — skipping viz batch")
            self._latent_viz_batch = None
            self._latent_viz_labels = None
            self._latent_viz_lognorm_params = None
            return

        # Fancy indexing already creates a new independent array (no .copy() needed)
        all_indices = np.concatenate(selected_indices)
        self._latent_viz_batch = concat_data[all_indices]
        self._latent_viz_labels = np.array(selected_labels, dtype="U20")
        self._latent_viz_lognorm_params = (
            lognorm_params[all_indices] if lognorm_params is not None else None
        )

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
            rng=derive_rng(self.config.training.seed, STREAM_VIZ, 1),
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
        """Load model weights.

        An explicit `tag` must exist in the target directory (FileNotFoundError otherwise);
        only the tag=None default falls back to 'final' and then the latest tag present —
        see _resolve_load_tag() and issue #142.
        """
        if dir is not None:
            base_dir = os.path.join(self.config.model_path, dir)
        else:
            base_dir = self.config.model_path

        tag = _resolve_load_tag(base_dir, tag)

        # Construct filepaths
        encoder_path = os.path.join(base_dir, f"vae_encoder_{tag}.keras")
        decoder_path = os.path.join(base_dir, f"vae_decoder_{tag}.keras")
        rf_path = os.path.join(base_dir, f"random_forest_{tag}.joblib")

        # Load the models
        try:
            logger.info(f"Loading models from {base_dir} with tag '{tag}'")

            # Load encoder & decoder
            checkpoint_encoder = tf.keras.models.load_model(encoder_path)
            checkpoint_decoder = tf.keras.models.load_model(decoder_path)

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
                self._rf_loaded_from_tag = tag
                logger.info("Random Forest loaded successfully")
            else:
                logger.info(
                    f"Random Forest not found at {rf_path} - this is normal if RF hasn't been trained yet"
                )

            logger.info(f"Successfully loaded models from {base_dir} with tag '{tag}'")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise  # Re-raise to propagate error

    def _run_plot_group(self, plot_calls: list[tuple[Callable, str]]) -> None:
        """
        Run a group of plot methods, attempting every one even when some fail (a single
        broken plot mustn't block the others), then raise a summary error listing the
        failures so the stage machine can record the group as failed.
        """
        failures = []
        for func, name in plot_calls:
            try:
                func()
            except Exception as e:
                logger.error(f"Failed to execute {name}: {e}")
                failures.append(name)
        if failures:
            raise RuntimeError(f"{len(failures)} plot(s) failed: {', '.join(failures)}")

    # NOTE: combine plot_beta_vae_loss_curves(), plot_beta_vae_training_stability(), and plot_latent_space_gif() into plot_training_progress()?
    def plot_vae_diagnostics(self) -> None:
        """The vae_plots stage: final loss curves, training stability, injection stats, the
        latent-space GIF, and the latent-dimension traversal. Attempts every plot even if one
        fails, then raises listing the failures — the stage machine records them without
        costing a data regeneration."""
        self._run_plot_group(
            [
                (self.plot_beta_vae_loss_curves, "plot_beta_vae_loss_curves"),
                (self.plot_beta_vae_training_stability, "plot_beta_vae_training_stability"),
                (self.plot_injection_stats, "plot_injection_stats"),
                (self.plot_latent_space_gif, "plot_latent_space_gif"),
                (self.plot_latent_traversal, "plot_latent_traversal"),
            ]
        )

    def plot_rf_diagnostics(self) -> None:
        """
        The rf_plots stage: the ten Random Forest visualizations. All plots consume the
        eval-artifact joblib persisted by train_random_forest() (the five SHAP plots
        additionally share a SHAP-values joblib); every plot is attempted even when one
        fails (e.g. an optional dep like SHAP missing), then a summary error is raised for
        the stage machine to record. plot_rf_latent_decision_boundary relies on the
        cadence-level UMAP joblibs written by plot_latent_space_gif (vae_plots stage), so
        stage ordering matters.
        """
        self._run_plot_group(
            [
                (self.plot_rf_confusion_matrices, "plot_rf_confusion_matrices"),
                (self.plot_rf_classification_curves, "plot_rf_classification_curves"),
                (self.plot_rf_shap_summary, "plot_rf_shap_summary"),
                (self.plot_rf_shap_dependence, "plot_rf_shap_dependence"),
                (self.plot_rf_shap_interactions, "plot_rf_shap_interactions"),
                (self.plot_rf_shap_loss_monitoring, "plot_rf_shap_loss_monitoring"),
                (
                    self.plot_rf_shap_explanation_clustering,
                    "plot_rf_shap_explanation_clustering",
                ),
                (self.plot_rf_calibration_curve, "plot_rf_calibration_curve"),
                (self.plot_rf_ensemble_accuracy_curve, "plot_rf_ensemble_accuracy_curve"),
                (
                    self.plot_rf_latent_decision_boundary,
                    "plot_rf_latent_decision_boundary",
                ),
            ]
        )

    def final_save(self) -> None:
        """The final_save stage: persist the final models plus the resolved config JSON.
        The config dump used to live in main.py's train_command after the retry loop; doing
        it here puts it under the stage machine's retry coverage."""
        self.save_models()

        config_path = os.path.join(
            self.config.output_path, f"config_{self.config.checkpoint.save_tag}.json"
        )
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Training configuration saved to {config_path}")

    def upload_to_hf(self) -> None:
        """The hf_upload stage: publish the final artifacts (staged under stable names) plus
        a generated model card to the configured HuggingFace Hub repo, then tag the commit
        with this run's save_tag. Requires HF_TOKEN in the environment (via .env)."""
        upload_run_to_hf(
            repo_id=self.config.hf.repo_id,
            tag=self.config.checkpoint.save_tag,
            model_path=self.config.model_path,
            output_path=self.config.output_path,
            force=self.config.checkpoint.force_tag,
        )


def _execute_training_stages(pipeline) -> None:
    """
    The training stage machine: an explicit ordered stage list with skip-if-done and
    persist-on-success semantics driven by the pipeline's run manifest.

    Critical stages (vae_rounds, rf_train, final_save) raise on failure — the retry loop in
    main.py rebuilds the pipeline and the manifest resumes it here. Non-critical stages
    (vae_plots, rf_plots, hf_upload) are recorded in stages_failed and execution continues:
    a broken plot or Hub upload mustn't cost a retry cycle including data regeneration, but
    main.py exits nonzero at the very end if any recorded failure never recovers, so lost
    artifacts stay loud.

    `pipeline` is duck-typed (a TrainingPipeline in production) so unit tests can drive the
    stage logic with a lightweight stub.
    """
    state = pipeline.run_state

    # Stage 1/5: vae_rounds — the beta-VAE round loop (per-round sub-resume happens inside
    # train_beta_vae via the manifest's completed_rounds)
    if state.is_stage_done(STAGE_VAE_ROUNDS):
        logger.info(f"Stage '{STAGE_VAE_ROUNDS}' already complete — skipping")
    else:
        pipeline.train_beta_vae()
        pipeline._mark_stage_done(STAGE_VAE_ROUNDS)

    # Stage 2/5: vae_plots (non-critical)
    if state.is_stage_done(STAGE_VAE_PLOTS):
        logger.info(f"Stage '{STAGE_VAE_PLOTS}' already complete — skipping")
    else:
        try:
            with stage_timer("train.vae_plots"):
                pipeline.plot_vae_diagnostics()
            pipeline._mark_stage_done(STAGE_VAE_PLOTS)
        except Exception as e:
            logger.error(f"Stage '{STAGE_VAE_PLOTS}' failed: {e}")
            pipeline._record_stage_failure(STAGE_VAE_PLOTS)
        finally:
            # The withheld viz batch's last consumer (plot_latent_traversal) has run —
            # free it (a few hundred MB at full-scale defaults) before rf_train
            pipeline._clear_latent_viz_data()

    # Stage 3/5: rf_train. On skip, the persisted RF from the attempt that completed the
    # stage is loaded back (rf_plots and final_save need a live model); if that reload
    # fails (e.g. the joblib was deleted), fall through to a full retrain
    if state.is_stage_done(STAGE_RF_TRAIN) and pipeline.try_load_rf_for_resume():
        logger.info(f"Stage '{STAGE_RF_TRAIN}' already complete — loaded persisted RF model")
    else:
        try:
            # Umbrella span for the RF stage — the data_generation / encode / fit
            # sub-stages inside train_random_forest nest under it
            with stage_timer("train.rf"):
                pipeline.train_random_forest()
        except Exception as e:
            logger.error(f"Error in train_random_forest(): {e}")
            # Attempt to save models on RF training failure
            pipeline.rf_model = None  # Avoid saving incomplete RF model state
            _safe_call(pipeline.save_models, "save_models")
            raise  # Re-raise to propagate error
        pipeline._mark_stage_done(STAGE_RF_TRAIN)

    # Stage 4/5: rf_plots (non-critical)
    if state.is_stage_done(STAGE_RF_PLOTS):
        logger.info(f"Stage '{STAGE_RF_PLOTS}' already complete — skipping")
    else:
        try:
            with stage_timer("train.rf_plots"):
                pipeline.plot_rf_diagnostics()
            pipeline._mark_stage_done(STAGE_RF_PLOTS)
        except Exception as e:
            logger.error(f"Stage '{STAGE_RF_PLOTS}' failed: {e}")
            pipeline._record_stage_failure(STAGE_RF_PLOTS)
        finally:
            # RF plots are done (or abandoned) — drop the shared eval-artifact / SHAP
            # caches so the (large) features arrays they hold don't hang around through
            # final_save and teardown. Clearing only in this branch is deliberate: the
            # caches are populated exclusively by the plot calls above, so the
            # skip-if-done path never has anything to clear
            pipeline._clear_rf_caches()

    # Stage 5/5: final_save — final models + config JSON
    if state.is_stage_done(STAGE_FINAL_SAVE):
        logger.info(f"Stage '{STAGE_FINAL_SAVE}' already complete — skipping")
    else:
        with stage_timer("train.final_save"):
            pipeline.final_save()
        pipeline._mark_stage_done(STAGE_FINAL_SAVE)

    # Stage 6: hf_upload (opt-in via config.hf.upload_after_training; non-critical) —
    # publish the final artifacts + model card to the HuggingFace Hub. Failure is recorded
    # in the manifest but never fails the run (the weights are already safe locally);
    # re-running the identical command retries just this stage.
    if not pipeline.config.hf.upload_after_training:
        if STAGE_HF_UPLOAD in state.stages_failed:
            # The user opted out after a failed upload attempt — the upload is no longer
            # pending, so drop the stale failure (it must not force a nonzero exit forever)
            logger.info(
                f"HF upload disabled — clearing the stale '{STAGE_HF_UPLOAD}' failure "
                f"recorded by a previous attempt"
            )
            pipeline._clear_stage_failure(STAGE_HF_UPLOAD)
    elif state.is_stage_done(STAGE_HF_UPLOAD):
        logger.info(f"Stage '{STAGE_HF_UPLOAD}' already complete — skipping")
    else:
        try:
            pipeline.upload_to_hf()
            pipeline._mark_stage_done(STAGE_HF_UPLOAD)
        except Exception as e:
            logger.error(f"Stage '{STAGE_HF_UPLOAD}' failed: {e}")
            pipeline._record_stage_failure(STAGE_HF_UPLOAD)


def run_training_pipeline(
    background_data: np.ndarray, strategy: tf.distribute.Strategy = None
) -> TrainingPipeline:
    """
    End-to-end training entry point: construct a TrainingPipeline from preprocessed background
    data (shape (n, 6, 16, 512)) and an optional tf.distribute strategy, then run the ordered
    training stages (vae_rounds -> vae_plots -> rf_train -> rf_plots -> final_save ->
    hf_upload), skipping any the persisted run manifest already records as done.
    """
    try:
        # Create pipeline (no cleanup needed on failure)
        pipeline = TrainingPipeline(background_data, strategy)
    except Exception as e:
        logger.error(f"Error creating TrainingPipeline: {e}")
        raise  # Re-raise to propagate error

    try:
        _execute_training_stages(pipeline)

        if pipeline.run_state.stages_failed:
            logger.warning(
                f"Training pipeline finished, but non-critical stage(s) failed: "
                f"{', '.join(pipeline.run_state.stages_failed)}"
            )
        logger.info("Training complete!")

        return pipeline

    finally:
        # Free shared resources on exit. The viz-batch clear matters on the failure path:
        # a vae_rounds crash skips the vae_plots-stage clear, and the retry loop builds a
        # fresh pipeline — don't let the dying one pin a few hundred MB of viz data
        pipeline._clear_latent_viz_data()
        pipeline.data_generator.close()


def _safe_call(func: Callable, name: str, args: tuple | None = None) -> None:
    """
    Invoke a callable, log-and-swallow any exception.

    Used during error cleanup where a best-effort call is still worth attempting (e.g.
    saving the VAE weights after an RF-training failure). Plot dispatch uses
    TrainingPipeline._run_plot_group instead, which also attempts every call but reports
    the failures so the stage machine can record them.
    """
    try:
        func(*args) if args else func()
    except Exception as e:
        logger.warning(f"Failed to execute {name}: {e}")
