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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils import shuffle as sklearn_shuffle
from tensorflow.keras.initializers import GlorotNormal, HeNormal
from tensorflow.keras.layers import Conv2D, Dense

from aetherscan.benchmark import round_stage_name, stage_timer
from aetherscan.config import get_config
from aetherscan.data_generation import DataGenerator
from aetherscan.db import get_db, get_system_metadata
from aetherscan.hf_hub import upload_run_to_hf
from aetherscan.latent_gif import run_umap_gif_sweep
from aetherscan.latent_variants import (
    VARIANT_ORDER,
    active_latent_dims,
    apply_probability_calibrator,
    build_variant_features,
    build_z_aug_training_set,
    expected_calibration_error,
    fit_probability_calibrator,
    latent_dim_variances,
    recall_at_fpr,
    sample_z_flat,
    select_winner,
)
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
from aetherscan.seeding import (
    STREAM_DATASET,
    STREAM_KMEANS,
    STREAM_PLOT,
    STREAM_RF,
    STREAM_RF_PLOTS,
    STREAM_SHAP_SAMPLES,
    STREAM_UMAP,
    STREAM_VIZ,
    derive_rng,
    derive_seed,
    seed_tensorflow,
)
from aetherscan.shap_parallel import parallel_shap

logger = logging.getLogger(__name__)

# TEST: is clipping still needed? currently every step seems to be getting clipped. what
# happens if we just don't?
# Global L2-norm gradient-clip threshold (see _build_accumulated_train_step for rationale):
# aggressive enough to catch exploding gradients, permissive enough not to dampen learning —
# aim for no more than ~1-5% of steps triggering the clip
_GRADIENT_CLIP_NORM = 1.0


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


def _model_pair_exists(base_dir: str, tag: str) -> bool:
    """True when the encoder/decoder pair for `tag` both exist in base_dir."""
    return os.path.exists(os.path.join(base_dir, f"vae_encoder_{tag}.keras")) and os.path.exists(
        os.path.join(base_dir, f"vae_decoder_{tag}.keras")
    )


def _resolve_load_tag(base_dir: str, tag: str | None) -> str:
    """
    Resolve the tag load_models() should load from base_dir.

    An explicitly requested tag must exist on disk — we never fall back to "the latest" tag,
    which could silently resume from a stale, unrelated model while reporting success (issue
    #142). The tag=None default loads the conventional "final" model, or fails loudly.
    """
    if tag is not None:
        if _model_pair_exists(base_dir, tag):
            return tag
        msg = (
            f"No models tagged '{tag}' in {base_dir} — refusing to fall back to the latest tag "
            f"for an explicitly requested tag."
        )
        # The per-round-checkpoint hint only helps for a round_XX tag; for any other explicit
        # tag (e.g. a typo'd full run tag) it's a red herring, so only append it for round tags.
        if re.fullmatch(r"round_\d+", tag):
            msg += (
                " If you meant to resume from a per-round checkpoint, pass --load-dir checkpoints"
            )
        raise FileNotFoundError(msg)

    # No explicit tag: only the conventional "final" model may be loaded implicitly. We do NOT
    # scan for the "latest" tag in base_dir — that could silently load a stale, unrelated run's
    # model (issue #142). If there's no "final" pair, fail loudly.
    logger.info("No tag specified. Defaulting to 'final'")
    if _model_pair_exists(base_dir, "final"):
        return "final"
    raise FileNotFoundError(
        f"No models tagged 'final' in {base_dir} and no explicit --load-tag given — refusing to "
        f"guess a tag. Pass --load-tag (a full run tag, or --load-dir checkpoints --load-tag round_XX)."
    )


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


def build_epoch_history(
    all_stats: list[dict], epochs_per_round: int
) -> tuple[range, dict[str, list[float]]]:
    """
    Arrange training-stat rows onto the REAL global-epoch axis for plotting (#277).

    Each row's x position is (round_number - 1) * epochs_per_round + epoch_number, so a value
    lands where the epoch actually happened; positions with no committed row hold NaN, which
    matplotlib renders as a visible gap in the line. The previous positional 1..N axis
    discarded epoch_number after sorting, so any missing rows silently re-registered later
    epochs onto earlier x positions — a partially-drained write queue could mis-plot rather
    than merely stop short. Duplicate (round, epoch) rows keep the last value seen; rows
    without round/epoch numbers are skipped.

    Returns (epochs, history): epochs = range(1, max_position + 1), and history maps
    stat_name to a value list aligned with epochs.
    """
    per_stat: dict[str, dict[int, float]] = {}
    max_position = 0
    for stat in all_stats:
        round_number = stat.get("round_number")
        epoch_number = stat.get("epoch_number")
        if round_number is None or epoch_number is None:
            continue
        position = (round_number - 1) * epochs_per_round + epoch_number
        per_stat.setdefault(stat["stat_name"], {})[position] = stat["value"]
        max_position = max(max_position, position)

    epochs = range(1, max_position + 1)
    history = {
        stat_name: [values.get(position, float("nan")) for position in epochs]
        for stat_name, values in per_stat.items()
    }
    return epochs, history


def check_posterior_collapse(
    kl_per_dim: np.ndarray,
    low_kl_streaks: np.ndarray,
    kl_epsilon: float,
    min_active_fraction: float,
    patience: int,
    tag: str,
) -> bool:
    """
    Advisory posterior-collapse guard (#282), same idiom as check_val_auc_floor: never fails
    the run, WARNs loudly (reaches Slack) when latent capacity is going dark.

    A dim counts ACTIVE when its batch-mean KL exceeds kl_epsilon; the round alarms when the
    active fraction drops below min_active_fraction OR any dim's KL has sat under epsilon
    for `patience` consecutive epochs. With beta > 1 some pruning is expected and even
    desirable — 6-8 active dims of 8 is healthy, 1-2 is pathological — so the caller-facing
    remedy ladder (KL warm-up -> free bits -> lower beta -> shrink latent_dim) lives in
    docs/TRAINING_PIPELINE.md, not in code. Returns True when collapse was flagged.
    """
    kl_per_dim = np.asarray(kl_per_dim, dtype=np.float64).ravel()
    low_kl_streaks = np.asarray(low_kl_streaks).ravel()
    latent_dim = len(kl_per_dim)
    active = kl_per_dim > kl_epsilon
    n_active = int(active.sum())
    stuck_dims = [int(d) for d in np.nonzero(low_kl_streaks >= patience)[0]]

    collapsed = (n_active < min_active_fraction * latent_dim) or bool(stuck_dims)
    if collapsed:
        logger.warning(
            f"POSTERIOR COLLAPSE WARNING ({tag}): {n_active}/{latent_dim} latent dims active "
            f"(KL > {kl_epsilon}); dims stuck below epsilon for >= {patience} consecutive "
            f"epochs: {stuck_dims or 'none'}. Per-dim KL: "
            f"{np.array2string(kl_per_dim, precision=4)}. See the posterior-collapse "
            "playbook in docs/TRAINING_PIPELINE.md (KL warm-up, free bits, lower beta, or "
            "shrink latent_dim)."
        )
    else:
        logger.info(
            f"Posterior-collapse check passed ({tag}): {n_active}/{latent_dim} latent dims "
            f"active (KL > {kl_epsilon})"
        )
    return collapsed


def check_screening_threshold(
    test_labels: np.ndarray,
    pass1_probas: np.ndarray,
    mc_mean_probas: np.ndarray,
    screening_threshold: float,
    science_threshold: float,
    recall_tolerance: float,
    tag: str,
) -> dict[str, float]:
    """
    Validate the two-pass cascade's screening threshold on labeled held-out data (#282):
    anything pass 1 rejects never gets a second look, so the cascade must lose ~zero recall
    versus MC-scoring EVERYTHING at the science threshold. Advisory (WARNs loudly, never
    fails the run), same idiom as check_val_auc_floor. Returns the measured numbers,
    including the largest screening threshold that would have lost nothing on this split.
    """
    test_labels = np.asarray(test_labels).astype(bool)
    pass1_probas = np.asarray(pass1_probas)
    mc_mean_probas = np.asarray(mc_mean_probas)

    positives = test_labels
    mc_pass = mc_mean_probas > science_threshold
    n_positive_mc = int((mc_pass & positives).sum())
    recall_full = float(mc_pass[positives].mean()) if positives.any() else float("nan")
    cascade_pass = (pass1_probas > screening_threshold) & mc_pass
    recall_cascade = float(cascade_pass[positives].mean()) if positives.any() else float("nan")
    recall_loss = recall_full - recall_cascade
    max_safe = float(pass1_probas[positives & mc_pass].min()) if n_positive_mc else float("nan")

    stats = {
        "screen_recall_mc_everything": recall_full,
        "screen_recall_cascade": recall_cascade,
        "screen_recall_loss": recall_loss,
        "screen_max_safe_threshold": max_safe,
    }
    if recall_loss > recall_tolerance:
        logger.warning(
            f"SCREENING THRESHOLD UNSAFE ({tag}): the two-pass cascade at "
            f"screening_threshold={screening_threshold} loses {recall_loss:.4f} recall vs "
            f"MC-on-everything at the science threshold ({recall_full:.4f} -> "
            f"{recall_cascade:.4f}); the largest zero-loss screen on this split is "
            f"{max_safe:.4f}. Lower inference.screening_threshold before a science run."
        )
    else:
        logger.info(
            f"Screening threshold validated ({tag}): cascade recall {recall_cascade:.4f} vs "
            f"MC-on-everything {recall_full:.4f} (loss {recall_loss:.4f} <= tolerance "
            f"{recall_tolerance}); largest zero-loss screen on this split: {max_safe:.4f}"
        )
    return stats


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


# Eigen's EIGEN_MAX_ALIGN_BYTES: TF CHECK-fails — an uncatchable SIGABRT, not an exception
# — when a CPU kernel touches a tensor whose buffer is less aligned than this, so zero-copy
# wrapping is gated on the source pointer's alignment below
_TF_CPU_TENSOR_ALIGN = 64


def _as_cpu_tensor(array: np.ndarray) -> tf.Tensor:
    """
    Wrap `array` as a CPU tensor for graph-side gathers, zero-copy when possible.

    from_dlpack on a writable, 64-byte-aligned ndarray — which the mmap_mode="c"
    copy-on-write memmaps load_round_arrays returns always are (page-aligned mapping + the
    npy format's 64-byte header padding) — aliases the existing buffer (no allocation, no
    copy), so a ~98 GB round array becomes a TF tensor in microseconds and gathers stream
    through the OS page cache exactly like numpy fancy-indexing did.

    Everything else falls back to tf.convert_to_tensor, which copies — fine for small
    in-RAM test arrays, loudly logged for anything big enough to matter. The fallback
    covers: read-only arrays (numpy refuses to export them over dlpack) and buffers below
    Eigen alignment (plain numpy allocations are typically 16-byte aligned; TF would abort
    the process at kernel time, not raise, so this is checked proactively).
    """
    with tf.device("/CPU:0"):
        if array.ctypes.data % _TF_CPU_TENSOR_ALIGN == 0:
            try:
                return tf.experimental.dlpack.from_dlpack(array.__dlpack__())
            except (AttributeError, BufferError, TypeError, RuntimeError, ValueError):
                pass
        if array.nbytes > 1 << 30:
            logger.warning(
                f"Zero-copy dlpack wrap unavailable for a {array.nbytes / 1e9:.1f} GB array "
                f"(read-only, misaligned, or non-exportable); falling back to an in-RAM "
                f"copy. If this is round data, check that load_round_arrays uses "
                f"mmap_mode='c'."
            )
        return tf.convert_to_tensor(array)


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

    `data` must contain 'concatenated', 'true', 'false', and 'labels' — typically the
    copy-on-write memmaps returned by round_data.load_round_arrays(), though plain in-RAM
    ndarrays work too. The split is stratified across the 4 signal types (false_no_signal,
    false_with_rfi, true_only_eti, true_eti_rfi) — generation lays labels out sequentially
    within chunks, so a naive positional split would over-represent later signal types in val.

    The datasets are built as cheap index generators followed by a parallel, deterministic
    .map() of pure tf.gather ops over zero-copy tensor views of the backing arrays
    (_as_cpu_tensor), so the entire steady-state gather path runs in tf.data's C++ threadpool
    with no Python — and therefore no GIL — involvement. This is the durable #276 fix: the
    previous tf.numpy_function gather re-entered the interpreter from every map worker, which
    made its +38% idle-host win invert to a loss whenever any other Python thread (the #277
    DB drainer, the producer relay) competed for the GIL. deterministic=True preserves the
    exact batch order of the index stream, so parallelism changes throughput only, never
    epoch composition or ordering. Randomness lives at the epoch level (train_indices are
    reshuffled each pass); sorting *within* a batch only improves read locality and the model
    is order-invariant within a batch.

    Each index-stream element is one PER-REPLICA batch (replica r of global batch g is
    element g*num_replicas + r): strategy.distribute_datasets_from_function hands consecutive
    elements to consecutive replicas, reproducing exactly the contiguous split that
    experimental_distribute_dataset used to apply to whole global batches, while skipping the
    split op and letting InputOptions prefetch each replica's batches straight to its GPU
    (the host→device copy overlaps compute instead of serializing in the step). Consumers see
    the same distributed-dataset interface as before; with one replica the elements are
    byte-identical to the old whole-global-batch stream.

    Page-cache framing: gathers pull pages of the mmap_mode="c" arrays through the OS page
    cache, so after the first epoch a round's ~294 GB (at full-scale defaults) is served at
    RAM speed from otherwise-free memory on the 503 GB training nodes — but under memory
    pressure the kernel evicts pages instead of OOM-killing the process, which is exactly the
    failure mode the old in-RAM arrays hit.

    Each logical global batch has the signature ((concat, true, false), concat). Sample
    counts are trimmed to the global / effective batch size to keep all replicas evenly fed
    (so every epoch pass yields whole batches exactly); the holder is shared by both index
    generators so neither pays a memory cost beyond index subsets.
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
    # full copies via fancy indexing (~2x peak memory). Instead, both datasets gather
    # per-batch slices from the same original arrays using their respective index subsets —
    # only the in-flight batches are materialized.
    train_holder = TrainDataHolder(data["concatenated"], data["true"], data["false"])

    # Zero-copy CPU tensor views of the backing arrays for the graph-side gather map below.
    # The map functions capture these by reference; they keep the underlying mappings alive
    # until the datasets are dropped, which is why train_round's teardown deletes the
    # datasets before removing the round directory (deletion is safe even with a lingering
    # mapping — POSIX inode semantics — the pages just stay reclaimable until the drop).
    concat_t = _as_cpu_tensor(data["concatenated"])
    true_t = _as_cpu_tensor(data["true"])
    false_t = _as_cpu_tensor(data["false"])

    # Create index-generator functions yielding ONE EPOCH of per-replica index batches per
    # yield, as a (batches_per_epoch * num_replicas, per_replica_batch) int64 array — row
    # g*num_replicas + r is replica r's slice of global batch g (with one replica, the old
    # whole-global-batch stream unchanged). A pure-TF .unbatch() downstream streams the rows
    # out one per-replica batch at a time, so the interpreter is touched ONCE per epoch: at
    # heavy GIL contention (the #277 drainer flood regime), a per-micro-batch Python yield
    # measured a further 35% off end-to-end throughput; the per-epoch yield is immune. The
    # gathers happen in a parallel, deterministic .map() of pure TF ops below — never in
    # Python (#276: first the one-thread gather starved the GPUs, then the numpy_function
    # gather contended for the GIL under load).
    # All randomness stays in the generators, so the epoch-level rng consumption order — and
    # therefore the #49 reproducibility contract (same seed => same split AND same per-epoch
    # batch order) — is byte-identical to the previous implementations.
    def train_index_generator():
        while True:  # Make generators infinite to reset state between epochs
            with train_holder._lock:
                if train_holder._cleared:
                    return  # Exit if data already cleared

            # Copy train_indices because rng.shuffle mutates in-place
            indices = train_indices.copy()
            if shuffle:
                # Perform global shuffle on each epoch so each pass through the data is unique
                rng.shuffle(indices)
            epoch = indices.astype(np.int64, copy=False).reshape(-1, global_train_batch_size)
            if shuffle:
                # Within-batch sorted order improves memmap read locality; random batch
                # membership is already guaranteed by the epoch-level shuffle above
                epoch = np.sort(epoch, axis=1)
            yield epoch.reshape(-1, per_replica_batch_size)

    def val_index_generator():
        while True:  # Make generators infinite to reset state between epochs
            with train_holder._lock:
                if train_holder._cleared:
                    return  # Exit if data already cleared

            # Maintain val_indices order on each epoch (already sorted above): no gradients are
            # calculated during validation, and train_random_forest relies on the i-th encoded
            # val cadence corresponding to val_indices[i]
            yield val_indices.astype(np.int64, copy=False).reshape(-1, per_replica_val_batch_size)

    def map_gather(batch_indices):
        # Pure tf.data ops on CPU: the gather runs in TF's C++ threadpool, releases nothing
        # to Python, and parallel map workers scale with cores instead of convoying on the
        # GIL. tf.cast is a no-op passthrough for float32 round arrays and the host-side
        # upcast under round_array_dtype="float16" (the gather itself moves half the bytes;
        # the training graph sees float32 either way).
        with tf.device("/CPU:0"):
            concat_batch = tf.cast(tf.gather(concat_t, batch_indices), tf.float32)
            true_batch = tf.cast(tf.gather(true_t, batch_indices), tf.float32)
            false_batch = tf.cast(tf.gather(false_t, batch_indices), tf.float32)
        return (concat_batch, true_batch, false_batch), concat_batch

    # deterministic=True keeps the emitted batch ORDER identical to the index stream even
    # with parallel in-flight gathers (and matches --tf-deterministic-ops semantics, which
    # would force it anyway).
    def _distributed_from_index_generator(generator_fn, per_replica_rows):
        index_spec = tf.TensorSpec(shape=(None, per_replica_rows), dtype=tf.int64)

        def dataset_fn(_input_context):
            return (
                tf.data.Dataset.from_generator(generator_fn, output_signature=index_spec)
                .unbatch()  # epoch table -> per-replica index batches, in pure TF
                .map(map_gather, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
                .repeat()
                .prefetch(tf.data.AUTOTUNE)
            )

        # fetch_to_device prefetches each replica's next batches into device memory, so the
        # host→device copy overlaps compute instead of serializing inside the step. The
        # buffer costs per_replica_buffer_size * ~76 MB of VRAM per GPU at production shapes.
        input_options = tf.distribute.InputOptions(
            experimental_fetch_to_device=True,
            experimental_per_replica_buffer_size=2,
        )
        return strategy.distribute_datasets_from_function(dataset_fn, input_options)

    logger.info(
        f"Creating infinite per-replica datasets from index generators with global batch size - "
        f"Train: {global_train_batch_size}, Val: {global_val_batch_size} "
        f"(distributed across {num_replicas} GPUs)"
    )

    train_dataset_distributed = _distributed_from_index_generator(
        train_index_generator, per_replica_batch_size
    )
    val_dataset_distributed = _distributed_from_index_generator(
        val_index_generator, per_replica_val_batch_size
    )

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

    # Zero-copy CPU tensor view for the graph-side gather (same machinery as the train
    # datasets — see prepare_distributed_train_dataset / _as_cpu_tensor). This feeds
    # _capture_latent_snapshot every latent_viz_step_interval training steps, so Python in
    # this path used to tax every capture during the epoch loop.
    viz_t = _as_cpu_tensor(padded_data)

    # Index generator yielding one full pass of PER-REPLICA contiguous index slices per
    # yield (row g*num_replicas + r is replica r's slice of global batch g — the same
    # consecutive-element convention as the train datasets, reproducing the old contiguous
    # global-batch split exactly); a pure-TF .unbatch() streams the rows out, so Python is
    # touched once per pass rather than once per batch.
    # WARN: DO NOT SHUFFLE viz indices, OR ELSE YOU'LL BREAK plot_latent_space_gif() —
    # contiguous in-order slices preserve the original cadence order on every pass
    # (n_padded is an exact multiple of the global batch size, so slices are whole)
    def viz_index_generator():
        while True:  # Make generator infinite to reset state between passes
            with viz_holder._lock:
                if viz_holder._cleared:
                    return  # Exit if data already cleared

            yield np.arange(n_padded, dtype=np.int64).reshape(-1, per_replica_inf_batch_size)

    index_spec = tf.TensorSpec(shape=(None, per_replica_inf_batch_size), dtype=tf.int64)

    def map_gather(batch_indices):
        with tf.device("/CPU:0"):
            return tf.cast(tf.gather(viz_t, batch_indices), tf.float32)

    logger.info(
        f"Creating infinite per-replica viz dataset with global batch size: "
        f"{global_viz_batch_size} (distributed across {num_replicas} GPUs)"
    )

    def dataset_fn(_input_context):
        return (
            tf.data.Dataset.from_generator(viz_index_generator, output_signature=index_spec)
            .unbatch()  # pass table -> per-replica index batches, in pure TF
            .map(map_gather, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
            # NOTE: do we need repeat for viz dataset? run test without repeat & see if anything breaks?
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )

    input_options = tf.distribute.InputOptions(
        experimental_fetch_to_device=True,
        experimental_per_replica_buffer_size=2,
    )
    viz_dataset_distributed = strategy.distribute_datasets_from_function(dataset_fn, input_options)

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

        # Reproducibility (#279): seed TF's global RNG before any model/variable creation so
        # weight initialization (HeNormal/GlorotNormal) draws a deterministic stream. The
        # shared seeding helper is used by BOTH pipeline constructors so training and
        # inference can't drift; train_round re-seeds per round (sub-key = round number) so a
        # resumed run reproduces an uninterrupted one. numpy/python randomness is NOT
        # globally seeded here — each consumer derives its own independent stream from the
        # same root (see aetherscan.seeding). No-op when the root seed is None
        applied_tf_seed = seed_tensorflow(
            self.config.reproducibility.seed, self.config.reproducibility.tf_deterministic_ops, 0
        )
        if applied_tf_seed is not None:
            logger.info(
                f"Seeded TF global RNG from root seed {self.config.reproducibility.seed} "
                f"(derived training stream seed {applied_tf_seed})"
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

        # Opt-in bf16 mixed precision (A/B-gated; see BetaVAEConfig.mixed_precision): the
        # global policy must be set BEFORE the models are built so every layer picks it up.
        # bf16 needs no loss scaling, and keras keeps variables/optimizer state — and hence
        # the gradients reaching the fp32 accumulators and NaN guard — in fp32. Flag off =>
        # no policy call at all, preserving the fp32 pipeline's numerics byte-for-byte.
        if self.config.beta_vae.mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
            logger.info(
                "Mixed precision enabled: keras global policy set to mixed_bfloat16 "
                "(fp32 islands: z_mean/z_log_var/Sampling, decoder output, loss math)"
            )

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
        self._latent_viz_iterator = None
        self._latent_viz_n_padded = None
        self._latent_viz_n_samples = None
        self._latent_viz_steps = None
        self._latent_viz_holder = None
        self._viz_encode_fn = None

        # Graph-side accumulation state (built lazily by _ensure_accumulation_state on the
        # first epoch: the accumulator variables must be created inside strategy.scope, and
        # doing it here would pay the allocation even for resume paths that never train).
        # _accumulated_train_step_fns / _val_loop_fns cache one traced tf.function per
        # (accumulation_steps,) / (val_steps,) value so a config change can never silently
        # reuse a graph built for a different K.
        self._grad_accumulators: list[tf.Variable] | None = None
        self._train_loss_accumulators: dict[str, tf.Variable] | None = None
        self._val_loss_accumulator: tf.Variable | None = None
        self._unconnected_grad_indices: set[int] = set()
        self._accumulated_train_step_fns: dict[int, Callable] = {}
        self._val_loop_fns: dict[int, Callable] = {}

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
            # Load models from checkpoints per the resume plan (explicit user flags win;
            # otherwise a manifest-driven resume reloads the last completed round's checkpoint).
            # The decision is factored into _resume_load_plan() so it stays unit-testable
            # without building the TF graph.
            plan = self._resume_load_plan()
            if plan is not None:
                self.load_models(tag=plan[0], dir=plan[1])

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

    def _resume_load_plan(self) -> tuple[str | None, str | None] | None:
        """Decide which checkpoint ``(tag, dir)`` __init__ should load, or ``None`` for a fresh
        run (no load).

        Raises ``FileNotFoundError`` when a resume-in-place run has neither its own final weights
        nor a completed round to fall back to. This is a pure decision — its only I/O is the
        on-disk model-pair existence check — so the resume logic is unit-testable without building
        the TF graph.
        """
        load_tag = self.config.checkpoint.load_tag
        load_dir = self.config.checkpoint.load_dir
        save_tag = self.config.checkpoint.save_tag
        # Resume-in-place: a full-tag --load-tag that equals this run's save-tag (the tag was
        # adopted by resolve_save_tag, which never adopts a round_XX load-tag — so load_tag here is
        # always a full {prefix}_{datetime} tag). If the run's final weights aren't on disk yet
        # (the VAE didn't finish before the interruption), fall back to the manifest's last
        # completed round for THIS tag — never another run's checkpoints; fail loudly if neither.
        resume_in_place = load_tag is not None and load_tag == save_tag
        if resume_in_place and not _model_pair_exists(self.config.model_path, load_tag):
            if self._start_round > 1:
                resume_tag = f"round_{self._start_round - 1:02d}"
                logger.info(
                    f"No final weights for run {load_tag!r} yet; resuming from {resume_tag} "
                    f"(this run's last completed round, per manifest)"
                )
                return resume_tag, "checkpoints"
            raise FileNotFoundError(
                f"Cannot resume run {load_tag!r}: no final weights on disk and no completed "
                f"rounds recorded in its manifest. Refusing to load another run's checkpoints."
            )
        if load_tag or load_dir:
            logger.info("Resuming from checkpoint")
            return load_tag, load_dir
        if self._start_round > 1:
            resume_tag = f"round_{self._start_round - 1:02d}"
            logger.info(f"Resuming from checkpoint {resume_tag} (per run manifest)")
            return resume_tag, "checkpoints"
        return None

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

        # Resume-in-place (--load-tag {full_tag} == this run's save_tag) is a CONTINUATION of the
        # same run, not a user override — so the manifest (completed_rounds) drives resume, exactly
        # as a same-command rerun would. Only a genuinely explicit checkpoint (a round_XX/other
        # --load-tag, or --load-dir) is treated as an override that restarts from start_round and
        # clears the manifest.
        resume_in_place = (
            self.config.checkpoint.load_tag is not None and self.config.checkpoint.load_tag == tag
        )
        explicit_checkpoint = (not resume_in_place) and (
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

        plot_checkpoints_dir = self._training_plots_dir("checkpoints")
        archive_directory(plot_checkpoints_dir, target_dirs=None, round_num=start_round)

        # Disk-backed round-data directory for this tag: delete round dirs >= start_round,
        # keep earlier ones only if their .done manifest validates (round-data mirror of the
        # checkpoint archiving above, minus the archiving — a round is ~295 GB)
        round_data_root = self.config.training.round_data_dir or self.config.get_training_file_path(
            "round_data"
        )
        self._round_data_base_dir = os.path.join(round_data_root, self.config.checkpoint.save_tag)
        prepare_round_data_dir(
            self._round_data_base_dir,
            start_round,
            expected_array_dtype=self.config.training.round_array_dtype,
        )

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
                    seed=self.config.reproducibility.seed,
                    array_dtype=self.config.training.round_array_dtype,
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

        # Re-seed TF's global RNG per round, sub-keyed by round number (#279): a resumed run
        # skips completed rounds, which would otherwise leave the single __init__-time stream
        # at a different position than an uninterrupted run — with per-round keys, round k's
        # TF draws (VAE sampling, dropout-free here but future-proof) depend only on
        # (root, round). deterministic_ops=False: already applied once in __init__
        seed_tensorflow(self.config.reproducibility.seed, False, 0, round_number)

        # Fresh low-KL streak counters for this round's posterior-collapse guard (#282)
        self._kl_low_streaks = np.zeros(self.config.beta_vae.latent_dim, dtype=np.int64)

        # Obtain this round's disk-backed data: reuse a validated on-disk dataset if one
        # exists, otherwise wait on the background producer (which was asked to generate it
        # while the previous round trained) or generate in-process (overlap disabled)
        if (
            validate_done_manifest(
                paths,
                expected_n_samples=n_samples,
                expected_array_dtype=self.config.training.round_array_dtype,
            )
            is not None
        ):
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
            rng=derive_rng(self.config.reproducibility.seed, STREAM_DATASET, round_number),
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

        train_iterator = None
        val_iterator = None
        try:
            # ONE distributed iterator per dataset for the WHOLE round, not one per epoch:
            # the datasets are infinite (the index generators reshuffle at every wrap, so
            # epoch boundaries are purely step-counted), and a fresh per-epoch iterator both
            # discards whatever the pipeline had prefetched across the boundary and hands
            # the traced step functions a new iterator object every epoch — under
            # MirroredStrategy that meant a retrace of the accumulated-step and val-loop
            # graphs per epoch, which dominated the epoch wall at production step counts.
            train_iterator = iter(train_dataset)
            val_iterator = iter(val_dataset)

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
                        train_iterator,
                        steps_per_epoch,
                        accumulation_steps,
                        time.time(),
                    )

                    # Validation
                    val_losses, val_duration = self._validate_epoch(
                        val_iterator, val_steps, time.time()
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
                        ("reg_loss", "reg"),
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

                    # Per-dimension KL (#282 posterior-collapse diagnostics): latent_dim
                    # rows per epoch — negligible volume, powers the KL heatmap and the
                    # active-units-vs-epoch curve
                    kl_per_dim = np.asarray(epoch_losses["kl_per_dim"]).ravel()
                    for dim_idx, dim_kl in enumerate(kl_per_dim):
                        self.db.write_training_stat(
                            model_name="beta_vae",
                            stat_name=f"kl_dim_{dim_idx:02d}",
                            value=float(dim_kl),
                            round_number=round_idx + 1,
                            epoch_number=epoch + 1,
                            tag=self.config.checkpoint.save_tag,
                            timestamp=current_time,
                        )
                    # Consecutive low-KL epoch streaks feed check_posterior_collapse at
                    # round end (a dim parked under epsilon for `patience` epochs is
                    # collapsing even if the batch-mean wobbles above zero)
                    low = kl_per_dim < self.config.training.posterior_collapse_kl_epsilon
                    self._kl_low_streaks = np.where(low, self._kl_low_streaks + 1, 0)

                    # Validation losses
                    for stat_name, key in [
                        ("val_total_loss", "total"),
                        ("val_reconstruction_loss", "reconstruction"),
                        ("val_kl_loss", "kl"),
                        ("val_true_loss", "true"),
                        ("val_false_loss", "false"),
                        ("val_reg_loss", "reg"),
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
                        f"Reg: {epoch_losses['reg']:.4f}, "
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
                        f"Reg: {val_losses['reg']:.4f}, "
                        f"Duration: {val_duration:.2f} "
                    )

                    # Adaptive learning rate
                    self._update_learning_rate(val_losses)

            # Posterior-collapse guard (#282): WARN loudly (reaches Slack) when latent dims
            # are going dark — same non-fatal advisory idiom as check_val_auc_floor
            check_posterior_collapse(
                kl_per_dim=np.asarray(epoch_losses["kl_per_dim"]).ravel(),
                low_kl_streaks=self._kl_low_streaks,
                kl_epsilon=self.config.training.posterior_collapse_kl_epsilon,
                min_active_fraction=self.config.training.min_active_units_fraction,
                patience=self.config.training.posterior_collapse_patience,
                tag=f"round_{round_idx + 1:02d}",
            )

            # NOTE: combine plot_beta_vae_loss_curves(), plot_beta_vae_training_stability(), and plot_latent_space_gif() into plot_training_progress()?
            with stage_timer("plots"):
                # Per-round plots are best-effort (#277): a failed or skipped plot must never
                # fail the round (a retry would regenerate the round data from scratch). The
                # canonical full-run figures render again in the vae_plots stage, where
                # _run_plot_group records failures as non-critical.
                per_round_plots = [
                    (
                        "plot_beta_vae_loss_curves",
                        lambda: self.plot_beta_vae_loss_curves(
                            tag=f"round_{round_idx + 1:02d}", dir="checkpoints"
                        ),
                    ),
                    (
                        "plot_beta_vae_training_stability",
                        lambda: self.plot_beta_vae_training_stability(
                            tag=f"round_{round_idx + 1:02d}", dir="checkpoints"
                        ),
                    ),
                    (
                        "plot_injection_stats",
                        lambda: self.plot_injection_stats(
                            tag=f"round_{round_idx + 1:02d}",
                            dir="checkpoints",
                            round_number=round_idx + 1,
                        ),
                    ),
                ]
                for plot_name, plot_fn in per_round_plots:
                    try:
                        plot_fn()
                    except Exception as e:
                        logger.error(
                            f"Skipped/failed {plot_name} for round {round_idx + 1} "
                            f"(non-critical, round continues): {e}"
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
            # Iterators first (bench_input_pipeline teardown order: iterator -> dataset ->
            # clear), then clear intermediate data
            del train_iterator, val_iterator
            train_holder.clear()
            del train_dataset, val_dataset

            # Clear latent viz distributed dataset (rebuilt each round); iterator dropped
            # BEFORE the holder clear, same teardown order as the train/val iterators
            self._latent_viz_iterator = None
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

    # Maps epoch-loss dict keys to the keys compute_total_loss returns; "kl_per_dim" is the
    # vector-valued (latent_dim,) #282 diagnostic and accumulates like the scalars
    _LOSS_KEY_MAP = (
        ("total", "total_loss"),
        ("reconstruction", "reconstruction_loss"),
        ("kl", "kl_loss"),
        ("true", "true_loss"),
        ("false", "false_loss"),
        ("reg", "reg_loss"),
        ("kl_per_dim", "kl_per_dim"),
    )

    def _ensure_accumulation_state(self):
        """
        Lazily create the graph-side accumulation state: one gradient accumulator per
        trainable variable, the train loss accumulators (the _LOSS_KEY_MAP scalars + the
        (latent_dim,) per-dim KL vector), and the scalar-count-sized val loss accumulator.

        ON_READ synchronization makes assign_add inside strategy.run a replica-LOCAL update
        (no communication per micro-batch); reading the variable back in cross-replica
        context performs a single aggregation — SUM across replicas for gradients, MEAN for
        losses. One optimizer step therefore costs exactly one cross-replica reduction per
        variable, instead of one per variable per micro-batch as the pre-#276-follow-up
        implementation did. Created inside strategy.scope() per the repo hard rule.
        """
        if self._grad_accumulators is not None:
            return

        with self.strategy.scope():
            self._grad_accumulators = [
                tf.Variable(
                    tf.zeros_like(v),
                    trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ,
                    aggregation=tf.VariableAggregation.SUM,
                )
                for v in self.vae.trainable_variables
            ]

            def _loss_accumulator(shape):
                return tf.Variable(
                    tf.zeros(shape, dtype=tf.float32),
                    trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ,
                    aggregation=tf.VariableAggregation.MEAN,
                )

            self._train_loss_accumulators = {
                name: _loss_accumulator(
                    [self.config.beta_vae.latent_dim] if name == "kl_per_dim" else []
                )
                for name, _ in self._LOSS_KEY_MAP
            }
            # One vector for the val losses, ordered as _LOSS_KEY_MAP minus kl_per_dim
            self._val_loss_accumulator = _loss_accumulator([len(self._LOSS_KEY_MAP) - 1])

    def _accumulate_micro_batch(self, batch_data):
        """
        Per-replica micro-batch: forward + backward, accumulating gradients and losses into
        the replica-local ON_READ accumulators. Runs under strategy.run inside the
        accumulated-train-step graph — there is no cross-replica communication here.
        """
        x, y = batch_data
        main_data = x[0]
        true_data = x[1]
        false_data = x[2]

        with tf.GradientTape() as tape:
            losses = self.vae.compute_total_loss(main_data, true_data, false_data, y, training=True)

        gradients = tape.gradient(losses["total_loss"], self.vae.trainable_variables)

        for idx, (accumulator, grad) in enumerate(
            zip(self._grad_accumulators, gradients, strict=False)
        ):
            if grad is None:
                # Trace-time bookkeeping (this branch never becomes graph ops): the variable
                # is structurally disconnected from the loss, so the apply step must skip it
                # — exactly what apply_gradients did with the None entries the previous
                # Python-side accumulation forwarded. Expected empty for the beta-VAE.
                # The set is instance-scoped and shared across every traced K: that is safe
                # only because loss↔variable connectivity is a property of the model graph,
                # independent of accumulation depth — if compute_total_loss ever grows
                # K-dependent branches, give each traced K its own set.
                self._unconnected_grad_indices.add(idx)
            else:
                accumulator.assign_add(grad)

        for name, loss_key in self._LOSS_KEY_MAP:
            self._train_loss_accumulators[name].assign_add(losses[loss_key])

    def _accumulate_val_micro_batch(self, batch_data):
        """Per-replica val micro-batch: forward only, losses into the scalar accumulator."""
        x, y = batch_data
        losses = self.vae.compute_total_loss(x[0], x[1], x[2], y, training=False)
        self._val_loss_accumulator.assign_add(
            tf.stack([losses[key] for _, key in self._LOSS_KEY_MAP[:-1]])
        )

    def _get_accumulated_train_step(self, accumulation_steps: int) -> Callable:
        """Traced accumulated-train-step for this K, built once and cached."""
        fn = self._accumulated_train_step_fns.get(accumulation_steps)
        if fn is None:
            fn = self._build_accumulated_train_step(accumulation_steps)
            self._accumulated_train_step_fns[accumulation_steps] = fn
        return fn

    def _build_accumulated_train_step(self, accumulation_steps: int) -> Callable:
        """
        Build the tf.function that performs ONE full optimizer step: K micro-batches
        accumulated per replica, one cross-replica reduction per variable, NaN/Inf guard,
        global-norm clip, apply, reset. The interpreter is re-entered once per optimizer
        step instead of once per micro-batch (plus per-variable eager ops), which removes
        the Python/GIL cost that dominated the #276 audit's host-side wall — and makes the
        steady-state loop immune to GIL competition from the DB writer / producer threads.

        tf.range (not python range) keeps the K micro-batches as a sequential graph loop:
        one traced body instead of K unrolled copies, and — critically for the 16 GB
        A4000 release host — at most one micro-batch's activations in flight per replica,
        so peak VRAM stays at the K=1 level (an unrolled loop measured 23.4 GB/GPU at
        K=12 on blpc3 from overlapped activations).

        Gradient semantics match the previous implementation exactly (up to float summation
        order): per-replica sums over K micro-batches, MEAN across replicas, averaged over
        K, then clipped by global L2 norm at 1.0 and applied once. Clipping rationale
        (unchanged): the global-norm form preserves the gradient direction while bounding
        magnitude — appropriate for this beta-VAE's heterogeneous gradient scales
        (reconstruction + KL components); per-tensor clipping would distort direction.
        Healthy training keeps global_norm below clip_norm with only occasional clips; if
        clipping becomes frequent, raise clip_norm rather than dampening learning.
        """
        strategy = self.strategy

        @tf.function
        def accumulated_train_step(iterator):
            for _ in tf.range(accumulation_steps):
                strategy.run(self._accumulate_micro_batch, args=(next(iterator),))

            # Reading the ON_READ accumulators in cross-replica context SUMs across
            # replicas; scale to the mean-over-replicas, mean-over-micro-batches gradient
            scale = tf.cast(accumulation_steps * strategy.num_replicas_in_sync, tf.float32)
            gradients = [acc.read_value() / scale for acc in self._grad_accumulators]

            trainable = self.vae.trainable_variables
            included = [
                (grad, var)
                for idx, (grad, var) in enumerate(zip(gradients, trainable, strict=False))
                if idx not in self._unconnected_grad_indices
            ]
            included_grads = [grad for grad, _ in included]

            # NaN/Inf guard on the averaged pre-clip gradients (same point the previous
            # implementation checked); the apply is skipped on a bad step so the weights
            # are never corrupted — the Python caller then raises, matching the old
            # behavior observable to everything downstream
            all_finite = tf.reduce_all(
                tf.stack([tf.reduce_all(tf.math.is_finite(g)) for g in included_grads])
            )
            global_norm = tf.linalg.global_norm(included_grads)

            def _clip_and_apply():
                clipped, _ = tf.clip_by_global_norm(
                    included_grads, _GRADIENT_CLIP_NORM, use_norm=global_norm
                )
                self.vae.optimizer.apply_gradients(
                    zip(clipped, [var for _, var in included], strict=False)
                )
                return tf.constant(True)

            applied = tf.cond(all_finite, _clip_and_apply, lambda: tf.constant(False))

            # Aggregated (MEAN across replicas) losses, averaged over the K micro-batches.
            # Auto control dependencies order these reads after the accumulation loop and
            # the resets below after the reads — resource ops on one variable never reorder
            losses = {
                name: self._train_loss_accumulators[name].read_value()
                / tf.cast(accumulation_steps, tf.float32)
                for name, _ in self._LOSS_KEY_MAP
            }

            for acc in self._grad_accumulators:
                acc.assign(tf.zeros_like(acc))
            for var in self._train_loss_accumulators.values():
                var.assign(tf.zeros_like(var))

            return losses, global_norm, applied

        return accumulated_train_step

    def _get_val_loop(self, val_steps: int) -> Callable:
        """Traced whole-epoch val loop for this step count, built once and cached."""
        fn = self._val_loop_fns.get(val_steps)
        if fn is None:
            fn = self._build_val_loop(val_steps)
            self._val_loop_fns[val_steps] = fn
        return fn

    def _build_val_loop(self, val_steps: int) -> Callable:
        """One tf.function for the whole validation epoch: val_steps forward micro-batches
        accumulated per replica, one MEAN aggregation at the end — a single interpreter
        re-entry per val epoch instead of one (plus five strategy.reduce calls) per batch."""
        strategy = self.strategy

        @tf.function
        def val_loop(iterator):
            for _ in tf.range(val_steps):
                strategy.run(self._accumulate_val_micro_batch, args=(next(iterator),))
            totals = self._val_loss_accumulator.read_value() / tf.cast(val_steps, tf.float32)
            self._val_loss_accumulator.assign(tf.zeros_like(self._val_loss_accumulator))
            return totals

        return val_loop

    def _train_epoch(
        self,
        round_idx,
        epoch_idx,
        snr_base,
        snr_range,
        iterator,
        steps_per_epoch,
        accumulation_steps=1,
        start_time=None,
    ):
        """
        Perform a single training epoch with gradient accumulation.

        `iterator` is the round-scoped distributed iterator train_round creates once and
        passes to every epoch (the datasets are infinite; epoch boundaries are step-counted,
        and the underlying index generators reshuffle at each wrap). Handing the SAME
        iterator object to the traced step function every epoch is what keeps it from
        retracing — a fresh per-epoch iterator forced a per-epoch retrace of the
        accumulated-step graph under MirroredStrategy.

        Each optimizer step is one call into the traced accumulated-train-step graph (see
        _build_accumulated_train_step); Python-side work per step is a handful of scalar
        fetches and dict updates, plus the periodic latent snapshot.
        """
        if not start_time:
            start_time = time.time()

        self._ensure_accumulation_state()
        train_step_fn = self._get_accumulated_train_step(accumulation_steps)

        epoch_losses = {
            "total": 0.0,
            "reconstruction": 0.0,
            "kl": 0.0,
            "true": 0.0,
            "false": 0.0,
            "reg": 0.0,
            # Vector-valued (latent_dim,) — accumulates like the scalars (#282 diagnostics)
            "kl_per_dim": np.zeros(self.config.beta_vae.latent_dim, dtype=np.float64),
        }
        epoch_gradient_norms = []

        try:
            for step in range(steps_per_epoch):
                step_losses, global_norm, applied = train_step_fn(iterator)

                if not bool(applied.numpy()):
                    logger.error(f"Step {step + 1}: NaN/Inf detected in averaged gradients!")
                    raise RuntimeError(f"NaN/Inf gradients at step {step + 1}")

                epoch_gradient_norms.append(float(global_norm.numpy()))

                # Accumulate epoch losses over training steps (already averaged over
                # micro-batches and replicas in-graph)
                for key, value in step_losses.items():
                    if key == "kl_per_dim":
                        epoch_losses[key] += value.numpy().astype(np.float64)
                    else:
                        epoch_losses[key] += float(value.numpy())

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

            # Average epoch losses over training steps
            for key in epoch_losses:
                epoch_losses[key] /= steps_per_epoch

            # Calculate train epoch duration
            train_duration = time.time() - start_time

            return epoch_losses, epoch_gradient_norms, train_duration

        except Exception as e:
            logger.error(f"Error in _train_epoch(): {e}")
            raise  # Re-raise to propagate error

        # Run cleanup regardless if epoch finishes successfully or not (the round-scoped
        # iterator is train_round's to tear down)
        finally:
            gc.collect()

    def _validate_epoch(self, iterator, steps, start_time=None):
        """
        Perform a single validation epoch (one traced graph call — see _build_val_loop).
        `iterator` is round-scoped, like _train_epoch's — same retrace rationale.
        """
        if not start_time:
            start_time = time.time()

        self._ensure_accumulation_state()
        val_loop_fn = self._get_val_loop(steps)

        try:
            totals = val_loop_fn(iterator).numpy()
            val_losses = {
                name: float(value)
                for (name, _), value in zip(self._LOSS_KEY_MAP[:-1], totals, strict=False)
            }

            # Calculate val epoch duration
            val_duration = time.time() - start_time

            return val_losses, val_duration

        except Exception as e:
            logger.error(f"Error in _validate_epoch(): {e}")
            raise  # Re-raise to propagate error

        # Run cleanup regardless if epoch finishes successfully or not (the round-scoped
        # iterator is train_round's to tear down)
        finally:
            gc.collect()

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
        self, dataset, n_steps, encode_fn, n_samples, latent_dim, logging=False, iterator=None
    ):
        """
        Run a provided @tf.function (`encode_fn`) over `n_steps` of a distributed `dataset` and
        return a list of (n_samples, latent_dim) ndarrays — one per tensor that encode_fn yields.

        The number of output arrays is inferred from encode_fn's return on the first step: a bare
        PerReplica tensor yields 1 output, a tuple of PerReplica tensors yields N. Per-replica
        results are gathered via experimental_local_results + np.concatenate, which is faster than
        a strategy-level gather over NCCL for the small latent payload.

        `iterator` lets a caller that encodes REPEATEDLY from the same in-order infinite
        dataset (the per-snapshot viz encode) supply a persistent iterator instead of paying
        iter() + a gc pass per call (~1.2 s of the ~1.5 s per-snapshot cost, measured); the
        caller then owns the iterator's lifecycle. One-shot callers (the RF stage) omit it
        and keep the original create/teardown behavior.
        """
        # Process all batches
        owns_iterator = iterator is None
        if owns_iterator:
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
            if owns_iterator:
                del iterator
                # One collection per encode pass — a per-batch gc.collect() here used to
                # burn GIL-holding milliseconds hundreds of times per pass; callers with a
                # persistent iterator skip even this (0.33 s/call measured, and the epoch
                # loop's snapshot path is exactly that caller)
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

        # Re-seed TF for the RF stage, sub-keyed by the same num_training_rounds+1 sentinel
        # as its data generation (#279): the sampled-z encodes below then reproduce whether
        # the stage runs after 20 in-process rounds or first thing on a resumed attempt
        seed_tensorflow(
            self.config.reproducibility.seed, False, 0, self.config.training.num_training_rounds + 1
        )

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
        if (
            validate_done_manifest(
                rf_paths,
                expected_n_samples=n_samples,
                expected_array_dtype=self.config.training.round_array_dtype,
            )
            is not None
        ):
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
            # when _start_round == num_training_rounds + 1. Since #277, plot_injection_stats
            # scopes its queries to rounds 1..num_training_rounds, so this sentinel round is
            # deliberately OUTSIDE every plotted range — RF-phase rows never appear in the
            # injection figures. No other consumer filters or groups on round_number, so this
            # is safe without touching mark_superseded or the DB schema.
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
            # RF dataset uses the round-0 STREAM_DATASET key (beta-VAE rounds are 1-based,
            # so no collision). Its DATA GENERATION uses the num_training_rounds+1 sentinel
            # on STREAM_DATA_GEN (see the round_num comment above) — different stream ids,
            # so the keys need not (and do not) match.
            rng=derive_rng(self.config.reproducibility.seed, STREAM_DATASET, 0),
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

        # Create distributed inference function. Since #282 it returns ALL THREE encoder
        # outputs — z_mean and z_log_var used to be computed and thrown away here, which is
        # exactly what makes the 8-variant representation sweep below free on the GPU side.
        @tf.function
        def rf_encode_fn(batch_data):
            """Encode batch data using distributed strategy"""

            def encode_fn(data):
                """Per-replica encoding step"""
                (concat_data, _, _), _ = data

                # Reshape for encoder: (batch, 6, 16, 512) -> (batch * 6, 16, 512, 1)
                concat_reshaped = tf.reshape(concat_data, [-1, time_bins, width_bin, 1])

                # Encode (returns z_mean, z_log_var, z)
                z_mean, z_log_var, concat_z = self.vae.encoder(concat_reshaped, training=False)

                return z_mean, z_log_var, concat_z

            return self.strategy.run(encode_fn, args=(batch_data,))

        train_z_mean = train_z_log_var = train_latents = None
        val_z_mean = val_z_log_var = val_latents = None

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
                [train_z_mean, train_z_log_var, train_latents] = self._distributed_encode(
                    dataset=train_dataset,
                    n_steps=train_encode_steps,
                    encode_fn=rf_encode_fn,
                    n_samples=n_train_trimmed * num_observations,
                    latent_dim=latent_dim,
                    logging=True,
                )

                [val_z_mean, val_z_log_var, val_latents] = self._distributed_encode(
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

            # ---------------------------------------------------------------- #282 sweep
            # Flatten every encoder output into per-cadence blocks once; every variant's
            # features are hstack compositions of these (see aetherscan.latent_variants)
            train_mean_flat = prepare_latent_features(train_z_mean, num_observations)
            train_logvar_flat = prepare_latent_features(train_z_log_var, num_observations)
            train_z_flat = prepare_latent_features(train_latents, num_observations)
            val_mean_flat = prepare_latent_features(val_z_mean, num_observations)
            val_logvar_flat = prepare_latent_features(val_z_log_var, num_observations)

            root_seed = self.config.reproducibility.seed
            rf_seed = self.config.resolved_rf_seed()
            classification_threshold = self.config.inference.classification_threshold
            tag = self.config.checkpoint.save_tag

            # Active Units (Burda et al.) on the train z_mean pool: sizes the collapse
            # problem retroactively and gates the z_mean_logvar_active variant
            active_dims = active_latent_dims(
                train_mean_flat,
                num_observations,
                latent_dim,
                self.config.rf.active_units_threshold,
            )
            # Per-dim variances logged alongside the count: an AU flip between runs is only
            # interpretable with the margins (hovering-at-threshold = noise, dark = signal)
            dim_variances = latent_dim_variances(train_mean_flat, num_observations, latent_dim)
            logger.info(
                f"Active latent dims (z_mean variance > "
                f"{self.config.rf.active_units_threshold}): {len(active_dims)}/{latent_dim} "
                f"-> {active_dims}; per-dim variances: "
                f"{[round(float(v), 5) for v in dim_variances]}"
            )

            # Partition the val split: selection (variant choice) / calibration (calibrator
            # fit) / test (held-out release metrics — best-of-8 on the selection split alone
            # would be optimistically biased). Seeded so the partition reproduces.
            n_val_rows = len(val_binary_labels)
            partition_rng = derive_rng(root_seed, STREAM_RF, 1)
            permutation = partition_rng.permutation(n_val_rows)
            n_selection = int(round(self.config.rf.val_selection_fraction * n_val_rows))
            n_calibration = int(round(self.config.rf.val_calibration_fraction * n_val_rows))
            selection_idx = permutation[:n_selection]
            calibration_idx = permutation[n_selection : n_selection + n_calibration]
            test_idx = permutation[n_selection + n_calibration :]

            # Train EVERY variant on the same data and split; evaluate each under its
            # DETERMINISTIC inference-time form (z_mean in the lead slot — for the z/z_aug
            # variants that is deliberately the deployed configuration, not the training
            # one). Cheap metrics for all; full diagnostics later for the winner only.
            def _variant_train_matrix(variant):
                if variant == "z":
                    return train_z_flat, train_binary_labels
                if variant == "z_aug":
                    return build_z_aug_training_set(
                        train_mean_flat,
                        train_logvar_flat,
                        train_binary_labels,
                        self.config.rf.z_aug_draws,
                        derive_rng(root_seed, STREAM_RF, 2),
                    )
                features = build_variant_features(
                    variant,
                    train_mean_flat,
                    train_logvar_flat,
                    num_observations,
                    latent_dim,
                    active_dims,
                )
                return features, train_binary_labels

            variant_models: dict[str, RandomForestClassifier] = {}
            variant_val_probas: dict[str, np.ndarray] = {}
            variant_metrics: dict[str, dict[str, float]] = {}
            with stage_timer("fit"):
                for variant in VARIANT_ORDER:
                    fit_features, fit_labels = _variant_train_matrix(variant)
                    fit_features, fit_labels = sklearn_shuffle(
                        fit_features, fit_labels, random_state=rf_seed
                    )
                    clf = RandomForestClassifier(
                        n_estimators=self.config.rf.n_estimators,
                        bootstrap=self.config.rf.bootstrap,
                        max_features=self.config.rf.max_features,
                        n_jobs=self.config.rf.n_jobs,
                        random_state=rf_seed,
                    )
                    clf.fit(fit_features, fit_labels)
                    variant_models[variant] = clf

                    eval_features = build_variant_features(
                        variant,
                        val_mean_flat,
                        val_logvar_flat,
                        num_observations,
                        latent_dim,
                        active_dims,
                    )
                    probas = clf.predict_proba(eval_features)[:, 1].astype(np.float32)
                    variant_val_probas[variant] = probas

                    sel_labels = val_binary_labels[selection_idx]
                    sel_probas = probas[selection_idx]
                    metrics = {
                        "recall_at_fpr": recall_at_fpr(
                            sel_labels, sel_probas, self.config.rf.selection_max_fpr
                        ),
                        "roc_auc": float(roc_auc_score(sel_labels, sel_probas)),
                        "brier": float(brier_score_loss(sel_labels, sel_probas)),
                        "ece": expected_calibration_error(sel_labels, sel_probas),
                        "n_features": int(fit_features.shape[1]),
                    }
                    variant_metrics[variant] = metrics
                    logger.info(
                        f"Variant '{variant}' (F={metrics['n_features']}): "
                        f"recall@{self.config.rf.selection_max_fpr:g}FPR="
                        f"{metrics['recall_at_fpr']:.4f}, AUC={metrics['roc_auc']:.4f}, "
                        f"Brier={metrics['brier']:.4f}, ECE={metrics['ece']:.4f}"
                    )
                    variant_path = os.path.join(
                        self.config.model_path, f"random_forest_{tag}_{variant}.joblib"
                    )
                    os.makedirs(os.path.dirname(variant_path), exist_ok=True)
                    joblib.dump(clf, variant_path)
                    del fit_features, fit_labels, eval_features
                    gc.collect()

            winner, selection_recalls = select_winner(
                val_binary_labels[selection_idx],
                {name: probas[selection_idx] for name, probas in variant_val_probas.items()},
                self.config.rf.selection_max_fpr,
                self.config.rf.selection_bootstrap_rounds,
                derive_rng(root_seed, STREAM_RF, 3),
            )
            recall_summary = ", ".join(f"{k}={v:.4f}" for k, v in selection_recalls.items())
            logger.info(f"LATENT VARIANT WINNER: '{winner}' (selection recalls: {recall_summary})")
            # MC coherence note for the record (docs carry the full explanation): only a
            # z-trained forest makes MC averaging a true posterior-predictive expectation
            if winner == "z":
                logger.info(
                    "MC semantics: the winner trains on sampled z, so pass-2 MC averaging is "
                    "a posterior-predictive expectation"
                )
            else:
                logger.info(
                    "MC semantics: the winner trains on deterministic features, so pass-2 MC "
                    "is a sensitivity/robustness probe (documented in INFERENCE_PIPELINE.md)"
                )

            # The winning fitted forest becomes THE model: canonical filename, HF upload,
            # and release tagging all pick it up unchanged
            self.rf_model.model = variant_models[winner]
            self.rf_model.is_trained = True
            # Record the winner + calibration outcome on the config singleton so
            # final_save's config_{tag}.json tells inference exactly how to rebuild
            # features. Single-threaded orchestration point (same precedent as the
            # startup-time save_tag resolution) — not a mid-flight mutation.
            self.config.rf.latent_variant = winner
            self.config.rf.active_dims = active_dims

            val_probas = variant_val_probas[winner]
            val_features = build_variant_features(
                winner, val_mean_flat, val_logvar_flat, num_observations, latent_dim, active_dims
            )
            if winner == "z":
                train_features = train_z_flat
            elif winner == "z_aug":
                train_features = train_mean_flat
            else:
                train_features = build_variant_features(
                    winner,
                    train_mean_flat,
                    train_logvar_flat,
                    num_observations,
                    latent_dim,
                    active_dims,
                )

            # -------------------------------------------------- #282 calibration (winner)
            # Measure ECE on the held-out calibration split; auto-fit a calibrator only when
            # the configured limit is exceeded, and KEEP it only if it demonstrably improves
            # ECE (without worsening Brier) on the further held-out test split.
            calibrator = None
            cal_labels = val_binary_labels[calibration_idx]
            cal_probas = val_probas[calibration_idx]
            measured_ece = expected_calibration_error(cal_labels, cal_probas)
            logger.info(
                f"Winner ECE on the calibration split: {measured_ece:.4f} "
                f"(rf.max_ece={self.config.rf.max_ece})"
            )
            if measured_ece > self.config.rf.max_ece:
                candidate = fit_probability_calibrator(
                    cal_probas, cal_labels, self.config.rf.calibration_min_isotonic
                )
                test_labels = val_binary_labels[test_idx]
                raw_test = val_probas[test_idx]
                calibrated_test = apply_probability_calibrator(candidate, raw_test)
                ece_before = expected_calibration_error(test_labels, raw_test)
                ece_after = expected_calibration_error(test_labels, calibrated_test)
                brier_before = float(brier_score_loss(test_labels, raw_test))
                brier_after = float(brier_score_loss(test_labels, calibrated_test))
                if ece_after < ece_before and brier_after <= brier_before * 1.001:
                    calibrator = candidate
                    logger.info(
                        f"Calibrator ({candidate['method']}) KEPT: test ECE "
                        f"{ece_before:.4f} -> {ece_after:.4f}, Brier {brier_before:.4f} -> "
                        f"{brier_after:.4f}"
                    )
                else:
                    logger.warning(
                        f"Calibrator ({candidate['method']}) DISCARDED — did not improve on "
                        f"the held-out test split (ECE {ece_before:.4f} -> {ece_after:.4f}, "
                        f"Brier {brier_before:.4f} -> {brier_after:.4f})"
                    )
            else:
                logger.info("Calibration not applied: measured ECE within rf.max_ece")

            self.config.rf.calibration_active = calibrator is not None
            self.config.rf.calibration_method = calibrator["method"] if calibrator else None
            if calibrator is not None:
                calibrator_path = os.path.join(
                    self.config.model_path, f"rf_calibrator_{tag}.joblib"
                )
                joblib.dump(calibrator, calibrator_path)
                logger.info(f"Saved probability calibrator to {calibrator_path}")

            # Deployment-scored probabilities/predictions for artifacts + plots: calibrated
            # when a calibrator is active (inference applies it identically), raw otherwise
            val_probas_deployed = apply_probability_calibrator(calibrator, val_probas).astype(
                np.float32
            )
            val_preds = (val_probas_deployed >= classification_threshold).astype(np.int64)

            # ------------------------------------------- #282 held-out test-split metrics
            test_labels = val_binary_labels[test_idx]
            test_deployed = val_probas_deployed[test_idx]
            release_metrics = {
                "test_recall_at_fpr": recall_at_fpr(
                    test_labels, test_deployed, self.config.rf.selection_max_fpr
                ),
                "test_roc_auc": float(roc_auc_score(test_labels, test_deployed)),
                "test_brier": float(brier_score_loss(test_labels, test_deployed)),
                "test_ece": expected_calibration_error(test_labels, test_deployed),
            }
            release_summary = ", ".join(f"{k}={v:.4f}" for k, v in release_metrics.items())
            logger.info(f"Held-out test metrics (deployment scoring): {release_summary}")

            # -------------------------------- #282 screening-threshold validation (cascade)
            # The two-pass cascade must lose ~zero recall vs MC-scoring EVERYTHING at the
            # science threshold — anything pass 1 drops never gets a second look
            mc_rng = derive_rng(root_seed, STREAM_RF, 4)
            test_mean_flat = val_mean_flat[test_idx]
            test_logvar_flat = val_logvar_flat[test_idx]
            draw_probas = np.zeros((self.config.inference.mc_draws, len(test_idx)))
            for draw_index in range(self.config.inference.mc_draws):
                draw_flat = sample_z_flat(test_mean_flat, test_logvar_flat, mc_rng)
                draw_features = build_variant_features(
                    winner, draw_flat, test_logvar_flat, num_observations, latent_dim, active_dims
                )
                draw_probas[draw_index] = apply_probability_calibrator(
                    calibrator, self.rf_model.model.predict_proba(draw_features)[:, 1]
                )
            mc_mean = draw_probas.mean(axis=0)
            screen_stats = check_screening_threshold(
                test_labels=test_labels,
                pass1_probas=test_deployed,
                mc_mean_probas=mc_mean,
                screening_threshold=self.config.inference.screening_threshold,
                science_threshold=classification_threshold,
                recall_tolerance=self.config.rf.screen_recall_tolerance,
                tag=tag,
            )
            del draw_probas, mc_mean, test_mean_flat, test_logvar_flat

            # NOTE: come back to this later (what is this artifact for? is it handled properly by archiving functions on startup?)
            # Persist a single eval-artifact joblib that every RF plot function consumes.
            # val_probas stays the winner's RAW deterministic-representation probabilities
            # (rank plots are calibration-invariant); val_probas_deployed adds the calibrated
            # view; val_preds reflects the deployment operating point.
            artifacts = {
                "train_features": train_features,
                "train_binary_labels": train_binary_labels,
                "train_subtype_labels": train_subtype_labels,
                "val_features": val_features,
                "val_binary_labels": val_binary_labels,
                "val_subtype_labels": val_subtype_labels,
                "val_probas": val_probas,
                "val_probas_deployed": val_probas_deployed,
                "val_preds": val_preds,
                "classification_threshold": classification_threshold,
                "snr_base": snr_base,
                "snr_range": snr_range,
                "tag": tag,
                "latent_variant": winner,
                "active_dims": active_dims,
                "variant_metrics": variant_metrics,
                "release_metrics": release_metrics,
                "calibration": {
                    "active": calibrator is not None,
                    "method": calibrator["method"] if calibrator else None,
                    "measured_ece": measured_ece,
                    "max_ece": self.config.rf.max_ece,
                },
                "val_partition": {
                    "selection_idx": selection_idx,
                    "calibration_idx": calibration_idx,
                    "test_idx": test_idx,
                },
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
                rf_metrics["screening_threshold"] = float(self.config.inference.screening_threshold)
                rf_metrics["active_units"] = float(len(active_dims))
                rf_metrics.update({k: float(v) for k, v in release_metrics.items()})
                rf_metrics.update({k: float(v) for k, v in screen_stats.items()})
                for variant_name, metrics in variant_metrics.items():
                    for metric_name, value in metrics.items():
                        rf_metrics[f"variant_{variant_name}_{metric_name}"] = float(value)
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
            # train_binary_labels / train_*_flat are deliberately NOT del'd: the
            # _variant_train_matrix closure references them and ruff flags closure names
            # deleted by the enclosing scope (see the identical NOTE near plot_dual_axis);
            # they are freed when the frame exits
            del (
                artifacts,
                train_features,
                val_features,
                train_subtype_labels,
                val_subtype_labels,
                val_binary_labels,
                val_probas,
                val_probas_deployed,
                val_preds,
                variant_models,
                variant_val_probas,
                val_mean_flat,
                val_logvar_flat,
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
            # Restore the sweep outcome onto the config (#282): the resumed attempt skips
            # train_random_forest's selection, but final_save still dumps the config, which
            # must record the variant/calibration the persisted model was actually trained
            # with — otherwise inference would rebuild the wrong features
            artifacts = joblib.load(artifact_path)
            self.config.rf.latent_variant = artifacts.get("latent_variant", "z")
            self.config.rf.active_dims = artifacts.get("active_dims")
            calibration = artifacts.get("calibration", {})
            self.config.rf.calibration_active = bool(calibration.get("active", False))
            self.config.rf.calibration_method = calibration.get("method")
            logger.info(
                f"Restored sweep outcome from artifacts: latent_variant="
                f"'{self.config.rf.latent_variant}', calibration_active="
                f"{self.config.rf.calibration_active}"
            )
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

        rng = derive_rng(self.config.reproducibility.seed, STREAM_SHAP_SAMPLES)
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
    def _training_plots_dir(self, subdir: str | None = None) -> str:
        """This run's training-plots base: ``{output_path}/plots/training/{save_tag}[/subdir]``."""
        base = os.path.join(
            self.config.output_path, "plots", "training", self.config.checkpoint.save_tag
        )
        return os.path.join(base, subdir) if subdir else base

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
            # #277: never render a partial result set as if it were complete. The rows are
            # still queued (nothing is lost) and the figure is regenerable once the writer
            # catches up; callers treat plot failures as non-critical, so raising skips the
            # figure without failing the run.
            raise RuntimeError(
                "Database flush timed out with training stats still queued - skipping the "
                "beta-VAE loss curves plot rather than rendering an incomplete result set"
            )
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

        # Arrange rows on the real global-epoch axis — missing epochs render as gaps (#277)
        epochs, history = build_epoch_history(all_stats, self.config.training.epochs_per_round)

        del all_stats
        gc.collect()

        # Add SNR range background shading to all axes
        snr_by_round = self._get_snr_by_round(current_time)
        epochs_per_round = self.config.training.epochs_per_round

        # Scale figure width for many rounds
        num_rounds = len(snr_by_round)
        base_width = 25
        fig_width = base_width * (1 + max(0, num_rounds - 10) * 0.05)  # +5% width per round over 10

        # Create figure & setup gridspec
        fig = plt.figure(figsize=(fig_width, 12))
        gs = fig.add_gridspec(2, 5, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

        # Top subplot spanning full width - Total Loss
        ax_top = fig.add_subplot(gs[0, :])

        # Bottom subplots - Individual losses
        ax_recon = fig.add_subplot(gs[1, 0])
        ax_kl = fig.add_subplot(gs[1, 1])
        ax_true = fig.add_subplot(gs[1, 2])
        ax_false = fig.add_subplot(gs[1, 3])
        ax_reg = fig.add_subplot(gs[1, 4])

        fig.suptitle(
            f"Beta-VAE Loss Curves ({tag}, {machine_name})", fontsize=18, fontweight="bold"
        )

        # Top subplot gets shading + text annotations, bottom subplots get shading only
        self._add_snr_range_shading(
            ax_top, snr_by_round, epochs_per_round, use_rounds=False, show_text_annotations=True
        )
        for ax in [ax_recon, ax_kl, ax_true, ax_false, ax_reg]:
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
        # L1/L2 regularization penalties (activated 2026-07; absent for runs predating it)
        plot_dual_axis(ax_reg, "Regularization", "reg_loss", "val_reg_loss")

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
        save_path = os.path.join(self._training_plots_dir(dir), f"beta_vae_loss_curves_{tag}.png")

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
            # #277: same skip-don't-render-partial contract as plot_beta_vae_loss_curves
            raise RuntimeError(
                "Database flush timed out with training stats still queued - skipping the "
                "training stability plot rather than rendering an incomplete result set"
            )
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

        # Arrange rows on the real global-epoch axis — missing epochs render as gaps (#277)
        epochs, history = build_epoch_history(all_stats, self.config.training.epochs_per_round)

        del all_stats
        gc.collect()

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
        save_path = os.path.join(
            self._training_plots_dir(dir), f"beta_vae_training_stability_{tag}.png"
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

    def plot_posterior_collapse(self, tag: str | None = None, dir: str | None = None):
        """
        Posterior-collapse diagnostics (#282): a KL-per-dimension heatmap (dim x global
        epoch — collapsing dims visibly go dark) plus the active-units count per epoch
        (dims with KL > training.posterior_collapse_kl_epsilon), from the kl_dim_* rows
        written each epoch. Follows the loss-curve plots' skip-don't-render-partial flush
        contract (#277).
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        if self.db is None:
            raise RuntimeError("No database instance detected - cannot plot posterior collapse")

        logger.info("Flushing database before plotting...")
        if not self.db.flush():
            raise RuntimeError(
                "Database flush timed out with training stats still queued - skipping the "
                "posterior-collapse plot rather than rendering an incomplete result set"
            )
        logger.info("Database flushed")

        latent_dim = self.config.beta_vae.latent_dim
        current_time = time.time()
        all_stats = self.db.query_training_stat(
            model_name="beta_vae",
            stat_name=[f"kl_dim_{d:02d}" for d in range(latent_dim)],
            start_round_number=1,
            tag=self.config.checkpoint.save_tag,
            start_time=self.start_time,
            end_time=current_time,
        )
        if not all_stats:
            logger.warning("No per-dimension KL rows to plot (pre-#282 run?) — skipping")
            return

        epochs, history = build_epoch_history(all_stats, self.config.training.epochs_per_round)
        n_epochs = len(epochs)
        kl_matrix = np.full((latent_dim, n_epochs), np.nan)
        for dim_idx in range(latent_dim):
            values = history.get(f"kl_dim_{dim_idx:02d}")
            if values is not None:
                kl_matrix[dim_idx] = values

        epsilon = self.config.training.posterior_collapse_kl_epsilon
        active_counts = np.nansum(kl_matrix > epsilon, axis=0)

        fig, (ax_heat, ax_active) = plt.subplots(
            2, 1, figsize=(16, 9), height_ratios=[3, 1], sharex=True
        )
        image = ax_heat.imshow(
            kl_matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            extent=(1, max(n_epochs, 2), latent_dim - 0.5, -0.5),
        )
        fig.colorbar(image, ax=ax_heat, label="mean KL per dim (nats)")
        ax_heat.set_ylabel("latent dimension")
        ax_heat.set_yticks(range(latent_dim))
        ax_heat.set_title(
            f"Per-dimension KL over training ({tag}) — dark rows = collapsing dims",
            fontsize=13,
            fontweight="bold",
        )

        ax_active.plot(list(epochs), active_counts, color="tab:red", linewidth=2)
        ax_active.axhline(
            self.config.training.min_active_units_fraction * latent_dim,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )
        ax_active.set_ylim(0, latent_dim + 0.5)
        ax_active.set_xlabel("Epoch (global)")
        ax_active.set_ylabel(f"active dims\n(KL > {epsilon:g})")
        ax_active.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self._training_plots_dir(dir), f"posterior_collapse_{tag}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Posterior-collapse diagnostics saved to: {save_path}")

        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                save_path, title=f"Posterior Collapse Diagnostics - ({tag})"
            )

    # TODO: reorder plot methods (def & call sites): train -> latent -> injection
    # TODO: move injection plots to data_generation.py & call at end of generate_round_to_memmap() (instead of at the end of train_round() & run_training_pipeline())
    # NOTE: there's a ton of improvements we could make to this function (and subsequent _plot functions), but i just care that it works well enough for now
    def plot_injection_stats(
        self, tag: str | None = None, dir: str | None = None, round_number: int | None = None
    ):
        """
        Generate 8 figures for bias/leakage analysis of the injection pipeline: 1 injected signal
        characteristics, 1 injection stability, 4 global intensity distributions (one per
        signal_type), 1 A->B global intensity biases, and 1 final global intensity biases.

        tag defaults to config.checkpoint.save_tag and is used in filenames; dir is an optional
        subdirectory under plots/ (e.g. "checkpoints" for per-round outputs). round_number
        scopes every query to that single round (the per-round call); None scopes to the full
        beta-VAE round range 1..num_training_rounds (the end-of-run call) — either way the
        pre-generated NEXT round's rows and the RF-phase sentinel round can no longer bleed
        into the figures (#277).
        """
        if tag is None:
            tag = self.config.checkpoint.save_tag

        save_dir = self._training_plots_dir(dir)
        os.makedirs(save_dir, exist_ok=True)

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        current_time = time.time()

        if self.db is None:
            raise RuntimeError(
                "No database instance detected - cannot generate injection stats plot"
            )

        # Round-scope every query below (#277): the background producer pre-generates round
        # N+1's rows during round N, so an unscoped query silently blends them in
        if round_number is not None:
            start_round_number, end_round_number = round_number, round_number
        else:
            start_round_number = 1
            end_round_number = self.config.training.num_training_rounds

        # Injection rows ride the DB's bulk lane, which flush() deliberately does not cover.
        # Gate on the bulk backlog instead: if any row for the rounds being plotted is still
        # queued, the figures would be incomplete — skip (non-critical) rather than render a
        # partial result set (#277). Rows for LATER rounds (the producer's pre-generation
        # flood) don't block, since the round scoping excludes them from the queries anyway.
        backlog = self.db.injection_backlog_rows(max_round=end_round_number)
        if backlog:
            raise RuntimeError(
                f"{backlog} injection-stat row(s) for rounds <= {end_round_number} are still "
                "queued in the DB bulk lane - skipping injection plots rather than rendering "
                "an incomplete result set"
            )

        # Flush the foreground lane too (covers direct write_injection_stat callers)
        logger.info("Flushing database before plotting...")
        if not self.db.flush():
            raise RuntimeError(
                "Database flush timed out with writes still queued - skipping injection "
                "plots rather than rendering an incomplete result set"
            )
        logger.info("Database flushed")

        # Tighten the timestamp window to the queried rounds' actual row span:
        # idx_injection_stats_filter leads with (tag, timestamp) and round_number is not in
        # it, so with the run-wide window every query below re-scans the tag's entire row
        # history (measured 10.5x slower at 12M rows) — and this function issues ~165 such
        # queries per call. The span is MIN/MAX over ALL rows for these rounds (superseded
        # included), so intersecting it with the existing [run-start, now] window can only
        # narrow each scan — never change a result set.
        window_start, window_end = self.start_time, current_time
        span = self.db.query_injection_stat_time_span(
            tag=self.config.checkpoint.save_tag,
            start_round_number=start_round_number,
            end_round_number=end_round_number,
        )
        if span is not None:
            window_start = max(window_start, span[0] - 1.0)
            window_end = min(window_end, span[1] + 1.0)

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
                start_round_number=start_round_number,
                end_round_number=end_round_number,
                tag=self.config.checkpoint.save_tag,
                start_time=window_start,
                end_time=window_end,
                columns=["value"],
            )
            eti_stats[stat_name] = [r["value"] for r in results]
            del results

            # Query RFI stats (from false_with_rfi and true_eti_rfi) in a single call
            results = self.db.query_injection_stat(
                stat_name=f"rfi_{stat_name}",
                signal_type=["false_with_rfi", "true_eti_rfi"],
                start_round_number=start_round_number,
                end_round_number=end_round_number,
                tag=self.config.checkpoint.save_tag,
                start_time=window_start,
                end_time=window_end,
                columns=["value"],
            )
            rfi_stats[stat_name] = [r["value"] for r in results]
            del results

        # Query background_index values for ETI and RFI signal types in single calls
        results = self.db.query_injection_stat(
            stat_name="global_mean",  # Any stat works here. Select "mean" to reduce rows queried
            injection_stage="A",  # Any stage works here. Select "A" to reduce rows queried
            signal_type=["true_only_eti", "true_eti_rfi"],
            start_round_number=start_round_number,
            end_round_number=end_round_number,
            tag=self.config.checkpoint.save_tag,
            start_time=window_start,
            end_time=window_end,
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
            start_round_number=start_round_number,
            end_round_number=end_round_number,
            tag=self.config.checkpoint.save_tag,
            start_time=window_start,
            end_time=window_end,
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
                start_round_number=start_round_number,
                end_round_number=end_round_number,
                tag=self.config.checkpoint.save_tag,
                start_time=window_start,
                end_time=window_end,
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
            start_round_number=start_round_number,
            end_round_number=end_round_number,
            tag=self.config.checkpoint.save_tag,
            start_time=window_start,
            end_time=window_end,
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
                        start_round_number=start_round_number,
                        end_round_number=end_round_number,
                        tag=self.config.checkpoint.save_tag,
                        start_time=window_start,
                        end_time=window_end,
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
                    start_round_number=start_round_number,
                    end_round_number=end_round_number,
                    tag=self.config.checkpoint.save_tag,
                    start_time=window_start,
                    end_time=window_end,
                    columns=["value"],
                )
                values_a = [r["value"] for r in results_a]
                del results_a

                # Query stage B values
                results_b = self.db.query_injection_stat(
                    stat_name=stat_name,
                    injection_stage="B",
                    signal_type=signal_type,
                    start_round_number=start_round_number,
                    end_round_number=end_round_number,
                    tag=self.config.checkpoint.save_tag,
                    start_time=window_start,
                    end_time=window_end,
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
                    start_round_number=start_round_number,
                    end_round_number=end_round_number,
                    tag=self.config.checkpoint.save_tag,
                    start_time=window_start,
                    end_time=window_end,
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
        rng = derive_rng(self.config.reproducibility.seed, STREAM_PLOT)

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

        save_dir = self._training_plots_dir(dir)

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

            # Sub-key 10 (obs) / 11 (cadence): distinct from the UMAP-fit sub-keys below
            rng = derive_rng(
                self.config.reproducibility.seed,
                STREAM_UMAP,
                10 if mode_label.startswith("obs") else 11,
            )
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
        # NOTE: instead of temp_dir, save frames in persistent dir. update dir archiving to handle
        temp_dir = tempfile.mkdtemp(prefix="latent_gif_")

        gif_paths = {}
        duration_ms = self.config.training.latent_viz_gif_duration_ms

        n_neighbors_values = self.config.training.latent_viz_umap_n_neighbors
        min_dist_values = self.config.training.latent_viz_umap_min_dist

        # Bundle the shared inputs once for the combo workers: every (mode, nn, md) combo
        # reads the same fit pools / snapshot coords / labels read-only, so they ship to the
        # forkserver workers through one on-disk joblib bundle instead of a per-task pickle.
        bundle_path = os.path.join(temp_dir, "sweep_inputs.joblib")
        joblib.dump(
            {
                "fit_pool_obs": fit_pool_obs,
                "fit_pool_cadence": fit_pool_cadence,
                "coords_obs": all_coords_obs,
                "coords_cadence": all_coords_cadence,
                "labels_obs": all_snapshot_labels_obs,
                "labels_cadence": all_snapshot_labels_cadence,
                "onoff_obs": all_snapshot_onoff_obs,
                "snapshot_metadata": snapshot_metadata,
            },
            bundle_path,
        )

        # One task per (nn, md, level): the whole combo pipeline (UMAP fit + persist +
        # per-snapshot transforms + frame render + GIF assembly) runs in a forkserver worker
        # (aetherscan.latent_gif.run_umap_gif_sweep). The sweep was strictly serial here at
        # ~95% single-core (~1.7-1.9 h per run) even after #278 parallelized frame
        # rendering; combos are independent fits with their own derived random_state, so
        # process isolation preserves every output byte — unlike the rejected within-fit
        # ideas (batched transforms / precomputed-knn reuse, see latent_gif.py).
        sweep_tasks = []
        for nn in n_neighbors_values:
            for md in min_dist_values:
                for level_idx, mode in ((0, "obs"), (1, "cadence")):
                    method_name = f"{mode}_umap_nn{nn}_md{md}"
                    level_display = "Obs-level" if mode == "obs" else "Cadence-level"
                    sweep_tasks.append(
                        {
                            "mode": mode,
                            "nn": nn,
                            "md": md,
                            # random_state derives from the root seed, sub-keyed (level, nn,
                            # md) so every fit gets its own reproducible stream (#279; md
                            # keyed at 1e-3 resolution) — the same derivation the serial
                            # loop used
                            "seed": derive_seed(
                                self.config.reproducibility.seed,
                                STREAM_UMAP,
                                level_idx,
                                nn,
                                round(md * 1000),
                            ),
                            "method_name": method_name,
                            "display_method": (
                                f"{level_display} UMAP (n_neighbors: {nn}, min_dist: {md})"
                            ),
                            "bundle_path": bundle_path,
                            "frames_dir": temp_dir,
                            "gif_path": os.path.join(
                                save_dir, f"latent_space_{method_name}_{tag}.gif"
                            ),
                            "umap_path": os.path.join(
                                self.config.model_path,
                                f"umap_{mode}_nn{nn}_md{md}_{tag}.joblib",
                            ),
                            "duration_ms": duration_ms,
                        }
                    )

        # The parent's copies of the bundled inputs are dead weight during the sweep
        del fit_pool_obs, fit_pool_cadence
        del all_coords_obs, all_coords_cadence
        del all_snapshot_labels_obs, all_snapshot_labels_cadence, all_snapshot_onoff_obs
        gc.collect()

        n_sweep_workers = min(len(sweep_tasks), self.config.manager.n_processes)
        logger.info(
            f"Running {len(sweep_tasks)} UMAP GIF combos across {n_sweep_workers} worker(s)"
        )
        sweep_results = run_umap_gif_sweep(sweep_tasks, n_workers=n_sweep_workers)

        # Logging, bookkeeping and Slack uploads stay parental (forkserver workers have no
        # log handler and no Slack client), in task order — the serial loop's output order
        for result in sweep_results:
            for warning in result["warnings"]:
                logger.warning(warning)
            if os.path.exists(result["umap_path"]):
                logger.info(f"Saved UMAP model: {result['umap_path']}")
            gif_paths[result["method_name"]] = result["gif_path"]
            if result["n_frames"] > 0:
                logger.info(
                    f"Latent space {result['method_name'].upper()} GIF saved: "
                    f"{result['gif_path']} ({result['n_frames']} frames)"
                )
                logger_instance = get_logger()
                if logger_instance:
                    logger_instance.upload_image_to_slack(
                        result["gif_path"],
                        title=f"Latent Space {result['display_method']} - ({tag})",
                    )

        # Cleanup (frame PNGs + the sweep input bundle)
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
        save_path = os.path.join(self._training_plots_dir(dir), filename)
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

        save_path = os.path.join(self._training_plots_dir(dir), f"rf_confusion_matrices_{tag}.png")
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

        save_path = os.path.join(
            self._training_plots_dir(dir), f"rf_classification_curves_{tag}.png"
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

        save_path = os.path.join(self._training_plots_dir(dir), f"rf_shap_summary_{tag}.png")
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

        save_path = os.path.join(self._training_plots_dir(dir), f"rf_shap_dependence_{tag}.png")
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

        save_path = os.path.join(self._training_plots_dir(dir), f"rf_shap_interactions_{tag}.png")
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

        save_path = os.path.join(
            self._training_plots_dir(dir), f"rf_shap_loss_monitoring_{tag}.png"
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
            umap_model = umap.UMAP(
                n_components=2,
                random_state=derive_seed(self.config.reproducibility.seed, STREAM_UMAP, 2),
                n_neighbors=15,
                min_dist=0.1,
            ).fit(shap_values)
            embedding = umap_model.transform(shap_values)
            # NOTE: come back to this later (are these values of n_clusters & n_init appropriate always?)
            kmeans = KMeans(
                n_clusters=4,
                random_state=derive_seed(self.config.reproducibility.seed, STREAM_KMEANS),
                n_init=10,
            )
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

        save_path = os.path.join(
            self._training_plots_dir(dir), f"rf_shap_explanation_clustering_{tag}.png"
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

        save_path = os.path.join(self._training_plots_dir(dir), f"rf_calibration_curve_{tag}.png")
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

        rng = derive_rng(self.config.reproducibility.seed, STREAM_RF_PLOTS, 0)
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

        save_path = os.path.join(self._training_plots_dir(dir), f"rf_oob_accuracy_curve_{tag}.png")
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
            rng = derive_rng(self.config.reproducibility.seed, STREAM_RF_PLOTS, 1)
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
                    # The persisted cadence UMAP was fit on the num_obs*latent_dim z_mean
                    # cadence space; wide #282 variants prepend that exact block, so
                    # project the lead columns only
                    lead_width = self.config.data.num_observations * self.config.beta_vae.latent_dim
                    embedding = umap_model.transform(pts_features[:, :lead_width])
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

                # #282: the cadence UMAP lives in the 48-dim z_mean space, but the winning
                # variant may carry extra uncertainty features the inverse transform cannot
                # reconstruct — hold those at their training-set means (a documented
                # approximation: the boundary is then the slice through typical uncertainty)
                n_rf_features = self.rf_model.model.n_features_in_
                if n_rf_features > grid_48d.shape[1]:
                    extras_mean = artifacts["train_features"][:, grid_48d.shape[1] :].mean(axis=0)
                    grid_features = np.hstack(
                        [grid_48d, np.tile(extras_mean, (grid_48d.shape[0], 1))]
                    )
                else:
                    grid_features = grid_48d
                grid_probas = self.rf_model.model.predict_proba(grid_features)[:, 1]
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
                save_path = os.path.join(self._training_plots_dir(dir), filename)
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
        rng = derive_rng(self.config.reproducibility.seed, STREAM_VIZ, 0)

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

        # Fancy indexing already creates a new independent array (no .copy() needed); the
        # astype is a no-op for float32 rounds and upcasts float16 rounds so traversal math
        # and plain-model encodes always see float32 (mirrors map_gather's cast)
        all_indices = np.concatenate(selected_indices)
        self._latent_viz_batch = concat_data[all_indices].astype(np.float32, copy=False)
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
            rng=derive_rng(self.config.reproducibility.seed, STREAM_VIZ, 1),
        )

        self._latent_viz_dataset = viz_results["viz_dataset"]
        self._latent_viz_n_padded = viz_results["n_padded"]
        self._latent_viz_n_samples = viz_results["n_samples"]
        self._latent_viz_steps = viz_results["viz_steps"]
        self._latent_viz_holder = viz_results["_viz_holder"]
        # ONE iterator for the round, shared by every snapshot: iter() on a distributed
        # dataset costs ~0.9 s on a 5-GPU strategy, and each capture consumes exactly one
        # in-order pass of the infinite dataset, so a persistent iterator reproduces the
        # per-capture encode order byte-for-byte (fresh-eyes audit: the fresh per-capture
        # iterator was ~60% of the snapshot's measured cost)
        self._latent_viz_iterator = iter(self._latent_viz_dataset)
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
        # Encode all cadences using distributed inference. The persistent round-scoped
        # iterator (each capture consumes exactly one in-order pass of the infinite viz
        # dataset) skips the ~0.9 s iter() + ~0.3 s gc that a fresh per-capture iterator
        # measured — this runs INSIDE the epoch loop, up to 6x per epoch at full scale
        [all_z_mean] = self._distributed_encode(
            dataset=self._latent_viz_dataset,
            n_steps=self._latent_viz_steps,
            encode_fn=self._viz_encode_fn,
            n_samples=n_padded * num_obs,
            latent_dim=latent_dim,
            logging=False,
            iterator=self._latent_viz_iterator,
        )

        # Truncate padding and reshape to per-cadence: (n_samples, 6, latent_dim)
        all_z_mean = all_z_mean[: n_samples * num_obs]
        z_mean_per_cadence = all_z_mean.reshape(n_samples, num_obs, latent_dim)
        del all_z_mean

        # Write to DB in one batched call: one metadata lookup and one array-wide round
        # instead of per-cadence Python (NOTE: 8 decimal precision for stored latents)
        rounded = np.round(z_mean_per_cadence, 8)
        self.db.write_latent_snapshots_bulk(
            model_name="beta_vae",
            round_number=round_idx + 1,
            epoch_number=epoch + 1,
            step_number=step + 1,
            snr_base=snr_base,
            snr_range=snr_range,
            tag=self.config.checkpoint.save_tag,
            timestamp=time.time(),
            snapshots=[
                (cadence_idx, str(self._latent_viz_labels[cadence_idx]), rounded[cadence_idx])
                for cadence_idx in range(n_samples)
            ],
        )

        del z_mean_per_cadence, rounded

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
        the tag=None default loads only the conventional 'final' pair, else fails loudly —
        there is no scan for "the latest tag present" (see _resolve_load_tag() and issue #142).
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
                (self.plot_posterior_collapse, "plot_posterior_collapse"),
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
