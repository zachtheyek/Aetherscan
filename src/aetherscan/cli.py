"""
CLI argument parsing for Aetherscan Pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime
from math import gcd, lcm
from typing import Any

from aetherscan.config import get_config

logger = logging.getLogger(__name__)


# Validation primitives are expressed as global variables and shared by
# validate_args() and utility scripts (e.g. utils/find_optimal_configs.py)

# Tag scheme (see resolve_save_tag): a run's save-tag is `{command}_{YYYYMMDD_HHMMSS}`. The user
# supplies only the command prefix on --save-tag; the datetime is appended at runtime. --load-tag
# takes a full resolved tag (resume that run in place) or a per-round checkpoint (round_XX).
_SAVE_TAG_PREFIXES = ("test", "train", "inf", "bench")
# --save-tag accepts one of the prefixes above (bare); the datetime is auto-appended.
_SAVE_TAG_PATTERN = re.compile(r"^(?:test|train|inf|bench)$")
# A fully-resolved run tag `{prefix}_{YYYYMMDD_HHMMSS}` — used to detect a resume-in-place --load-tag.
_FULL_TAG_PATTERN = re.compile(r"^(?:test|train|inf|bench)_\d{8}_\d{6}$")
# --load-tag accepts a full resolved tag (resume that run in place) or a per-round checkpoint.
_LOAD_TAG_PATTERN = re.compile(r"^(?:(?:test|train|inf|bench)_\d{8}_\d{6}|round_\d+)$")

# The round_XX subset of _LOAD_TAG_PATTERN: per-round checkpoints are saved under the checkpoints/
# subdirectory, so loading one requires --load-dir checkpoints (issue #142)
_ROUND_TAG_PATTERN = re.compile(r"^round_\d+$")

# Default --save-tag prefix per subcommand when --save-tag is omitted.
_COMMAND_TAG_PREFIX = {"train": "train", "inference": "inf"}


def resolve_save_tag(command: str | None, save_tag: str | None, load_tag: str | None) -> str:
    """Resolve the run's full `{prefix}_{YYYYMMDD_HHMMSS}` save-tag at runtime.

    - A full `{prefix}_{datetime}` ``--load-tag`` resumes that run **in place**: its tag is
      adopted so the resumed attempt writes under the same ``run_state`` / DB rows / artifacts.
      (A ``round_XX`` ``--load-tag`` does NOT adopt — it seeds a fresh run from that checkpoint.)
    - Otherwise the tag is ``{prefix}_{datetime}``, where the prefix is ``--save-tag``
      (``test|train|inf|bench``) or, when omitted, the subcommand default (``train`` → ``train``,
      ``inference`` → ``inf``). The datetime is stamped once, here, at run time.
    """
    if load_tag is not None and _FULL_TAG_PATTERN.match(load_tag):
        return load_tag
    if save_tag is not None:
        prefix = save_tag
    elif command in _COMMAND_TAG_PREFIX:
        prefix = _COMMAND_TAG_PREFIX[command]
    else:
        # Fail loudly rather than emit an unloadable tag (a prefix outside {test,train,inf,bench}
        # would not match _LOAD_TAG_PATTERN, so its artifacts could never be --load-tag'd).
        raise ValueError(
            f"Cannot derive a save-tag prefix for command {command!r}; pass --save-tag explicitly."
        )
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# sklearn's RandomForestClassifier accepts these string values for max_features
_RF_MAX_FEATURES_STR_VALUES = {"sqrt", "log2"}

# Allowed values for curriculum_schedule
_CURRICULUM_SCHEDULES = {"linear", "exponential", "step"}

# Allowed values for bandpass_method (energy-detection bandpass flattening)
_BANDPASS_METHODS = {"pfb", "spline"}

# HuggingFace Hub repo ids are "namespace/name" (user or org namespace)
_HF_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")


@dataclass
class ValidationError:
    """A single validation failure with enough structure for utility scripts to propose fixes."""

    field: str  # e.g. "training.effective_batch_size"
    current: Any
    message: str
    fix_kind: str  # one of: clamp_low, clamp_high, range, enum, divisibility, file_exists, cross_param, format
    min_val: Any = None
    max_val: Any = None
    allowed: Any = None
    divisor: Any = None
    extra: dict = field(default_factory=dict)


def _resolve(args: argparse.Namespace, arg_name: str, default: Any) -> Any:
    """Return `args.<arg_name>` if present and not None, otherwise `default` (typically the config value at startup time)."""
    val = getattr(args, arg_name, None)
    return val if val is not None else default


def _estimate_round_data_nbytes(
    n_samples: int, num_observations: int, time_bins: int, width_bin_downsampled: int
) -> int:
    """
    Estimate the on-disk size of one round's disk-backed dataset (see round_data.py): three
    float32 arrays of shape (n_samples, num_observations, time_bins, width_bin_downsampled)
    plus a tiny U20 labels array. Kept stdlib-only (no numpy) so utils/print_cli_help.py can
    keep importing cli.py without the scientific stack.
    """
    per_sample_bytes = num_observations * time_bins * width_bin_downsampled * 4  # float32
    labels_bytes = n_samples * 20 * 4  # numpy "U20" = 20 UCS-4 code points per label
    return 3 * n_samples * per_sample_bytes + labels_bytes


def _nearest_existing_ancestor(path: str) -> str:
    """Walk up from `path` to the closest directory that exists (for shutil.disk_usage on a
    round-data dir that hasn't been created yet)."""
    probe = os.path.abspath(path)
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return probe


def _resolve_num_replicas(args: argparse.Namespace) -> int | None:
    """Resolve the replica count to validate cross-replica constraints against, and
    fail fast if an explicitly-requested count is unusable on this host.

    Returns the count `collect_validation_errors` should check batch/sample divisibility
    against, or None to skip those checks.

    Resolution:
    - If the user passed --num-replicas, or `config.gpu.num_replicas` is set on the
      singleton, e.g. loaded from a saved JSON config via `apply_saved_config` (flag has
      priority), that value must be >= 1 and <= the number of GPUs TF reports; either
      violation raises ValueError so a bad count hard-stops *before* the divisibility
      checks consume it as a divisor. Otherwise it is returned, so those checks run
      against the same count the strategy will actually use.
    - With no explicit request, the returned count is the number of GPUs TF reports.
    - If TF is unavailable (e.g. a dev box) or reports zero GPUs, None is returned
      regardless of whether a count was requested: nothing can be confirmed against the
      hardware, so the divisibility checks are skipped and handling is deferred to
      `setup_gpu_strategy` (which determines whether to fail on GPU-required runs, or to
      proceed with a CPU-only runtime). Note, the >= 1 check still runs first, so a
      nonsensical explicit request is rejected even here.

    TF is imported lazily so utility scripts (which call `collect_validation_errors`
    directly and never reach this) don't pay the init cost.
    """
    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")

    # Resolve the requested count and remember which source supplied it, so any failure
    # below names the right one (the flag vs. the saved/programmatic config value).
    if getattr(args, "num_replicas", None) is not None:
        requested: int | None = int(args.num_replicas)
        source = "--num-replicas"
    elif config.gpu.num_replicas is not None:
        requested = int(config.gpu.num_replicas)
        source = "config.gpu.num_replicas"
    else:
        requested = None
        source = None

    # Positivity is checked first and without TF, so a nonsensical explicit request fails
    # fast even on a no-GPU box.
    if requested is not None and requested < 1:
        raise ValueError(
            f"{source} must be >= 1 (or unset to use all available GPUs), got {requested}"
        )

    # Everything below needs the host's GPU count. If TF is unavailable or reports zero
    # GPUs we can neither detect nor verify a count, so skip the cross-replica checks and
    # defer to setup_gpu_strategy rather than guessing.
    try:
        import tensorflow as tf  # noqa: PLC0415

        gpus = tf.config.list_physical_devices("GPU")
    except Exception:
        return None
    if not gpus:
        return None
    total = len(gpus)

    # No explicit request: use every GPU TF reports.
    if requested is None:
        return total

    # Explicit request: never allow more replicas than the host has GPUs — propagating
    # batch/sample sizes validated against the wrong replica count would silently corrupt
    # the run.
    if requested > total:
        raise ValueError(
            f"{source}={requested} exceeds the number of GPUs available on this node "
            f"({total}). Set it to <= {total}, or unset it to use all available GPUs."
        )
    return requested


def setup_argument_parser() -> argparse.ArgumentParser:
    """Build the top-level Aetherscan argparse parser with `train` and `inference` subcommands."""
    parser = argparse.ArgumentParser(
        description="Aetherscan Pipeline -- Breakthrough Listen's first end-to-end production-grade DL pipeline for SETI @ scale"
    )

    # Add commands. required=True: a subcommand is mandatory. Without it a bare
    # `python -m aetherscan.main` leaves command=None, and main()'s resolve_save_tag(None, ...)
    # would raise a ValueError traceback before the logger exists; argparse instead emits a clean
    # usage error (caught by main()'s SystemExit handler, which prints full help).
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")

    # Train command
    _add_train_arguments(subparsers)
    # Inference command
    _add_inference_arguments(subparsers)

    return parser


def _add_train_arguments(subparsers):
    """Register the `train` subcommand and populate it with all training-mode flags."""
    train_parser = subparsers.add_parser("train", help="Execute training pipeline")
    _add_train_flags_to(train_parser)


# TODO: improve flag help descriptions
def _add_reproducibility_flags_to(parser):
    """
    Add the shared reproducibility flags (#279) — registered on BOTH subparsers via one
    helper so the train and inference surfaces cannot drift.
    """
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Root random seed for reproducible runs: every random stream derives from it — data generation, dataset split/shuffles, TF weight init, the VAE sampling layer (training AND inference), the random forest, UMAP/KMeans plot fits, and plot subsampling. Defaults to a concrete value (reproducible out of the box); must be >= 0.",
    )
    parser.add_argument(
        "--tf-deterministic-ops",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force deterministic TensorFlow/cuDNN op implementations (tf.config.experimental.enable_op_determinism) for bit-exact GPU reproducibility at some speed cost. Only meaningful together with a seed (default: disabled)",
    )


def _add_train_flags_to(parser):
    """
    Add all training-mode CLI flags to `parser`. Defined separately from the subparser wrapper
    so that utility scripts (e.g. utils/find_optimal_configs.py) can expose the same flag
    surface without re-declaring every argument.
    """
    # Shared reproducibility flags (#279) — one helper, both subparsers
    _add_reproducibility_flags_to(parser)

    # Path arguments (overrides environment variables)
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data directory (overrides AETHERSCAN_DATA_PATH environment variable)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model directory (overrides AETHERSCAN_MODEL_PATH environment variable)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to output directory (overrides AETHERSCAN_OUTPUT_PATH environment variable)",
    )
    parser.add_argument(
        "--dashboard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Auto-launch the live monitoring Streamlit dashboard for this run; SSH-forward the port to view it (default: on). Use --no-dashboard to disable",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=None,
        help="Port for the auto-launched live dashboard (default: 8501)",
    )
    parser.add_argument(
        "--benchmark-report",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render the end-of-run benchmark report (stage timeline + bottleneck suggestions) and post it to Slack (default: on). Use --no-benchmark-report to disable",
    )

    # BetaVAE model configuration
    parser.add_argument(
        "--vae-latent-dim",
        type=int,
        default=None,
        help="Dimensionality of the VAE latent space (bottleneck size)",
    )
    parser.add_argument(
        "--vae-dense-layer-size",
        type=int,
        default=None,
        help="Size of dense layer in VAE architecture (should match frequency bins after downsampling)",
    )
    parser.add_argument(
        "--vae-kernel-size",
        type=int,
        nargs=2,
        default=None,
        help="Kernel size for Conv2D layers as two integers (e.g., --vae-kernel-size 3 3)",
    )
    parser.add_argument(
        "--vae-beta",
        type=float,
        default=None,
        help="Beta coefficient for KL divergence loss term in beta-VAE (controls disentanglement)",
    )
    parser.add_argument(
        "--vae-alpha",
        type=float,
        default=None,
        help="Alpha coefficient for clustering loss term in VAE (controls cluster separation)",
    )

    # Random Forest configuration
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=None,
        help="Number of decision trees in the random forest ensemble",
    )
    parser.add_argument(
        "--rf-bootstrap",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Whether to use bootstrap sampling when building trees (enables bagging)",
    )
    parser.add_argument(
        "--rf-max-features",
        type=str,
        default=None,
        help="Number of features to consider for splits: 'sqrt', 'log2', or a float (fraction of features)",
    )
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for random forest training (-1 uses all CPU cores)",
    )
    parser.add_argument(
        "--rf-seed",
        type=int,
        default=None,
        help="DEPRECATED: explicit random forest seed override. The RF seed now derives from the root --seed (#279); this alias remains for existing scripts and logs a deprecation warning when used.",
    )

    # GPU configuration
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=None,
        help="Number of GPUs to use for the distributed strategy. If omitted, the strategy uses every GPU visible to TF; otherwise it is restricted to the first N physical GPUs and the rest are left untouched. Must be >= 1 and <= the number of physical GPUs on your machine.",
    )
    parser.add_argument(
        "--gpu-memory-limit-mb",
        type=int,
        default=None,
        help="Per-GPU memory cap in MiB. Omit to use memory-growth-only (recommended on Blackwell). Set for TF to allocate a fixed logical device of a given size per physical GPU (e.g. 14000)",
    )
    parser.add_argument(
        "--nccl-num-packs",
        type=int,
        default=None,
        help="num_packs for NCCL/HierarchicalCopy all-reduce. Lower values (e.g. 1) reduces tiny-tensor latency; higher values (e.g. >=4) can help bandwidth on >4-GPU topologies.",
    )
    parser.add_argument(
        "--async-allocator",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle TF_GPU_ALLOCATOR=cuda_malloc_async (default: enabled). Pass --no-async-allocator as a workaround for NGC 25.02 multi-GPU OOM bugs.",
    )

    # Data configuration
    parser.add_argument(
        "--num-observations",
        type=int,
        default=None,
        help="Number of observations per cadence snippet (e.g., 6 for 3 ON + 3 OFF)",
    )
    parser.add_argument(
        "--width-bin",
        type=int,
        default=None,
        help="Number of frequency bins per observation (spectral resolution)",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=None,
        help="Downsampling factor for frequency bins (reduces spectral dimension)",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=None,
        help="Number of time bins per observation (temporal resolution)",
    )
    parser.add_argument(
        "--freq-resolution",
        type=float,
        default=None,
        help="Frequency resolution in Hz (determined by instrument)",
    )
    parser.add_argument(
        "--time-resolution",
        type=float,
        default=None,
        help="Time resolution in seconds (determined by instrument)",
    )
    parser.add_argument(
        "--num-target-backgrounds",
        type=int,
        default=None,
        help="Number of background (noise-only) cadences to load for training data generation",
    )
    parser.add_argument(
        "--background-load-chunk-size",
        type=int,
        default=None,
        help="Maximum number of background cadences to process at once during loading (memory management)",
    )
    parser.add_argument(
        "--max-chunks-per-file",
        type=int,
        default=None,
        help="Maximum number of chunks to load from a single data file (limits per-file contribution)",
    )
    parser.add_argument(
        "--train-files",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated list of training data file names (e.g., real_filtered_LARGE_HIP110750.npy)",
    )

    # Training configuration
    parser.add_argument(
        "--num-training-rounds",
        type=int,
        default=None,
        help="Total number of training rounds in curriculum learning schedule",
    )
    parser.add_argument(
        "--epochs-per-round",
        type=int,
        default=None,
        help="Number of epochs to train the VAE per curriculum learning round",
    )
    parser.add_argument(
        "--num-samples-beta-vae",
        type=int,
        default=None,
        # NOTE: divisible by 4 or num_replicas?
        help="Number of training samples to generate for beta-VAE per round (must be divisible by 4)",
    )
    parser.add_argument(
        "--num-samples-rf",
        type=int,
        default=None,
        # NOTE: divisible by 4 or num_replicas?
        help="Number of training samples to generate for random forest (must be divisible by 4)",
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        default=None,
        help="Fraction of data to use for training vs validation (e.g., 0.8 = 80%% train, 20%% val)",
    )
    parser.add_argument(
        "--per-replica-batch-size",
        type=int,
        default=None,
        help="Batch size per GPU/device replica during training",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=None,
        help="Effective batch size for gradient accumulation across all replicas",
    )
    parser.add_argument(
        "--per-replica-val-batch-size",
        type=int,
        default=None,
        help="Batch size per GPU/device replica during validation",
    )
    parser.add_argument(
        "--signal-injection-chunk-size",
        type=int,
        default=None,
        # NOTE: divisible by 4 or num_replicas?
        help="Maximum cadences to process at once during synthetic signal injection (must be divisible by 4)",
    )
    parser.add_argument(
        "--data-gen-task-size",
        type=int,
        default=None,
        help="Cadences per batched signal-injection worker task (workers write results straight into the round's on-disk memmap; must be >= 1)",
    )
    parser.add_argument(
        "--round-data-dir",
        type=str,
        default=None,
        help="Directory for disk-backed per-round training datasets (defaults to <data-path>/training/round_data; needs ~2.2x one round's size free when data-generation overlap is enabled, ~1.1x otherwise)",
    )
    parser.add_argument(
        "--overlap-data-generation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Generate round k+1's training data in a background producer process while round k trains (default: enabled). Pass --no-overlap-data-generation to fall back to sequential in-process generation for debugging",
    )
    parser.add_argument(
        "--keep-round-data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retain each round's on-disk training data after that round finishes (default: disabled — round k's data directory is deleted as soon as round k's training completes). Enable for debugging",
    )
    parser.add_argument(
        "--plot-injection-subsampling-count",
        type=int,
        default=None,
        help="Max points per stat name, per signal type, for A→B intensity bias scatter plots. Outliers are prioritized, with the difference made up from randomly sampling without replacement the remaining points",
    )
    parser.add_argument(
        "--plot-injection-outlier-percentile",
        type=float,
        default=None,
        help="Threshold for points to always be included in A→B intensity bias scatter plots",
    )
    parser.add_argument(
        "--latent-viz-num-cadences-per-type",
        type=int,
        default=None,
        help="Number of cadences per signal type for latent space visualization batch (total points = 4× this value × 6 observations per cadence)",
    )
    parser.add_argument(
        "--latent-viz-step-interval",
        type=int,
        default=None,
        help="Capture a latent space snapshot every N training steps (lower = more snapshots, more DB writes, and larger storage costs)",
    )
    parser.add_argument(
        "--latent-viz-umap-fit-max-samples",
        type=int,
        default=None,
        help="Maximum number of pooled latent vectors used to fit the UMAP model (remaining vectors are projected via transform; lower = faster, higher = more faithful embedding)",
    )
    parser.add_argument(
        "--latent-viz-umap-n-neighbors",
        type=int,
        nargs="+",
        default=None,
        help="UMAP n_neighbors values to sweep for latent space visualization (e.g., --latent-viz-umap-n-neighbors 5 15 30 50)",
    )
    parser.add_argument(
        "--latent-viz-umap-min-dist",
        type=float,
        nargs="+",
        default=None,
        help="UMAP min_dist values to sweep for latent space visualization (e.g., --latent-viz-umap-min-dist 0.0 0.1 0.5)",
    )
    parser.add_argument(
        "--latent-viz-gif-max-frames",
        type=int,
        default=None,
        help="Maximum number of frames in latent space GIF output (snapshots beyond this limit are log-subsampled, prioritizing earlier training steps)",
    )
    parser.add_argument(
        "--latent-viz-gif-duration-ms",
        type=int,
        default=None,
        help="Milliseconds per frame in latent space GIF output",
    )
    parser.add_argument(
        "--latent-traversal-every-round",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render latent-dimension traversal figures at the end of every training round, in addition to the end-of-training set (default: disabled)",
    )
    parser.add_argument(
        "--latent-traversal-num-steps",
        type=int,
        default=None,
        help="Number of traversal steps per latent dimension (must be odd and >= 3 so the center column is the unperturbed class-mean decode)",
    )
    parser.add_argument(
        "--latent-traversal-max-sigma",
        type=float,
        default=None,
        help="Latent traversal range in per-dimension standard deviations: steps span [-max_sigma, +max_sigma] (must be > 0)",
    )

    parser.add_argument(
        "--snr-base",
        type=int,
        default=None,
        help="Base signal-to-noise ratio for curriculum learning (minimum SNR difficulty level)",
    )
    parser.add_argument(
        "--initial-snr-range",
        type=int,
        default=None,
        help="SNR range for initial (easiest) training rounds (signals sampled from snr_base to snr_base + initial_snr_range)",
    )
    parser.add_argument(
        "--final-snr-range",
        type=int,
        default=None,
        help="SNR range for final (hardest) training rounds (signals sampled from snr_base to snr_base + final_snr_range). Ignored if only training for 1 round",
    )
    parser.add_argument(
        "--curriculum-schedule",
        type=str,
        default=None,
        help="Curriculum difficulty progression schedule: 'linear', 'exponential', or 'step'",
    )
    parser.add_argument(
        "--exponential-decay-rate",
        type=float,
        default=None,
        help="Decay rate for exponential curriculum schedule (must be negative; more negative = faster difficulty increase)",
    )
    parser.add_argument(
        "--step-easy-rounds",
        type=int,
        default=None,
        help="Number of rounds with easy signals when using step curriculum schedule",
    )
    parser.add_argument(
        "--step-hard-rounds",
        type=int,
        default=None,
        help="Number of rounds with hard signals when using step curriculum schedule",
    )
    parser.add_argument(
        "--base-learning-rate",
        type=float,
        default=None,
        help="Initial learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=None,
        help="Learning rate floor for adaptive learning rate reduction",
    )
    parser.add_argument(
        "--min-pct-improvement",
        type=float,
        default=None,
        help="Minimum fractional validation loss improvement to avoid LR reduction (e.g., 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--patience-threshold",
        type=int,
        default=None,
        help="Number of consecutive epochs without minimum improvement before reducing learning rate",
    )
    parser.add_argument(
        "--lr-reduction-factor",
        type=float,
        default=None,
        help="Multiplicative factor for learning rate reduction (e.g., 0.2 reduces LR by 20%%)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum number of retry attempts when training fails due to errors",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=None,
        help="Delay in seconds between retry attempts after training failure",
    )

    # HuggingFace Hub configuration
    parser.add_argument(
        "--hf-upload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Upload the final model artifacts (encoder, decoder, random forest, config) plus a generated model card to the HuggingFace Hub after training completes, tagging the commit with --save-tag (default: disabled = local-only). Requires HF_TOKEN in the environment (via .env)",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="HuggingFace model repo id (namespace/name) for weight upload/download (default: zachtheyek/aetherscan)",
    )

    # Checkpoint configuration
    parser.add_argument(
        "--load-dir",
        type=str,
        default=None,
        help="Subdirectory for checkpoint loading (relative to --model-path)",
    )
    parser.add_argument(
        "--load-tag",
        type=str,
        default=None,
        help="Checkpoint to load. A full run tag ({command}_YYYYMMDD_HHMMSS) resumes that run in place (its tag is adopted, so the resumed attempt writes under the same run). round_XX (requires --load-dir checkpoints) seeds a fresh run from that per-round checkpoint, resuming from round XX+1 unless --start-round is given.",
    )
    parser.add_argument(
        "--start-round",
        type=int,
        default=None,
        help="Round to begin/resume training from",
    )
    parser.add_argument(
        "--save-tag",
        type=str,
        default=None,
        help="Run label prefix: one of test, train, inf, bench. The datetime is appended automatically at runtime (e.g. train_20260101_120000). Defaults to the subcommand (train->train, inference->inf) if omitted.",
    )
    parser.add_argument(
        "--force-tag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the fail-early save-tag collision guard: proceed even when an explicitly-provided --save-tag matches existing artifacts, DB rows, or (with --hf-upload) an existing HuggingFace tag (default: disabled)",
    )


def _add_inference_arguments(subparsers):
    """Register the `inference` subcommand and populate it with all inference-mode flags."""
    inf_parser = subparsers.add_parser("inference", help="Execute inference pipeline")
    _add_inference_flags_to(inf_parser)


# TODO: improve flag help descriptions
def _add_inference_flags_to(parser):
    """
    Add all inference-mode CLI flags to `parser`. Defined separately from the subparser wrapper
    so that utility scripts (e.g. utils/find_optimal_configs.py) can expose the same flag
    surface without re-declaring every argument.
    """
    # Shared reproducibility flags (#279) — one helper, both subparsers
    _add_reproducibility_flags_to(parser)

    # Path arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data directory (overrides AETHERSCAN_DATA_PATH environment variable)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model directory (overrides AETHERSCAN_MODEL_PATH environment variable)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to output directory (overrides AETHERSCAN_OUTPUT_PATH environment variable)",
    )
    parser.add_argument(
        "--dashboard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Auto-launch the live monitoring Streamlit dashboard for this run; SSH-forward the port to view it (default: on). Use --no-dashboard to disable",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=None,
        help="Port for the auto-launched live dashboard (default: 8501)",
    )
    parser.add_argument(
        "--benchmark-report",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render the end-of-run benchmark report (stage timeline + bottleneck suggestions) and post it to Slack (default: on). Use --no-benchmark-report to disable",
    )

    # GPU configuration
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=None,
        help="Number of GPUs to use for the distributed strategy. If omitted, the strategy uses every GPU visible to TF; otherwise it is restricted to the first N physical GPUs and the rest are left untouched. Must be >= 1 and <= the number of physical GPUs on your machine.",
    )
    parser.add_argument(
        "--gpu-memory-limit-mb",
        type=int,
        default=None,
        help="Per-GPU memory cap in MiB. Omit to use memory-growth-only (recommended on Blackwell). Set for TF to allocate a fixed logical device of a given size per physical GPU (e.g. 14000)",
    )
    parser.add_argument(
        "--async-allocator",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle TF_GPU_ALLOCATOR=cuda_malloc_async (default: enabled). Pass --no-async-allocator as a workaround for NGC 25.02 multi-GPU OOM bugs.",
    )

    # Data configuration
    parser.add_argument(
        "--test-files",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated list of testing data file names (e.g., real_filtered_LARGE_test_HIP15638.npy)",
    )
    parser.add_argument(
        "--inference-files",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated list of inference catalog file names (e.g. complete_cadences_catalog.csv). Expects .h5 filepaths to individual observations, and sufficient metadata for recovering cadence groupings. If provided, triggers the energy detection preprocessing pipeline and takes precedence over --test-files",
    )

    # Inference configuration
    parser.add_argument(
        "--encoder-path",
        type=str,
        default=None,
        help="Path to trained VAE encoder model file (.keras). Optional: when none of --encoder-path/--rf-path/--config-path are given, the artifacts are downloaded from the HuggingFace Hub (see --hf-repo-id/--hf-revision); provide either all three local paths or none",
    )
    parser.add_argument(
        "--rf-path",
        type=str,
        default=None,
        help="Path to trained Random Forest model file (.joblib). Optional: see --encoder-path for the all-three-or-none rule",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config file from corresponding training run (.json). Optional: see --encoder-path for the all-three-or-none rule",
    )
    parser.add_argument(
        "--per-replica-batch-size",
        type=int,
        default=None,
        help="Batch size per GPU/device replica during inference",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=None,
        help="Classification threshold for candidate detection",
    )

    # Energy detection preprocessing
    parser.add_argument(
        "--cadence-group-by-cols",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated list of CSV column names whose joint value defines cadence membership (e.g., Target Session Band 'Cadence ID' Frequency)",
    )
    parser.add_argument(
        "--cadence-h5-path-col",
        type=str,
        default=None,
        help="CSV column containing the .h5 file path for each observation (default: '.h5 path')",
    )
    parser.add_argument(
        "--cadence-expected-obs",
        type=int,
        default=None,
        help="Required number of observations per cadence (default: 6 for ABACAD)",
    )
    parser.add_argument(
        "--coarse-channel-width",
        type=int,
        default=None,
        help="Number of fine channels per coarse channel (default: 1048576)",
    )
    parser.add_argument(
        "--coarse-channel-log-interval",
        type=int,
        default=None,
        help="Progress-logging chunk size for energy detection, in coarse channels per log line (default: the number of worker processes). Parallelism itself comes from the persistent worker pool, not this knob.",
    )
    parser.add_argument(
        "--bandpass-method",
        type=str,
        default=None,
        help="Bandpass flattening method for energy detection: 'pfb' (default) divides each coarse channel by the instrument's static polyphase-filterbank response; 'spline' fits and subtracts a per-channel spline",
    )
    parser.add_argument(
        "--pfb-taps-per-channel",
        type=int,
        default=None,
        help="PFB prototype-filter taps per coarse channel for --bandpass-method pfb (default: 12, the GBT/Breakthrough Listen backend value). INSTRUMENT-DEPENDENT: must match the backend that produced the .h5 files",
    )
    parser.add_argument(
        "--bandpass-debug-plot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save a per-cadence bandpass-flattening overlay debug plot (raw vs flattened integrated spectrum for a few sampled coarse channels) under plots/inference/{save-tag}/ (default: off)",
    )
    parser.add_argument(
        "--spline-order",
        type=int,
        default=None,
        help="Spline order for bandpass fitting with --bandpass-method spline (default: 16)",
    )
    parser.add_argument(
        "--detection-window-size",
        type=int,
        default=None,
        help="Sliding window size in fine channels for normality test (default: 256)",
    )
    parser.add_argument(
        "--detection-step-size",
        type=int,
        default=None,
        help="Step size in fine channels for sliding window (default: 128)",
    )
    parser.add_argument(
        "--stat-threshold",
        type=float,
        default=None,
        help="D'Agostino-Pearson statistic threshold for hit detection (default: 2048.0)",
    )
    parser.add_argument(
        "--stamp-width",
        type=int,
        default=None,
        help="Width in fine channels of the extracted stamp around each hit (default: 4096; must equal --width-bin)",
    )
    parser.add_argument(
        "--store-downsampled-stamps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Downsample stamps along frequency (by --downsample-factor) at extraction time, storing stamp_width // downsample_factor bins per stamp (~8x smaller at defaults; default: enabled). Pass --no-store-downsampled-stamps to archive raw-resolution stamps; loading handles both layouts.",
    )
    parser.add_argument(
        "--overlap-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Additionally extract stamps offset by ±overlap_fraction*stamp_width around each hit. Pass --no-overlap-search to disable when the config default is True.",
    )
    parser.add_argument(
        "--overlap-fraction",
        type=float,
        default=None,
        help="Fractional offset (relative to stamp_width) for overlap-search stamps (default: 0.5)",
    )
    parser.add_argument(
        "--preprocess-output-dir",
        type=str,
        default=None,
        help="Directory for per-cadence .npy outputs from preprocessing. Default: a per-CSV, tag-scoped directory {data_path}/inference/preprocessed/<csv_stem>_<save_tag>/ — retrying with the same tag resumes from existing .npy files, while a new tag starts clean. Pass an old run's directory explicitly to reuse its preprocessing (shared across CSVs)",
    )

    # Visualization suite
    parser.add_argument(
        "--inference-viz",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render the inference visualization suite (energy detection distributions, hit spectrum, bandpass overlay, stamp/candidate galleries, confidence distribution, latent projection, summary card) at the end of a CSV inference run, saved under plots/inference/{save_tag}/ and uploaded to Slack (default: enabled). Pass --no-inference-viz to disable.",
    )
    parser.add_argument(
        "--stamp-gallery-top-k",
        type=int,
        default=None,
        help="Number of top-statistic stamps shown in the stamp gallery figure, each as a 6-observation waterfall grid (default: 12)",
    )
    parser.add_argument(
        "--max-candidate-plots",
        type=int,
        default=None,
        help="Maximum number of per-candidate figures rendered per run, highest confidence first (default: 50; the candidate gallery is unaffected)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum number of retry attempts for inference (including preprocessing) on failure",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=None,
        help="Delay in seconds between inference retry attempts",
    )

    # HuggingFace Hub configuration
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="HuggingFace model repo id (namespace/name) for weight upload/download (default: zachtheyek/aetherscan)",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="HuggingFace revision (tag, branch, or commit hash) to pin the model download to when no local artifact paths are given (default: v{package version} when running as an installed release, else the repo's latest release tag — highest semver vX.Y.Z tag; a release tag is required for a no-artifact download)",
    )

    # Checkpoint configuration
    parser.add_argument(
        "--save-tag",
        type=str,
        default=None,
        help="Run label prefix: one of test, train, inf, bench. The datetime is appended automatically at runtime (e.g. inf_20260101_120000). Defaults to the subcommand (train->train, inference->inf) if omitted.",
    )
    parser.add_argument(
        "--force-tag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the fail-early save-tag collision guard: proceed even when an explicitly-provided --save-tag matches a previous run's saved config or DB rows (default: disabled)",
    )


# NOTE: come back to this later
def apply_saved_config(config_path: str) -> None:
    """Layer a saved JSON config (e.g. from a prior training run) onto the singleton.

    Used in inference mode to make `--config-path my_run.json` actually take effect:
    the saved file's values become the new "defaults" that `validate_args` sees via
    `_resolve`, and any CLI flags the user also passes will override them in the
    subsequent `apply_args_to_config` call.

    Priority order ends up as:

        defaults  <  saved config  <  CLI args

    For each top-level key in the JSON the corresponding sub-dataclass on the
    singleton is located by name (`data`, `inference`, `gpu`, ...) and the saved
    fields are written via setattr. Unknown keys/fields are skipped to stay
    forward-compat with newer/older saved configs. Top-level scalar entries
    (`data_path`, `model_path`, `output_path`) are applied directly.

    The `checkpoint` section is skipped entirely: a saved *training* config's
    checkpoint fields (most damagingly `save_tag`) are never what an inference run
    wants — layering them would make this run masquerade under the training run's
    tag, corrupting DB provenance and output paths. This run's resolved save_tag
    (set once in main() by resolve_save_tag) stays authoritative.

    Raises `ValueError` if the file is missing or malformed — caught by main.py's
    wrapper alongside `validate_args` failures.
    """
    if not os.path.exists(config_path):
        raise ValueError(f"--config-path: file does not exist on disk: {config_path}")
    try:
        with open(config_path) as f:
            saved = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--config-path: {config_path} is not valid JSON: {exc}") from exc

    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")

    for key, value in saved.items():
        if key == "checkpoint":
            # Never layer a saved training run's checkpoint section (save_tag/load_tag/
            # load_dir/start_round) under CLI flags — see docstring.
            continue
        target = getattr(config, key, None)
        if target is None:
            continue
        if is_dataclass(target) and isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if hasattr(target, sub_key):
                    setattr(target, sub_key, sub_val)
        else:
            # Top-level scalar (data_path, model_path, output_path, ...).
            setattr(config, key, value)


def apply_args_to_config(args: argparse.Namespace) -> None:
    """Mutate the singleton config in place with any non-None overrides from the parsed CLI
    namespace. Only attributes actually present on `args` are considered; missing ones fall back
    to the config defaults."""
    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")

    # Path overrides (must be done first as they affect file loading)
    if hasattr(args, "data_path") and args.data_path is not None:
        config.data_path = args.data_path
    if hasattr(args, "model_path") and args.model_path is not None:
        config.model_path = args.model_path
    if hasattr(args, "output_path") and args.output_path is not None:
        config.output_path = args.output_path
    if hasattr(args, "dashboard") and args.dashboard is not None:
        config.monitor.dashboard_enabled = args.dashboard
    if hasattr(args, "dashboard_port") and args.dashboard_port is not None:
        config.monitor.dashboard_port = args.dashboard_port
    if hasattr(args, "benchmark_report") and args.benchmark_report is not None:
        config.monitor.benchmark_report_enabled = args.benchmark_report

    # BetaVAE configuration
    if hasattr(args, "vae_latent_dim") and args.vae_latent_dim is not None:
        config.beta_vae.latent_dim = args.vae_latent_dim
    if hasattr(args, "vae_dense_layer_size") and args.vae_dense_layer_size is not None:
        config.beta_vae.dense_layer_size = args.vae_dense_layer_size
    if hasattr(args, "vae_kernel_size") and args.vae_kernel_size is not None:
        config.beta_vae.kernel_size = tuple(args.vae_kernel_size)
    if hasattr(args, "vae_beta") and args.vae_beta is not None:
        config.beta_vae.beta = args.vae_beta
    if hasattr(args, "vae_alpha") and args.vae_alpha is not None:
        config.beta_vae.alpha = args.vae_alpha

    # Random Forest configuration
    if hasattr(args, "rf_n_estimators") and args.rf_n_estimators is not None:
        config.rf.n_estimators = args.rf_n_estimators
    if hasattr(args, "rf_bootstrap") and args.rf_bootstrap is not None:
        config.rf.bootstrap = args.rf_bootstrap
    if hasattr(args, "rf_max_features") and args.rf_max_features is not None:
        config.rf.max_features = args.rf_max_features
    if hasattr(args, "rf_n_jobs") and args.rf_n_jobs is not None:
        config.rf.n_jobs = args.rf_n_jobs
    if hasattr(args, "rf_seed") and args.rf_seed is not None:
        # Deprecated alias (#279): the RF seed normally derives from the root --seed via
        # STREAM_RF; an explicit override still works but is discouraged
        logger.warning(
            "--rf-seed is DEPRECATED: the random forest seed now derives from the root "
            "--seed. The explicit override will be honored this run."
        )
        config.rf.seed = args.rf_seed

    # GPU configuration
    if hasattr(args, "num_replicas") and args.num_replicas is not None:
        config.gpu.num_replicas = args.num_replicas
    if hasattr(args, "gpu_memory_limit_mb") and args.gpu_memory_limit_mb is not None:
        config.gpu.per_gpu_memory_limit_mb = args.gpu_memory_limit_mb
    if hasattr(args, "nccl_num_packs") and args.nccl_num_packs is not None:
        config.gpu.nccl_num_packs = args.nccl_num_packs
    # async_allocator uses argparse.BooleanOptionalAction with default=None so that
    # the CLI can express "leave the config default" (omit), "force on"
    # (--async-allocator), and "force off" (--no-async-allocator). The `is not None`
    # guard preserves the config default when the user passes neither
    if hasattr(args, "async_allocator") and args.async_allocator is not None:
        config.gpu.use_async_allocator = args.async_allocator

    # Data configuration
    if hasattr(args, "num_observations") and args.num_observations is not None:
        config.data.num_observations = args.num_observations
    if hasattr(args, "width_bin") and args.width_bin is not None:
        config.data.width_bin = args.width_bin
    if hasattr(args, "downsample_factor") and args.downsample_factor is not None:
        config.data.downsample_factor = args.downsample_factor
    if hasattr(args, "time_bins") and args.time_bins is not None:
        config.data.time_bins = args.time_bins
    if hasattr(args, "freq_resolution") and args.freq_resolution is not None:
        config.data.freq_resolution = args.freq_resolution
    if hasattr(args, "time_resolution") and args.time_resolution is not None:
        config.data.time_resolution = args.time_resolution
    if hasattr(args, "num_target_backgrounds") and args.num_target_backgrounds is not None:
        config.data.num_target_backgrounds = args.num_target_backgrounds
    if hasattr(args, "background_load_chunk_size") and args.background_load_chunk_size is not None:
        config.data.background_load_chunk_size = args.background_load_chunk_size
    if hasattr(args, "max_chunks_per_file") and args.max_chunks_per_file is not None:
        config.data.max_chunks_per_file = args.max_chunks_per_file
    if hasattr(args, "train_files") and args.train_files is not None:
        config.data.train_files = args.train_files
    if hasattr(args, "test_files") and args.test_files is not None:
        config.data.test_files = args.test_files
    if hasattr(args, "inference_files") and args.inference_files is not None:
        config.data.inference_files = args.inference_files

    # Reproducibility (#279): shared by both subparsers
    if hasattr(args, "seed") and args.seed is not None:
        config.reproducibility.seed = args.seed
    # tf_deterministic_ops uses argparse.BooleanOptionalAction with default=None so the CLI
    # can express "leave the config default" (omit), "force on", and "force off" — same
    # pattern as async_allocator above
    if hasattr(args, "tf_deterministic_ops") and args.tf_deterministic_ops is not None:
        config.reproducibility.tf_deterministic_ops = args.tf_deterministic_ops

    # Training configuration
    if hasattr(args, "num_training_rounds") and args.num_training_rounds is not None:
        config.training.num_training_rounds = args.num_training_rounds
    if hasattr(args, "epochs_per_round") and args.epochs_per_round is not None:
        config.training.epochs_per_round = args.epochs_per_round
    if hasattr(args, "num_samples_beta_vae") and args.num_samples_beta_vae is not None:
        config.training.num_samples_beta_vae = args.num_samples_beta_vae
    if hasattr(args, "num_samples_rf") and args.num_samples_rf is not None:
        config.training.num_samples_rf = args.num_samples_rf
    if hasattr(args, "train_val_split") and args.train_val_split is not None:
        config.training.train_val_split = args.train_val_split
    if (
        hasattr(args, "per_replica_batch_size")
        and args.per_replica_batch_size is not None
        and getattr(args, "command", None) == "train"
    ):
        config.training.per_replica_batch_size = args.per_replica_batch_size
    if hasattr(args, "effective_batch_size") and args.effective_batch_size is not None:
        config.training.effective_batch_size = args.effective_batch_size
    if hasattr(args, "per_replica_val_batch_size") and args.per_replica_val_batch_size is not None:
        config.training.per_replica_val_batch_size = args.per_replica_val_batch_size
    if (
        hasattr(args, "signal_injection_chunk_size")
        and args.signal_injection_chunk_size is not None
    ):
        config.training.signal_injection_chunk_size = args.signal_injection_chunk_size
    if hasattr(args, "data_gen_task_size") and args.data_gen_task_size is not None:
        config.training.data_gen_task_size = args.data_gen_task_size
    if hasattr(args, "round_data_dir") and args.round_data_dir is not None:
        config.training.round_data_dir = args.round_data_dir
    # overlap_data_generation / keep_round_data use argparse.BooleanOptionalAction with
    # default=None so that the CLI can express "leave the config default" (omit), "force on"
    # (--overlap-data-generation), and "force off" (--no-overlap-data-generation). The
    # `is not None` guard preserves the config default when the user passes neither
    if hasattr(args, "overlap_data_generation") and args.overlap_data_generation is not None:
        config.training.overlap_data_generation = args.overlap_data_generation
    if hasattr(args, "keep_round_data") and args.keep_round_data is not None:
        config.training.keep_round_data = args.keep_round_data
    if (
        hasattr(args, "plot_injection_subsampling_count")
        and args.plot_injection_subsampling_count is not None
    ):
        config.training.plot_injection_subsampling_count = args.plot_injection_subsampling_count
    if (
        hasattr(args, "plot_injection_outlier_percentile")
        and args.plot_injection_outlier_percentile is not None
    ):
        config.training.plot_injection_outlier_percentile = args.plot_injection_outlier_percentile
    if (
        hasattr(args, "latent_viz_num_cadences_per_type")
        and args.latent_viz_num_cadences_per_type is not None
    ):
        config.training.latent_viz_num_cadences_per_type = args.latent_viz_num_cadences_per_type
    if hasattr(args, "latent_viz_step_interval") and args.latent_viz_step_interval is not None:
        config.training.latent_viz_step_interval = args.latent_viz_step_interval
    if (
        hasattr(args, "latent_viz_umap_fit_max_samples")
        and args.latent_viz_umap_fit_max_samples is not None
    ):
        config.training.latent_viz_umap_fit_max_samples = args.latent_viz_umap_fit_max_samples
    if (
        hasattr(args, "latent_viz_umap_n_neighbors")
        and args.latent_viz_umap_n_neighbors is not None
    ):
        config.training.latent_viz_umap_n_neighbors = args.latent_viz_umap_n_neighbors
    if hasattr(args, "latent_viz_umap_min_dist") and args.latent_viz_umap_min_dist is not None:
        config.training.latent_viz_umap_min_dist = args.latent_viz_umap_min_dist
    if hasattr(args, "latent_viz_gif_max_frames") and args.latent_viz_gif_max_frames is not None:
        config.training.latent_viz_gif_max_frames = args.latent_viz_gif_max_frames
    if hasattr(args, "latent_viz_gif_duration_ms") and args.latent_viz_gif_duration_ms is not None:
        config.training.latent_viz_gif_duration_ms = args.latent_viz_gif_duration_ms
    # latent_traversal_every_round uses argparse.BooleanOptionalAction with default=None
    # (same tri-state as overlap_data_generation / keep_round_data above)
    if (
        hasattr(args, "latent_traversal_every_round")
        and args.latent_traversal_every_round is not None
    ):
        config.training.latent_traversal_every_round = args.latent_traversal_every_round
    if hasattr(args, "latent_traversal_num_steps") and args.latent_traversal_num_steps is not None:
        config.training.latent_traversal_num_steps = args.latent_traversal_num_steps
    if hasattr(args, "latent_traversal_max_sigma") and args.latent_traversal_max_sigma is not None:
        config.training.latent_traversal_max_sigma = args.latent_traversal_max_sigma
    if hasattr(args, "snr_base") and args.snr_base is not None:
        config.training.snr_base = args.snr_base
    if hasattr(args, "initial_snr_range") and args.initial_snr_range is not None:
        config.training.initial_snr_range = args.initial_snr_range
    if hasattr(args, "final_snr_range") and args.final_snr_range is not None:
        config.training.final_snr_range = args.final_snr_range
    if hasattr(args, "curriculum_schedule") and args.curriculum_schedule is not None:
        config.training.curriculum_schedule = args.curriculum_schedule
    if hasattr(args, "exponential_decay_rate") and args.exponential_decay_rate is not None:
        config.training.exponential_decay_rate = args.exponential_decay_rate
    if hasattr(args, "step_easy_rounds") and args.step_easy_rounds is not None:
        config.training.step_easy_rounds = args.step_easy_rounds
    if hasattr(args, "step_hard_rounds") and args.step_hard_rounds is not None:
        config.training.step_hard_rounds = args.step_hard_rounds
    if hasattr(args, "base_learning_rate") and args.base_learning_rate is not None:
        config.training.base_learning_rate = args.base_learning_rate
    if hasattr(args, "min_learning_rate") and args.min_learning_rate is not None:
        config.training.min_learning_rate = args.min_learning_rate
    if hasattr(args, "min_pct_improvement") and args.min_pct_improvement is not None:
        config.training.min_pct_improvement = args.min_pct_improvement
    if hasattr(args, "patience_threshold") and args.patience_threshold is not None:
        config.training.patience_threshold = args.patience_threshold
    if hasattr(args, "lr_reduction_factor") and args.lr_reduction_factor is not None:
        config.training.reduction_factor = args.lr_reduction_factor
    if (
        hasattr(args, "max_retries")
        and args.max_retries is not None
        and getattr(args, "command", None) == "train"
    ):
        config.training.max_retries = args.max_retries
    if (
        hasattr(args, "retry_delay")
        and args.retry_delay is not None
        and getattr(args, "command", None) == "train"
    ):
        config.training.retry_delay = args.retry_delay

    # HuggingFace Hub configuration
    # hf_upload uses argparse.BooleanOptionalAction with default=None so that the CLI can
    # express "leave the config default" (omit), "force on" (--hf-upload), and "force off"
    # (--no-hf-upload). The `is not None` guard preserves the config default when the user
    # passes neither
    if hasattr(args, "hf_upload") and args.hf_upload is not None:
        config.hf.upload_after_training = args.hf_upload
    if hasattr(args, "hf_repo_id") and args.hf_repo_id is not None:
        config.hf.repo_id = args.hf_repo_id
    if hasattr(args, "hf_revision") and args.hf_revision is not None:
        config.hf.revision = args.hf_revision

    # Checkpoint configuration
    if hasattr(args, "load_dir") and args.load_dir is not None:
        config.checkpoint.load_dir = args.load_dir
    if hasattr(args, "load_tag") and args.load_tag is not None:
        config.checkpoint.load_tag = args.load_tag
        config.checkpoint.infer_start_round()  # Try inferring start_round from load_tag first
    if hasattr(args, "start_round") and args.start_round is not None:
        config.checkpoint.start_round = args.start_round  # Override start_round if provided
    # NOTE: config.checkpoint.save_tag is resolved once in main() (before init_logger, so the log
    # file / config / artifacts share one datetime) via resolve_save_tag — not applied from args here.
    # force_tag uses argparse.BooleanOptionalAction with default=None (same tri-state as
    # hf_upload above)
    if hasattr(args, "force_tag") and args.force_tag is not None:
        config.checkpoint.force_tag = args.force_tag

    # Inference configuration
    if hasattr(args, "encoder_path") and args.encoder_path is not None:
        config.inference.encoder_path = args.encoder_path
    if hasattr(args, "rf_path") and args.rf_path is not None:
        config.inference.rf_path = args.rf_path
    if hasattr(args, "config_path") and args.config_path is not None:
        config.inference.config_path = args.config_path
    if hasattr(args, "classification_threshold") and args.classification_threshold is not None:
        config.inference.classification_threshold = args.classification_threshold
    if (
        hasattr(args, "per_replica_batch_size")
        and args.per_replica_batch_size is not None
        and getattr(args, "command", None) == "inference"
    ):
        config.inference.per_replica_batch_size = args.per_replica_batch_size

    # Energy detection preprocessing
    if hasattr(args, "cadence_group_by_cols") and args.cadence_group_by_cols is not None:
        config.inference.cadence_group_by_cols = args.cadence_group_by_cols
    if hasattr(args, "cadence_h5_path_col") and args.cadence_h5_path_col is not None:
        config.inference.cadence_h5_path_col = args.cadence_h5_path_col
    if hasattr(args, "cadence_expected_obs") and args.cadence_expected_obs is not None:
        config.inference.cadence_expected_obs = args.cadence_expected_obs
    if hasattr(args, "coarse_channel_width") and args.coarse_channel_width is not None:
        config.inference.coarse_channel_width = args.coarse_channel_width
    if (
        hasattr(args, "coarse_channel_log_interval")
        and args.coarse_channel_log_interval is not None
    ):
        config.inference.coarse_channel_log_interval = args.coarse_channel_log_interval
    if hasattr(args, "bandpass_method") and args.bandpass_method is not None:
        config.inference.bandpass_method = args.bandpass_method
    if hasattr(args, "pfb_taps_per_channel") and args.pfb_taps_per_channel is not None:
        config.inference.pfb_taps_per_channel = args.pfb_taps_per_channel
    # bandpass_debug_plot uses argparse.BooleanOptionalAction with default=None so the CLI
    # can express "leave the config default" (omit), "force on" (--bandpass-debug-plot),
    # and "force off" (--no-bandpass-debug-plot).
    if hasattr(args, "bandpass_debug_plot") and args.bandpass_debug_plot is not None:
        config.inference.bandpass_debug_plot = args.bandpass_debug_plot
    if hasattr(args, "spline_order") and args.spline_order is not None:
        config.inference.spline_order = args.spline_order
    if hasattr(args, "detection_window_size") and args.detection_window_size is not None:
        config.inference.detection_window_size = args.detection_window_size
    if hasattr(args, "detection_step_size") and args.detection_step_size is not None:
        config.inference.detection_step_size = args.detection_step_size
    if hasattr(args, "stat_threshold") and args.stat_threshold is not None:
        config.inference.stat_threshold = args.stat_threshold
    if hasattr(args, "stamp_width") and args.stamp_width is not None:
        config.inference.stamp_width = args.stamp_width
    # store_downsampled_stamps uses argparse.BooleanOptionalAction with default=None so that
    # the CLI can express "leave the config default" (omit), "force on"
    # (--store-downsampled-stamps), and "force off" (--no-store-downsampled-stamps)
    if hasattr(args, "store_downsampled_stamps") and args.store_downsampled_stamps is not None:
        config.inference.store_downsampled_stamps = args.store_downsampled_stamps
    # overlap_search uses argparse.BooleanOptionalAction with default=None so that
    # the CLI can express "leave the config default" (omit), "force on"
    # (--overlap-search), and "force off" (--no-overlap-search). The `is not None`
    # guard preserves the config default when the user passes neither.
    if hasattr(args, "overlap_search") and args.overlap_search is not None:
        config.inference.overlap_search = args.overlap_search
    if hasattr(args, "overlap_fraction") and args.overlap_fraction is not None:
        config.inference.overlap_fraction = args.overlap_fraction
    if hasattr(args, "preprocess_output_dir") and args.preprocess_output_dir is not None:
        config.inference.preprocess_output_dir = args.preprocess_output_dir

    # Visualization suite
    # inference_viz uses argparse.BooleanOptionalAction with default=None so that the CLI
    # can express "leave the config default" (omit), "force on" (--inference-viz), and
    # "force off" (--no-inference-viz)
    if hasattr(args, "inference_viz") and args.inference_viz is not None:
        config.inference.inference_viz_enabled = args.inference_viz
    if hasattr(args, "stamp_gallery_top_k") and args.stamp_gallery_top_k is not None:
        config.inference.stamp_gallery_top_k = args.stamp_gallery_top_k
    if hasattr(args, "max_candidate_plots") and args.max_candidate_plots is not None:
        config.inference.max_candidate_plots = args.max_candidate_plots
    if (
        hasattr(args, "max_retries")
        and args.max_retries is not None
        and getattr(args, "command", None) == "inference"
    ):
        config.inference.max_retries = args.max_retries
    if (
        hasattr(args, "retry_delay")
        and args.retry_delay is not None
        and getattr(args, "command", None) == "inference"
    ):
        config.inference.retry_delay = args.retry_delay


# NOTE: come back to this later (make sure these checks are both comprehensive and implemented correctly)
def collect_validation_errors(
    args: argparse.Namespace, num_replicas: int | None
) -> list[ValidationError]:
    """
    Collect semantic and cross-param validation failures for the parsed CLI namespace, merging
    args with config defaults via _resolve(). Returns a list rather than raising so utility
    scripts (e.g. utils/find_optimal_configs.py) can post-process violations and propose fixes;
    validate_args() is the thin wrapper that turns this list into a ValueError.
    """

    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")

    cmd = getattr(args, "command", None)
    errors: list[ValidationError] = []

    # --num-replicas validation (>= 1 and <= available GPUs) is performed upstream
    # by _resolve_num_replicas(), which validate_args() calls before
    # collect_validation_errors(). By the time we get here, num_replicas is known to
    # be a sane positive int (or None on a no-GPU dev box, in which case cross-
    # replica divisibility checks are skipped further down).

    # Checks below are ordered to follow the parser sections in _add_train_flags_to /
    # _add_inference_flags_to, which themselves mirror the sub-dataclass order in
    # config.py. Within each section, fields appear in the same order they are
    # registered on the parser. Cross-parameter checks live in the section of the
    # primary constrained field.

    # ============================================================================
    # COMMON CHECKS — flags shared by both subcommands
    # ============================================================================
    # --hf-repo-id must look like an HF "namespace/name" repo id (shared Pattern B flag).
    hf_repo_id = _resolve(args, "hf_repo_id", config.hf.repo_id)
    if hf_repo_id is not None and not _HF_REPO_ID_PATTERN.match(hf_repo_id):
        errors.append(
            ValidationError(
                field="hf.repo_id",
                current=hf_repo_id,
                message=f"--hf-repo-id must be a HuggingFace repo id of the form namespace/name, got {hf_repo_id!r}",
                fix_kind="format",
            )
        )

    # Reproducibility (#279, shared flags): the root seed must be non-negative
    # (np.random.SeedSequence and tf.random.set_seed both reject negatives at runtime)
    seed = _resolve(args, "seed", config.reproducibility.seed)
    if seed is not None and seed < 0:
        errors.append(
            ValidationError(
                field="reproducibility.seed",
                current=seed,
                message=f"--seed must be a non-negative integer, got {seed}",
                fix_kind="clamp_low",
                min_val=0,
            )
        )

    # Same bound for the deprecated explicit RF override (train-only flag; _resolve falls
    # back to the config default on the inference namespace)
    rf_seed = _resolve(args, "rf_seed", config.rf.seed)
    if rf_seed is not None and rf_seed < 0:
        errors.append(
            ValidationError(
                field="rf.seed",
                current=rf_seed,
                message=f"--rf-seed must be a non-negative integer, got {rf_seed}",
                fix_kind="clamp_low",
                min_val=0,
            )
        )

    # ============================================================================
    # TRAINING-MODE CHECKS — match _add_train_flags_to layout
    # ============================================================================
    if cmd == "train":
        # ---------------------------------------------------------------- BetaVAE
        # --vae-dense-layer-size must match width_bin // downsample_factor (a Data
        # constraint that lands on a BetaVAE field — gate on the BetaVAE field).
        wb = _resolve(args, "width_bin", config.data.width_bin)
        df = _resolve(args, "downsample_factor", config.data.downsample_factor)
        vds = _resolve(args, "vae_dense_layer_size", config.beta_vae.dense_layer_size)
        if wb and df and vds is not None:
            expected = wb // df
            if vds != expected:
                errors.append(
                    ValidationError(
                        field="beta_vae.dense_layer_size",
                        current=vds,
                        message=f"--vae-dense-layer-size ({vds}) must equal width_bin // downsample_factor ({wb} // {df} = {expected})",
                        fix_kind="range",
                        min_val=expected,
                        max_val=expected,
                    )
                )

        # ---------------------------------------------------------- Random Forest
        rfmf = _resolve(args, "rf_max_features", config.rf.max_features)
        if rfmf is not None and not (
            rfmf in _RF_MAX_FEATURES_STR_VALUES or isinstance(rfmf, (int, float))
        ):
            errors.append(
                ValidationError(
                    field="rf.max_features",
                    current=rfmf,
                    message=f"--rf-max-features must be one of {sorted(_RF_MAX_FEATURES_STR_VALUES)} or a number, got {rfmf!r}",
                    fix_kind="enum",
                    allowed=sorted(_RF_MAX_FEATURES_STR_VALUES),
                )
            )

        rfj = _resolve(args, "rf_n_jobs", config.rf.n_jobs)
        n_cores = os.cpu_count() or 1
        if rfj is not None and not (-1 <= rfj <= n_cores):
            errors.append(
                ValidationError(
                    field="rf.n_jobs",
                    current=rfj,
                    message=f"--rf-n-jobs must satisfy -1 <= n_jobs <= cpu_count ({n_cores}), got {rfj}",
                    fix_kind="range",
                    min_val=-1,
                    max_val=n_cores,
                )
            )

        # --------------------------------------------------------------------- GPU
        # (--num-replicas <= 0 lives in the COMMON checks block above; nothing else
        # in GPUConfig requires train-side validation.)

        # -------------------------------------------------------------------- Data
        # TODO: Deferred -- time_bin & width_bin must match data shape — requires
        # loading actual data files; validated at runtime by the data loader.
        data_path = _resolve(args, "data_path", config.data_path)
        train_files = _resolve(args, "train_files", config.data.train_files) or []
        for f in train_files:
            full = config.get_training_file_path(f, data_path)
            if not os.path.exists(full):
                errors.append(
                    ValidationError(
                        field="data.train_files",
                        current=f,
                        message=f"--train-files: file does not exist on disk: {full}",
                        fix_kind="file_exists",
                    )
                )

        # ---------------------------------------------------------------- Training
        # Resolve the fields needed by multiple checks up front so the cross-field
        # validations below can refer to them without re-resolving.
        ntr = _resolve(args, "num_training_rounds", config.training.num_training_rounds)
        nsb = _resolve(args, "num_samples_beta_vae", config.training.num_samples_beta_vae)
        nsr = _resolve(args, "num_samples_rf", config.training.num_samples_rf)
        tvs = _resolve(args, "train_val_split", config.training.train_val_split)
        prb = _resolve(args, "per_replica_batch_size", config.training.per_replica_batch_size)
        eb = _resolve(args, "effective_batch_size", config.training.effective_batch_size)
        prvb = _resolve(
            args, "per_replica_val_batch_size", config.training.per_replica_val_batch_size
        )
        sic = _resolve(
            args, "signal_injection_chunk_size", config.training.signal_injection_chunk_size
        )
        lvc = _resolve(
            args,
            "latent_viz_num_cadences_per_type",
            config.training.latent_viz_num_cadences_per_type,
        )
        cs = _resolve(args, "curriculum_schedule", config.training.curriculum_schedule)

        # NOTE: come back to this later (should we parametrize num_signal_types = 4 in config.py?)
        # num_samples_beta_vae divisible by 4 (balanced class generation)
        if nsb is not None and nsb % 4 != 0:
            errors.append(
                ValidationError(
                    field="training.num_samples_beta_vae",
                    current=nsb,
                    message=f"--num-samples-beta-vae must be divisible by 4 for balanced class generation, got {nsb}",
                    fix_kind="divisibility",
                    divisor=4,
                )
            )
        # num_samples_rf must be divisible by 4 (class balance) AND by 2 (triplet batches).
        if nsr is not None and nsr % 4 != 0:
            errors.append(
                ValidationError(
                    field="training.num_samples_rf",
                    current=nsr,
                    message=f"--num-samples-rf must be divisible by 4 for balanced class generation, got {nsr}",
                    fix_kind="divisibility",
                    divisor=4,
                )
            )
        if nsr is not None and nsr % 2 != 0:
            errors.append(
                ValidationError(
                    field="training.num_samples_rf",
                    current=nsr,
                    message=f"--num-samples-rf must be divisible by 2 for the balanced true/false halves in data generation, got {nsr}",
                    fix_kind="divisibility",
                    divisor=2,
                )
            )

        # train_val_split bounds
        if tvs is not None and not (0 <= tvs <= 1):
            errors.append(
                ValidationError(
                    field="training.train_val_split",
                    current=tvs,
                    message=f"--train-val-split must satisfy 0 <= split <= 1, got {tvs}",
                    fix_kind="range",
                    min_val=0.0,
                    max_val=1.0,
                )
            )

        # signal_injection_chunk_size divisible by 4 (class balance)
        if sic is not None and sic % 4 != 0:
            errors.append(
                ValidationError(
                    field="training.signal_injection_chunk_size",
                    current=sic,
                    message=f"--signal-injection-chunk-size must be divisible by 4 for balanced class generation, got {sic}",
                    fix_kind="divisibility",
                    divisor=4,
                )
            )

        # data_gen_task_size >= 1 (batched memmap worker tasks)
        dgts = _resolve(args, "data_gen_task_size", config.training.data_gen_task_size)
        if dgts is not None and dgts < 1:
            errors.append(
                ValidationError(
                    field="training.data_gen_task_size",
                    current=dgts,
                    message=f"--data-gen-task-size must be >= 1, got {dgts}",
                    fix_kind="clamp_low",
                    min_val=1,
                )
            )

        # Latent traversal: odd step count >= 3 (so the center column is the unperturbed
        # class-mean decode) and a strictly positive sigma range. No upper step bound:
        # decode cost scales linearly and stays trivial at any plausible value (99 steps
        # ~= 800 decoder calls), and an oversized figure is immediately self-evident
        lts = _resolve(
            args, "latent_traversal_num_steps", config.training.latent_traversal_num_steps
        )
        if lts is not None and (lts < 3 or lts % 2 == 0):
            errors.append(
                ValidationError(
                    field="training.latent_traversal_num_steps",
                    current=lts,
                    message=f"--latent-traversal-num-steps must be odd and >= 3 so the center column is the unperturbed decode, got {lts}",
                    # No fix_kind captures "odd and >= 3"; clamp_low is closest and the
                    # proposal seed is the field's default (7 — odd and valid), matching the
                    # max_sigma block below. propose_simple_fix only ever suggests this value.
                    fix_kind="clamp_low",
                    min_val=7,
                )
            )
        ltms = _resolve(
            args, "latent_traversal_max_sigma", config.training.latent_traversal_max_sigma
        )
        if ltms is not None and ltms <= 0:
            errors.append(
                ValidationError(
                    field="training.latent_traversal_max_sigma",
                    current=ltms,
                    message=f"--latent-traversal-max-sigma must be > 0, got {ltms}",
                    fix_kind="clamp_low",
                    min_val=3.0,  # Proposal seed: the field's default (matches num_steps above)
                )
            )

        # Disk budget for the disk-backed round datasets (round_data.py): with overlap enabled
        # two rounds coexist on disk (round k trains while round k+1 generates), so require
        # 2.2x one round's estimated size free; 1.1x when overlap is disabled. Estimated from
        # sample counts only — actual usage tracks the estimate closely since the arrays are
        # fixed-shape float32.
        nob = _resolve(args, "num_observations", config.data.num_observations)
        tb = _resolve(args, "time_bins", config.data.time_bins)
        overlap = _resolve(args, "overlap_data_generation", config.training.overlap_data_generation)
        round_data_dir = _resolve(args, "round_data_dir", config.training.round_data_dir)
        if round_data_dir is None:
            # data_path was resolved in the Data block above; round data are generated
            # training data, so they live under the training-data root
            round_data_dir = config.get_training_file_path("round_data", data_path)
        if all(v is not None for v in (nsb, nob, tb, wb)) and df:
            round_nbytes = _estimate_round_data_nbytes(nsb, nob, tb, wb // df)
            required_factor = 2.2 if overlap else 1.1
            required_bytes = required_factor * round_nbytes
            try:
                free_bytes = shutil.disk_usage(_nearest_existing_ancestor(round_data_dir)).free
            except OSError as e:
                logger.warning(
                    f"Could not check free disk space for --round-data-dir "
                    f"({round_data_dir}): {e} — skipping the disk-budget check"
                )
            else:
                if free_bytes < required_bytes:
                    errors.append(
                        ValidationError(
                            field="training.round_data_dir",
                            current=round_data_dir,
                            message=(
                                f"--round-data-dir ({round_data_dir}) needs >= "
                                f"{required_bytes / 1e9:.1f} GB free ({required_factor}x one "
                                f"round's ~{round_nbytes / 1e9:.1f} GB"
                                f"{' with data-generation overlap enabled' if overlap else ''}), "
                                f"but only {free_bytes / 1e9:.1f} GB is available. Free up disk "
                                f"space, point --round-data-dir at a larger volume, reduce "
                                f"--num-samples-beta-vae, or pass --no-overlap-data-generation "
                                f"to halve the requirement"
                            ),
                            fix_kind="file_exists",
                        )
                    )

        # SNR sanity (positivity + curriculum ordering)
        snr_base = _resolve(args, "snr_base", config.training.snr_base)
        snr_init = _resolve(args, "initial_snr_range", config.training.initial_snr_range)
        snr_fin = _resolve(args, "final_snr_range", config.training.final_snr_range)
        if snr_base is not None and snr_base <= 0:
            errors.append(
                ValidationError(
                    field="training.snr_base",
                    current=snr_base,
                    message=f"--snr-base must be > 0, got {snr_base}",
                    fix_kind="clamp_low",
                    min_val=1,
                )
            )
        if snr_init is not None and snr_init <= 0:
            errors.append(
                ValidationError(
                    field="training.initial_snr_range",
                    current=snr_init,
                    message=f"--initial-snr-range must be > 0, got {snr_init}",
                    fix_kind="clamp_low",
                    min_val=1,
                )
            )
        if snr_fin is not None and snr_fin <= 0:
            errors.append(
                ValidationError(
                    field="training.final_snr_range",
                    current=snr_fin,
                    message=f"--final-snr-range must be > 0, got {snr_fin}",
                    fix_kind="clamp_low",
                    min_val=1,
                )
            )
        if snr_init is not None and snr_fin is not None and snr_init < snr_fin:
            errors.append(
                ValidationError(
                    field="training.initial_snr_range",
                    current=snr_init,
                    message=f"--initial-snr-range ({snr_init}) must be >= --final-snr-range ({snr_fin}) — curriculum schedules from easy to hard",
                    fix_kind="clamp_low",
                    min_val=snr_fin,
                )
            )

        # Curriculum schedule enum + exponential / step parameters
        if cs is not None and cs not in _CURRICULUM_SCHEDULES:
            errors.append(
                ValidationError(
                    field="training.curriculum_schedule",
                    current=cs,
                    message=f"--curriculum-schedule must be one of {sorted(_CURRICULUM_SCHEDULES)}, got {cs!r}",
                    fix_kind="enum",
                    allowed=sorted(_CURRICULUM_SCHEDULES),
                )
            )

        edr = _resolve(args, "exponential_decay_rate", config.training.exponential_decay_rate)
        # exponential_decay_rate only governs the exponential schedule; skip it otherwise.
        if cs == "exponential" and edr is not None and edr >= 0:
            errors.append(
                ValidationError(
                    field="training.exponential_decay_rate",
                    current=edr,
                    message=f"--exponential-decay-rate must be < 0 (more negative = faster difficulty ramp), got {edr}",
                    fix_kind="clamp_high",
                    max_val=-0.01,
                )
            )

        ser = _resolve(args, "step_easy_rounds", config.training.step_easy_rounds)
        shr = _resolve(args, "step_hard_rounds", config.training.step_hard_rounds)
        # step_easy_rounds / step_hard_rounds only apply to the step schedule; skip their
        # range checks otherwise.
        if cs == "step" and ntr is not None and ser is not None and not (0 <= ser <= ntr):
            errors.append(
                ValidationError(
                    field="training.step_easy_rounds",
                    current=ser,
                    message=f"--step-easy-rounds must satisfy 0 <= rounds <= num_training_rounds ({ntr}), got {ser}",
                    fix_kind="range",
                    min_val=0,
                    max_val=ntr,
                )
            )
        if cs == "step" and ntr is not None and shr is not None and not (0 <= shr <= ntr):
            errors.append(
                ValidationError(
                    field="training.step_hard_rounds",
                    current=shr,
                    message=f"--step-hard-rounds must satisfy 0 <= rounds <= num_training_rounds ({ntr}), got {shr}",
                    fix_kind="range",
                    min_val=0,
                    max_val=ntr,
                )
            )
        # step_easy + step_hard only need to sum to num_training_rounds when step
        # schedule is selected.
        if (
            cs == "step"
            and ntr is not None
            and ser is not None
            and shr is not None
            and ser + shr != ntr
        ):
            errors.append(
                ValidationError(
                    field="training.step_easy_rounds",
                    current=ser,
                    message=f"--step-easy-rounds + --step-hard-rounds ({ser} + {shr} = {ser + shr}) must equal --num-training-rounds ({ntr}) when curriculum_schedule=step",
                    fix_kind="cross_param",
                    extra={"step_hard_rounds": shr, "num_training_rounds": ntr},
                )
            )

        # Learning rate / patience
        blr = _resolve(args, "base_learning_rate", config.training.base_learning_rate)
        mlr = _resolve(args, "min_learning_rate", config.training.min_learning_rate)
        if blr is not None and mlr is not None and blr < mlr:
            errors.append(
                ValidationError(
                    field="training.base_learning_rate",
                    current=blr,
                    message=f"--base-learning-rate ({blr}) must be >= --min-learning-rate ({mlr})",
                    fix_kind="clamp_low",
                    min_val=mlr,
                )
            )
        mpi = _resolve(args, "min_pct_improvement", config.training.min_pct_improvement)
        if mpi is not None and mpi < 0:
            errors.append(
                ValidationError(
                    field="training.min_pct_improvement",
                    current=mpi,
                    message=f"--min-pct-improvement must be >= 0, got {mpi}",
                    fix_kind="clamp_low",
                    min_val=0.0,
                )
            )
        pt = _resolve(args, "patience_threshold", config.training.patience_threshold)
        if pt is not None and pt < 1:
            errors.append(
                ValidationError(
                    field="training.patience_threshold",
                    current=pt,
                    message=f"--patience-threshold must be >= 1 (must wait at least one epoch before reducing LR), got {pt}",
                    fix_kind="clamp_low",
                    min_val=1,
                )
            )
        lrf = _resolve(args, "lr_reduction_factor", config.training.reduction_factor)
        if lrf is not None and not (0 < lrf < 1):
            errors.append(
                ValidationError(
                    field="training.reduction_factor",
                    current=lrf,
                    message=f"--lr-reduction-factor must satisfy 0 < factor < 1 (it's a reduction multiplier), got {lrf}",
                    fix_kind="range",
                    min_val=0.01,
                    max_val=0.99,
                )
            )

        # Retries (training-scoped — args.max_retries / args.retry_delay also exist
        # in the inference subparser, so the Pattern C gate on cmd kept this clean).
        mr = _resolve(args, "max_retries", config.training.max_retries)
        if mr is not None and mr < 1:
            errors.append(
                ValidationError(
                    field="training.max_retries",
                    current=mr,
                    message=f"--max-retries must be >= 1 (the run needs at least one attempt), got {mr}",
                    fix_kind="clamp_low",
                    min_val=1,
                )
            )
        rd = _resolve(args, "retry_delay", config.training.retry_delay)
        if rd is not None and rd < 0:
            errors.append(
                ValidationError(
                    field="training.retry_delay",
                    current=rd,
                    message=f"--retry-delay must be >= 0, got {rd}",
                    fix_kind="clamp_low",
                    min_val=0,
                )
            )

        # Cross-replica batch / sample constraints. Tied to multiple Training fields
        # (effective_batch_size, per_replica_*_batch_size, num_samples_*, latent_viz_*)
        # plus num_replicas, so they live at the end of the Training section.
        if all(v is not None for v in (prb, eb, prvb, lvc, nsb, nsr, tvs)):
            if num_replicas is None:
                logger.warning(
                    "GPU count unknown (TF reports 0 GPUs or is unavailable) — skipping "
                    "cross-replica divisibility checks; they need a TF-visible GPU count "
                    "even when --num-replicas is passed. The effective-batch-size check "
                    "re-runs at training time once the real replica count is known."
                )
            elif num_replicas < 1:
                # _resolve_num_replicas would have raised before we got here for a real
                # pipeline invocation. Reachable only via direct callers (e.g.
                # utils/find_optimal_configs.py) that bypass validate_args; skip the
                # cross-replica section to avoid div-by-zero.
                pass
            else:
                train_samples = nsb * tvs
                val_samples = nsb * (1 - tvs)
                global_train_batch = prb * num_replicas
                global_val_batch = prvb * num_replicas
                latent_total = lvc * 4

                if not (global_train_batch <= eb <= train_samples):
                    errors.append(
                        ValidationError(
                            field="training.effective_batch_size",
                            current=eb,
                            message=f"--effective-batch-size ({eb}) must satisfy per_replica_batch_size * num_replicas ({prb} * {num_replicas} = {global_train_batch}) <= effective_batch_size <= num_samples_beta_vae * train_val_split ({nsb} * {tvs} = {train_samples})",
                            fix_kind="cross_param",
                            extra={
                                "per_replica_batch_size": prb,
                                "num_replicas": num_replicas,
                                "num_samples_beta_vae": nsb,
                                "train_val_split": tvs,
                            },
                        )
                    )
                if global_val_batch > val_samples:
                    errors.append(
                        ValidationError(
                            field="training.per_replica_val_batch_size",
                            current=prvb,
                            message=f"--per-replica-val-batch-size * num_replicas ({prvb} * {num_replicas} = {global_val_batch}) must be <= num_samples_beta_vae * (1 - train_val_split) ({nsb} * {1 - tvs:.4f} = {val_samples})",
                            fix_kind="cross_param",
                        )
                    )
                # The RF splits num_samples_rf by train_val_split (like the beta-VAE), so the
                # val set the runtime actually builds is nsr * (1 - train_val_split), then
                # trims it to a multiple of the global val batch. Validate that val portion,
                # not raw nsr — otherwise train_random_forest fails at runtime with
                # "val_steps < 1: n_val_trimmed (0) ..." when the val split is too small.
                rf_val_samples = round(nsr * (1 - tvs))
                if global_val_batch > rf_val_samples:
                    errors.append(
                        ValidationError(
                            field="training.num_samples_rf",
                            current=nsr,
                            message=f"num_samples_rf * (1 - train_val_split) ({nsr} * {1 - tvs:.4f} = {rf_val_samples}) must be >= per_replica_val_batch_size * num_replicas ({prvb} * {num_replicas} = {global_val_batch})",
                            fix_kind="cross_param",
                        )
                    )
                if latent_total > val_samples:
                    errors.append(
                        ValidationError(
                            field="training.latent_viz_num_cadences_per_type",
                            current=lvc,
                            message=f"--latent-viz-num-cadences-per-type * 4 ({lvc} * 4 = {latent_total}) must be <= num_samples_beta_vae * (1 - train_val_split) ({val_samples})",
                            fix_kind="cross_param",
                        )
                    )
                if eb % global_train_batch != 0:
                    errors.append(
                        ValidationError(
                            field="training.effective_batch_size",
                            current=eb,
                            message=f"--effective-batch-size ({eb}) must be divisible by per_replica_batch_size * num_replicas ({global_train_batch})",
                            fix_kind="cross_param",
                            divisor=global_train_batch,
                        )
                    )
                # Round to int — train_samples/val_samples come from `nsb * tvs` and
                # `nsb * (1 - tvs)` which suffer IEEE-754 noise (e.g.
                # 499200 * 0.2 = 99839.999...). Use round() so the intended sample
                # counts survive the cast.
                train_samples_i = round(train_samples)
                val_samples_i = round(val_samples)
                if train_samples_i % eb != 0:
                    errors.append(
                        ValidationError(
                            field="training.num_samples_beta_vae",
                            current=nsb,
                            message=f"num_samples_beta_vae * train_val_split ({train_samples_i}) must be divisible by --effective-batch-size ({eb})",
                            fix_kind="cross_param",
                            divisor=eb,
                        )
                    )
                if val_samples_i % global_val_batch != 0:
                    errors.append(
                        ValidationError(
                            field="training.num_samples_beta_vae",
                            current=nsb,
                            message=f"num_samples_beta_vae * (1 - train_val_split) ({val_samples_i}) must be divisible by per_replica_val_batch_size * num_replicas ({global_val_batch})",
                            fix_kind="cross_param",
                            divisor=global_val_batch,
                        )
                    )
                if nsr % global_val_batch != 0:
                    errors.append(
                        ValidationError(
                            field="training.num_samples_rf",
                            current=nsr,
                            message=f"--num-samples-rf ({nsr}) must be divisible by per_replica_val_batch_size * num_replicas ({global_val_batch})",
                            fix_kind="cross_param",
                            divisor=global_val_batch,
                        )
                    )
                if latent_total % global_val_batch != 0:
                    errors.append(
                        ValidationError(
                            field="training.latent_viz_num_cadences_per_type",
                            current=lvc,
                            message=f"--latent-viz-num-cadences-per-type * 4 ({latent_total}) must be divisible by per_replica_val_batch_size * num_replicas ({global_val_batch})",
                            fix_kind="cross_param",
                            divisor=global_val_batch,
                        )
                    )

        # -------------------------------------------------------------- Checkpoint
        # TODO: Deferred -- save_tag uniqueness check requires the DB (init_db runs
        # after validate_args).
        load_tag = _resolve(args, "load_tag", config.checkpoint.load_tag)
        if load_tag is not None and not _LOAD_TAG_PATTERN.match(load_tag):
            errors.append(
                ValidationError(
                    field="checkpoint.load_tag",
                    current=load_tag,
                    message=f"--load-tag must be a full run tag ({{test|train|inf|bench}}_YYYYMMDD_HHMMSS) or round_XX; got {load_tag!r}",
                    fix_kind="format",
                )
            )

        # Resume footgun (issue #142): per-round checkpoints are saved under the checkpoints/
        # subdirectory, so `--load-tag round_XX` without `--load-dir checkpoints` searches the
        # models root instead and used to silently resume from a stale, unrelated model.
        load_dir = _resolve(args, "load_dir", config.checkpoint.load_dir)
        if (
            load_tag is not None
            and _ROUND_TAG_PATTERN.match(load_tag)
            and load_dir != "checkpoints"
        ):
            errors.append(
                ValidationError(
                    field="checkpoint.load_dir",
                    current=load_dir,
                    message=(
                        f"--load-tag {load_tag!r} names a per-round checkpoint, and those are "
                        f"saved under the 'checkpoints/' subdirectory — pass --load-dir "
                        f"checkpoints to resume from it. (If you genuinely keep a final model "
                        f"tagged {load_tag!r} in the models root, re-save it under a full run "
                        f"tag ({{test|train|inf|bench}}_YYYYMMDD_HHMMSS) and load that instead.)"
                    ),
                    fix_kind="cross_param",
                )
            )

        sr = _resolve(args, "start_round", config.checkpoint.start_round)
        if sr is not None and ntr is not None and not (1 <= sr <= ntr):
            errors.append(
                ValidationError(
                    field="checkpoint.start_round",
                    current=sr,
                    message=f"--start-round must satisfy 1 <= round <= num_training_rounds ({ntr}), got {sr}",
                    fix_kind="range",
                    min_val=1,
                    max_val=ntr,
                )
            )

        save_tag = getattr(args, "save_tag", None)
        if save_tag is not None and not _SAVE_TAG_PATTERN.match(save_tag):
            errors.append(
                ValidationError(
                    field="checkpoint.save_tag",
                    current=save_tag,
                    message=f"--save-tag must be one of: test, train, inf, bench (the datetime is appended automatically); got {save_tag!r}",
                    fix_kind="format",
                )
            )

    # ============================================================================
    # INFERENCE-MODE CHECKS — match _add_inference_flags_to layout
    # ============================================================================
    if cmd == "inference":
        # ----------------------------------------------------- Inference model artifacts
        # The encoder/RF/config artifact trio is all-or-none: with all three paths given
        # they must each exist on disk; with none given, main.py's
        # resolve_inference_artifacts() has already filled `args` with HuggingFace-
        # downloaded cache paths before validation runs (utility scripts that skip that
        # step simply defer to the Hub default). A partial trio is an error — mixing local
        # and Hub-sourced artifacts would silently pair mismatched models.
        artifact_specs = (
            ("encoder_path", "encoder_path", "--encoder-path"),
            ("rf_path", "rf_path", "--rf-path"),
            ("config_path", "config_path", "--config-path"),
        )
        artifact_paths = {
            arg_name: _resolve(args, arg_name, getattr(config.inference, cfg_attr))
            for arg_name, cfg_attr, _flag in artifact_specs
        }
        n_provided = sum(path is not None for path in artifact_paths.values())
        for arg_name, cfg_attr, flag in artifact_specs:
            path = artifact_paths[arg_name]
            if path is None and 0 < n_provided < len(artifact_specs):
                errors.append(
                    ValidationError(
                        field=f"inference.{cfg_attr}",
                        current=None,
                        message=f"{flag} is missing: provide all three of --encoder-path/--rf-path/--config-path, or omit all three to download the artifacts from the HuggingFace Hub",
                        fix_kind="file_exists",
                    )
                )
            elif path is not None and not os.path.exists(path):
                errors.append(
                    ValidationError(
                        field=f"inference.{cfg_attr}",
                        current=path,
                        message=f"{flag}: file does not exist on disk: {path}",
                        fix_kind="file_exists",
                    )
                )

        # -------------------------------------------------------------------- Data
        data_path = _resolve(args, "data_path", config.data_path)
        test_files = _resolve(args, "test_files", config.data.test_files) or []
        for f in test_files:
            full = config.get_test_file_path(f, data_path)
            if not os.path.exists(full):
                errors.append(
                    ValidationError(
                        field="data.test_files",
                        current=f,
                        message=f"--test-files: file does not exist on disk: {full}",
                        fix_kind="file_exists",
                    )
                )
        inf_files = _resolve(args, "inference_files", config.data.inference_files)
        for f in inf_files or []:
            full = config.get_inference_file_path(f, data_path)
            if not os.path.exists(full):
                errors.append(
                    ValidationError(
                        field="data.inference_files",
                        current=f,
                        message=f"--inference-files: file does not exist on disk: {full}",
                        fix_kind="file_exists",
                    )
                )

        # --------------------------------------------------------------- Inference
        ct = _resolve(args, "classification_threshold", config.inference.classification_threshold)
        if ct is not None and not (0 <= ct <= 1):
            errors.append(
                ValidationError(
                    field="inference.classification_threshold",
                    current=ct,
                    message=f"--classification-threshold must satisfy 0 <= threshold <= 1, got {ct}",
                    fix_kind="range",
                    min_val=0.0,
                    max_val=1.0,
                )
            )

        # -------------------------------------------- Energy detection preprocessing
        # When inference_files is set the energy-detection pipeline runs before
        # inference; stamp_width must equal data.width_bin so the extracted
        # (n_hits, 6, 16, stamp_width) tensor matches the downstream downsample +
        # log-norm path.
        if inf_files is not None:
            sw_inf = _resolve(args, "stamp_width", config.inference.stamp_width)
            wb_inf = _resolve(args, "width_bin", config.data.width_bin)
            if sw_inf is not None and wb_inf is not None and sw_inf != wb_inf:
                errors.append(
                    ValidationError(
                        field="inference.stamp_width",
                        current=sw_inf,
                        message=(
                            f"--stamp-width ({sw_inf}) must equal --width-bin ({wb_inf}) "
                            "when --inference-files is set"
                        ),
                        fix_kind="range",
                        min_val=wb_inf,
                        max_val=wb_inf,
                    )
                )
            group_cols = _resolve(
                args, "cadence_group_by_cols", config.inference.cadence_group_by_cols
            )
            if group_cols is not None and len(group_cols) == 0:
                errors.append(
                    ValidationError(
                        field="inference.cadence_group_by_cols",
                        current=group_cols,
                        message="--cadence-group-by-cols must be non-empty when --inference-files is provided",
                        fix_kind="clamp_low",
                    )
                )
            # Downsample-at-extraction stores stamp_width // downsample_factor bins per
            # stamp, so the division must be exact for the stored width to be well-defined
            store_ds = _resolve(
                args, "store_downsampled_stamps", config.inference.store_downsampled_stamps
            )
            df_inf = _resolve(args, "downsample_factor", config.data.downsample_factor)
            if (
                store_ds
                and sw_inf is not None
                and df_inf is not None
                and df_inf > 0
                and sw_inf % df_inf != 0
            ):
                errors.append(
                    ValidationError(
                        field="inference.stamp_width",
                        current=sw_inf,
                        message=(
                            f"--stamp-width ({sw_inf}) must be divisible by "
                            f"--downsample-factor ({df_inf}) when stamps are downsampled at "
                            "extraction (--store-downsampled-stamps)"
                        ),
                        fix_kind="divisibility",
                        divisor=df_inf,
                    )
                )

        coarse_channel_width = _resolve(
            args, "coarse_channel_width", config.inference.coarse_channel_width
        )
        coarse_channel_log_interval = _resolve(
            args, "coarse_channel_log_interval", config.inference.coarse_channel_log_interval
        )
        spline_order = _resolve(args, "spline_order", config.inference.spline_order)
        detection_window_size = _resolve(
            args, "detection_window_size", config.inference.detection_window_size
        )
        detection_step_size = _resolve(
            args, "detection_step_size", config.inference.detection_step_size
        )
        stat_threshold = _resolve(args, "stat_threshold", config.inference.stat_threshold)
        stamp_width = _resolve(args, "stamp_width", config.inference.stamp_width)
        overlap_fraction = _resolve(args, "overlap_fraction", config.inference.overlap_fraction)

        # Positivity checks for the bare-int / bare-float fields, in parser order.
        for name, val in (
            ("coarse_channel_width", coarse_channel_width),
            ("coarse_channel_log_interval", coarse_channel_log_interval),
            ("stat_threshold", stat_threshold),
            ("stamp_width", stamp_width),
        ):
            if val is not None and val <= 0:
                errors.append(
                    ValidationError(
                        field=f"inference.{name}",
                        current=val,
                        message=f"--{name.replace('_', '-')} must be > 0, got {val}",
                        fix_kind="clamp_low",
                        min_val=1,
                    )
                )
        if spline_order is not None and spline_order < 1:
            errors.append(
                ValidationError(
                    field="inference.spline_order",
                    current=spline_order,
                    message=f"--spline-order must be >= 1, got {spline_order}",
                    fix_kind="clamp_low",
                    min_val=1,
                )
            )
        bandpass_method = _resolve(args, "bandpass_method", config.inference.bandpass_method)
        if bandpass_method is not None and bandpass_method not in _BANDPASS_METHODS:
            errors.append(
                ValidationError(
                    field="inference.bandpass_method",
                    current=bandpass_method,
                    message=f"--bandpass-method must be one of {sorted(_BANDPASS_METHODS)}, got {bandpass_method!r}",
                    fix_kind="enum",
                    allowed=sorted(_BANDPASS_METHODS),
                )
            )
        pfb_taps = _resolve(args, "pfb_taps_per_channel", config.inference.pfb_taps_per_channel)
        if pfb_taps is not None and pfb_taps < 1:
            errors.append(
                ValidationError(
                    field="inference.pfb_taps_per_channel",
                    current=pfb_taps,
                    message=f"--pfb-taps-per-channel must be >= 1, got {pfb_taps}",
                    fix_kind="clamp_low",
                    min_val=1,
                )
            )
        # detection_window_size <= stamp_width
        if (
            detection_window_size is not None
            and stamp_width is not None
            and detection_window_size > stamp_width
        ):
            errors.append(
                ValidationError(
                    field="inference.detection_window_size",
                    current=detection_window_size,
                    message=(
                        f"--detection-window-size ({detection_window_size}) must be"
                        f" <= --stamp-width ({stamp_width})"
                    ),
                    fix_kind="clamp_high",
                    max_val=stamp_width,
                )
            )
        # detection_step_size > 0 and <= detection_window_size
        if detection_step_size is not None and detection_step_size <= 0:
            errors.append(
                ValidationError(
                    field="inference.detection_step_size",
                    current=detection_step_size,
                    message=f"--detection-step-size must be > 0, got {detection_step_size}",
                    fix_kind="clamp_low",
                    min_val=1,
                )
            )
        if (
            detection_step_size is not None
            and detection_window_size is not None
            and detection_step_size > detection_window_size
        ):
            errors.append(
                ValidationError(
                    field="inference.detection_step_size",
                    current=detection_step_size,
                    message=(
                        f"--detection-step-size ({detection_step_size}) must be"
                        f" <= --detection-window-size ({detection_window_size})"
                    ),
                    fix_kind="clamp_high",
                    max_val=detection_window_size,
                )
            )
        if overlap_fraction is not None and not (0 <= overlap_fraction <= 1):
            errors.append(
                ValidationError(
                    field="inference.overlap_fraction",
                    current=overlap_fraction,
                    message=f"--overlap-fraction must satisfy 0 <= fraction <= 1, got {overlap_fraction}",
                    fix_kind="range",
                    min_val=0.0,
                    max_val=1.0,
                )
            )

        # Visualization suite
        stamp_gallery_top_k = _resolve(
            args, "stamp_gallery_top_k", config.inference.stamp_gallery_top_k
        )
        if stamp_gallery_top_k is not None and stamp_gallery_top_k < 1:
            errors.append(
                ValidationError(
                    field="inference.stamp_gallery_top_k",
                    current=stamp_gallery_top_k,
                    message=f"--stamp-gallery-top-k must be >= 1, got {stamp_gallery_top_k}",
                    fix_kind="clamp_low",
                    min_val=1,
                )
            )
        max_candidate_plots = _resolve(
            args, "max_candidate_plots", config.inference.max_candidate_plots
        )
        if max_candidate_plots is not None and max_candidate_plots < 0:
            errors.append(
                ValidationError(
                    field="inference.max_candidate_plots",
                    current=max_candidate_plots,
                    message=f"--max-candidate-plots must be >= 0, got {max_candidate_plots}",
                    fix_kind="clamp_low",
                    min_val=0,
                )
            )

        # Retries (inference-scoped)
        mr = _resolve(args, "max_retries", config.inference.max_retries)
        if mr is not None and mr < 0:
            errors.append(
                ValidationError(
                    field="inference.max_retries",
                    current=mr,
                    message=f"--max-retries must be >= 0, got {mr}",
                    fix_kind="clamp_low",
                    min_val=0,
                )
            )
        rd = _resolve(args, "retry_delay", config.inference.retry_delay)
        if rd is not None and rd < 0:
            errors.append(
                ValidationError(
                    field="inference.retry_delay",
                    current=rd,
                    message=f"--retry-delay must be >= 0, got {rd}",
                    fix_kind="clamp_low",
                    min_val=0,
                )
            )

        # -------------------------------------------------------------- Checkpoint
        save_tag = getattr(args, "save_tag", None)
        if save_tag is not None and not _SAVE_TAG_PATTERN.match(save_tag):
            errors.append(
                ValidationError(
                    field="checkpoint.save_tag",
                    current=save_tag,
                    message=f"--save-tag must be one of: test, train, inf, bench (the datetime is appended automatically); got {save_tag!r}",
                    fix_kind="format",
                )
            )

    return errors


# NOTE: come back to this later (everything in this section)
# ---------------------------------------------------------------------------
# Fix proposal surface — used both by validate_args (to include suggestions in
# the ValueError message on failure) and by utils/find_optimal_configs.py (to
# print suggestions to stdout when run directly). Kept here in cli.py rather
# than utils/ so cli.py can call them without a circular import or sys.path
# gymnastics, and so the validation + proposal logic stays colocated.
# ---------------------------------------------------------------------------

# Neighborhoods for the fields the solver prefers to keep near their base value (data sizes +
# the throughput-optimal per-replica batch). effective_batch_size and per_replica_val_batch_size
# are deliberately NOT here: they are generated exactly from the divisibility structure (divisors
# of the train split / of the gcd of the val-side counts), so the search is small and complete
# rather than a coarse grid that either explodes or steps over the only valid values.
_SEARCH_RANGES: dict[str, tuple[int, int, int]] = {
    # (min, max, step)
    "num_samples_beta_vae": (400_000, 600_000, 10_240),
    "num_samples_rf": (80_000, 120_000, 2_048),
    "per_replica_batch_size": (64, 256, 64),
}

_CROSS_PARAM_FIELDS = (
    "num_samples_beta_vae",
    "num_samples_rf",
    "per_replica_batch_size",
    "effective_batch_size",
    "per_replica_val_batch_size",
)


def _divisors(n: int) -> list[int]:
    """Sorted positive divisors of ``n`` (empty for n < 1)."""
    if n < 1:
        return []
    small: list[int] = []
    large: list[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return small + large[::-1]


def _check_cross_constraints(
    num_samples_beta_vae: int,
    num_samples_rf: int,
    train_val_split: float,
    per_replica_batch_size: int,
    effective_batch_size: int,
    per_replica_val_batch_size: int,
    num_replicas_list: list[int],
    latent_total: int | None = None,
) -> bool:
    """Return True iff the six-tuple satisfies all cross-replica constraints for every
    num_replicas in the list. Mirrors the cross-replica checks in
    :func:`collect_validation_errors`. When ``latent_total`` (= latent_viz_num_cadences_per_type
    * 4) is supplied, the latent-viz divisibility check is included too, so a returned solution is
    valid against every cross-replica constraint the pipeline enforces — pass None to skip it (e.g.
    when the caller doesn't vary anything the latent batch depends on)."""
    if not (0 <= train_val_split <= 1):
        return False
    if num_samples_rf % 2 != 0:
        return False
    for nr in num_replicas_list:
        gtrain = per_replica_batch_size * nr
        gval = per_replica_val_batch_size * nr
        # round() rather than int() — `num_samples_beta_vae * (1 - train_val_split)`
        # suffers IEEE-754 noise (e.g. 499200 * 0.2 = 99839.999...) and int() would
        # truncate to 99839.
        train_samples = round(num_samples_beta_vae * train_val_split)
        val_samples = round(num_samples_beta_vae * (1 - train_val_split))
        if not (gtrain <= effective_batch_size <= num_samples_beta_vae * train_val_split):
            return False
        if gval > num_samples_beta_vae * (1 - train_val_split):
            return False
        if gval > num_samples_rf:
            return False
        if effective_batch_size % gtrain != 0:
            return False
        if train_samples % effective_batch_size != 0:
            return False
        if val_samples % gval != 0:
            return False
        if num_samples_rf % gval != 0:
            return False
        if latent_total is not None and latent_total % gval != 0:
            return False
    return True


def _solve_cross_param_constraints(
    base: dict[str, int | float],
    num_replicas_list: list[int],
    max_candidates: int = 5_000_000,
) -> dict[str, int | float] | None:
    """Find the batch/sample configuration nearest (L1) to ``base`` that satisfies every
    cross-replica constraint for all replica counts in ``num_replicas_list``, or None if none
    exists. ``base`` may carry an optional ``latent_total`` key (= latent_viz_num_cadences_per_type
    * 4) so the latent-viz constraint is honored.

    Rather than a blind grid (which both explodes past the candidate budget and steps over the only
    valid values), the effective batch and per-replica val batch — the two purely divisibility-bound
    fields — are enumerated *exactly* from the structure: the effective batch must be a divisor of
    the train split that is a multiple of ``per_replica_batch_size * num_replicas`` for every replica
    count; the global val batch must divide the gcd of the val split, num_samples_rf, and (if given)
    latent_total. The data sizes and per-replica batch are only searched over small neighborhoods
    (seeded with the base value, so they're preferred when a solution keeps them) — but because the
    result is chosen purely by L1 distance, a multi-count solve can still move per_replica_batch when
    doing so is closer than raising the effective batch to the larger multiple all counts share (e.g.
    a 4/5/6-GPU solve drops it to 64 rather than lifting the effective batch to lcm(128*{4,5,6})).
    Every candidate is confirmed with the shared :func:`_check_cross_constraints`, which stays the
    single source of truth."""
    latent_total = base.get("latent_total")
    tvs = base["train_val_split"]
    b_nsb = base["num_samples_beta_vae"]
    b_nsr = base["num_samples_rf"]
    b_p = base["per_replica_batch_size"]
    b_e = base["effective_batch_size"]
    b_v = base["per_replica_val_batch_size"]

    def _ok(nsb, nsr, p, e, v):
        return _check_cross_constraints(
            nsb, nsr, tvs, p, e, v, num_replicas_list, latent_total=latent_total
        )

    if _ok(b_nsb, b_nsr, b_p, b_e, b_v):
        return {
            "num_samples_beta_vae": b_nsb,
            "num_samples_rf": b_nsr,
            "train_val_split": tvs,
            "per_replica_batch_size": b_p,
            "effective_batch_size": b_e,
            "per_replica_val_batch_size": b_v,
        }

    def _neighborhood(field: str) -> list[int]:
        lo, hi, step = _SEARCH_RANGES[field]
        return sorted({base[field]} | set(range(lo, hi + 1, step)))

    nsb_cands = _neighborhood("num_samples_beta_vae")
    nsr_cands = _neighborhood("num_samples_rf")
    p_cands = _neighborhood("per_replica_batch_size")

    best: dict[str, int | float] | None = None
    best_dist = float("inf")
    checked = 0
    for nsb in nsb_cands:
        train_samples = round(nsb * tvs)
        val_samples = round(nsb * (1 - tvs))
        train_divisors = _divisors(train_samples)
        for p in p_cands:
            gtrains = [p * nr for nr in num_replicas_list]
            lcm_gtrain = 1
            for gt in gtrains:
                lcm_gtrain = lcm(lcm_gtrain, gt)
            min_e = max(gtrains)
            e_cands = [d for d in train_divisors if d >= min_e and d % lcm_gtrain == 0]
            if not e_cands:
                continue
            for nsr in nsr_cands:
                if nsr % 2 != 0:
                    continue
                g = gcd(val_samples, nsr)
                if latent_total is not None:
                    g = gcd(g, latent_total)
                v_cands = [
                    d for d in _divisors(g) if all(g % (d * nr) == 0 for nr in num_replicas_list)
                ]
                for e in e_cands:
                    for v in v_cands:
                        if checked >= max_candidates:
                            return best
                        checked += 1
                        if not _ok(nsb, nsr, p, e, v):
                            continue
                        dist = (
                            abs(nsb - b_nsb)
                            + abs(nsr - b_nsr)
                            + abs(p - b_p)
                            + abs(e - b_e)
                            + abs(v - b_v)
                        )
                        if dist < best_dist:
                            best_dist = dist
                            best = {
                                "num_samples_beta_vae": nsb,
                                "num_samples_rf": nsr,
                                "train_val_split": tvs,
                                "per_replica_batch_size": p,
                                "effective_batch_size": e,
                                "per_replica_val_batch_size": v,
                            }
    return best


def _round_to_multiple(val: int, divisor: int) -> int:
    """Round ``val`` to the nearest multiple of ``divisor`` (ties round up)."""
    q, r = divmod(val, divisor)
    return divisor * (q if r * 2 < divisor else q + 1)


def propose_simple_fix(err: ValidationError) -> object | None:
    """Suggest a single replacement value for a non-cross-param violation. Returns None when
    a violation cannot be auto-fixed (e.g. a missing file)."""
    if err.fix_kind == "clamp_low" and err.min_val is not None:
        return err.min_val
    if err.fix_kind == "clamp_high" and err.max_val is not None:
        return err.max_val
    if err.fix_kind == "range" and err.min_val is not None and err.max_val is not None:
        if isinstance(err.current, (int, float)):
            return max(err.min_val, min(err.current, err.max_val))
        return err.min_val
    if err.fix_kind == "enum" and err.allowed:
        return err.allowed[0]
    if err.fix_kind == "divisibility" and err.divisor and isinstance(err.current, int):
        return _round_to_multiple(err.current, err.divisor)
    return None  # format / file_exists / cross_param handled separately


def _format_proposal_value(val: object) -> str:
    """Format a proposed value for the CLI flag hint shown to the user."""
    if isinstance(val, float):
        return f"{val:g}"
    if isinstance(val, list):
        return "[" + ", ".join(str(x) for x in val) + "]"
    return str(val)


def _build_suggestion_block(errors: list[ValidationError], num_replicas: int | None) -> str:
    """Compose the 'Suggested fixes:' block appended to the ValueError message.

    Mirrors the proposer logic in utils/find_optimal_configs.py: simple violations get
    a per-field replacement value via :func:`propose_simple_fix`; cross-replica
    violations trigger a single grid search via :func:`_solve_cross_param_constraints`.
    Returns an empty string if nothing actionable was found.
    """
    config = get_config()
    simple = [e for e in errors if e.fix_kind != "cross_param"]
    cross = [e for e in errors if e.fix_kind == "cross_param"]
    # Keyed on (field, fix_kind): distinct violations of the SAME field (e.g. the
    # stamp_width equality and divisibility checks both firing) must not silently
    # overwrite each other's proposal — one flag line per violation is honest, even
    # when that repeats a flag with different candidate values.
    proposals: dict[tuple[str, str], object] = {}

    for err in simple:
        fix = propose_simple_fix(err)
        if fix is not None:
            proposals[(err.field, err.fix_kind)] = fix

    if cross and num_replicas is not None and config is not None:
        base = {
            "num_samples_beta_vae": config.training.num_samples_beta_vae,
            "num_samples_rf": config.training.num_samples_rf,
            "train_val_split": config.training.train_val_split,
            "per_replica_batch_size": config.training.per_replica_batch_size,
            "effective_batch_size": config.training.effective_batch_size,
            "per_replica_val_batch_size": config.training.per_replica_val_batch_size,
            "latent_total": config.training.latent_viz_num_cadences_per_type * 4,
        }
        solution = _solve_cross_param_constraints(base, [num_replicas])
        if solution is not None:
            for f_name in _CROSS_PARAM_FIELDS + ("train_val_split",):
                if solution[f_name] != base[f_name]:
                    proposals[(f"training.{f_name}", "cross_param")] = solution[f_name]

    if not proposals:
        return ""

    flag_lines: list[str] = []
    for (f_name, _fix_kind), value in proposals.items():
        # Drop the config-section prefix and switch underscores to hyphens.
        short = f_name.split(".", 1)[-1]
        flag = "--" + short.replace("_", "-")
        flag_lines.append(f"      {flag} {_format_proposal_value(value)}")

    body = " \\\n".join(flag_lines)
    return (
        "\n\nSuggested fixes (run `python utils/find_optimal_configs.py` for the full "
        "report):\n" + body
    )


def validate_args(args: argparse.Namespace) -> None:
    """
    Pre-flight semantic and cross-parameter validation for the parsed CLI namespace, run before
    apply_args_to_config(). Delegates to _resolve_num_replicas() (which also fails fast on a
    replica count that's unusable on this host) and collect_validation_errors(), and raises
    ValueError if any failures came back. On failure the ValueError message includes a list of
    suggested CLI flags from the same proposer surface that powers utils/find_optimal_configs.py.
    Syntax and type checks are handled earlier by argparse itself in ArgumentParser.parse_args
    (called from main.py:main).
    """
    num_replicas = _resolve_num_replicas(args)
    # NOTE: come back to this later (should we log a help message to use utils/find_optimal_configs.py if the inline `Suggested fixes: ` block isn't enough? alternatively, why don't we make it such that validate_args and the utility script share the exact same proposer surface -- e.g. bounded grid search on both, not just utility script. then, the purpose of the utility script would simply be to test hypothetical configurations? update docs/CONFIG_AND_CLI.md if any changes.)
    errors = collect_validation_errors(args, num_replicas)
    if errors:
        violation_block = "\n".join(f"  - {e.message}" for e in errors)
        suggestions = _build_suggestion_block(errors, num_replicas)
        raise ValueError(f"Invalid arguments detected:\n{violation_block}{suggestions}")
