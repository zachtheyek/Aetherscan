"""
CLI argument parsing for Aetherscan Pipeline
"""

from __future__ import annotations

import argparse
import logging

from aetherscan.config import get_config

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Build the top-level Aetherscan argparse parser with `train` and `inference` subcommands."""
    parser = argparse.ArgumentParser(
        description="Aetherscan Pipeline -- Breakthrough Listen's first end-to-end production-grade DL pipeline for SETI @ scale"
    )

    # Add commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # NOTE: how does the pipeline know which one to use?
    # Train command
    _add_train_arguments(subparsers)
    # Inference command
    _add_inference_arguments(subparsers)

    return parser


# TODO: update descriptions
def _add_train_arguments(subparsers):
    """Add training command arguments to subparser"""
    train_parser = subparsers.add_parser("train", help="Execute training pipeline")

    # Path arguments (overrides environment variables)
    train_parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data directory (overrides AETHERSCAN_DATA_PATH environment variable)",
    )
    train_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model directory (overrides AETHERSCAN_MODEL_PATH environment variable)",
    )
    train_parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to output directory (overrides AETHERSCAN_OUTPUT_PATH environment variable)",
    )

    # BetaVAE model configuration
    train_parser.add_argument(
        "--vae-latent-dim",
        type=int,
        default=None,
        help="Dimensionality of the VAE latent space (bottleneck size)",
    )
    train_parser.add_argument(
        "--vae-dense-layer-size",
        type=int,
        default=None,
        help="Size of dense layer in VAE architecture (should match frequency bins after downsampling)",
    )
    train_parser.add_argument(
        "--vae-kernel-size",
        type=int,
        nargs=2,
        default=None,
        help="Kernel size for Conv2D layers as two integers (e.g., --vae-kernel-size 3 3)",
    )
    train_parser.add_argument(
        "--vae-beta",
        type=float,
        default=None,
        help="Beta coefficient for KL divergence loss term in beta-VAE (controls disentanglement)",
    )
    train_parser.add_argument(
        "--vae-alpha",
        type=float,
        default=None,
        help="Alpha coefficient for clustering loss term in VAE (controls cluster separation)",
    )

    # Random Forest configuration
    train_parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=None,
        help="Number of decision trees in the random forest ensemble",
    )
    train_parser.add_argument(
        "--rf-bootstrap",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Whether to use bootstrap sampling when building trees (enables bagging)",
    )
    train_parser.add_argument(
        "--rf-max-features",
        type=str,
        default=None,
        help="Number of features to consider for splits: 'sqrt', 'log2', or a float (fraction of features)",
    )
    train_parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for random forest training (-1 uses all CPU cores)",
    )
    train_parser.add_argument(
        "--rf-seed",
        type=int,
        default=None,
        help="Random seed for random forest reproducibility",
    )

    # GPU configuration
    train_parser.add_argument(
        "--gpu-memory-limit-mb",
        type=int,
        default=None,
        help="Per-GPU memory cap in MiB. Omit to use memory-growth-only (recommended on Blackwell). Set for TF to allocate a fixed logical device of a given size per physical GPU (e.g. 14000)",
    )
    train_parser.add_argument(
        "--nccl-num-packs",
        type=int,
        default=None,
        help="num_packs for NCCL/HierarchicalCopy all-reduce. Lower values (e.g. 1) reduces tiny-tensor latency; higher values (e.g. >=4) can help bandwidth on >4-GPU topologies.",
    )
    train_parser.add_argument(
        "--async-allocator",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle TF_GPU_ALLOCATOR=cuda_malloc_async (default: enabled). Pass --no-async-allocator as a workaround for NGC 25.02 multi-GPU OOM bugs.",
    )

    # Data configuration
    train_parser.add_argument(
        "--num-observations",
        type=int,
        default=None,
        help="Number of observations per cadence snippet (e.g., 6 for 3 ON + 3 OFF)",
    )
    train_parser.add_argument(
        "--width-bin",
        type=int,
        default=None,
        help="Number of frequency bins per observation (spectral resolution)",
    )
    train_parser.add_argument(
        "--downsample-factor",
        type=int,
        default=None,
        help="Downsampling factor for frequency bins (reduces spectral dimension)",
    )
    train_parser.add_argument(
        "--time-bins",
        type=int,
        default=None,
        help="Number of time bins per observation (temporal resolution)",
    )
    train_parser.add_argument(
        "--freq-resolution",
        type=float,
        default=None,
        help="Frequency resolution in Hz (determined by instrument)",
    )
    train_parser.add_argument(
        "--time-resolution",
        type=float,
        default=None,
        help="Time resolution in seconds (determined by instrument)",
    )
    train_parser.add_argument(
        "--num-target-backgrounds",
        type=int,
        default=None,
        help="Number of background (noise-only) cadences to load for training data generation",
    )
    train_parser.add_argument(
        "--background-load-chunk-size",
        type=int,
        default=None,
        help="Maximum number of background cadences to process at once during loading (memory management)",
    )
    train_parser.add_argument(
        "--max-chunks-per-file",
        type=int,
        default=None,
        help="Maximum number of chunks to load from a single data file (limits per-file contribution)",
    )
    train_parser.add_argument(
        "--train-files",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated list of training data file names (e.g., real_filtered_LARGE_HIP110750.npy)",
    )

    # Training configuration
    train_parser.add_argument(
        "--num-training-rounds",
        type=int,
        default=None,
        help="Total number of training rounds in curriculum learning schedule",
    )
    train_parser.add_argument(
        "--epochs-per-round",
        type=int,
        default=None,
        help="Number of epochs to train the VAE per curriculum learning round",
    )
    train_parser.add_argument(
        "--num-samples-beta-vae",
        type=int,
        default=None,
        # NOTE: divisible by 4 or num_replicas?
        help="Number of training samples to generate for beta-VAE per round (must be divisible by 4)",
    )
    train_parser.add_argument(
        "--num-samples-rf",
        type=int,
        default=None,
        # NOTE: divisible by 4 or num_replicas?
        help="Number of training samples to generate for random forest (must be divisible by 4)",
    )
    train_parser.add_argument(
        "--train-val-split",
        type=float,
        default=None,
        help="Fraction of data to use for training vs validation (e.g., 0.8 = 80%% train, 20%% val)",
    )
    train_parser.add_argument(
        "--per-replica-batch-size",
        type=int,
        default=None,
        help="Batch size per GPU/device replica during training",
    )
    train_parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=None,
        help="Effective batch size for gradient accumulation across all replicas",
    )
    train_parser.add_argument(
        "--per-replica-val-batch-size",
        type=int,
        default=None,
        help="Batch size per GPU/device replica during validation",
    )
    train_parser.add_argument(
        "--signal-injection-chunk-size",
        type=int,
        default=None,
        # NOTE: divisible by 4 or num_replicas?
        help="Maximum cadences to process at once during synthetic signal injection (must be divisible by 4)",
    )
    train_parser.add_argument(
        "--plot-injection-subsampling-count",
        type=int,
        default=None,
        help="Max points per stat name, per signal type, for A→B intensity bias scatter plots. Outliers are prioritized, with the difference made up from randomly sampling without replacement the remaining points",
    )
    train_parser.add_argument(
        "--plot-injection-outlier-percentile",
        type=float,
        default=None,
        help="Threshold for points to always be included in A→B intensity bias scatter plots",
    )
    train_parser.add_argument(
        "--latent-viz-num-cadences-per-type",
        type=int,
        default=None,
        help="Number of cadences per signal type for latent space visualization batch (total points = 4× this value × 6 observations per cadence)",
    )
    train_parser.add_argument(
        "--latent-viz-step-interval",
        type=int,
        default=None,
        help="Capture a latent space snapshot every N training steps (lower = more snapshots, more DB writes, and larger storage costs)",
    )
    train_parser.add_argument(
        "--latent-viz-umap-fit-max-samples",
        type=int,
        default=None,
        help="Maximum number of pooled latent vectors used to fit the UMAP model (remaining vectors are projected via transform; lower = faster, higher = more faithful embedding)",
    )
    train_parser.add_argument(
        "--latent-viz-umap-n-neighbors",
        type=int,
        nargs="+",
        default=None,
        help="UMAP n_neighbors values to sweep for latent space visualization (e.g., --latent-viz-umap-n-neighbors 5 15 30 50)",
    )
    train_parser.add_argument(
        "--latent-viz-umap-min-dist",
        type=float,
        nargs="+",
        default=None,
        help="UMAP min_dist values to sweep for latent space visualization (e.g., --latent-viz-umap-min-dist 0.0 0.1 0.5)",
    )
    train_parser.add_argument(
        "--latent-viz-gif-max-frames",
        type=int,
        default=None,
        help="Maximum number of frames in latent space GIF output (snapshots beyond this limit are log-subsampled, prioritizing earlier training steps)",
    )
    train_parser.add_argument(
        "--latent-viz-gif-duration-ms",
        type=int,
        default=None,
        help="Milliseconds per frame in latent space GIF output",
    )

    train_parser.add_argument(
        "--snr-base",
        type=int,
        default=None,
        help="Base signal-to-noise ratio for curriculum learning (minimum SNR difficulty level)",
    )
    train_parser.add_argument(
        "--initial-snr-range",
        type=int,
        default=None,
        help="SNR range for initial (easiest) training rounds (signals sampled from snr_base to snr_base + initial_snr_range)",
    )
    train_parser.add_argument(
        "--final-snr-range",
        type=int,
        default=None,
        help="SNR range for final (hardest) training rounds (signals sampled from snr_base to snr_base + final_snr_range). Ignored if only training for 1 round",
    )
    train_parser.add_argument(
        "--curriculum-schedule",
        type=str,
        default=None,
        help="Curriculum difficulty progression schedule: 'linear', 'exponential', or 'step'",
    )
    train_parser.add_argument(
        "--exponential-decay-rate",
        type=float,
        default=None,
        help="Decay rate for exponential curriculum schedule (must be negative; more negative = faster difficulty increase)",
    )
    train_parser.add_argument(
        "--step-easy-rounds",
        type=int,
        default=None,
        help="Number of rounds with easy signals when using step curriculum schedule",
    )
    train_parser.add_argument(
        "--step-hard-rounds",
        type=int,
        default=None,
        help="Number of rounds with hard signals when using step curriculum schedule",
    )
    train_parser.add_argument(
        "--base-learning-rate",
        type=float,
        default=None,
        help="Initial learning rate for Adam optimizer",
    )
    train_parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=None,
        help="Learning rate floor for adaptive learning rate reduction",
    )
    train_parser.add_argument(
        "--min-pct-improvement",
        type=float,
        default=None,
        help="Minimum fractional validation loss improvement to avoid LR reduction (e.g., 0.001 = 0.1%%)",
    )
    train_parser.add_argument(
        "--patience-threshold",
        type=int,
        default=None,
        help="Number of consecutive epochs without minimum improvement before reducing learning rate",
    )
    train_parser.add_argument(
        "--lr-reduction-factor",
        type=float,
        default=None,
        help="Multiplicative factor for learning rate reduction (e.g., 0.2 reduces LR by 20%%)",
    )
    train_parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum number of retry attempts when training fails due to errors",
    )
    train_parser.add_argument(
        "--retry-delay",
        type=int,
        default=None,
        help="Delay in seconds between retry attempts after training failure",
    )

    # Checkpoint configuration
    train_parser.add_argument(
        "--load-dir",
        type=str,
        default=None,
        help="Subdirectory for checkpoint loading (relative to --model-path)",
    )
    train_parser.add_argument(
        "--load-tag",
        type=str,
        default=None,
        help="Model tag for checkpoint loading. Accepted formats: final_vX, round_XX, YYYYMMDD_HHMMSS, test_vX. If round_XX format used, and --start-round not specified, training will resume from round following loaded checkpoint (i.e., XX + 1)",
    )
    train_parser.add_argument(
        "--start-round",
        type=int,
        default=None,
        help="Round to begin/resume training from",
    )
    train_parser.add_argument(
        "--save-tag",
        type=str,
        default=None,
        help="Tag for current pipeline run. Accepted formats: final_vX, round_XX, test_vX. Current timestamp used (YYYYMMDD_HHMMSS) if none specified",
    )


# TODO: update descriptions
def _add_inference_arguments(subparsers):
    """Add inference command arguments to subparser"""
    inf_parser = subparsers.add_parser("inference", help="Execute inference pipeline")

    # Path arguments
    inf_parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data directory (overrides AETHERSCAN_DATA_PATH environment variable)",
    )
    inf_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model directory (overrides AETHERSCAN_MODEL_PATH environment variable)",
    )
    inf_parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to output directory (overrides AETHERSCAN_OUTPUT_PATH environment variable)",
    )

    # GPU configuration
    inf_parser.add_argument(
        "--gpu-memory-limit-mb",
        type=int,
        default=None,
        help="Per-GPU memory cap in MiB. Omit to use memory-growth-only (recommended on Blackwell). Set for TF to allocate a fixed logical device of a given size per physical GPU (e.g. 14000)",
    )
    inf_parser.add_argument(
        "--async-allocator",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle TF_GPU_ALLOCATOR=cuda_malloc_async (default: enabled). Pass --no-async-allocator as a workaround for NGC 25.02 multi-GPU OOM bugs.",
    )

    # Data configuration
    inf_parser.add_argument(
        "--test-files",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated list of testing data file names (e.g., real_filtered_LARGE_test_HIP15638.npy)",
    )
    inf_parser.add_argument(
        "--inference-files",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated list of inference catalog file names (e.g. complete_cadences_catalog.csv). Expects .h5 filepaths to individual observations, and sufficient metadata for recovering cadence groupings. If provided, triggers the energy detection preprocessing pipeline and takes precedence over --test-files",
    )

    # Inference configuration
    inf_parser.add_argument(
        "--encoder-path",
        type=str,
        default=None,
        help="Path to trained VAE encoder model file (.keras)",
    )
    inf_parser.add_argument(
        "--rf-path",
        type=str,
        default=None,
        help="Path to trained Random Forest model file (.joblib)",
    )
    inf_parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config file from corresponding training run (.json)",
    )
    inf_parser.add_argument(
        "--per-replica-batch-size",
        type=int,
        default=None,
        help="Batch size per GPU/device replica during inference",
    )
    inf_parser.add_argument(
        "--classification-threshold",
        type=float,
        default=None,
        help="Classification threshold for candidate detection",
    )

    # Energy detection preprocessing
    inf_parser.add_argument(
        "--cadence-group-by-cols",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated list of CSV column names whose joint value defines cadence membership (e.g., Target Session Band 'Cadence ID' Frequency)",
    )
    inf_parser.add_argument(
        "--cadence-h5-path-col",
        type=str,
        default=None,
        help="CSV column containing the .h5 file path for each observation (default: '.h5 path')",
    )
    inf_parser.add_argument(
        "--cadence-expected-obs",
        type=int,
        default=None,
        help="Required number of observations per cadence (default: 6 for ABACAD)",
    )
    inf_parser.add_argument(
        "--coarse-channel-width",
        type=int,
        default=None,
        help="Number of fine channels per coarse channel (default: 1048576)",
    )
    inf_parser.add_argument(
        "--parallel-coarse-chans",
        type=int,
        default=None,
        help="Number of coarse channels to process in parallel per block (default: 28)",
    )
    inf_parser.add_argument(
        "--spline-order",
        type=int,
        default=None,
        help="Spline order for bandpass fitting (default: 16)",
    )
    inf_parser.add_argument(
        "--detection-window-size",
        type=int,
        default=None,
        help="Sliding window size in fine channels for normality test (default: 256)",
    )
    inf_parser.add_argument(
        "--detection-step-size",
        type=int,
        default=None,
        help="Step size in fine channels for sliding window (default: 128)",
    )
    inf_parser.add_argument(
        "--stat-threshold",
        type=float,
        default=None,
        help="D'Agostino-Pearson statistic threshold for hit detection (default: 2048.0)",
    )
    inf_parser.add_argument(
        "--stamp-width",
        type=int,
        default=None,
        help="Width in fine channels of the extracted stamp around each hit (default: 4096; must equal --width-bin)",
    )
    inf_parser.add_argument(
        "--overlap-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Additionally extract stamps offset by ±overlap_fraction*stamp_width around each hit. Pass --no-overlap-search to disable when the config default is True.",
    )
    inf_parser.add_argument(
        "--overlap-fraction",
        type=float,
        default=None,
        help="Fractional offset (relative to stamp_width) for overlap-search stamps (default: 0.5)",
    )
    inf_parser.add_argument(
        "--preprocess-output-dir",
        type=str,
        default=None,
        help="Directory for per-cadence .npy outputs from preprocessing",
    )
    inf_parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum number of retry attempts for inference (including preprocessing) on failure",
    )
    inf_parser.add_argument(
        "--retry-delay",
        type=int,
        default=None,
        help="Delay in seconds between inference retry attempts",
    )

    # Checkpoint configuration
    inf_parser.add_argument(
        "--save-tag",
        type=str,
        default=None,
        help="Tag for current pipeline run. Current timestamp used (YYYYMMDD_HHMMSS) if none specified",
    )


# NOTE: how to ensure only train_parser/inf_parser args get applied depending on whether train/inference is ran?
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
        config.rf.seed = args.rf_seed

    # GPU configuration
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

    # Checkpoint configuration
    if hasattr(args, "load_dir") and args.load_dir is not None:
        config.checkpoint.load_dir = args.load_dir
    if hasattr(args, "load_tag") and args.load_tag is not None:
        config.checkpoint.load_tag = args.load_tag
        config.checkpoint.infer_start_round()  # Try inferring start_round from load_tag first
    if hasattr(args, "start_round") and args.start_round is not None:
        config.checkpoint.start_round = args.start_round  # Override start_round if provided
    if hasattr(args, "save_tag") and args.save_tag is not None:
        config.checkpoint.save_tag = args.save_tag

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
    if hasattr(args, "parallel_coarse_chans") and args.parallel_coarse_chans is not None:
        config.inference.parallel_coarse_chans = args.parallel_coarse_chans
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


# NOTE: should we change validate_args() to validate_config() and run the tests on the final "applied" config singleton? or should we check both separately?
def validate_args(args: argparse.Namespace) -> None:
    """
    Run semantic and cross-param validation on the parsed CLI namespace before it is applied to
    the config — checks divisibility/range constraints, A-requires-B relationships, and
    existence of referenced files. Syntax and type checks are not handled here; those run
    earlier in parser.parse_args() (called from main.py:main).
    """
    errors = []

    # NOTE: comment out temporarily for bla0
    # # Check: signal-injection-chunk-size must be divisible by 4 (for balanced class generation)
    # if (
    #     hasattr(args, "signal_injection_chunk_size")
    #     and args.signal_injection_chunk_size is not None
    #     and args.signal_injection_chunk_size % 4 != 0
    # ):
    #     errors.append(
    #         f"--signal-injection-chunk-size must be divisible by 4 for balanced class generation, "
    #         f"got {args.signal_injection_chunk_size}"
    #     )
    #
    # # Check: num-samples-beta-vae must be divisible by 4 (for balanced class generation)
    # if (
    #     hasattr(args, "num_samples_beta_vae")
    #     and args.num_samples_beta_vae is not None
    #     and args.num_samples_beta_vae % 4 != 0
    # ):
    #     errors.append(
    #         f"--num-samples-beta-vae must be divisible by 4 for balanced class generation, "
    #         f"got {args.num_samples_beta_vae}"
    #     )
    #
    # # Check: num-samples-rf must be divisible by 4 (for balanced class generation)
    # if (
    #     hasattr(args, "num_samples_rf")
    #     and args.num_samples_rf is not None
    #     and args.num_samples_rf % 4 != 0
    # ):
    #     errors.append(
    #         f"--num-samples-rf must be divisible by 4 for balanced class generation, "
    #         f"got {args.num_samples_rf}"
    #     )

    # TODO: come back to this later
    # NOTE: double check if these are correct
    # rf_max_features, curriculum_schedule, load_tag, save_tag not following the accepted formats
    # make sure save tag is unique (check db? is db initialized before validate_args()?)
    # vae_dense_layer_size = freq_resolution // downsample_factor
    # -1 <= rf_n_jobs <= num_cores
    # time_bin & width bin match data (randomly sample a few files)
    # train_files & test_files exist
    # 0 <= train_val_split <= 1
    # per_replica_batch_size * num_replicas <= effective_batch_size <= num_samples_beta_vae * train_val_split
    # per_replica_val_batch_size * num_replicas <= num_samples_beta_vae * (1 - train_val_split)
    # per_replica_val_batch_size * num_replicas <= num_samples_rf
    # latent_viz_num_cadences_per_type * 4 <= num_samples_beta_vae * (1 - train_val_split)
    # effective_batch_size is divisible by per_replica_batch_size * num_replicas
    # num_samples_beta_vae * train_val_split is divisible by effective_batch_size
    # num_samples_beta_vae * (1 - train_val_split) is divisible by per_replica_val_batch_size * num_replicas
    # num_samples_rf is divisible by per_replica_val_batch_size * num_replicas
    # num_samples_rf is divisible by 2 (for generate_triplet_batch)
    # latent_viz_num_cadences_per_type * 4 is divisible by per_replica_val_batch_size * num_replicas
    # # how to ensure divisibility for actual n_samples during inference?
    # snr_base, initial_snr_range, final_snr_range > 0
    # initial_snr_range >= final_snr_range
    # exponential_decay_rate < 0
    # 0 <= step_easy_rounds, step_hard_rounds <= num_training_rounds
    # step_easy_rounds + step_hard_rounds = num_training_rounds
    # base_learning_rate >= min_learning_rate
    # min_pct_improvement >= 0
    # patience_threshold >= 0 (1?)
    # lr_reduction_factor > 0
    # max_retries >= 0
    # retry_delay >= 0
    # start_round < num_training_rounds
    # do directories specified have to exist? or we assume we'll create them on the fly and handle errors in-flight? (currently train_command, TrainingPipeline(), and load_models() all attempt to handle errors on the fly. is this optimal behavior?)
    # add tests for other configs (inference, db, monitor, etc.)
    #
    # Template for adding more checks:
    # if hasattr(args, "some_param") and args.some_param is not None:
    #     if <validation_condition>:
    #         errors.append("Error message explaining the problem")

    # NOTE: come back to this later (are runtime checks properly implemented? does config defaults kick in before cli args are applied? if some but not all csvs are invalid, do we still process valid csvs & throw errors for the invalid ones with proper checkpointing, or do we fail loudly & immediately? are these checks comprehensive for energy detection + inference modules, e.g. overlap_fraction between 0 & 1?)
    # # Energy detection preprocessing checks
    # # If inference_files is provided, cadence_group_by_cols must be non-empty.
    # # Per-CSV column-existence checks are deferred to runtime (CSVs aren't loaded here).
    # # stamp_width == width_bin is validated at runtime in inference_command, where both
    # # are resolved together.
    # if hasattr(args, "inference_files") and args.inference_files is not None:
    #     group_cols = getattr(args, "cadence_group_by_cols", None)
    #     # If the user didn't pass --cadence-group-by-cols, the dataclass default kicks in
    #     # (a non-empty list). Only fail if the user explicitly provided an empty list.
    #     if group_cols is not None and len(group_cols) == 0:
    #         errors.append(
    #             "--cadence-group-by-cols must be non-empty when --inference-files is provided"
    #         )
    #
    # if hasattr(args, "detection_window_size") and args.detection_window_size is not None:
    #     stamp_width = getattr(args, "stamp_width", None)
    #     if stamp_width is not None and args.detection_window_size > stamp_width:
    #         errors.append(
    #             f"--detection-window-size ({args.detection_window_size}) must be "
    #             f"<= --stamp-width ({stamp_width})"
    #         )

    # Throw an error if any validation fails
    if errors:
        raise ValueError("Invalid arguments detected:\n" + "\n".join(f"  - {e}" for e in errors))
