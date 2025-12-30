"""
CLI argument parsing for Aetherscan Pipeline
"""

from __future__ import annotations

import argparse
import logging

from aetherscan.config import get_config

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for Aetherscan CLI

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Aetherscan Pipeline -- Breakthrough Listen's first end-to-end production-grade DL pipeline for SETI @ scale"
    )

    # Add commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Train command
    _add_train_arguments(subparsers)
    # Inference command
    _add_inference_arguments(subparsers)
    # Evaluate command
    _add_evaluate_arguments(subparsers)

    return parser


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
    train_parser.add_argument(
        "--test-files",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated list of testing data file names (e.g., real_filtered_LARGE_test_HIP15638.npy)",
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
        "--num-samples-vae",
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
        "--global-batch-size",
        type=int,
        default=None,
        help="Effective global batch size for gradient accumulation across all replicas",
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
        help="Model tag for checkpoint loading. Accepted formats: final_vX, round_XX, YYYYMMDD_HHMMSS, test_vX. If round_XX format used, and --start-round not specified, training will resume from round proceeding loaded checkpoint (i.e., XX + 1)",
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


# NOTE: come back to this later
def _add_inference_arguments(subparsers):
    """Add inference command arguments to subparser"""
    inf_parser = subparsers.add_parser("inference", help="Execute inference pipeline")

    # TODO: finish adding inference_command args
    # inf_parser.add_argument('vae_model', type=str, help='Path to VAE encoder model')
    # inf_parser.add_argument('rf_model', type=str, help='Path to Random Forest model')
    # inf_parser.add_argument('--config', type=str, help='Path to saved config file')
    # inf_parser.add_argument('--n-bands', type=int, default=16,
    #                       help='Number of frequency bands to process')
    # inf_parser.add_argument('--output', type=str, help='Output file path')
    # ...


# NOTE: come back to this later
def _add_evaluate_arguments(subparsers):
    """Add evaluation command arguments to subparser"""
    eval_parser = subparsers.add_parser("evaluate", help="Execute evaluation pipeline")

    # TODO: finish adding evaluate_command args
    # eval_parser.add_argument('vae_model', type=str, help='Path to VAE encoder model')
    # eval_parser.add_argument('rf_model', type=str, help='Path to Random Forest model')
    # eval_parser.add_argument('--test-data', type=str, help='Path to test data')
    # ...


def apply_args_to_config(args: argparse.Namespace) -> None:
    """
    Override config values with CLI arguments

    Args:
        args: Parsed CLI arguments
    """
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

    # Training configuration
    if hasattr(args, "num_training_rounds") and args.num_training_rounds is not None:
        config.training.num_training_rounds = args.num_training_rounds
    if hasattr(args, "epochs_per_round") and args.epochs_per_round is not None:
        config.training.epochs_per_round = args.epochs_per_round
    if hasattr(args, "num_samples_vae") and args.num_samples_vae is not None:
        config.training.num_samples_beta_vae = args.num_samples_vae
    if hasattr(args, "num_samples_rf") and args.num_samples_rf is not None:
        config.training.num_samples_rf = args.num_samples_rf
    if hasattr(args, "train_val_split") and args.train_val_split is not None:
        config.training.train_val_split = args.train_val_split
    if hasattr(args, "per_replica_batch_size") and args.per_replica_batch_size is not None:
        config.training.per_replica_batch_size = args.per_replica_batch_size
    if hasattr(args, "global_batch_size") and args.global_batch_size is not None:
        config.training.global_batch_size = args.global_batch_size
    if hasattr(args, "per_replica_val_batch_size") and args.per_replica_val_batch_size is not None:
        config.training.per_replica_val_batch_size = args.per_replica_val_batch_size
    if (
        hasattr(args, "signal_injection_chunk_size")
        and args.signal_injection_chunk_size is not None
    ):
        config.training.signal_injection_chunk_size = args.signal_injection_chunk_size
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
    if hasattr(args, "max_retries") and args.max_retries is not None:
        config.training.max_retries = args.max_retries
    if hasattr(args, "retry_delay") and args.retry_delay is not None:
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

    # TODO: finish adding args for inference & evaluate
    # ...


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate parsed arguments are compatible with downstream pipeline

    Does handle:
      - logic constraints (e.g., divisible by X, within range Y, etc.)
      - cross-param relationships (e.g., A requires B, etc.)
      - semantic validation (e.g., file exists, etc.)

    Does not handle:
      - syntax & type validity (handled by parser.parse_args() in main.py:main())

    Args:
        args: Parsed CLI arguments
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
    # # Check: num-samples-vae must be divisible by 4 (for balanced class generation)
    # if (
    #     hasattr(args, "num_samples_vae")
    #     and args.num_samples_vae is not None
    #     and args.num_samples_vae % 4 != 0
    # ):
    #     errors.append(
    #         f"--num-samples-vae must be divisible by 4 for balanced class generation, "
    #         f"got {args.num_samples_vae}"
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
    # double check if these are correct
    # rf_max_features, curriculum_schedule, load_tag, save_tag not following the accepted formats
    # vae_dense_layer_size = freq_resolution // downsample_factor
    # -1 <= rf_n_jobs <= num_cores
    # time_bin & width bin match data (randomly sample a few files)
    # train_files & test_files exist
    # 0 <= train_val_split <= 1
    # per_replica_batch_size <= global_batch_size <= num_samples_vae * train_val_split
    # per_replica_val_batch_size <= num_samples_vae * (1 - train_val_split)
    # per_replica_val_batch_size <= num_samples rf
    # snr_base, initial_snr_range, final_snr_range > 0
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
    #
    # Template for adding more checks:
    # if hasattr(args, "some_param") and args.some_param is not None:
    #     if <validation_condition>:
    #         errors.append("Error message explaining the problem")

    # Throw an error if any validation fails
    if errors:
        raise ValueError("Invalid arguments detected:\n" + "\n".join(f"  - {e}" for e in errors))
