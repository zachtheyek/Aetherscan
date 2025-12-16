"""
Entry point for Aetherscan Pipeline
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf

from cli import apply_args_to_config, setup_argument_parser, validate_args
from config import get_config, init_config
from db import init_db
from logger import init_logger
from manager import init_manager, register_logger
from monitor import init_monitor
from preprocessing import DataPreprocessor
from train import get_latest_tag, train_full_pipeline

logger = logging.getLogger(__name__)


def setup_gpu_strategy():
    """Configure GPU memory growth, memory limits, multi-GPU strategy with load balancing & async allocator"""

    os.environ["TF_GPU_ALLOCATOR"] = (
        "cuda_malloc_async"  # Prevent memory fragmentation within each GPU
    )
    os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"] = (
        "true"  # Aggressive cleanup of intermediate tensors
    )

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Set equal memory limits for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=14000)
                    ],  # 14GiB limit per GPU
                )

            # Set distributed strategy to prevent uneven GPU memory usage
            try:
                # Primary choice: NCCL for NVIDIA GPUs
                strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.NcclAllReduce(num_packs=2)
                )
                logger.info("Using NcclAllReduce for optimal NVIDIA GPU performance")

            except Exception as e:
                # Fallback: HierarchicalCopyAllReduce
                logger.warning(f"NCCL failed ({e}), using HierarchicalCopyAllReduce")
                strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=2)
                )

            logger.info(f"Distributed strategy: {strategy.num_replicas_in_sync} GPUs")
            return strategy

        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
            return None

    else:
        logger.warning("No GPUs detected, running on CPU")
        return None


def train_command():
    """Execute training pipeline with distributed strategy & fault tolerance"""
    logger.info("=" * 60)
    logger.info("Starting Aetherscan Training Pipeline")
    logger.info("=" * 60)

    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")

    logger.info("Configuration:")
    logger.info(f"  Number of rounds: {config.training.num_training_rounds}")
    logger.info(f"  Epochs per round: {config.training.epochs_per_round}")
    logger.info(f"  Data path: {config.data_path}")
    logger.info(f"  Model path: {config.model_path}")
    logger.info(f"  Output path: {config.output_path}")

    # Setup GPU strategy
    try:
        strategy = setup_gpu_strategy()
    except Exception as e:
        logger.error(f"Failed to setup GPU strategy: {e}")
        sys.exit(1)

    # Initialize preprocessor & load background data
    # Note, we load this in train_command() to avoid reloading backgrounds on training pipeline retries
    # This gives us faster startup times at the expense of holding onto more memory during training
    # Should be fine since backgrounds only take up low ~10^1 Gb in RAM (benchmarked: Dec '25)
    # However, if we decide to trade off reduced memory pressure for slower startup times in future,
    # then we should consider moving this into TrainingPipeline proper
    try:
        preprocessor = DataPreprocessor()
        background_data = preprocessor.load_background_data().astype(np.float32)
        # NOTE: close preprocessing pools and/or shared memory?
    except Exception as e:
        logger.error(f"Failed to load backgrounds: {e}")
        sys.exit(1)

    # Train models with fault tolerance
    logger.info("Starting training pipeline...")

    max_retries = config.training.max_retries
    retry_delay = config.training.retry_delay

    for attempt in range(max_retries):
        try:
            logger.info(f"Training attempt: {attempt + 1}/{max_retries}")

            if attempt > 0:
                logger.info(f"Retrying training from round {config.checkpoint.start_round}")

            # Reinitialize training pipeline on each attempt so no corrupted state is persisted
            train_full_pipeline(background_data=background_data, strategy=strategy)

            break  # If we get here, training succeeded

        except KeyboardInterrupt:
            # Don't retry on user interruption
            # Re-raise to propagate traceback
            logger.info("Training interrupted by user")
            raise

        except Exception as e:
            logger.error(f"Training attempt {attempt + 1} failed with error: {e}")

            if attempt < max_retries - 1:
                # Retry training
                logger.info(
                    f"Attempting to recover from failure: attempt {attempt + 2}/{max_retries}"
                )

                try:
                    # Collect garbage
                    gc.collect()

                    # Find the latest checkpoint & determine where to resume from
                    config.checkpoint.load_dir = "checkpoints"
                    config.checkpoint.load_tag = get_latest_tag(
                        os.path.join(config.model_path, config.checkpoint.load_dir)
                    )
                    config.checkpoint.infer_start_round()

                    if not config.checkpoint.load_tag.startswith("round_"):
                        raise ValueError("No valid checkpoints loaded")

                    logger.info(
                        f"Found latest checkpoint from round {config.checkpoint.start_round - 1}"
                    )
                    logger.info(f"Waiting {retry_delay} seconds before retry...")

                except Exception as recovery_error:
                    # If no checkpoints loaded, restart from last valid start_round
                    logger.error(f"Recovery failed: {recovery_error}")
                    logger.info(
                        f"Restarting training from round {config.checkpoint.start_round} in {retry_delay} seconds..."
                    )
                    config.checkpoint.load_dir = (
                        None  # Reset to prevent TrainingPipeline() from loading phantom checkpoints
                    )

                finally:
                    time.sleep(retry_delay)

            else:
                # Max retries exceeded
                logger.error(f"Training attempts exceeded maximum retries ({max_retries})")
                logger.error(f"Final error: {e}")
                sys.exit(1)

    # Save training configuration
    config_path = os.path.join(config.model_path, f"config_{config.checkpoint.save_tag}.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)  # Create dir if it doesn't exist

    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Configuration saved to {config_path}")

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


# NOTE: come back to this later
# def inference_command(args):
#     """Execute inference command"""
#     logger.info("Starting inference pipeline...")
#
#     # Setup GPU
#     setup_gpu_config()
#
#     # Load configuration
#     config = Config()
#
#     # Load saved config if provided
#     if args.config:
#         with open(args.config) as f:
#             saved_config = json.load(f)
#             # Update config with saved values
#             for section_key, section_value in saved_config.items():
#                 if hasattr(config, section_key) and isinstance(section_value, dict):
#                     for key, value in section_value.items():
#                         if hasattr(getattr(config, section_key), key):
#                             setattr(getattr(config, section_key), key, value)
#
#     # Prepare observation files
#     observation_files = []
#
#     # Check for prepared test cadences
#     test_dir = os.path.join(config.data_path, "testing", "prepared_cadences")
#     if os.path.exists(test_dir):
#         # Load prepared cadences
#         for cadence_idx in range(args.n_bands):
#             cadence_files = []
#             for obs_idx in range(6):
#                 obs_file = os.path.join(test_dir, f"cadence_{cadence_idx:04d}_obs_{obs_idx}.npy")
#                 if os.path.exists(obs_file):
#                     cadence_files.append(obs_file)
#
#             if len(cadence_files) == 6:
#                 observation_files.append(cadence_files)
#
#     if not observation_files:
#         logger.error("No observation files found. Please prepare test data first.")
#         sys.exit(1)
#
#     logger.info(f"Found {len(observation_files)} cadences for inference")
#
#     # Run inference
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_path = args.output or f"/outputs/seti/detections_{timestamp}.csv"
#
#     results = run_inference(config, observation_files, args.vae_model, args.rf_model, output_path)
#
#     logger.info(f"Inference completed. Results saved to {output_path}")
#
#     # Print summary
#     if results is not None and not results.empty:
#         n_total = len(results)
#         n_high_conf = len(results[results["confidence"] > 0.9])
#         logger.info(f"Total detections: {n_total}")
#         logger.info(f"High confidence (>90%): {n_high_conf}")


# NOTE: come back to this later
# def evaluate_command(args):
#     """Execute evaluation command"""
#     logger.info("Starting model evaluation...")
#
#     # Setup GPU
#     setup_gpu_config()
#
#     # Load configuration
#     config = Config()
#
#     # Import necessary modules
#     import tensorflow as tf
#     from models.random_forest import RandomForestModel
#
#     # Load models
#     logger.info(f"Loading VAE encoder from {args.vae_model}")
#     vae_encoder = tf.keras.models.load_model(args.vae_model)
#
#     logger.info(f"Loading Random Forest from {args.rf_model}")
#     rf_model = RandomForestModel(config)
#     rf_model.load(args.rf_model)
#
#     # Load or generate test data
#     if args.test_data:
#         logger.info(f"Loading test data from {args.test_data}")
#         test_data = np.load(args.test_data, allow_pickle=True).item()
#     else:
#         logger.info("Generating synthetic test data...")
#         # Load some background for generation
#         background_data = load_background_data(config)
#         generator = DataGenerator(config, background_data[:100])  # Use subset
#         test_data = generator.generate_test_set()
#
#     # Evaluate
#     preprocessor = DataPreprocessor(config)
#
#     # Prepare test data
#     test_true = preprocessor.prepare_batch(test_data['true'])
#     test_false = preprocessor.prepare_batch(test_data['false'])
#
#     # Get predictions
#     _, _, true_latents = vae_encoder.predict(test_true, batch_size=64)
#     _, _, false_latents = vae_encoder.predict(test_false, batch_size=64)
#
#     true_preds = rf_model.predict(true_latents)
#     false_preds = rf_model.predict(false_latents)
#
#     # Calculate metrics
#     tpr = np.mean(true_preds == 1)
#     fpr = np.mean(false_preds == 1)
#     accuracy = np.mean(np.concatenate([true_preds == 1, false_preds == 0]))
#
#     logger.info("="*60)
#     logger.info("Evaluation Results:")
#     logger.info(f"  True Positive Rate: {tpr:.3f}")
#     logger.info(f"  False Positive Rate: {fpr:.3f}")
#     logger.info(f"  Overall Accuracy: {accuracy:.3f}")
#     logger.info("="*60)


def main():
    """Main entry point to Aetherscan pipeline"""
    # Initialize config
    try:
        init_config()
    except Exception as e:
        # Note, can't log before init_logger()
        sys.exit(1)

    # Initialize logger
    try:
        init_logger()
        logger.info("Logger initialization successful, but not yet registered for cleanup.")
        logger.info("Awaiting resource manager initialization. Do not terminate the process!")
    except Exception as e:
        # Note, can't log if init_logger() fails
        sys.exit(1)

    # Initialize resource manager
    try:
        init_manager()
        logger.info("Resource manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize resource manager: {e}")
        sys.exit(1)

    # Register logger
    try:
        register_logger()
        logger.info("Logger registered successfully")
    except Exception as e:
        logger.error(f"Failed to register logger: {e}")
        sys.exit(1)

    # Setup CLI argument parser
    try:
        parser = setup_argument_parser()
        logger.info("CLI argument parser setup successfully")
    except Exception as e:
        logger.error(f"Failed to setup argument parser: {e}")
        sys.exit(1)

    # Parse arguments
    try:
        args = parser.parse_args()
        logger.info("CLI arguments parsed. No issues found")
    except SystemExit as e:
        # argparse calls sys.exit(2) on parse errors (invalid types, missing required args, etc.)
        # which is why we catch with SystemExit instead of Exception
        # Exit code 2 = command line syntax error (standard for CLI tools)
        if e.code == 2:  # argparse error (syntax/type error)
            logger.error("Invalid CLI arguments received")
            logger.error("See usage")
            # Note, argparse already prints error message, no need to print_help() again
        # Here, we simply let the original traceback propagate by re-raising
        # so cleanup handlers still run via atexit
        raise

    # Validate arguments (handles everything else parse_args() missed)
    try:
        validate_args(args)
        logger.info("CLI arguments validated. No issues found")
    except Exception as e:
        # Print help message & exit if validate_args() fails
        parser.print_help()
        logger.error(f"Invalid CLI arguments received: {e}")
        logger.error("See usage")
        sys.exit(1)

    # Override default config values with CLI arguments
    try:
        apply_args_to_config(args)
        logger.info("CLI arguments applied successfully")
    except Exception as e:
        logger.error(f"Failed to apply CLI args: {e}")
        sys.exit(1)

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

    # Initialize resource monitoring
    try:
        init_monitor()
        logger.info("Resource monitor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize resource monitor: {e}")
        sys.exit(1)

    # Execute command
    if args.command == "train":
        train_command()
    # NOTE: come back to this later
    # elif args.command == "inference":
    #     inference_command(args)
    # NOTE: come back to this later
    # elif args.command == 'evaluate':
    #     evaluate_command(args)
    else:
        # Print help message & exit if no valid command provided
        parser.print_help()
        logger.error("Invalid CLI command received")
        logger.error("See usage")
        sys.exit(1)


if __name__ == "__main__":
    main()
