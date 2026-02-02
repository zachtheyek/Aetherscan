# NOTE: is there a way to parallelize the CPU-GPU processing (e.g. while GPU is working on training/inference, CPU starts working on data_generation/preprocessing)
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

from aetherscan.cli import apply_args_to_config, setup_argument_parser, validate_args
from aetherscan.config import get_config, init_config
from aetherscan.db import init_db
from aetherscan.inference import run_inference_pipeline
from aetherscan.logger import init_logger
from aetherscan.manager import get_manager, init_manager, register_logger
from aetherscan.monitor import init_monitor
from aetherscan.preprocessing import DataPreprocessor
from aetherscan.train import get_latest_tag, run_training_pipeline

logger = logging.getLogger(__name__)


# NOTE: split this into strategy.py? have config option & cli flag to set single-GPU, multi-GPU, multi-node strategy. test thoroughly for each option
# NOTE: how to only run on specific GPUs, rather than all GPUs?
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
    logger.info(f"  Data path: {config.data_path}")
    logger.info(f"  Model path: {config.model_path}")
    logger.info(f"  Output path: {config.output_path}")
    logger.info(f"  Number of rounds: {config.training.num_training_rounds}")
    logger.info(f"  Epochs per round: {config.training.epochs_per_round}")

    # Setup GPU strategy
    try:
        strategy = setup_gpu_strategy()
    except Exception as e:
        logger.error(f"Failed to setup GPU strategy: {e}")
        sys.exit(1)

    # NOTE: come back to this later (test whether pipeline runs on <4, <6 GPUs, on single GPU, and on CPU. if all is robust, remove no strategy error)
    if strategy is None:
        logger.error("No GPU strategy available. Training requires GPU.")
        sys.exit(1)

    # Initialize preprocessor & load training backgrounds
    # Note, we load this in train_command() to avoid reloading backgrounds on training pipeline retries
    # This gives us faster startup times at the expense of holding onto more memory during training
    # Should be fine since backgrounds only take up low ~10^1 Gb in RAM (benchmarked: Dec '25)
    # However, if we decide to trade off reduced memory pressure for slower startup times in future,
    # then we should consider moving this into TrainingPipeline proper
    try:
        preprocessor = DataPreprocessor()
        background_data = preprocessor.load_train_data().astype(np.float32)
        # NOTE: do we need to close preprocessing pools and/or shared memory?
    except Exception as e:
        logger.error(f"Failed to load train backgrounds: {e}")
        sys.exit(1)

    # Train models with fault tolerance
    logger.info("Starting training pipeline...")

    max_retries = config.training.max_retries
    retry_delay = config.training.retry_delay

    for attempt in range(max_retries):
        pipeline = None

        try:
            logger.info(f"Training attempt: {attempt + 1}/{max_retries}")

            if attempt > 0:
                logger.info(f"Retrying training from round {config.checkpoint.start_round}")
            else:
                logger.info(f"Starting training from round {config.checkpoint.start_round}")

            # Reinitialize training pipeline on each attempt so no corrupted state is persisted
            pipeline = run_training_pipeline(background_data=background_data, strategy=strategy)

            break  # If we get here, training succeeded

        except KeyboardInterrupt:
            # Don't retry on user interruption
            # Re-raise to propagate traceback
            logger.info("Training interrupted by user")
            raise

        # NOTE: fault tolerance currently only accounts for beta-vae training failure. what about cases when train_random_forest fails and we wish to resume from there? should we add a check where if new round number is greater than specified rounds, skip directly to train RF? how would this look like? what about when save_models or plot_beta_vae_training_progress fails? should also add a db flag on training retry, such that we can easily filter out "corrupted" data when reading from db (e.g. for plotting). would need to return start_time so we can retain state on retries
        except Exception as e:
            logger.error(f"Training attempt {attempt + 1} failed with error: {e}")

            if attempt < max_retries - 1:
                # Retry training
                logger.info(
                    f"Attempting to recover from failure: attempt {attempt + 2}/{max_retries}"
                )

                # Collect garbage
                if pipeline:
                    del pipeline
                gc.collect()

                # Save original checkpoint values in case of failure recovery
                original_dir = config.checkpoint.load_dir
                original_tag = config.checkpoint.load_tag
                original_round = config.checkpoint.start_round

                try:
                    # Find the latest checkpoint & determine where to resume from
                    config.checkpoint.load_dir = "checkpoints"
                    config.checkpoint.load_tag = get_latest_tag(
                        os.path.join(config.model_path, config.checkpoint.load_dir)
                    )
                    if config.checkpoint.load_tag.startswith("round_"):
                        config.checkpoint.infer_start_round()
                    else:
                        raise ValueError("No valid checkpoints loaded")

                    logger.info(
                        f"Found latest checkpoint from round {config.checkpoint.start_round - 1}"
                    )
                    logger.info(f"Waiting {retry_delay} seconds before retry...")

                except Exception as recovery_error:
                    # If no checkpoints loaded, restart from last valid point
                    config.checkpoint.load_dir = original_dir
                    config.checkpoint.load_tag = original_tag
                    config.checkpoint.start_round = original_round

                    logger.error(f"Recovery failed: {recovery_error}")
                    logger.info(
                        f"Restarting training from round {config.checkpoint.start_round} in {retry_delay} seconds..."
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
    logger.info(f"Training configuration saved to {config_path}")

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


# NOTE: we need to load the saved config from the corresponding training run, but when/where should we do that, and how does that play with apply_args_to_config()?
def inference_command():
    """Execute inference pipeline with distributed strategy & fault tolerance"""
    logger.info("=" * 60)
    logger.info("Starting Aetherscan Inference Pipeline")
    logger.info("=" * 60)

    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")

    # Sanity check
    if not all(
        [
            config.inference.encoder_path,
            config.inference.rf_path,
            config.inference.config_path,
        ]
    ):
        logger.error("Encoder, RF, or config path not specified")
        sys.exit(1)

    # TODO: add a sanity check that verifies encoder, RF, and config path all have the same tag. throw a warning if false

    logger.info("Configuration:")
    logger.info(f"  Data path: {config.data_path}")
    logger.info(f"  Model path: {config.model_path}")
    logger.info(f"  Output path: {config.output_path}")
    logger.info(f"  Encoder path: {config.inference.encoder_path}")
    logger.info(f"  Random Forest path: {config.inference.rf_path}")
    logger.info(f"  Config path: {config.inference.config_path}")
    logger.info(f"  Files to process: {config.data.test_files}")
    logger.info(f"  Classification threshold: {config.inference.classification_threshold}")

    # Setup GPU strategy
    try:
        strategy = setup_gpu_strategy()
    except Exception as e:
        logger.error(f"Failed to setup GPU strategy: {e}")
        sys.exit(1)

    # NOTE: come back to this later (test whether pipeline runs on <4, <6 GPUs, on single GPU, and on CPU. if all is robust, remove no strategy error)
    if strategy is None:
        logger.error("No GPU strategy available. Inference requires GPU.")
        sys.exit(1)

    # NOTE: come back to this later (this is where we convert .h5 into .npy, and then downsample + log norm the .npy data)
    # NOTE: will this lead to us holding too much data in memory for the sake of inference fault tolerance? should we add some async/back-and-forth design patterns (e.g. preproc X files, inference X files, clear, repeat) to reduce memory pressure? is this the most efficient architecture we can use? add comments about memory/performance trade-offs once inference pipeline complete (see preproc section in train_command())
    # TODO: implement fault tolerance (should preproc have separate fault tolerance to inference?)
    # Initialize preprocessor & load inference cadences
    try:
        preprocessor = DataPreprocessor()
        cadence_data = preprocessor.load_inference_data().astype(np.float32)
        # NOTE: do we need to close preprocessing pools and/or shared memory?
    except Exception as e:
        logger.error(f"Failed to load inference cadences: {e}")
        sys.exit(1)

    # Run inference with fault tolerance
    logger.info("Starting inference pipeline...")

    # TODO: implement fault tolerance
    try:
        results = run_inference_pipeline(
            cadence_data=cadence_data,
            npy_path=config.data.test_files[0],  # TODO: Handle multiple test files properly
            strategy=strategy,
            # TODO: figure out how to pass preproc metadata into InferencePipeline (target, session, cadence_id, band, frequency_mhz, timestamp_observed, h5_path). should we roll these metadata + npy_path into a list/dict from preproc, then unroll them inside run_inference_pipeline()?
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)

    # NOTE: should we create dedicated (tagged) directories inside output_path to store inference results (plots, configs, etc.)? note, data still written to db regardless
    # Save inference configuration
    config_path = os.path.join(config.output_path, f"config_{config.checkpoint.save_tag}.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)  # Create dir if it doesn't exist

    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Inference configuration saved to {config_path}")

    logger.info("=" * 60)
    logger.info("Inference completed successfully!")
    logger.info("Summary:")
    logger.info(f"  Total cadence snippets: {results['n_cadence_snippets']}")
    logger.info(f"    Processed: {results['n_processed']}")
    logger.info(f"    Candidates found: {results['n_candidates']}")
    logger.info("=" * 60)


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
#         background_data = load_train_data(config)
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
            # Print help message & exit if parse_args() fails
            # Note, argparse prints its own error message, but we call print_help() again
            # just to be safe
            parser.print_help()
            logger.error("Invalid CLI arguments received")
            logger.error("See usage")
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

    try:
        # Execute command
        if args.command == "train":
            train_command()
        elif args.command == "inference":
            inference_command()
        # NOTE: come back to this later
        # elif args.command == 'evaluate':
        #     evaluate_command(args)
        else:
            # Print help message & exit if no valid command provided
            parser.print_help()
            logger.error("Invalid CLI command received")
            logger.error("See usage")
            sys.exit(1)

    finally:
        # Explicitly call cleanup_all() before exiting to avoid deadlock
        # Without this, non-daemon threads block sys.exit() from running atexit handlers (race condition)
        # NOTE: do the other sys.exit() calls in main.py get blocked by non-daemon threads as well?
        # BUG: sys.exit() calls DO get blocked. directly call manager.cleanup_all() instead. actually sometimes it works? further testing required
        manager = get_manager()  # NOTE: is this needed? since manager initialized in main?
        if manager:
            manager.cleanup_all()


if __name__ == "__main__":
    main()
