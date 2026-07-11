# NOTE: remove time filters and strictly query db using tag. would need to ensure save-tag is unique at the start of each run
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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv

from aetherscan.cli import (
    apply_args_to_config,
    apply_saved_config,
    setup_argument_parser,
    validate_args,
)
from aetherscan.config import get_config, init_config
from aetherscan.db import init_db
from aetherscan.inference import InferencePipeline, run_inference_pipeline
from aetherscan.logger import init_logger
from aetherscan.manager import get_manager, init_manager, register_logger
from aetherscan.monitor import init_monitor
from aetherscan.preprocessing import DataPreprocessor, derive_cadence_provenance
from aetherscan.train import run_training_pipeline

logger = logging.getLogger(__name__)


# NOTE: verify that our current GPU config gracefully handles cases where the node has a single GPU (vs multiple)
# TODO: run performance benchmarks using different num_gpus on a single node (and in future, multi-node as well)
# TODO: add a way to specify (either number or name) the specific GPUs on a system we wish to use (currently defaults to all available). extend to cli.py too
# TEST: make sure this works, especially with _build_optimizer() in train.py (do they conflict? is one unnecessary vs the other?)
def _warmup_collective(strategy):
    """Trigger a tiny cross-device reduction to surface NCCL failures at setup time.

    MirroredStrategy construction with NcclAllReduce never fails on its own — NCCL
    errors only surface on the first actual collective. Doing a 1-element reduce
    here lets us catch (and fall back from) NCCL failures before training starts,
    rather than mid-epoch.
    """

    @tf.function
    def _per_replica():
        return tf.constant(1.0)

    per_replica_value = strategy.run(_per_replica)
    _ = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_value, axis=None)


def setup_gpu_strategy():
    """Configure GPU memory growth, memory limits, multi-GPU strategy with load balancing & async allocator"""

    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")

    # Both env vars here are read lazily by TF when the GPU runtime first initializes (that is,
    # on first GPU memory allocation), which happens below on set_memory_growth()
    if config.gpu.use_async_allocator:
        # Prevent memory fragmentation within each GPU.
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    else:
        # Explicitly clear so a previous run's env doesn't leak in.
        os.environ.pop("TF_GPU_ALLOCATOR", None)

    # Enable aggressive cleanup of intermediate tensors
    os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"] = "true"

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.warning("No GPUs detected")
        return None

    # Apply config.gpu.num_replicas. None means "use every visible GPU" (default);
    # a positive int wired through set_visible_devices restricts TF to the first N
    # GPUs and leaves the rest untouched for other workloads
    #
    # Note that config.gpu.num_replicas is guaranteed to be None or in [1, total_gpus].
    # Any other values are caught at validate_args time before reaching this point. The
    # remaining edge-case where 0-GPUs are reported by TF is handled by train_command()
    # and inference_command() individually (here we simply return None)
    total_gpus = len(gpus)
    requested = config.gpu.num_replicas
    if requested is not None and requested < total_gpus:
        gpus = gpus[:requested]
        # set_visible_devices must run before any GPU memory-growth or logical-device
        # call, since those initialize the GPU runtime and freeze the visible set.
        tf.config.set_visible_devices(gpus, "GPU")
        logger.info(
            f"Restricting TF to {requested} of {total_gpus} GPUs "
            f"(per config.gpu.num_replicas={requested}); GPUs "
            f"{list(range(requested, total_gpus))} are left untouched."
        )

    try:
        for gpu in gpus:
            # set_memory_growth stayed under tf.config.experimental in TF 2.17.
            tf.config.experimental.set_memory_growth(gpu, True)
            # When per_gpu_memory_limit_mb is None we skip this call entirely and
            # rely on memory-growth only (recommended default on ~96 GB Blackwell cards).
            if config.gpu.per_gpu_memory_limit_mb is not None:
                # set_logical_device_configuration is the stable replacement for the
                # deprecated experimental.set_virtual_device_configuration.
                tf.config.set_logical_device_configuration(
                    gpu,
                    [
                        tf.config.LogicalDeviceConfiguration(
                            memory_limit=config.gpu.per_gpu_memory_limit_mb
                        )
                    ],
                )

        num_packs = config.gpu.nccl_num_packs

        # Set distributed strategy to prevent uneven VRAM usage
        # Try NCCL first; fall back to HierarchicalCopyAllReduce only if the
        # warmup all-reduce actually fails. NCCL 2.25.1 is the first NCCL with
        # official sm_120 (Blackwell) support, so this path is especially load-bearing
        # on the Blackwell cluster.
        try:
            strategy = tf.distribute.MirroredStrategy(
                cross_device_ops=tf.distribute.NcclAllReduce(num_packs=num_packs)
            )
            _warmup_collective(strategy)
            logger.info("Using NcclAllReduce for optimal NVIDIA GPU performance")
        except Exception as e:
            logger.warning(
                f"NCCL warmup all-reduce failed ({e}), falling back to HierarchicalCopyAllReduce"
            )
            strategy = tf.distribute.MirroredStrategy(
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=num_packs)
            )
            _warmup_collective(strategy)

        logger.info(f"Distributed strategy: {strategy.num_replicas_in_sync} GPUs")
        return strategy

    except Exception as e:
        # This exception catch is broad on purpose:
        # catches RuntimeError from device-config calls plus
        # non-RuntimeError failures (ValueError, tf.errors.*) from the fallback
        # warmup, so any GPU setup failure resolves to the graceful return-None
        # path the callers expect rather than escaping this function.
        logger.error(f"GPU configuration error: {e}")
        return None


def _report_final_training_status(pipeline) -> None:
    """
    Emit the terminal training status and exit nonzero on any permanently-failed stage.

    Extracted from train_command so the exit contract is unit-testable: a fully-successful run
    exits 0, a run with any recorded non-critical stage failure (vae_plots/rf_plots that never
    recovered across attempts) exits 1, and a missing pipeline exits 1 rather than reporting a
    false success.
    """
    if pipeline is None:
        # Defensive: the retry loop always sets pipeline or sys.exit(1)s first, and
        # --max-retries >= 1 is validated, so this is unreachable in practice — but never
        # report success without a completed pipeline.
        logger.error("Training produced no pipeline — treating as failure")
        sys.exit(1)

    # Plot stages are non-critical (a broken plot mustn't cost a retry cycle including data
    # regeneration), but their failures are recorded in the run manifest — surface them
    # loudly and exit nonzero so lost artifacts can't go unnoticed.
    failed_stages = pipeline.run_state.stages_failed
    if failed_stages:
        logger.error("=" * 60)
        logger.error(
            f"Training finished, but non-critical stage(s) permanently failed: "
            f"{', '.join(failed_stages)}"
        )
        logger.error(
            "Re-run the identical command to retry them — completed stages are skipped "
            "via the run manifest"
        )
        logger.error("=" * 60)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


def train_command():
    """Execute training pipeline with distributed strategy & fault tolerance"""
    logger.info("=" * 60)
    logger.info("Starting Aetherscan Training Pipeline")
    logger.info("=" * 60)

    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")

    # NOTE: come back to this later (print more descriptive info)
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

    # NOTE: come back to this later (should we provide a CPU-only mode?)
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
    pipeline = None

    for attempt in range(max_retries):
        try:
            logger.info(f"Training attempt: {attempt + 1}/{max_retries}")

            # Reinitialize the training pipeline on each attempt so no corrupted state is
            # persisted. The persisted run manifest (run_state_{save_tag}.json) tells the new
            # pipeline which rounds/stages already completed, so the attempt resumes exactly
            # where the previous one died (works identically for a full process relaunch)
            pipeline = run_training_pipeline(background_data=background_data, strategy=strategy)

            break  # If we get here, training succeeded

        except KeyboardInterrupt:
            # Don't retry on user interruption
            # Re-raise to propagate traceback
            logger.info("Training interrupted by user")
            raise

        except Exception as e:
            logger.error(f"Training attempt {attempt + 1} failed with error: {e}")

            if attempt < max_retries - 1:
                logger.info(
                    f"Attempting to recover from failure: attempt {attempt + 2}/{max_retries}"
                )
                logger.info(f"Waiting {retry_delay} seconds before retry...")

                # Collect garbage
                gc.collect()
                time.sleep(retry_delay)

            else:
                # Max retries exceeded
                logger.error(f"Training attempts exceeded maximum retries ({max_retries})")
                logger.error(f"Final error: {e}")
                sys.exit(1)

    # Note, the training configuration JSON is saved by the pipeline's final_save stage
    # (so it's covered by the retry machinery), not here
    _report_final_training_status(pipeline)


def _run_streaming_csv_inference(preprocessor: DataPreprocessor, strategy) -> dict:
    """
    Per-cadence streaming inference over the configured CSV catalogs.

    Flow (peak memory = one cadence's stamps + the next cadence's in-flight preprocessing,
    independent of catalog size):

        units = plan_cadences()                      # cadence groups + .npy paths, no work yet
        pipeline = InferencePipeline(strategy)       # models loaded once for the whole run
        for each cadence (prefetch depth 1):
            [background thread] preprocess cadence i+1 (energy detection; the persistent pool's
                                child processes do the real work — the thread mostly waits)
            [main thread]       load cadence i's stamps (memmap -> log-norm) -> encode on GPUs
                                -> RF -> write per-cadence results with per-cadence provenance

    Provenance (target/session/band/cadence_id/per-stamp frequency/timestamp/h5_path) is
    derived per cadence from its group key + metadata JSON via derive_cadence_provenance.
    Returns aggregate {n_cadence_snippets, n_processed, n_candidates, n_cadences}. Exceptions
    from the load/encode stages propagate to the retry loop in inference_command; per-cadence
    preprocessing failures are logged and skipped inside process_pending_cadence.
    """
    config = get_config()

    units = preprocessor.plan_cadences()
    if not units:
        logger.error("No cadence work units produced from the configured inference CSVs")
        sys.exit(1)
    logger.info(f"Streaming inference over {len(units)} cadence(s)")

    # Load models once; every cadence reuses this pipeline
    pipeline = InferencePipeline(strategy=strategy)

    totals = {"n_cadence_snippets": 0, "n_processed": 0, "n_candidates": 0, "n_cadences": 0}

    # Start the persistent energy-detection pool from the main thread (forking after
    # background threads exist risks inheriting mid-operation locks in the children)
    preprocessor.start_energy_detection_pool()
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="preproc_prefetch") as prefetch:
            future = prefetch.submit(preprocessor.process_pending_cadence, units[0])

            for i, unit in enumerate(units):
                cadence_result = future.result()

                # Prefetch depth 1: kick off cadence i+1's CPU preprocessing while the main
                # thread loads + encodes cadence i on the GPUs
                if i + 1 < len(units):
                    future = prefetch.submit(preprocessor.process_pending_cadence, units[i + 1])

                if cadence_result is None:
                    logger.info(f"Cadence {unit.group.key}: no stamps produced; skipping")
                    continue

                # Per-cadence provenance from the group key + metadata JSON
                try:
                    with open(cadence_result.metadata_path) as f:
                        metadata = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning(
                        f"Cadence {cadence_result.key}: could not read metadata at "
                        f"{cadence_result.metadata_path} ({e}); provenance will be sparse"
                    )
                    metadata = {"h5_paths": cadence_result.h5_paths}
                provenance = derive_cadence_provenance(
                    key=cadence_result.key,
                    group_by_cols=config.inference.cadence_group_by_cols,
                    metadata=metadata,
                )

                cadence_data = preprocessor.load_inference_data(
                    override_filepaths=[cadence_result.npy_path]
                ).astype(np.float32)

                results = pipeline.run_inference(
                    data=cadence_data,
                    npy_path=cadence_result.npy_path,
                    **provenance,
                )
                del cadence_data
                gc.collect()

                totals["n_cadence_snippets"] += results["n_cadence_snippets"]
                totals["n_processed"] += results["n_processed"]
                totals["n_candidates"] += results["n_candidates"]
                totals["n_cadences"] += 1

                logger.info(
                    f"Cadence {cadence_result.key} ({totals['n_cadences']} done, "
                    f"{len(units) - i - 1} to go): {results['n_processed']} snippets, "
                    f"{results['n_candidates']} candidate(s)"
                )
    finally:
        preprocessor.stop_energy_detection_pool()
        # Release TF dataset/iterator state once per run, after the loaded models are done
        tf.keras.backend.clear_session()
        logger.info("Cleared TensorFlow session state")

    return totals


# NOTE: we need to load the saved config from the corresponding training run, but when/where should we do that, and how does that play with apply_args_to_config()?
def inference_command():
    """Execute inference pipeline with distributed strategy & fault tolerance"""
    logger.info("=" * 60)
    logger.info("Starting Aetherscan Inference Pipeline")
    logger.info("=" * 60)

    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")

    # Required artifacts (encoder/rf/config paths) and stamp_width == width_bin
    # are enforced upstream by collect_validation_errors() in cli.py, so by the
    # time inference_command() runs those preconditions are guaranteed to hold.
    # TODO: add a sanity check that verifies encoder, RF, and config path all have the same tag. throw a warning if false

    # NOTE: come back to this later (print more descriptive info)
    logger.info("Configuration:")
    logger.info(f"  Data path: {config.data_path}")
    logger.info(f"  Model path: {config.model_path}")
    logger.info(f"  Output path: {config.output_path}")
    logger.info(f"  Encoder path: {config.inference.encoder_path}")
    logger.info(f"  Random Forest path: {config.inference.rf_path}")
    logger.info(f"  Config path: {config.inference.config_path}")
    if config.data.inference_files is not None:
        logger.info(f"  Inference CSVs to process: {config.data.inference_files}")
    else:
        logger.info(f"  Files to process: {config.data.test_files}")
    logger.info(f"  Classification threshold: {config.inference.classification_threshold}")

    # Setup GPU strategy
    try:
        strategy = setup_gpu_strategy()
    except Exception as e:
        logger.error(f"Failed to setup GPU strategy: {e}")
        sys.exit(1)

    # NOTE: come back to this later (should we provide a CPU-only mode?)
    if strategy is None:
        logger.error("No GPU strategy available. Inference requires GPU.")
        sys.exit(1)

    # NOTE: come back to this later (does fault tolerance work properly with inference?)
    # Run preprocessing + inference with fault tolerance.
    # Recovery is state-based, not checkpoint-based: preprocessing writes per-cadence
    # .npy files as it goes and skips any whose .npy already exists, so simply
    # retrying resumes from where the last attempt died. No checkpoint metadata
    # is needed for the preprocessing stage.
    preprocessor = DataPreprocessor()
    max_retries = config.inference.max_retries
    retry_delay = config.inference.retry_delay
    results = None
    # Legacy --test-files path only: cache cadence_data across retry attempts so an
    # inference-only failure doesn't trigger a re-load / re-downsample / re-log-norm
    # pass. Mirrors how train_command loads background_data once outside its retry
    # loop. The streaming CSV path holds no catalog-sized state to cache — its
    # per-cadence .npy files are the resume mechanism.
    cadence_data: np.ndarray | None = None
    npy_path_for_logging: str | None = None

    for attempt in range(max_retries):
        try:
            logger.info(f"Inference attempt: {attempt + 1}/{max_retries}")

            if config.data.inference_files is not None:
                # Streaming CSV path: per-cadence preprocess -> load -> encode -> RF ->
                # write, with models loaded once and prefetch depth 1 (see
                # _run_streaming_csv_inference). Memory stays independent of catalog size.
                results = _run_streaming_csv_inference(preprocessor, strategy)
            else:
                if not config.data.test_files:
                    logger.error(
                        "Neither --inference-files nor --test-files is configured; "
                        "nothing to load for inference"
                    )
                    sys.exit(1)
                # Load stage. Skipped on retry if a previous attempt already produced
                # cadence_data (i.e. only the inference stage failed).
                if cadence_data is None:
                    cadence_data = preprocessor.load_inference_data().astype(np.float32)
                    npy_path_for_logging = config.data.test_files[0]
                else:
                    logger.info("Reusing cadence_data from previous attempt (skipping load)")

                # NOTE: come back to this later (inference-stage resume should skip cadences already in the DB. not yet implemented)
                # Inference stage
                results = run_inference_pipeline(
                    cadence_data=cadence_data,
                    npy_path=npy_path_for_logging,  # TODO: handle multiple test_files properly
                    strategy=strategy,
                )
            break  # success

        except KeyboardInterrupt:
            # Don't retry on user interruption; re-raise to propagate traceback
            logger.info("Inference interrupted by user")
            raise

        except Exception as e:
            logger.error(f"Inference attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                gc.collect()
                time.sleep(retry_delay)
            else:
                logger.error(f"Exceeded max retries ({max_retries}). Final error: {e}")
                sys.exit(1)

    # NOTE: come back to this later (should we create dedicated (tagged) directories inside output_path to store inference results (plots, configs, etc.)? note, data still written to db regardless)
    # Save inference configuration
    config_path = os.path.join(config.output_path, f"config_{config.checkpoint.save_tag}.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)  # Create dir if it doesn't exist

    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Inference configuration saved to {config_path}")

    logger.info("=" * 60)
    logger.info("Inference completed successfully!")
    logger.info("Summary:")
    if "n_cadences" in results:
        logger.info(f"  Total cadences: {results['n_cadences']}")
    logger.info(f"  Total cadence snippets: {results['n_cadence_snippets']}")
    logger.info(f"    Processed: {results['n_processed']}")
    logger.info(f"    Candidates found: {results['n_candidates']}")
    logger.info("=" * 60)


def main():
    """Main entry point to Aetherscan pipeline"""
    # Auto-load <repo>/.env (searched upward from CWD) into os.environ so
    # environment variables land in the process env before any aetherscan
    # module reads them — covers the Ampere conda workflow without needing
    # "source .env" or an inline VAR=val prefix, and harmlessly redundant in
    # the container workflow (utils/run_container.sh already passes --env for
    # the same keys, and load_dotenv() by default won't override existing
    # values). Multiprocess workers spawned later inherit os.environ from us.
    load_dotenv(find_dotenv())

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

    # NOTE: come back to this later
    # Inference mode: if the user pointed --config-path at a saved JSON config,
    # layer its values onto the singleton *before* validate_args runs. That way
    # validate_args sees the merged (saved + CLI) view via _resolve, and any
    # invariants that involve fields stored in the saved config (e.g. width_bin /
    # stamp_width / latent_dim / dense_layer_size) are checked against the actual
    # values inference will use rather than the dataclass defaults. Train mode
    # is unaffected.
    if args.command == "inference" and getattr(args, "config_path", None) is not None:
        try:
            apply_saved_config(args.config_path)
            logger.info(f"Saved config loaded from {args.config_path}")
        except Exception as e:
            parser.print_help()
            logger.error(f"Failed to load saved config: {e}")
            logger.error("See usage")
            sys.exit(1)

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
