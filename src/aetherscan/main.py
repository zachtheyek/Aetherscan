# NOTE: remove time filters and strictly query db using tag. would need to ensure save-tag is unique at the start of each run
# NOTE: is there a way to parallelize the CPU-GPU processing (e.g. while GPU is working on training/inference, CPU starts working on data_generation/preprocessing)
"""
Entry point for Aetherscan Pipeline
"""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import json
import logging
import os
import socket
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv

from aetherscan.benchmark import stage_timer
from aetherscan.candidate_figures import write_candidate_snippet_sidecar
from aetherscan.candidate_triage import partition_candidates_by_frequency, report_exclusion_ranges
from aetherscan.cli import (
    apply_args_to_config,
    apply_saved_config,
    resolve_save_tag,
    setup_argument_parser,
    validate_args,
)
from aetherscan.config import get_config, init_config
from aetherscan.dashboard_launcher import launch_dashboard
from aetherscan.db import get_db, get_machine_name, init_db
from aetherscan.display_tag import display_tag
from aetherscan.hf_hub import resolve_inference_artifacts
from aetherscan.inference import InferencePipeline, run_inference_pipeline, summarize_confidences
from aetherscan.inference_viz import InferenceVizCollector, render_inference_visualizations
from aetherscan.logger import get_logger, init_logger
from aetherscan.manager import get_manager, init_manager, register_logger
from aetherscan.monitor import init_monitor
from aetherscan.preprocessing import (
    CadenceResult,
    DataPreprocessor,
    PendingCadence,
    derive_cadence_provenance,
)
from aetherscan.run_state import inference_config_fingerprint
from aetherscan.tag_guards import enforce_tag_guards
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

    # Dedicated per-GPU kernel-launch threads (see GPUConfig.gpu_thread_mode). Read lazily by
    # TF at GPU-runtime init like the allocator var above.
    valid_thread_modes = {"global", "gpu_private", "gpu_shared"}
    if config.gpu.gpu_thread_mode not in valid_thread_modes:
        raise ValueError(
            f"gpu.gpu_thread_mode must be one of {sorted(valid_thread_modes)}, "
            f"got {config.gpu.gpu_thread_mode!r}"
        )
    if config.gpu.gpu_thread_mode == "global":
        # Explicitly clear so a previous run's env doesn't leak in.
        os.environ.pop("TF_GPU_THREAD_MODE", None)
        os.environ.pop("TF_GPU_THREAD_COUNT", None)
    else:
        os.environ["TF_GPU_THREAD_MODE"] = config.gpu.gpu_thread_mode
        os.environ["TF_GPU_THREAD_COUNT"] = str(config.gpu.gpu_thread_count)

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
    exits 0, a run with any recorded non-critical stage failure (vae_plots/rf_plots/hf_upload that
    never recovered across attempts) exits 1, and a missing pipeline exits 1 rather than reporting
    a false success. A run that skipped RF training because a pre-loaded RF was already trained
    exits 0 but with a warning annotation instead of the unqualified success line (issue #142).
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

    # Qualified success (issue #142): a run whose RF stage was skipped because an
    # already-trained Random Forest was pre-loaded (e.g. resumed from the wrong tag) must not
    # report unqualified success — the saved RF was never trained on this run's encoder.
    # Ordering is intentional: the failed_stages branch above exits(1) first, so a run that
    # both had a permanent stage failure AND skipped RF reports the failure (the stronger
    # signal), never this qualified-success annotation.
    skipped_rf_tag = getattr(pipeline, "rf_training_skipped_from_tag", None)
    if skipped_rf_tag:
        logger.warning("=" * 60)
        logger.warning(
            f"Training completed, but Random Forest training was SKIPPED — the saved RF was "
            f"loaded from tag '{skipped_rf_tag}', not trained during this run"
        )
        logger.warning("=" * 60)
        return

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


def _post_benchmark_report(tag: str) -> None:
    """
    Render the end-of-run benchmark report (utils/benchmark_report.py) and post it to Slack.

    Runs at the tail of train_command/inference_command rather than after monitor shutdown:
    the monitor never writes pipeline_stages rows (stage spans land via the async DB write
    queue), and by the time main()'s finally block stops the monitor the Slack handler is
    being torn down too. An explicit db.flush() is what guarantees every span row is on disk
    before the report reads them. Fully guarded: any failure (including the report tool's
    SystemExit) logs an error and never fails the run.
    """
    try:
        config = get_config()
        if config is None:
            raise ValueError("get_config() returned None")
        if not config.monitor.benchmark_report_enabled:
            logger.info("Benchmark report disabled (--no-benchmark-report)")
            return

        db = get_db()
        if db is None:
            raise ValueError("get_db() returned None")
        # Stage spans reach the DB through the async write queue — drain it so the report
        # sees every row of this run
        db.flush(timeout=config.db.flush_timeout)

        # utils/benchmark_report.py is deliberately import-free of aetherscan (stdlib
        # sqlite3 + numpy + matplotlib) so it stays cluster-portable — load it by file path
        # instead of importing, same pattern as tests/unit/test_benchmark.py. The
        # sys.modules registration must precede exec_module: the module's @dataclass
        # resolves its own module by name at class-creation time (PEP 563 string
        # annotations).
        report_path = Path(__file__).resolve().parents[2] / "utils" / "benchmark_report.py"
        if not report_path.exists():
            # e.g. a pip-installed package without the repo checkout alongside
            logger.warning(f"Benchmark report skipped: {report_path} does not exist")
            return
        spec = importlib.util.spec_from_file_location("benchmark_report", report_path)
        benchmark_report = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_report"] = benchmark_report
        spec.loader.exec_module(benchmark_report)

        rows = benchmark_report.load_rows(db.db_path, tag)
        if not rows:
            logger.warning(f"Benchmark report skipped: no pipeline_stages rows for tag {tag!r}")
            return
        root = benchmark_report.build_stage_tree(rows)
        # tag stays the plain DB tag for load_rows / build_suggestions; the display tag scopes the
        # PNG filename, on-figure title, and Slack message to this host. The import-free report tool
        # uses its tag arg only for the suptitle, so passing the display tag there is safe.
        display_tag_value = display_tag(tag, get_machine_name())
        png_path = os.path.join(
            config.output_path, "plots", f"benchmark_report_{display_tag_value}.png"
        )
        benchmark_report.render_report_png(root, rows, display_tag_value, png_path)
        logger.info(f"Benchmark report saved to {png_path}")

        # Bottleneck suggestions ride along as the upload's comment, landing in the run
        # thread right next to the figure
        suggestions = benchmark_report.build_suggestions(root, db.db_path, tag)
        message = "\n".join(f"- {s}" for s in suggestions) if suggestions else None

        logger_instance = get_logger()
        if logger_instance is None:
            raise ValueError("get_logger() returned None")
        if not logger_instance.upload_image_to_slack(
            png_path, title=f"Benchmark Report - {display_tag_value}", message=message
        ):
            logger.warning("Benchmark report rendered but Slack upload was skipped or failed")

    except (Exception, SystemExit) as e:
        # SystemExit included: load_rows raises it on a pre-benchmarking-schema DB.
        # Observability must never fail an otherwise-finished run. Runs at the very tail
        # of the command (after all real work is done), so this also swallowing a
        # KeyboardInterrupt-adjacent exit here has minimal blast radius — a Ctrl-C during
        # report generation still leaves the run's actual results intact.
        # TODO: if utils/benchmark_report.py's load_rows ever stops raising SystemExit for
        # a pre-benchmarking-schema DB, narrow this back to `except Exception`.
        logger.error(f"Benchmark report generation failed: {e}")


def _post_perband_report(tag: str) -> None:
    """
    Render the per-band inference-performance plot (utils/perband_report.py) and post it to
    Slack, at the tail of an inference run alongside _post_benchmark_report.

    Groups per-cadence energy-detection preprocessing wall-clock by band (L/S/C/X) and by
    frequency, joining the pipeline_stages umbrella spans to the run's inference catalog CSV(s)
    via the plan-index reconstruction. Gated by the same monitor.benchmark_report_enabled flag,
    and fully guarded: any failure logs an error and never fails the run. Skips quietly when the
    run has no inference catalog (the legacy --test-files path), when the join assumption doesn't
    hold, or when the report tool file isn't alongside the package (mirrors _post_benchmark_report).
    """
    try:
        config = get_config()
        if config is None:
            raise ValueError("get_config() returned None")
        if not config.monitor.benchmark_report_enabled:
            return  # _post_benchmark_report already logged the disabled notice

        # The per-band join needs the run's inference catalog CSV(s). The legacy --test-files
        # path has no catalog (it loads a pre-built .npy), so there is nothing to group by band.
        if not config.data.inference_files:
            logger.info(
                "Per-band inference plot skipped: no inference catalog CSV (--test-files run)"
            )
            return
        catalog_paths = [config.get_inference_file_path(f) for f in config.data.inference_files]

        db = get_db()
        if db is None:
            raise ValueError("get_db() returned None")
        db.flush(timeout=config.db.flush_timeout)

        # Load the tool by file path, preserving its no-aetherscan-imports contract (same
        # pattern as _post_benchmark_report / tests/unit/test_benchmark.py).
        report_path = Path(__file__).resolve().parents[2] / "utils" / "perband_report.py"
        if not report_path.exists():
            logger.warning(f"Per-band inference plot skipped: {report_path} does not exist")
            return
        spec = importlib.util.spec_from_file_location("perband_report", report_path)
        perband_report = importlib.util.module_from_spec(spec)
        sys.modules["perband_report"] = perband_report
        spec.loader.exec_module(perband_report)

        hostname = socket.gethostname()
        # tag stays the plain DB tag (it keys the pipeline_stages query inside
        # render_perband_report); the display tag scopes the PNG filename, on-figure title, and
        # Slack message to this host so cross-host artifacts don't collide (matches the other plots).
        display_tag_value = display_tag(tag, get_machine_name())
        png_path = os.path.join(
            config.output_path, "plots", f"perband_inference_perf_{display_tag_value}.png"
        )
        result = perband_report.render_perband_report(
            db.db_path, tag, catalog_paths, png_path, hostname, display_tag=display_tag_value
        )
        if result is None:
            return  # render_perband_report already logged why it skipped
        logger.info(f"Per-band inference plot saved to {png_path}")

        logger_instance = get_logger()
        if logger_instance is None:
            raise ValueError("get_logger() returned None")
        if not logger_instance.upload_image_to_slack(
            png_path, title=f"Inference performance by band - ({display_tag_value})"
        ):
            logger.warning(
                "Per-band inference plot rendered but Slack upload was skipped or failed"
            )

    except Exception as e:
        # Observability must never fail an otherwise-finished run (matches _post_benchmark_report)
        logger.error(f"Per-band inference plot generation failed: {e}")


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
        with stage_timer("train.load_backgrounds"):
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
            # where the previous one died. The save_tag is fixed for the life of the process, so
            # these in-process retries resume automatically; a full process relaunch mints a fresh
            # datetime tag and only resumes when re-invoked with --load-tag {full-tag}.
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

    # Post the end-of-run benchmark report before the terminal status report: the latter
    # sys.exit(1)s when any non-critical stage permanently failed, and the timing data is
    # exactly as valuable on those runs
    _post_benchmark_report(config.checkpoint.save_tag)

    # Note, the training configuration JSON is saved by the pipeline's final_save stage
    # (so it's covered by the retry machinery), not here
    _report_final_training_status(pipeline)


# Observed per-band worst-case in-flight cadence footprints (GB) for the RAM preflight
# (#408): the largest single-cadence loaded stamp array measured on the 350-cadence /datag
# subset benchmark, per band (max stamps/cadence x 196,608 B/snippet: C 329,749 stamps at
# 8438 MHz -> ~65 GB; L 73,004 -> ~15; S 56,880 -> ~12; X 37,139 -> ~8). These are observed
# tails, not bounds — a catalog can exceed them — so the preflight is a WARNING, never a
# clamp (#408: a silent step-down would make cross-host wall-clock comparisons quietly
# incomparable). Bands the table doesn't know assume the C-band worst case.
_BAND_WORST_CASE_INFLIGHT_GB = {"L": 15.0, "S": 12.0, "C": 65.0, "X": 8.0}
_UNKNOWN_BAND_WORST_CASE_GB = 65.0
# Fraction of total host RAM the worst case may occupy before the preflight warns: the
# remainder covers TF/models/pool overhead and page-cache headroom (the README's OOM note).
_RAM_PREFLIGHT_BUDGET_FRACTION = 0.9


def _prefetch_ram_preflight(
    pending: list, group_by_cols: list[str], depth: int, total_ram_gb: float
) -> str | None:
    """
    Catalog-derived RAM preflight for the prefetch pipeline (#408): estimate the worst-case
    resident footprint as (depth + 1) in-flight cadences (the prefetch slots plus the one
    being inferred) of the largest per-band footprint among the bands ACTUALLY present in
    the pending catalog, and return a warning string when it exceeds the host budget
    (None otherwise — the caller logs, nothing is ever clamped). Keying on the catalog's
    band mix rather than the static C-band worst case keeps the warning quiet on e.g. an
    X-band-only catalog, whose observed tail is ~8x smaller. Bands are read from the
    cadence group keys via the configured 'Band' group-by column; catalogs grouped without
    one assume the conservative unknown-band worst case.
    """
    if not pending or depth < 1 or total_ram_gb <= 0:
        return None
    try:
        band_index = group_by_cols.index("Band")
    except ValueError:
        band_index = None

    worst_gb = 0.0
    bands: set[str] = set()
    for unit in pending:
        key = unit.group.key
        if band_index is not None and band_index < len(key):
            band = str(key[band_index]).strip().upper() or "?"
        else:
            band = "?"
        bands.add(band)
        worst_gb = max(
            worst_gb, _BAND_WORST_CASE_INFLIGHT_GB.get(band, _UNKNOWN_BAND_WORST_CASE_GB)
        )

    budget_gb = total_ram_gb * _RAM_PREFLIGHT_BUDGET_FRACTION
    worst_total_gb = (depth + 1) * worst_gb
    if worst_total_gb <= budget_gb:
        return None

    # The largest depth whose worst case fits the budget ((d + 1) in-flight cadences)
    suggested_depth = max(1, int(budget_gb // worst_gb) - 1)
    return (
        f"RAM preflight (#408): --prefetch-depth {depth} budgets {depth + 1} in-flight "
        f"cadence(s) x ~{worst_gb:.0f} GB (worst observed for band(s) "
        f"{', '.join(sorted(bands))}) ≈ {worst_total_gb:.0f} GB, above "
        f"{_RAM_PREFLIGHT_BUDGET_FRACTION:.0%} of this host's {total_ram_gb:.0f} GB RAM. "
        f"The estimate is a per-band worst case, not a measurement — the run proceeds "
        f"unclamped — but if this host has OOM'd before, consider --prefetch-depth "
        f"{suggested_depth}."
    )


class NonRetryableInferenceError(RuntimeError):
    """A permanent inference failure (bad catalog/config) that retrying cannot fix.

    inference_command's retry loop re-raises this immediately instead of burning retry
    attempts on it; transient failures (I/O hiccups, GPU errors) stay plain exceptions and
    keep the existing retry semantics.
    """


def _prefetch_cadence(preprocessor: DataPreprocessor, unit: PendingCadence) -> tuple:
    """
    Prefetch-thread task for one cadence: preprocess (energy detection -> stamp .npy), then
    load + log-norm the stamps (#298 I5-overlap — the load_lognorm span used to run on the
    GPU main thread between cadences; here it hides under the prefetch pipeline).

    Returns (cadence_result, cadence_data). cadence_result is None when the cadence produced
    no stamps. cadence_data is None when the load failed — the load is retried on the
    inference thread inside _infer_cadence, whose per-cadence failure containment covers it
    (a prefetch-side load exception must not abort the whole pass one iteration later).
    """
    cadence_result = preprocessor.process_pending_cadence(unit)
    if cadence_result is None:
        return None, None
    try:
        # copy=False: the loader already returns float32; don't duplicate GBs of stamps.
        # parallel=False: this loads exactly one already-downsampled cadence .npy whose
        # per-cadence work is one vectorized log-norm pass, while the persistent
        # energy-detection pool is busy at full n_processes width — forking a second
        # chunk pool here would double-subscribe the CPU.
        with stage_timer("load_lognorm"):
            cadence_data = preprocessor.load_inference_data(
                override_filepaths=[cadence_result.npy_path], parallel=False
            ).astype(np.float32, copy=False)
    except Exception as e:
        logger.error(
            f"Cadence {cadence_result.key}: prefetch-side load failed ({e}); "
            f"the inference thread will retry the load"
        )
        cadence_data = None
    return cadence_result, cadence_data


def _resolve_prune_stamps(config) -> bool:
    """Resolve the stamp-cache pruning mode (#302, default flipped by #399): an explicit
    inference.prune_stamps wins; the None default means OFF — stamps are kept so the
    fingerprint-scoped cache makes re-scores (new weights / threshold sweeps under the
    same ED config) skip preprocessing out of the box. Catalog-scale runs must opt in
    with --prune-stamps or the stamp volume exceeds scratch (~30-90 TB unpruned)."""
    if config.inference.prune_stamps is not None:
        return bool(config.inference.prune_stamps)
    return False


def _prune_cadence_stamps(
    cadence_result: CadenceResult, results: dict, collector: InferenceVizCollector | None
) -> None:
    """Delete one scored cadence's stamp .npy (#302), keeping everything the rest of the
    run needs: the metadata .json (provenance + viz + resume guard), a ~196 KB
    .candidates.npz snippet sidecar per candidate (the candidate figures' read path), and
    the collector's bounded top-K pixel pool (the stamp gallery's). Runs strictly AFTER
    the cadence's 'inferred' manifest row and viz collection — resume rides the DB row
    and never touches the .npy. Best-effort: a pruning failure keeps the stamps and the
    run continues (disk pressure is a slow failure; a science pass must not die for it)."""
    npy_path = cadence_result.npy_path
    try:
        candidate_idx = np.nonzero(np.asarray(results["predictions"]) == 1)[0]
        if len(candidate_idx):
            write_candidate_snippet_sidecar(npy_path, candidate_idx)
        if collector is not None:
            collector.pool_gallery_pixels(cadence_result.metadata_path, npy_path)
        size_gb = 0.0
        with contextlib.suppress(OSError):
            size_gb = os.path.getsize(npy_path) / 1e9
        os.remove(npy_path)
        logger.info(
            f"Pruned stamp cache for cadence {cadence_result.key}: removed {npy_path} "
            f"({size_gb:.2f} GB; metadata kept, {len(candidate_idx)} candidate snippet(s) "
            f"sidecarred)"
        )
    except Exception as e:
        logger.error(
            f"Stamp pruning failed for cadence {cadence_result.key} ({e}); stamps kept; "
            f"run continues"
        )


def _infer_cadence(
    pipeline: InferencePipeline,
    preprocessor: DataPreprocessor,
    unit: PendingCadence,
    cadence_result: CadenceResult,
    config_fingerprint: str,
    cadence_data: np.ndarray | None = None,
) -> dict:
    """
    Run the inference stage for one preprocessed cadence: derive provenance, load its stamps
    (memmap -> log-norm), encode on the GPUs, classify with the RF, and record the result in
    both DB tables. Returns run_inference's results dict augmented with the provenance dict
    and the stage's duration_s. Exceptions propagate to the caller's per-cadence containment.

    Supersede-on-retry ordering (single-writer FIFO makes each step atomic w.r.t. the next):
    1. mark_superseded(inference_results, tag, npy_path) — partial positives written by a
       dead attempt can't mix with this attempt's rows;
    2. run_inference writes the fresh positives;
    3. mark_superseded(inference_cadences, tag, npy_path) — retires the 'preprocessed' row
       and any 'failed'/stale 'inferred' rows;
    4. write_inference_cadence(status='inferred') — the row the stage-aware resume keys on,
       carrying the aggregate stats (n_stamps/n_candidates/confidence summary) so run-level
       artifacts don't depend on the positives-only inference_results table.
    """
    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")
    db = get_db()
    if db is None:
        raise ValueError("get_db() returned None")
    tag = config.checkpoint.save_tag

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

    stage_start = time.time()

    # Umbrella stage span for this cadence's inference phase — the encode / rf / db_write
    # sub-stages inside nest under it via thread-local naming (load_lognorm normally ran
    # on the prefetch thread already — see _prefetch_cadence)
    with stage_timer(f"inference.infer_cadence_{unit.index:03d}"):
        if cadence_data is None:
            # Fallback: the prefetch task's load failed (or a caller passed none) — load
            # here so the per-cadence containment around this function covers a bad .npy
            with stage_timer("load_lognorm"):
                cadence_data = preprocessor.load_inference_data(
                    override_filepaths=[cadence_result.npy_path], parallel=False
                ).astype(np.float32, copy=False)

        # Step 1: retire any partial rows from a dead attempt before fresh ones land
        db.mark_superseded("inference_results", tag, npy_path=cadence_result.npy_path)

        results = pipeline.run_inference(
            data=cadence_data,
            npy_path=cadence_result.npy_path,
            # Stable per-catalog cadence index -> reproducible per-cadence TF stream (#279)
            seed_key=unit.index,
            **provenance,
        )
        # cadence_data is a plain ndarray (no cycles): the del refcount-frees it, and
        # run_inference's finally block already ran this cadence's one gc pass (#298 I7)
        del cadence_data

        duration_s = time.time() - stage_start

        # Steps 3 + 4: new-row-plus-supersede on the run manifest
        confidence_summary = summarize_confidences(
            results["proba_true"], config.inference.classification_threshold
        )
        db.mark_superseded("inference_cadences", tag, npy_path=cadence_result.npy_path)
        db.write_inference_cadence(
            npy_path=cadence_result.npy_path,
            status="inferred",
            tag=tag,
            csv_path=unit.group.csv_path,
            cadence_key=cadence_result.key,
            n_stamps=results["n_cadence_snippets"],
            n_candidates=results["n_candidates"],
            confidence_summary=confidence_summary,
            duration_s=duration_s,
            config_fingerprint=config_fingerprint,
        )

    results["provenance"] = provenance
    results["duration_s"] = duration_s
    return results


def _run_streaming_csv_inference(
    preprocessor: DataPreprocessor,
    strategy: tf.distribute.Strategy,
    gallery_pool: list | None = None,
) -> dict:
    """
    Per-cadence streaming inference over the configured CSV catalogs, with stage-aware
    resume off the inference_cadences run manifest.

    Flow (peak memory = up to inference.prefetch_depth in-flight cadences of stamps +
    loaded arrays plus the one being inferred, independent of catalog size):

        units = plan_cadences()                # cadence groups + .npy paths, no work yet
        skip units with a live 'inferred' manifest row for this tag (fold their stored
            aggregates into the totals); for the rest:
        start the ED pool + first prefetch, then InferencePipeline(strategy) — models load
            once, hidden under the first cadence's energy detection
        for each pending cadence (prefetch depth = inference.prefetch_depth, #298 N2):
            [prefetch thread(s)] preprocess + load/log-norm upcoming cadences (energy
                                 detection skipped when the stamp .npy already exists —
                                 preprocessing-artifact resume; see _prefetch_cadence)
            [main thread]        encode on GPUs -> RF -> write per-cadence results +
                                 manifest row (_infer_cadence), consuming cadences in
                                 COMPLETION order (#401) — scores are order-free (seeded
                                 on the catalog index), only row/log order varies

    Failure containment mirrors preprocessing's: a cadence whose inference stage fails is
    logged, recorded as status='failed' in the manifest, and the loop continues; the pass
    raises at the end so inference_command's retry loop re-attempts — and the manifest skip
    means only the failed cadences re-run. Raises NonRetryableInferenceError when the
    catalog yields no work units or no cadence produces a stamp .npy — permanent conditions
    the retry loop must not retry. On a fully successful pass the visualization suite is
    rendered (config.inference.inference_viz_enabled; every figure is exception-guarded).

    Returns aggregate {n_cadence_snippets, n_processed, n_candidates, n_cadences,
    n_skipped}, where skipped (resumed) cadences contribute their manifest aggregates.
    """
    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")
    db = get_db()
    if db is None:
        raise ValueError("get_db() returned None")
    tag = config.checkpoint.save_tag

    units = preprocessor.plan_cadences()
    if not units:
        raise NonRetryableInferenceError(
            "No cadence work units produced from the configured inference CSVs"
        )

    totals = {
        "n_cadence_snippets": 0,
        "n_processed": 0,
        "n_candidates": 0,
        "n_cadences": 0,
        "n_skipped": 0,
    }
    # gallery_pool (a run-scoped list from inference_command) persists the stamp-gallery
    # pixel pool across the in-process retry attempts (#305): a fresh collector per attempt
    # would otherwise blank the gallery for cadences an earlier attempt pruned.
    collector = (
        InferenceVizCollector(gallery_pool=gallery_pool)
        if config.inference.inference_viz_enabled
        else None
    )

    # Stage-aware resume: a live 'inferred' manifest row for (tag, npy_path) means the cadence
    # completed on an earlier attempt — skip it and reuse its aggregates, but ONLY when it was
    # written under the same inference config. The fingerprint guard stops a reused --save-tag
    # with a changed threshold/model/geometry from silently serving stale results (the inference
    # counterpart of the training-side config_fingerprint guard).
    # Flush first so rows queued by this process (an in-process retry) are visible.
    current_fingerprint = inference_config_fingerprint(config.to_dict())
    db.flush()
    inferred_rows = {
        row["npy_path"]: row for row in db.query_inference_cadences(tag=tag, status="inferred")
    }

    pending = []
    stale_config = 0
    for unit in units:
        manifest_row = inferred_rows.get(unit.npy_path)
        if manifest_row is None:
            pending.append(unit)
            continue
        if manifest_row.get("config_fingerprint") != current_fingerprint:
            # Live 'inferred' row, but written under a different inference config -> don't reuse.
            # Re-infer; _infer_cadence's supersede step retires the stale row.
            stale_config += 1
            pending.append(unit)
            continue
        n_snippets = int(manifest_row.get("n_stamps") or 0)
        n_candidates = int(manifest_row.get("n_candidates") or 0)
        totals["n_cadence_snippets"] += n_snippets
        totals["n_processed"] += n_snippets
        totals["n_candidates"] += n_candidates
        totals["n_cadences"] += 1
        totals["n_skipped"] += 1
        if collector is not None:
            # Viz collection must never fail the pass: a collector bug degrades the plots,
            # not the science (the render phase has the same contract via _viz_safe)
            try:
                collector.record_skipped(
                    unit.group.key,
                    unit.npy_path,
                    DataPreprocessor.cadence_metadata_path(unit.npy_path),
                    manifest_row,
                    catalog_index=unit.index,
                )
            except Exception as e:
                logger.error(
                    f"Viz collection failed for skipped cadence {unit.group.key} ({e}); "
                    f"plots will be degraded; run continues"
                )
        logger.info(
            f"Cadence {unit.group.key}: already inferred under tag {tag} "
            f"({n_snippets} snippets, {n_candidates} candidate(s)); skipping"
        )

    if stale_config:
        logger.warning(
            f"{stale_config} cadence(s) under tag {tag} have an 'inferred' manifest row written "
            f"with a DIFFERENT inference config; re-inferring rather than reusing stale results "
            f"(a reused --save-tag with a changed threshold/model/geometry is not resumed). Use a "
            f"fresh --save-tag to keep separate configs' results separate."
        )

    logger.info(
        f"Streaming inference over {len(pending)} cadence(s) "
        f"({totals['n_skipped']} already inferred, resumed from manifest)"
    )

    prune_stamps = _resolve_prune_stamps(config)
    logger.info(
        f"Stamp-cache pruning: {'ON' if prune_stamps else 'OFF'} "
        f"({'explicit' if config.inference.prune_stamps is not None else 'default'}; "
        f"{'fingerprint-scoped default cache dir' if config.inference.preprocess_output_dir is None else 'explicit --preprocess-output-dir'})"
    )
    if not prune_stamps:
        # Loud on purpose (#399): keeping the cache is the right default for subset-scale
        # work (re-scores under the same ED config skip preprocessing entirely), but a
        # full catalog writes ~30-90 TB of stamps — a catalog run without --prune-stamps
        # dies on disk after a few hundred cadences.
        logger.info(
            "Stamp cache will be RETAINED (~1 GB/cadence average; enables free re-scores "
            "under the same ED config). Catalog-scale runs should pass --prune-stamps to "
            "bound disk usage."
        )

    # RAM preflight (#408): warn — never clamp — when the catalog's per-band worst case
    # exceeds the host budget at the configured depth. psutil is deferred so this module
    # stays importable without it (the monitor already owns the hard dependency).
    try:
        import psutil  # noqa: PLC0415

        preflight_warning = _prefetch_ram_preflight(
            pending,
            config.inference.cadence_group_by_cols,
            max(1, config.inference.prefetch_depth),
            psutil.virtual_memory().total / 1e9,
        )
        if preflight_warning:
            logger.warning(preflight_warning)
    except Exception as e:
        logger.info(f"RAM preflight skipped ({e}); run is unaffected")

    failed_keys: list[tuple] = []
    if pending:
        # Start the persistent energy-detection pool from the main thread BEFORE any
        # background thread exists (forking after threads spin up risks inheriting
        # mid-operation locks in the children)
        preprocessor.start_energy_detection_pool()
        try:
            # Prefetch depth (#298 N2) with completion-order consumption (#401): `depth`
            # cadences preprocess+load concurrently ahead of the GPU stage, and the main
            # thread consumes WHICHEVER future finishes first, submitting a replacement
            # immediately — so one straggler (a 30-minute RFI-dense preprocess, a
            # 10-minute load of a monster stamp array) no longer head-of-line-blocks the
            # other slots into idleness (measured: 12% of the depth-3 baseline run's wall
            # had ZERO preprocessing running while finished slots waited behind a
            # straggler). Per-cadence science is order-free by construction — seeding
            # keys on unit.index (the catalog position, not execution order) and the
            # reference-cloud reservoir selects by content-derived keys (#401) — so
            # results are identical at any depth and any completion order; what does vary
            # run-to-run is manifest/DB ROW ORDER, log interleaving, and the
            # latent-projection figure's bounded non-candidate subsample (viz-only; the
            # render phase catalog-sorts the records for everything else). Row content is
            # unchanged; resume/queries key on tag+npy_path, never order. Cost: up to
            # `depth` in-flight cadences of RAM.
            depth = max(1, config.inference.prefetch_depth)
            with ThreadPoolExecutor(
                max_workers=depth, thread_name_prefix="preproc_prefetch"
            ) as prefetch:
                in_flight = {
                    prefetch.submit(_prefetch_cadence, preprocessor, unit): unit
                    for unit in pending[:depth]
                }
                next_submit = len(in_flight)

                # Load models AFTER the first prefetch is in flight (#298 rider): the
                # 10-60 s encoder+RF+calibrator load hides under the first cadence's
                # energy detection. The worker pool was forked before any thread existed,
                # and every cadence reuses this one pipeline.
                pipeline = InferencePipeline(strategy=strategy)

                n_consumed = 0
                while in_flight:
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    # NOTE: an exception inside a prefetched preprocessing task surfaces
                    # at .result() below, when its future completes — and then propagates
                    # to inference_command's retry loop. In practice
                    # process_pending_cadence swallows per-cadence failures (returns
                    # None) and _prefetch_cadence contains load failures, so only
                    # infrastructure-level errors (e.g. a broken worker pool) raise.
                    future = next(iter(done))
                    unit = in_flight.pop(future)
                    cadence_result, cadence_data = future.result()
                    # Drop the future AND the done set immediately: Future._result pins
                    # the (cadence_result, cadence_data) tuple, so a surviving reference
                    # would keep a cadence-sized array (~65 GB worst case) resident
                    # through this iteration's encode and the next blocking wait() —
                    # exactly what the `del cadence_data` below exists to prevent. (The
                    # old popleft().result() shape dropped the future as a temporary;
                    # this is that shape's explicit equivalent.)
                    del future, done
                    n_consumed += 1

                    # Keep `depth` cadences in flight while the main thread encodes this one
                    if next_submit < len(pending):
                        next_unit = pending[next_submit]
                        in_flight[prefetch.submit(_prefetch_cadence, preprocessor, next_unit)] = (
                            next_unit
                        )
                        next_submit += 1

                    if cadence_result is None:
                        logger.info(f"Cadence {unit.group.key}: no stamps produced; skipping")
                        continue

                    # Inference-stage failure containment: log, record in the manifest,
                    # move on — one bad cadence must not abort the catalog. The pass
                    # raises after the loop so the retry loop re-attempts failed cadences.
                    try:
                        results = _infer_cadence(
                            pipeline,
                            preprocessor,
                            unit,
                            cadence_result,
                            current_fingerprint,
                            cadence_data=cadence_data,
                        )
                        # Release the prefetched array as soon as _infer_cadence returns
                        # (its own del only clears the callee's reference): with
                        # prefetch_depth cadences already loading behind this one, holding
                        # it until the next iteration's rebind would stack an extra
                        # cadence-sized array on peak RAM
                        del cadence_data
                    except Exception as e:
                        logger.error(
                            f"Cadence {cadence_result.key}: inference stage failed ({e}); "
                            f"continuing with remaining cadences"
                        )
                        failed_keys.append(cadence_result.key)
                        db.write_inference_cadence(
                            npy_path=cadence_result.npy_path,
                            status="failed",
                            tag=tag,
                            csv_path=unit.group.csv_path,
                            cadence_key=cadence_result.key,
                            # Despite the historical field name, CadenceResult.n_hits is
                            # the .npy's stamp-row count — the same quantity as the
                            # 'preprocessed'/'inferred' rows' n_stamps, so the manifest
                            # stays consistent across all three statuses.
                            n_stamps=cadence_result.n_hits,
                        )
                        continue

                    totals["n_cadence_snippets"] += results["n_cadence_snippets"]
                    totals["n_processed"] += results["n_processed"]
                    totals["n_candidates"] += results["n_candidates"]
                    totals["n_cadences"] += 1

                    if collector is not None:
                        # Same contract as record_skipped above: an exception here would
                        # abort the whole pass after the cadence's inference already
                        # succeeded — swallow it and degrade the plots instead
                        try:
                            collector.record_processed(
                                cadence_result.key,
                                cadence_result.npy_path,
                                cadence_result.metadata_path,
                                results["provenance"],
                                results,
                                results["duration_s"],
                                catalog_index=unit.index,
                            )
                        except Exception as e:
                            logger.error(
                                f"Viz collection failed for cadence {cadence_result.key} "
                                f"({e}); plots will be degraded; run continues"
                            )

                    # Stamp-cache pruning (#302): strictly after the 'inferred' manifest
                    # row (inside _infer_cadence) and viz collection above, and only for
                    # stamps THIS run freshly extracted — a resumed run never deletes a
                    # cache it was handed
                    if prune_stamps and cadence_result.freshly_extracted:
                        _prune_cadence_stamps(cadence_result, results, collector)

                    logger.info(
                        f"Cadence {cadence_result.key} ({totals['n_cadences']} done, "
                        f"{len(pending) - n_consumed} to go): {results['n_processed']} snippets, "
                        f"{results['n_candidates']} candidate(s)"
                    )
                    del results
        finally:
            preprocessor.stop_energy_detection_pool()
            # Release TF dataset/iterator state once per run, after the loaded models are done
            tf.keras.backend.clear_session()
            logger.info("Cleared TensorFlow session state")

        if failed_keys:
            # Whole-pass retry over the remaining failed set: cadences that completed are
            # skipped via their 'inferred' manifest rows on the next attempt
            raise RuntimeError(
                f"Inference stage failed for {len(failed_keys)} cadence(s): {failed_keys}"
            )

        # Reference cloud (#282): MC-score the seeded reject reservoir once per successful
        # pass. On a resumed run only the final attempt's cadences feed the reservoir
        # (manifest-skipped cadences never re-offer their rejects) — the npz records the
        # subsample size and rejects seen for provenance. Best-effort: the cloud degrades
        # the uncertainty plot, never the science.
        try:
            with stage_timer("inference.reference_cloud"):
                pipeline.finalize_reference_cloud()
        except Exception as e:
            logger.error(
                f"Reference-cloud finalization failed ({e}); the candidate uncertainty "
                f"plot will lack the survey background; run continues"
            )

    if totals["n_cadences"] == 0:
        # Preserve the historical contract: preprocessing producing no stamp .npy at all is
        # an error (bad paths/catalog), not a legitimate empty result
        raise NonRetryableInferenceError("No cadence results produced by preprocessing")

    if collector is not None:
        # Every figure is individually exception-guarded — a plot bug can't fail the pass
        with stage_timer("inference.viz"):
            render_inference_visualizations(collector, preprocessor, totals)

    return totals


def _run_legacy_test_files_inference(
    preprocessor: DataPreprocessor, strategy: tf.distribute.Strategy
) -> dict:
    """Legacy --test-files path: load + infer in one shot. The load repeats on retry (the
    old cross-attempt cadence_data cache is gone — the manifest made it obsolete on the
    streaming path, and holding a catalog-sized array across attempts was its only
    remaining use).

    Before the fresh positives land, any inference_results rows a dead attempt wrote for
    this npy_path are retired — the legacy counterpart of _infer_cadence's
    supersede-on-retry step 1; without it a failure after partial writes plus a retry
    would leave duplicate live candidate rows for the same npy_path.
    """
    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")
    db = get_db()
    if db is None:
        raise ValueError("get_db() returned None")
    npy_path = config.data.test_files[0]  # TODO: handle multiple test_files properly

    with stage_timer("inference.load_lognorm"):
        cadence_data = preprocessor.load_inference_data().astype(np.float32)

    db.mark_superseded("inference_results", config.checkpoint.save_tag, npy_path=npy_path)
    results = run_inference_pipeline(
        cadence_data=cadence_data,
        npy_path=npy_path,
        strategy=strategy,
    )
    del cadence_data
    return results


def _log_report_exclusion_summary(config) -> None:
    """Report-time frequency exclusion tally (#395): with ranges configured, append the
    excluded/reported split under the final candidate count, sourced from the whole tag's
    live candidate rows. On a normal run this matches the totals (which also aggregate
    resumed cadences); reusing a tag across DIFFERENT catalogs can leave live rows the
    current catalog never supersedes, so this tag-wide tally is the more inclusive count.
    Best-effort — a tally bug must never fail a completed science run."""
    ranges = report_exclusion_ranges(config)
    if not ranges:
        return
    try:
        db = get_db()
        if db is None:
            return
        db.flush()
        rows = db.query_inference_result(
            tag=config.checkpoint.save_tag, prediction=1, columns=["frequency_mhz"]
        )
        reported, excluded = partition_candidates_by_frequency(rows, ranges)
        range_label = ", ".join(f"{start:g}-{end:g}" for start, end in ranges)
        logger.info(
            f"    Excluded by --report-exclude-frequency-range ({range_label} MHz): {len(excluded)}"
        )
        logger.info(f"    Reported after exclusion: {len(reported)}")
    except Exception as e:
        logger.error(f"Report-time exclusion tally failed ({e}); run is unaffected")


# NOTE: we need to load the saved config from the corresponding training run, but when/where should we do that, and how does that play with apply_args_to_config()?
def inference_command():
    """Execute inference pipeline with distributed strategy & fault tolerance"""
    logger.info("=" * 60)
    logger.info("Starting Aetherscan Inference Pipeline")
    logger.info("=" * 60)

    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")

    # The artifact trio (encoder/rf/config paths) is populated upstream — either explicit
    # local paths validated by collect_validation_errors() in cli.py, or HuggingFace-
    # downloaded cache paths from resolve_inference_artifacts() — and stamp_width ==
    # width_bin is enforced by validation, so by the time inference_command() runs those
    # preconditions are guaranteed to hold.
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

    # Run preprocessing + inference with fault tolerance.
    # Recovery is state-based, not checkpoint-based: preprocessing resumes off each
    # cadence's on-disk stamp .npy, and the inference stage resumes off the
    # inference_cadences run manifest in the DB (a live status='inferred' row means the
    # cadence is skipped entirely) — so a retry, in-process or a full relaunch of the
    # identical command, re-runs only what the previous attempt didn't finish.
    preprocessor = DataPreprocessor()
    max_retries = config.inference.max_retries
    retry_delay = config.inference.retry_delay
    results = None
    # Run-scoped stamp-gallery pixel pool: persists across the retry attempts below so a
    # cadence pruned in an earlier attempt still renders in the final attempt's gallery
    # (#305). Bounded to top-K pixels (~2.4 MB); the preprocessor likewise persists, so its
    # freshly-extracted set survives too.
    gallery_pool: list = []

    for attempt in range(max_retries):
        try:
            logger.info(f"Inference attempt: {attempt + 1}/{max_retries}")

            if config.data.inference_files is not None:
                # Streaming CSV path: per-cadence preprocess -> load -> encode -> RF ->
                # write, with models loaded once and inference.prefetch_depth cadences in
                # flight (see _run_streaming_csv_inference). Memory stays independent of
                # catalog size.
                results = _run_streaming_csv_inference(preprocessor, strategy, gallery_pool)
            else:
                if not config.data.test_files:
                    logger.error(
                        "Neither --inference-files nor --test-files is configured; "
                        "nothing to load for inference"
                    )
                    sys.exit(1)
                # Legacy --test-files path: load + infer in one shot, with stale rows from
                # a dead attempt superseded first (see _run_legacy_test_files_inference)
                results = _run_legacy_test_files_inference(preprocessor, strategy)
            break  # success

        except KeyboardInterrupt:
            # Don't retry on user interruption; re-raise to propagate traceback
            logger.info("Inference interrupted by user")
            raise

        except NonRetryableInferenceError as e:
            # Permanent failure (empty/invalid catalog): retrying can't fix it, so fail
            # fast instead of burning the remaining attempts
            logger.error(f"Inference failed permanently: {e}")
            sys.exit(1)

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
    config_path = os.path.join(
        config.output_path,
        f"config_{display_tag(config.checkpoint.save_tag, get_machine_name())}.json",
    )
    os.makedirs(os.path.dirname(config_path), exist_ok=True)  # Create dir if it doesn't exist

    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Inference configuration saved to {config_path}")

    logger.info("=" * 60)
    logger.info("Inference completed successfully!")
    logger.info("Summary:")
    if "n_cadences" in results:
        logger.info(f"  Total cadences: {results['n_cadences']}")
    if results.get("n_skipped"):
        logger.info(f"    Resumed from a previous attempt (skipped): {results['n_skipped']}")
    logger.info(f"  Total cadence snippets: {results['n_cadence_snippets']}")
    logger.info(f"    Processed: {results['n_processed']}")
    logger.info(f"    Candidates found: {results['n_candidates']}")
    _log_report_exclusion_summary(config)
    logger.info("=" * 60)

    _post_benchmark_report(config.checkpoint.save_tag)
    _post_perband_report(config.checkpoint.save_tag)


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
        # Note, can't log before init_logger() — write to stderr so the failure isn't silent
        # (print() is banned outside utils/ by the T20 lint rule).
        sys.stderr.write(f"Failed to initialize config: {e}\n")
        sys.exit(1)

    # Set up the CLI parser and parse arguments BEFORE init_logger so the logger can name this
    # run's log file with its effective save_tag (aetherscan_{save_tag}.log instead of a single
    # overwritten aetherscan.log). Arg parsing has no dependency on the logger/manager; it is
    # hoisted here only to resolve the tag. Like init_config above, these steps run before the
    # logger exists and so cannot log.
    try:
        parser = setup_argument_parser()
    except Exception as e:
        # Note, can't log before init_logger() — write to stderr so the failure isn't silent
        # (print() is banned outside utils/ by the T20 lint rule).
        sys.stderr.write(f"Failed to set up argument parser: {e}\n")
        sys.exit(1)

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # argparse calls sys.exit(2) on parse errors (invalid types, missing required args, etc.),
        # which is why we catch SystemExit instead of Exception. argparse already prints its own
        # error message + usage to stderr; we re-print full help and re-raise to preserve exit
        # code 2. No logger exists yet (see above), so nothing is logged here.
        if e.code == 2:  # argparse error (syntax/type error)
            parser.print_help()
        raise

    # Resolve the run's save-tag ONCE, here, at runtime: a full {command}_{datetime} --load-tag
    # resumes that run in place (its tag is adopted); otherwise {--save-tag prefix or subcommand}_
    # {datetime}. Done before init_logger() (which names the log file by tag) and validate_args()
    # so the log file, config, and every artifact share a single datetime. --save-tag stays a bare
    # prefix on `args` for validate_args to check; only the resolved full tag lands on the config.
    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")
    config.checkpoint.save_tag = resolve_save_tag(
        getattr(args, "command", None),
        getattr(args, "save_tag", None),
        getattr(args, "load_tag", None),
    )

    # Initialize logger, naming the run's log file with the resolved save_tag.
    try:
        init_logger(save_tag=config.checkpoint.save_tag)
        logger.info("Logger initialization successful, but not yet registered for cleanup.")
        logger.info("Awaiting resource manager initialization. Do not terminate the process!")
    except Exception as e:
        # Note, can't log if init_logger() fails
        sys.exit(1)

    # A full --load-tag resumes that run in place (its tag is adopted as the save_tag). If the user
    # ALSO passed an explicit --save-tag prefix, it was silently overridden — surface that so the
    # save-tag not taking effect isn't a mystery.
    if getattr(args, "save_tag", None) and config.checkpoint.save_tag == getattr(
        args, "load_tag", None
    ):
        logger.info(
            f"Resuming in place under --load-tag '{args.load_tag}' — the provided --save-tag "
            f"'{args.save_tag}' was ignored (a full --load-tag adopts that run's tag)."
        )

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

    # Inference mode: resolve the model artifact trio before anything reads it. When the
    # user provided none of --encoder-path/--rf-path/--config-path, the pinned
    # (--hf-revision) or latest-release revision is downloaded from the HuggingFace Hub
    # and the cached paths are written onto `args` — exactly as if they were passed on the
    # CLI — so validation, apply_saved_config, and the model-load path run unchanged.
    # Explicit local paths take precedence; a partial trio falls through to validate_args,
    # which reports the missing paths.
    if args.command == "inference":
        try:
            resolve_inference_artifacts(args)
        except Exception as e:
            logger.error(f"Failed to resolve inference model artifacts: {e}")
            sys.exit(1)

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

    # Auto-launch the live monitoring dashboard (opt out with --no-dashboard). Fully guarded:
    # a missing streamlit or a spawn failure only warns — the dashboard is optional observability
    # and must never abort the run.
    try:
        launch_dashboard()
    except Exception as e:
        logger.warning(f"Dashboard launch skipped: {e}")

    try:
        # Fail-early save-tag dedup guards: hard-stop before any expensive work when an
        # explicitly-provided --save-tag collides with a previous run's artifacts/DB rows
        # (resumable-run manifests exempt same-tag retries), and check the HF repo for a
        # tag collision when --hf-upload is set. --force-tag overrides.
        enforce_tag_guards(args)

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
