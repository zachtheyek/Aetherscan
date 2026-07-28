"""
Disk-backed (memmap) round datasets for Aetherscan training.

Each training round's data lives on disk as .npy memmaps under
{round_data_dir}/{save_tag}/round_{k:02d}/ instead of ~294 GB of main-process RAM:
workers write generated cadences straight into the memmaps, training reads them back
through np.load(mmap_mode="r"), and the OS page cache keeps steady-state reads at
RAM speed while remaining evictable under memory pressure (no more OOM kills).

This module owns:
- RoundDataPaths: the on-disk layout of one round's dataset.
- The atomic `.done` manifest protocol (written only after all workers finish, via
  .tmp -> os.replace like preprocessing's _extract_stamps_worker).
- Startup archive/reuse/delete semantics for a tag's round-data directory.
- RoundDataProducer: a dedicated process that generates round k+1 while round k
  trains, streaming stats back to the main process (DB writes stay in main — the
  DB queue is a thread queue.Queue, not process-safe).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import multiprocessing
import os
import queue
import re
import shutil
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from logging.handlers import QueueListener

import numpy as np
import psutil

from aetherscan.benchmark import record_stage, round_stage_name
from aetherscan.logger import init_worker_logging
from aetherscan.manager import get_manager

logger = logging.getLogger(__name__)

# The producer process is started with the SPAWN method, not fork: the training parent holds
# deep TF / NCCL / gRPC / CUDA thread state whose locks a forked child can inherit in a locked
# state — an early producer prototype deadlocked on a futex before reaching its own code. A
# spawned child runs a fresh interpreter with none of that baggage (its own pool workers are
# then forked from the clean single-threaded producer, which is safe).
_MP_CONTEXT = multiprocessing.get_context("spawn")

# Directory-name pattern for one round's dataset (1-based, mirrors checkpoint round_XX tags)
_ROUND_DIR_PATTERN = re.compile(r"^round_(\d+)$")

# Number of values sampled per array for the manifest's cheap corruption check
_MANIFEST_SAMPLE_COUNT = 8

# Pidfile written next to a tag's round dirs by RoundDataProducer.start() (removed on graceful
# shutdown). An ungraceful main-process death (SIGKILL/OOM) leaves the producer orphaned with
# an argv of `... spawn_main ...` that no process-discovery pattern can attribute to Aetherscan
# — this file is what lets utils/kill_pipeline.sh and _reap_stale_producer() find the tree.
_PRODUCER_PIDFILE = "producer.pid"

# How long the producer's request loop waits on its queue before re-checking that the parent
# process is still alive (see the ppid watch in _producer_main).
_PARENT_POLL_INTERVAL_S = 5.0


@dataclass(frozen=True)
class RoundDataPaths:
    """On-disk layout of one round's dataset: {main,true,false,labels}.npy, sibling
    {main,true,false}_lognorm.npy parameter arrays, + a .done manifest."""

    round_dir: str
    round_idx: int

    @classmethod
    def for_round(cls, base_dir: str, round_idx: int) -> RoundDataPaths:
        """Build the paths for 1-based round `round_idx` under `base_dir` (round_{k:02d})."""
        return cls(round_dir=os.path.join(base_dir, f"round_{round_idx:02d}"), round_idx=round_idx)

    @property
    def main_path(self) -> str:
        return os.path.join(self.round_dir, "main.npy")

    @property
    def true_path(self) -> str:
        return os.path.join(self.round_dir, "true.npy")

    @property
    def false_path(self) -> str:
        return os.path.join(self.round_dir, "false.npy")

    @property
    def labels_path(self) -> str:
        return os.path.join(self.round_dir, "labels.npy")

    @property
    def done_path(self) -> str:
        return os.path.join(self.round_dir, f"round_{self.round_idx:02d}.done")

    @property
    def array_paths(self) -> dict[str, str]:
        """Mapping of array name -> .npy path for the three cadence arrays."""
        return {"main": self.main_path, "true": self.true_path, "false": self.false_path}

    @property
    def lognorm_paths(self) -> dict[str, str]:
        """Mapping of array name -> sibling .npy path holding that array's per-observation
        (min_log, range_log) log-norm parameters, shape (n_samples, num_observations, 2)."""
        return {
            name: os.path.join(self.round_dir, f"{name}_lognorm.npy")
            for name in ("main", "true", "false")
        }


def _array_checksum(arr: np.ndarray) -> dict:
    """
    Cheap, sha-less checksum for a (possibly huge) memmap: total element count plus a few
    deterministically-sampled flat-index values. Positions are derived from the element count,
    so validation re-samples the same spots without storing an RNG state.

    NOTE: this is a smoke test, not integrity protection. It reliably catches truncation and
    gross corruption (wrong element count, shape, swapped arrays), but sub-threshold damage
    slips through: a randomly corrupted fraction f of one array is only detected with
    probability 1-(1-f)^_MANIFEST_SAMPLE_COUNT (e.g. a single lost page in a ~90 GB array is
    ~never sampled). That is acceptable because a crash leaves NO manifest at all (the dir is
    then treated as garbage and regenerated), so the checksum only guards against same-run
    disk/NFS corruption between writing .done and reading it back — not power-loss/torn-page
    scenarios. If stronger guarantees are ever needed, upgrade to a full content hash.
    """
    # NOTE: reshape(-1) is a zero-copy view here (and in validate_done_manifest) because
    # open_memmap/np.save arrays are always C-contiguous; on a non-contiguous array it would
    # silently materialize a full in-RAM copy — switch to .ravel()/.flat if the layout ever changes
    flat = arr.reshape(-1)
    n = int(flat.size)
    rng = np.random.default_rng(n)
    positions = sorted({int(i) for i in rng.integers(0, n, size=min(_MANIFEST_SAMPLE_COUNT, n))})
    if arr.dtype.kind in ("U", "S"):
        samples = [[i, str(flat[i])] for i in positions]
    else:
        samples = [[i, float(flat[i])] for i in positions]
    return {"elements": n, "samples": samples}


def _checksum_value_matches(recorded, actual) -> bool:
    """Compare one manifest sample against the on-disk value (NaN == NaN counts as a match)."""
    if isinstance(recorded, str):
        return recorded == str(actual)
    actual = float(actual)
    return recorded == actual or (math.isnan(recorded) and math.isnan(actual))


def _manifest_paths(paths: RoundDataPaths) -> dict[str, str]:
    """Every array the .done manifest covers: cadence arrays, lognorm siblings, labels."""
    return {
        **paths.array_paths,
        **{f"{name}_lognorm": p for name, p in paths.lognorm_paths.items()},
        "labels": paths.labels_path,
    }


def build_manifest(
    paths: RoundDataPaths,
    n_samples: int,
    snr_base: float,
    snr_range: float,
    wall_time_s: float,
    chunk_count: int,
    array_dtype: str = "float32",
) -> dict:
    """Assemble the .done manifest dict by re-opening the finished arrays read-only.
    `array_dtype` is the requested cadence-array dtype; the per-array `dtypes` map records
    what is actually on disk (labels/lognorm sidecars stay float32/U regardless)."""
    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    checksums: dict[str, dict] = {}
    for name, path in _manifest_paths(paths).items():
        arr = np.load(path, mmap_mode="r")
        shapes[name] = list(arr.shape)
        dtypes[name] = str(arr.dtype)
        checksums[name] = _array_checksum(arr)
        del arr
    return {
        "round_idx": paths.round_idx,
        "n_samples": int(n_samples),
        "shapes": shapes,
        "dtypes": dtypes,
        "array_dtype": str(array_dtype),
        "snr_base": float(snr_base),
        "snr_range": float(snr_range),
        "checksums": checksums,
        "wall_time_s": float(wall_time_s),
        "chunk_count": int(chunk_count),
        "created_at": time.time(),
    }


def write_done_manifest(paths: RoundDataPaths, manifest: dict) -> None:
    """Write the manifest atomically (.tmp -> os.replace) so a crash can't leave a partial
    .done file — a round dir either has a complete manifest or is treated as garbage."""
    tmp_path = paths.done_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, paths.done_path)
    logger.info(f"Wrote round-data manifest: {paths.done_path}")


def validate_done_manifest(
    paths: RoundDataPaths,
    expected_n_samples: int | None = None,
    expected_array_dtype: str | None = None,
) -> dict | None:
    """
    Validate a round dir's .done manifest against the arrays on disk. Returns the manifest
    dict when everything checks out (round index, optional expected sample count, per-array
    existence, shape, element count, and the sampled checksum values), else None.

    NOTE: validation does NOT compare the manifest's snr_base/snr_range against what the caller
    wants — it only proves the arrays match the manifest that was written with them. Reusing a
    validated round dir is safe TODAY only because _setup_directories deletes every round dir
    >= start_round before generation begins, so a reused dir is always one written earlier in
    THIS run's curriculum. If cross-run round reuse is ever added (the obvious ~295 GB-saving
    optimization), a dir generated under a different SNR curriculum would validate and be
    silently trained on — add an expected_snr guard here (and at the _producer_main reuse
    short-circuit) before relaxing the delete-on-startup policy.
    """
    try:
        if not os.path.isfile(paths.done_path):
            return None
        with open(paths.done_path) as f:
            manifest = json.load(f)

        if manifest.get("round_idx") != paths.round_idx:
            logger.warning(f"Manifest round_idx mismatch in {paths.done_path}")
            return None
        if expected_n_samples is not None and manifest.get("n_samples") != expected_n_samples:
            logger.warning(
                f"Manifest n_samples ({manifest.get('n_samples')}) != expected "
                f"({expected_n_samples}) in {paths.done_path}"
            )
            return None
        # Cadence-array dtype gate: a dir generated under a different round_array_dtype must
        # not be silently reused (a float16 round fed to a float32-config resume — or vice
        # versa — changes input numerics mid-run). Manifests predating the dtypes key are all
        # float32 rounds.
        if expected_array_dtype is not None:
            manifest_dtype = manifest.get("array_dtype", "float32")
            if manifest_dtype != expected_array_dtype:
                logger.warning(
                    f"Manifest array_dtype ({manifest_dtype}) != expected "
                    f"({expected_array_dtype}) in {paths.done_path}"
                )
                return None

        manifest_dtypes = manifest.get("dtypes")
        for name, path in _manifest_paths(paths).items():
            if not os.path.isfile(path):
                logger.warning(f"Manifest array missing on disk: {path}")
                return None
            arr = np.load(path, mmap_mode="r")
            try:
                if list(arr.shape) != manifest["shapes"][name]:
                    logger.warning(f"Shape mismatch for {path}")
                    return None
                if manifest_dtypes is not None and str(arr.dtype) != manifest_dtypes[name]:
                    logger.warning(f"Dtype mismatch for {path}")
                    return None
                checksum = manifest["checksums"][name]
                if int(arr.size) != checksum["elements"]:
                    logger.warning(f"Element-count mismatch for {path}")
                    return None
                flat = arr.reshape(-1)
                for position, recorded in checksum["samples"]:
                    if not _checksum_value_matches(recorded, flat[position]):
                        logger.warning(f"Sampled-value mismatch for {path} at {position}")
                        return None
            finally:
                del arr

        return manifest
    except Exception as e:
        logger.warning(f"Failed to validate round-data manifest {paths.done_path}: {e}")
        return None


def load_round_arrays(paths: RoundDataPaths) -> dict[str, np.ndarray]:
    """
    Open one round's arrays for training. The three cadence arrays come back as
    copy-on-write memmaps (np.load(mmap_mode="c")) so nothing is pulled into RAM until
    batches gather it — the OS page cache keeps hot pages resident and evicts them under
    memory pressure instead of OOM-killing the process. mmap_mode="c" (MAP_PRIVATE) rather
    than "r": reads behave identically and the on-disk files stay write-protected, but the
    mapping is writable from numpy's point of view, which lets train._as_cpu_tensor export
    it zero-copy over dlpack for the graph-side gather (numpy refuses dlpack export of
    read-only arrays; nothing ever writes, so no private pages are materialized). Labels and
    the main array's log-norm parameters ("lognorm", consumed by the latent-traversal plot)
    are small (~24 MB for the main array at full scale) and loaded eagerly.
    """
    return {
        "concatenated": np.load(paths.main_path, mmap_mode="c"),
        "true": np.load(paths.true_path, mmap_mode="c"),
        "false": np.load(paths.false_path, mmap_mode="c"),
        "labels": np.load(paths.labels_path),
        "lognorm": np.load(paths.lognorm_paths["main"]),
    }


def _reap_stale_producer(base_dir: str, term_timeout: float = 5.0) -> None:
    """
    Restart-race guard: a previous run's producer that survived an ungraceful main-process
    death (see the pidfile written by RoundDataProducer.start()) may still be writing into
    this tag's round_XX dirs — racing prepare_round_data_dir's rmtree against those live
    writes corrupts the very dirs this run is about to regenerate or reuse. Read the recorded
    PID and terminate that tree (SIGTERM first, so the producer's handler reaps its own pool;
    then SIGKILL any survivors) before any deletion happens. Always removes the pidfile.
    """
    pidfile = os.path.join(base_dir, _PRODUCER_PIDFILE)
    if not os.path.exists(pidfile):
        return

    try:
        with open(pidfile) as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        pid = None

    if pid is not None:
        try:
            proc = psutil.Process(pid)
            # PID-reuse guard: the recorded producer was already alive when its pidfile was
            # written, so a process created after the pidfile's mtime cannot be it.
            if proc.create_time() <= os.path.getmtime(pidfile) + 1.0:
                logger.warning(
                    f"Reaping orphaned RoundDataProducer (PID {pid}) recorded in {pidfile}"
                )
                # Snapshot the subtree while the producer is alive: once it dies, its
                # children reparent and can no longer be found through its PID.
                children = proc.children(recursive=True)
                proc.terminate()
                with contextlib.suppress(psutil.TimeoutExpired):
                    proc.wait(timeout=term_timeout)
                for child in children:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        child.kill()
                with contextlib.suppress(psutil.NoSuchProcess):
                    proc.kill()
        except psutil.NoSuchProcess:
            pass  # already gone — the common case (producer exited via its own ppid watch)
        except Exception as e:
            logger.warning(f"Failed to reap producer recorded in {pidfile}: {e}")

    with contextlib.suppress(OSError):
        os.remove(pidfile)


def prepare_round_data_dir(
    base_dir: str, start_round: int, expected_array_dtype: str = "float32"
) -> None:
    """
    Startup cleanup of a tag's round-data directory, mirroring checkpoint-archiving semantics
    (train.archive_directory) for resumable runs — except entries are deleted rather than
    archived (a round is ~295 GB; archiving would silently double the disk budget):

    - any orphaned producer recorded in the tag's pidfile is reaped first (it could still be
      writing into the round dirs this cleanup is about to delete or validate),
    - round dirs with index >= start_round are deleted (regenerated by this run),
    - round dirs with index < start_round are kept only if their .done manifest validates
      (reusable instead of regenerated); .done-less or corrupt dirs are deleted,
    - any non-round entry (e.g. a stale RF dataset dir) is deleted.
    """
    os.makedirs(base_dir, exist_ok=True)
    _reap_stale_producer(base_dir)

    for entry in sorted(os.listdir(base_dir)):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        match = _ROUND_DIR_PATTERN.match(entry)
        if match is None:
            logger.info(f"Deleting stale round-data entry: {entry_path}")
            shutil.rmtree(entry_path, ignore_errors=True)
            continue
        round_idx = int(match.group(1))
        if round_idx >= start_round:
            logger.info(f"Deleting round-data dir for round >= {start_round}: {entry_path}")
            shutil.rmtree(entry_path, ignore_errors=True)
        elif (
            validate_done_manifest(
                RoundDataPaths(entry_path, round_idx), expected_array_dtype=expected_array_dtype
            )
            is None
        ):
            logger.info(f"Deleting round-data dir with missing/invalid manifest: {entry_path}")
            shutil.rmtree(entry_path, ignore_errors=True)
        else:
            logger.info(f"Keeping validated round-data dir: {entry_path}")


# ---------------------------------------------------------------------------
# RoundDataProducer — background generation process
# ---------------------------------------------------------------------------

# Producer-process-local handle to its worker pool, so the SIGTERM handler can
# terminate the pool's children before the producer itself dies.
_PRODUCER_POOL = None


def _default_generate(paths, round_idx, snr_base, snr_range, pool, params, stats_cb, progress_cb):
    """Real generation entry point for the producer process (test hooks can replace it)."""
    # Deferred import: data_generation pulls in setigen/scipy, which the parent may not need
    # at round_data import time (cli.py must stay stdlib-importable via config/cli only).
    from aetherscan.data_generation import generate_round_to_memmap  # noqa: PLC0415

    return generate_round_to_memmap(
        paths=paths,
        n_samples=params["n_samples"],
        snr_base=snr_base,
        snr_range=snr_range,
        width_bin=params["width_bin"],
        num_observations=params["num_observations"],
        time_bins=params["time_bins"],
        chunk_size=params["chunk_size"],
        task_size=params["task_size"],
        freq_resolution=params["freq_resolution"],
        time_resolution=params["time_resolution"],
        pool=pool,
        round_num=round_idx,
        seed=params["seed"],
        stats_cb=stats_cb,
        progress_cb=progress_cb,
        array_dtype=params.get("array_dtype", "float32"),
    )


def _producer_main(
    request_queue,
    result_queue,
    params: dict,
    generate_fn=None,
    log_queue=None,
    parent_pid: int | None = None,
    poll_interval: float = _PARENT_POLL_INTERVAL_S,
) -> None:
    """
    Producer process entry point (spawn-started — see _MP_CONTEXT). Owns a private worker pool
    (workers attach to the background-plate shared memory created by the main process) and
    generates rounds on request, isolated from the main process's GIL (TF's prefetch threads no
    longer slow generation down, and generation no longer steals cycles from training).

    Protocol (multiprocessing.Queues):
    - in:  ("generate", round_idx, snr_base, snr_range) | ("shutdown",)
    - out: ("stats", round_idx, segment_dict)            per class-segment per chunk
           ("progress", round_idx, chunk, n_chunks)      per chunk
           ("timing", round_idx, start_ts, end_ts)       generation wall-clock span, sent
                                                         before "done" (skipped on reuse —
                                                         no generation happened); the
                                                         main-process drainer records it as
                                                         a pipeline_stages row (this
                                                         process can't reach the DB writer
                                                         queue, a thread queue.Queue)
           ("done", round_idx, manifest)                 on success (or valid reuse)
           ("error", round_idx, traceback_str)           on failure (producer keeps serving)
           ("shutdown_ack",)                             right before a graceful exit

    `log_queue` is a spawn-context multiprocessing queue relayed into the main process's
    logging pipeline by RoundDataProducer.start() — a spawned child has no inherited Logger
    singleton, so it (and its pool workers) attach QueueHandlers to this queue explicitly.
    `generate_fn` is a test seam: when provided, no worker pool is created and the stub is
    called in place of the real memmap generation (see tests/unit/test_round_data.py).

    `parent_pid` is the PID of the process this producer must not outlive (passed by
    RoundDataProducer.start(); defaults to os.getppid() for the thread-driven tests, where
    it never changes). The request loop polls it every `poll_interval` seconds: an
    ungraceful parent death (SIGKILL/OOM — no cleanup runs, no "shutdown" is ever sent)
    reparents this process, and the watch terminates the pool and exits instead of leaving
    an orphan generating at full CPU and pinning the background SHM forever.
    """
    global _PRODUCER_POOL

    if parent_pid is None:
        parent_pid = os.getppid()

    is_main_thread = threading.current_thread() is threading.main_thread()
    if is_main_thread:
        # The log queue is a multiprocessing.Queue — safe to use from this process.
        init_worker_logging(log_queue)

        # Ignore SIGINT: the main process's ResourceManager coordinates Ctrl-C cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # SIGTERM handler mirrors data_generation._init_worker's:
        # WARN: DO NOT LOG ANYTHING IN THIS HANDLER — the log QueueHandler's feeder thread
        # needs the GIL, which may be held elsewhere, and a blocked handler deadlocks
        # termination (see data_generation.py's canonical worker handler).
        def _cleanup_on_sigterm(signum, frame):
            with contextlib.suppress(Exception):
                if _PRODUCER_POOL is not None:
                    # Forward termination to the pool's children so they don't outlive us
                    # (SIGKILL on this process would otherwise orphan them).
                    _PRODUCER_POOL.terminate()
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)

        signal.signal(signal.SIGTERM, _cleanup_on_sigterm)

        if sys.platform == "linux":
            # Belt-and-braces (Linux only): ask the kernel to SIGTERM this process the moment
            # its parent dies — covers mid-generation parent death, when the request-loop
            # ppid watch below isn't polling. PR_SET_PDEATHSIG is set here in the child's
            # main entry (it fires on exit of the *thread* that set it, and is cleared
            # across this process's own fork() — the pool workers don't inherit it, but the
            # SIGTERM handler above terminates the pool for them).
            with contextlib.suppress(Exception):
                import ctypes  # noqa: PLC0415

                _pr_set_pdeathsig = 1  # linux/prctl.h
                ctypes.CDLL("libc.so.6", use_errno=True).prctl(
                    _pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0
                )

    pool = None
    if generate_fn is None:
        generate_fn = _default_generate
        # Deferred import (setigen/scipy) — only the real producer needs it.
        from aetherscan.data_generation import _init_worker  # noqa: PLC0415

        # Plain fork-context Pool: this spawned process is single-threaded, so forking its
        # workers is safe — and they attach to the parent-created background SHM by name.
        pool = multiprocessing.Pool(
            processes=max(1, params["n_processes"]),
            initializer=_init_worker,
            initargs=(
                params["shm_name"],
                tuple(params["background_shape"]),
                np.dtype(params["background_dtype"]),
                log_queue,
            ),
        )
        _PRODUCER_POOL = pool

    logger.info(f"RoundDataProducer started (PID {os.getpid()})")

    try:
        while True:
            try:
                message = request_queue.get(timeout=poll_interval)
            except queue.Empty:
                # Parent-death watch: reparenting (to init or a subreaper) changes our ppid.
                # Exit without a shutdown_ack — there is no one left to ack to.
                if os.getppid() != parent_pid:
                    logger.warning(
                        "RoundDataProducer: parent process died; terminating pool and exiting"
                    )
                    break
                continue
            if message[0] == "shutdown":
                logger.info("RoundDataProducer received shutdown request")
                result_queue.put(("shutdown_ack",))
                break
            if message[0] != "generate":
                logger.warning(f"RoundDataProducer ignoring unknown message: {message[0]!r}")
                continue

            _, round_idx, snr_base, snr_range = message
            paths = RoundDataPaths.for_round(params["base_dir"], round_idx)

            existing = validate_done_manifest(
                paths,
                expected_n_samples=params["n_samples"],
                expected_array_dtype=params.get("array_dtype", "float32"),
            )
            if existing is not None:
                logger.info(f"RoundDataProducer: reusing validated round {round_idx} data")
                result_queue.put(("done", round_idx, existing))
                continue

            try:
                logger.info(
                    f"RoundDataProducer: generating round {round_idx} data "
                    f"(SNR {snr_base}-{snr_base + snr_range})"
                )
                generation_start = time.time()
                manifest = generate_fn(
                    paths,
                    round_idx,
                    snr_base,
                    snr_range,
                    pool,
                    params,
                    lambda segment, _r=round_idx: result_queue.put(("stats", _r, segment)),
                    lambda chunk, n_chunks, _r=round_idx: result_queue.put(
                        ("progress", _r, chunk, n_chunks)
                    ),
                )
                logger.info(f"RoundDataProducer: round {round_idx} data complete")
                # Timing before done: queue FIFO means the drainer records the stage span
                # before await_round() unblocks on the done message
                result_queue.put(("timing", round_idx, generation_start, time.time()))
                result_queue.put(("done", round_idx, manifest))
            except Exception:
                result_queue.put(("error", round_idx, traceback.format_exc()))
    finally:
        _PRODUCER_POOL = None
        if pool is not None:
            with contextlib.suppress(Exception):
                pool.terminate()
                pool.join()


class RoundDataProducer:
    """
    Main-process handle for the background round-data generation process.

    Lifecycle: start() spawns the producer process (registered with ResourceManager for
    terminate -> join -> kill cleanup) plus a drainer thread that consumes the result queue —
    writing streamed injection stats to the DB from the main process (the DB writer queue is
    a thread queue.Queue, not process-safe) while the GPUs train. request_generation() is
    fire-and-forget; await_round() blocks until that round's done/error message arrives.
    """

    def __init__(
        self,
        *,
        base_dir: str,
        n_samples: int,
        shm_name: str,
        background_shape: tuple,
        background_dtype: str,
        n_processes: int,
        width_bin: int,
        num_observations: int,
        time_bins: int,
        chunk_size: int,
        task_size: int,
        freq_resolution: float,
        time_resolution: float,
        db,
        tag: str,
        seed: int | None = None,
        array_dtype: str = "float32",
    ):
        self._params = {
            "base_dir": base_dir,
            "n_samples": n_samples,
            "shm_name": shm_name,
            "background_shape": tuple(background_shape),
            "background_dtype": str(background_dtype),
            "n_processes": n_processes,
            "width_bin": width_bin,
            "num_observations": num_observations,
            "time_bins": time_bins,
            "chunk_size": chunk_size,
            "task_size": task_size,
            "freq_resolution": freq_resolution,
            "time_resolution": time_resolution,
            # Pipeline root seed (config.reproducibility.seed) — crosses the spawn boundary with the
            # rest of the params so producer-generated rounds derive the same per-round
            # streams as in-process generation. None = OS entropy
            "seed": seed,
            # On-disk dtype for the cadence arrays (config.training.round_array_dtype)
            "array_dtype": str(array_dtype),
        }
        self._db = db
        self._tag = tag
        # Queues come from the spawn context so they can cross the spawn pickling boundary
        # (mixing contexts raises "A SemLock created in a fork context is being shared with a
        # process in a spawn context" — which also rules out handing the child the Logger
        # singleton's fork-context queue directly; see the relay in start())
        self._request_queue: multiprocessing.Queue = _MP_CONTEXT.Queue()
        self._result_queue: multiprocessing.Queue = _MP_CONTEXT.Queue()
        self._producer_log_queue: multiprocessing.Queue = _MP_CONTEXT.Queue()
        self._log_relay: QueueListener | None = None
        self._process: multiprocessing.Process | None = None
        self._drainer: threading.Thread | None = None
        self._drainer_done = False
        self._results: dict[int, tuple[str, object]] = {}
        self._condition = threading.Condition()
        self._pidfile = os.path.join(base_dir, _PRODUCER_PIDFILE)

    def start(self) -> None:
        """Spawn the producer process and the main-side result drainer thread."""
        # A spawned child has no inherited Logger singleton, and the singleton's fork-context
        # queue can't cross the spawn boundary — so the producer tree logs into its own
        # spawn-context queue, and this main-side QueueListener relays each record into the
        # main process's root handlers (i.e. into the normal file/console/Slack pipeline).
        self._log_relay = QueueListener(
            self._producer_log_queue,
            *logging.getLogger().handlers,
            respect_handler_level=False,
        )
        self._log_relay.start()

        self._process = _MP_CONTEXT.Process(
            target=_producer_main,
            args=(
                self._request_queue,
                self._result_queue,
                self._params,
                None,
                self._producer_log_queue,
                # This process's real PID, not the child's getppid(): if the parent dies
                # before the child captures its ppid, the child would capture the reaper's
                # PID and its parent-death watch could never fire.
                os.getpid(),
            ),
            name="RoundDataProducer",
        )

        # The spawned child re-imports the aetherscan module chain, which includes TF; blank
        # out GPU visibility for the child's entire process tree so nothing in the producer
        # can ever initialize CUDA (generation is pure CPU).
        # NOTE: another thread reading os.environ during this brief window would see the
        # blanked value — intentionally acceptable ONLY because the parent's TF read the var
        # once at its own GPU init (long before this point) and nothing else in the pipeline
        # spawns GPU-visible subprocesses concurrently with training startup.
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            self._process.start()
        finally:
            if original_cuda_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible

        # Record the producer PID for post-mortem discovery: the orphan left by an ungraceful
        # main-process death has a bare spawn_main argv that utils/kill_pipeline.sh's pattern
        # can't attribute to Aetherscan, so the kill script (and the next run's
        # _reap_stale_producer) read this file instead. Removed on graceful shutdown.
        try:
            os.makedirs(self._params["base_dir"], exist_ok=True)
            with open(self._pidfile, "w") as f:
                f.write(str(self._process.pid))
        except OSError as e:
            logger.warning(f"Failed to write producer pidfile {self._pidfile}: {e}")

        manager = get_manager()
        if manager is not None:
            manager.register_process(self._process, name="RoundDataProducer")

        self._drainer = threading.Thread(
            target=self._drain_results, name="RoundDataDrainer", daemon=True
        )
        self._drainer.start()
        logger.info(f"RoundDataProducer process started (PID {self._process.pid})")

    def request_generation(self, round_idx: int, snr_base: float, snr_range: float) -> None:
        """Queue generation of 1-based round `round_idx` (non-blocking)."""
        logger.info(f"Requesting background generation of round {round_idx} data")
        self._request_queue.put(("generate", round_idx, snr_base, snr_range))

    def await_round(self, round_idx: int) -> dict:
        """
        Block until round `round_idx`'s done/error message arrives; return the manifest on
        success, raise RuntimeError (carrying the producer traceback) on generation failure
        or if the producer died without answering.
        """
        while True:
            with self._condition:
                if round_idx in self._results:
                    kind, payload = self._results[round_idx]
                    break
                if self._drainer_done:
                    raise RuntimeError(
                        f"RoundDataProducer exited before producing round {round_idx} data"
                    )
                # NOTE: the drainer's notify_all() wakes this immediately on done/error; the
                # 1 s timeout is not a polling latency, just a liveness re-check so a drainer
                # that died without notifying can't strand this wait forever
                self._condition.wait(timeout=1.0)

        if kind == "error":
            raise RuntimeError(f"Round {round_idx} data generation failed in producer:\n{payload}")
        return payload

    def shutdown(self, timeout: float = 10.0) -> None:
        """Request a graceful producer exit, escalating to terminate/kill via the manager's
        ManagedProcess if the producer doesn't wind down in time (e.g. mid-generation)."""
        if self._process is None:
            return

        with contextlib.suppress(Exception):
            self._request_queue.put(("shutdown",))
        self._process.join(timeout)

        manager = get_manager()
        if manager is not None:
            # No-op join if already exited; terminate -> join -> kill escalation otherwise.
            manager.close_process(self._process)
        elif self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout)

        if self._drainer is not None:
            self._drainer.join(timeout=timeout)
        if self._log_relay is not None:
            with contextlib.suppress(Exception):
                self._log_relay.stop()
            self._log_relay = None
        self._process = None
        # Graceful shutdown: the pidfile only exists to mark an orphan candidate.
        with contextlib.suppress(OSError):
            os.remove(self._pidfile)
        logger.info("RoundDataProducer shut down")

    def _handle_message(self, message: tuple) -> None:
        """Dispatch one producer message (runs on the drainer thread)."""
        kind = message[0]
        if kind == "stats":
            # Deferred import: keep round_data importable without the setigen stack.
            from aetherscan.data_generation import write_segment_stats  # noqa: PLC0415

            _, _round_idx, segment = message
            write_segment_stats(self._db, self._tag, segment)
        elif kind == "progress":
            _, round_idx, chunk, n_chunks = message
            logger.info(f"Round {round_idx} data generation: chunk {chunk}/{n_chunks} complete")
        elif kind == "timing":
            # The producer's generation wall-clock span, recorded from this (main) process
            # because the DB writer queue is thread-only
            _, round_idx, start_ts, end_ts = message
            record_stage(
                f"{round_stage_name(round_idx)}.data_generation",
                start_ts,
                end_ts,
                tag=self._tag,
                metadata={"source": "producer"},
            )
        elif kind in ("done", "error"):
            _, round_idx, payload = message
            with self._condition:
                self._results[round_idx] = (kind, payload)
                self._condition.notify_all()
        else:
            logger.warning(f"RoundDataDrainer ignoring unknown message: {kind!r}")

    def _drain_results(self) -> None:
        """Drainer thread body: consume producer messages until shutdown-ack or producer death,
        then sweep any messages still buffered in the queue. A failure handling one message
        (e.g. a DB hiccup on a stats write) must not kill the drainer — done/error messages
        for in-flight rounds still need to unblock await_round()."""

        def _handle_safely(message: tuple) -> None:
            try:
                self._handle_message(message)
            except Exception as e:
                logger.error(f"RoundDataDrainer failed to handle {message[0]!r} message: {e}")

        while True:
            try:
                message = self._result_queue.get(timeout=1.0)
            except queue.Empty:
                if self._process is None or not self._process.is_alive():
                    break
                continue
            if message[0] == "shutdown_ack":
                break
            _handle_safely(message)

        # Final sweep: the producer may have exited with messages still in the pipe.
        while True:
            try:
                message = self._result_queue.get_nowait()
            except queue.Empty:
                break
            if message[0] != "shutdown_ack":
                _handle_safely(message)

        with self._condition:
            self._drainer_done = True
            self._condition.notify_all()
