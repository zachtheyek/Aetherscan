# TODO: add warning when aetherscan.db file size exceeds 1TB (at that point we should consider migrating to PostgreSQL or sharding the SQLite db)
# TODO: add functions to delete entries in db based on query results (e.g. query_system_resource -> DELETE)
"""
Database for Aetherscan Pipeline
Uses SQLite with asynchronous queue-based writes to handle concurrent data collection from multiple
processes safely
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import socket
import sqlite3
import threading
import time
from contextlib import contextmanager
from queue import Empty, Full, Queue
from typing import Any

import numpy as np

from aetherscan.config import get_config
from aetherscan.manager import register_db

logger = logging.getLogger(__name__)

# Unique sentinel object (i.e. an object with no attributes, no methods beyond those inherited from
# Python's base object type, and no meaningful equality semantics except identity) for flush
# requests - writer thread recognizes this as a command to flush immediately rather than data to be
# written
_FLUSH_SENTINEL = object()

# Command sentinel for mark_superseded() requests routed through the writer thread (same
# identity-based recognition as _FLUSH_SENTINEL, so single-writer semantics are preserved)
_MARK_SUPERSEDED_SENTINEL = object()

# Current schema version, stored in SQLite's PRAGMA user_version (0 on any pre-versioning db).
# Bump this and extend _migrate_schema() with an `if version < N:` block for future changes.
# v1: added `superseded INTEGER DEFAULT 0` to training_stats, injection_stats,
#     latent_snapshots, and inference_results (stale-data supersede semantics on retry)
# v2: added the `inference_cadences` table (per-cadence inference run manifest driving
#     stage-aware retries). New tables need no ALTER step — the CREATE TABLE IF NOT EXISTS
#     statements in _init_database() run before migration for old and new databases alike —
#     so the version bump exists to record the change and keep future `if version < N:`
#     blocks ordered.
# v3: added `inference_cadences.config_fingerprint` so the stage-aware resume only skips a
#     cadence whose stored 'inferred' row was written under the same inference config — guards
#     the reused-tag-with-changed-config stale-reuse footgun (the inference counterpart of the
#     training-side config_fingerprint guard).
# v4: added the `pipeline_stages` table (always-on stage timing spans from
#     aetherscan.benchmark, consumed by utils/benchmark_report.py and the monitor's
#     stage-band overlay). New table -> no ALTER step, same as v2.
# v5: added `inference_results.screening_proba` / `mc_mean` / `mc_std` (#282 two-pass
#     inference: the deterministic pass-1 score plus the seeded MC mean/spread that carries
#     the science threshold for survivors).
_SCHEMA_VERSION = 5


# Per-process cache for get_system_metadata(): every field (hostname, user, outbound IP, PID)
# is constant for the life of a process, and the uncached version opened a UDP socket per call
# — a real cost on the injection-stat hot path, which calls this tens of millions of times per
# round (#277). A spawned child process re-imports the module and rebuilds its own cache, so
# the PID is always correct.
_SYSTEM_METADATA_CACHE: str | None = None


def get_system_metadata() -> str:
    """
    Collects system metadata (machine name, user name, IP address, process ID)
    and returns it as a JSON string suitable for database storage. Cached per process —
    every field is process-constant.
    """
    global _SYSTEM_METADATA_CACHE
    if _SYSTEM_METADATA_CACHE is not None:
        return _SYSTEM_METADATA_CACHE

    # Machine and user info
    machine_name = socket.gethostname()
    user_name = getpass.getuser()

    # IP Address
    def _get_ip_address():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(
                ("8.8.8.8", 80)  # Doesn't have to be reachable; used to infer the outbound IP
            )
            return s.getsockname()[0]
        except Exception:
            return "Unknown"
        finally:
            s.close()

    ip_address = _get_ip_address()

    # Process ID
    process_id = os.getpid()

    # Pack into JSON
    metadata = {
        "machine_name": machine_name,
        "user_name": user_name,
        "ip_address": ip_address,
        "process_id": process_id,
    }

    # Use sorted keys for deterministic ordering (optional, good for diffs)
    _SYSTEM_METADATA_CACHE = json.dumps(metadata, sort_keys=True)
    return _SYSTEM_METADATA_CACHE


class Database:
    """
    Thread-safe SQLite database with asynchronous queue-based writes.

    Multiple threads/processes push records to a shared queue; a single background writer thread
    drains it and commits to SQLite at a configured interval. Serializing writes through one
    thread eliminates SQLITE_BUSY contention that would otherwise arise from concurrent writers.
    """

    _instance = None  # Stores singleton instance
    _lock = threading.Lock()  # Ensures thread safety on object initialization

    # __new__ allocates the object in memory (constructor at the object-creation level)
    # __init__ initializes the object's attributes after it's created
    # since __new__ is called before __init__ every time we instantiate a class,
    # by overriding __new__, we can short-circuit object creation entirely, and control whether a
    # new instance is created, or just return the existing instance
    def __new__(cls):
        # Double-checked locking pattern:
        # First check if _instance is None, without lock (for performance)
        if cls._instance is None:
            # If None, acquire the lock to serialize the initialization path,
            # preventing race conditions (2 threads violating singleton semantics)
            with cls._lock:
                # Check if _instance is None again inside the lock
                # (since multiple threads can be calling simultaneously)
                if cls._instance is None:
                    # If still None, only then we construct the singleton instance
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False  # Mark as not initialized (for __init__)
        # Return the same instance for all subsequent constructor calls
        return cls._instance

    def __init__(self):
        """Initialize database"""
        # Note, __init__ is triggered every time the class's constructor is called,
        # even if __new__ returned the existing singleton instance
        # Hence, we use the _initialized flag to make sure __init__ only runs once
        if self._initialized:
            return

        self._initialized = True

        self.config = get_config()
        if self.config is None:
            raise ValueError("get_config() returned None")

        self.db_path = os.path.join(self.config.output_path, "db", "aetherscan.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)  # Create dir if it doesn't exist

        self.get_connection_timeout = self.config.db.get_connection_timeout
        self.stop_writer_timeout = self.config.db.stop_writer_timeout
        self.write_interval = self.config.db.write_interval
        self.write_buffer_max_size = self.config.db.write_buffer_max_size
        self.write_retry_delay = self.config.db.write_retry_delay
        self.flush_timeout = self.config.db.flush_timeout
        self.bulk_chunk_rows = self.config.db.bulk_chunk_rows
        self.stop_drain_timeout = self.config.db.stop_drain_timeout

        self.write_queue = Queue()
        # Bulk lane (#277): bounded queue for high-volume injection-stat chunks. The bound
        # applies backpressure to the enqueuer (the round-data drainer / in-process data-gen
        # stats callback — background work that can afford to wait), never the training path,
        # and caps writer-side memory (the old single unbounded queue grew to ~35 GB of
        # anonymous RSS on the release run). flush() only drains the foreground write_queue,
        # so plot flushes no longer wait behind a round's worth of injection rows.
        self.bulk_queue = Queue(maxsize=self.config.db.bulk_queue_max_items)
        # Pending bulk rows per round_number (None keyed as -1), so plot code can verify that
        # every injection row for rounds <= N is committed before rendering round N's figures
        self._bulk_pending_lock = threading.Lock()
        self._bulk_pending_by_round: dict[int, int] = {}
        self.writer_thread = None
        self.stop_event = threading.Event()  # Thread-safe flag for stopping
        self.drain_event = threading.Event()  # Graceful-shutdown flag: drain both lanes first

        # Initialize database schema
        self._init_database()

        logger.info(f"Database initialized at: {self.db_path}")
        db_stats = self.get_db_stats()
        for name, value in db_stats.items():
            logger.info(f"  {name}: {value}")
        logger.info(f"Write interval: {self.write_interval} seconds")
        logger.info(f"Max buffer size: {self.write_buffer_max_size} records")

    @classmethod
    def _reset(cls):
        """
        Teardown hook for the thread-safe singleton — discards the cached instance so the next
        constructor call yields a fresh one. Only safe to call after stop() has completed; calling
        it while the database is active leaves live threads holding a stale reference.
        """
        # Acquire lock to prevent race conditions
        with cls._lock:
            # Discard the singleton instance by removing the global reference
            # Guarantees the next constructor call will produce a fresh instance
            # Note, resources held by the old instance will remain alive unless explicitly closed beforehand
            cls._instance = None
            logger.info("Database singleton instance reset")

    # NOTE: should we consider migrating to PostgreSQL? or should we go all in on SQLite?
    def _init_database(self):
        """Create database tables if they don't exist, then run schema migrations"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # System resources table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_resources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    tag TEXT,
                    metadata TEXT
                )
            """)

            # NOTE: removing ORDER BY to reduce indexing costs
            # # Composite index optimized for query_system_resource ORDER BY pattern
            # # Also works for common filter patterns (tag + timestamp)
            # # Recall that composite indices follow the left-prefix rule
            # # That is, if your query contains a left prefix of the index columns,
            # # you'll still get (some of) the benefits of indexing
            # cursor.execute("""
            #     CREATE INDEX IF NOT EXISTS idx_system_resources_query
            #     ON system_resources(tag, timestamp, resource_type, resource_name)
            # """)

            # Composite index for common filter patterns (tag + timestamp)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_system_resources_filter
                ON system_resources(tag, timestamp)
            """)

            # Injection statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS injection_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    stat_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    round_number INTEGER,
                    chunk_number INTEGER,
                    sample_index INTEGER,
                    background_index INTEGER,
                    signal_class TEXT,
                    signal_type TEXT,
                    injection_stage TEXT,
                    is_finite INTEGER DEFAULT 1,
                    slope_clamped INTEGER DEFAULT 0,
                    tag TEXT,
                    metadata TEXT,
                    superseded INTEGER DEFAULT 0
                )
            """)

            # NOTE: removing ORDER BY to reduce indexing costs
            # # Composite index optimized for query_injection_stat ORDER BY pattern
            # cursor.execute("""
            #     CREATE INDEX IF NOT EXISTS idx_injection_stats_query
            #     ON injection_stats(tag, signal_class, signal_type, round_number, chunk_number, sample_index, stat_name)
            # """)

            # Composite index for common filter pattern (tag + timestamp + stat_name + signal_type + injection_stage)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_injection_stats_filter
                ON injection_stats(tag, timestamp, stat_name, signal_type, injection_stage)
            """)

            # Training statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    stat_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    round_number INTEGER,
                    epoch_number INTEGER,
                    tag TEXT,
                    metadata TEXT,
                    superseded INTEGER DEFAULT 0
                )
            """)

            # NOTE: removing ORDER BY to reduce indexing costs
            # # Composite index optimized for query_training_stat ORDER BY pattern
            # cursor.execute("""
            #     CREATE INDEX IF NOT EXISTS idx_training_stats_query
            #     ON training_stats(tag, model_name, round_number, epoch_number, stat_name)
            # """)

            # Composite index for common filter pattern (tag + timestamp + model_name + stat_name)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_stats_filter
                ON training_stats(tag, timestamp, model_name, stat_name)
            """)

            # Latent snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS latent_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    round_number INTEGER NOT NULL,
                    epoch_number INTEGER NOT NULL,
                    step_number INTEGER NOT NULL,
                    cadence_index INTEGER NOT NULL,
                    signal_type TEXT NOT NULL,
                    latent_vector TEXT NOT NULL,
                    snr_base INTEGER,
                    snr_range INTEGER,
                    tag TEXT,
                    metadata TEXT,
                    superseded INTEGER DEFAULT 0
                )
            """)

            # Composite index for common filter pattern (tag + timestamp + model_name + round_number + epoch_number + step_number)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_latent_snapshots_filter
                ON latent_snapshots(tag, timestamp, model_name, round_number, epoch_number, step_number)
            """)

            # Inference results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inference_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    npy_path TEXT NOT NULL,
                    snippet_index INTEGER NOT NULL,
                    prediction INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    latent_vector TEXT,
                    target TEXT,
                    session TEXT,
                    cadence_id INTEGER,
                    band TEXT,
                    frequency_mhz REAL,
                    timestamp_observed REAL,
                    h5_path TEXT,
                    tag TEXT,
                    metadata TEXT,
                    superseded INTEGER DEFAULT 0,
                    screening_proba REAL,
                    mc_mean REAL,
                    mc_std REAL
                )
            """)

            # Composite index for common filter pattern (tag + confidence + prediction)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_inference_results_filter
                ON inference_results(tag, timestamp, confidence, prediction)
            """)

            # Inference cadence manifest table (schema v2): one row per (cadence, stage
            # transition) — status 'preprocessed' when the stamp .npy lands, a superseding
            # 'inferred' row (with aggregate stats) when inference completes, and 'failed'
            # when the inference stage of a cadence dies. Drives stage-aware retry resume:
            # a live 'inferred' row means the cadence is skipped entirely on retry.
            # cadence_key and confidence_summary are JSON TEXT.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inference_cadences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    tag TEXT,
                    csv_path TEXT,
                    cadence_key TEXT,
                    npy_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    n_stamps INTEGER,
                    n_candidates INTEGER,
                    confidence_summary TEXT,
                    duration_s REAL,
                    config_fingerprint TEXT,
                    superseded INTEGER DEFAULT 0
                )
            """)

            # Composite index for the resume lookup pattern (tag + npy_path + status)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_inference_cadences_filter
                ON inference_cadences(tag, npy_path, status)
            """)

            # Pipeline stage timing table (schema v4): one row per timed pipeline stage
            # span, written by aetherscan.benchmark (stage_timer / record_stage). stage is
            # a hierarchical dot-name ("train.round_02.data_generation"); metadata is an
            # optional JSON TEXT blob (e.g. {"status": "failed", ...} for spans that ended
            # in an exception). Retried stages simply append new rows — consumers see every
            # attempt, each with its own span.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_stages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stage TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration_s REAL NOT NULL,
                    tag TEXT,
                    metadata TEXT
                )
            """)

            # Composite index for the common filter pattern (tag + start_time)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pipeline_stages_filter
                ON pipeline_stages(tag, start_time)
            """)

            conn.commit()

            # Bring pre-existing databases (older schema) up to the current version
            self._migrate_schema(conn)

            # Enable Write-Ahead Logging (WAL) mode for better concurrent read performance
            # WAL places writes in a separate log file so reads can still go through while writes happen
            # The WAL log is periodically merged back into the main db (i.e. checkpointing)
            cursor.execute("PRAGMA journal_mode=WAL")

            logger.info("Database schema initialized with WAL mode")

    def _migrate_schema(self, conn):
        """
        Minimal schema migration gated on SQLite's PRAGMA user_version.

        Each block below upgrades one version step via additive ALTER TABLE ... ADD COLUMN
        statements (the only in-place table change SQLite supports); a per-table column-existence
        check keeps every step idempotent even if user_version was lost (e.g. a db file copied
        without its journal). Fresh databases already get the full schema from the CREATE TABLE
        statements in _init_database(), so migration only stamps their version.
        """
        cursor = conn.cursor()
        version = cursor.execute("PRAGMA user_version").fetchone()[0]

        if version >= _SCHEMA_VERSION:
            return

        if version < 1:
            # v1: stale-data supersede semantics — rows from failed/superseded attempts are
            # flagged (never deleted) so default queries can filter them out
            for table in (
                "training_stats",
                "injection_stats",
                "latent_snapshots",
                "inference_results",
            ):
                columns = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
                if "superseded" not in columns:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN superseded INTEGER DEFAULT 0")
                    logger.info(f"Schema migration: added {table}.superseded")

        # v2 (inference_cadences table) needs no migration step here: the table is created for
        # old and new databases alike by the CREATE TABLE IF NOT EXISTS statements in
        # _init_database(), which always runs before this method.

        if version < 3:
            # v3: config_fingerprint on inference_cadences. A fresh db already has the column
            # from the CREATE TABLE above; this ALTER patches a db that created the v2 table
            # before the column existed. The column-existence check keeps it idempotent.
            columns = {row[1] for row in cursor.execute("PRAGMA table_info(inference_cadences)")}
            if "config_fingerprint" not in columns:
                cursor.execute("ALTER TABLE inference_cadences ADD COLUMN config_fingerprint TEXT")
                logger.info("Schema migration: added inference_cadences.config_fingerprint")

        # v4 (pipeline_stages table) needs no migration step here either: like v2, the table is
        # created for old and new databases alike by the CREATE TABLE IF NOT EXISTS statement in
        # _init_database(). Only the version stamp below advances.

        if version < 5:
            # v5: two-pass inference columns (#282). A fresh db already has them from the
            # CREATE TABLE above; the ALTERs patch a pre-v5 db. Column-existence checks keep
            # each step idempotent.
            columns = {row[1] for row in cursor.execute("PRAGMA table_info(inference_results)")}
            for column in ("screening_proba", "mc_mean", "mc_std"):
                if column not in columns:
                    cursor.execute(f"ALTER TABLE inference_results ADD COLUMN {column} REAL")
                    logger.info(f"Schema migration: added inference_results.{column}")

        # PRAGMA doesn't support parameter binding; _SCHEMA_VERSION is a module-level int constant
        cursor.execute(f"PRAGMA user_version = {_SCHEMA_VERSION:d}")
        conn.commit()
        logger.info(f"Database schema migrated from version {version} to {_SCHEMA_VERSION}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with proper cleanup"""
        conn = sqlite3.connect(self.db_path, timeout=self.get_connection_timeout)
        try:
            # Under WAL, synchronous=NORMAL only skips the per-commit WAL fsync (the WAL is
            # still synced at checkpoints) — a crash can lose the most recent commits but
            # never corrupts the database. That durability is ample for diagnostic telemetry
            # and removes the dominant per-transaction fsync stall (#277).
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            conn.close()

    def start(self):
        """Start the background writer thread"""
        if self.writer_thread is not None and self.writer_thread.is_alive():
            return

        self.stop_event.clear()
        self.drain_event.clear()
        # NOTE: should db be daemon or non-daemon thread?
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=False)
        self.writer_thread.start()
        logger.info("Database writer thread started")

    def _pending_row_counts(self) -> tuple[int, int]:
        """Approximate (foreground items, bulk rows) still queued — for drain progress logs."""
        with self._bulk_pending_lock:
            bulk_rows = sum(self._bulk_pending_by_round.values())
        return self.write_queue.qsize(), bulk_rows

    def stop(self):
        """
        Drain both write lanes to disk, then stop the background writer thread.

        The writer first empties the foreground and bulk queues (with progress heartbeats —
        the backlog can be large), bounded by stop_drain_timeout; only then does it exit. If
        the drain cap is hit, the writer is force-stopped and the exact number of dropped
        rows is logged at ERROR — never silently (#277: the old stop() discarded everything
        still queued, losing the most recent rows first).
        """
        if self.writer_thread is None:
            return

        fg_items, bulk_rows = self._pending_row_counts()
        logger.info(
            f"Stopping database writer thread (draining backlog: ~{fg_items} foreground "
            f"item(s), ~{bulk_rows} bulk row(s))..."
        )
        self.drain_event.set()  # Ask the writer to drain both lanes, then exit

        deadline = time.time() + self.stop_drain_timeout
        while self.writer_thread.is_alive() and time.time() < deadline:
            self.writer_thread.join(timeout=5.0)
            if self.writer_thread.is_alive():
                fg_items, bulk_rows = self._pending_row_counts()
                logger.info(
                    f"Draining DB backlog before shutdown: ~{fg_items} foreground item(s), "
                    f"~{bulk_rows} bulk row(s) remaining"
                )

        if self.writer_thread.is_alive():
            # Drain cap hit — force the writer out and account for every dropped row
            self.stop_event.set()
            self.writer_thread.join(timeout=self.stop_writer_timeout)
            fg_items, bulk_rows = self._pending_row_counts()
            logger.error(
                f"Database writer did not drain within {self.stop_drain_timeout}s; "
                f"dropping ~{fg_items} foreground item(s) and ~{bulk_rows} bulk row(s) "
                "still queued"
            )
            if self.writer_thread.is_alive():
                logger.warning("Database writer thread did not stop cleanly")
        else:
            self.stop_event.set()  # Writer already exited; keep post-stop flush()/writes short-circuiting
            logger.info("Database writer thread stopped (backlog drained)")

    def flush(self, timeout: float | None = None) -> bool:
        """
        Block until queued FOREGROUND writes are drained to the database.

        Returns False on timeout or if shutdown is initiated mid-wait. A None timeout falls back
        to the configured flush_timeout. Call before any read that needs to observe writes still
        sitting in the queue.

        The bulk injection-stat lane is deliberately NOT covered (#277): a flush must never
        queue behind a round's worth of bulk rows. Readers of injection_stats should gate on
        injection_backlog_rows() instead.
        """
        if timeout is None:
            timeout = self.flush_timeout

        # No-op if writer isn't running
        if self.writer_thread is None or not self.writer_thread.is_alive():
            logger.info("Flush called but writer thread not running")
            return True

        # Check if shutdown is already in progress
        if self.stop_event.is_set():
            logger.warning("Flush called during shutdown, skipping")
            return False

        # Create a completion event for this specific flush request
        flush_complete = threading.Event()

        # Put sentinel in queue - when writer sees this, it will flush and signal
        self.write_queue.put((_FLUSH_SENTINEL, flush_complete))

        # Block until writer signals completion, timeout, or shutdown
        # Use a loop with short waits to also check stop_event
        wait_interval = 0.1  # Check stop_event every 100ms
        elapsed = 0.0

        while elapsed < timeout:
            if flush_complete.wait(timeout=wait_interval):
                return True  # Flush completed successfully

            # Check if shutdown was initiated while waiting
            if self.stop_event.is_set():
                logger.warning("Shutdown initiated during flush, aborting wait")
                return False

            elapsed += wait_interval

        logger.warning(f"Database flush timed out after {timeout} seconds")
        return False

    # Tables that support supersede marking, mapped to the optional filters each accepts
    # (round_ge needs a round_number column; npy_path exists on inference_results and
    # inference_cadences)
    _SUPERSEDE_TABLES = {
        "training_stats": frozenset({"round_ge"}),
        "injection_stats": frozenset({"round_ge"}),
        "latent_snapshots": frozenset({"round_ge"}),
        "inference_results": frozenset({"npy_path"}),
        "inference_cadences": frozenset({"npy_path"}),
    }

    def mark_superseded(
        self,
        table: str,
        tag: str,
        *,
        round_ge: int | None = None,
        npy_path: str | None = None,
        timeout: float | None = None,
    ) -> bool:
        """
        Flag existing rows for `tag` as superseded (stale data from a failed attempt) so the
        default query_* filters ignore them. Rows are never deleted — pass
        include_superseded=True to a query method to inspect them.

        round_ge narrows the mark to round_number >= round_ge (training_stats /
        injection_stats / latent_snapshots only); npy_path narrows to one cadence file
        (inference_results / inference_cadences only).

        The command executes on the background writer thread (a command tuple through the
        write queue, like the flush sentinel) so single-writer semantics are preserved:
        buffered rows are flushed first, then the UPDATE runs — queue FIFO ordering
        guarantees every row queued before this call gets marked while later writes keep
        superseded = 0. Blocks until the mark lands; returns False on timeout or shutdown.
        When the writer thread isn't running there are no queued rows to order against, so
        the UPDATE executes synchronously in the caller thread.
        """
        if table not in self._SUPERSEDE_TABLES:
            raise ValueError(
                f"mark_superseded does not support table {table!r}; "
                f"expected one of {sorted(self._SUPERSEDE_TABLES)}"
            )
        allowed_filters = self._SUPERSEDE_TABLES[table]
        if round_ge is not None and "round_ge" not in allowed_filters:
            raise ValueError(f"round_ge is not supported for table {table!r}")
        if npy_path is not None and "npy_path" not in allowed_filters:
            raise ValueError(
                "npy_path is only supported for tables 'inference_results' and 'inference_cadences'"
            )
        if not tag:
            raise ValueError("mark_superseded requires a non-empty tag")

        if timeout is None:
            timeout = self.flush_timeout

        # No writer thread -> nothing queued to order against; run inline
        if self.writer_thread is None or not self.writer_thread.is_alive():
            self._execute_mark_superseded(table, tag, round_ge, npy_path)
            return True

        if self.stop_event.is_set():
            logger.warning("mark_superseded called during shutdown, skipping")
            return False

        mark_complete = threading.Event()
        self.write_queue.put(
            (_MARK_SUPERSEDED_SENTINEL, (table, tag, round_ge, npy_path), mark_complete)
        )

        # Block until the writer signals completion, timeout, or shutdown (same loop as flush)
        wait_interval = 0.1
        elapsed = 0.0
        while elapsed < timeout:
            if mark_complete.wait(timeout=wait_interval):
                return True
            if self.stop_event.is_set():
                logger.warning("Shutdown initiated during mark_superseded, aborting wait")
                return False
            elapsed += wait_interval

        logger.warning(f"mark_superseded timed out after {timeout} seconds")
        return False

    def _execute_mark_superseded(
        self, table: str, tag: str, round_ge: int | None, npy_path: str | None
    ) -> None:
        """Apply the supersede UPDATE (runs on the writer thread, or inline via
        mark_superseded when no writer thread is running). `table` was already validated
        against _SUPERSEDE_TABLES, so the f-string interpolation below is safe."""
        query = f"UPDATE {table} SET superseded = 1 WHERE superseded = 0 AND tag = ?"
        params: list = [tag]

        if round_ge is not None:
            query += " AND round_number >= ?"
            params.append(round_ge)

        if npy_path is not None:
            query += " AND npy_path = ?"
            params.append(npy_path)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                logger.info(
                    f"Marked {cursor.rowcount} row(s) superseded in {table} "
                    f"(tag={tag}, round_ge={round_ge}, npy_path={npy_path})"
                )
        except Exception as e:
            # Mirror _flush_buffer's log-and-continue semantics: a failed mark must not
            # kill the writer thread (stale rows would then merely reappear in plots)
            logger.error(f"Error marking rows superseded in {table}: {e}")

    def _consume_bulk_item(self) -> bool:
        """
        Pull one bulk chunk (if any), commit it together with any buffered foreground rows in
        a single transaction, and settle the pending-row accounting. Runs on the writer
        thread. Returns True if a chunk was consumed.
        """
        try:
            table, rows, round_counts = self.bulk_queue.get_nowait()
        except Empty:
            return False

        self.buffer.extend((table, values) for values in rows)
        self._flush_buffer()
        self.buffer.clear()

        # Settle the per-round pending accounting only after the commit attempt, so
        # injection_backlog_rows() == 0 really means "every enqueued row reached the db
        # engine" (_flush_buffer logs and drops a failed batch — those rows are gone either
        # way, so they must not linger in the backlog count)
        with self._bulk_pending_lock:
            for round_key, count in round_counts.items():
                remaining = self._bulk_pending_by_round.get(round_key, 0) - count
                if remaining > 0:
                    self._bulk_pending_by_round[round_key] = remaining
                else:
                    self._bulk_pending_by_round.pop(round_key, None)
        return True

    def _writer_loop(self):
        """
        Background loop that consumes data from both write lanes and writes to the database.

        The foreground write_queue (training stats, latent snapshots, sentinels, ...) has
        strict priority; the bounded bulk lane (injection-stat chunks) is serviced whenever
        the foreground lane is idle. On graceful shutdown (drain_event) both lanes are fully
        drained to disk before the thread exits (#277).
        """
        self.buffer = []
        last_write_time = time.time()

        # Keep looping until told to stop (hard) or drain (graceful)
        while not self.stop_event.is_set() and not self.drain_event.is_set():
            try:
                # Calculate how much time remains until the next scheduled write
                # Don't wait longer than 1s so we check the stop flag regularly
                # Don't wait more than 0.1s to avoid wasting CPU resources
                # Retrive items from the queue one-by-one & append to local buffer
                # When bulk chunks are waiting, poll the foreground lane briefly instead so
                # the bulk lane drains at full speed while foreground items still win ties
                if self.bulk_queue.empty():
                    timeout = max(
                        0.1, min(1.0, self.write_interval - (time.time() - last_write_time))
                    )
                else:
                    timeout = 0.01
                metric = self.write_queue.get(timeout=timeout)

                # Check for flush sentinel - signals immediate flush request
                if metric[0] is _FLUSH_SENTINEL:
                    flush_complete_event = metric[1]
                    # Flush current buffer immediately
                    if self.buffer:
                        # Write all buffered data to db
                        self._flush_buffer()
                        # Clear the buffer (empty the list)
                        self.buffer.clear()
                        # Reset the timer
                        last_write_time = time.time()
                    # Signal that flush is complete
                    flush_complete_event.set()
                    continue

                # Check for mark-superseded command. Flush buffered rows first: queue FIFO
                # guarantees every row enqueued before the command is already in the buffer,
                # so the UPDATE covers them, while rows enqueued after keep superseded = 0
                if metric[0] is _MARK_SUPERSEDED_SENTINEL:
                    _, payload, mark_complete_event = metric
                    try:
                        if self.buffer:
                            self._flush_buffer()
                            self.buffer.clear()
                            last_write_time = time.time()
                        self._execute_mark_superseded(*payload)
                    finally:
                        # Always unblock the caller, even if the flush/UPDATE errored
                        mark_complete_event.set()
                    continue

                self.buffer.append(metric)

                # Write when buffer is full or interval elapsed
                current_time = time.time()
                if (
                    len(self.buffer) >= self.write_buffer_max_size
                    or (current_time - last_write_time) >= self.write_interval
                ):
                    # Write all buffered data to db
                    self._flush_buffer()
                    # Clear the buffer (empty the list)
                    self.buffer.clear()
                    # Reset the timer
                    last_write_time = current_time

            except Empty:
                # Foreground lane idle — service one bulk chunk if available (commits the
                # current buffer alongside it), else fall back to the interval flush check
                if self._consume_bulk_item():
                    last_write_time = time.time()
                    continue
                # If get() timesout (queue was empty) but interval elapsed, write buffered data anyway
                current_time = time.time()
                if self.buffer and (current_time - last_write_time) >= self.write_interval:
                    # Write all buffered data to db
                    self._flush_buffer()
                    # Clear the buffer (empty the list)
                    self.buffer.clear()
                    # Reset the timer
                    last_write_time = current_time
                continue

            except Exception as e:
                logger.error(f"Error in db writer loop: {e}")
                # Sleep (interruptible for faster shutdown)
                self.stop_event.wait(self.write_retry_delay)

        # Graceful shutdown: drain BOTH lanes to disk before exiting (#277 — the old
        # implementation dropped everything still queued, silently losing the most recent
        # rows). A hard stop (stop_event without drain_event, or the drain cap escalation
        # in stop()) skips this and only flushes the in-memory buffer below.
        if self.drain_event.is_set() and not self.stop_event.is_set():
            self._drain_queues()

        # Final flush on shutdown
        if self.buffer:
            flushed_count = len(self.buffer)
            self._flush_buffer()
            self.buffer.clear()
            logger.info(f"Flushed {flushed_count} remaining data on shutdown")

    def _drain_queues(self) -> None:
        """Empty both write lanes to disk (graceful-shutdown path; see stop())."""
        drained_foreground = 0
        drained_bulk_chunks = 0
        while not self.stop_event.is_set():
            progressed = False
            try:
                metric = self.write_queue.get_nowait()
                progressed = True
                if metric[0] is _FLUSH_SENTINEL:
                    if self.buffer:
                        self._flush_buffer()
                        self.buffer.clear()
                    metric[1].set()
                elif metric[0] is _MARK_SUPERSEDED_SENTINEL:
                    _, payload, mark_complete_event = metric
                    try:
                        if self.buffer:
                            self._flush_buffer()
                            self.buffer.clear()
                        self._execute_mark_superseded(*payload)
                    finally:
                        mark_complete_event.set()
                else:
                    self.buffer.append(metric)
                    drained_foreground += 1
                    if len(self.buffer) >= self.write_buffer_max_size:
                        self._flush_buffer()
                        self.buffer.clear()
            except Empty:
                pass

            if self._consume_bulk_item():
                progressed = True
                drained_bulk_chunks += 1

            if not progressed:
                break

        if drained_foreground or drained_bulk_chunks:
            logger.info(
                f"Drained {drained_foreground} foreground row(s) and {drained_bulk_chunks} "
                "bulk chunk(s) to the database at shutdown"
            )

    # In commit 08fc37d, we switched from using sequential execute() to executemany()
    # The sequential approaches parses SQL statements N times & performs N round trips to the db
    # engine, in exchange for lower peak memory & more granular errors (can identify exactly which
    # row failed). In contrast, executemany() only parses SQL statements once & performs only 1
    # round trip to the db engine, but requires all rows to be in memory & fails the entire batch
    # if any row fails
    # In general, sequential execute() is preferred when db buffer sizes are small, if write
    # frequency is low, if each record in the buffer is guaranteed to go to different tables, or
    # if we require per-row error handling
    # In our case, we have a larger buffer (100+ records), most records in a batch will go to the
    # same table, and we're okay with all-or-nothing batch semantics. So the increased write speeds
    # in exchange for increased memory pressure is worth it
    # As well, since our db writes happen in a background thread & don't block the main process,
    # either approach should have minimal practical impact
    def _flush_buffer(self, buffer: list | None = None):
        """
        Write buffered data to database in a single transaction using executemany().

        Groups records by table and uses executemany() for bulk inserts, which is more efficient
        than individual execute() calls because the SQL is parsed once and reused for all rows.
        Defaults to the writer thread's buffer; an explicit `buffer` supports the inline
        no-writer path of write_injection_stats_bulk.
        """
        if buffer is None:
            buffer = self.buffer
        if not buffer:
            return

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Group records by table for bulk inserts
                system_resources_records: list[tuple] = []
                injection_stats_records: list[tuple] = []
                training_stats_records: list[tuple] = []
                latent_snapshots_records: list[tuple] = []
                inference_results_records: list[tuple] = []
                inference_cadences_records: list[tuple] = []
                pipeline_stages_records: list[tuple] = []

                for table, values in buffer:
                    if table == "system_resources":
                        system_resources_records.append(values)
                    elif table == "injection_stats":
                        injection_stats_records.append(values)
                    elif table == "training_stats":
                        training_stats_records.append(values)
                    elif table == "latent_snapshots":
                        latent_snapshots_records.append(values)
                    elif table == "inference_results":
                        inference_results_records.append(values)
                    elif table == "inference_cadences":
                        inference_cadences_records.append(values)
                    elif table == "pipeline_stages":
                        pipeline_stages_records.append(values)

                # Bulk insert each table type
                if system_resources_records:
                    cursor.executemany(
                        """
                        INSERT INTO system_resources
                        (timestamp, resource_type, resource_name, value, unit, tag, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        system_resources_records,
                    )

                if injection_stats_records:
                    cursor.executemany(
                        """
                        INSERT INTO injection_stats
                        (timestamp, stat_name, value, round_number, chunk_number, sample_index,
                         background_index, signal_class, signal_type, injection_stage, is_finite,
                         slope_clamped, tag, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        injection_stats_records,
                    )

                if training_stats_records:
                    cursor.executemany(
                        """
                        INSERT INTO training_stats
                        (timestamp, model_name, stat_name, value, round_number, epoch_number,
                         tag, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        training_stats_records,
                    )

                if latent_snapshots_records:
                    cursor.executemany(
                        """
                        INSERT INTO latent_snapshots
                        (timestamp, model_name, round_number, epoch_number, step_number,
                         cadence_index, signal_type, latent_vector, snr_base, snr_range, tag,
                         metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        latent_snapshots_records,
                    )

                if inference_results_records:
                    cursor.executemany(
                        """
                        INSERT INTO inference_results
                        (timestamp, npy_path, snippet_index, prediction, confidence, latent_vector,
                         target, session, cadence_id, band, frequency_mhz, timestamp_observed,
                         h5_path, tag, metadata, screening_proba, mc_mean, mc_std)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        inference_results_records,
                    )

                if inference_cadences_records:
                    cursor.executemany(
                        """
                        INSERT INTO inference_cadences
                        (timestamp, tag, csv_path, cadence_key, npy_path, status, n_stamps,
                         n_candidates, confidence_summary, duration_s, config_fingerprint)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        inference_cadences_records,
                    )

                if pipeline_stages_records:
                    cursor.executemany(
                        """
                        INSERT INTO pipeline_stages
                        (stage, start_time, end_time, duration_s, tag, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        pipeline_stages_records,
                    )

                conn.commit()

        except Exception as e:
            logger.error(f"Error flushing data buffer: {e}")

    # TODO: write checks to sanitize values before writing to db. raise error if problematic value passed
    def write_system_resource(
        self,
        resource_type: str,
        resource_name: str,
        value: float,
        unit: str | None = None,
        tag: str | None = None,
        timestamp: float | None = None,
    ):
        """
        Queue a non-blocking write to system_resources.

        resource_type takes broad categories like 'cpu', 'ram', 'gpu'; resource_name takes
        finer sub-labels like 'system_total' or 'process_tree'. unit is a free-form label
        ('percent', 'MB', etc.). Timestamp defaults to current wall time if omitted.
        """
        metadata_json = get_system_metadata()

        self.write_queue.put(
            (
                "system_resources",
                (
                    timestamp or time.time(),
                    resource_type,
                    resource_name,
                    value,
                    unit,
                    tag,
                    metadata_json,
                ),
            )
        )

    # TODO: write checks to sanitize values before writing to db. raise error if problematic value passed
    def write_injection_stat(
        self,
        stat_name: str,
        value: float,
        round_number: int | None = None,
        chunk_number: int | None = None,
        sample_index: int | None = None,
        background_index: int | None = None,
        signal_class: str | None = None,
        signal_type: str | None = None,
        injection_stage: str | None = None,
        slope_clamped: bool | None = None,
        tag: str | None = None,
        timestamp: float | None = None,
    ):
        """
        Queue a non-blocking write to injection_stats.

        stat_name labels the metric (global_mean, eti_snr, rfi_drift_rate, num_samples, etc.).
        signal_class is one of {main, false, true}; signal_type is one of
        {false_no_signal, false_with_rfi, true_only_eti, true_eti_rfi}; injection_stage is one of
        {A (pre-inj pre-norm), B (post-inj pre-norm), C (post-inj post-norm)}. Non-finite values
        are coerced to 0.0 with is_finite=0 so queries can drop them via the default only_finite
        filter — this is required because the schema's NOT NULL constraint forbids storing NaN/Inf
        directly. Timestamp defaults to current wall time.
        """
        metadata_json = get_system_metadata()

        # Since higher-order moments from _compute_intensity_stats() could lead to NaN/Inf values
        # We explicitly check if a value is finite to set the flag accordingly
        # We will opt to store these values as 0.0 due to the schema's NOT NULL constraint
        # However, queries for injection_stats should by default use the is_finite flag, unless
        # sanitization is explicitly not needed
        is_finite = 1 if np.isfinite(value) else 0
        sanitized_value = float(value) if is_finite else 0.0

        if not is_finite:
            logger.warning(
                f"write_injection_stat: {stat_name} is {value}, storing as 0.0 with is_finite=0"
            )

        # Convert slope_clamped bool to int (0 or 1), defaulting to 0 if None
        slope_clamped_int = 1 if slope_clamped else 0

        self.write_queue.put(
            (
                "injection_stats",
                (
                    timestamp or time.time(),
                    stat_name,
                    sanitized_value,
                    round_number,
                    chunk_number,
                    sample_index,
                    background_index,
                    signal_class,
                    signal_type,
                    injection_stage,
                    is_finite,
                    slope_clamped_int,
                    tag,
                    metadata_json,
                ),
            )
        )

    def write_injection_stats_bulk(self, stats: list[dict], tag: str | None = None) -> None:
        """
        Queue a batch of injection-stat rows on the bounded bulk lane (#277).

        Each dict takes the same keys as write_injection_stat's parameters (stat_name and
        value required; the rest optional). Semantics per row are identical — NaN/Inf
        coercion with is_finite=0, slope_clamped bool -> int — but the batch shares ONE
        system-metadata lookup and is enqueued in bulk_chunk_rows-sized chunks, so a
        ~300K-row segment costs a handful of queue operations instead of ~300K (the per-row
        Python overhead that capped the writer at ~590 rows/s). put() blocks when the bulk
        lane is full — deliberate backpressure on the (background) enqueuer; never call this
        from the training-critical path. With no writer thread running the rows are written
        inline, mirroring mark_superseded's no-writer path.
        """
        metadata_json = get_system_metadata()
        now = time.time()

        rows: list[tuple] = []
        round_counts: dict[int, int] = {}
        non_finite = 0
        for stat in stats:
            value = stat["value"]
            is_finite = 1 if np.isfinite(value) else 0
            if not is_finite:
                non_finite += 1
            round_number = stat.get("round_number")
            rows.append(
                (
                    stat.get("timestamp") or now,
                    stat["stat_name"],
                    float(value) if is_finite else 0.0,
                    round_number,
                    stat.get("chunk_number"),
                    stat.get("sample_index"),
                    stat.get("background_index"),
                    stat.get("signal_class"),
                    stat.get("signal_type"),
                    stat.get("injection_stage"),
                    is_finite,
                    1 if stat.get("slope_clamped") else 0,
                    tag,
                    metadata_json,
                )
            )
            # None rounds key as -1 so injection_backlog_rows() counts them conservatively
            round_key = round_number if round_number is not None else -1
            round_counts[round_key] = round_counts.get(round_key, 0) + 1

        if non_finite:
            logger.warning(
                f"write_injection_stats_bulk: coerced {non_finite} non-finite value(s) to 0.0 "
                "with is_finite=0"
            )

        if not rows:
            return

        # No writer thread -> no consumer for the bounded queue; write inline instead
        if self.writer_thread is None or not self.writer_thread.is_alive():
            self._flush_buffer([("injection_stats", values) for values in rows])
            return

        for start in range(0, len(rows), self.bulk_chunk_rows):
            chunk = rows[start : start + self.bulk_chunk_rows]
            chunk_counts: dict[int, int] = {}
            for values in chunk:
                round_key = values[3] if values[3] is not None else -1
                chunk_counts[round_key] = chunk_counts.get(round_key, 0) + 1
            with self._bulk_pending_lock:
                for round_key, count in chunk_counts.items():
                    self._bulk_pending_by_round[round_key] = (
                        self._bulk_pending_by_round.get(round_key, 0) + count
                    )
            # Blocks when the lane is full (bounded queue) — the backpressure is the point.
            # But never blocks FOREVER: a bounded put with no consumer would deadlock the
            # enqueuer if the writer died after the liveness check above, so re-check
            # liveness on every timeout and fall back to an inline write if it's gone
            while True:
                if self.writer_thread is None or not self.writer_thread.is_alive():
                    logger.warning(
                        "DB writer thread died while the bulk lane was full - writing "
                        f"{len(chunk)} injection row(s) inline"
                    )
                    self._flush_buffer([("injection_stats", values) for values in chunk])
                    with self._bulk_pending_lock:
                        for round_key, count in chunk_counts.items():
                            remaining = self._bulk_pending_by_round.get(round_key, 0) - count
                            if remaining > 0:
                                self._bulk_pending_by_round[round_key] = remaining
                            else:
                                self._bulk_pending_by_round.pop(round_key, None)
                    break
                try:
                    self.bulk_queue.put(("injection_stats", chunk, chunk_counts), timeout=5.0)
                    break
                except Full:
                    continue

    def injection_backlog_rows(self, max_round: int | None = None) -> int:
        """
        Bulk-lane injection rows enqueued but not yet committed for rounds <= max_round
        (None = all rounds). Rows with round_number None are counted against every
        max_round (conservative). Plot code uses this to verify a round's injection rows
        are all committed before rendering its figures (#277).
        """
        with self._bulk_pending_lock:
            if max_round is None:
                return sum(self._bulk_pending_by_round.values())
            return sum(
                count
                for round_key, count in self._bulk_pending_by_round.items()
                if round_key == -1 or round_key <= max_round
            )

    # TODO: write checks to sanitize values before writing to db. raise error if problematic value passed
    def write_training_stat(
        self,
        model_name: str,
        stat_name: str,
        value: float,
        round_number: int | None = None,
        epoch_number: int | None = None,
        tag: str | None = None,
        timestamp: float | None = None,
    ):
        """
        Queue a non-blocking write to training_stats.

        model_name labels the model ('beta_vae', 'rf'); stat_name labels the metric ('total_loss',
        'reconstruction_loss', 'learning_rate', etc.). Timestamp defaults to current wall time.
        """
        metadata_json = get_system_metadata()

        self.write_queue.put(
            (
                "training_stats",
                (
                    timestamp or time.time(),
                    model_name,
                    stat_name,
                    value,
                    round_number,
                    epoch_number,
                    tag,
                    metadata_json,
                ),
            )
        )

    # TODO: write checks to sanitize values before writing to db. raise error if problematic value passed
    def write_latent_snapshot(
        self,
        model_name: str,
        round_number: int,
        epoch_number: int,
        step_number: int,
        cadence_index: int,
        signal_type: str,
        latent_vector: list[list[float]],
        snr_base: int | None = None,
        snr_range: int | None = None,
        tag: str | None = None,
        timestamp: float | None = None,
    ):
        """
        Queue a non-blocking write to latent_snapshots.

        cadence_index is the position within the viz batch (0 to num_cadences-1). signal_type is
        one of {false_no_signal, false_with_rfi, true_only_eti, true_eti_rfi}. latent_vector is a
        nested list of shape (6, latent_dim), serialized to JSON before storage. Timestamp defaults
        to current wall time.
        """
        metadata_json = get_system_metadata()
        latent_vector_json = json.dumps(latent_vector)

        self.write_queue.put(
            (
                "latent_snapshots",
                (
                    timestamp or time.time(),
                    model_name,
                    round_number,
                    epoch_number,
                    step_number,
                    cadence_index,
                    signal_type,
                    latent_vector_json,
                    snr_base,
                    snr_range,
                    tag,
                    metadata_json,
                ),
            )
        )

    def write_latent_snapshots_bulk(
        self,
        model_name: str,
        round_number: int,
        epoch_number: int,
        step_number: int,
        snapshots: list[tuple],
        snr_base: int | None = None,
        snr_range: int | None = None,
        tag: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        """
        Queue one snapshot capture's worth of latent_snapshots rows in a single call.

        `snapshots` is a list of (cadence_index, signal_type, latent_vector) tuples sharing
        the given capture-level fields; latent_vector is array-like of shape
        (num_observations, latent_dim). Row semantics are identical to per-row
        write_latent_snapshot calls, but the system-metadata lookup happens once per batch
        instead of once per cadence — the per-row Python cost ran inside the epoch loop via
        _capture_latent_snapshot (measured ~0.12 s per capture at 3,840 rows).
        """
        metadata_json = get_system_metadata()
        ts = timestamp or time.time()
        for cadence_index, signal_type, latent_vector in snapshots:
            self.write_queue.put(
                (
                    "latent_snapshots",
                    (
                        ts,
                        model_name,
                        round_number,
                        epoch_number,
                        step_number,
                        cadence_index,
                        signal_type,
                        json.dumps(np.asarray(latent_vector).tolist()),
                        snr_base,
                        snr_range,
                        tag,
                        metadata_json,
                    ),
                )
            )

    # TODO: write checks to sanitize values before writing to db. raise error if problematic value passed
    def write_inference_result(
        self,
        npy_path: str,
        snippet_index: int,
        prediction: int,
        confidence: float,
        latent_vector: np.ndarray | None = None,
        target: str | None = None,
        session: str | None = None,
        cadence_id: int | None = None,
        band: str | None = None,
        frequency_mhz: float | None = None,
        timestamp_observed: float | None = None,
        h5_path: str | None = None,
        tag: str | None = None,
        timestamp: float | None = None,
        screening_proba: float | None = None,
        mc_mean: float | None = None,
        mc_std: float | None = None,
    ):
        """
        Queue a non-blocking write to inference_results.

        prediction is 0 (RFI) or 1 (candidate); confidence is in [0.0, 1.0]. latent_vector, when
        provided, is a (6 * latent_dim,) array that gets serialized to JSON for later analysis.
        target, session, cadence_id, band, frequency_mhz, timestamp_observed, and h5_path carry
        the observational provenance (e.g. target='DDO210', session='AGBT18A_999_103', band='L').
        Timestamp defaults to current wall time (distinct from timestamp_observed, which is the
        original observation time). screening_proba / mc_mean / mc_std carry the #282 two-pass
        scores: the deterministic pass-1 probability and the seeded MC mean/spread for
        pass-2 survivors.
        """
        metadata_json = get_system_metadata()

        # Convert latent vector to JSON string if provided
        latent_json = None
        if latent_vector is not None:
            latent_json = json.dumps(latent_vector.tolist())

        self.write_queue.put(
            (
                "inference_results",
                (
                    timestamp or time.time(),
                    npy_path,
                    snippet_index,
                    prediction,
                    confidence,
                    latent_json,
                    target,
                    session,
                    cadence_id,
                    band,
                    frequency_mhz,
                    timestamp_observed,
                    h5_path,
                    tag,
                    metadata_json,
                    screening_proba,
                    mc_mean,
                    mc_std,
                ),
            )
        )

    # TODO: write checks to sanitize values before writing to db. raise error if problematic value passed
    def write_inference_cadence(
        self,
        npy_path: str,
        status: str,
        tag: str | None = None,
        csv_path: str | None = None,
        cadence_key: tuple | list | None = None,
        n_stamps: int | None = None,
        n_candidates: int | None = None,
        confidence_summary: dict | None = None,
        duration_s: float | None = None,
        config_fingerprint: str | None = None,
        timestamp: float | None = None,
    ):
        """
        Queue a non-blocking write to the inference_cadences run manifest.

        One row per (cadence, stage transition): status='preprocessed' when the cadence's
        stamp .npy lands (n_stamps set, aggregates None), status='inferred' when inference
        completes (aggregates set; the caller supersedes older rows for the same
        (tag, npy_path) first), status='failed' when the cadence's inference stage died
        (so the failure is inspectable and the retry pass re-attempts it). cadence_key
        (the CSV group-by key) and confidence_summary (quantile stats from
        inference.summarize_confidences) are JSON-serialized for storage. duration_s is
        the stage's wall-clock duration. Timestamp defaults to current wall time.
        """
        cadence_key_json = None
        if cadence_key is not None:
            cadence_key_json = json.dumps([str(part) for part in cadence_key])
        confidence_summary_json = None
        if confidence_summary is not None:
            confidence_summary_json = json.dumps(confidence_summary)

        self.write_queue.put(
            (
                "inference_cadences",
                (
                    timestamp or time.time(),
                    tag,
                    csv_path,
                    cadence_key_json,
                    npy_path,
                    status,
                    n_stamps,
                    n_candidates,
                    confidence_summary_json,
                    duration_s,
                    config_fingerprint,
                ),
            )
        )

    # TODO: write checks to sanitize values before writing to db. raise error if problematic value passed
    def write_pipeline_stage(
        self,
        stage: str,
        start_time: float,
        end_time: float,
        tag: str | None = None,
        metadata: str | None = None,
    ):
        """
        Queue a non-blocking write to pipeline_stages.

        stage is a hierarchical dot-name ("train.round_02.data_generation"); start_time /
        end_time are unix timestamps bounding the span (duration_s is derived here so the
        table stays internally consistent). metadata, when given, is an already-serialized
        JSON string (aetherscan.benchmark owns the serialization). Prefer the stage_timer /
        record_stage helpers in aetherscan.benchmark over calling this directly.
        """
        self.write_queue.put(
            (
                "pipeline_stages",
                (
                    stage,
                    start_time,
                    end_time,
                    end_time - start_time,
                    tag,
                    metadata,
                ),
            )
        )

    # Column whitelists per table (for SQL injection prevention when using column projection)
    _SYSTEM_RESOURCES_COLUMNS = {
        "id",
        "timestamp",
        "resource_type",
        "resource_name",
        "value",
        "unit",
        "tag",
        "metadata",
    }
    _INJECTION_STATS_COLUMNS = {
        "id",
        "timestamp",
        "stat_name",
        "value",
        "round_number",
        "chunk_number",
        "sample_index",
        "background_index",
        "signal_class",
        "signal_type",
        "injection_stage",
        "is_finite",
        "slope_clamped",
        "tag",
        "metadata",
        "superseded",
    }
    _TRAINING_STATS_COLUMNS = {
        "id",
        "timestamp",
        "model_name",
        "stat_name",
        "value",
        "round_number",
        "epoch_number",
        "tag",
        "metadata",
        "superseded",
    }
    _LATENT_SNAPSHOTS_COLUMNS = {
        "id",
        "timestamp",
        "model_name",
        "round_number",
        "epoch_number",
        "step_number",
        "cadence_index",
        "signal_type",
        "latent_vector",
        "snr_base",
        "snr_range",
        "tag",
        "metadata",
        "superseded",
    }
    _INFERENCE_RESULTS_COLUMNS = {
        "id",
        "timestamp",
        "npy_path",
        "snippet_index",
        "prediction",
        "confidence",
        "latent_vector",
        "target",
        "session",
        "cadence_id",
        "band",
        "frequency_mhz",
        "timestamp_observed",
        "h5_path",
        "tag",
        "metadata",
        "superseded",
        "screening_proba",
        "mc_mean",
        "mc_std",
    }
    _INFERENCE_CADENCES_COLUMNS = {
        "id",
        "timestamp",
        "tag",
        "csv_path",
        "cadence_key",
        "npy_path",
        "status",
        "n_stamps",
        "n_candidates",
        "confidence_summary",
        "duration_s",
        "config_fingerprint",
        "superseded",
    }
    _PIPELINE_STAGES_COLUMNS = {
        "id",
        "stage",
        "start_time",
        "end_time",
        "duration_s",
        "tag",
        "metadata",
    }

    @staticmethod
    def _build_select(table: str, columns: list[str] | None, whitelist: set[str]) -> str:
        """
        Build SELECT clause with optional column projection
        Use SELECT * if no columns provided; Use SELECT col_1, ..., col_n if columns provided
        Validates columns exist in table from whitelist. Raise ValueError if any column is invalid

        Note that this API is not meant for public consumption, and should never be called with
        user input! Since `table` is interpolated directly, this represents a possible SQL injection
        vector
        """
        if columns is None:
            return f"SELECT * FROM {table}"
        # Validate all requested columns against whitelist
        invalid = set(columns) - whitelist
        if invalid:
            raise ValueError(f"Invalid column(s) for {table}: {invalid}")
        cols = ", ".join(columns)
        return f"SELECT {cols} FROM {table}"

    @staticmethod
    def _add_str_filter(
        query: str,
        params: list,
        column: str,
        value: str | list[str],
    ) -> str:
        """
        Handle str or list[str] filter
        Use = for str; Use IN for list

        Note that this is an asymmetric API:
            params (list -> mutable) is modified in place,
            query (str -> immutable) is returned as a new value.
        A cleaner approach would be to either return both (query, params), or modify both in-place
        However, functionally, this method is correct as is, and should be noted by the caller
        """
        if isinstance(value, list):
            if not value:
                return query
            placeholders = ", ".join("?" for _ in value)
            query += f" AND {column} IN ({placeholders})"
            params.extend(value)
        else:
            query += f" AND {column} = ?"
            params.append(value)
        return query

    # NOTE: how to additionally filter by metadata (e.g. machine_name)?
    # NOTE: should we also let user filter by value (e.g. >= or <= some value)?
    def query_system_resource(
        self,
        resource_type: str | list[str] | None = None,
        resource_name: str | list[str] | None = None,
        tag: str | list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query rows from system_resources as a list of dicts.

        resource_type, resource_name, and tag accept either a single value (= filter) or a list
        (IN filter). start_time/end_time bound the timestamp range (unix time, inclusive on both
        ends). columns lets callers project a subset of fields; values are validated against
        _SYSTEM_RESOURCES_COLUMNS to block SQL injection through column names.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            select = self._build_select("system_resources", columns, self._SYSTEM_RESOURCES_COLUMNS)
            # WHERE 1=1 is a way of building parametrized queries
            # Since 1=1 is always true, it does nothing functionally
            # But it allows us to safely add more conditions by appending AND clauses
            # While not breaking the query if none are added
            query = f"{select} WHERE 1=1"
            params = []

            # Build the query dynamically based on user-specified conditions
            if resource_type:
                query = self._add_str_filter(query, params, "resource_type", resource_type)

            if resource_name:
                query = self._add_str_filter(query, params, "resource_name", resource_name)

            if tag:
                query = self._add_str_filter(query, params, "tag", tag)

            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)

            # NOTE: removing ORDER BY to reduce indexing costs
            # Intentionally hard-coded. Update if schema changes
            # query += " ORDER BY tag, timestamp, resource_type, resource_name"

            cursor.execute(query, params)

            # Create a list of column names using query result's metadata
            result_columns = [desc[0] for desc in cursor.description]
            # Pair column names with values and return to user as a dict
            return [dict(zip(result_columns, row, strict=False)) for row in cursor.fetchall()]

    # NOTE: how to additionally filter by metadata (e.g. machine_name)?
    # NOTE: should we also let user filter by value (e.g. >= or <= some value)?
    def query_injection_stat(
        self,
        stat_name: str | list[str] | None = None,
        start_round_number: int | None = None,
        end_round_number: int | None = None,
        start_chunk_number: int | None = None,
        end_chunk_number: int | None = None,
        start_sample_index: int | None = None,
        end_sample_index: int | None = None,
        start_background_index: int | None = None,
        end_background_index: int | None = None,
        signal_class: str | list[str] | None = None,
        signal_type: str | list[str] | None = None,
        injection_stage: str | list[str] | None = None,
        only_finite: bool = True,
        only_slope_clamped: bool | None = None,
        tag: str | list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        columns: list[str] | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Query rows from injection_stats as a list of dicts.

        String filters (stat_name, signal_class, signal_type, injection_stage, tag) accept either
        a single value (= filter) or a list (IN filter). signal_class is one of {main, false, true};
        signal_type is one of {false_no_signal, false_with_rfi, true_only_eti, true_eti_rfi};
        injection_stage is one of {A=pre-inj pre-norm, B=post-inj pre-norm, C=post-inj post-norm}.
        start_*/end_* pairs bound the corresponding integer column. only_finite (default True)
        drops rows where the stored value was non-finite at write time; only_slope_clamped, when
        not None, filters by the slope_clamped flag. include_superseded (default False) controls
        whether rows flagged stale by mark_superseded() are returned. columns is validated
        against _INJECTION_STATS_COLUMNS.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            select = self._build_select("injection_stats", columns, self._INJECTION_STATS_COLUMNS)
            # WHERE 1=1 is a way of building parametrized queries
            # Since 1=1 is always true, it does nothing functionally
            # But it allows us to safely add more conditions by appending AND clauses
            # While not breaking the query if none are added
            query = f"{select} WHERE 1=1"
            params = []

            # Build the query dynamically based on user-specified conditions
            if stat_name:
                query = self._add_str_filter(query, params, "stat_name", stat_name)

            if start_round_number is not None:
                query += " AND round_number >= ?"
                params.append(start_round_number)

            if end_round_number is not None:
                query += " AND round_number <= ?"
                params.append(end_round_number)

            if start_chunk_number is not None:
                query += " AND chunk_number >= ?"
                params.append(start_chunk_number)

            if end_chunk_number is not None:
                query += " AND chunk_number <= ?"
                params.append(end_chunk_number)

            if start_sample_index is not None:
                query += " AND sample_index >= ?"
                params.append(start_sample_index)

            if end_sample_index is not None:
                query += " AND sample_index <= ?"
                params.append(end_sample_index)

            if start_background_index is not None:
                query += " AND background_index >= ?"
                params.append(start_background_index)

            if end_background_index is not None:
                query += " AND background_index <= ?"
                params.append(end_background_index)

            if signal_class:
                query = self._add_str_filter(query, params, "signal_class", signal_class)

            if signal_type:
                query = self._add_str_filter(query, params, "signal_type", signal_type)

            if injection_stage:
                query = self._add_str_filter(query, params, "injection_stage", injection_stage)

            if only_finite:
                query += " AND is_finite = 1"

            if only_slope_clamped is not None:
                query += " AND slope_clamped = ?"
                params.append(1 if only_slope_clamped else 0)

            if not include_superseded:
                query += " AND superseded = 0"

            if tag:
                query = self._add_str_filter(query, params, "tag", tag)

            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)

            # NOTE: removing ORDER BY to reduce indexing costs
            # Intentionally hard-coded. Update if schema changes
            # query += " ORDER BY tag, signal_class, signal_type, round_number, chunk_number, sample_index, stat_name"

            cursor.execute(query, params)

            # Create a list of column names using query result's metadata
            result_columns = [desc[0] for desc in cursor.description]
            # Pair column names with values and return to user as a dict
            return [dict(zip(result_columns, row, strict=False)) for row in cursor.fetchall()]

    def query_injection_stat_stability(
        self,
        stat_name: str | list[str] | None = None,
        start_round_number: int | None = None,
        end_round_number: int | None = None,
        injection_stage: str | list[str] | None = None,
        tag: str | list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Per-round aggregation of injection_stats for sanitization and clamping rates.

        Returns dicts of {round_number, total_count, non_finite_count, clamped_count} so callers
        can compute sanitization rate (non_finite_count / total_count) and clamping rate
        (clamped_count / total_count) without fetching every row. Preferred over
        query_injection_stat + Python-side aggregation: SQLite's native COUNT/SUM in C beats
        materializing rows into Python dicts and iterating. String filters accept str or list[str];
        start_round_number/end_round_number bound the rounds aggregated (#277 — callers scope to
        the rounds being plotted so pre-generated later rounds don't show partial bars).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    round_number,
                    COUNT(*) as total_count,
                    SUM(CASE WHEN is_finite = 0 THEN 1 ELSE 0 END) as non_finite_count,
                    SUM(CASE WHEN slope_clamped = 1 THEN 1 ELSE 0 END) as clamped_count
                FROM injection_stats
                WHERE 1=1
            """
            params: list = []

            if stat_name:
                query = self._add_str_filter(query, params, "stat_name", stat_name)

            if start_round_number is not None:
                query += " AND round_number >= ?"
                params.append(start_round_number)

            if end_round_number is not None:
                query += " AND round_number <= ?"
                params.append(end_round_number)

            if injection_stage:
                query = self._add_str_filter(query, params, "injection_stage", injection_stage)

            if tag:
                query = self._add_str_filter(query, params, "tag", tag)

            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if not include_superseded:
                query += " AND superseded = 0"

            query += " GROUP BY round_number ORDER BY round_number"

            cursor.execute(query, params)

            # Create a list of column names using query result's metadata
            result_columns = [desc[0] for desc in cursor.description]
            # Pair column names with values and return to user as a dict
            return [dict(zip(result_columns, row, strict=False)) for row in cursor.fetchall()]

    # NOTE: how to additionally filter by metadata (e.g. machine_name)?
    # NOTE: should we also let user filter by value (e.g. >= or <= some value)?
    def query_training_stat(
        self,
        model_name: str | list[str] | None = None,
        stat_name: str | list[str] | None = None,
        start_round_number: int | None = None,
        end_round_number: int | None = None,
        start_epoch_number: int | None = None,
        end_epoch_number: int | None = None,
        tag: str | list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        columns: list[str] | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Query rows from training_stats as a list of dicts.

        model_name, stat_name, and tag accept either a single value (= filter) or a list
        (IN filter). start_*/end_* pairs bound the corresponding integer column (inclusive).
        include_superseded (default False) controls whether rows flagged stale by
        mark_superseded() are returned. columns is validated against _TRAINING_STATS_COLUMNS.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            select = self._build_select("training_stats", columns, self._TRAINING_STATS_COLUMNS)
            # WHERE 1=1 is a way of building parametrized queries
            # Since 1=1 is always true, it does nothing functionally
            # But it allows us to safely add more conditions by appending AND clauses
            # While not breaking the query if none are added
            query = f"{select} WHERE 1=1"
            params = []

            # Build the query dynamically based on user-specified conditions
            if model_name:
                query = self._add_str_filter(query, params, "model_name", model_name)

            if stat_name:
                query = self._add_str_filter(query, params, "stat_name", stat_name)

            if start_round_number is not None:
                query += " AND round_number >= ?"
                params.append(start_round_number)

            if end_round_number is not None:
                query += " AND round_number <= ?"
                params.append(end_round_number)

            if start_epoch_number is not None:
                query += " AND epoch_number >= ?"
                params.append(start_epoch_number)

            if end_epoch_number is not None:
                query += " AND epoch_number <= ?"
                params.append(end_epoch_number)

            if not include_superseded:
                query += " AND superseded = 0"

            if tag:
                query = self._add_str_filter(query, params, "tag", tag)

            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)

            # NOTE: removing ORDER BY to reduce indexing costs
            # Intentionally hard-coded. Update if schema changes
            # query += " ORDER BY tag, model_name, round_number, epoch_number, stat_name"

            cursor.execute(query, params)

            # Create a list of column names using query result's metadata
            result_columns = [desc[0] for desc in cursor.description]
            # Pair column names with values and return to user as a dict
            return [dict(zip(result_columns, row, strict=False)) for row in cursor.fetchall()]

    # NOTE: how to additionally filter by metadata (e.g. machine_name)?
    def query_latent_snapshots(
        self,
        model_name: str | list[str] | None = None,
        round_number: int | None = None,
        epoch_number: int | None = None,
        step_number: int | None = None,
        signal_type: str | list[str] | None = None,
        tag: str | list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        columns: list[str] | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Query rows from latent_snapshots as a list of dicts.

        model_name, signal_type, and tag accept either a single value (= filter) or a list
        (IN filter). round_number/epoch_number/step_number are exact-match filters (no range
        variant). The returned latent_vector field is a JSON string — callers parse it with
        json.loads. include_superseded (default False) controls whether rows flagged stale by
        mark_superseded() are returned. columns is validated against _LATENT_SNAPSHOTS_COLUMNS.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            select = self._build_select("latent_snapshots", columns, self._LATENT_SNAPSHOTS_COLUMNS)
            # WHERE 1=1 is a way of building parametrized queries
            # Since 1=1 is always true, it does nothing functionally
            # But it allows us to safely add more conditions by appending AND clauses
            # While not breaking the query if none are added
            query = f"{select} WHERE 1=1"
            params = []

            # Build the query dynamically based on user-specified conditions
            if model_name:
                query = self._add_str_filter(query, params, "model_name", model_name)

            if round_number is not None:
                query += " AND round_number = ?"
                params.append(round_number)

            if epoch_number is not None:
                query += " AND epoch_number = ?"
                params.append(epoch_number)

            if step_number is not None:
                query += " AND step_number = ?"
                params.append(step_number)

            if signal_type:
                query = self._add_str_filter(query, params, "signal_type", signal_type)

            if tag:
                query = self._add_str_filter(query, params, "tag", tag)

            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if not include_superseded:
                query += " AND superseded = 0"

            # NOTE: hard-coded ORDER BY? add template index to init_db

            cursor.execute(query, params)

            # Create a list of column names using query result's metadata
            result_columns = [desc[0] for desc in cursor.description]
            # Pair column names with values and return to user as a dict
            return [dict(zip(result_columns, row, strict=False)) for row in cursor.fetchall()]

    def query_latent_snapshot_keys(
        self,
        tag: str | list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Distinct snapshot keys (model_name, round_number, epoch_number, step_number, snr_base,
        snr_range) sorted by training progression. Avoids the cost of loading every row through
        query_latent_snapshots() when the caller only needs the set of available keys.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT DISTINCT model_name, round_number, epoch_number, step_number, snr_base, snr_range
                FROM latent_snapshots
                WHERE 1=1
            """
            params: list = []

            # Build the query dynamically based on user-specified conditions
            if tag:
                query = self._add_str_filter(query, params, "tag", tag)

            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if not include_superseded:
                query += " AND superseded = 0"

            # This ORDER BY doesn't make full use of idx_latent_snapshots_filter, since the ORDER BY
            # columns don't follow contiguously in the index
            # However, as long as both tag & timestamp are present in the query's WHERE clause,
            # idx_latent_snapshots_filter can still be used to optimize the query
            # The remaining rows form a small set of DISTINCT keys that aren't expensive for SQLite
            # to perform a filesort
            # If you measure a bottleneck in the future, consider adding a second schema index like
            # (tag, model_name, round_number, epoch_number, step_number, timestamp) and then sorting
            # the rows in that order
            query += " ORDER BY model_name, round_number, epoch_number, step_number"

            cursor.execute(query, params)

            # Create a list of column names using query result's metadata
            result_columns = [desc[0] for desc in cursor.description]
            # Pair column names with values and return to user as a dict
            return [dict(zip(result_columns, row, strict=False)) for row in cursor.fetchall()]

    # NOTE: how to additionally filter by metadata (e.g. machine_name)?
    def query_inference_result(
        self,
        npy_path: str | list[str] | None = None,
        start_snippet_index: int | None = None,
        end_snippet_index: int | None = None,
        prediction: int | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        target: str | list[str] | None = None,
        session: str | list[str] | None = None,
        cadence_id: int | None = None,
        band: str | list[str] | None = None,
        min_frequency_mhz: float | None = None,
        max_frequency_mhz: float | None = None,
        start_timestamp_observed: float | None = None,
        end_timestamp_observed: float | None = None,
        h5_path: str | list[str] | None = None,
        tag: str | list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        columns: list[str] | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Query rows from inference_results as a list of dicts.

        String filters (npy_path, target, session, band, h5_path, tag) accept either a single
        value (= filter) or a list (IN filter). prediction is exact-match (0=RFI, 1=candidate).
        start_*/end_*/min_*/max_* pairs bound the corresponding numeric column (inclusive).
        timestamp_observed is the original observation time (distinct from the write-time
        timestamp). include_superseded (default False) controls whether rows flagged stale by
        mark_superseded() are returned. columns is validated against _INFERENCE_RESULTS_COLUMNS.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            select = self._build_select(
                "inference_results", columns, self._INFERENCE_RESULTS_COLUMNS
            )
            # WHERE 1=1 is a way of building parametrized queries
            # Since 1=1 is always true, it does nothing functionally
            # But it allows us to safely add more conditions by appending AND clauses
            # While not breaking the query if none are added
            query = f"{select} WHERE 1=1"
            params = []

            # Build the query dynamically based on user-specified conditions
            if npy_path:
                query = self._add_str_filter(query, params, "npy_path", npy_path)

            if start_snippet_index is not None:
                query += " AND snippet_index >= ?"
                params.append(start_snippet_index)

            if end_snippet_index is not None:
                query += " AND snippet_index <= ?"
                params.append(end_snippet_index)

            if prediction is not None:
                query += " AND prediction = ?"
                params.append(prediction)

            if min_confidence is not None:
                query += " AND confidence >= ?"
                params.append(min_confidence)

            if max_confidence is not None:
                query += " AND confidence <= ?"
                params.append(max_confidence)

            if target:
                query = self._add_str_filter(query, params, "target", target)

            if session:
                query = self._add_str_filter(query, params, "session", session)

            if cadence_id is not None:
                query += " AND cadence_id = ?"
                params.append(cadence_id)

            if band:
                query = self._add_str_filter(query, params, "band", band)

            if min_frequency_mhz is not None:
                query += " AND frequency_mhz >= ?"
                params.append(min_frequency_mhz)

            if max_frequency_mhz is not None:
                query += " AND frequency_mhz <= ?"
                params.append(max_frequency_mhz)

            if start_timestamp_observed is not None:
                query += " AND timestamp_observed >= ?"
                params.append(start_timestamp_observed)

            if end_timestamp_observed is not None:
                query += " AND timestamp_observed <= ?"
                params.append(end_timestamp_observed)

            if h5_path:
                query = self._add_str_filter(query, params, "h5_path", h5_path)

            if tag:
                query = self._add_str_filter(query, params, "tag", tag)

            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if not include_superseded:
                query += " AND superseded = 0"

            # NOTE: hard-coded ORDER BY? add template index to init_db

            cursor.execute(query, params)

            # Create a list of column names using query result's metadata
            result_columns = [desc[0] for desc in cursor.description]
            # Pair column names with values and return to user as a dict
            return [dict(zip(result_columns, row, strict=False)) for row in cursor.fetchall()]

    def query_inference_cadences(
        self,
        npy_path: str | list[str] | None = None,
        status: str | list[str] | None = None,
        csv_path: str | list[str] | None = None,
        tag: str | list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        columns: list[str] | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Query rows from the inference_cadences run manifest as a list of dicts.

        String filters (npy_path, status, csv_path, tag) accept either a single value
        (= filter) or a list (IN filter); status is one of {'preprocessed', 'inferred',
        'failed'}. The returned cadence_key and confidence_summary fields are JSON strings —
        callers parse them with json.loads. include_superseded (default False) controls
        whether rows flagged stale by mark_superseded() are returned — the resume flow relies
        on the default so a cadence whose 'inferred' row was superseded is re-attempted.
        columns is validated against _INFERENCE_CADENCES_COLUMNS.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            select = self._build_select(
                "inference_cadences", columns, self._INFERENCE_CADENCES_COLUMNS
            )
            # WHERE 1=1 is a way of building parametrized queries
            # Since 1=1 is always true, it does nothing functionally
            # But it allows us to safely add more conditions by appending AND clauses
            # While not breaking the query if none are added
            query = f"{select} WHERE 1=1"
            params: list = []

            # Build the query dynamically based on user-specified conditions
            if npy_path:
                query = self._add_str_filter(query, params, "npy_path", npy_path)

            if status:
                query = self._add_str_filter(query, params, "status", status)

            if csv_path:
                query = self._add_str_filter(query, params, "csv_path", csv_path)

            if tag:
                query = self._add_str_filter(query, params, "tag", tag)

            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if not include_superseded:
                query += " AND superseded = 0"

            cursor.execute(query, params)

            # Create a list of column names using query result's metadata
            result_columns = [desc[0] for desc in cursor.description]
            # Pair column names with values and return to user as a dict
            return [dict(zip(result_columns, row, strict=False)) for row in cursor.fetchall()]

    def query_pipeline_stages(
        self,
        stage: str | list[str] | None = None,
        tag: str | list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query rows from pipeline_stages as a list of dicts, ordered by start_time.

        stage and tag accept either a single value (= filter) or a list (IN filter).
        start_time/end_time bound the row's start_time column (unix time, inclusive on both
        ends) — a span that started inside the window but ended after end_time is still
        returned. The returned metadata field is a JSON string or None — callers parse it
        with json.loads. columns is validated against _PIPELINE_STAGES_COLUMNS.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            select = self._build_select("pipeline_stages", columns, self._PIPELINE_STAGES_COLUMNS)
            # WHERE 1=1 is a way of building parametrized queries
            # Since 1=1 is always true, it does nothing functionally
            # But it allows us to safely add more conditions by appending AND clauses
            # While not breaking the query if none are added
            query = f"{select} WHERE 1=1"
            params: list = []

            # Build the query dynamically based on user-specified conditions
            if stage:
                query = self._add_str_filter(query, params, "stage", stage)

            if tag:
                query = self._add_str_filter(query, params, "tag", tag)

            if start_time is not None:
                query += " AND start_time >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND start_time <= ?"
                params.append(end_time)

            # Chronological order is what every consumer (report tree, timeline plot,
            # monitor overlay) wants; the table stays small (one row per stage span), so
            # the filesort on top of idx_pipeline_stages_filter is cheap
            query += " ORDER BY start_time"

            cursor.execute(query, params)

            # Create a list of column names using query result's metadata
            result_columns = [desc[0] for desc in cursor.description]
            # Pair column names with values and return to user as a dict
            return [dict(zip(result_columns, row, strict=False)) for row in cursor.fetchall()]

    # NOTE: this call gets expensive as db grows. create a separate schema to track num_rows_added per pipeline run. then count & update as part of db cleanup routine? or use SQLite's dbstat virtual table or periodic ANALYZE?
    def get_db_stats(self) -> dict[str, Any]:
        """Get summary statistics for the database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Row counts
            cursor.execute("SELECT COUNT(*) FROM system_resources")
            stats["system_resources_row_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM injection_stats")
            stats["injection_stats_row_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM training_stats")
            stats["training_stats_row_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM latent_snapshots")
            stats["latent_snapshots_row_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM inference_results")
            stats["inference_results_row_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM inference_cadences")
            stats["inference_cadences_row_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM pipeline_stages")
            stats["pipeline_stages_row_count"] = cursor.fetchone()[0]

            # Time range
            # Use system_resources as proxy
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp)
                FROM system_resources
            """)
            min_time, max_time = cursor.fetchone()
            stats["min_timestamp"] = min_time
            stats["max_timestamp"] = max_time

            # Database size
            cursor.execute(
                "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
            )
            stats["db_size_bytes"] = cursor.fetchone()[0]
            stats["db_size_mb"] = stats["db_size_bytes"] / (1024 * 1024)

            return stats


def init_db() -> Database:
    """
    Initialize global database instance (call once at startup)
    """
    db = Database()
    db.start()

    register_db(db)

    return db


def get_db() -> Database | None:
    """Get the global database instance"""
    db = Database._instance

    if db is None:
        logger.warning("No database instance initialized")

    return db


# TODO: complete this function
# NOTE: where should source of truth be? (bla0, blpc2, blpc3, other)
def merge_db() -> None:
    """Merge (join & dedup) two different aetherscan.db files into a single source-of-truth"""
    pass


def shutdown_db() -> None:
    """Shutdown the global database instance (call on exit)"""
    db = Database._instance

    if db is None:
        logger.warning("No database instance initialized")
        return

    db.stop()
    Database._reset()
