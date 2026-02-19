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
from queue import Empty, Queue
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


def get_system_metadata() -> str:
    """
    Collects system metadata (machine name, user name, IP address, process ID)
    and returns it as a JSON string suitable for database storage.
    """
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
    return json.dumps(metadata, sort_keys=True)


class Database:
    """
    Thread-safe SQLite database for storing data with asynchronous queue-based writes.

    Architecture:
    - Multiple threads/processes send data to a shared queue
    - Single writer thread consumes from queue and writes to SQLite periodically
    - Eliminates concurrent write issues and SQLITE_BUSY errors
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

        self.write_queue = Queue()
        self.writer_thread = None
        self.stop_event = threading.Event()  # Thread-safe flag for stopping

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
        Teardown hook for thread-safe singleton
        Resets the db instance to None

        WARNING: Only use for testing or cleanup after shutdown.
        Calling this while the database is active will cause issues.
        Should only be called after stop() has completed.
        """
        # Acquire lock to prevent race conditions
        with cls._lock:
            # Discard the singleton instance by removing the global reference
            # Guarantees the next constructor call will produce a fresh instance
            # Note, resources held by the old instance will remain alive unless explicitly closed beforehand
            cls._instance = None
            logger.info("Database singleton instance reset")

    # TODO:
    # We currently don't support schema versioning or migration scripting for schema changes
    # This means that while new tables can be easily added using CREATE TABLE IF NOT EXISTS
    # statements, existing tables cannot be easily modified (i.e. no ALTER TABLE support for adding
    # new columns to tables, modifying column constraints, renaming columns, or dropping obsolete
    # columns), and no rollback mechanisms exist
    # This makes it impossible to evolve existing schema without manual database updates. As well as
    # having no audit trail of schema changes, or no rollback capability for failed upgrades. It's
    # also difficult to coordinate deployments across multiple db instances
    # NOTE: should we consider migrating to PostgreSQL? or should we go all in on SQLite?
    def _init_database(self):
        """Create database tables if they don't exist"""
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
                    metadata TEXT
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
                    metadata TEXT
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
                    metadata TEXT
                )
            """)

            # Composite index for common filter pattern (tag + confidence + prediction)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_inference_results_filter
                ON inference_results(tag, timestamp, confidence, prediction)
            """)

            # Latent snapshots table (for latent space visualization)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS latent_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    round_number INTEGER NOT NULL,
                    epoch_number INTEGER NOT NULL,
                    step_number INTEGER NOT NULL,
                    cadence_index INTEGER NOT NULL,
                    signal_type TEXT NOT NULL,
                    latent_vector TEXT NOT NULL,
                    snr_base INTEGER,
                    snr_range INTEGER,
                    tag TEXT,
                    metadata TEXT
                )
            """)

            # Composite index for common filter pattern (tag + round + epoch + step)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_latent_snapshots_filter
                ON latent_snapshots(tag, round_number, epoch_number, step_number)
            """)

            conn.commit()

            # Enable Write-Ahead Logging (WAL) mode for better concurrent read performance
            # WAL places writes in a separate log file so reads can still go through while writes happen
            # The WAL log is periodically merged back into the main db (i.e. checkpointing)
            cursor.execute("PRAGMA journal_mode=WAL")

            logger.info("Database schema initialized with WAL mode")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with proper cleanup"""
        conn = sqlite3.connect(self.db_path, timeout=self.get_connection_timeout)
        try:
            yield conn
        finally:
            conn.close()

    def start(self):
        """Start the background writer thread"""
        if self.writer_thread is not None and self.writer_thread.is_alive():
            return

        self.stop_event.clear()
        # NOTE: should db be daemon or non-daemon thread?
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=False)
        self.writer_thread.start()
        logger.info("Database writer thread started")

    def stop(self):
        """Stop the background writer thread and flush remaining data"""
        if self.writer_thread is None:
            return

        logger.info("Stopping database writer thread...")
        self.stop_event.set()  # Signal thread to stop

        # Wait for writer thread to finish
        self.writer_thread.join(timeout=self.stop_writer_timeout)

        if self.writer_thread.is_alive():
            logger.warning("Database writer thread did not stop cleanly")
        else:
            logger.info("Database writer thread stopped")

    def flush(self, timeout: float | None = None) -> bool:
        """
        Block until all queued writes are flushed to database.

        This method ensures data consistency by waiting for all pending writes
        to complete before returning. Use this before any operation that reads
        from the database and expects to see recently written data.

        Args:
            timeout: Maximum time to wait for flush completion (seconds).
                     If None, uses the default configured flush_timeout.

        Returns:
            True if flush completed successfully, False if timed out or
            shutdown was initiated during the wait.
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

    def _writer_loop(self):
        """Background loop that consumes data from queue and writes to database"""
        self.buffer = []
        last_write_time = time.time()

        # Keep looping until told to stop
        while not self.stop_event.is_set():
            try:
                # Calculate how much time remains until the next scheduled write
                # Don't wait longer than 1s so we check the stop flag regularly
                # Don't wait more than 0.1s to avoid wasting CPU resources
                # Retrive items from the queue one-by-one & append to local buffer
                timeout = max(0.1, min(1.0, self.write_interval - (time.time() - last_write_time)))
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

        # Final flush on shutdown
        if self.buffer:
            self._flush_buffer()
            self.buffer.clear()
            logger.info(f"Flushed {len(self.buffer)} remaining data on shutdown")

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
    def _flush_buffer(self):
        """
        Write buffered data to database in a single transaction using executemany().

        Groups records by table and uses executemany() for bulk inserts, which is more efficient
        than individual execute() calls because the SQL is parsed once and reused for all rows.
        """
        if not self.buffer:
            return

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Group records by table for bulk inserts
                system_resources_records: list[tuple] = []
                injection_stats_records: list[tuple] = []
                training_stats_records: list[tuple] = []
                inference_results_records: list[tuple] = []
                latent_snapshots_records: list[tuple] = []

                for table, values in self.buffer:
                    if table == "system_resources":
                        system_resources_records.append(values)
                    elif table == "injection_stats":
                        injection_stats_records.append(values)
                    elif table == "training_stats":
                        training_stats_records.append(values)
                    elif table == "inference_results":
                        inference_results_records.append(values)
                    elif table == "latent_snapshots":
                        latent_snapshots_records.append(values)

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

                if inference_results_records:
                    cursor.executemany(
                        """
                        INSERT INTO inference_results
                        (timestamp, npy_path, snippet_index, prediction, confidence, latent_vector,
                         target, session, cadence_id, band, frequency_mhz, timestamp_observed,
                         h5_path, tag, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        inference_results_records,
                    )

                if latent_snapshots_records:
                    cursor.executemany(
                        """
                        INSERT INTO latent_snapshots
                        (timestamp, round_number, epoch_number, step_number, cadence_index,
                         signal_type, latent_vector, snr_base, snr_range, tag, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        latent_snapshots_records,
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
        Queue write to system_resources table (non-blocking)

        Args:
            resource_type: Type of resource (e.g. 'cpu', 'ram', 'gpu')
            resource_name: Name of resource (e.g. 'system_total', 'process_tree')
            value: Resource value
            unit: Optional unit of measurement (e.g. 'percent', 'MB')
            tag: Optional tag for current pipeline run
            timestamp: Optional timestamp when resource was logged (uses current time if not provided)
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
        Queue write to injection_stats table (non-blocking)

        Args:
            stat_name: Stat name (e.g. global_mean, eti_snr, rfi_drift_rate, num_samples, etc.)
            value: Stat value
            round_number: Optional current training round number
            chunk_number: Optional current injection chunk number
            sample_index: Optional sample index within batch (0 to N-1)
            background_index: Optional index of background plate used for this sample
            signal_class: Optional signal class (e.g. main, false, true)
            signal_type: Optional signal type (e.g. false_no_signal, false_with_rfi, true_only_eti, true_eti_rfi)
            injection_stage: Optional injection stage (e.g. A: pre-inj pre-norm, B: post-inj pre-norm, C: post-inj post-norm)
            slope_clamped: Optional per-sample flag indicating if slope was clamped during injection
            tag: Optional tag for current pipeline run
            timestamp: Optional timestamp when stat was logged (uses current time if not provided)
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
        Queue write to training_stats table (non-blocking)

        Args:
            model_name: Model name (e.g. 'beta_vae', 'rf')
            stat_name: Stat name (e.g. 'total_loss', 'reconstruction_loss', 'learning_rate')
            value: Stat value
            round_number: Optional current training round number
            epoch_number: Optional current training epoch number
            tag: Optional tag for current pipeline run
            timestamp: Optional timestamp when stat was logged (uses current time if not provided)
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
    ):
        """
        Queue write to inference_results table (non-blocking)

        Args:
            npy_path: Path to the .npy file containing the snippet (e.g. "real_filtered_LARGE_test_HIP15638.npy")
            snippet_index: Index of the snippet within the .npy file
            prediction: Classification prediction (0=RFI, 1=candidate)
            confidence: Classification confidence score (0.0 to 1.0)
            latent_vector: Optional latent vector (6 x 8 features) for later analysis
            target: Optional observation target name (e.g. "DDO210")
            session: Optional observing session identifier (e.g. "AGBT18A_999_103")
            cadence_id: Optional cadence ID from the observation (e.g. "24777")
            band: Optional frequency band (e.g. "L", "S", "C", "X")
            frequency_mhz: Optional center frequency in MHz
            timestamp_observed: Optional timestamp when observation was made (unix time)
            h5_path: Optional path to the source HDF5 file
            tag: Optional tag for current pipeline run
            timestamp: Optional timestamp when result was logged (uses current time if not provided)
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
                ),
            )
        )

    def write_latent_snapshot(
        self,
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
        Queue write to latent_snapshots table (non-blocking)

        Args:
            round_number: Current training round number
            epoch_number: Current training epoch number
            step_number: Current training step number
            cadence_index: Position within the viz batch (0 to num_cadences-1)
            signal_type: Signal type (e.g. false_no_signal, false_with_rfi, true_only_eti, true_eti_rfi)
            latent_vector: Latent vectors for this cadence, shape (6, latent_dim) as nested list
            snr_base: Optional SNR base value
            snr_range: Optional SNR range value
            tag: Optional tag for current pipeline run
            timestamp: Optional timestamp (uses current time if not provided)
        """
        metadata_json = get_system_metadata()
        latent_vector_json = json.dumps(latent_vector)

        self.write_queue.put(
            (
                "latent_snapshots",
                (
                    timestamp or time.time(),
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
    }
    _LATENT_SNAPSHOTS_COLUMNS = {
        "id",
        "timestamp",
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
        Query from system_resources table

        Args:
            resource_type: Type of resource (e.g. 'cpu', 'ram', 'gpu'). Accepts str or list[str].
            resource_name: Name of resource (e.g. 'system_total', 'process_tree'). Accepts str or list[str].
            tag: Tag for current pipeline run. Accepts str or list[str].
            start_time: Start timestamp (unix time)
            end_time: End timestamp (unix time)
            columns: Optional list of columns to select (default: all). Validated against schema.

        Returns:
            List of metric dictionaries
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
    ) -> list[dict[str, Any]]:
        """
        Query from injection_stats table

        Args:
            stat_name: Stat name (e.g. global_mean, eti_snr, rfi_drift_rate, num_samples, etc.). Accepts str or list[str].
            start_round_number: Start round number
            end_round_number: End round number
            start_chunk_number: Start chunk number
            end_chunk_number: End chunk number
            start_sample_index: Start sample index
            end_sample_index: End sample index
            start_background_index: Start background index
            end_background_index: End background index
            signal_class: Signal class (e.g. main, false, true). Accepts str or list[str].
            signal_type: Signal type (e.g. false_no_signal, false_with_rfi, true_only_eti, true_eti_rfi). Accepts str or list[str].
            injection_stage: Optional injection stage (e.g. A: pre-inj pre-norm, B: post-inj pre-norm, C: post-inj post-norm). Accepts str or list[str].
            only_finite: If True (default), only return rows where is_finite=1
            only_slope_clamped: If True, only return rows where slope_clamped=1; if False, only where slope_clamped=0; if None (default), no filter
            tag: Tag for current pipeline run. Accepts str or list[str].
            start_time: Start timestamp (unix time)
            end_time: End timestamp (unix time)
            columns: Optional list of columns to select (default: all). Validated against schema.

        Returns:
            List of metric dictionaries
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
        injection_stage: str | list[str] | None = None,
        tag: str | list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        SQL-level aggregation of injection_stats sanitization and clamping rates.

        Preferred over calling query_injection_stat() & performing Python-level aggregation,
        since SQLite's internal operations (native C) are inherently faster thanPython's object
        creation & iteration

        Returns per-round counts for computing sanitization rate (non_finite_count / total_count)
        and clamping rate (clamped_count / total_count) without fetching all rows.

        Args:
            stat_name: Stat name filter. Accepts str or list[str].
            injection_stage: Injection stage filter. Accepts str or list[str].
            tag: Tag filter. Accepts str or list[str].
            start_time: Start timestamp (unix time)
            end_time: End timestamp (unix time)

        Returns:
            List of dicts with {round_number, total_count, non_finite_count, clamped_count}
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
    ) -> list[dict[str, Any]]:
        """
        Query from training_stats table

        Args:
            model_name: Model name (e.g. 'beta_vae', 'rf'). Accepts str or list[str].
            stat_name: Stat name (e.g. 'total_loss', 'reconstruction_loss', 'learning_rate'). Accepts str or list[str].
            start_round_number: Start round number
            end_round_number: End round number
            start_epoch_number: Start epoch number
            end_epoch_number: End epoch number
            tag: Tag for current pipeline run. Accepts str or list[str].
            start_time: Start timestamp (unix time)
            end_time: End timestamp (unix time)
            columns: Optional list of columns to select (default: all). Validated against schema.

        Returns:
            List of metric dictionaries
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
    # NOTE: should we also let user filter by value (e.g. >= or <= some value)?
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
    ) -> list[dict[str, Any]]:
        """
        Query from inference_results table

        Args:
            npy_path: Path to the .npy file containing the snippet. Accepts str or list[str].
            start_snippet_index: Start snippet index
            end_snippet_index: End snippet index
            prediction: Classification prediction (0=RFI, 1=candidate)
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            target: Observation target name (e.g. "HIP110750"). Accepts str or list[str].
            session: Observing session identifier (e.g. "AGBT18A_999_103"). Accepts str or list[str].
            cadence_id: Cadence ID from the observation (e.g. "24777")
            band: Frequency band (e.g. "L", "S", "C", "X"). Accepts str or list[str].
            min_frequency_mhz: Minimum center frequency in MHz
            max_frequency_mhz: Maximum center frequency in MHz
            start_timestamp_observed: Start observation timestamp (unix time)
            end_timestamp_observed: End observation timestamp (unix time)
            h5_path: Path to the source HDF5 file. Accepts str or list[str].
            tag: Tag for current pipeline run. Accepts str or list[str].
            start_time: Start timestamp (unix time)
            end_time: End timestamp (unix time)
            columns: Optional list of columns to select (default: all). Validated against schema.

        Returns:
            List of result dictionaries
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

            # NOTE: hard-coded ORDER BY? add template index to init_db

            cursor.execute(query, params)

            # Create a list of column names using query result's metadata
            result_columns = [desc[0] for desc in cursor.description]
            # Pair column names with values and return to user as a dict
            return [dict(zip(result_columns, row, strict=False)) for row in cursor.fetchall()]

    def query_latent_snapshots(
        self,
        tag: str | list[str] | None = None,
        round_number: int | None = None,
        epoch_number: int | None = None,
        step_number: int | None = None,
        signal_type: str | list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query from latent_snapshots table

        Args:
            tag: Tag for current pipeline run. Accepts str or list[str].
            round_number: Filter by exact round number
            epoch_number: Filter by exact epoch number
            step_number: Filter by exact step number
            signal_type: Signal type filter. Accepts str or list[str].
            start_time: Start timestamp (unix time)
            end_time: End timestamp (unix time)
            columns: Optional list of columns to select (default: all). Validated against schema.

        Returns:
            List of snapshot dictionaries (latent_vector is JSON string — caller parses with json.loads)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            select = self._build_select("latent_snapshots", columns, self._LATENT_SNAPSHOTS_COLUMNS)
            query = f"{select} WHERE 1=1"
            params = []

            if tag:
                query = self._add_str_filter(query, params, "tag", tag)

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

            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)

            cursor.execute(query, params)

            result_columns = [desc[0] for desc in cursor.description]
            return [dict(zip(result_columns, row, strict=False)) for row in cursor.fetchall()]

    def query_latent_snapshot_keys(
        self,
        tag: str | list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get distinct snapshot keys (round, epoch, step, snr_base, snr_range) sorted by progression.

        Returns:
            List of dicts with {round_number, epoch_number, step_number, snr_base, snr_range}
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT DISTINCT round_number, epoch_number, step_number, snr_base, snr_range
                FROM latent_snapshots
                WHERE 1=1
            """
            params: list = []

            if tag:
                query = self._add_str_filter(query, params, "tag", tag)

            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY round_number, epoch_number, step_number"

            cursor.execute(query, params)

            result_columns = [desc[0] for desc in cursor.description]
            return [dict(zip(result_columns, row, strict=False)) for row in cursor.fetchall()]

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

            cursor.execute("SELECT COUNT(*) FROM inference_results")
            stats["inference_results_row_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM latent_snapshots")
            stats["latent_snapshots_row_count"] = cursor.fetchone()[0]

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
