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


# TEST: is this still needed? is there a more sensible fallback value than 0.0? should we remove the NaN values manually from db writes and plotting? does this conflict with _compute_intensity_stats() from data_generation.py? should we add metrics to track sanitization frequency (indicating possible data corruption)
def _sanitize_float(value: float, fallback: float = 0.0, name: str = "value") -> float:
    """Replace NaN/inf with fallback for SQLite compatibility."""
    if not np.isfinite(value):
        logger.warning(f"_sanitize_float: {name} is {value}, replacing with {fallback}")
        return fallback
    return value


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

    # We currently don't support schema versioning or migration scripting for schema changes
    # For the current pipeline design, CREATE TABLE IF NOT EXISTS is sufficient
    # Migratin frameworks add complexity that simply aren't necessary for our current use cases
    # For now, we'll manually change schemas as-needed
    # Revisit this if schema changes become frequent
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

            # TODO: add additional index for frequent queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON system_resources(timestamp)
            """)
            # -- For query_system_resource ORDER BY pattern
            # CREATE INDEX IF NOT EXISTS idx_system_resources_query
            # ON system_resources(tag, timestamp, resource_type, resource_name);

            # Injection statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS injection_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    stat_name TEXT NOT NULL,
                    value REAL,
                    round_number INTEGER,
                    chunk_number INTEGER,
                    sample_index INTEGER,
                    background_index INTEGER,
                    signal_class TEXT,
                    signal_type TEXT,
                    injection_stage TEXT,
                    tag TEXT,
                    metadata TEXT,
                    is_valid INTEGER DEFAULT 1
                )
            """)

            # TODO: add additional index for frequent queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON injection_stats(timestamp)
            """)
            # -- For query_injection_stat ORDER BY pattern
            # CREATE INDEX IF NOT EXISTS idx_injection_stats_query
            # ON injection_stats(tag, signal_class, signal_type, round_number, chunk_number, sample_index, stat_name);

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

            # TODO: add additional index for frequent queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON training_stats(timestamp)
            """)
            # -- For query_training_stat ORDER BY pattern
            # CREATE INDEX IF NOT EXISTS idx_training_stats_query
            # ON training_stats(tag, model_name, round_number, epoch_number, stat_name);

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

    def _flush_buffer(self):
        """
        Write buffered data to database in a single transaction using executemany().

        Groups records by table type and uses executemany() for bulk inserts, which is
        more efficient than individual execute() calls because the SQL is parsed once
        and reused for all rows.
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

                for table, values in self.buffer:
                    if table == "system_resources":
                        system_resources_records.append(values)
                    elif table == "injection_stats":
                        injection_stats_records.append(values)
                    elif table == "training_stats":
                        training_stats_records.append(values)

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
                         background_index, signal_class, signal_type, injection_stage, tag, metadata, is_valid)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

                conn.commit()

        except Exception as e:
            logger.error(f"Error flushing data buffer: {e}")

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
            tag: Optional tag for current pipeline run
            timestamp: Optional timestamp when stat was logged (uses current time if not provided)
        """
        metadata_json = get_system_metadata()

        # Check if value is valid (finite) and set is_valid flag accordingly
        # Store NULL for invalid values instead of clamping to 0.0
        is_valid = 1 if np.isfinite(value) else 0
        sanitized_value = float(value) if is_valid else None

        if not is_valid:
            logger.warning(
                f"write_injection_stat: {stat_name} is {value}, storing as NULL with is_valid=0"
            )

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
                    tag,
                    metadata_json,
                    is_valid,
                ),
            )
        )

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

    # NOTE: how to additionally filter by metadata (e.g. machine_name)?
    # NOTE: should we also let user filter by value (e.g. >= or <= some value)?
    # NOTE: how to allow str args to be lists & filter accordingly?
    def query_system_resource(
        self,
        resource_type: str | None = None,
        resource_name: str | None = None,
        tag: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query from system_resources table

        Args:
            resource_type: Type of resource (e.g. 'cpu', 'ram', 'gpu')
            resource_name: Name of resource (e.g. 'system_total', 'process_tree')
            tag: Tag for current pipeline run
            start_time: Start timestamp (unix time)
            end_time: End timestamp (unix time)

        Returns:
            List of metric dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # WHERE 1=1 is a trick for building dynamic queries
            # Since it's always true, it does nothing
            # But it lets us safely add more conditions with AND
            query = "SELECT * FROM system_resources WHERE 1=1"
            params = []

            # Build the query dynamically based on user-specified conditions
            if resource_type:
                query += " AND resource_type = ?"
                params.append(resource_type)

            if resource_name:
                query += " AND resource_name = ?"
                params.append(resource_name)

            if tag:
                query += " AND tag = ?"
                params.append(tag)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            # Intentionally hard-coded. Update if schema changes
            query += " ORDER BY tag, timestamp, resource_type, resource_name"

            cursor.execute(query, params)

            # Create a list of column names using query result's metadata
            columns = [desc[0] for desc in cursor.description]
            # Pair column names with values and return to user as a dictionary
            return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]

    # NOTE: how to additionally filter by metadata (e.g. machine_name)?
    # NOTE: should we also let user filter by value (e.g. >= or <= some value)?
    # NOTE: how to allow str args to be lists & filter accordingly?
    def query_injection_stat(
        self,
        stat_name: str | None = None,
        start_round_number: int | None = None,
        end_round_number: int | None = None,
        start_chunk_number: int | None = None,
        end_chunk_number: int | None = None,
        start_sample_index: int | None = None,
        end_sample_index: int | None = None,
        start_background_index: int | None = None,
        end_background_index: int | None = None,
        signal_class: str | None = None,
        signal_type: str | None = None,
        injection_stage: str | None = None,
        tag: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        only_valid: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Query from injection_stats table

        Args:
            stat_name: Stat name (e.g. global_mean, eti_snr, rfi_drift_rate, num_samples, etc.)
            start_round_number: Start round number
            end_round_number: End round number
            start_chunk_number: Start chunk number
            end_chunk_number: End chunk number
            start_sample_index: Start sample index
            end_sample_index: End sample index
            start_background_index: Start background index
            end_background_index: End background index
            signal_class: Signal class (e.g. main, false, true)
            signal_type: Signal type (e.g. false_no_signal, false_with_rfi, true_only_eti, true_eti_rfi)
            injection_stage: Optional injection stage (e.g. A: pre-inj pre-norm, B: post-inj pre-norm, C: post-inj post-norm)
            tag: Tag for current pipeline run
            start_time: Start timestamp (unix time)
            end_time: End timestamp (unix time)
            only_valid: If True (default), only return rows where is_valid=1

        Returns:
            List of metric dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # WHERE 1=1 is a trick for building dynamic queries
            # Since it's always true, it does nothing
            # But it lets us safely add more conditions with AND
            query = "SELECT * FROM injection_stats WHERE 1=1"
            params = []

            # Build the query dynamically based on user-specified conditions
            if stat_name:
                query += " AND stat_name = ?"
                params.append(stat_name)

            if start_round_number:
                query += " AND round_number >= ?"
                params.append(start_round_number)

            if end_round_number:
                query += " AND round_number <= ?"
                params.append(end_round_number)

            if start_chunk_number:
                query += " AND chunk_number >= ?"
                params.append(start_chunk_number)

            if end_chunk_number:
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
                query += " AND signal_class = ?"
                params.append(signal_class)

            if signal_type:
                query += " AND signal_type = ?"
                params.append(signal_type)

            if injection_stage:
                query += " AND injection_stage = ?"
                params.append(injection_stage)

            if tag:
                query += " AND tag = ?"
                params.append(tag)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if only_valid:
                query += " AND is_valid = 1"

            # Intentionally hard-coded. Update if schema changes
            query += " ORDER BY tag, signal_class, signal_type, round_number, chunk_number, sample_index, stat_name"

            cursor.execute(query, params)

            # Create a list of column names using query result's metadata
            columns = [desc[0] for desc in cursor.description]
            # Pair column names with values and return to user as a dictionary
            return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]

    # NOTE: how to additionally filter by metadata (e.g. machine_name)?
    # NOTE: should we also let user filter by value (e.g. >= or <= some value)?
    # NOTE: how to allow str args to be lists & filter accordingly?
    def query_training_stat(
        self,
        model_name: str | None = None,
        stat_name: str | None = None,
        start_round_number: int | None = None,
        end_round_number: int | None = None,
        start_epoch_number: int | None = None,
        end_epoch_number: int | None = None,
        tag: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query from training_stats table

        Args:
            model_name: Model name (e.g. 'beta_vae', 'rf')
            stat_name: Stat name (e.g. 'total_loss', 'reconstruction_loss', 'learning_rate')
            start_round_number: Start round number
            end_round_number: End round number
            start_epoch_number: Start epoch number
            end_epoch_number: End epoch number
            tag: Tag for current pipeline run
            start_time: Start timestamp (unix time)
            end_time: End timestamp (unix time)

        Returns:
            List of metric dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # WHERE 1=1 is a trick for building dynamic queries
            # Since it's always true, it does nothing
            # But it lets us safely add more conditions with AND
            query = "SELECT * FROM training_stats WHERE 1=1"
            params = []

            # Build the query dynamically based on user-specified conditions
            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)

            if stat_name:
                query += " AND stat_name = ?"
                params.append(stat_name)

            if start_round_number:
                query += " AND round_number >= ?"
                params.append(start_round_number)

            if end_round_number:
                query += " AND round_number <= ?"
                params.append(end_round_number)

            if start_epoch_number:
                query += " AND epoch_number >= ?"
                params.append(start_epoch_number)

            if end_epoch_number:
                query += " AND epoch_number <= ?"
                params.append(end_epoch_number)

            if tag:
                query += " AND tag = ?"
                params.append(tag)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            # Intentionally hard-coded. Update if schema changes
            query += " ORDER BY tag, model_name, round_number, epoch_number, stat_name"

            cursor.execute(query, params)

            # Create a list of column names using query result's metadata
            columns = [desc[0] for desc in cursor.description]
            # Pair column names with values and return to user as a dictionary
            return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]

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
