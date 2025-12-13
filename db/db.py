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

from config import get_config
from manager import register_db

logger = logging.getLogger(__name__)


def get_system_metadata() -> str:
    """
    Collects system metadata (machine name, user name, IP address)
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

    # Pack into JSON
    metadata = {
        "machine_name": machine_name,
        "user_name": user_name,
        "ip_address": ip_address,
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

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON system_resources(timestamp)
            """)

            # Background statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS background_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    stat_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    filename TEXT,
                    process_id INTEGER,
                    chunk_number INTEGER,
                    duration_seconds REAL,
                    tag TEXT,
                    metadata TEXT
                )
            """)

            # TODO: add additional index for frequent queries, then add wrapper func below
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON background_stats(timestamp)
            """)

            # Injection statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS injection_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    stat_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    model_name TEXT,
                    round_number INTEGER,
                    process_id INTEGER,
                    chunk_number INTEGER,
                    duration_seconds REAL,
                    tag TEXT,
                    metadata TEXT
                )
            """)

            # TODO: add additional index for frequent queries, then add wrapper func below
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON injection_stats(timestamp)
            """)

            # Training loss table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_losses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    loss_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    round_number INTEGER,
                    epoch_number INTEGER,
                    learning_rate REAL,
                    duration_seconds REAL,
                    tag TEXT,
                    metadata TEXT
                )
            """)

            # TODO: add additional index for frequent queries, then add wrapper func below
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON training_losses(timestamp)
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
        """Write buffered data to database in a single transaction"""
        if not self.buffer:
            return

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                for table, values in self.buffer:
                    if table == "system_resources":
                        cursor.execute(
                            """
                            INSERT INTO system_resources
                            (timestamp, resource_type, resource_name, value, unit, tag, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                            values,
                        )
                    elif table == "background_stats":
                        cursor.execute(
                            """
                            INSERT INTO background_stats
                            (timestamp, stat_name, value, unit, filename, process_id,
                             chunk_number, duration_seconds, tag, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            values,
                        )
                    elif table == "injection_stats":
                        cursor.execute(
                            """
                            INSERT INTO injection_stats
                            (timestamp, stat_name, value, unit, model_name, round_number,
                             process_id, chunk_number, duration_seconds, tag, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            values,
                        )
                    elif table == "training_losses":
                        cursor.execute(
                            """
                            INSERT INTO training_losses
                            (timestamp, model_name, loss_name, value, round_number,
                             epoch_number, learning_rate, duration_seconds, tag, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            values,
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

    def write_background_stat(
        self,
        stat_name: str,
        value: float,
        unit: str | None = None,
        filename: str | None = None,
        process_id: int | None = None,
        chunk_number: int | None = None,
        duration_seconds: float | None = None,
        tag: str | None = None,
    ):
        """
        Queue write to background_stats table (non-blocking)

        Args:
            stat_name: Type of statistic (e.g. mean, stdev, skew, kurtosis, etc.)
            value: Statistic value
            unit: Optional unit of measurement
            filename: Optional filename of background
            process_id: Optional process ID that calculated the statistic
            chunk_number: Optional chunk number being processed
            duration_seconds: Optional operation duration in seconds
            tag: Optional tag for current pipeline run
        """
        metadata_json = get_system_metadata()

        self.write_queue.put(
            (
                "background_stats",
                (
                    time.time(),
                    stat_name,
                    value,
                    unit,
                    filename,
                    process_id,
                    chunk_number,
                    duration_seconds,
                    tag,
                    metadata_json,
                ),
            )
        )

    def write_injection_stat(
        self,
        stat_name: str,
        value: float,
        unit: str | None = None,
        model_name: str | None = None,
        round_number: int | None = None,
        process_id: int | None = None,
        chunk_number: int | None = None,
        duration_seconds: float | None = None,
        tag: str | None = None,
    ):
        """
        Queue write to injection_stats table (non-blocking)

        Args:
            stat_name: Type of statistic (e.g. mean, stdev, skew, kurtosis, etc.)
            value: Statistic value
            unit: Optional unit of measurement
            model_name: Optional model name that requested injection (e.g. 'beta_vae', 'rf')
            round_number: Optional current training round number
            process_id: Optional process ID that calculated the statistic
            chunk_number: Optional chunk number being processed
            duration_seconds: Optional operation duration in seconds
            tag: Optional tag for current pipeline run
        """
        metadata_json = get_system_metadata()

        self.write_queue.put(
            (
                "injection_stats",
                (
                    time.time(),
                    stat_name,
                    value,
                    unit,
                    model_name,
                    round_number,
                    process_id,
                    chunk_number,
                    duration_seconds,
                    tag,
                    metadata_json,
                ),
            )
        )

    def write_training_loss(
        self,
        model_name: str,
        loss_name: str,
        value: float,
        round_number: int | None = None,
        epoch_number: int | None = None,
        learning_rate: float | None = None,
        duration_seconds: float | None = None,
        tag: str | None = None,
    ):
        """
        Queue write to training_losses table (non-blocking)

        Args:
            model_name: Model name (e.g. 'beta_vae', 'rf')
            loss_name: Loss name (e.g. 'total_loss', 'reconstruction_loss')
            value: Loss value
            round_number: Optional current training round number
            epoch_number: Optional current training epoch number
            learning_rate: Optional current learning rate
            duration_seconds: Optional operation duration in seconds
            tag: Optional tag for current pipeline run
        """
        metadata_json = get_system_metadata()

        self.write_queue.put(
            (
                "training_losses",
                (
                    time.time(),
                    model_name,
                    loss_name,
                    value,
                    round_number,
                    epoch_number,
                    learning_rate,
                    duration_seconds,
                    tag,
                    metadata_json,
                ),
            )
        )

    def query_system_resource(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query from system_resources table

        Args:
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
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp"

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

            cursor.execute("SELECT COUNT(*) FROM background_stats")
            stats["background_stats_row_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM injection_stats")
            stats["injection_stats_row_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM training_losses")
            stats["training_losses_row_count"] = cursor.fetchone()[0]

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


def shutdown_db() -> None:
    """Shutdown the global database instance (call on exit)"""
    db = Database._instance

    if db is None:
        logger.warning("No database instance initialized")
        return

    db.stop()
    Database._reset()
