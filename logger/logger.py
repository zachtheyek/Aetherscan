"""
Logger for Aetherscan Pipeline
Runs as background thread & uses thread-safe queue-based logging to avoid deadlocks and corrupted
outputs from concurrent writes (e.g. from worker processes)
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue

import tensorflow as tf

from manager import register_logger

logger = logging.getLogger(__name__)


class StreamToLogger:
    """Redirect stream (stdout/stderr) to main logging system"""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        # Flush any remaining content in linebuf if needed
        if self.linebuf:
            self.logger.log(self.level, self.linebuf.rstrip())
            self.linebuf = ""

        # Flush all handlers attached to the logger
        for handler in self.logger.handlers:
            handler.flush()


class Logger:
    """
    Thread-safe logging system with multiprocessing support

    Architecture:
    - Main process runs a QueueListener in a background thread
    - Worker processes send log messages to a shared queue
    - Listener consumes from queue and writes to file/console
    - Eliminates concurrent write issues and corrupted outputs
    """

    _instance = None  # Stores singleton instance
    _lock = threading.Lock()  # Ensures thread safety on object initialization

    # __new__ allocates the object in memory (constructor at the object-creation level)
    # __init__ initializes the object's attributes after it's created
    # since __new__ is called before __init__ every time we instantiate a class,
    # by overriding __new__, we can short-circuit object creation entirely, and control whether a
    # new instance is created, or just return the existing instance
    def __new__(cls, log_filepath: str):
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

    def __init__(self, log_filepath: str):
        """
        Initialize logger

        Args:
            log_filepath: Path to log file
        """
        # Note, __init__ is triggered every time the class's constructor is called,
        # even if __new__ returned the existing singleton instance
        # Hence, we use the _initialized flag to make sure __init__ only runs once
        if self._initialized:
            return

        self._initialized = True
        self.log_filepath = log_filepath

        # Create queue for worker processes (no size limit)
        self.log_queue = Queue(-1)

        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)  # Ignore DEBUG level logs
        root_logger.handlers.clear()  # Clear existing handlers

        # Create formatter
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

        # Setup file handler (only used by main process via listener)
        file_handler = logging.FileHandler(log_filepath, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Setup stream handler (only used by main process via listener)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        # Create queue listener - runs in background thread, writes logs from queue
        self.log_listener = QueueListener(
            self.log_queue, file_handler, stream_handler, respect_handler_level=True
        )
        self.log_listener.start()

        # Add queue handler to root logger (both main and workers use this)
        queue_handler = QueueHandler(self.log_queue)
        root_logger.addHandler(queue_handler)

        # Redirect TensorFlow logs to Python logging
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Show all TF logs
        tf.get_logger().setLevel(logging.INFO)
        tf_logger = tf.get_logger()
        tf_logger.handlers = []  # Remove TF's default handlers
        tf_logger.propagate = True  # Use root logger handlers

        # Capture Python warnings module output
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.setLevel(logging.WARNING)

        # Redirect stdout and stderr to logging
        # This captures print statements and C library output
        # Note that workers will reset these with init_worker_logging to avoid inheritance issues
        sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
        sys.stderr = StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

        logger.info(f"Logger initialized at: {log_filepath}")

    @classmethod
    def _reset(cls):
        """
        Teardown hook for thread-safe singleton
        Resets the logger instance to None

        WARNING: Only use for testing or cleanup after shutdown.
        Calling this while the logger is active will cause issues.
        Should only be called after stop() has completed.
        """
        # Acquire lock to prevent race conditions
        with cls._lock:
            # Discard the singleton instance by removing the global reference
            # Guarantees the next constructor call will produce a fresh instance
            # Note, resources held by the old instance will remain alive unless explicitly closed beforehand
            cls._instance = None
            # Note, can't log here after tear down

    def stop(self):
        """Stop the queue listener thread"""
        if self.log_listener is not None:
            self.log_listener.stop()
            # Note, can't log after this point -- listener thread has stopped
            # All subsequent logs will get queued but never logged


def init_logger(log_filepath: str) -> Logger:
    """
    Initialize global logger instance (call once at startup)

    Args:
        log_filepath: Path to log file

    Returns:
        Logger instance
    """
    logger_instance = Logger(log_filepath)

    register_logger(logger_instance)

    return logger_instance


def init_worker_logging():
    """
    Initialize logging for multiprocessing workers.

    Resets stdout/stderr to avoid inherited StreamToLogger from parent
    and configures queue-based logging for process-safe logging.

    Args:
        log_queue: Queue for sending log messages to main process (optional)
    """
    logger_instance = Logger._instance

    if logger_instance is None:
        logger.warning(
            "No logger instance initialized - disabling worker logging to avoid conflicts"
        )
        logging.getLogger().handlers.clear()
        logging.getLogger().addHandler(logging.NullHandler())
        return

    log_queue = logger_instance.log_queue

    # Reset stdout/stderr to avoid inherited StreamToLogger from parent
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    # Configure process-local logging to use queue
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(QueueHandler(log_queue))
    root_logger.setLevel(logging.INFO)


def get_logger() -> Logger | None:
    """Get the global logger instance"""
    logger_instance = Logger._instance

    if logger_instance is None:
        logger.warning("No logger instance initialized")

    return logger_instance


def shutdown_logger():
    """Shutdown the global logger instance (call on exit)"""
    logger_instance = Logger._instance

    if logger_instance is None:
        logger.warning("No logger instance initialized")
        return

    logger_instance.stop()
    Logger._reset()
