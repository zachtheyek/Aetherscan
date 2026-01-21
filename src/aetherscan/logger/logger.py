# TODO: add tag to file log & archive old logs
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

from aetherscan.config import get_config

# NOTE: can this just be from aetherscan.logger import SlackHandler?
from aetherscan.logger.slack_handler import SlackHandler

logger = logging.getLogger(__name__)


def _parse_level(level_str: str) -> int:
    """
    Convert log level string to logging constant.

    Args:
        level_str: Log level name (e.g., "INFO", "WARNING", "DEBUG")

    Returns:
        Logging level constant (e.g., logging.INFO)
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level_str.upper(), logging.INFO)


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
        """Initialize logger"""
        # Note, __init__ is triggered every time the class's constructor is called,
        # even if __new__ returned the existing singleton instance
        # Hence, we use the _initialized flag to make sure __init__ only runs once
        if self._initialized:
            return

        self._initialized = True

        self.config = get_config()
        if self.config is None:
            raise ValueError("get_config() returned None")

        self.log_path = os.path.join(self.config.output_path, "logs", "aetherscan.log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)  # Create dir if it doesn't exist

        # Create queue for worker processes (no size limit)
        self.log_queue = Queue(-1)

        # Setup root logger - set to DEBUG to let handlers filter
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()  # Clear existing handlers

        # Create formatter
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

        # Setup file handler (only used by main process via listener)
        file_handler = logging.FileHandler(self.log_path, mode="w")
        file_handler.setLevel(_parse_level(self.config.logger.file_level))
        file_handler.setFormatter(formatter)

        # Setup stream handler (only used by main process via listener)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(_parse_level(self.config.logger.console_level))
        stream_handler.setFormatter(formatter)

        # Build handler list for QueueListener
        handlers: list[logging.Handler] = [file_handler, stream_handler]

        # Initialize Slack handler if enabled
        self.slack_handler: SlackHandler | None = None
        slack_init_message: str | None = None
        slack_init_level: int = logging.INFO

        if self.config.logger.slack_enabled:
            slack_token = os.environ.get("SLACK_BOT_TOKEN")
            slack_channel = os.environ.get("SLACK_CHANNEL", self.config.logger.slack_channel)

            # Defer logging until after the queue infrastructure is ready
            if not slack_token:
                slack_init_message = (
                    "Slack logging enabled but SLACK_BOT_TOKEN not found in environment. "
                    "Slack logging disabled."
                )
                slack_init_level = logging.WARNING
            elif not slack_channel:
                slack_init_message = (
                    "Slack logging enabled but no channel configured. "
                    "Set SLACK_CHANNEL env var or config.logger.slack_channel. Slack logging disabled."
                )
                slack_init_level = logging.WARNING
            else:
                try:
                    self.slack_handler = SlackHandler(
                        token=slack_token,
                        channel=slack_channel,
                        username=self.config.logger.slack_username,
                        timeout=self.config.logger.slack_timeout,
                        retry_attempts=self.config.logger.slack_retry_attempts,
                        buffer_size=self.config.logger.slack_buffer_size,
                        flush_interval=self.config.logger.slack_flush_interval,
                        broadcast_level=_parse_level(self.config.logger.slack_broadcast_level),
                    )
                    self.slack_handler.setLevel(_parse_level(self.config.logger.slack_level))
                    slack_formatter = logging.Formatter("%(message)s")
                    self.slack_handler.setFormatter(slack_formatter)
                    handlers.append(self.slack_handler)
                    slack_init_message = f"Slack handler initialized for channel: {slack_channel}"
                    slack_init_level = logging.INFO
                except Exception as e:
                    slack_init_message = f"Failed to initialize Slack handler: {e}"
                    slack_init_level = logging.WARNING

        # Create queue listener - runs in background thread, writes logs from queue
        self.log_listener = QueueListener(self.log_queue, *handlers, respect_handler_level=True)
        self.log_listener.start()

        # Add queue handler to root logger (both main and workers use this)
        queue_handler = QueueHandler(self.log_queue)
        root_logger.addHandler(queue_handler)

        # Start the Slack run thread FIRST (posts run summary, all logs become replies)
        # This must happen before any log messages so they appear in the thread
        if self.slack_handler is not None:
            self.slack_handler.start_run()

        # Now that logging infrastructure is ready, log Slack initialization status
        if slack_init_message:
            logger.log(slack_init_level, slack_init_message)

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

        logger.info(f"Logger initialized at: {self.log_path}")

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
        """Stop the queue listener thread and flush Slack messages"""
        # Flush and close Slack handler first (while listener is still running)
        if self.slack_handler is not None:
            self.slack_handler.close()

        if self.log_listener is not None:
            self.log_listener.stop()
            # Note, can't log after this point -- listener thread has stopped
            # All subsequent logs will get queued but never logged

    # NOTE:
    # currently only title is broadcasted to main channel,
    # since there's a delay in the title vs image appearing in the thread
    # not sure if this is a fundamental limitation to slack webhooks?
    def upload_image_to_slack(
        self,
        file_path: str,
        channels: str | list[str] | None = None,
        title: str | None = None,
        message: str | None = None,
        broadcast: bool = True,
    ) -> bool:
        """
        Upload an image file to Slack.

        The image is uploaded to the current run's thread. If broadcast=True (default),
        a message is also posted to the main channel with a link back to the thread
        (similar to the "Also send to channel" checkbox in Slack).

        Args:
            file_path: Path to the image file to upload
            channels: Channel(s) to upload to (defaults to handler's configured channel)
            title: Title for the image
            message: Comment to add with the image
            broadcast: If True, also echo to main channel with link to thread

        Returns:
            True if upload succeeded, False otherwise
        """
        if self.slack_handler is None:
            logger.debug("Slack handler not initialized, skipping image upload")
            return False

        return self.slack_handler.upload_file(
            file_path=file_path,
            channels=channels,
            title=title,
            initial_comment=message,
            broadcast=broadcast,
        )


def init_logger() -> Logger:
    """
    Initialize global logger instance (call once at startup)
    """
    logger_instance = Logger()

    # Note, unlike other modules, due to dependency chains,
    # we wait to call register_logger() inside main.py:main()

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
