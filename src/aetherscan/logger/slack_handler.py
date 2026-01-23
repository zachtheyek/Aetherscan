# NOTE: come back to this later
# BUG: batched slack messages should be colored by the highest priority message (e.g. red if any message in the batch has priority ERROR). currently seeing INFO + WARN + ERROR be colored yellow (WARN)
"""
Slack logging handler for Aetherscan Pipeline
Provides custom logging handler that sends messages to Slack API with:
- Batched message delivery (buffer size + time interval)
- Thread-based logging (all logs as replies to a run summary message)
- Color-coded messages by log level
- Retry logic with exponential backoff
- Image upload functionality

Threading Model:
- Main thread calls emit() to buffer log records
- Background flush thread (_flush_loop) periodically sends buffered messages
- _cooldown_lock protects cooldown state (_cooldown_until, _consecutive_failures)
- _buffer_lock protects the message buffer

Security Note (Log Injection):
- Log messages are sent to Slack without sanitization of Slack markdown characters
- This is acceptable because logs are internal pipeline output, not user-generated content
- If logs could contain untrusted user input in the future, consider escaping <, >, &
- Timestamps are wrapped in backticks which provides some protection
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, TypedDict

# Defer runtime import to _get_client() as lazy import
# Allows slack_sdk to be an optional dependency that doesn't crash the entire application if it's
# not installed (just throws a warning)
if TYPE_CHECKING:
    from slack_sdk import WebClient

logger = logging.getLogger(__name__)


# Color mapping for Slack message attachments based on log level
LEVEL_COLORS = {
    logging.CRITICAL: "#FF0000",  # Red
    logging.ERROR: "#FFA500",  # Orange
    logging.WARNING: "#FFFF00",  # Yellow
    logging.INFO: "#36A64F",  # Green
    logging.DEBUG: "#FFFFFF",  # White
}

# Level name to emoji mapping for batched messages
LEVEL_EMOJI = {
    logging.CRITICAL: ":rotating_light:",
    logging.ERROR: ":x:",
    logging.WARNING: ":warning:",
    logging.INFO: ":information_source:",
    logging.DEBUG: ":mag:",
}

# Priority mapping for determining batch severity (higher = more severe)
LEVEL_PRIORITY = {
    logging.CRITICAL: 5,
    logging.ERROR: 4,
    logging.WARNING: 3,
    logging.INFO: 2,
    logging.DEBUG: 1,
}

# NOTE: should we parametrize this into config.py?
# Configuration constants
FLUSH_CHECK_INTERVAL = 1.0  # Seconds between flush thread checks
THREAD_STOP_TIMEOUT = 2.0  # Seconds to wait for flush thread to stop
GPU_INFO_TIMEOUT = 5.0  # Seconds to wait for nvidia-smi
MAX_MESSAGE_LENGTH = 500  # Max characters per individual log message
MAX_COMBINED_LENGTH = 3000  # Max characters for combined batch message
MIN_CHANNEL_ID_LENGTH = 9  # Slack channel IDs are C/G/D/Z + 8 chars
CONVERSATIONS_PAGE_SIZE = 200  # Pagination limit for conversations.list API
EXPONENTIAL_BACKOFF_BASE = 2  # Base for exponential backoff calculation


class BufferedMessage(TypedDict):
    """Type definition for buffered log messages."""

    text: str
    color: str
    emoji: str
    level: str
    name: str
    timestamp: str


class SlackHandler(logging.Handler):
    """
    Custom logging handler that sends messages to Slack.

    Features:
    - Batched message delivery to prevent rate limiting
    - Thread-based logging (replies to initial run summary)
    - Color-coded messages by log level
    - Retry logic with exponential backoff
    - Error throttling to prevent spam on consecutive failures
    - Graceful degradation (never crashes the application)
    - Image upload support via upload_file()
    """

    def __init__(
        self,
        token: str,
        channel: str,
        username: str,
        timeout: float,
        retry_attempts: int,
        buffer_size: int,
        flush_interval: float,
        broadcast_level: int,
    ):
        """
        Initialize SlackHandler.

        Args:
            token: Slack Bot User OAuth Token
            channel: Default channel to post messages to (e.g., "#aetherscan-logs")
            username: Bot username displayed in Slack
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
            buffer_size: Maximum messages to buffer before flushing
            flush_interval: Seconds between automatic buffer flushes
            broadcast_level: Log level at which messages are broadcast to main channel
        """
        super().__init__()

        self.channel = channel
        self.username = username
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.broadcast_level = broadcast_level

        # Track consecutive failures for error throttling
        # Protected by _cooldown_lock for thread-safe access
        self._consecutive_failures = 0
        self._max_failures_before_cooldown = 5
        self._cooldown_duration = 300  # 5 minutes
        self._cooldown_until: float | None = None
        self._cooldown_lock = threading.Lock()

        # Initialize Slack client lazily to avoid import errors if slack_sdk not installed
        self._client: WebClient | None = None
        self._token = token

        # Thread-based logging: store the thread timestamp for replies
        self._thread_ts: str | None = None
        self._run_started = False

        # Channel ID cache (files_upload_v2 requires ID, not name)
        # NOTE: This cache is unbounded, but in practice only a few channels (1-3) are used
        # per pipeline run. If dynamic channel generation becomes a use case, consider using
        # functools.lru_cache or adding a TTL-based eviction policy.
        self._channel_id_cache: dict[str, str] = {}

        # Message batching
        self._buffer: list[BufferedMessage] = []
        self._buffer_lock = threading.Lock()
        self._last_flush_time = time.time()
        self._flush_thread: threading.Thread | None = None
        self._stop_flush_thread = threading.Event()

        # Start the background flush thread
        self._start_flush_thread()

    def _start_flush_thread(self):
        """Start the background thread that periodically flushes the buffer."""
        self._stop_flush_thread.clear()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self):
        """Background loop that flushes buffer at regular intervals."""
        while not self._stop_flush_thread.is_set():
            time.sleep(FLUSH_CHECK_INTERVAL)
            # Extract messages while holding lock, then send outside lock
            messages = None
            with self._buffer_lock:
                elapsed = time.time() - self._last_flush_time
                if self._buffer and elapsed >= self.flush_interval:
                    messages = self._extract_buffer_locked()
            # Network call outside the lock to avoid blocking emit()
            if messages:
                self._send_batched_messages(messages)

    def _get_client(self) -> WebClient | None:
        """Lazily initialize and return the Slack WebClient."""
        if self._client is None:
            try:
                from slack_sdk import WebClient  # noqa: PLC0415

                self._client = WebClient(token=self._token, timeout=self.timeout)
            except ImportError:
                logger.warning("slack_sdk not installed. Slack logging disabled.")
                return None
            except Exception as e:
                logger.warning(f"Failed to initialize Slack client: {e}")
                return None
        return self._client

    def _resolve_channel_id(self, channel: str) -> str | None:
        """
        Resolve a channel name to its ID.

        The files_upload_v2 API requires a channel ID, not a name.
        This method converts channel names (like #aetherscan-logs) to IDs (like C01234ABCD).

        Args:
            channel: Channel name (with or without #) or channel ID

        Returns:
            Channel ID, or None if resolution failed
        """
        # If it already looks like a channel ID, return as-is
        # Slack channel IDs start with C (public), G (private), D (DM), or Z (app)
        if channel and len(channel) >= MIN_CHANNEL_ID_LENGTH and channel[0] in "CGDZ":
            return channel

        # Check cache
        cache_key = channel.lstrip("#")
        if cache_key in self._channel_id_cache:
            return self._channel_id_cache[cache_key]

        client = self._get_client()
        if client is None:
            return None

        try:
            # Use conversations.list to find the channel
            channel_name = channel.lstrip("#")
            cursor = None

            while True:
                response = client.conversations_list(
                    types="public_channel,private_channel",
                    limit=CONVERSATIONS_PAGE_SIZE,
                    cursor=cursor,
                )

                if not response.get("ok"):
                    return None

                for ch in response.get("channels", []):
                    if ch.get("name") == channel_name:
                        channel_id = ch.get("id")
                        self._channel_id_cache[cache_key] = channel_id
                        return channel_id

                # Check for pagination
                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

            # Channel not found
            print(f"Could not find channel ID for: {channel}", file=sys.__stderr__)
            return None

        except Exception as e:
            print(f"Failed to resolve channel ID for {channel}: {e}", file=sys.__stderr__)
            return None

    def _is_in_cooldown(self) -> bool:
        """Check if handler is in cooldown period due to consecutive failures."""
        with self._cooldown_lock:
            if self._cooldown_until is None:
                return False
            if time.time() < self._cooldown_until:
                return True
            # Cooldown expired, reset
            self._cooldown_until = None
            self._consecutive_failures = 0
            return False

    def _record_failure(self):
        """Record a failure and enter cooldown if threshold exceeded."""
        with self._cooldown_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_failures_before_cooldown:
                self._cooldown_until = time.time() + self._cooldown_duration
                print(
                    f"Slack handler entering cooldown for {self._cooldown_duration}s "
                    f"after {self._consecutive_failures} consecutive failures",
                    file=sys.__stderr__,
                )

    def _record_success(self):
        """Record a successful send, resetting failure counter."""
        with self._cooldown_lock:
            self._consecutive_failures = 0
            self._cooldown_until = None

    def start_run(self, cli_args: list[str] | None = None) -> bool:
        """
        Post the initial run summary message to Slack.

        All subsequent log messages will be posted as replies to this message.

        Args:
            cli_args: Command line arguments for this run

        Returns:
            True if the run summary was posted successfully
        """
        if self._run_started:
            return True

        if self._is_in_cooldown():
            return False

        client = self._get_client()
        if client is None:
            return False

        # Gather system information
        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = "unknown"

        try:
            cpu_count = os.cpu_count() or "unknown"
        except Exception:
            cpu_count = "unknown"

        # Try to get GPU info
        gpu_info = self._get_gpu_info()

        # Format CLI args
        if cli_args is None:
            cli_args = sys.argv
        cli_args_str = " ".join(cli_args)

        # Build the run summary message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_lines = [
            "*Aetherscan Pipeline Run Started*",
            "",
            f"*Start Time:* {timestamp}",
            f"*Machine:* {hostname}",
            f"*OS:* {platform.system()} {platform.release()}",
            f"*CPU Cores:* {cpu_count}",
        ]

        if gpu_info:
            summary_lines.append(f"*GPUs:* {gpu_info}")

        summary_lines.extend(
            [
                "",
                "*Command:*",
                f"```{cli_args_str}```",
                "",
                "_Logs will appear as batched replies to this message._",
            ]
        )

        summary_text = "\n".join(summary_lines)

        try:
            response = self._send_with_retry_return(
                lambda: client.chat_postMessage(
                    channel=self.channel,
                    text=summary_text,
                    username=self.username,
                    mrkdwn=True,
                )
            )

            if response and response.get("ok"):
                self._thread_ts = response.get("ts")
                self._run_started = True
                return True

        except Exception as e:
            print(f"Failed to post run summary to Slack: {e}", file=sys.__stderr__)

        return False

    def _get_gpu_info(self) -> str | None:
        """Get GPU information if available, grouped by GPU type."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
                timeout=GPU_INFO_TIMEOUT,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpus = result.stdout.strip().split("\n")

                # Count GPUs by name and sum total VRAM
                gpu_counts: dict[str, int] = {}
                total_vram_mb = 0.0

                for gpu in gpus:
                    # Split on comma (with or without space) for robustness
                    parts = gpu.split(",")
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        try:
                            mem_mb = float(parts[1].strip())
                            total_vram_mb += mem_mb
                            gpu_counts[name] = gpu_counts.get(name, 0) + 1
                        except ValueError:
                            # Skip this GPU if memory value can't be parsed
                            continue

                if not gpu_counts:
                    return None

                # Format as "N x GPU_NAME" for each type
                gpu_strs = []
                for name, count in gpu_counts.items():
                    if count > 1:
                        gpu_strs.append(f"{count} x {name}")
                    else:
                        gpu_strs.append(name)

                # Convert total VRAM to GB
                total_vram_gb = total_vram_mb / 1024

                # Format output
                gpu_list = ", ".join(gpu_strs)
                return f"{gpu_list} ({total_vram_gb:.0f}GB combined)"

        except Exception:
            pass
        return None

    def emit(self, record: logging.LogRecord) -> None:
        """
        Buffer log record for batched sending to Slack.

        Args:
            record: The log record to send
        """
        if self._is_in_cooldown():
            return

        try:
            # Format the message
            msg = self.format(record)
            color = LEVEL_COLORS.get(record.levelno, "#808080")
            emoji = LEVEL_EMOJI.get(record.levelno, ":speech_balloon:")

            # Add to buffer and check if flush needed
            messages_to_send = None
            with self._buffer_lock:
                self._buffer.append(
                    BufferedMessage(
                        text=msg,
                        color=color,
                        emoji=emoji,
                        level=record.levelname,
                        name=record.name,
                        timestamp=datetime.now().strftime("%H:%M:%S"),
                    )
                )

                # Extract messages if buffer is full
                if len(self._buffer) >= self.buffer_size:
                    messages_to_send = self._extract_buffer_locked()

            # Send outside the lock to avoid blocking other emit() calls
            if messages_to_send:
                self._send_batched_messages(messages_to_send)

        except Exception as e:
            print(f"SlackHandler.emit() error: {e}", file=sys.__stderr__)
            self._record_failure()

    def _extract_buffer_locked(self) -> list[BufferedMessage] | None:
        """
        Extract messages from buffer. Must be called with _buffer_lock held.

        Returns:
            List of buffered messages, or None if buffer was empty.
        """
        if not self._buffer:
            return None

        messages = self._buffer.copy()
        self._buffer.clear()
        self._last_flush_time = time.time()
        return messages

    def flush(self):
        """Flush any buffered messages immediately."""
        messages = None
        with self._buffer_lock:
            messages = self._extract_buffer_locked()
        # Send outside the lock
        if messages:
            self._send_batched_messages(messages)

    def _send_batched_messages(self, messages: list[BufferedMessage]):
        """Send a batch of messages as a single Slack message."""
        if self._is_in_cooldown():
            return

        client = self._get_client()
        if client is None:
            return

        # Determine the highest severity level for the batch color
        max_level = logging.DEBUG
        for msg in messages:
            level_name = msg.get("level", "INFO")
            level_num = getattr(logging, level_name, logging.INFO)
            if LEVEL_PRIORITY.get(level_num, 0) > LEVEL_PRIORITY.get(max_level, 0):
                max_level = level_num

        batch_color = LEVEL_COLORS.get(max_level, "#808080")

        # Build the combined message text
        lines = []
        for msg in messages:
            emoji = msg.get("emoji", "")
            timestamp = msg.get("timestamp", "")
            text = msg.get("text", "")
            # Truncate very long messages
            if len(text) > MAX_MESSAGE_LENGTH:
                text = text[: MAX_MESSAGE_LENGTH - 3] + "..."
            lines.append(f"{emoji} `{timestamp}` {text}")

        combined_text = "\n".join(lines)

        # Truncate if total message is too long (Slack limit is ~40k chars)
        if len(combined_text) > MAX_COMBINED_LENGTH:
            combined_text = combined_text[: MAX_COMBINED_LENGTH - 4] + "\n..."

        # Build the message with text field to avoid warnings
        try:
            kwargs = {
                "channel": self.channel,
                "text": f"{len(messages)} log message(s)",  # Fallback text for notifications
                "username": self.username,
                "attachments": [
                    {
                        "color": batch_color,
                        "text": combined_text,
                        "fallback": f"{len(messages)} log message(s)",  # Fixes warning
                        "mrkdwn_in": ["text"],
                    }
                ],
            }

            # Add thread_ts if we have a run thread
            if self._thread_ts:
                kwargs["thread_ts"] = self._thread_ts
                # Broadcast to main channel if max severity >= broadcast_level
                if max_level >= self.broadcast_level:
                    kwargs["reply_broadcast"] = True

            self._send_with_retry(lambda: client.chat_postMessage(**kwargs))

        except Exception as e:
            print(f"Failed to send batched messages to Slack: {e}", file=sys.__stderr__)
            self._record_failure()

    def _send_with_retry(self, send_func):
        """
        Execute send function with exponential backoff retry.

        Args:
            send_func: Callable that performs the Slack API call
        """
        last_exception = None

        for attempt in range(self.retry_attempts + 1):
            try:
                send_func()
                self._record_success()
                return
            except Exception as e:
                last_exception = e
                if attempt < self.retry_attempts:
                    wait_time = EXPONENTIAL_BACKOFF_BASE**attempt
                    time.sleep(wait_time)

        self._record_failure()
        if last_exception:
            print(
                f"Slack send failed after {self.retry_attempts + 1} attempts: {last_exception}",
                file=sys.__stderr__,
            )

    def _send_with_retry_return(self, send_func):
        """
        Execute send function with retry and return the response.

        Args:
            send_func: Callable that performs the Slack API call

        Returns:
            The API response dict, or None on failure
        """
        last_exception = None

        for attempt in range(self.retry_attempts + 1):
            try:
                response = send_func()
                self._record_success()
                return response
            except Exception as e:
                last_exception = e
                if attempt < self.retry_attempts:
                    wait_time = EXPONENTIAL_BACKOFF_BASE**attempt
                    time.sleep(wait_time)

        self._record_failure()
        if last_exception:
            print(
                f"Slack send failed after {self.retry_attempts + 1} attempts: {last_exception}",
                file=sys.__stderr__,
            )
        return None

    def upload_file(
        self,
        file_path: str,
        channels: str | list[str] | None = None,
        title: str | None = None,
        initial_comment: str | None = None,
        broadcast: bool = True,
    ) -> bool:
        """
        Upload a file to Slack.

        When uploading to a thread, the file is posted in the thread. If broadcast=True,
        a message is also posted to the main channel with a link back to the thread
        (similar to the "Also send to channel" checkbox in Slack).

        Args:
            file_path: Path to the file to upload
            channels: Channel(s) to upload to (defaults to handler's channel)
            title: Title for the file
            initial_comment: Comment to add with the file
            broadcast: If True and in a thread, also echo to main channel

        Returns:
            True if upload succeeded, False otherwise
        """
        if not os.path.exists(file_path):
            print(f"File not found for Slack upload: {file_path}", file=sys.__stderr__)
            return False

        if self._is_in_cooldown():
            return False

        client = self._get_client()
        if client is None:
            return False

        if channels is None:
            target_channel = self.channel
        elif isinstance(channels, list):
            # For simplicity, use first channel for file upload
            target_channel = channels[0] if channels else self.channel
        else:
            target_channel = channels

        if not target_channel:
            print("No channel specified for Slack file upload", file=sys.__stderr__)
            return False

        # Resolve channel name to ID (required for files_upload_v2)
        channel_id = self._resolve_channel_id(target_channel)
        if not channel_id:
            print(f"Could not resolve channel ID for: {target_channel}", file=sys.__stderr__)
            return False

        try:
            kwargs = {
                "channel": channel_id,
                "file": file_path,
                "title": title,
                "initial_comment": initial_comment,
            }

            # Upload in thread if we have one
            if self._thread_ts:
                kwargs["thread_ts"] = self._thread_ts

            response = self._send_with_retry_return(lambda: client.files_upload_v2(**kwargs))

            # If we're in a thread and broadcast is enabled, post a broadcast message
            # This echoes the upload to the main channel with a link to the thread
            if response and self._thread_ts and broadcast:
                self._broadcast_file_upload(client, target_channel, title, initial_comment)

            return response is not None
        except Exception as e:
            print(f"Failed to upload file to Slack: {e}", file=sys.__stderr__)
            return False

    def _broadcast_file_upload(
        self,
        client: WebClient,
        channel: str,
        title: str | None,
        comment: str | None,
    ):
        """
        Post a broadcast message to the main channel announcing a file upload in the thread.

        Args:
            client: Slack WebClient instance
            channel: Channel to broadcast to
            title: Title of the uploaded file
            comment: Comment associated with the file
        """
        if not self._thread_ts:
            return

        # Build a brief announcement message
        if title:
            text = f":chart_with_upwards_trend: *{title}*"
        else:
            text = ":chart_with_upwards_trend: *New plot uploaded*"

        if comment:
            text += f"\n{comment}"

        try:
            client.chat_postMessage(
                channel=channel,
                text=text,
                thread_ts=self._thread_ts,
                reply_broadcast=True,  # This echoes to the main channel
                username=self.username,
            )
        except Exception as e:
            # Don't fail the upload if broadcast fails
            print(f"Failed to broadcast file upload: {e}", file=sys.__stderr__)

    def close(self):
        """Close the handler and clean up resources."""
        # Stop the flush thread
        self._stop_flush_thread.set()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=THREAD_STOP_TIMEOUT)
            if self._flush_thread.is_alive():
                print(
                    f"SlackHandler flush thread did not stop within {THREAD_STOP_TIMEOUT}s",
                    file=sys.__stderr__,
                )

        # Flush any remaining messages
        self.flush()

        super().close()
        self._client = None
