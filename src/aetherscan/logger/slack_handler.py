"""
Slack logging handler for Aetherscan Pipeline
Provides custom logging handler that sends messages to Slack API with retry logic,
color-coded messages, and image upload functionality.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slack_sdk import WebClient

logger = logging.getLogger(__name__)


# Color mapping for Slack message attachments based on log level
LEVEL_COLORS = {
    logging.CRITICAL: "#8B0000",  # Dark red
    logging.ERROR: "#FF0000",  # Red
    logging.WARNING: "#FFA500",  # Orange/Yellow
    logging.INFO: "#36A64F",  # Green
    logging.DEBUG: "#808080",  # Gray
}


class SlackHandler(logging.Handler):
    """
    Custom logging handler that sends messages to Slack.

    Features:
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
        username: str = "Aetherscan Bot",
        icon_emoji: str = ":robot_face:",
        timeout: float = 5.0,
        retry_attempts: int = 2,
    ):
        """
        Initialize SlackHandler.

        Args:
            token: Slack Bot User OAuth Token
            channel: Default channel to post messages to (e.g., "#aetherscan-logs")
            username: Bot username displayed in Slack
            icon_emoji: Emoji icon for the bot
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
        """
        super().__init__()

        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji
        self.timeout = timeout
        self.retry_attempts = retry_attempts

        # Track consecutive failures for error throttling
        self._consecutive_failures = 0
        self._max_failures_before_cooldown = 5
        self._cooldown_duration = 300  # 5 minutes
        self._cooldown_until: float | None = None

        # Initialize Slack client lazily to avoid import errors if slack_sdk not installed
        self._client: WebClient | None = None
        self._token = token

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

    def _is_in_cooldown(self) -> bool:
        """Check if handler is in cooldown period due to consecutive failures."""
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
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_failures_before_cooldown:
            self._cooldown_until = time.time() + self._cooldown_duration
            # Use stderr directly to avoid recursive logging
            print(
                f"Slack handler entering cooldown for {self._cooldown_duration}s "
                f"after {self._consecutive_failures} consecutive failures",
                file=sys.__stderr__,
            )

    def _record_success(self):
        """Record a successful send, resetting failure counter."""
        self._consecutive_failures = 0
        self._cooldown_until = None

    def emit(self, record: logging.LogRecord) -> None:
        """
        Send log record to Slack.

        Args:
            record: The log record to send
        """
        # Skip if in cooldown
        if self._is_in_cooldown():
            return

        client = self._get_client()
        if client is None:
            return

        try:
            # Format the message
            msg = self.format(record)

            # Get color based on level
            color = LEVEL_COLORS.get(record.levelno, "#808080")

            # Build attachment with color
            attachments = [
                {
                    "color": color,
                    "text": msg,
                    "footer": f"{record.name} | {record.levelname}",
                }
            ]

            # Send with retry logic
            self._send_with_retry(
                lambda: client.chat_postMessage(
                    channel=self.channel,
                    username=self.username,
                    icon_emoji=self.icon_emoji,
                    attachments=attachments,
                )
            )

        except Exception:
            # Silently fail - don't crash the application
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
                    # Exponential backoff: 1s, 2s, 4s, ...
                    wait_time = 2**attempt
                    time.sleep(wait_time)

        # All retries failed
        self._record_failure()
        if last_exception:
            # Use stderr directly to avoid recursive logging
            print(
                f"Slack send failed after {self.retry_attempts + 1} attempts: {last_exception}",
                file=sys.__stderr__,
            )

    def upload_file(
        self,
        file_path: str,
        channels: str | list[str] | None = None,
        title: str | None = None,
        initial_comment: str | None = None,
    ) -> bool:
        """
        Upload a file to Slack.

        Args:
            file_path: Path to the file to upload
            channels: Channel(s) to upload to (defaults to handler's channel)
            title: Title for the file
            initial_comment: Comment to add with the file

        Returns:
            True if upload succeeded, False otherwise
        """
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found for Slack upload: {file_path}")
            return False

        # Skip if in cooldown
        if self._is_in_cooldown():
            return False

        client = self._get_client()
        if client is None:
            return False

        # Determine target channel(s)
        if channels is None:
            target_channels = self.channel
        elif isinstance(channels, list):
            target_channels = ",".join(channels)
        else:
            target_channels = channels

        if not target_channels:
            logger.warning("No channel specified for Slack file upload")
            return False

        try:
            self._send_with_retry(
                lambda: client.files_upload_v2(
                    channel=target_channels,
                    file=file_path,
                    title=title,
                    initial_comment=initial_comment,
                )
            )
            return True
        except Exception as e:
            logger.debug(f"Failed to upload file to Slack: {e}")
            return False

    def close(self):
        """Close the handler and clean up resources."""
        super().close()
        self._client = None
