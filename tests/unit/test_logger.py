"""Unit tests for aetherscan.logger.logger.StreamToLogger: the file-protocol probe surface
that libraries writing to the redirected sys.stdout/sys.stderr rely on (tqdm progress bars
via huggingface_hub probe isatty(); others probe writable()/fileno())."""

from __future__ import annotations

import io
import logging

import pytest

from aetherscan.logger.logger import StreamToLogger


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def capture_logger():
    log = logging.getLogger("test_stream_to_logger")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    handler = _CaptureHandler()
    log.addHandler(handler)
    yield log, handler
    log.removeHandler(handler)


class TestStreamToLogger:
    def test_write_routes_lines_to_logger(self, capture_logger):
        log, handler = capture_logger
        stream = StreamToLogger(log, logging.INFO)
        stream.write("line one\nline two\n")
        assert [r.getMessage() for r in handler.records] == ["line one", "line two"]
        assert all(r.levelno == logging.INFO for r in handler.records)

    def test_flush_is_safe(self, capture_logger):
        log, _handler = capture_logger
        StreamToLogger(log, logging.INFO).flush()  # must not raise

    def test_isatty_is_false(self, capture_logger):
        # The exact probe that killed the hf_upload stage: huggingface_hub's tqdm progress
        # machinery calls sys.stdout.isatty() on the redirected stream.
        log, _handler = capture_logger
        assert StreamToLogger(log, logging.INFO).isatty() is False

    def test_capability_probes(self, capture_logger):
        log, _handler = capture_logger
        stream = StreamToLogger(log, logging.INFO)
        assert stream.writable() is True
        assert stream.readable() is False

    def test_fileno_raises_unsupported_operation(self, capture_logger):
        # io.UnsupportedOperation subclasses both OSError and ValueError — the two types
        # libraries guard fileno() probes against (matching io.StringIO's behavior).
        log, _handler = capture_logger
        stream = StreamToLogger(log, logging.INFO)
        with pytest.raises(io.UnsupportedOperation):
            stream.fileno()
        assert issubclass(io.UnsupportedOperation, OSError)
        assert issubclass(io.UnsupportedOperation, ValueError)
