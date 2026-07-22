"""Unit tests for aetherscan.logger.logger:
- StreamToLogger: the file-protocol probe surface that libraries writing to the redirected
  sys.stdout/sys.stderr rely on (tqdm progress bars via huggingface_hub probe isatty(); others
  probe writable()/fileno()).
- log_path_for_tag: the pure tag->path derivation that gives each run its own tag-scoped log file
  (aetherscan_{save_tag}.log) instead of overwriting a single aetherscan.log."""

from __future__ import annotations

import io
import logging
import os

import pytest

from aetherscan.logger.logger import Logger, StreamToLogger, log_path_for_tag, shutdown_logger


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


class TestLogPathForTag:
    """The pure tag->path derivation (issue #215): each run's log lives at
    {output_path}/logs/aetherscan_{save_tag}.log."""

    def test_explicit_tag(self):
        # An explicit --save-tag (e.g. test_v30) names the file aetherscan_test_v30.log.
        path = log_path_for_tag("/datax/out", "test_v30")
        assert path == os.path.join("/datax/out", "logs", "aetherscan_test_v30.log")
        assert os.path.basename(path) == "aetherscan_test_v30.log"

    def test_timestamp_default_tag(self):
        # Untagged runs fall back to the import-time YYYYMMDD_HHMMSS tag; it flows into the
        # filename unchanged, so even untagged runs stop clobbering each other.
        tag = "20260723_141516"
        path = log_path_for_tag("/datax/out", tag)
        assert path == os.path.join("/datax/out", "logs", f"aetherscan_{tag}.log")

    def test_different_tags_yield_distinct_files_same_dir(self):
        # The crux of #215: two tags -> two separate files under the same logs/ dir; neither
        # overwrites the other.
        a = log_path_for_tag("/out", "final_v1")
        b = log_path_for_tag("/out", "final_v2")
        assert a != b
        assert os.path.dirname(a) == os.path.dirname(b) == os.path.join("/out", "logs")


class TestLoggerUsesTaggedPath:
    """Light integration check that Logger actually builds its FileHandler from the tagged path."""

    def test_logger_targets_tag_named_file(self):
        # A Logger built with an explicit tag points log_path (and its FileHandler) at the
        # tag-named file under the config's output_path, and creates it. get_config() is the
        # conftest-initialized singleton (output_path scoped to tmp_path); shutdown_logger()
        # tears the singleton down so it can't leak into the next test.
        from aetherscan.config import get_config  # noqa: PLC0415

        output_path = get_config().output_path
        try:
            instance = Logger(save_tag="test_v30")
            assert instance.log_path == log_path_for_tag(output_path, "test_v30")
            assert os.path.basename(instance.log_path) == "aetherscan_test_v30.log"
            assert os.path.exists(instance.log_path)
        finally:
            shutdown_logger()
