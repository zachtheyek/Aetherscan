"""Unit tests for aetherscan.logger.logger:
- StreamToLogger: the file-protocol probe surface that libraries writing to the redirected
  sys.stdout/sys.stderr rely on (tqdm progress bars via huggingface_hub probe isatty(); others
  probe writable()/fileno()).
- log_path_for_tag: the pure tag->path derivation that gives each run its own tag-scoped log file
  (aetherscan_{save_tag}.log) instead of overwriting a single aetherscan.log."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
from unittest.mock import patch

import pytest

from aetherscan.logger.logger import (
    Logger,
    ShutdownSafeQueueHandler,
    StreamToLogger,
    log_path_for_tag,
    shutdown_logger,
)


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

    def test_logger_falls_back_to_config_tag_when_omitted(self):
        # No save_tag passed (the untagged-run / default-timestamp-tag path, arguably more
        # common than the explicit-tag path) — Logger must fall back to
        # config.checkpoint.save_tag rather than name the file after a None/sentinel tag.
        from aetherscan.config import get_config  # noqa: PLC0415

        config = get_config()
        try:
            instance = Logger()
            expected = log_path_for_tag(config.output_path, config.checkpoint.save_tag)
            assert instance.log_path == expected
            assert os.path.exists(instance.log_path)
        finally:
            shutdown_logger()


class TestLoggerStop:
    """Regression guard for the #221 interpreter-hang fix in Logger.stop()."""

    def test_stop_disposes_log_queue_feeder_thread(self):
        # stop() must call BOTH log_queue.close() AND log_queue.cancel_join_thread(): once the
        # QueueListener (consumer) is stopped, an mp.Queue with a live feeder thread hangs the
        # interpreter's exit-time join unless both are called (issue #221, logger.py stop()).
        # Dropping either call currently only surfaces as a >90s suite timeout — this converts
        # it into a fast, localized failure. Build a real Logger (the conftest-initialized
        # config singleton scopes output_path to tmp_path) and spy on the queue's disposal calls.
        instance = Logger(save_tag="test_v31")
        try:
            with (
                patch.object(instance.log_queue, "close") as mock_close,
                patch.object(instance.log_queue, "cancel_join_thread") as mock_cancel,
            ):
                instance.stop()
            mock_close.assert_called_once()
            mock_cancel.assert_called_once()
        finally:
            # stop() ran with close()/cancel_join_thread() mocked out, so the real queue was
            # never disposed — do it for real now so its feeder thread can't outlive the test
            # (the very hang #221 fixed). Then drop the singleton via _reset() rather than
            # shutdown_logger(): stop() is not idempotent (QueueListener.stop() raises on a
            # second call), so shutdown_logger()'s stop() would crash after the explicit one above.
            with contextlib.suppress(Exception):
                instance.log_queue.close()
                instance.log_queue.cancel_join_thread()
            Logger._reset()


class TestTeardownRecursionFix:
    """#281: Logger.stop() must restore the real streams and disarm the logging error path,
    so teardown-time output can never recurse through a redirected stderr."""

    def _build_logger(self):
        return Logger(save_tag="test_v281")

    def test_stop_restores_real_streams(self):
        instance = self._build_logger()
        try:
            assert isinstance(sys.stdout, StreamToLogger)
            assert isinstance(sys.stderr, StreamToLogger)
            instance.stop()
            assert sys.stdout is sys.__stdout__
            assert sys.stderr is sys.__stderr__
        finally:
            Logger._reset()

    def test_stop_leaves_foreign_redirect_alone(self):
        instance = self._build_logger()
        foreign = io.StringIO()
        try:
            sys.stderr = foreign  # someone else redirected after us (e.g. pytest capture)
            instance.stop()
            assert sys.stderr is foreign  # not clobbered
            assert sys.stdout is sys.__stdout__  # ours was restored
        finally:
            Logger._reset()

    def test_logging_after_stop_is_a_silent_noop(self):
        instance = self._build_logger()
        try:
            instance.stop()
            # The root logger still carries the (shutdown-safe) queue handler; emitting must
            # neither raise nor recurse now that the queue is closed
            logging.getLogger("post_stop_probe").error("logged after Logger.stop()")
            sys.stderr.write("written after Logger.stop()\n")
        finally:
            Logger._reset()

    def test_stop_is_idempotent(self):
        instance = self._build_logger()
        try:
            instance.stop()
            instance.stop()  # QueueListener.stop() would raise without the guard
        finally:
            Logger._reset()

    def test_shutdown_safe_handler_drops_on_closed_queue(self):
        from multiprocessing import Queue as MpQueue  # noqa: PLC0415

        queue = MpQueue(-1)
        handler = ShutdownSafeQueueHandler(queue)
        record = logging.LogRecord("probe", logging.INFO, __file__, 1, "msg", None, None)
        queue.close()
        queue.cancel_join_thread()
        handler.emit(record)  # must not raise (vanilla QueueHandler would ValueError)

    def test_stream_to_logger_write_is_reentrancy_guarded(self, capture_logger):
        log, handler = capture_logger
        stream = StreamToLogger(log, logging.INFO)

        class _Recurser(logging.Handler):
            def emit(self, record):
                stream.write("recursed")  # second entry must be dropped, not recurse

        recurser = _Recurser()
        log.addHandler(recurser)
        try:
            stream.write("outer")
        finally:
            log.removeHandler(recurser)
        # Only the outer write landed; the re-entrant one was dropped instead of recursing
        assert [r.getMessage() for r in handler.records] == ["outer"]
