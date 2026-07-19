"""
Always-on stage timing for the Aetherscan pipeline.

stage_timer() wraps a pipeline stage (context manager or decorator) and records one
(stage, start_time, end_time, duration_s, tag, metadata) row into the pipeline_stages DB
table through the database's writer queue — two time.time() calls plus one queue put, so
the overhead is negligible and the timers stay on in production runs.

Stage names are hierarchical dot-names ("train.round_02.data_generation"). Nesting is
automatic within a thread: a stage_timer entered while another is active on the same
thread records its name relative to the active one, so instrumented library code (e.g.
the encode/rf sub-stages inside InferencePipeline.run_inference) inherits whatever
umbrella span the caller opened without name plumbing. Timers on different threads don't
interact (the active-stage stack is thread-local), which matters because training,
prefetch preprocessing, and the producer drainer all time stages concurrently.

record_stage() writes a span measured elsewhere (explicit start/end timestamps) — used by
the round-data producer, whose generation happens in another process: the producer can't
touch the DB writer queue (a thread queue.Queue), so it reports (start, end) over its
result-message channel and the main-process drainer records it here.

Rows are read back by utils/benchmark_report.py (stage tree + timeline plot) and by
monitor._save_plot()'s stage-band overlay.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import ContextDecorator

logger = logging.getLogger(__name__)

# Thread-local stack of active stage names (full dotted names), driving relative naming
_ACTIVE_STAGES = threading.local()


def _stage_stack() -> list[str]:
    """The current thread's stack of active (full) stage names."""
    stack = getattr(_ACTIVE_STAGES, "stack", None)
    if stack is None:
        stack = []
        _ACTIVE_STAGES.stack = stack
    return stack


def current_stage() -> str | None:
    """Full dotted name of the innermost stage_timer active on this thread, or None."""
    stack = _stage_stack()
    return stack[-1] if stack else None


def round_stage_name(round_number: int) -> str:
    """Canonical umbrella stage name for a 1-based training round ("train.round_02").

    Shared by the trainer (which opens the umbrella span) and the round-data producer
    (which, running in another process, records the data_generation child by absolute name
    and so can't rely on thread-local nesting). Routing both sites through this helper keeps
    the two name constructions from drifting — a mismatch would orphan the producer's
    data_generation span outside the round subtree in the report tree.
    """
    return f"train.round_{round_number:02d}"


def record_stage(
    stage: str,
    start_time: float,
    end_time: float,
    tag: str | None = None,
    metadata: dict | None = None,
) -> None:
    """
    Record one pipeline stage span with explicit timestamps (no nesting resolution — the
    name is used as-is). tag defaults to the run's save tag. Never raises: benchmarking
    must not be able to fail the pipeline, so a missing DB (unit tests, dev scripts) or a
    serialization hiccup downgrades to a debug/warning log.
    """
    try:
        # Late imports keep this module import-light and avoid any import-cycle risk
        # (db imports config + manager; nothing imports benchmark back).
        from aetherscan.config import get_config  # noqa: PLC0415
        from aetherscan.db import get_db  # noqa: PLC0415

        db = get_db()
        if db is None:
            logger.debug(f"No database instance — dropping stage timing for {stage!r}")
            return

        if tag is None:
            config = get_config()
            # NOTE: tag stays None if a timer fires before checkpoint.save_tag is wired
            # (early init, pre-CLI-parse). Such rows won't match a tag="..." query filter.
            # In practice timers don't fire that early, so this is a documented edge, not a bug.
            tag = config.checkpoint.save_tag if config is not None else None

        db.write_pipeline_stage(
            stage=stage,
            start_time=start_time,
            end_time=end_time,
            tag=tag,
            metadata=json.dumps(metadata) if metadata else None,
        )
    except Exception as e:
        logger.warning(f"Failed to record stage timing for {stage!r}: {e}")


class _StageTimer(ContextDecorator):
    """Context manager / decorator that times a block and records it via record_stage().

    One instance holds per-entry state (full_name/start_time), so a decorated function is
    fine to call repeatedly from one thread but not concurrently from several — for
    concurrent callers, create the timer inside the function (`with stage_timer(...)`)."""

    def __init__(self, stage: str, tag: str | None = None, metadata: dict | None = None):
        self.stage = stage
        self.tag = tag
        self.metadata = metadata
        self.full_name: str | None = None
        self.start_time: float | None = None

    def __enter__(self):
        parent = current_stage()
        self.full_name = f"{parent}.{self.stage}" if parent else self.stage
        _stage_stack().append(self.full_name)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        end_time = time.time()
        stack = _stage_stack()
        # Pop our own frame (guarded: a corrupted stack must not mask the block's result)
        if stack and stack[-1] == self.full_name:
            stack.pop()
        else:
            logger.warning(
                f"stage_timer stack mismatch on exit of {self.full_name!r} "
                f"(top: {(stack[-1] if stack else None)!r})"
            )

        metadata = self.metadata
        if exc_type is not None:
            # Record the failure but never suppress it (return None -> exception propagates)
            metadata = dict(metadata or {})
            metadata["status"] = "failed"
            metadata["error"] = f"{exc_type.__name__}: {exc_value}"

        # Implicit None return -> exceptions propagate (never suppressed)
        record_stage(
            self.full_name,
            self.start_time,
            end_time,
            tag=self.tag,
            metadata=metadata,
        )


def stage_timer(stage: str, tag: str | None = None, metadata: dict | None = None) -> _StageTimer:
    """
    Time a pipeline stage and queue a pipeline_stages DB row on exit.

        with stage_timer("train.round_02.data_generation"):
            ...

        @stage_timer("inference.viz")
        def render(): ...

    `stage` is a dotted hierarchical name. If another stage_timer is active on the same
    thread, `stage` is recorded relative to it (entering stage_timer("encode") while
    "inference.infer_cadence_001" is active records "inference.infer_cadence_001.encode");
    with no active parent it is recorded as-is. On exception the span is still recorded
    (metadata carries status=failed plus the error) and the exception propagates.
    `metadata`, when given, is a small JSON-serializable dict stored on the row.
    """
    return _StageTimer(stage, tag=tag, metadata=metadata)
