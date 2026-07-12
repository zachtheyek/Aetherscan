"""
Persisted training-run state for fault-tolerant resume.

A TrainingRunState manifest lives at {output_path}/run_state_{save_tag}.json and carries
everything a retry (in-process or a full relaunch of the identical command) needs to resume
where the previous attempt died:

- run_start_time: wall clock of attempt 1. TrainingPipeline.__init__ seeds self.start_time
  from it, so every DB query/plot spans the whole run rather than just the current attempt.
- completed_rounds: beta-VAE rounds whose checkpoint was saved; resume starts at max + 1.
- stages_done / stages_failed: drive run_training_pipeline's stage machine — done stages are
  skipped, failed non-critical stages (plots) are retried on the next run and force a nonzero
  exit if they never succeed.

Writes are atomic (tmp -> os.replace, mirroring round_data's .done manifest protocol) so a
crash mid-write can never leave a truncated manifest — the file is either the previous state
or the new one.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# Ordered stage names for run_training_pipeline's stage machine.
STAGE_VAE_ROUNDS = "vae_rounds"
STAGE_VAE_PLOTS = "vae_plots"
STAGE_RF_TRAIN = "rf_train"
STAGE_RF_PLOTS = "rf_plots"
STAGE_FINAL_SAVE = "final_save"
TRAINING_STAGES = (
    STAGE_VAE_ROUNDS,
    STAGE_VAE_PLOTS,
    STAGE_RF_TRAIN,
    STAGE_RF_PLOTS,
    STAGE_FINAL_SAVE,
)


def run_state_path(output_path: str, tag: str) -> str:
    """Manifest location for one training run: {output_path}/run_state_{save_tag}.json."""
    return os.path.join(output_path, f"run_state_{tag}.json")


@dataclass
class TrainingRunState:
    """Persisted state of one training run (identified by its save tag) across attempts."""

    tag: str
    run_start_time: float  # Wall clock of attempt 1 — used by ALL DB queries/plots
    attempt: int = 1  # Incremented per retry (each TrainingPipeline rebuild)
    completed_rounds: list[int] = field(default_factory=list)
    stages_done: list[str] = field(default_factory=list)
    stages_failed: list[str] = field(default_factory=list)

    def is_stage_done(self, stage: str) -> bool:
        return stage in self.stages_done

    def mark_stage_done(self, stage: str) -> None:
        """Record a stage success; a previously recorded failure of the same stage is cleared."""
        if stage not in self.stages_done:
            self.stages_done.append(stage)
        if stage in self.stages_failed:
            self.stages_failed.remove(stage)

    def record_stage_failure(self, stage: str) -> None:
        """Record a non-critical stage failure (the stage stays out of stages_done, so the
        next attempt/relaunch retries it)."""
        if stage not in self.stages_failed:
            self.stages_failed.append(stage)

    def mark_round_completed(self, round_number: int) -> None:
        if round_number not in self.completed_rounds:
            self.completed_rounds.append(round_number)
            self.completed_rounds.sort()

    @property
    def resume_round(self) -> int:
        """1-based round the beta-VAE loop should (re)start from."""
        return max(self.completed_rounds) + 1 if self.completed_rounds else 1


def save_run_state(state: TrainingRunState, path: str) -> None:
    """Persist the manifest atomically (tmp -> os.replace): a crash mid-write leaves either
    the previous manifest or the new one, never a truncated file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(asdict(state), f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def load_run_state(path: str) -> TrainingRunState | None:
    """Load a persisted manifest; returns None when the file is missing or unreadable
    (a corrupt manifest downgrades to a fresh run rather than blocking training)."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        return TrainingRunState(
            tag=str(payload["tag"]),
            run_start_time=float(payload["run_start_time"]),
            attempt=int(payload.get("attempt", 1)),
            completed_rounds=sorted(int(r) for r in payload.get("completed_rounds", [])),
            stages_done=[str(s) for s in payload.get("stages_done", [])],
            stages_failed=[str(s) for s in payload.get("stages_failed", [])],
        )
    except Exception as e:
        logger.warning(f"Failed to load run state from {path}: {e} — treating as a fresh run")
        return None
