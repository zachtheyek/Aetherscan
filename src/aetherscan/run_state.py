"""
Persisted training-run state for fault-tolerant resume.

A TrainingRunState manifest lives at {output_path}/run_state_{save_tag}.json and carries
everything a retry (in-process or a full relaunch of the identical command) needs to resume
where the previous attempt died:

- run_start_time: wall clock of attempt 1. TrainingPipeline.__init__ seeds self.start_time
  from it, so every DB query/plot spans the whole run rather than just the current attempt.
- completed_rounds: beta-VAE rounds whose checkpoint was saved; resume starts at max + 1.
- stages_done / stages_failed: drive run_training_pipeline's stage machine — done stages are
  skipped, failed non-critical stages (plots, hf_upload) are retried on the next run and force a nonzero
  exit if they never succeed.

Writes are atomic (tmp -> os.replace, mirroring round_data's .done manifest protocol) so a
crash mid-write can never leave a truncated manifest — the file is either the previous state
or the new one.
"""

from __future__ import annotations

import contextlib
import hashlib
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
STAGE_HF_UPLOAD = "hf_upload"  # Opt-in (config.hf.upload_after_training); non-critical
TRAINING_STAGES = (
    STAGE_VAE_ROUNDS,
    STAGE_VAE_PLOTS,
    STAGE_RF_TRAIN,
    STAGE_RF_PLOTS,
    STAGE_FINAL_SAVE,
    STAGE_HF_UPLOAD,
)


def run_state_path(output_path: str, tag: str) -> str:
    """Manifest location for one training run: {output_path}/run_state_{save_tag}.json."""
    return os.path.join(output_path, f"run_state_{tag}.json")


# Config sections excluded from a run's fingerprint: pure infra/runtime (db, manager, monitor,
# logger), resume-control (checkpoint), environment paths, the inference config, and the HF
# upload config — none of which change the training result, so a change to any of them must NOT
# force a fresh run. In particular, toggling --hf-upload or changing its repo/revision must never
# be read as training drift (that would discard the manifest and overwrite a completed model).
_FINGERPRINT_EXCLUDE_SECTIONS = frozenset(
    {"db", "manager", "monitor", "logger", "inference", "paths", "checkpoint", "hf"}
)
# Retry knobs live in the training section but only control the retry loop, not the result.
_FINGERPRINT_EXCLUDE_TRAINING_KEYS = frozenset({"max_retries", "retry_delay"})


def config_fingerprint(config_dict: dict) -> str:
    """
    Stable hash of the training-result-affecting config (from Config.to_dict()), used to detect
    that a run's config changed under a reused --save-tag. Excludes infra/runtime, resume-control,
    environment paths, the inference config, and the retry knobs (see the constants above) so a
    change to any of those does not spuriously force a fresh run.
    """
    relevant = {
        section: values
        for section, values in config_dict.items()
        if section not in _FINGERPRINT_EXCLUDE_SECTIONS
    }
    training = relevant.get("training")
    if isinstance(training, dict):
        relevant["training"] = {
            k: v for k, v in training.items() if k not in _FINGERPRINT_EXCLUDE_TRAINING_KEYS
        }
    canonical = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def config_changed(state: TrainingRunState | None, current_fingerprint: str) -> bool:
    """
    True when a persisted manifest exists but was written under a different config fingerprint —
    i.e. the resolved training config changed under a reused save tag. The caller downgrades to a
    fresh run so a stale model is never silently reused/reported as success (a manifest predating
    fingerprinting carries an empty string, which mismatches any real fingerprint -> fresh run).
    """
    return state is not None and state.config_fingerprint != current_fingerprint


# Inference-result-affecting config for inference_config_fingerprint(). We DENYLIST the inference
# keys that don't change per-cadence results (I/O paths, batching, retry knobs, and viz/debug
# toggles) rather than allowlisting the ones that do — a denylist can only over-invalidate (force a
# harmless re-inference), never under-invalidate (the actual stale-reuse footgun). The data section
# is the reverse: it is mostly training params, so we ALLOWLIST only the geometry fields that drive
# stamp extraction and the encoder input.
_INFERENCE_FINGERPRINT_EXCLUDE_INFERENCE_KEYS = frozenset(
    {
        "config_path",  # source-config file path, not a result input
        "per_replica_batch_size",  # batching only; #120's pad+truncate makes results batch-invariant
        "coarse_channel_log_interval",  # progress-logging chunk size only (inert)
        "bandpass_debug_plot",  # opt-in debug artifact
        "preprocess_output_dir",  # where stamps are written (folded into npy_path, not values)
        "inference_viz_enabled",  # viz toggle
        "inference_viz_scope",  # viz only (#301)
        "stamp_gallery_top_k",  # viz only
        "max_candidate_plots",  # viz only
        "max_retries",  # retry loop only
        "retry_delay",  # retry loop only
        "prefetch_depth",  # scheduling only; per-cadence results are depth-invariant (#298 N2)
        # Stamp-cache retention only (#302): deletes already-scored artifacts, never
        # changes what a cadence scores. MUST stay excluded — as a new inference key it
        # would otherwise enter BOTH fingerprints, staling every 'inferred' resume row
        # and renaming every ED cache directory on upgrade.
        "prune_stamps",
        # Report-time only (#395): filters the run tallies and Slack candidate uploads,
        # never what a cadence scores or which rows land in inference_results. Same
        # MUST-stay-excluded reasoning as prune_stamps above.
        "report_exclude_frequency_ranges",
    }
)
_INFERENCE_FINGERPRINT_DATA_KEYS = frozenset(
    {"downsample_factor", "width_bin", "num_observations", "time_bins"}
)
# Scoring/model keys excluded ON TOP of the inference denylist for the PREPROCESSING
# fingerprint (#298 I3): energy detection is deterministic given (csv files, h5 files, ED
# config), so a changed encoder/RF/threshold must REUSE stamps — that is the whole point of
# the fingerprint-scoped stamp cache. Everything else in the inference section (cadence
# grouping columns, channelization, bandpass method/taps, detection windows/threshold, stamp
# geometry, overlap, the downsample toggle) stays IN the hash. This too is a DENYLIST, never
# an ED-key allowlist: an allowlist that missed a key (say coarse_channel_width) would
# silently REUSE WRONG STAMPS when that key changes; a denylist can only over-invalidate.
_PREPROCESSING_FINGERPRINT_EXTRA_EXCLUDE_KEYS = frozenset(
    {
        "encoder_path",
        "rf_path",
        "classification_threshold",
        "screening_threshold",
        "mc_draws",
        "reference_cloud_size",
    }
)


def preprocessing_config_fingerprint(config_dict: dict) -> str:
    """
    Stable hash of the PREPROCESSING-result-affecting config — everything that changes what
    energy detection writes into a stamp .npy (#298 I3). Keys the default stamp cache
    directory ({data_path}/inference/preprocessed/<csv_stem>_ed<hash12>/), is persisted into
    each cadence's metadata JSON as ed_config_fingerprint, and is verified by the resume
    guard — so runs sharing an ED config share stamps, and any ED-config change lands in a
    different directory by construction.
    """
    inference = config_dict.get("inference") or {}
    data = config_dict.get("data") or {}
    excluded = (
        _INFERENCE_FINGERPRINT_EXCLUDE_INFERENCE_KEYS
        | _PREPROCESSING_FINGERPRINT_EXTRA_EXCLUDE_KEYS
    )
    relevant = {
        "inference": {k: v for k, v in inference.items() if k not in excluded},
        "data": {k: data[k] for k in _INFERENCE_FINGERPRINT_DATA_KEYS if k in data},
    }
    canonical = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def inference_config_fingerprint(config_dict: dict) -> str:
    """
    Stable hash of the inference-result-affecting config (from Config.to_dict()) — the inference
    counterpart of config_fingerprint. Used to detect that inference was re-run under a reused
    --save-tag with a changed config so a stale 'inferred' manifest row is not silently reused as
    a skip. Covers the inference section minus non-result knobs (see the denylist above) plus the
    data-section geometry that drives stamp extraction and the encoder input.
    """
    inference = config_dict.get("inference") or {}
    data = config_dict.get("data") or {}
    relevant = {
        "inference": {
            k: v
            for k, v in inference.items()
            if k not in _INFERENCE_FINGERPRINT_EXCLUDE_INFERENCE_KEYS
        },
        "data": {k: data[k] for k in _INFERENCE_FINGERPRINT_DATA_KEYS if k in data},
    }
    canonical = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class TrainingRunState:
    """Persisted state of one training run (identified by its save tag) across attempts."""

    tag: str
    run_start_time: float  # Wall clock of attempt 1 — used by ALL DB queries/plots
    attempt: int = 1  # Incremented per retry (each TrainingPipeline rebuild)
    config_fingerprint: str = ""  # Hash of the result-affecting config (config_fingerprint())
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
        """Record a non-critical stage failure; not in stages_done, so relaunches retry it."""
        if stage not in self.stages_failed:
            self.stages_failed.append(stage)

    def clear_stage_failure(self, stage: str) -> None:
        """Drop a recorded failure without marking the stage done — used when the user opts
        out of an optional stage (e.g. re-running without --hf-upload) so a stale failure
        can't force a nonzero exit forever."""
        if stage in self.stages_failed:
            self.stages_failed.remove(stage)

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
            config_fingerprint=str(payload.get("config_fingerprint", "")),
            completed_rounds=sorted(int(r) for r in payload.get("completed_rounds", [])),
            stages_done=[str(s) for s in payload.get("stages_done", [])],
            stages_failed=[str(s) for s in payload.get("stages_failed", [])],
        )
    except KeyError as e:
        logger.warning(
            f"Run state at {path} is missing required field {e} — treating as a fresh run"
        )
        return None
    except Exception as e:
        logger.warning(f"Failed to load run state from {path}: {e} — treating as a fresh run")
        return None
