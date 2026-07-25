"""
Fail-early save-tag dedup guards.

Reusing a save_tag silently mixes a new run's artifacts, config JSON, and DB rows with a
previous run's — the stale-artifact confusion the cluster runbook used to work around by
manually incrementing test_vNN tags. These guards hard-stop a run at startup (before any
expensive work or writes) when its resolved save-tag collides with a completed run's state,
while staying out of the way of the two legitimate same-tag flows:

- Training retries/relaunches: a run-state manifest (run_state_{tag}.json) marks the tag as
  an in-progress resumable run, so the guard is skipped entirely (PR-04's supersede
  semantics are what make same-tag retries safe).
- Inference resume: manifest rows in the inference_cadences DB table mark an in-progress
  streaming run. Only a completed run — evidenced by its saved config_{tag}.json, written at
  the very end of a successful pass — or stale legacy-path DB rows count as collisions.

Because every resolved save-tag carries a fresh second-resolution {command}_{datetime} stamp,
a fresh run can't collide by construction; the guards bite only if a completed run's tag is
deliberately reused, and the resumable-run manifests above exempt the legitimate retry/resume
flows. --force-tag consciously overrides every guard.

enforce_tag_guards() is called from main() immediately before command dispatch
(post-validation, post-DB-init, pre-any-work).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from aetherscan.config import get_config
from aetherscan.db import get_db
from aetherscan.run_state import STAGE_FINAL_SAVE, load_run_state, run_state_path

logger = logging.getLogger(__name__)


def find_train_tag_collisions(tag: str) -> list[str]:
    """
    Human-readable descriptions of existing training state that would collide with `tag`:
    the final encoder artifact under model_path, the saved config JSON under output_path,
    and non-superseded training_stats DB rows. Empty list = no collisions.
    """
    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")
    db = get_db()
    collisions: list[str] = []

    encoder_path = os.path.join(config.model_path, f"vae_encoder_{tag}.keras")
    if os.path.exists(encoder_path):
        collisions.append(f"model artifact exists: {encoder_path}")

    config_path = os.path.join(config.output_path, f"config_{tag}.json")
    if os.path.exists(config_path):
        collisions.append(f"saved run config exists: {config_path}")

    if db is not None:
        rows = db.query_training_stat(tag=tag, columns=["tag"])
        if rows:
            collisions.append(
                f"{len(rows)} non-superseded training_stats DB row(s) carry tag '{tag}'"
            )

    return collisions


def find_inference_tag_collisions(tag: str) -> list[str]:
    """
    Collisions scoped to what inference writes. A saved config_{tag}.json marks a *completed*
    run under this tag (it is written at the very end of a successful pass) — reusing its tag
    is always a collision. DB rows alone are only a collision on the legacy --test-files path
    (no inference_cadences manifest rows): with manifest rows present, same-tag DB state is
    exactly what the streaming path's resume flow consumes, so it must not be flagged.
    """
    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")
    db = get_db()
    collisions: list[str] = []

    config_path = os.path.join(config.output_path, f"config_{tag}.json")
    if os.path.exists(config_path):
        collisions.append(f"saved run config exists (completed run marker): {config_path}")

    if db is not None and not db.query_inference_cadences(tag=tag, columns=["tag"]):
        rows = db.query_inference_result(tag=tag, columns=["tag"])
        if rows:
            collisions.append(
                f"{len(rows)} non-superseded inference_results DB row(s) carry tag '{tag}' "
                f"(no resumable run manifest)"
            )

    return collisions


def _exit_on_collisions(tag: str, collisions: list[str], resume_hint: str) -> None:
    """Log the collision list plus the user's options, then hard-exit."""
    logger.error("=" * 60)
    logger.error(f"--save-tag '{tag}' collides with existing state from a previous run:")
    for collision in collisions:
        logger.error(f"  - {collision}")
    logger.error("Options:")
    logger.error("  - pick a new --save-tag")
    logger.error(f"  - {resume_hint}")
    logger.error("  - pass --force-tag to consciously override this guard")
    logger.error("=" * 60)
    sys.exit(1)


def _guard_hf_tag(tag: str, force: bool) -> None:
    """
    HF-side dedup guard, run at startup when --hf-upload is enabled so a tag collision on the
    Hub surfaces now, not after ~30 h of training. A failed check (network/auth) only warns:
    the upload stage itself is non-critical, so its guard must not block training either.
    """
    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")
    repo_id = config.hf.repo_id
    from aetherscan.hf_hub import hf_tag_exists  # noqa: PLC0415

    try:
        exists = hf_tag_exists(repo_id, tag)
    except Exception as e:
        logger.warning(f"Could not check HF repo '{repo_id}' for tag collisions ({e}) — proceeding")
        return
    if not exists:
        return
    if force:
        logger.warning(
            f"Tag '{tag}' already exists on HF repo '{repo_id}' — --force-tag set, the "
            f"upload stage will move it to the new commit"
        )
        return
    logger.error("=" * 60)
    logger.error(f"--hf-upload is enabled but tag '{tag}' already exists on HF repo '{repo_id}'.")
    logger.error("Options: pick a new --save-tag, or pass --force-tag to move the HF tag.")
    logger.error("=" * 60)
    sys.exit(1)


def enforce_tag_guards(args: argparse.Namespace) -> None:
    """
    Fail-early save-tag dedup guards, dispatched by mode (see module docstring). Reads the
    post-apply config singleton for the effective tag/force values and `args` only to learn
    whether --save-tag was explicitly provided (default datetime tags are immune by
    construction, so the local guards are skipped for them).
    """
    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")
    tag = config.checkpoint.save_tag
    force = bool(config.checkpoint.force_tag)
    explicit = getattr(args, "save_tag", None) is not None
    command = getattr(args, "command", None)

    if command == "train":
        if explicit:
            manifest_path = run_state_path(config.output_path, tag)
            # Only an UNFINISHED run's manifest exempts the collision guard. The manifest
            # persists on success, so a completed run's manifest must NOT keep disabling dedup
            # for that tag — otherwise a reused --save-tag would silently overwrite a finished
            # model with no warning (TG-1). "Unfinished" = final_save not yet done, or a
            # recorded non-critical stage failure still pending retry.
            state = load_run_state(manifest_path)
            resumable = state is not None and (
                not state.is_stage_done(STAGE_FINAL_SAVE) or bool(state.stages_failed)
            )
            if resumable:
                logger.info(
                    f"Run-state manifest at {manifest_path} marks an unfinished run — tag "
                    f"'{tag}' is resumable, skipping the collision guard"
                )
            else:
                collisions = find_train_tag_collisions(tag)
                if collisions and force:
                    logger.warning(
                        f"--force-tag set: proceeding despite {len(collisions)} "
                        f"collision(s) on tag '{tag}'"
                    )
                elif collisions:
                    _exit_on_collisions(
                        tag,
                        collisions,
                        "to resume the previous run, re-run its identical command "
                        "(resume requires its run-state manifest)",
                    )
        if config.hf.upload_after_training:
            _guard_hf_tag(tag, force)

    elif command == "inference" and explicit:
        collisions = find_inference_tag_collisions(tag)
        if collisions and force:
            logger.warning(
                f"--force-tag set: proceeding despite {len(collisions)} collision(s) on tag '{tag}'"
            )
        elif collisions:
            _exit_on_collisions(
                tag,
                collisions,
                "an in-progress run (manifest rows, no saved config JSON) resumes by "
                "re-running its identical command",
            )
