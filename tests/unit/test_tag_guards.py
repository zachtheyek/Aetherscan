"""Unit tests for aetherscan.tag_guards: the fail-early save-tag dedup guard matrix (local
train/inference collisions, resumable-run exemptions, --force-tag override, HF-side check)."""

from __future__ import annotations

import argparse
import json
import os

import pytest

from aetherscan import hf_hub, tag_guards
from aetherscan.config import get_config
from aetherscan.run_state import run_state_path
from aetherscan.tag_guards import (
    enforce_tag_guards,
    find_inference_tag_collisions,
    find_train_tag_collisions,
)

TAG = "test_v5"


class _FakeDb:
    """Stands in for the Database singleton: query methods return canned row lists."""

    def __init__(self, training_rows=0, inference_rows=0, manifest_rows=0):
        self._training = [{"tag": TAG}] * training_rows
        self._inference = [{"tag": TAG}] * inference_rows
        self._manifest = [{"tag": TAG}] * manifest_rows

    def query_training_stat(self, tag=None, columns=None, **kwargs):
        return list(self._training)

    def query_inference_result(self, tag=None, columns=None, **kwargs):
        return list(self._inference)

    def query_inference_cadences(self, tag=None, columns=None, **kwargs):
        return list(self._manifest)


@pytest.fixture
def fake_db(monkeypatch):
    db = _FakeDb()
    monkeypatch.setattr(tag_guards, "get_db", lambda: db)
    return db


def _args(command, save_tag=TAG, **kwargs):
    return argparse.Namespace(command=command, save_tag=save_tag, **kwargs)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("stub")


def _set_tag(tag=TAG, force=False):
    config = get_config()
    config.checkpoint.save_tag = tag
    config.checkpoint.force_tag = force
    return config


class TestFindTrainTagCollisions:
    def test_clean_slate_has_no_collisions(self, fake_db):
        assert find_train_tag_collisions(TAG) == []

    def test_encoder_artifact_collides(self, fake_db):
        config = get_config()
        _touch(os.path.join(config.model_path, f"vae_encoder_{TAG}.keras"))
        collisions = find_train_tag_collisions(TAG)
        assert len(collisions) == 1
        assert "vae_encoder" in collisions[0]

    def test_config_json_collides(self, fake_db):
        config = get_config()
        _touch(os.path.join(config.output_path, f"config_{TAG}.json"))
        assert any("config_" in c for c in find_train_tag_collisions(TAG))

    def test_db_rows_collide(self, fake_db):
        fake_db._training = [{"tag": TAG}] * 3
        collisions = find_train_tag_collisions(TAG)
        assert any("3 non-superseded training_stats" in c for c in collisions)

    def test_all_three_reported_together(self, fake_db):
        config = get_config()
        _touch(os.path.join(config.model_path, f"vae_encoder_{TAG}.keras"))
        _touch(os.path.join(config.output_path, f"config_{TAG}.json"))
        fake_db._training = [{"tag": TAG}]
        assert len(find_train_tag_collisions(TAG)) == 3


class TestFindInferenceTagCollisions:
    def test_clean_slate_has_no_collisions(self, fake_db):
        assert find_inference_tag_collisions(TAG) == []

    def test_completed_run_config_json_collides(self, fake_db):
        config = get_config()
        _touch(os.path.join(config.output_path, f"config_{TAG}.json"))
        assert any("completed run marker" in c for c in find_inference_tag_collisions(TAG))

    def test_legacy_db_rows_without_manifest_collide(self, fake_db):
        fake_db._inference = [{"tag": TAG}] * 2
        collisions = find_inference_tag_collisions(TAG)
        assert any("inference_results" in c for c in collisions)

    def test_manifest_rows_mark_resumable_run_not_collision(self, fake_db):
        # An in-progress streaming run: manifest rows exist, no completed-run config JSON.
        # Same-tag DB state is exactly what the resume flow consumes — never a collision.
        fake_db._manifest = [{"tag": TAG}]
        fake_db._inference = [{"tag": TAG}] * 10
        assert find_inference_tag_collisions(TAG) == []


class TestEnforceTagGuardsTrain:
    def test_default_datetime_tag_skips_guard(self, fake_db):
        # args.save_tag is None -> tag was not explicitly provided -> immune by construction
        config = _set_tag()
        _touch(os.path.join(config.model_path, f"vae_encoder_{TAG}.keras"))
        enforce_tag_guards(_args("train", save_tag=None))  # must not exit

    def test_explicit_tag_with_collision_exits(self, fake_db):
        config = _set_tag()
        _touch(os.path.join(config.model_path, f"vae_encoder_{TAG}.keras"))
        with pytest.raises(SystemExit) as excinfo:
            enforce_tag_guards(_args("train"))
        assert excinfo.value.code == 1

    def test_explicit_tag_clean_slate_passes(self, fake_db):
        _set_tag()
        enforce_tag_guards(_args("train"))

    def test_resumable_manifest_exempts_same_tag_retry(self, fake_db):
        config = _set_tag()
        _touch(os.path.join(config.model_path, f"vae_encoder_{TAG}.keras"))
        with open(run_state_path(config.output_path, TAG), "w") as f:
            json.dump({"tag": TAG, "run_start_time": 1.0}, f)
        enforce_tag_guards(_args("train"))  # must not exit

    def test_force_tag_overrides_collision(self, fake_db):
        config = _set_tag(force=True)
        _touch(os.path.join(config.model_path, f"vae_encoder_{TAG}.keras"))
        enforce_tag_guards(_args("train"))  # must not exit


class TestEnforceTagGuardsHf:
    def test_hf_collision_exits_when_upload_enabled(self, fake_db, monkeypatch):
        config = _set_tag()
        config.hf.upload_after_training = True
        monkeypatch.setattr(hf_hub, "hf_tag_exists", lambda repo_id, tag: True)
        with pytest.raises(SystemExit):
            enforce_tag_guards(_args("train"))

    def test_hf_check_runs_even_for_default_tags(self, fake_db, monkeypatch):
        config = _set_tag()
        config.hf.upload_after_training = True
        monkeypatch.setattr(hf_hub, "hf_tag_exists", lambda repo_id, tag: True)
        with pytest.raises(SystemExit):
            enforce_tag_guards(_args("train", save_tag=None))

    def test_hf_clean_tag_passes(self, fake_db, monkeypatch):
        config = _set_tag()
        config.hf.upload_after_training = True
        monkeypatch.setattr(hf_hub, "hf_tag_exists", lambda repo_id, tag: False)
        enforce_tag_guards(_args("train"))

    def test_hf_check_failure_warns_but_does_not_block(self, fake_db, monkeypatch):
        config = _set_tag()
        config.hf.upload_after_training = True

        def boom(repo_id, tag):
            raise ConnectionError("no network")

        monkeypatch.setattr(hf_hub, "hf_tag_exists", boom)
        enforce_tag_guards(_args("train"))  # must not exit

    def test_hf_collision_with_force_passes(self, fake_db, monkeypatch):
        config = _set_tag(force=True)
        config.hf.upload_after_training = True
        monkeypatch.setattr(hf_hub, "hf_tag_exists", lambda repo_id, tag: True)
        enforce_tag_guards(_args("train"))

    def test_hf_check_skipped_when_upload_disabled(self, fake_db, monkeypatch):
        _set_tag()

        def boom(repo_id, tag):
            raise AssertionError("HF must not be contacted when --hf-upload is off")

        monkeypatch.setattr(hf_hub, "hf_tag_exists", boom)
        enforce_tag_guards(_args("train"))


class TestEnforceTagGuardsInference:
    def test_completed_run_collision_exits(self, fake_db):
        config = _set_tag()
        _touch(os.path.join(config.output_path, f"config_{TAG}.json"))
        with pytest.raises(SystemExit):
            enforce_tag_guards(_args("inference"))

    def test_resumable_manifest_rows_pass(self, fake_db):
        _set_tag()
        fake_db._manifest = [{"tag": TAG}]
        fake_db._inference = [{"tag": TAG}] * 4
        enforce_tag_guards(_args("inference"))

    def test_default_tag_skips_guard(self, fake_db):
        config = _set_tag()
        _touch(os.path.join(config.output_path, f"config_{TAG}.json"))
        enforce_tag_guards(_args("inference", save_tag=None))

    def test_force_tag_overrides(self, fake_db):
        config = _set_tag(force=True)
        _touch(os.path.join(config.output_path, f"config_{TAG}.json"))
        enforce_tag_guards(_args("inference"))
