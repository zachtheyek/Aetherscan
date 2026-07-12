"""Unit tests for aetherscan.hf_hub: revision selection/resolution, inference artifact
resolution, upload staging/arg shapes, and model card content. huggingface_hub is never
contacted — the module's _hf_api()/_hf_hub_download() seams are monkeypatched."""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pytest

from aetherscan import hf_hub
from aetherscan.config import get_config
from aetherscan.hf_hub import (
    HF_CARD_FILENAME,
    HF_CONFIG_FILENAME,
    HF_DECODER_FILENAME,
    HF_ENCODER_FILENAME,
    HF_RF_FILENAME,
    compute_rf_metrics,
    generate_model_card,
    resolve_hf_revision,
    resolve_inference_artifacts,
    select_default_revision,
    upload_run_to_hf,
)


class _FakeHfApi:
    """Records every HfApi call. upload_folder snapshots the staging directory's contents
    (the TemporaryDirectory is deleted right after the call, so assertions need a copy)."""

    def __init__(self, fail_on=()):
        self.calls = []
        self.uploaded_files: dict[str, str] = {}
        self._fail_on = set(fail_on)

    def _record(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if name in self._fail_on:
            raise RuntimeError(f"{name} failed")

    def create_repo(self, repo_id, **kwargs):
        self._record("create_repo", repo_id=repo_id, **kwargs)

    def upload_folder(self, *, repo_id, folder_path, commit_message):
        for name in sorted(os.listdir(folder_path)):
            with open(os.path.join(folder_path, name), errors="replace") as f:
                self.uploaded_files[name] = f.read()
        self._record(
            "upload_folder",
            repo_id=repo_id,
            folder_path=folder_path,
            commit_message=commit_message,
        )

    def create_tag(self, repo_id, *, tag):
        self._record("create_tag", repo_id=repo_id, tag=tag)

    def delete_tag(self, repo_id, *, tag):
        self._record("delete_tag", repo_id=repo_id, tag=tag)

    def list_repo_refs(self, repo_id, repo_type=None):
        self._record("list_repo_refs", repo_id=repo_id, repo_type=repo_type)
        raise AssertionError("list_repo_refs must be stubbed per-test when needed")


@pytest.fixture
def fake_api(monkeypatch):
    api = _FakeHfApi()
    monkeypatch.setattr(hf_hub, "_hf_api", lambda: api)
    return api


def _write_run_artifacts(config, tag, with_eval_artifacts=True):
    """Create the on-disk artifacts final_save leaves behind for `tag`."""
    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.output_path, exist_ok=True)
    for name in (f"vae_encoder_{tag}.keras", f"vae_decoder_{tag}.keras"):
        with open(os.path.join(config.model_path, name), "w") as f:
            f.write("stub-keras")
    with open(os.path.join(config.model_path, f"random_forest_{tag}.joblib"), "w") as f:
        f.write("stub-joblib")
    with open(os.path.join(config.output_path, f"config_{tag}.json"), "w") as f:
        json.dump(config.to_dict(), f)
    if with_eval_artifacts:
        joblib.dump(
            {
                "val_binary_labels": np.array([0, 0, 1, 1]),
                "val_probas": np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32),
                "classification_threshold": 0.99,
            },
            os.path.join(config.model_path, f"rf_eval_artifacts_{tag}.joblib"),
        )


class TestSelectDefaultRevision:
    def test_semver_outranks_final(self):
        assert select_default_revision(["final_v9", "v1.2.3", "v0.1.0"]) == "v1.2.3"

    def test_semver_sorted_numerically_not_lexicographically(self):
        assert select_default_revision(["v1.9.9", "v1.10.0"]) == "v1.10.0"
        assert select_default_revision(["v2.0.0", "v10.0.0", "v9.9.9"]) == "v10.0.0"

    def test_final_tags_sorted_numerically(self):
        assert select_default_revision(["final_v2", "final_v12", "final_v3"]) == "final_v12"

    def test_other_tag_families_never_win(self):
        assert select_default_revision(["test_v17", "20260712_010203", "round_02"]) is None

    def test_partial_semver_is_not_a_release_tag(self):
        assert select_default_revision(["v1.2", "v1", "final_v1"]) == "final_v1"

    def test_empty(self):
        assert select_default_revision([]) is None


class TestResolveHfRevision:
    def test_explicit_revision_returned_without_listing(self, monkeypatch):
        def boom(repo_id):
            raise AssertionError("must not list tags for an explicit revision")

        monkeypatch.setattr(hf_hub, "list_hf_tags", boom)
        assert resolve_hf_revision("ns/repo", "test_v17") == "test_v17"

    def test_latest_release_selected(self, monkeypatch):
        monkeypatch.setattr(hf_hub, "list_hf_tags", lambda repo_id: ["final_v1", "v0.2.0"])
        assert resolve_hf_revision("ns/repo", None) == "v0.2.0"

    def test_no_resolvable_tag_raises_with_guidance(self, monkeypatch):
        monkeypatch.setattr(hf_hub, "list_hf_tags", lambda repo_id: ["test_v1"])
        with pytest.raises(RuntimeError, match="--hf-revision"):
            resolve_hf_revision("ns/repo", None)


class TestListHfTags:
    def test_names_extracted_from_refs(self, monkeypatch, fake_api):
        class Ref:
            def __init__(self, name):
                self.name = name

        class Refs:
            tags = [Ref("v0.1.0"), Ref("final_v2")]

        fake_api.list_repo_refs = lambda repo_id, repo_type=None: Refs()
        assert hf_hub.list_hf_tags("ns/repo") == ["v0.1.0", "final_v2"]

    def test_missing_repo_yields_empty(self, monkeypatch, fake_api):
        import httpx  # noqa: PLC0415
        from huggingface_hub.errors import RepositoryNotFoundError  # noqa: PLC0415

        response = httpx.Response(
            404, request=httpx.Request("GET", "https://huggingface.co/api/models/ns/repo")
        )

        def raise_missing(repo_id, repo_type=None):
            raise RepositoryNotFoundError("nope", response=response)

        fake_api.list_repo_refs = raise_missing
        assert hf_hub.list_hf_tags("ns/repo") == []


class TestResolveInferenceArtifacts:
    def _args(self, **kwargs):
        defaults = {
            "command": "inference",
            "encoder_path": None,
            "rf_path": None,
            "config_path": None,
            "hf_repo_id": None,
            "hf_revision": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_all_local_paths_are_a_noop(self, monkeypatch):
        monkeypatch.setattr(
            hf_hub,
            "download_inference_artifacts",
            lambda *a, **k: pytest.fail("must not download"),
        )
        args = self._args(encoder_path="e.keras", rf_path="r.joblib", config_path="c.json")
        resolve_inference_artifacts(args)
        assert (args.encoder_path, args.rf_path, args.config_path) == (
            "e.keras",
            "r.joblib",
            "c.json",
        )

    def test_local_paths_outrank_hf_revision(self, monkeypatch):
        monkeypatch.setattr(
            hf_hub,
            "download_inference_artifacts",
            lambda *a, **k: pytest.fail("must not download"),
        )
        args = self._args(
            encoder_path="e.keras",
            rf_path="r.joblib",
            config_path="c.json",
            hf_revision="v0.1.0",
        )
        resolve_inference_artifacts(args)
        assert args.encoder_path == "e.keras"

    def test_partial_trio_left_untouched_for_validation(self, monkeypatch):
        monkeypatch.setattr(
            hf_hub,
            "download_inference_artifacts",
            lambda *a, **k: pytest.fail("must not download"),
        )
        args = self._args(encoder_path="e.keras")
        resolve_inference_artifacts(args)
        assert args.encoder_path == "e.keras"
        assert args.rf_path is None
        assert args.config_path is None

    def test_no_paths_downloads_pinned_revision(self, monkeypatch):
        downloads = []

        def fake_download(repo_id, filename, revision):
            downloads.append((repo_id, filename, revision))
            return f"/cache/{filename}"

        monkeypatch.setattr(
            hf_hub,
            "_hf_hub_download",
            lambda **kw: fake_download(kw["repo_id"], kw["filename"], kw["revision"]),
        )
        args = self._args(hf_revision="test_v17")
        resolve_inference_artifacts(args)
        # Revision-pinned, correct repo (config default), correct filenames.
        repo_id = get_config().hf.repo_id
        assert downloads == [
            (repo_id, HF_ENCODER_FILENAME, "test_v17"),
            (repo_id, HF_RF_FILENAME, "test_v17"),
            (repo_id, HF_CONFIG_FILENAME, "test_v17"),
        ]
        assert args.encoder_path == f"/cache/{HF_ENCODER_FILENAME}"
        assert args.rf_path == f"/cache/{HF_RF_FILENAME}"
        assert args.config_path == f"/cache/{HF_CONFIG_FILENAME}"

    def test_no_paths_no_revision_resolves_latest_and_records_provenance(self, monkeypatch):
        monkeypatch.setattr(hf_hub, "list_hf_tags", lambda repo_id: ["v0.1.0", "final_v3"])
        monkeypatch.setattr(hf_hub, "_hf_hub_download", lambda **kw: f"/cache/{kw['filename']}")
        args = self._args()
        resolve_inference_artifacts(args)
        # The resolved revision is written back to args so it lands in config.hf.revision
        # (and the saved inference config) via apply_args_to_config.
        assert args.hf_revision == "v0.1.0"

    def test_repo_id_flag_overrides_config_default(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            hf_hub,
            "_hf_hub_download",
            lambda **kw: (seen.append(kw["repo_id"]), f"/cache/{kw['filename']}")[1],
        )
        args = self._args(hf_repo_id="other/repo", hf_revision="final_v1")
        resolve_inference_artifacts(args)
        assert set(seen) == {"other/repo"}


class TestComputeRfMetrics:
    def test_metrics_from_eval_artifacts(self):
        config = get_config()
        _write_run_artifacts(config, "test_v1")
        metrics = compute_rf_metrics(config.model_path, "test_v1")
        # val_probas orders one positive below one negative -> AUC 0.75 for these arrays.
        assert metrics["val_roc_auc"] == pytest.approx(0.75)
        assert 0.0 <= metrics["val_average_precision"] <= 1.0
        assert metrics["classification_threshold"] == pytest.approx(0.99)
        assert metrics["n_val"] == 4

    def test_missing_artifact_returns_none(self):
        assert compute_rf_metrics(get_config().model_path, "test_v9") is None

    def test_corrupt_artifact_returns_none(self):
        config = get_config()
        os.makedirs(config.model_path, exist_ok=True)
        with open(os.path.join(config.model_path, "rf_eval_artifacts_test_v2.joblib"), "w") as f:
            f.write("not a joblib")
        assert compute_rf_metrics(config.model_path, "test_v2") is None


class TestGenerateModelCard:
    def _card(self, metrics=None):
        return generate_model_card(
            tag="test_v17",
            repo_id="zachtheyek/aetherscan",
            config_dict=get_config().to_dict(),
            metrics=metrics,
            versions={"python": "3.10.0", "tensorflow": "2.17.1"},
        )

    def test_contains_tag_config_and_links(self):
        card = self._card()
        assert "**Training tag**: `test_v17`" in card
        config = get_config()
        assert f"`{config.training.num_training_rounds}`" in card
        assert f"`{config.training.curriculum_schedule}`" in card
        assert hf_hub.GITHUB_URL in card
        for filename in (
            HF_ENCODER_FILENAME,
            HF_DECODER_FILENAME,
            HF_RF_FILENAME,
            HF_CONFIG_FILENAME,
        ):
            assert filename in card
        assert "3.10.0" in card and "2.17.1" in card
        # Frontmatter block for the Hub card metadata.
        assert card.startswith("---\n")

    def test_metrics_section_present_when_available(self):
        card = self._card(
            metrics={
                "val_roc_auc": 0.9876,
                "val_average_precision": 0.8765,
                "classification_threshold": 0.99,
                "n_val": 128,
            }
        )
        assert "0.9876" in card
        assert "0.8765" in card
        assert "| Validation samples | 128 |" in card

    def test_metrics_section_omitted_when_unavailable(self):
        card = self._card(metrics=None)
        assert "ROC AUC" not in card
        assert "No evaluation artifacts" in card


class TestUploadRunToHf:
    def test_stages_stable_names_and_tags_commit(self, fake_api):
        config = get_config()
        _write_run_artifacts(config, "test_v1")
        upload_run_to_hf(
            repo_id="ns/repo",
            tag="test_v1",
            model_path=config.model_path,
            output_path=config.output_path,
        )

        assert sorted(fake_api.uploaded_files) == sorted(
            [
                HF_ENCODER_FILENAME,
                HF_DECODER_FILENAME,
                HF_RF_FILENAME,
                HF_CONFIG_FILENAME,
                HF_CARD_FILENAME,
            ]
        )
        assert "test_v1" in fake_api.uploaded_files[HF_CARD_FILENAME]

        names = [name for name, _ in fake_api.calls]
        assert names == ["create_repo", "upload_folder", "create_tag"]
        create_repo = dict(fake_api.calls[0][1])
        assert create_repo == {
            "repo_id": "ns/repo",
            "repo_type": "model",
            "private": False,
            "exist_ok": True,
        }
        upload = dict(fake_api.calls[1][1])
        assert upload["repo_id"] == "ns/repo"
        assert upload["commit_message"] == "test_v1"  # commit message = save_tag
        assert dict(fake_api.calls[2][1]) == {"repo_id": "ns/repo", "tag": "test_v1"}

    def test_force_moves_existing_tag(self, fake_api):
        config = get_config()
        _write_run_artifacts(config, "test_v1")
        upload_run_to_hf(
            repo_id="ns/repo",
            tag="test_v1",
            model_path=config.model_path,
            output_path=config.output_path,
            force=True,
        )
        names = [name for name, _ in fake_api.calls]
        assert names == ["create_repo", "upload_folder", "delete_tag", "create_tag"]

    def test_missing_artifact_raises_before_any_api_call(self, fake_api):
        config = get_config()
        _write_run_artifacts(config, "test_v1")
        os.remove(os.path.join(config.model_path, "vae_decoder_test_v1.keras"))
        with pytest.raises(FileNotFoundError, match="vae_decoder_test_v1.keras"):
            upload_run_to_hf(
                repo_id="ns/repo",
                tag="test_v1",
                model_path=config.model_path,
                output_path=config.output_path,
            )
        assert fake_api.calls == []

    def test_upload_failure_propagates_to_caller(self, monkeypatch):
        config = get_config()
        _write_run_artifacts(config, "test_v1")
        api = _FakeHfApi(fail_on={"upload_folder"})
        monkeypatch.setattr(hf_hub, "_hf_api", lambda: api)
        with pytest.raises(RuntimeError, match="upload_folder failed"):
            upload_run_to_hf(
                repo_id="ns/repo",
                tag="test_v1",
                model_path=config.model_path,
                output_path=config.output_path,
            )
