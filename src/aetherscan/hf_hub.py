"""
HuggingFace Hub integration for Aetherscan model artifacts.

One public model repo (config.hf.repo_id, default zachtheyek/aetherscan) carries the released
weights at stable filenames in the repo root — vae_encoder.keras, vae_decoder.keras,
random_forest.joblib, config.json, plus an auto-generated model card README.md — with HF git
tags carrying the versioning. Two tag families exist on the Hub: training tags (= a run's
save_tag, e.g. train_20260101_120000, created by upload_run_to_hf after training) and release
tags (vX.Y.Z, added by the release runbook pointing at the blessed weights commit).

Upload (train, opt-in via --hf-upload) stages the four run artifacts under the stable names,
commits them with the save_tag as the commit message, and tags the commit. Download
(inference, the default when no local artifact paths are given) resolves a revision —
explicit --hf-revision, else v{__version__} when running as an installed release (see
version_default_revision — this is what makes `pip install aetherscan==1.0.0` + bare
inference pull exactly the v1.0.0 weights), else the highest semver vX.Y.Z release tag — and
pulls the artifact trio revision-pinned via hf_hub_download. A no-artifact download requires a
release tag; training tags never name the default revision.

Auth: uploads require HF_TOKEN in the environment (loaded from the gitignored .env);
huggingface_hub reads it implicitly. The repo is public, so downloads need no token. The
token value is never logged.

huggingface_hub is imported lazily inside the thin _hf_api()/_hf_hub_download() seams so this
module stays importable (and unit-testable with those seams monkeypatched) without the
dependency installed — e.g. in the pre-rebuild NGC container, where only actual Hub calls
should fail.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import os
import platform
import re
import shutil
import tempfile
from typing import Any

from aetherscan.config import get_config

logger = logging.getLogger(__name__)

# Stable artifact filenames at the HF repo root (versioning lives in git tags, not names).
HF_ENCODER_FILENAME = "vae_encoder.keras"
HF_DECODER_FILENAME = "vae_decoder.keras"
HF_RF_FILENAME = "random_forest.joblib"
HF_CONFIG_FILENAME = "config.json"
HF_CARD_FILENAME = "README.md"

GITHUB_URL = "https://github.com/zachtheyek/Aetherscan"

# Release tags (vX.Y.Z, from the release runbook) name the default download revision. Training
# tags (the {command}_{datetime} run tags) never name it — a no-artifact inference download
# requires a blessed release tag.
_SEMVER_TAG_PATTERN = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _hf_api():
    """Construct an HfApi client (lazy import — see module docstring). HF_TOKEN is picked up
    from the environment implicitly for authenticated calls."""
    from huggingface_hub import HfApi  # noqa: PLC0415

    return HfApi()


def _hf_hub_download(**kwargs) -> str:
    """Thin seam over huggingface_hub.hf_hub_download (lazy import — see module docstring)."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    return hf_hub_download(**kwargs)


def _disable_hub_progress_bars() -> None:
    """Disable huggingface_hub's tqdm progress bars for this process. The pipeline redirects
    sys.stdout/sys.stderr to loggers (StreamToLogger), so interactive progress rendering is
    line-spam at best — and tty/capability probing by the progress machinery must never be a
    failure surface for uploads/downloads (hf_upload once died on a stdout isatty() probe).
    Tolerant of a missing huggingface_hub: the subsequent Hub call raises its own error."""
    try:
        from huggingface_hub.utils import disable_progress_bars  # noqa: PLC0415
    except Exception:
        return
    disable_progress_bars()


def _is_tag_conflict(exc: Exception) -> bool:
    """True when `exc` is the Hub's tag-already-exists HTTP 409 conflict."""
    try:
        from huggingface_hub.errors import HfHubHTTPError  # noqa: PLC0415
    except Exception:
        return False
    if not isinstance(exc, HfHubHTTPError):
        return False
    return getattr(getattr(exc, "response", None), "status_code", None) == 409


def list_hf_tags(repo_id: str) -> list[str]:
    """Return the names of every git tag on the HF model repo; [] when the repo doesn't exist
    yet (first upload creates it). Network/auth errors propagate to the caller."""
    from huggingface_hub.errors import RepositoryNotFoundError  # noqa: PLC0415

    try:
        refs = _hf_api().list_repo_refs(repo_id, repo_type="model")
    except RepositoryNotFoundError:
        return []
    return [ref.name for ref in refs.tags]


def hf_tag_exists(repo_id: str, tag: str) -> bool:
    """True when `tag` already exists on the HF repo (used by the fail-early dedup guard)."""
    return tag in list_hf_tags(repo_id)


def select_default_revision(tags: list[str]) -> str | None:
    """
    Pick the default download revision from a repo's tag list: the highest semver vX.Y.Z
    release tag, or None if the repo carries no release tag. Comparison is numeric
    (v1.10.0 > v1.9.9); training tags (the {command}_{datetime} run tags) never win — a
    no-artifact inference download requires a blessed release tag.
    """
    semver = [
        (tuple(int(g) for g in m.groups()), t) for t in tags if (m := _SEMVER_TAG_PATTERN.match(t))
    ]
    if semver:
        return max(semver)[1]
    return None


def version_default_revision() -> str | None:
    """
    The version-coupled default HF revision — f"v{__version__}" — or None when this run's
    version can't name a release tag. This is the runtime half of the release contract
    (docs/RELEASE.md): `pip install aetherscan==1.0.0` + inference with no model-path flags
    downloads exactly the v1.0.0-blessed weights, because the installed package version is
    the default revision.

    Guard: the default only activates when v{__version__} lands in the release-tag family
    (_SEMVER_TAG_PATTERN, strict vX.Y.Z). The importlib.metadata fallback ("0.0.0.dev0" —
    source-tree/container runs), .dev pre-releases (e.g. "0.9.0.dev0" — a pip install of a
    between-releases tree), and rc/post/local versions all fail the match and fall through
    to latest-release resolution instead. Deliberately NOT existence-checked: an installed
    release whose weights tag is missing must fail the download loudly (the release
    runbook's blessing step was skipped) rather than silently pull some other version's
    weights.
    """
    import aetherscan  # noqa: PLC0415  (late import so tests can monkeypatch __version__)

    candidate = f"v{aetherscan.__version__}"
    if not _SEMVER_TAG_PATTERN.match(candidate):
        return None
    return candidate


def resolve_hf_revision(repo_id: str, revision: str | None) -> str:
    """
    Resolve the HF revision inference should download from: an explicitly requested revision
    is returned as-is (existence is checked by the download itself); otherwise an installed
    release pins its own version's tag (version_default_revision — also download-checked);
    otherwise the repo's tags are listed and select_default_revision picks the latest
    release. Raises RuntimeError with guidance when nothing is resolvable.
    """
    if revision is not None:
        return revision
    version_pinned = version_default_revision()
    if version_pinned is not None:
        logger.info(
            f"Resolved HF revision '{version_pinned}' (pinned to the installed aetherscan "
            f"release; override with --hf-revision)"
        )
        return version_pinned
    try:
        tags = list_hf_tags(repo_id)
    except Exception as e:
        raise RuntimeError(
            f"Could not list tags on HF repo '{repo_id}' to resolve the latest release: {e}. "
            f"Check that --hf-repo-id is correct, the repo is public (or HF_TOKEN grants "
            f"access), and this host can reach huggingface.co — or pin a revision with "
            f"--hf-revision / pass all three local artifact paths."
        ) from e
    selected = select_default_revision(tags)
    if selected is None:
        raise RuntimeError(
            f"No release tag (vX.Y.Z) found on HF repo "
            f"'{repo_id}' to download model artifacts from. Either pin a revision with "
            f"--hf-revision, point --hf-repo-id at a repo with released weights, or pass "
            f"all three local artifact paths (--encoder-path/--rf-path/--config-path)."
        )
    logger.info(f"Resolved HF revision '{selected}' (latest release tag on {repo_id})")
    return selected


def download_inference_artifacts(repo_id: str, revision: str) -> tuple[str, str, str]:
    """
    Download the (encoder, random forest, config) artifact trio from the HF repo at the
    pinned revision. Returns the local cache paths (under HF_HOME / ~/.cache/huggingface;
    repeated runs hit the cache). Public repo — no token required.
    """
    _disable_hub_progress_bars()
    try:
        paths = tuple(
            _hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
            for filename in (HF_ENCODER_FILENAME, HF_RF_FILENAME, HF_CONFIG_FILENAME)
        )
    except Exception as e:
        # Wrap raw huggingface_hub errors (404s, network failures, auth) with operator
        # guidance — this surfaces via main.py's resolution path at startup.
        raise RuntimeError(
            f"Failed to download model artifacts from {repo_id}@{revision}: {e}. Check "
            f"that the revision exists (--hf-revision), the repo id is correct "
            f"(--hf-repo-id), the repo is public (or HF_TOKEN grants access), and this "
            f"host can reach huggingface.co."
        ) from e
    logger.info(f"Downloaded model artifacts from {repo_id}@{revision}")
    return paths


def resolve_inference_artifacts(args: argparse.Namespace) -> None:
    """
    Ensure the inference artifact trio (encoder/rf/config paths) is populated on `args`
    before validation runs, in resolution order: explicit local paths (highest) >
    --hf-revision > v{__version__} when running as an installed release (see
    version_default_revision) > latest release tag on the HF repo.

    When none of the three paths were given, the resolved revision's artifacts are
    downloaded and their cache paths written onto `args` — exactly as if the user had passed
    them on the CLI, so the existing validate_args / apply_saved_config / model-load path is
    reused unchanged. The resolved revision is also written to args.hf_revision so it lands
    in config.hf.revision (and thus the saved inference config) for provenance.

    A partial set of local paths is left untouched: collect_validation_errors reports the
    missing ones (mixing local and Hub-sourced artifacts would silently pair mismatched
    models). Raises RuntimeError (via resolve_hf_revision) or huggingface_hub errors when HF
    resolution fails — the caller exits with the message as guidance.
    """
    provided = [
        getattr(args, name, None) is not None for name in ("encoder_path", "rf_path", "config_path")
    ]
    if all(provided):
        if getattr(args, "hf_revision", None) is not None:
            logger.info("Explicit local artifact paths take precedence — ignoring --hf-revision")
        return
    if any(provided):
        # Partial trio: leave args as-is; validation reports the missing paths.
        return

    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")
    repo_id = getattr(args, "hf_repo_id", None) or config.hf.repo_id
    revision = resolve_hf_revision(repo_id, getattr(args, "hf_revision", None))
    logger.info(
        f"No local artifact paths given — downloading model artifacts from HF repo "
        f"'{repo_id}' at revision '{revision}'"
    )
    args.encoder_path, args.rf_path, args.config_path = download_inference_artifacts(
        repo_id, revision
    )
    args.hf_revision = revision


def _collect_library_versions() -> dict[str, str]:
    """Best-effort version stamps for the model card; import failures record 'unknown'."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for card_name, module_name in (
        ("tensorflow", "tensorflow"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("huggingface_hub", "huggingface_hub"),
    ):
        try:
            module = __import__(module_name)
            versions[card_name] = str(module.__version__)
        except Exception:
            versions[card_name] = "unknown"
    return versions


def compute_rf_metrics(model_path: str, tag: str) -> dict[str, Any] | None:
    """
    Derive validation metrics for the model card from the run's persisted RF eval artifacts
    (rf_eval_artifacts_{tag}.joblib, written by train_random_forest). Returns None when the
    artifact is missing or unreadable — the card simply omits its metrics section.
    """
    artifact_path = os.path.join(model_path, f"rf_eval_artifacts_{tag}.joblib")
    if not os.path.exists(artifact_path):
        logger.info(f"No RF eval artifacts at {artifact_path} — model card omits metrics")
        return None
    try:
        import joblib  # noqa: PLC0415
        from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: PLC0415

        artifacts = joblib.load(artifact_path)
        labels = artifacts["val_binary_labels"]
        probas = artifacts["val_probas"]
        return {
            "val_roc_auc": float(roc_auc_score(labels, probas)),
            "val_average_precision": float(average_precision_score(labels, probas)),
            "classification_threshold": float(artifacts["classification_threshold"]),
            "n_val": int(len(labels)),
        }
    except Exception as e:
        logger.warning(f"Failed to compute RF metrics from {artifact_path}: {e}")
        return None


def generate_model_card(
    *,
    tag: str,
    config_dict: dict[str, Any],
    metrics: dict[str, Any] | None,
    versions: dict[str, str],
) -> str:
    """
    Render the repo card (README.md) uploaded alongside the weights: pipeline description,
    training tag, config summary (rounds / samples / SNR schedule), validation metrics when
    available, library versions, and the GitHub link + citation pointer.
    """
    training = config_dict.get("training", {})
    beta_vae = config_dict.get("beta_vae", {})
    rf = config_dict.get("rf", {})

    def row(name: str, section: dict[str, Any], key: str) -> str:
        return f"| {name} | `{section.get(key, 'n/a')}` |"

    config_rows = "\n".join(
        [
            row("Training rounds", training, "num_training_rounds"),
            row("Epochs per round", training, "epochs_per_round"),
            row("Beta-VAE samples per round", training, "num_samples_beta_vae"),
            row("Random Forest samples", training, "num_samples_rf"),
            row("Curriculum schedule", training, "curriculum_schedule"),
            row("SNR base", training, "snr_base"),
            row("Initial SNR range", training, "initial_snr_range"),
            row("Final SNR range", training, "final_snr_range"),
            row("Latent dimensions", beta_vae, "latent_dim"),
            row("Beta (KL weight)", beta_vae, "beta"),
            row("Alpha (clustering weight)", beta_vae, "alpha"),
            row("RF estimators", rf, "n_estimators"),
        ]
    )

    if metrics is not None:
        metrics_section = (
            "## Evaluation (validation split)\n\n"
            "| Metric | Value |\n|---|---|\n"
            f"| ROC AUC | {metrics['val_roc_auc']:.4f} |\n"
            f"| Average precision | {metrics['val_average_precision']:.4f} |\n"
            f"| Classification threshold | {metrics['classification_threshold']} |\n"
            f"| Validation samples | {metrics['n_val']} |\n"
        )
    else:
        metrics_section = "## Evaluation\n\nNo evaluation artifacts were available for this run.\n"

    versions_rows = "\n".join(f"| {name} | `{ver}` |" for name, ver in versions.items())

    return f"""---
license: bsd-3-clause
library_name: keras
tags:
- seti
- radio-astronomy
- anomaly-detection
- beta-vae
- random-forest
---

# Aetherscan

[Breakthrough Listen](https://breakthroughinitiatives.org/initiative/1)'s deep-learning SETI
pipeline: a two-stage architecture where a **Beta-VAE encoder** compresses each observation of
a 6-observation cadence (3 ON / 3 OFF, ABACAD) into an 8-dimensional latent, and a **Random
Forest** classifies the cadence's concatenated latents as a technosignature candidate or not.

This repository carries the released model weights at stable filenames, versioned via git
tags: training tags match the pipeline run's save tag (e.g. `train_20260101_120000`), and
release tags (`vX.Y.Z`) mark blessed weights.

**Training tag**: `{tag}`

## Files

| File | Description |
|---|---|
| `{HF_ENCODER_FILENAME}` | Beta-VAE encoder (Keras) — the inference feature extractor |
| `{HF_DECODER_FILENAME}` | Beta-VAE decoder (Keras) — for reconstruction/traversal analysis |
| `{HF_RF_FILENAME}` | Random Forest cadence classifier (joblib) |
| `{HF_CONFIG_FILENAME}` | Full resolved training configuration for this run |

## Training configuration

| Parameter | Value |
|---|---|
{config_rows}

The complete configuration is in `{HF_CONFIG_FILENAME}`.

{metrics_section}
## Library versions

| Library | Version |
|---|---|
{versions_rows}

## Usage

Aetherscan inference downloads these weights by default when no local artifact paths are
given (pin a version with `--hf-revision`):

```bash
python -m aetherscan.main inference --hf-revision {tag} --inference-files <catalog.csv>
```

## Links & citation

Source code, documentation, and issue tracker: [{GITHUB_URL}]({GITHUB_URL}).
If you use Aetherscan in your research, please cite it via the repository's `CITATION.cff`.
"""


# Environment-specific config fields stripped before config.json is published to the PUBLIC Hub
# repo (HFSEC-1): whole sections that are pure host layout, and per-field absolute paths / real
# observation filenames. None are reproducible off-host, and they would otherwise disclose the
# operator's username, directory layout, and dataset file names on huggingface.co.
_UPLOAD_REDACT_SECTIONS = ("paths",)
_UPLOAD_REDACT_FIELDS = {
    "data": ("train_files", "test_files", "inference_files"),
    "inference": ("encoder_path", "rf_path", "config_path", "preprocess_output_dir"),
    "checkpoint": ("load_dir",),
}


def _sanitize_config_for_upload(config_dict: dict) -> dict:
    """Return a copy of the run config with host-specific fields (absolute paths, real dataset
    filenames) stripped, so the config.json published to the public Hub repo carries only
    reproducibility-relevant hyperparameters. See HFSEC-1."""
    sanitized = copy.deepcopy(config_dict)
    for section in _UPLOAD_REDACT_SECTIONS:
        sanitized.pop(section, None)
    for section, fields in _UPLOAD_REDACT_FIELDS.items():
        block = sanitized.get(section)
        if isinstance(block, dict):
            for field in fields:
                block.pop(field, None)
    return sanitized


def upload_run_to_hf(
    *, repo_id: str, tag: str, model_path: str, output_path: str, force: bool = False
) -> None:
    """
    Publish one training run's final artifacts to the HF model repo: stage the four artifacts
    under their stable names plus a generated model card, commit them (commit message = the
    run's save_tag), and tag the commit with the save_tag. Creates the (public) repo when it
    doesn't exist yet. With force=True a pre-existing identical tag is moved to the new
    commit (the startup dedup guard was consciously overridden via --force-tag).

    Raises on any failure — the caller (the hf_upload training stage) records it in the run
    manifest without failing the run.
    """
    sources = {
        HF_ENCODER_FILENAME: os.path.join(model_path, f"vae_encoder_{tag}.keras"),
        HF_DECODER_FILENAME: os.path.join(model_path, f"vae_decoder_{tag}.keras"),
        HF_RF_FILENAME: os.path.join(model_path, f"random_forest_{tag}.joblib"),
        HF_CONFIG_FILENAME: os.path.join(output_path, f"config_{tag}.json"),
    }
    missing = [path for path in sources.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Cannot upload run '{tag}' to HF: missing artifact(s): {', '.join(missing)}"
        )

    with open(sources[HF_CONFIG_FILENAME]) as f:
        config_dict = json.load(f)
    # The repo is PUBLIC, so strip environment-specific fields before publishing config.json:
    # absolute host paths (username + internal layout) and real observation filenames, none of
    # which are reproducible off-host (HFSEC-1). The model card renders only hyperparameters, so
    # it is safe to build from the sanitized config too.
    public_config = _sanitize_config_for_upload(config_dict)
    card = generate_model_card(
        tag=tag,
        config_dict=public_config,
        metrics=compute_rf_metrics(model_path, tag),
        versions=_collect_library_versions(),
    )

    _disable_hub_progress_bars()
    api = _hf_api()
    with tempfile.TemporaryDirectory(prefix="aetherscan_hf_upload_") as staging:
        for stable_name, source in sources.items():
            if stable_name == HF_CONFIG_FILENAME:
                with open(os.path.join(staging, stable_name), "w") as f:
                    json.dump(public_config, f, indent=2)
            else:
                shutil.copy2(source, os.path.join(staging, stable_name))
        with open(os.path.join(staging, HF_CARD_FILENAME), "w") as f:
            f.write(card)

        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=staging, commit_message=tag)

    if force:
        # --force-tag semantics: repoint the existing tag at the fresh commit rather than
        # failing (delete errors are ignored — the tag may simply not exist yet).
        # NOTE: delete_tag -> create_tag is not atomic (the Hub has no tag-move primitive):
        # a crash between the two calls leaves the tag missing until the stage is retried —
        # hf_upload stays un-done in the run manifest, so re-running the identical command
        # recreates it. Acceptable for a consciously-forced override.
        with contextlib.suppress(Exception):
            api.delete_tag(repo_id, tag=tag)
    try:
        api.create_tag(repo_id, tag=tag)
    except Exception as e:
        if _is_tag_conflict(e):
            # TOCTOU: the startup dedup guard checked this tag hours ago (a full-scale run
            # trains for ~30 h) — a concurrent run may have created it since. The artifacts
            # are safely uploaded on the main branch; only the tag is missing.
            raise RuntimeError(
                f"Tag '{tag}' already exists on {repo_id} (created after this run's "
                f"startup check, e.g. by a concurrent run). The artifacts were uploaded "
                f"but left untagged — re-run the identical command with --force-tag to "
                f"move the tag to this run's upload."
            ) from e
        raise
    logger.info(f"Uploaded run '{tag}' to https://huggingface.co/{repo_id} and tagged the commit")
