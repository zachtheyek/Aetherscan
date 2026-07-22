# Release Engineering

This document is the release contract for Aetherscan: how one version string couples PyPI,
GitHub, and HuggingFace; what the release CD does (and refuses to do); the one-time setup a
maintainer needs; and the step-by-step runbook for cutting a release. The packaging and the
CD workflow (`.github/workflows/release.yml`) are implemented to satisfy this contract —
when they and this document disagree, fix whichever is wrong *as a PR that says so*.

## TL;DR — the version-coupling contract

**One version string, e.g. `v1.0.0`, names three synchronized objects:**

1. A **git tag** `v1.0.0` on `master` — which GitHub decorates into a **GitHub Release**
   (notes + source tarballs) and which the CD builds into the **PyPI release** `1.0.0`.
2. An **HF weights tag** `v1.0.0` on the model repo `zachtheyek/aetherscan`, pointing at the
   blessed trained weights.

The coupling is **by convention, enforced by verification**: CD checks that the HF tag
exists before publishing anything, but never creates weights — weights can only come from
cluster training runs (CI has neither the GPUs nor the data). The installed package pulls
its matching weights at runtime because the package version is the default HF revision:

```
pip install aetherscan==1.0.0
aetherscan.main inference ...        # no model-path flags
  → resolves HF revision v1.0.0 → downloads the blessed weights → runs
```

Two invariants to internalize:

- **PyPI versions are immutable.** A botched release means `v1.0.1`, never a re-upload.
- **Installing `aetherscan==1.0.0` *is* the tagged source** — the sdist/wheel are built from
  the tag by CD. There is nothing further to "sync" between PyPI and GitHub.

## Packaging

- **Build system**: hatchling (`[build-system] requires = ["hatchling"]` in
  [`pyproject.toml`](../pyproject.toml)), wheel target `src/aetherscan`.
  `requires-python = ">=3.10,<3.13"`.
- **Dependencies**: `[project].dependencies` mirrors
  [`requirements-container.txt`](../requirements-container.txt)'s ranges **plus** the
  packages the NGC base image provides implicitly (`tensorflow[and-cuda]==2.17.*`, `h5py`,
  `hdf5plugin`, `psutil`) — a pip install has no base image to lean on.
  Optional extras: `dev = ["ruff", "pre-commit", "pytest"]` and
  `dashboard = ["streamlit>=1.39,<1.41", "plotly>=5.24,<6", "pandas>=2.0,<3"]` —
  `pip install aetherscan[dashboard]` pulls the stack for the packaged live dashboard
  ([`src/aetherscan/dashboard.py`](../src/aetherscan/dashboard.py)). Dependency *versions* follow
  [`SECURITY.md`](../SECURITY.md)'s selection policy, and the documented ceilings
  (`numpy<2.0`, the NGC TF 2.17 ABI) apply to the package exactly as to the container.
- **Version single-sourcing**: the static `version` in `pyproject.toml` is the only place
  the version is written. `src/aetherscan/__init__.py` exposes `__version__` via
  `importlib.metadata.version("aetherscan")`, falling back to `"0.0.0.dev0"` on
  `PackageNotFoundError` so source-tree runs (`PYTHONPATH=src`, the container) work without
  an install.
- The `Development Status` classifier is a per-release decision (Beta vs Production/Stable)
  — revisit it in each release PR.

## Weight resolution (runtime side of the contract)

Inference resolves models in this precedence order:

1. **Explicit local paths** (`--encoder-path` / `--rf-path` / `--config-path`) — always win;
   the offline/cluster path.
2. **`--hf-revision <tag>`** — pin any HF revision (a training tag like `final_v1`, a
   release tag like `v1.0.0`, or a commit).
3. **`v{__version__}`** — when running as an installed release, the package's own version is
   the default revision. This is the line that makes `pip install aetherscan==1.0.0` +
   bare inference pull exactly the `v1.0.0` weights.
4. **Latest `v*` semver tag** on the HF repo, then **latest `final_vX`** training tag.
5. Otherwise: error with guidance.

Downloads happen **lazily at first inference**, never at import time (an import-time network
download would be hostile), revision-pinned and cached under the standard HF cache
(`~/.cache/huggingface`; set `HF_HOME` if home isn't writable/bound in your container
setup). Public repo — downloads need no token.

### HF repo layout and tag families

One model repo, **`zachtheyek/aetherscan`**, with **stable filenames** at the repo root —
`vae_encoder.keras`, `vae_decoder.keras`, `random_forest.joblib`, `config.json`, and an
auto-generated model card `README.md` — and **revisions (HF git tags) carrying the
versioning**:

| Tag family | Created by | Points at |
| --- | --- | --- |
| Training tags (`final_v1`, `test_v17`, ...) | Training runs with `--hf-upload` (tag = the run's `save_tag`) | The commit that upload produced |
| Release tags (`v1.0.0`, ...) | The release runbook (step 3 below) | The blessed training upload's commit |

Uploads need a write-scoped `HF_TOKEN` in the environment (`.env` on the clusters —
gitignored, forwarded into the container; never logged, never committed). Upload failure
never fails a training run — weights are already safe locally.

## The CD workflow: `release.yml`

`on: push: tags: ["v*"]`. Steps, in order — each gate exists to make a half-released state
impossible:

1. **Signed-tag gate** — the `guard` job's "Enforce a signed release tag" step rejects
   lightweight tags and any tag whose GPG signature GitHub does not verify against a key
   registered to the tagger's account (releases carry the same provenance as this repo's
   signed commits). No key material lives in CI — GitHub does the verification. (Skipped on
   the TestPyPI dry run, which has no tag.)
2. **Version guard** — the pushed tag must equal `v` + the `pyproject.toml` version;
   otherwise fail loudly before anything publishes. (Prevents tagging a commit whose release
   PR didn't land.)
3. **Unit tests** — the same selection as `tests.yml`
   (`pytest -m "not gpu and not cluster"`). A release build must not outrun a red suite.
4. **HF weights verification** — confirm the matching `v*` tag exists on
   `zachtheyek/aetherscan` (public repo, no token needed). **Verify, never create**: if this
   fails, you skipped the weight-blessing step; the error says so and how to fix it.
5. **Build** — `python -m build` (sdist + wheel) from the tagged source, then **smoke the
   wheel**: install it `--no-deps` and assert `aetherscan.__version__` equals the guarded
   version (catches a packaging/version-single-sourcing mismatch before anything publishes).
6. **Publish to PyPI via trusted publishing** — `pypa/gh-action-pypi-publish` (SHA-pinned)
   with `permissions: id-token: write` and the `pypi` environment. OIDC-based: **no long-lived
   PyPI API token is stored anywhere** in the repo or its secrets.
7. **GitHub Release** — created from the tag (`gh release create --verify-tag
   --generate-notes`) with the built sdist + wheel attached; curate the body from the
   per-PR `claude-release-notes` comments (see
   [`GITHUB_AUTOMATION.md`](GITHUB_AUTOMATION.md)) — they are the raw material, written one
   merge at a time for exactly this purpose.

A workflow-level `concurrency:` block groups runs by ref
(`group: release-${{ github.ref }}`, `cancel-in-progress: false`), so two runs for the same
tag — e.g. an accidental double tag-push, or a `test_pypi: true` dry run overlapping a real
release for the same ref — are serialized rather than run in parallel, and an in-flight
publish is never cancelled by a newer run. The grouping is at the release-workflow level, so
it covers the whole gate chain above (not just the publish step). Unrelated tags do not
serialize against each other.

An optional `workflow_dispatch` input (`test_pypi: true`) publishing to test.pypi.org is the
recommended dry-run before the first real release.

### One-time setup (maintainer)

- **PyPI**: create the `aetherscan` project and add GitHub as a **trusted publisher** —
  repository `zachtheyek/Aetherscan`, workflow `release.yml`, environment `pypi`. This is
  what step 6 authenticates against; no token to rotate, nothing to leak.
- **HF**: confirm the model repo `zachtheyek/aetherscan` exists and that the clusters'
  `.env` files carry a write-scoped `HF_TOKEN` (for training uploads only — CD never
  writes to HF).

## Release runbook

Steps for cutting `vX.Y.Z` (maintainer + agent together):

1. **Prereqs** — one-time setup above is done; all intended PRs are merged; `master` is
   green.
2. **Train the release model** — full-scale
   `train --save-tag final_vN --hf-upload` on the chosen cluster. Weights land locally and
   on HF tagged `final_vN`. **Review the training artifacts** — the loss curves, injection
   stats, latent diagnostics, and RF plots ([`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md))
   are the release-qualification evidence.
3. **Bless the weights** — create the release tag on HF pointing at the training upload:
   `utils/hf_tag_release.py --save-tag final_vN --release vX.Y.Z` (a thin wrapper over
   `HfApi.create_tag`). This is the human "these weights are the release" decision — CD
   deliberately cannot make it.
4. **Release PR** — bump `version = "X.Y.Z"` in `pyproject.toml`, revisit the Development
   Status classifier, draft the release notes (curate the `claude-release-notes` comments),
   regenerate anything version-stamped. Merge through normal review.
5. **Tag** — from the release PR's merge commit on master:
   `git tag -s vX.Y.Z && git push origin vX.Y.Z` (signed, like every commit in this repo).
6. **CD does the rest** — guard → tests → HF verify → build → PyPI → GitHub Release. Watch
   the run. If the HF-verify step fails, you skipped step 3: don't touch the git tag (it's
   already correct) — run step 3, then re-run the failed workflow.
7. **Smoke the release** — on a clean venv/machine:
   `pip install aetherscan==X.Y.Z`, then run inference with **no model-path flags** against
   a small catalog; confirm it downloads the `vX.Y.Z` weights and completes. That closes the
   loop on the contract.

## FAQ

- *Can GitHub releases and HF tagged weights get out of sync?* Only by skipping step 3 —
  and then CD refuses to publish. The same tag string on both sides, with CD enforcing
  existence, is the whole synchronization mechanism.
- *Why can't CI produce the weights?* Training needs cluster GPUs, hundreds of GB of
  scratch, and real background data. Hence verify-don't-create: the pipeline's most
  expensive artifact is produced and reviewed by humans, and automation only checks it's
  where it should be.
- *What if a release is broken?* PyPI is immutable: fix forward with `vX.Y.(Z+1)` (a new
  release PR + tag). You can yank the bad version on PyPI so resolvers skip it, but the
  version number is spent.
- *Do source-tree users see any of this?* No — the container/`PYTHONPATH=src` workflows are
  untouched; `__version__` just reads `0.0.0.dev0` and explicit model paths keep working.
