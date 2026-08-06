# Release Engineering

This document is the release contract for Aetherscan: how versions are numbered (SemVer) and
what each bump promises; how one version string couples PyPI, GitHub, and HuggingFace; what
the release CD does (and refuses to do); the one-time setup a maintainer needs; and the
step-by-step runbook for cutting a release. The packaging and the CD workflow
(`.github/workflows/release.yml`) are implemented to satisfy this contract — when they and
this document disagree, fix whichever is wrong *as a PR that says so*.

## TL;DR — the version-coupling contract

**One version string, e.g. `v1.0.0`, names four synchronized objects:**

1. A **git tag** `v1.0.0` on `master` — which GitHub decorates into a **GitHub Release**
   (notes + source tarballs) and which the CD builds into the **PyPI release** `1.0.0`.
2. An **HF weights tag** `v1.0.0` on the model repo `zachtheyek/aetherscan`, pointing at the
   blessed trained weights.
3. A **GHCR image tag** `v1.0.0` on `ghcr.io/zachtheyek/aetherscan` — the prebuilt runtime
   container that `utils/run_container.sh` pulls instead of building the `.sif` locally.

The coupling is **by convention, enforced in CD**: CD verifies the HF weights tag exists and
itself builds+pushes the container image, and only then publishes PyPI + the GitHub Release — so
the four never drift. CD **verifies** weights but never creates them (weights come only from
cluster training runs — CI has neither the GPUs nor the data); it **does** create the image
(a Dockerfile build needs neither GPUs nor data). The installed package pulls its matching
weights at runtime because the package version is the default HF revision:

```
pip install aetherscan==1.0.0
python -m aetherscan.main inference ...   # no model-path flags
  → resolves HF revision v1.0.0 → downloads the blessed weights → runs
```

Two invariants to internalize:

- **PyPI versions are immutable.** A botched release means `v1.0.1`, never a re-upload.
- **Installing `aetherscan==1.0.0` *is* the tagged source** — the sdist/wheel are built from
  the tag by CD. There is nothing further to "sync" between PyPI and GitHub.
- **HF and GHCR carry a tag *per version*, but reuse *content* when nothing changed.** A release
  whose weights didn't change re-points its HF tag at the same training commit; a release whose
  image inputs (the `Dockerfile` — base digest, labels, layers — and `requirements-container.txt`)
  didn't change retags the same GHCR digest. So `v1.0.0` and `v1.1.0` can be byte-identical
  weights/image under distinct
  version tags — each checkout still pulls *its own* tag. See
  [Version bump with no weights/image change](#version-bump-with-no-weights-or-image-change).

## Versioning policy (SemVer)

Aetherscan follows [Semantic Versioning 2.0.0](https://semver.org): every release is
`MAJOR.MINOR.PATCH` (`vX.Y.Z`). Now that the project is past `1.0.0` the number is a
**compatibility contract**, not just a date stamp. The public surface the contract covers is:

- the `python -m aetherscan.main {train,inference}` **CLI** — subcommand set, flag names, and the
  defaults that decide behavior when a flag is omitted;
- the **saved-artifact + config-JSON formats** and the **model contract** inference reads from them
  (`latent_dim`, the RF feature layout / `latent_variant` — plus the forest's additive,
  backward-compatible `aetherscan_latent_variant_` / `aetherscan_active_dims_` stamps, which
  inference's identity checks simply no-op on when absent — the `.keras`/`.joblib` formats, the HF
  repo layout and weight-resolution rules);
- the **supported runtimes** (NGC container — built from `aetherscan.def` or pulled prebuilt from
  GHCR — / conda env / PyPI package) and their documented floors.

Internal implementation details, private helpers, log wording, and plot cosmetics are **not** part
of the contract. Bump the leftmost segment that applies:

- **MAJOR (`X`) — incompatible changes** that force a user to change how they invoke, configure, or
  consume the pipeline, or that make an existing released artifact unusable with the new code.
  Examples: removing or renaming a CLI flag/subcommand without a compatibility alias; a config-schema
  or artifact-format change a prior release's saved config/weights can no longer load; changing the
  model contract so an old artifact is silently reinterpreted; raising the Python/CUDA/TF floor so a
  previously-supported runtime stops working; a backward-incompatible DB-schema change. A retrain
  whose weights need new code to load is a MAJOR change **to the weights**.
- **MINOR (`Y`) — backward-compatible additions**: anything a user could ignore and keep working
  exactly as before. Examples: a new CLI flag whose default preserves prior behavior; a new optional
  config field; new observability/plots; a new latent-variant; a performance change with identical
  outputs; or **new blessed weights from a retrain whose artifact + config + model contract are
  unchanged** (same `latent_dim`, feature layout, and file formats — old configs still load and new
  code loads old weights). The weights are the product, so re-blessing on the same contract is a
  user-facing feature → at least a MINOR bump.
- **PATCH (`Z`) — backward-compatible fixes**: no new capability, no contract change. Examples: a bug
  fix (e.g. the off-cluster `tf_keras` weight-load fix for
  [#323](https://github.com/zachtheyek/Aetherscan/issues/323)); a dependency security bump inside
  the documented version ranges; a docs-only correction; a packaging fix. If a user would see no
  behavioral difference except that something broken now works, it's a PATCH.

Two rules make bundled releases unambiguous:

1. **Highest bump wins.** A release takes the largest bump any single change in it requires — one
   breaking change makes the whole release MAJOR no matter how many MINOR/PATCH changes ride along.
2. **Code and weights share the one version string** (see the coupling contract above): if *either*
   the code or the weights warrant a given bump, the release takes at least that bump. (So the next
   release after `1.0.0` bundling new features, the `tf_keras` PATCH fix, and a same-contract retrain
   is a **MINOR** → `v1.1.0`.)

Between releases `master` carries a `.devN` pre-release version (e.g. `1.0.1.dev0`) so it never
advertises itself as a shipped stable version — see the dev-version reset in the runbook below. That
`.devN` number is only a *not-yet-released* placeholder, **not** a commitment to the next version:
the actual next release number is chosen at release time by the rules above, so `1.0.1.dev0` on
`master` does not mean the next release is `1.0.1`. The pre-`1.0.0` `0.y.z` line made no
compatibility promises; from `1.0.0` onward, these rules hold.

## Packaging

- **Build system**: hatchling (`[build-system] requires = ["hatchling"]` in
  [`pyproject.toml`](../pyproject.toml)), wheel target `src/aetherscan`.
  `requires-python = ">=3.10,<3.13"`.
- **Dependencies**: `[project].dependencies` mirrors
  [`requirements-container.txt`](../requirements-container.txt)'s ranges **plus** the
  packages the NGC base image provides implicitly (`tensorflow[and-cuda]==2.17.*`, `tf_keras`,
  `h5py`, `hdf5plugin`, `psutil`) — a pip install has no base image to lean on.
  Optional extras: `dev = ["ruff", "pre-commit", "pytest>=9.0.3"]` (the `pytest` floor is a
  security floor — GHSA-6w46-j5rx-g56g, tmpdir handling; dev/test-only) and
  `dashboard = ["streamlit>=1.54.0,<1.55", "plotly>=5.24,<6", "pandas>=2.0,<3"]` —
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
2. **`--hf-revision <tag>`** — pin any HF revision (a training tag like
   `train_20260101_120000`, a release tag like `v1.0.0`, or a commit).
3. **`v{__version__}`** — when running as an installed release, the package's own version is
   the default revision. This is the line that makes `pip install aetherscan==1.0.0` +
   bare inference pull exactly the `v1.0.0` weights.
4. **Latest `v*` semver tag** on the HF repo. Training tags never name the default download —
   a no-artifact inference download requires a blessed release tag.
5. Otherwise: error with guidance.

Downloads happen **lazily at first inference**, never at import time (an import-time network
download would be hostile), revision-pinned and cached under the standard HF cache
(`~/.cache/huggingface`; set `HF_HOME` to redirect it — e.g. to scratch — if home isn't
writable/bound in your container setup: `utils/run_container.sh` binds and forwards `HF_HOME`
when set, see [`GPU_RUNTIME_GUIDE.md`](GPU_RUNTIME_GUIDE.md)). Public repo — downloads need no token.

### HF repo layout and tag families

One model repo, **`zachtheyek/aetherscan`**, with **stable filenames** at the repo root —
`vae_encoder.keras`, `vae_decoder.keras`, `random_forest.joblib`, `config.json`, and an
auto-generated model card `README.md` — and **revisions (HF git tags) carrying the
versioning**:

| Tag family | Created by | Points at |
| --- | --- | --- |
| Training tags (`train_20260101_120000`, ...) | Training runs with `--hf-upload` (tag = the run's `save_tag`) | The commit that upload produced |
| Release tags (`v1.0.0`, ...) | The release runbook (the bless, step 5 below) | The blessed training upload's commit |

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
   (`pytest -m "not gpu and not cluster and not integration"`; the trailing
   `and not integration` is a defense-in-depth leak-guard — see
   [`TESTING.md`](TESTING.md#ci)). The release workflow reuses `tests.yml`
   directly ([`.github/workflows/release.yml`](../.github/workflows/release.yml)),
   so the two selections stay in sync automatically. A release build must not
   outrun a red suite.
4. **HF weights verification** — confirm the matching `v*` tag exists on
   `zachtheyek/aetherscan` (public repo, no token needed). **Verify, never create**: if this
   fails, you skipped the weight-blessing step; the error says so and how to fix it.
5. **Build** — `python -m build` (sdist + wheel) from the tagged source, then **smoke the
   wheel**: install it `--no-deps` and assert `aetherscan.__version__` equals the guarded
   version (catches a packaging/version-single-sourcing mismatch before anything publishes).
6. **Container image** — the reusable
   [`publish-image.yml`](../.github/workflows/publish-image.yml) builds the OCI image from
   [`Dockerfile`](../Dockerfile) and pushes it to `ghcr.io/zachtheyek/aetherscan:vX.Y.Z` (auth is
   the built-in `GITHUB_TOKEN` with `packages: write` — no PAT). It runs in parallel with the
   build and **gates the PyPI publish** (step 7 `needs` it), so a real release never reaches PyPI
   unless the image is up. The image is rebuilt only when its inputs (the whole `Dockerfile`, base
   digest included, plus `requirements-container.txt`) change; otherwise it **retags the existing
   digest** under the new version — the Aetherscan code is bind-mounted at runtime, not baked in,
   so a code-only release
   reuses the prior image (see
   [Version bump with no weights/image change](#version-bump-with-no-weights-or-image-change)). On
   the TestPyPI dry run it validates only (no build, no push), so the gate stays uniform.
7. **Publish to PyPI via trusted publishing** — `pypa/gh-action-pypi-publish` (SHA-pinned)
   with `permissions: id-token: write` and the `pypi` environment. OIDC-based: **no long-lived
   PyPI API token is stored anywhere** in the repo or its secrets.
8. **GitHub Release** — created from the tag (`gh release create --verify-tag
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

All of this is already done for `zachtheyek/Aetherscan` — it is listed so a fork or a future
maintainer can reproduce it, and so the runbook's prereq check (step 1) has something concrete to
verify against.

- **PyPI (real release)**: create the `aetherscan` project on pypi.org and add GitHub as a
  **trusted publisher** — repository `zachtheyek/Aetherscan`, workflow `release.yml`, environment
  `pypi` — then create the `pypi` environment in this repo (Settings → Environments). The publish
  step authenticates against this via OIDC; no API token is ever stored or rotated.
- **TestPyPI (dry run)**: the **same** trusted-publisher setup on test.pypi.org, with a `testpypi`
  environment in this repo. Required for the dry-run step (runbook step 4) to publish — without it
  the dry run's build/tests still run but its final publish step fails at authentication.
- **HF**: confirm the model repo `zachtheyek/aetherscan` exists and is **public** (inference
  downloads need no token), and that wherever the bless is run (a cluster, or any machine running
  `hf_tag_release.py`) has a **write-scoped** `HF_TOKEN` in its gitignored `.env`. CD never writes
  to HF — only the training upload and the bless do.
- **GHCR (container image)**: no secret to create — CD authenticates with the built-in
  `GITHUB_TOKEN` (the `image` job grants it `packages: write`). After the **first** image push the
  package `ghcr.io/zachtheyek/aetherscan` is created *private*; set it **public** once (GitHub →
  Packages → `aetherscan` → Package settings → Change visibility → Public), so clusters pull with
  no token. The `org.opencontainers.image.source` label links it back to this repo. Then confirm
  the package page stays license-compliant (see [Container image licensing](#container-image-licensing)).
- **Verify** the environments exist before a release:
  `gh api repos/zachtheyek/Aetherscan/environments --jq '.environments[].name'` must list both
  `pypi` and `testpypi`.

## Release runbook

The concrete, in-order sequence for cutting `vX.Y.Z`. **This is the exact process used for the
v1.0.0 release** — substitute your version and training tag. Roles: **maintainer** = a human with
repo admin + a registered GPG signing key + (for the bless) a write `HF_TOKEN`; **agent** = steps
the assistant can drive; **CD** = automatic. Each step gates the next; do not reorder.

1. **Prereqs (maintainer/agent).** One-time setup above is done — verify with
   `gh api repos/zachtheyek/Aetherscan/environments --jq '.environments[].name'` (must list `pypi`
   and `testpypi`). Every intended PR is merged and `master` is green
   (`gh run list --branch master --limit 1`). Decide two things up front: the version `X.Y.Z` and
   the Development-Status classifier (`4 - Beta` vs `5 - Production/Stable`).

2. **Train the release model (maintainer).** Full-scale `train --save-tag train --hf-upload` on
   the chosen cluster. The run stamps its own datetime, so the tag resolves to
   `train_{YYYYMMDD_HHMMSS}` — **copy it from the startup log; every later step needs it.** Weights
   land locally *and* on HF under that training tag (this upload is separate from the release tag
   created in step 5). **Review the artifacts** (loss curves, injection stats, latent diagnostics,
   RF plots — [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)); this is the release-qualification
   evidence. Recommended: also do a Phase-3 inference pass on a real `/datag` catalog subset — it
   sanity-checks the weights on real data and captures the inference RAM/VRAM/disk numbers (from
   the always-on `system_resources` DB rows) that the README update in step 3 needs.

3. **Release PR (agent drafts, maintainer merges).** One PR that:
   - bumps `version` in `pyproject.toml` to `X.Y.Z` — the single source of truth; CD enforces the
     pushed git tag equals `v` + this;
   - sets the Development-Status classifier per step 1;
   - updates the README **System Requirements** with the real RAM/VRAM/disk numbers from the
     training + inference runs (issue #183; read them off the `system_resources` DB rows — cite the
     run tags + hardware for provenance);
   - bumps `CITATION.cff` `version` and `date-released`;
   - links the release issue (`Closes #183`).
   Run the normal review loop. **This repo allows merge commits only** (rebase/squash are
   disabled), and branch protection keeps the PR at `mergeStateStatus: BLOCKED` until it has a
   review approval — merge after approval, or with `gh pr merge <n> --admin --merge --delete-branch`
   if an admin is cutting the release directly.

4. **Dry-run to TestPyPI (agent/maintainer).** *After* the release PR merges (so the build carries
   the real `X.Y.Z`, not the dev pre-release), trigger the dry run on master and watch it:
   ```bash
   gh workflow run release.yml -f test_pypi=true --ref master
   gh run watch "$(gh run list --workflow=release.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
   ```
   The dry run does guard → tests → build + **wheel smoke** + **image (validate-only)** → publish to
   test.pypi.org. It **skips** the signed-tag gate, the HF-weights verification, and the GitHub
   Release — so it needs **no bless and no git tag**. Its whole job is to catch packaging / version-single-sourcing /
   build breakage *before* you spend the immutable real version number. Confirm it went green (and,
   if you like, that `aetherscan X.Y.Z` shows up on test.pypi.org) before continuing.

5. **Bless the weights (maintainer, or agent if authorized + `HF_TOKEN` is present).** Create the
   release tag *on HF*, pointing at the training upload's commit:
   ```bash
   python utils/hf_tag_release.py --save-tag train_{YYYYMMDD_HHMMSS} --release vX.Y.Z
   ```
   Run it where a write-scoped `HF_TOKEN` is available (a cluster `.env`, etc.); it is a thin
   wrapper over `HfApi.create_tag`. This is the human "these weights **are** the release" decision
   — CD verifies the tag but deliberately cannot create it. It must be done before step 6 (the real
   CD's HF-verify checks it); re-running for an existing tag errors clearly rather than
   duplicating.

   > **No retrain this release?** Skip step 2 and bless the **existing** training tag under the new
   > version — `hf_tag_release.py --save-tag <the same old train tag> --release vX.Y.Z` re-points a
   > new HF tag at the same commit, so the old and new versions resolve to identical weights. See
   > [Version bump with no weights/image change](#version-bump-with-no-weights-or-image-change).

6. **Sign + push the tag (maintainer ONLY).** Update local `master` to the release PR's merge
   commit (`git checkout master && git pull`), then:
   ```bash
   git tag -s vX.Y.Z && git push origin vX.Y.Z
   ```
   Signed with your GPG key — **the assistant cannot do this step.** If unrelated work merged to
   `master` after the release PR, don't tag `HEAD` — tag that PR's merge commit explicitly so the
   release captures exactly the reviewed tree: `git tag -s vX.Y.Z <release-PR-merge-sha>`. The push
   is the point of no return: it triggers the real CD, and PyPI versions are immutable.

7. **CD does the rest (automatic — watch it).** guard (signed tag + `tag == v+version`) → tests →
   HF-weights verify + build + wheel smoke + image build/push to GHCR → publish to PyPI (trusted
   publishing) → GitHub Release (notes + sdist/wheel). If **HF-verify** fails, you skipped step 5:
   the git tag is already correct, so **do NOT retag** — run step 5, then re-run the failed
   workflow (`gh run rerun <run-id> --failed`). If the **image** job fails (e.g. a transient
   runner/registry error), likewise re-run the failed job — the git tag is already correct, don't
   re-push it; PyPI/GitHub-Release are gated behind the image, so a failed image blocks them rather
   than half-releasing.

8. **Smoke the release (agent/maintainer).** On a clean venv/machine (**not** the source tree, so
   `__version__` comes from the installed package):
   ```bash
   pip install aetherscan==X.Y.Z
   python -m aetherscan.main inference --inference-files <small_catalog.csv>   # no --encoder-path/--rf-path/--config-path
   ```
   Confirm it lazily downloads the `vX.Y.Z` weights from HF and completes. That closes the loop on
   the version-coupling contract.

9. **Reset the dev version (agent drafts, maintainer merges).** Small follow-up chore PR right
   after the release lands: bump `pyproject.toml` `version` from `X.Y.Z` to the next pre-release
   (e.g. `X.Y.(Z+1).dev0`) so `master` stops advertising itself as a shipped stable version between
   releases (`src/aetherscan/__init__.py` reads it back via `importlib.metadata`). Revisit the
   Development-Status classifier only if the maturity level actually changed.

> **Release notes.** CD creates the GitHub Release with `--generate-notes`; curate the body
> afterward from the per-merge `claude-release-notes` comments (see
> [`GITHUB_AUTOMATION.md`](GITHUB_AUTOMATION.md)) — they are the raw material, written one merge at
> a time for exactly this purpose.
>
> **Recovering from a broken release.** PyPI versions are immutable: if a published release is
> broken, fix forward with `vX.Y.(Z+1)` (a fresh release PR + tag). You can `pip`-yank the bad
> version on PyPI so resolvers skip it, but the number is spent. Releases are always cut forward
> from `master`, never on a maintenance branch — note that an out-of-order tag push (tagging an
> older line *after* a newer one shipped) would also drag GHCR `:latest` backward onto the older
> image, since `:latest` is a mutable pointer whoever pushed last owns (unlike PyPI, which orders
> versions). The fix-forward-only rule keeps that from happening.

## Version bump with no weights or image change

Most releases change only code (a bug fix, a new flag). Weights change only on a retrain; the
container image changes only when the `Dockerfile` (its base digest, labels, or layers) or
`requirements-container.txt` changes (the Aetherscan code is **bind-mounted at runtime, not baked
into the image**). So a release often needs
**no new weights and no new image** — but the vX.Y.Z contract still wants a tag for each, so a
`v1.0.0` and a `v1.1.0` checkout each pull *their own* tag. The rule is **new tag, reused content**:

| Object | This release… | What the release does |
| --- | --- | --- |
| PyPI / GitHub | always (code is versioned) | build + publish `vX.Y.Z` from the tag |
| **HF weights** | did **not** retrain | re-bless: `hf_tag_release.py --save-tag <old train tag> --release vX.Y.Z` → a new HF tag on the **same commit** as the prior release |
| **HF weights** | retrained | train `--hf-upload`, then bless the **new** training tag (runbook steps 2 + 5) |
| **GHCR image** | `Dockerfile` (base digest, labels, layers) + `requirements-container.txt` unchanged | CD **retags the existing digest** to `vX.Y.Z` automatically — no rebuild (its input fingerprint `fp-<hash>` already exists in GHCR) |
| **GHCR image** | any `Dockerfile` or `requirements-container.txt` change | CD **rebuilds** and pushes a new digest as `vX.Y.Z` |

You do nothing special for the image: the `image` job fingerprints its inputs (the whole `Dockerfile`
— base digest, labels, and layers — plus `requirements-container.txt`) into an `fp-<hash>` marker tag
and decides build-vs-retag on its own. For weights the only manual
choice is *which* training tag to bless — the existing one (no retrain) or a fresh one (retrain).
Either way both `v1.0.0` and `v1.1.0` end up as real tags on HF **and** GHCR: a v1.0.0 checkout
pulls `:v1.0.0`, a v1.1.0 checkout pulls `:v1.1.0`, and when nothing changed those tags resolve to
the same underlying commit/digest.

## Container image licensing

The GHCR image is a derivative of NVIDIA's NGC TensorFlow container, so **the image as a whole is
governed by the [NVIDIA Deep Learning Container License](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf)**
— **not** this repo's BSD-3-Clause, which covers only the Aetherscan source (bind-mounted at
runtime, absent from the image). Publishing is expressly permitted (license §1(c), distributing a
"Compatible derived CONTAINER") as long as we keep meeting the following, which the
[`Dockerfile`](../Dockerfile) labels and the package-visibility step encode — **future releases
must preserve them**:

- **§2(a) material added functionality** — the image adds the Aetherscan runtime stack; never
  publish the bare NGC base.
- **§2(b) required notice** — the Dockerfile sets `LABEL com.nvidia.notice="This software contains
  source code provided by NVIDIA Corporation."`; keep it.
- **§2(c) / §4(g) at-least-as-protective terms, no OSS-license infection** — the
  `org.opencontainers.image.licenses` label is `LicenseRef-NVIDIA-Deep-Learning-Container-License`;
  present the GHCR package as governed by that license, and **do not** relabel the whole image
  under BSD-3-Clause or any OSI license.
- **§4(d) no implied endorsement** — don't describe the image as sponsored/endorsed by NVIDIA.
- **§7 / §15 third-party components & export** — TF (Apache-2.0), CUDA/cuDNN (their own EULAs) keep
  their licenses; a public GHCR image is a distribution subject to US export rules.

If any of this is ever in doubt for a release, the license routes questions to
`nvidia-compute-license-questions@nvidia.com`.

### Backfilling an older release's image (e.g. v1.0.0)

v1.0.0 predates the `Dockerfile`, so its image is published once by dispatching the reusable
workflow **after this change lands on `master`** (`workflow_dispatch` only runs workflows present on
the default branch). It builds from the current `Dockerfile` + that version's
`requirements-container.txt` and pushes `:vX.Y.Z` (auth is the built-in `GITHUB_TOKEN` — no PAT):

```bash
gh workflow run publish-image.yml -f version=1.0.0 -f ref=v1.0.0
gh run watch "$(gh run list --workflow=publish-image.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
```

The `:latest` tag tracks the **newest** release: a real release (`workflow_call` from `release.yml`)
moves `:latest`, but a backfill dispatch does **not** (its `latest` input defaults to `false`) — so
publishing an older version never regresses `:latest`, which `run_container.sh` pulls for `.devN`
checkouts. If you ever need to move it, pass `-f latest=true`.

**A backfill never creates or moves `:latest`** — the dispatch above publishes the version tag
(plus its internal `fp-<hash>` marker tag) only. Historical note: between the v1.0.0 backfill and
the v1.1.0 release there was **no** `:latest` at all, so `.devN`/`master` checkouts had to build
from `aetherscan.def` — deliberate, because master's `requirements-container.txt` had already
moved past v1.0.0's (notably the `streamlit` security bump), and pinning `:latest` to the
backfilled image would have served dev checkouts a knowingly stale dependency set. Since v1.1.0's
CD run, `:latest` exists and tracks the newest real release, so dev checkouts pull normally
(until their `requirements-container.txt` moves past the last release again).

Verify: `docker buildx imagetools inspect ghcr.io/zachtheyek/aetherscan:v1.0.0`, or just pull it on
a cluster via `utils/run_container.sh`.

## FAQ

- *Can GitHub releases and HF tagged weights get out of sync?* Only by skipping the bless (step 5) —
  and then CD refuses to publish. The same tag string on both sides, with CD enforcing
  existence, is the whole synchronization mechanism.
- *Can the GHCR image get out of sync with the release?* No — CD builds/pushes it during the
  release run and **gates PyPI on it**, so `vX.Y.Z` exists on GHCR whenever the PyPI/GitHub release
  does. When the image inputs didn't change, CD retags the existing digest instead of rebuilding
  (see [Version bump with no weights/image change](#version-bump-with-no-weights-or-image-change)).
- *Why can't CI produce the weights?* Training needs cluster GPUs, hundreds of GB of
  scratch, and real background data. Hence verify-don't-create: the pipeline's most
  expensive artifact is produced and reviewed by humans, and automation only checks it's
  where it should be.
- *What if a release is broken?* PyPI is immutable: fix forward with `vX.Y.(Z+1)` (a new
  release PR + tag). You can yank the bad version on PyPI so resolvers skip it, but the
  version number is spent.
- *Do source-tree users see any of this?* The `PYTHONPATH=src` / conda path is untouched
  (`__version__` reads `0.0.0.dev0`, explicit model paths keep working). The **container** path now
  pulls the release-pinned image from GHCR when no local `.sif` is present (or prints
  `aetherscan.def` build instructions if the pull fails) — a convenience, not a contract change: a
  checkout of a release tag pulls that version's image, a `.devN` checkout falls back to `:latest`.
