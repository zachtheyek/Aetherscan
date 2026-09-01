# Contributing to Aetherscan

Thank you for your interest in contributing to Aetherscan! This document describes the process for contributing to the project.

---

## Getting Started

### AI Usage

The Aetherscan project has strict rules for AI usage. Please see the [AI usage policy](/AI_POLICY.md) before proceeding. **This is very important**.

### Prerequisites

Aetherscan supports two install paths off the same source tree; you only need the prerequisites for the one you plan to use locally:

- **NGC container path (canonical, both clusters)** — Apptainer 1.4+ or SingularityCE 4.1+, plus an NVIDIA GPU with driver ≥570 (Blackwell) or ≥550 (Ampere via CUDA forward compatibility). Python 3.12 / TF 2.17 / CUDA 12.8 live inside the container.
- **Conda env (alternative, Ampere only)** — Conda or Mamba, Python 3.10, CUDA 12.3+ driver, NVIDIA Ampere GPU.

See [`README.md`](README.md#system-requirements) for the full system requirements matrix.

Contributors should additionally have the following:

- Git with [GPG signing configured](#commit-signing-gpg)
- [pre-commit](https://pre-commit.com/)
- [tmux](https://github.com/tmux/tmux)

### Development Setup

Pick whichever install path matches your dev environment.

**Container path** (canonical; works on both clusters):

```bash
git clone https://github.com/zachtheyek/Aetherscan.git
cd Aetherscan

# The image is acquired automatically on the first `utils/run_container.sh` run: it pulls the
# prebuilt image from GHCR (ghcr.io/zachtheyek/aetherscan) and caches it as aetherscan-ngc25.02.sif.
# A fresh `master` clone resolves the newest published release at or below its own version (#424),
# so you normally build nothing. Build here only when the pull can't serve you — offline,
# requirements-container.txt has moved past that release, non-x86_64, or a driver below the
# CUDA 12.8 floor.
# singularity build aetherscan-ngc25.02.sif aetherscan.def   # (or apptainer) only if the pull can't serve you

# Launch tmux session
# All subsequent commands should be ran in the top pane of the pipeline
# window (in focus on startup)
./utils/start_tmux_session.sh

# Install pre-commit hooks
pre-commit install

# Sanity check
./utils/run_container.sh python utils/print_cli_help.py top
```

**Conda env path** (alternative, Ampere only):

```bash
git clone https://github.com/zachtheyek/Aetherscan.git
cd Aetherscan

conda env create -f environment.yml

# Launch tmux session
# All subsequent commands should be ran in the top pane of the pipeline
# window (in focus on startup)
./utils/start_tmux_session.sh

# Install pre-commit hooks
pre-commit install

# Sanity check
PYTHONPATH=src python utils/print_cli_help.py top
```

See [`README.md`](README.md#installation) for the full walkthrough including `.env` configuration and how to launch the pipeline.

---

## Commit Signing (GPG)

Every commit that lands in Aetherscan must carry a **verified GPG signature** — this is enforced by branch protection, so unsigned or unverifiable commits will block your PR. If you've never set this up, the one-time process below takes a few minutes.

> [!NOTE]
> GitHub shows a signature as **Verified** only when all three of these line up: (1) the commit is signed with a GPG key uploaded to your account, (2) the commit's author email matches a user ID (UID) on that key, and (3) that same email is a **verified** email on your GitHub account. Most "Unverified" badges come from one of these three drifting out of sync.

### Already have a GPG key?

List your secret keys and grab the fingerprint (the 40-character hex string on the line below `sec`):

```bash
gpg --list-secret-keys --keyid-format=long
```

If the key's UID email is the address you commit with (and it's verified on GitHub), skip ahead to [Point git at your key](#point-git-at-your-key). If the key lacks that email, either add it as a UID (`gpg --edit-key <FINGERPRINT>` → `adduid` → `save`) or generate a fresh key below.

### Generate a new key

```bash
gpg --full-generate-key
```

Recommended answers at the prompts:

| Prompt     | Choose                                                                 |
| ---------- | ---------------------------------------------------------------------- |
| Key type   | `(9) ECC (sign and encrypt)` → `(1) Curve 25519` (or RSA 4096)         |
| Expiry     | Your call — `0` for no expiry, or e.g. `2y` and renew before it lapses |
| Real name  | Your name                                                              |
| Email      | An address that is **verified on your GitHub account**                 |
| Passphrase | A strong passphrase (cached by `gpg-agent` / your OS keychain)         |

ed25519 (Curve 25519) keys are smaller and faster; RSA 4096 is the conservative choice if you need maximum tooling compatibility. GitHub accepts either.

### Add the public key to GitHub

Export the armored public key:

```bash
gpg --armor --export <FINGERPRINT> > my-gpg-key.asc
```

Then either paste the whole `-----BEGIN PGP PUBLIC KEY BLOCK-----` block into **GitHub → Settings → SSH and GPG keys → New GPG key**, or use the CLI:

```bash
gh auth refresh -s write:gpg_key    # one-time: grant gh permission to manage GPG keys
gh gpg-key add my-gpg-key.asc
```

### Point git at your key

This is the part that tailors signing to your commits. Set it globally, or drop `--global` to scope signing to this repo only:

```bash
git config --global user.signingkey <FINGERPRINT>    # which key to sign with
git config --global commit.gpgsign true              # sign every commit
git config --global tag.gpgsign true                 # sign every tag (optional)
git config --global user.email "you@verified-email"  # must match a key UID + a verified GitHub email
git config --global gpg.program "$(command -v gpg)"  # only needed if you have multiple gpg installs
```

### Verify

```bash
git commit --allow-empty -m "test: gpg signing"
git log --show-signature -1     # expect "Good signature from ..."
git reset --soft HEAD~1         # discard the throwaway commit
```

Once pushed, the commit should show a green **Verified** badge on GitHub.

### Troubleshooting

> [!TIP]
>
> - **`error: gpg failed to sign the data`** — the agent can't reach a prompt for your passphrase. Add `export GPG_TTY=$(tty)` to your shell rc (`~/.zshrc` / `~/.bashrc`) and re-source it.
> - **macOS passphrase prompt never appears** — install a GUI pinentry: `brew install pinentry-mac`, add `pinentry-program $(brew --prefix)/bin/pinentry-mac` to `~/.gnupg/gpg-agent.conf`, then `gpgconf --kill gpg-agent`.
> - **Commit shows "Unverified" on GitHub** — your commit email isn't a verified account email, or isn't a UID on the uploaded key. Reconcile the three: `git config user.email`, `gpg --list-secret-keys`, and GitHub → Settings → Emails.
> - **`gpg` commands hang (newer GnuPG with the `use-keyboxd` backend)** — a wedged `keyboxd` / `gpg-agent` daemon. Run `gpgconf --kill all` and retry; if it persists, remove the stale `~/.gnupg/S.keyboxd` socket and try again.

---

## Project Structure

```
Aetherscan/
├── src/aetherscan/             # Main package
│   ├── __init__.py             # Package initialization
│   ├── main.py                 # Entry point, command dispatch
│   ├── cli.py                  # Argument parsing, validation, config override
│   ├── config.py               # Configuration defaults
│   ├── train.py                # Training orchestration
│   ├── round_data.py           # Disk-backed (memmap) round datasets + producer process
│   ├── run_state.py            # Persisted training-run manifest (stage-aware resume)
│   ├── inference.py            # Inference orchestration
│   ├── inference_viz.py        # Inference visualization suite
│   ├── candidate_figures.py    # Per-candidate figure renderer (TF-free; forkserver pool; called by inference_viz.py)
│   ├── candidate_triage.py     # Report-time frequency exclusion + OOD review ordering (TF-free)
│   ├── preprocessing.py        # Data preprocessing + energy detection
│   ├── pfb.py                  # PFB static passband equalization
│   ├── data_generation.py      # Synthetic signal injection
│   ├── seeding.py              # Root-seed stream derivation (reproducible train + inference runs)
│   ├── benchmark.py            # Always-on stage timing → pipeline_stages table
│   ├── dashboard.py            # Streamlit live-monitoring dashboard (DB-driven)
│   ├── dashboard_launcher.py   # Spawns the headless dashboard subprocess (guarded)
│   ├── dashboard_cli.py        # Console entry point for manual dashboard runs (aetherscan-dashboard)
│   ├── hf_hub.py               # HuggingFace Hub artifact upload/download
│   ├── tag_guards.py           # Fail-early --save-tag dedup guards
│   ├── display_tag.py          # Machine-scoped display tag for filenames + plot titles (stdlib-only)
│   ├── rf_metrics.py           # Pure RF eval-metric helper (persisted to training_stats via train.py)
│   ├── shap_parallel.py        # RF SHAP process-pool wrapper (TF-free; called by train.py)
│   ├── latent_variants.py      # Latent-representation variant catalogue + selection/calibration (TF-free)
│   ├── latent_gif.py           # Process-parallel latent-GIF frame renderer (TF-free; called by train.py)
│   ├── models/
│   │   ├── __init__.py         # Model exports
│   │   ├── vae.py              # Beta-VAE architecture
│   │   └── random_forest.py    # RF architecture
│   ├── db/
│   │   ├── __init__.py         # Database exports
│   │   └── db.py               # SQLite async writer
│   ├── logger/
│   │   ├── __init__.py         # Logger exports
│   │   ├── logger.py           # Logging configuration
│   │   └── slack_handler.py    # Slack integration
│   ├── monitor/
│   │   ├── __init__.py         # Monitor exports
│   │   └── monitor.py          # Resource monitoring
│   └── manager/
│       ├── __init__.py         # Manager exports
│       └── manager.py          # Resource lifecycle management
├── docs/                       # Technical documentation suite, one doc per pipeline surface
│                               #   (indexed in docs/README.md; start at docs/ARCHITECTURE.md)
├── tests/                      # Pytest suite (see Testing section below)
│   ├── conftest.py                  # Singleton-reset fixtures, tmp paths, synthetic data factories
│   ├── fixtures/                    # Small recorded data fixtures (e.g. real ED channel slice)
│   ├── unit/                        # Fast, hardware-independent unit tests (run in CI)
│   └── integration/                 # gpu+cluster-marked end-to-end smoke tests
├── benchmarks/                 # Standalone micro-benchmarks (not collected by pytest;
│                               #   see benchmarks/README.md + docs/BENCHMARKING.md)
├── utils/                      # Utility scripts
│   ├── benchmark_report.py          # Render the stage-timing benchmark report
│   ├── candidate_rfi_report.py      # Standalone read-only candidate RFI-triage report (reads the run DB)
│   ├── fetch_run_outputs.sh         # rsync a run's outputs from a remote node
│   ├── find_optimal_configs.py      # Per-host config helper
│   ├── get_system_info.sh           # System info helper
│   ├── hf_tag_release.py            # Bless trained weights as a release tag on HF
│   ├── kill_pipeline.sh             # Stop a running pipeline (main + workers)
│   ├── perband_report.py            # Render the per-band inference-performance plot
│   ├── probe_candidate_location.py  # Trace a (cadence, frequency) location through the scoring cascade
│   ├── print_cli_help.py            # CLI reference regen helper
│   ├── run_container.sh             # Apptainer/SingularityCE wrapper
│   ├── start_tmux_session.sh        # tmux session template helper
│   └── verify_train_test_files.py   # Data sanity check helper
├── .claude/                    # Claude Code config + on-demand skills (skills/aetherscan-repo-context/SKILL.md)
├── .github/                    # CI/CD workflows, issue templates, etc.
├── .gitignore                  # Local gitignore
├── .pre-commit-config.yaml     # Pre-commit hook configuration
├── aetherscan.def              # Apptainer/SingularityCE build recipe (NGC container)
├── Dockerfile                  # OCI twin of aetherscan.def; published to GHCR by publish-image.yml
├── requirements-container.txt  # Pip extras layered into NGC container
├── environment.yml             # Conda dependencies
├── pyproject.toml              # Package metadata, ruff config
├── CLAUDE.md                   # Lean always-on agent rules
├── AI_POLICY.md                # AI usage policy
├── CITATION.cff                # Citation metadata
├── CODE_OF_CONDUCT.md          # Core values guidelines
├── CODEOWNERS                  # Code ownership
├── CONTRIBUTING.md             # Contributing guidelines
├── KNOWN_ISSUES.md             # Catalog of known issues and workarounds
├── LICENSE                     # Project license
├── README.md                   # Project overview, installation & usage guides
└── SECURITY.md                 # Security policy
```

### Module Responsibilities

| Module                    | Purpose                                                                |
| ------------------------- | ---------------------------------------------------------------------- |
| `main.py`                 | CLI entry point, command routing, GPU strategy setup (NCCL + fallback) |
| `cli.py`                  | Argument parsing, validation, config override                          |
| `config.py`               | All configuration dataclasses and defaults                             |
| `train.py`                | Training orchestration, curriculum learning, checkpointing             |
| `round_data.py`           | Disk-backed round datasets, `.done` manifests, background producer     |
| `run_state.py`            | Persisted training-run manifest driving stage-aware resume             |
| `inference.py`            | Model inference, candidate detection                                   |
| `inference_viz.py`        | End-of-run inference visualization suite                               |
| `candidate_figures.py`    | Per-candidate figure rendering (TF-free; forkserver + empty preload; called by `inference_viz.py`) |
| `candidate_triage.py`     | Report-time candidate triage: frequency-exclusion partitioning for tallies/Slack surfaces and Mahalanobis OOD scores for review ordering (TF-free; stdlib-only at import so `cli.py` can import its validator) |
| `preprocessing.py`        | Data loading / downsampling / log-normalization + energy detection     |
| `pfb.py`                  | Polyphase-filterbank static passband response (bandpass flattening)    |
| `data_generation.py`      | Synthetic signal injection using setigen                               |
| `seeding.py`              | Root-seed stream derivation (reproducible train + inference runs)      |
| `benchmark.py`            | Always-on stage timing (`stage_timer`/`record_stage`) to `pipeline_stages` |
| `dashboard.py`            | Streamlit live-monitoring dashboard read from the run SQLite DB        |
| `dashboard_launcher.py`   | `launch_dashboard()` spawns/reaps the headless dashboard subprocess    |
| `dashboard_cli.py`        | `aetherscan-dashboard` console script; re-execs Streamlit for manual DB inspection |
| `hf_hub.py`               | HuggingFace Hub artifact upload/download, version-coupled revisions    |
| `tag_guards.py`           | Fail-early `--save-tag` dedup guards (local + HF collisions)           |
| `display_tag.py`          | Derives the presentation-only `{command}_{machine}_{datetime}` display tag used for artifact filenames, plot titles and Slack messages (stdlib-only; the DB tag stays `{command}_{datetime}`) |
| `rf_metrics.py`           | Pure RF eval-metric helper written to `training_stats` (`model_name='rf'`) |
| `shap_parallel.py`        | Process-pool wrapper for RF SHAP passes (TF-free; forkserver + empty preload; called by `train.py`) |
| `latent_variants.py`      | Latent-representation variant catalogue, selection metrics, probability calibrator (TF-free; shared by `train.py` and `inference.py`) |
| `latent_gif.py`           | Process-parallel latent-GIF frame rendering (TF-free; forkserver + empty preload; called by `train.py`) |
| `models/vae.py`           | Beta-VAE architecture with composite clustering loss                   |
| `models/random_forest.py` | Scikit-learn RF wrapper                                                |
| `db/db.py`                | Thread-safe SQLite with async queue-based writes                       |
| `monitor/monitor.py`      | Background resource monitoring (CPU, RAM, GPU)                         |
| `manager/manager.py`      | Resource lifecycle management (pools, shared memory)                   |
| `logger/`                 | Multi-handler logging with Slack integration                           |

For how these modules fit together — data flow, process/thread topology, initialization
order, and where every artifact lands — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md);
the full per-surface documentation suite is indexed in [`docs/README.md`](docs/README.md).

---

## Contribution Workflow

> [!TIP]
> **All issues are actionable, and all PRs must be tied to an existing issue.**

### 1. Start a Discussion

Before making any changes:

- First check to see if any related PRs, issues, or discussions already exist
- If not, open a [GitHub Discussion](https://github.com/zachtheyek/Aetherscan/discussions) or [Slack thread](https://breakthroughlisten.slack.com/archives/C0A3CDALQD8)

> [!TIP]
> Consider asking @claude whether your query has any existing PRs, issues, or discussions

### 2. Open an Issue

Once a discussion has reached a well-understood problem statement:

- Open a [GitHub Issue](https://github.com/zachtheyek/Aetherscan/issues) by filling out the appropriate issue template completely
- Claude will automatically triage and assign your issue a label
- Optionally, if you'd like to tackle this PR, make your interest known to the maintainers within the issue itself

> [!WARNING]
> "Drive-by" issues (i.e. issues opened without prior discussions with maintainers) may be closed without review or explanation

### 3. Create a Feature Branch

Branch naming convention: `category/description`

| Category   | Use Case                      | Example                     |
| ---------- | ----------------------------- | --------------------------- |
| `feature/` | New functionality             | `feature/db_integration`    |
| `hotfix/`  | Bug fixes                     | `hotfix/cpu_sampling_rate`  |
| `misc/`    | Housekeeping tasks            | `misc/update_docs`          |
| `claude/`  | Reserved for Claude assistant | `claude/fix_typo_in_readme` |

```bash
git checkout -b feature/my_new_feature
```

### 4. Implement Changes

- Keep commits focused and well-described
- Ensure all pre-commit hooks pass
- Follow the code conventions in `pyproject.toml` (PEP-8 with minor relaxations, enforced via [ruff](https://docs.astral.sh/ruff/))
- Write tests and update documentation if applicable

### 5. Submit a Pull Request

- Ensure your branch is up-to-date with `master` (use `git rebase`, not `git merge`)
- All commits must have [verified GPG signatures](#commit-signing-gpg)
- Fill out the appropriate PR template completely
- Link your PR to the associated issue using GitHub's **"Development" sidebar** or by including `Closes #N` / `Fixes #N` in the PR body. This creates a formal link that enables automatic label syncing

> [!WARNING]
> PRs not tied to an existing issue may be closed without review or explanation

> [!NOTE]
> Labels from linked issues are automatically synced to PRs. Claude also checks that PRs have linked issues and that issues have prior discussions — non-compliant contributions will receive a warning comment and label.

### 6. Code Review

- PRs require:
  - all status checks to pass
  - at least one maintainer approval
  - all conversations to be resolved
  - branches to be up to date
- Address review feedback promptly
- PR approvals are voided when new commits are pushed

> [!NOTE]
> Claude will automatically provide an initial code review. You do not need to address every point raised. Use your own judgement and discuss with a maintainer if you're unsure.

---

## Pre-commit Hooks

This project uses pre-commit hooks for code quality:

```yaml
# .pre-commit-config.yaml hooks:
- ruff # Linting
- ruff-format # Formatting
- pre-commit-hooks # General-purpose checks
- gitleaks # Secret detection
```

Once installed using `pre-commit install`, the hooks should automatically run on every commit, and block changes that don't pass every hook. Pre-commit will attempt to fix "simple" issues, so if any hooks are failing, you may just need to run `git add` and `git commit` again. For more "complex" cases, manual intervention is needed. See the pre-commit messages for details.

### Running Manually

```bash
# Run all hooks on staged files
pre-commit run

# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
```

### Bypassing Hooks (Use Sparingly)

```bash
# Skip hooks for a single commit (not recommended)
git commit --no-verify -m "message"
```

---

## Code Style

### Ruff Configuration

This project uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting, and follows [PEP-8](https://peps.python.org/pep-0008/) with minor relaxations. The full configuration lives in [`pyproject.toml`](pyproject.toml) under `[tool.ruff]`; highlights below.

- **Target version**: Python 3.10 (lowest common denominator across the conda 3.10 path and the container 3.12 path)
- **Line length**: 100 characters (formatter wraps; `E501` is intentionally ignored so wrapping is the formatter's job, not the linter's)
- **Enabled rule families**: PEP-8 (`E`, `W`), pyflakes (`F`), isort (`I`), pep8-naming (`N`), pyupgrade (`UP`), bugbear (`B`), comprehensions (`C4`), simplify (`SIM`), pylint (`PL`), flake8-print (`T20`), plus select flake8-logging-format checks (`G001`–`G003`)
- **Notable allowances**: unused vars (`F841`), many-param functions (`PLR0913`), and magic-number comparisons (`PLR2004`) are permitted intentionally
- **Per-file ignores**: `F401` (unused imports) is allowed in every `__init__.py`; `T20` (`print()`) is allowed throughout `utils/` (one-off scripts); and `T201` is allowed in `logger/slack_handler.py` (a logging handler can't log through itself without recursion)
- **Import sorting**: isort-compatible, with `aetherscan` declared as the only first-party namespace; isort's `required-imports` also makes `from __future__ import annotations` mandatory atop every module (`I002` auto-inserts it)
- **Quote style**: Double quotes
- **Indentation**: 4 spaces

### Key Style Rules

> [!TIP]
> Most of these are enforced automatically by ruff (lint) and ruff-format. The list below highlights the patterns you'll see throughout the codebase and what new code should match.

#### Modern Python typing

Every module starts with `from __future__ import annotations` so type expressions are stringified at import time (lets us use modern syntax under the 3.10 target). Use **PEP 604 unions** (`X | None`) instead of `Optional[X]` and **PEP 585 generics** (`list[int]`, `dict[str, float]`, `tuple[X, ...]`) instead of `typing.List` / `typing.Dict`. Annotate both arguments and return types on function signatures.

```python
from __future__ import annotations

def load_inference_data(
    self, override_filepaths: list[str] | None = None
) -> np.ndarray:
    ...
```

`ruff`'s `UP` (pyupgrade) family enforces these idioms (`from typing import List, Dict, Optional, Tuple, ...` is flagged and auto-fixed), and isort's `required-imports` setting (`I002`) auto-inserts the `from __future__ import annotations` line in any module missing it.

#### Module and function docstrings

Short, plain prose. No Sphinx / Google / Numpy-style section markers — describe what the module or function does and any non-obvious design choice. One-line docstrings are fine for self-evident helpers; expand only when behavior isn't clear from the signature.

```python
"""
Inference orchestration for Aetherscan Pipeline
Implements distributed model inference and candidate detection.
Supports distributed datasets & latent generation
"""
```

```python
def _warmup_collective(strategy):
    """Trigger a tiny cross-device reduction to surface NCCL failures at setup time."""
    ...
```

#### Inline comment markers

Five markers are used consistently across the codebase — they are grep-friendly entry points for things still in flight or worth thinking about:

| Marker    | Use for                                                          |
| --------- | ---------------------------------------------------------------- |
| `# TODO:` | Concrete, actionable work item                                   |
| `# NOTE:` | Clarification, rationale, or question worth coming back to       |
| `# FIX:`  | Known issue you don't have time to fix right now                 |
| `# BUG:`  | Known bug, often paired with a workaround in the next lines      |
| `# TEST:` | Behavior that needs verifying (informal test plan, no suite yet) |

Prefer `# NOTE:` over `# TODO:` when there's no obvious action — the latter implies someone owes follow-through.

#### Logging

Get a module-level logger named after the module and use f-strings for messages. Avoid bare `print()` outside of one-off scripts under `utils/`.

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Loaded {len(backgrounds)} backgrounds from {path}")
logger.warning(f"NCCL warmup failed ({e}), falling back to HierarchicalCopy")
```

The Slack handler attaches automatically when `SLACK_BOT_TOKEN` is set in the env, so anything you log at `INFO+` may also surface in Slack — keep messages information-dense and free of secrets.

`ruff` backs part of this: `T20` rejects bare `print()` in package code (allowed only under `utils/`, plus the self-logging `slack_handler.py`), and `G001`–`G003` reject `%` / `str.format()` / `+` pre-formatted log messages in favour of f-strings. The remaining conventions — module-level placement, the `__name__` logger name, and the no-secrets rule — aren't expressible as ruff rules and are spot-checked by the post-merge style-check workflow instead.

#### Config singleton access

`config.py` exposes a thread-safe singleton via `get_config()`. The None-guard is the canonical idiom: the getter is annotated as `Config | None` (so type checkers don't complain at the call site), but in practice it's only `None` if `init_config()` hasn't run yet — which is a programming error.

```python
from aetherscan.config import get_config

config = get_config()
if config is None:
    raise ValueError("get_config() returned None")

batch_size = config.training.per_replica_batch_size
```

#### Dataclass mutable defaults

Always use `field(default_factory=...)` for mutable defaults — a bare `[...]` default is shared across every instance of the dataclass, which is a Python footgun and a real source of bugs:

```python
# Good
train_files: list[str] = field(
    default_factory=lambda: [
        "real_filtered_LARGE_HIP110750.npy",
        "real_filtered_LARGE_HIP13402.npy",
    ]
)

# Bad: shared mutable state across instances
train_files: list[str] = ["real_filtered_LARGE_HIP110750.npy", ...]
```

`ruff`'s `B` (bugbear) family will flag the bad form.

#### Retry / error-handling pattern

Pipeline retry loops follow a consistent shape: catch `KeyboardInterrupt` separately and re-raise so traceback propagates, log the exception with `logger.error`, decide whether to retry or `sys.exit(1)`, and `time.sleep(retry_delay)` before the next attempt. The pattern lives in `train_command` and `inference_command` if you need a reference.

```python
for attempt in range(max_retries):
    try:
        ...
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
        else:
            sys.exit(1)
```

#### Variable naming

Prefer descriptive, full-word names: `num_training_rounds`, `per_replica_batch_size`, `coarse_channel_width` — not `n`, `bs`, `ccw`. Single-letter names are reserved for tight loops, math expressions, and array indexing. See the [Naming Conventions](#naming-conventions) table below for the casing rules.

### Naming Conventions

| Element       | Convention  | Example                  |
| ------------- | ----------- | ------------------------ |
| Classes       | PascalCase  | `DataGenerator`          |
| Functions     | snake_case  | `run_training_pipeline`  |
| Constants     | UPPER_SNAKE | `MAX_RETRIES`            |
| Private       | \_prefix    | `_init_worker`           |
| Config fields | snake_case  | `per_replica_batch_size` |

---

## Testing

The test suite lives in `tests/` and runs on [pytest](https://docs.pytest.org/) (configured in `pyproject.toml` under `[tool.pytest.ini_options]` — `pythonpath = ["src"]` makes `import aetherscan` work without a `PYTHONPATH` prefix).

### Running the suite

```bash
# Default selection — everything that doesn't need GPUs or cluster-resident data.
# Matches what CI runs (.github/workflows/tests.yml); CI additionally appends
# `and not integration` as a defense-in-depth leak-guard against a future test
# marked only `integration` (which skips the isolation fixture) — today the two
# expressions select the same set because every `integration` test is also
# `gpu`+`cluster`.
pytest -m "not gpu and not cluster" -q

# Cluster integration smokes (end-to-end train + CSV inference), inside the NGC container:
./utils/run_container.sh python -m pytest tests/ -m "gpu or cluster" -q

# A single file / test while iterating:
pytest tests/unit/test_config.py -q
pytest tests/unit/test_db.py::TestFlushSentinel -q
```

### Markers

| Marker        | Meaning                                                                                       |
| ------------- | --------------------------------------------------------------------------------------------- |
| `slow`        | Slower tests (e.g. builds TF graphs on CPU); included in the default CI selection             |
| `gpu`         | Requires one or more physical GPUs; excluded from CI                                          |
| `cluster`     | Requires cluster-resident data/models (blpc3/bla0); excluded from CI                          |
| `integration` | End-to-end pipeline runs; skips the per-test singleton/env isolation fixture in `conftest.py` |

### Test environment & fixtures

- Every non-integration test runs isolated: `tests/conftest.py` points `AETHERSCAN_{DATA,MODEL,OUTPUT}_PATH` at a pytest `tmp_path`, resets all singletons (`Config`, `ResourceManager`, `Database`, `Logger`, `ResourceMonitor`) via their `_reset()` hooks, and re-runs `init_config()` — so tests never touch real data or leak state into each other.
- Synthetic data factories are provided as fixtures: `make_background_npy` (tiny background plates), `make_h5_observation` (tiny filterbank-style `.h5` files), and `make_inference_csv` (tiny cadence-grouping CSVs).
- CI (`.github/workflows/tests.yml`) runs the default selection on Ubuntu with `tensorflow-cpu==2.17.*` + the pins from `requirements-container.txt`, on Python 3.10, 3.11, and 3.12 (3.10 and 3.12 being the conda and container runtimes respectively, 3.11 the in-between version `requires-python` permits). A few tests assert Linux-only behavior (e.g. PSS-based memory accounting) and self-skip elsewhere.

### Writing tests

- Every PR that adds or changes logic should ship unit tests for it (`tests/unit/`), reusing the conftest fixtures rather than instantiating singletons directly.
- Mark anything that needs hardware or cluster data with the appropriate marker so the default selection stays runnable on a laptop and in CI.
- Test style follows the same ruff lint+format rules as `src/` (pre-commit runs on `tests/` too).

---

## New Version Releases

Releases are cut through the CD workflow ([`.github/workflows/release.yml`](.github/workflows/release.yml)): pushing a signed `v*` tag builds the package, publishes it to PyPI, and creates the GitHub Release. The full contract — the SemVer versioning policy (which segment to bump, and how a retrain's weights are classified), the PyPI/HuggingFace version-coupling, the signed-tag / version / weights CD gates, and the step-by-step runbook — lives in [`docs/RELEASE.md`](docs/RELEASE.md). Before tagging, bless the matching trained weights on the HF model repo with [`utils/hf_tag_release.py`](utils/hf_tag_release.py).

---

## Sensitive Data Warning

**DO NOT commit sensitive information.** Pre-commit hooks scan for secrets using [gitleaks](https://github.com/gitleaks/gitleaks), but this is not foolproof.

Never commit:

- API keys or tokens (`.env` files)
- Credentials or passwords
- Private data files
- Internal URLs or IP addresses

If you accidentally commit sensitive data, see [SECURITY.md](SECURITY.md) for remediation steps.

---

## Questions?

- Check [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for known problems
- Open a [GitHub Discussion](https://github.com/zachtheyek/Aetherscan/discussions) or [Slack thread](https://breakthroughlisten.slack.com/archives/C0A3CDALQD8) for other questions
