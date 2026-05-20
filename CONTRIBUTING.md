# Contributing to Aetherscan

Thank you for your interest in contributing to Aetherscan! This document describes the process for contributing to the project.

---

## Getting Started

### AI Usage

The Aetherscan project has strict rules for AI usage. Please see the [AI usage policy](/AI_POLICY.md) before proceeding. **This is very important**.

### Prerequisites

Aetherscan supports two install paths off the same source tree; you only need the prerequisites for the one you plan to use locally:

- **NGC container path (canonical, both clusters)** — Apptainer 1.4+ or SingularityCE 4.1+, plus an NVIDIA GPU with driver ≥570 (Blackwell) or ≥550 (Ampere via CUDA forward compatibility). Python 3.12 / TF 2.17 / CUDA 12.8 live inside the container.
- **Conda env (alternative, Ampere only)** — Conda or Mamba, Python 3.10, CUDA 12.4+ driver, NVIDIA Ampere GPU.
- **Both paths** — Git with GPG signing configured; [pre-commit](https://pre-commit.com/) (`pip install pre-commit` or `brew install pre-commit`).

See [`README.md`](README.md#system-requirements) for the full system requirements matrix.

### Development Setup

Pick whichever install path matches your dev environment, then install the pre-commit hooks.

**Container path** (canonical; works on both clusters):

```bash
git clone https://github.com/zachtheyek/Aetherscan.git
cd Aetherscan

# Build the .sif image with whichever runtime is installed on the host:
singularity build aetherscan-ngc25.02.sif aetherscan.def
# or:
apptainer build aetherscan-ngc25.02.sif aetherscan.def

# Sanity check
./utils/run_container.sh python utils/print_cli_help.py top
```

**Conda env path** (alternative, Ampere only):

```bash
git clone https://github.com/zachtheyek/Aetherscan.git
cd Aetherscan

conda env create -f environment.yml
conda activate aetherscan

# Sanity check
PYTHONPATH=src python utils/print_cli_help.py top
```

**Pre-commit hooks** (required for both paths):

```bash
pre-commit install
```

See [`README.md`](README.md#installation) for the full walkthrough including `.env` configuration and how to launch the pipeline.

---

## Project Structure

```
Aetherscan/
├── src/aetherscan/             # Main package
│   ├── __init__.py             # Package initialization, version
│   ├── main.py                 # Entry point, command dispatch, GPU strategy setup
│   ├── cli.py                  # Argument parsing, validation, config override
│   ├── config.py               # Configuration dataclasses
│   ├── train.py                # Training orchestration
│   ├── inference.py            # Inference orchestration
│   ├── evaluate.py             # Evaluation pipeline (stub; not yet wired)
│   ├── preprocessing.py        # Data preprocessing + energy detection
│   ├── data_generation.py      # Synthetic signal injection
│   ├── models/
│   │   ├── __init__.py         # Model exports
│   │   ├── vae.py              # Beta-VAE architecture
│   │   └── random_forest.py    # RF classifier
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
├── docs/                       # Runbooks and guides
│   └── BLACKWELL_MIGRATION.md  # Container + dual-cluster migration runbook
├── tests/                      # Test suite (placeholder; no tests yet)
├── utils/                      # Utility scripts
│   ├── run_container.sh             # Apptainer/SingularityCE auto-detecting wrapper
│   ├── start_tmux_session.sh        # Monitoring tmux session helper
│   ├── print_cli_help.py            # README CLI Reference regen helper
│   ├── find_optimal_configs.py      # Per-host config sweep
│   ├── verify_train_test_files.py   # Training/test data sanity check
│   └── get_system_info.sh           # System info dump (for bug reports)
├── .github/                    # CI/CD workflows, issue templates, CODEOWNERS, etc.
├── .gitignore                  # Local gitignore
├── .pre-commit-config.yaml     # Pre-commit hook configuration
├── aetherscan.def              # Apptainer/SingularityCE build recipe for the NGC container
├── requirements-container.txt  # Pip extras layered into the NGC container
├── environment.yml             # Conda dependencies (Ampere conda env path)
├── pyproject.toml              # Package metadata, ruff config
├── AGENTS.md                   # AI agent guidelines
├── AI_POLICY.md                # AI usage policy
├── CITATION.cff                # Citation metadata
├── CODE_OF_CONDUCT.md          # Core values guidelines
├── CODEOWNERS                  # Code ownership
├── CONTRIBUTING.md             # This file
├── KNOWN_ISSUES.md             # Known issues and workarounds
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
| `inference.py`            | Model inference, candidate detection                                   |
| `evaluate.py`             | Evaluation pipeline (stub; not yet dispatched from `main.py`)          |
| `preprocessing.py`        | Data loading / downsampling / log-normalization + energy detection     |
| `data_generation.py`      | Synthetic signal injection using setigen                               |
| `models/vae.py`           | Beta-VAE architecture with custom clustering loss                      |
| `models/random_forest.py` | Scikit-learn RF wrapper                                                |
| `db/db.py`                | Thread-safe SQLite with async queue-based writes                       |
| `monitor/monitor.py`      | Background resource monitoring (CPU, RAM, GPU)                         |
| `manager/manager.py`      | Resource lifecycle management (pools, shared memory)                   |
| `logger/`                 | Multi-handler logging with Slack integration                           |

> [!WARNING]
>
> # TODO: add an architecture section?

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
- All commits must have verified GPG signatures
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
- Note: PR approvals are voided when new commits are pushed

> [!NOTE]
> Claude will automatically provide an initial code review. You do not need to address every point raised. Use your own judgement and discuss with a maintainer if you're unsure.

### 7. After Merge

Once your PR is merged, delete the remote feature branch from `origin` so the branch list stays clean. Either:

```bash
# CLI
git push origin --delete <branchname>
```

…or click the **"Delete branch"** button GitHub shows on the merged PR page. Locally, you can also prune the tracking ref with `git fetch --prune`.

> [!TIP]
> If the repository's **Settings → General → Pull Requests → Automatically delete head branches** option is enabled, GitHub does this for you on merge and this step becomes a no-op.

---

## Pre-commit Hooks

The project uses pre-commit hooks for code quality:

```yaml
# .pre-commit-config.yaml hooks:
- ruff # Linting
- ruff-format # Formatting
- pre-commit-hooks # General-purpose checks
- gitleaks # Secret detection
```

Once installed using `pre-commit install`, the hooks should automatically run on every commit, and block changes that don't pass every hook. Note that pre-commit will attempt to fix "simple" issues, so if any hooks are failing, you may just need to run `git add` and `git commit` again. For more "complex" cases, manual intervention is needed. See the pre-commit messages for details.

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

The project uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting, and follows PEP-8 with minor relaxations. The full configuration lives in [`pyproject.toml`](pyproject.toml) under `[tool.ruff]`; highlights below.

- **Line length**: 100 characters (formatter wraps; `E501` is intentionally ignored so wrapping is the formatter's job, not the linter's)
- **Target version**: Python 3.10 (lowest common denominator across the conda 3.10 path and the container 3.12 path)
- **Quote style**: Double quotes
- **Indentation**: 4 spaces
- **Import sorting**: isort-compatible, with `aetherscan` declared as the only first-party namespace
- **Enabled rule families**: PEP-8 (`E`, `W`), pyflakes (`F`), isort (`I`), pep8-naming (`N`), pyupgrade (`UP`), bugbear (`B`), comprehensions (`C4`), simplify (`SIM`), pylint (`PL`)
- **Notable allowances**: unused vars (`F841`), many-param functions (`PLR0913`), and magic-number comparisons (`PLR2004`) are permitted intentionally; a few `PLR0911/12/15` and `PLW0603` ignores are tracked as temporary in `pyproject.toml`
- **Per-file ignores**: `F401` (unused imports) is allowed in every `__init__.py`

### Key Style Rules

```python
# Good: Type hints with future annotations
from __future__ import annotations

def process_data(data: np.ndarray, config: Config) -> dict[str, float]:
    ...

# Good: Descriptive variable names
num_training_rounds = config.training.num_training_rounds
per_replica_batch_size = config.training.per_replica_batch_size

# Good: Docstrings for public functions
def load_train_data(config: Config) -> tuple[np.ndarray, np.ndarray]:
    """
    Load training data from configured files.

    Args:
        config: Configuration object with data paths

    Returns:
        Tuple of (backgrounds, labels) arrays
    """
    ...
```

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

> [!WARNING]
>
> # TODO: update this section once test suite is operational

---

## New Version Releases

> [!WARNING]
>
> # TODO: add tagged releases workflow when available

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
