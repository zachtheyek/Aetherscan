# Contributing to Aetherscan

Thank you for your interest in contributing to Aetherscan! This document describes the process for contributing to the project.

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA 12.2+ with compatible NVIDIA GPU
- Conda or Mamba for environment management
- Git with GPG signing configured

### Development Setup

```bash
# Clone the repository
git clone https://github.com/zachtheyek/Aetherscan.git
cd Aetherscan

# Create and activate environment
conda env create -f environment.yml
conda activate aetherscan

# Install pre-commit hooks
pre-commit install
```

### AI Usage

The Aetherscan project has strict rules for AI usage. Please see the [AI usage policy](/AI_POLICY.md) before proceeding. **This is very important**.

### Code of Conduct

Basically just don't be an asshole. See [`CODE_OF_CONDUCT.md`](/CODE_OF_CONDUCT.md).

---

## Contribution Workflow

> [!TIP]
> **All issues are actionable, and all PRs must be tied to an existing issue.**

### 1. Start a Discussion

Before making any changes:

- First check to see if any related PRs, issues, or discussions already exist
- If not, open a [GitHub Discussion](https://github.com/zachtheyek/Aetherscan/discussions) or [Slack thread](https://breakthroughlisten.slack.com/archives/C0A3CDALQD8)

### 2. Open an Issue

Once a discussion has reached a well-understood problem statement:

- Open a [GitHub Issue](https://github.com/zachtheyek/Aetherscan/issues) using the appropriate template
- Claude will automatically triage and assign your issue a label
- Optionally, if you'd like to tackle this PR, make your interest known to the maintainers within the issue itself

> [!WARNING]
>
> "Drive-by" issues (i.e. issues opened without prior discussions with maintainers) may be closed without review or explanation

### 3. Create a Feature Branch

Branch naming convention: `category/description`

| Category   | Use Case           | Example                    |
| ---------- | ------------------ | -------------------------- |
| `feature/` | New functionality  | `feature/db_integration`   |
| `hotfix/`  | Bug fixes          | `hotfix/cpu_sampling_rate` |
| `misc/`    | Housekeeping tasks | `misc/update_docs`         |

```bash
git checkout -b feature/my_new_feature
```

### 4. Implement Changes

- Keep commits focused and well-described
- Follow the code conventions in `pyproject.toml` (PEP-8 with minor relaxations, enforced via [ruff](https://docs.astral.sh/ruff/))
- Write tests and update documentation if applicable

### 5. Submit a Pull Request

- Ensure your branch is up-to-date with `master` (use `git rebase`, not `git merge`)
- All commits must have verified GPG signatures
- Fill out the PR template completely
- Link your PR to the associated issue

> [!WARNING]
>
> PRs not tied to an existing issue may be closed without review or explanation

### 6. Code Review

- PRs require at least one maintainer approval
- Address review feedback promptly
- Note: PR approvals are voided when new commits are pushed

> [!NOTE]
> Claude will perform an automatic code review when your PR is first set to "ready for review". You do not need to address every point raised. Use your own judgement and discuss with a maintainer if you're unsure.

---

## Project Structure

```
Aetherscan/
├── src/aetherscan/           # Main package
│   ├── __init__.py           # Package initialization, version
│   ├── main.py               # Entry point, command dispatch
│   ├── cli.py                # Argument parsing, validation
│   ├── config.py             # Configuration dataclasses
│   ├── train.py              # Training orchestration
│   ├── inference.py          # Inference orchestration
│   ├── preprocessing.py      # Data preprocessing
│   ├── data_generation.py    # Synthetic signal injection
│   ├── models/
│   │   ├── __init__.py       # Model exports
│   │   ├── vae.py            # Beta-VAE architecture
│   │   └── random_forest.py  # RF classifier
│   ├── db/
│   │   ├── __init__.py       # Database exports
│   │   └── db.py             # SQLite async writer
│   ├── logger/
│   │   ├── __init__.py       # Logger exports
│   │   ├── logger.py         # Logging configuration
│   │   └── slack_handler.py  # Slack integration
│   ├── monitor/
│   │   ├── __init__.py       # Monitor exports
│   │   └── monitor.py        # Resource monitoring
│   └── manager/
│       ├── __init__.py       # Manager exports
│       └── manager.py        # Resource lifecycle management
├── docs/                     # Additional documentation
│   └── ...
├── tests/                    # Test suite
│   └── ...
├── utils/                    # Utility scripts
│   └── ...
├── .github/                  # CI/CD workflows
│   └── ...
├── .gitignore                # Local gitignore
├── .pre-commit-config.yaml   # Pre-commit hook configuration
├── environment.yml           # Conda dependencies
├── pyproject.toml            # Package metadata, ruff config
├── AGENTS.md                 # AI agent guidelines
├── AI_POLICY.md              # AI usage policy
├── CITATION.cff              # Citation metadata
├── CODE_OF_CONDUCT.md        # Core values guidelines
├── CODEOWNERS                # Code ownership
├── CONTRIBUTING.md           # This file
├── KNOWN_ISSUES.md           # Known issues and workarounds
├── LICENSE                   # Project license
├── README.md                 # Project overview, installation & usage guides
└── SECURITY.md               # Security policy
```

### Module Responsibilities

| Module                    | Purpose                                                    |
| ------------------------- | ---------------------------------------------------------- |
| `main.py`                 | CLI entry point, command routing                           |
| `cli.py`                  | Argument parsing, validation, config override              |
| `config.py`               | All configuration dataclasses and defaults                 |
| `train.py`                | Training orchestration, curriculum learning, checkpointing |
| `inference.py`            | Model inference, candidate detection                       |
| `preprocessing.py`        | Data preprocessing, normalization                          |
| `data_generation.py`      | Synthetic signal injection using setigen                   |
| `models/vae.py`           | Beta-VAE architecture with custom clustering loss          |
| `models/random_forest.py` | Scikit-learn RF wrapper                                    |
| `db/db.py`                | Thread-safe SQLite with async queue-based writes           |
| `monitor/monitor.py`      | Background resource monitoring (CPU, RAM, GPU)             |
| `manager/manager.py`      | Resource lifecycle management (pools, shared memory)       |
| `logger/`                 | Multi-handler logging with Slack integration               |

> [!WARNING]
>
> # TODO: add an architecture section?

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

Once installed using `pre-commit install`, the hooks should run automatically on every commit, and block changes that don't pass all hooks. Note that pre-commit will attempt to automatically fix "simple" issues, so if any hooks are failing, you may simply need to run `git add` and `git commit` again. For more "complex" cases, manual intervention is needed. See the pre-commit messages for details.

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

The project uses Ruff for linting and formatting, and follows the PEP-8 style guides with minor relaxations (see `pyproject.toml`):

- **Line length**: 100 characters
- **Target version**: Python 3.10
- **Quote style**: Double quotes
- **Import sorting**: isort-compatible

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

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_config.py

# With coverage report
pytest --cov=aetherscan --cov-report=html tests/

# Verbose output
pytest -v tests/
```

### Writing Tests

- Place tests in `tests/` directory
- Use `pytest` fixtures for common setup
- Mock external dependencies (GPU, Slack, etc.)
- Use `Config._reset()` in teardown

```python
import pytest
from aetherscan.config import Config, init_config

@pytest.fixture
def config():
    """Provide fresh config for each test."""
    cfg = init_config()
    yield cfg
    Config._reset()

def test_config_defaults(config):
    assert config.beta_vae.latent_dim == 8
    assert config.training.num_training_rounds == 20
```

---

## New Version Releases

> [!WARNING]
>
> # TODO: add tagged releases workflow when available

When releasing a new version, update these files:

| File                         | Location                                  | Example                                             |
| ---------------------------- | ----------------------------------------- | --------------------------------------------------- |
| `pyproject.toml`             | `version = "X.Y.Z"`                       | `version = "1.0.0"`                                 |
| `src/aetherscan/__init__.py` | `__version__ = "X.Y.Z"`                   | `__version__ = "1.0.0"`                             |
| `CITATION.cff`               | `version:` and `date-released:`           | `version: 1.0.0` and `date-released: 2026-01-01`    |
| `SECURITY.md`                | Under "Supported Versions", if applicable | see [`SECURITY.md`](SECURITY.md#supported-versions) |

### Dependency Updates

When updating dependencies, ensure all relevant files are synchronized:

| File                                       | Dependencies                                        | When to Update                                        |
| ------------------------------------------ | --------------------------------------------------- | ----------------------------------------------------- |
| `environment.yml`                          | Conda/pip packages (Python, TensorFlow, CUDA, etc.) | Adding/updating any Python or CUDA dependency         |
| `pyproject.toml`                           | Dev dependencies (ruff), Python version             | Adding dev tools, changing Python version requirement |
| `.pre-commit-config.yaml`                  | Pre-commit hooks (ruff, gitleaks, pre-commit-hooks) | Updating linter/formatter or adding new hooks         |
| `README.md`                                | Version badges and system requirements              | Major version changes to Python, TensorFlow, or CUDA  |
| `CONTRIBUTING.md`                          | Prerequisites section                               | Major version changes to Python, CUDA, or tooling     |
| `.github/workflows/pre-commit.yml`         | Python version, action versions                     | Changing Python version or updating CI actions        |
| `.github/workflows/claude*.yml`            | Claude action versions, model specification         | Updating Claude Code action or model                  |
| `.github/workflows/auto-assign-author.yml` | GitHub action versions                              | Updating GitHub Actions versions                      |

> [!TIP]
> When adding a new Python package, always update `environment.yml` first, then test with a fresh conda environment before committing.

---

## Communications

- [**GitHub Discussions**](https://github.com/zachtheyek/Aetherscan/discussions) or [**Slack**](https://breakthroughlisten.slack.com/archives/C0A3CDALQD8): Questions and general development discussions
- [**GitHub Issues**](https://github.com/zachtheyek/Aetherscan/issues): Actionable bug reports and feature requests

---

## Issue and PR Guidelines

### Issue Authors

- Automatically assigned as issue assignee
- Provide clear reproduction steps for bugs
- Include system information (use `utils/system_info.sh` and append outputs as attachments)

### PR Authors

- Link to the associated issue
- Provide a clear description of changes
- Update tests and documentation if applicable
- Respond to review feedback

### Reviewers

- [`CODEOWNERS`](/CODEOWNERS) are automatically assigned as reviewers
- Focus on correctness, performance, and maintainability
- Be constructive and specific in feedback

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

- Check [README.md](README.md) for project overview
- Check [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for known problems
- Open a [GitHub Discussions](https://github.com/zachtheyek/Aetherscan/discussions) or [Slack thread](https://breakthroughlisten.slack.com/archives/C0A3CDALQD8) for other questions
