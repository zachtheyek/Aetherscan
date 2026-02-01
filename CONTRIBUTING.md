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

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## Contribution Workflow

### 1. Start a Discussion

Before implementing significant changes:

- **Questions or problems**: Start a [GitHub Discussion](https://github.com/zachtheyek/Aetherscan/discussions) or Slack thread ([#aetherscan-dev](https://breakthroughlisten.slack.com/archives/C0A3CDALQD8))
- **Feature requests**: Open a GitHub Discussion first to gauge interest
- **Bug reports**: Check existing issues, then open a new one with reproduction steps

### 2. Open an Issue

Once maintainers have acknowledged your query:

- Open a GitHub Issue using the appropriate template
- Wait for maintainer approval before starting implementation
- All PRs must be tied to an existing issue

### 3. Create a Feature Branch

Branch naming convention: `category/description`

| Category | Use Case | Example |
|----------|----------|---------|
| `feature/` | New functionality | `feature/db_integration` |
| `hotfix/` | Critical bug fixes | `hotfix/cpu_sampling_rate` |
| `release/` | Release preparation | `release/aetherscan_1.0.0` |
| `misc/` | Documentation, tooling | `misc/plot_improvements` |

```bash
git checkout -b feature/my_new_feature
```

### 4. Implement Changes

- Follow the code conventions in [AGENTS.md](AGENTS.md)
- Write tests for new functionality
- Update documentation if needed
- Keep commits focused and well-described

### 5. Submit a Pull Request

- Ensure your branch is up-to-date with `master` (use `git rebase`, not `git merge`)
- All commits must have verified GPG signatures
- Fill out the PR template completely
- Link the associated issue

### 6. Code Review

- PRs require at least one maintainer approval
- Address review feedback promptly
- Note: PR approvals are voided when new commits are pushed
- Claude will automatically provide a code review when the PR is set to "ready for review"

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
│   ├── inference.py          # Inference pipeline
│   ├── evaluate.py           # Evaluation metrics
│   ├── preprocessing.py      # Data preprocessing
│   ├── data_generation.py    # Synthetic signal injection
│   ├── models/
│   │   ├── __init__.py       # Model exports
│   │   ├── vae.py            # Beta-VAE architecture
│   │   └── random_forest.py  # RF classifier wrapper
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
├── tests/                    # Test suite
│   ├── test_config.py
│   ├── test_data_generation.py
│   └── ...
├── docs/                     # Additional documentation
├── environment.yml           # Conda dependencies
├── pyproject.toml            # Package metadata, ruff config
├── .pre-commit-config.yaml   # Pre-commit hook configuration
├── AGENTS.md                 # AI agent guidelines
├── CONTRIBUTING.md           # This file
├── KNOWN_ISSUES.md           # Known issues and workarounds
├── SECURITY.md               # Security policy
├── LICENSE                   # BSD-3-Clause
└── CITATION.cff              # Citation metadata
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `main.py` | CLI entry point, command routing |
| `cli.py` | Argument parsing, validation, config override |
| `config.py` | All configuration dataclasses and defaults |
| `train.py` | Training orchestration, curriculum learning, checkpointing |
| `inference.py` | Model inference, candidate detection |
| `evaluate.py` | Model evaluation metrics |
| `preprocessing.py` | Data preprocessing, normalization |
| `data_generation.py` | Synthetic signal injection using setigen |
| `models/vae.py` | Beta-VAE architecture with custom clustering loss |
| `models/random_forest.py` | Scikit-learn RF wrapper |
| `db/db.py` | Thread-safe SQLite with async queue-based writes |
| `monitor/monitor.py` | Background resource monitoring (CPU, RAM, GPU) |
| `manager/manager.py` | Resource lifecycle management (pools, shared memory) |
| `logger/` | Multi-handler logging with Slack integration |

---

## Pre-commit Hooks

The project uses pre-commit hooks for code quality:

```yaml
# .pre-commit-config.yaml hooks:
- ruff          # Linting and formatting
- ruff-format   # Code formatting
- gitleaks      # Secret detection
```

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

The project uses Ruff for linting and formatting (see `pyproject.toml`):

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

---

## Testing

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

## Version Updates

When releasing a new version, update these files:

| File | Location | Example |
|------|----------|---------|
| `pyproject.toml` | `version = "X.Y.Z"` | `version = "1.0.0"` |
| `src/aetherscan/__init__.py` | `__version__ = "X.Y.Z"` | `__version__ = "1.0.0"` |
| `CITATION.cff` | `version:` and `date-released:` | `version: 1.0.0` |

---

## Communication

- **Slack**: [#aetherscan-dev](https://breakthroughlisten.slack.com/archives/C0A3CDALQD8) for development discussions
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion

---

## Issue and PR Guidelines

### Issue Authors

- Automatically assigned as issue assignee
- Provide clear reproduction steps for bugs
- Include system information (OS, GPU, CUDA version)

### PR Authors

- Link to the associated issue
- Provide a clear description of changes
- Update tests and documentation
- Respond to review feedback

### Reviewers

- CODEOWNERS are automatically assigned as reviewers
- Focus on correctness, performance, and maintainability
- Be constructive and specific in feedback

---

## Sensitive Data Warning

**DO NOT commit sensitive information.** Pre-commit hooks scan for secrets using gitleaks, but this is not foolproof.

Never commit:
- API keys or tokens (`.env` files)
- Credentials or passwords
- Private data files
- Internal URLs or IP addresses

If you accidentally commit sensitive data, see [SECURITY.md](SECURITY.md) for remediation steps.

---

## Questions?

- Check [README.md](README.md) for project overview
- Check [AGENTS.md](AGENTS.md) for code conventions
- Check [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for known problems
- Open a GitHub Discussion for other questions
