# AGENTS.md

## Overview

Aetherscan: Deep learning pipeline for SETI technosignature detection. Two-stage ML (Beta-VAE → Random Forest) with multi-GPU distributed training/inference.

---

## Commands

```bash
# Setup
conda env create -f environment.yml && conda activate aetherscan

# Train
PYTHONPATH=src python -m aetherscan.main train --save-tag final_v1

# Inference
PYTHONPATH=src python -m aetherscan.main inference --encoder-path /path/to/encoder --rf-path /path/to/rf

# Test
pytest tests/ -v

# Lint
ruff check src/ && ruff format src/
```

---

## Code Style

**Formatter**: ruff (100-char lines, see pyproject.toml)

| Element       | Convention  | Example                  |
| ------------- | ----------- | ------------------------ |
| Classes       | PascalCase  | `DataGenerator`          |
| Functions     | snake_case  | `run_training_pipeline`  |
| Constants     | UPPER_SNAKE | `MAX_RETRIES`            |
| Private       | \_prefix    | `_init_worker`           |
| Config fields | snake_case  | `per_replica_batch_size` |

---

## Key Patterns

- **Distributed training/inference**: All TensorFlow model ops must occur within `strategy.scope()`
- **Thread-safe singletons**: `Config`, `Database`, `ResourceManager` — always use `get_config()`, `get_db()`, `get_manager()` accessors
- **Shared memory**: Inter-process data via `manager.create_shared_memory()` — cleanup handled by ResourceManager
- **DataHolder**: Wraps data with lock for memory-safe cleanup

---

## Boundaries

### Always

- Create models inside `strategy.scope()`
- Use accessor functions (`get_config()`, `get_db()`, `get_manager()`)
- Register pools/shared memory with ResourceManager
- Call `holder.clear()` after processing completes

### Ask First

- Modifying config dataclass fields
- Adding new singleton patterns
- Changing cleanup order in ResourceManager
- Altering checkpoint/resume logic

### Never

- Mutate config after initialization in multi-threaded code
- Call `shm.unlink()` from worker processes (only creator)
- Log inside SIGTERM handlers (causes deadlock)
- Create TensorFlow models outside `strategy.scope()`

---

## Common Modifications

**Adding CLI arguments**:

1. Add to parser in `cli.py`
2. Add field to config dataclass in `config.py`
3. Apply in `apply_args_to_config()`
4. Add to `to_dict()` for serialization
5. Update README.md CLI Reference section

**Adding model components**:

1. Create in `src/aetherscan/models/`
2. Export from `models/__init__.py`
3. Instantiate within `strategy.scope()`

**Database schema changes**:

1. Modify `_create_tables()` in `db/db.py`
2. Add write/query methods
3. No migrations — requires manual migration or fresh DB

---

## Key Files

| File                 | Purpose                                        |
| -------------------- | ---------------------------------------------- |
| `main.py`            | Entry point, orchestration, GPU strategy setup |
| `cli.py`             | Argument parsing, validation                   |
| `config.py`          | Config singleton & dataclasses                 |
| `train.py`           | Training pipeline orchestration                |
| `models/vae.py`      | Beta-VAE with clustering loss                  |
| `manager/manager.py` | Resource lifecycle management                  |
| `db/db.py`           | Async SQLite writer                            |

---

## Pull Request Guidelines

PRs must have an associated issue. If an issue doesn't yet exist for your PR, create the issue first, then link it to the PR with `Closes #N` or `Fixes #N` in the PR body. Do NOT rely on simply mentioning `#N` in the PR body

---

## References

- [README.md](README.md) — Installation, usage examples, CLI reference
- [CONTRIBUTING.md](CONTRIBUTING.md) — Git workflow, PR guidelines
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) — Known bugs and workarounds
