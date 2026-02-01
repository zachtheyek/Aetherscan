# Agent Guidelines for Aetherscan

This document provides guidelines for Claude Code and other AI agents working on the Aetherscan codebase.

---

## Project Overview

**Aetherscan** is a production-grade deep learning pipeline for detecting technosignatures (potential signs of extraterrestrial intelligence) in radio telescope data. It extends the research approach from Ma et al. 2023 into a scalable system for processing the Breakthrough Listen archive.

### Key Characteristics

- **Two-stage ML pipeline**: Beta-VAE for feature extraction + Random Forest for classification
- **Distributed training**: Multi-GPU support via TensorFlow MirroredStrategy
- **Production-focused**: Fault tolerance, monitoring, async database, Slack alerts
- **Domain-specific**: Exploits ON/OFF cadence pattern unique to SETI observations

### Tech Stack

- Python 3.10, TensorFlow 2.16, scikit-learn
- CUDA 12.2, cuDNN 8.9
- SQLite (async writes), Slack SDK
- setigen (synthetic signal injection)

---

## Code Conventions

### Style

- **Formatter**: Ruff with 100-character line length
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with `isort` (first-party: `aetherscan`)
- **Type hints**: Use for function signatures; `from __future__ import annotations` at top

### Naming

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `BetaVAE`, `DataGenerator` |
| Functions | snake_case | `load_train_data`, `run_inference` |
| Constants | UPPER_SNAKE | `_FLUSH_SENTINEL` |
| Private | Leading underscore | `_instance`, `_initialized` |
| Config classes | Suffix with `Config` | `BetaVAEConfig`, `TrainingConfig` |

### Docstrings

- Use triple-quoted docstrings for modules, classes, and public functions
- Include `Args:` and `Returns:` sections for non-trivial functions
- Keep module-level docstrings brief (1-2 sentences describing purpose)

### Comments

- Use `# TODO:` for planned improvements
- Use `# NOTE:` for important implementation details
- Use `# BUG:` for known issues that need investigation
- Reference GitHub issues where applicable: `# BUG: see #12`

---

## Architecture Patterns

### Singleton Pattern

Several core classes use thread-safe singletons to ensure single instances:

```python
class Config:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
```

**Used by**: `Config`, `Database`, `ResourceManager`

**Access pattern**: Use `get_config()`, `get_db()`, etc. rather than direct instantiation.

**Reset pattern**: `Config._reset()` exists for testing only. Never call in production.

### DataHolder Pattern

For memory-efficient data handling with TensorFlow distributed datasets:

```python
class DataHolder:
    def __init__(self, data):
        self._cleared = False
        self._lock = threading.Lock()
        self.data = data

    def clear(self):
        with self._lock:
            if self._cleared:
                return
            self._cleared = True
            self.data = None
```

**Purpose**: Allows explicit memory cleanup while generators hold references.

**Critical**: Always call `holder.clear()` after epoch completion. Generators cache local references, so clearing mid-epoch won't free memory until indices are exhausted.

### Shared Memory Pattern

For inter-process data sharing without copying:

```python
from multiprocessing.shared_memory import SharedMemory

# Create
shm = SharedMemory(create=True, size=data.nbytes)
shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
shm_array[:] = data[:]

# Cleanup
shm.close()
shm.unlink()  # Only call unlink() from creator process
```

**Important**: `shm.unlink()` must be called exactly once, by the creating process. Use `ResourceManager` to track and cleanup.

### Generator Pattern for Distributed Datasets

```python
def data_generator():
    while True:  # Infinite generator for .repeat()
        with holder._lock:
            if holder._cleared:
                return
            data = holder.data  # Cache reference

        indices = np.random.permutation(len(data))
        for idx in indices:
            yield data[idx]

        del data  # Allow GC

dataset = tf.data.Dataset.from_generator(...)
    .batch(global_batch_size)
    .repeat()
    .prefetch(tf.data.AUTOTUNE)
```

---

## Common Tasks

### Adding a New CLI Argument

1. **Add to parser** in `cli.py`:
   ```python
   train_parser.add_argument(
       "--my-new-arg",
       type=int,
       default=None,
       help="Description of argument",
   )
   ```

2. **Add to config** in `config.py`:
   ```python
   @dataclass
   class TrainingConfig:
       my_new_arg: int = 42  # Default value
   ```

3. **Apply in `apply_args_to_config()`**:
   ```python
   if hasattr(args, "my_new_arg") and args.my_new_arg is not None:
       config.training.my_new_arg = args.my_new_arg
   ```

4. **Add to `to_dict()`** for serialization:
   ```python
   "training": {
       ...
       "my_new_arg": self.training.my_new_arg,
   }
   ```

### Adding a New Model Component

1. Create in `src/aetherscan/models/`
2. Export from `models/__init__.py`
3. Follow existing patterns (e.g., `create_beta_vae_model()`)
4. Models should be created within `strategy.scope()` for distributed training

### Database Schema Changes

1. Modify `_create_tables()` in `db/db.py`
2. Add new write method (e.g., `write_new_metric()`)
3. Add corresponding query method
4. Update `to_dict()` if config-related
5. **Note**: SQLite has no built-in migrations. Schema changes require manual migration or fresh DB.

### Adding Resource Monitoring

1. Add collection logic in `monitor/monitor.py`
2. Write to DB using `db.write_system_resource()`
3. Update plotting functions if visualization needed

---

## Testing Guidelines

### Running Tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_config.py

# With coverage
pytest --cov=aetherscan tests/
```

### Test Patterns

- Use `Config._reset()` in teardown to ensure clean state
- Mock GPU operations for CI environments
- Use small sample sizes for unit tests

### What to Test

- Configuration loading and override precedence
- Data generation produces correct shapes
- Model forward passes don't crash
- Database writes are thread-safe
- Cleanup actually frees resources

---

## Known Technical Debt

1. **Hard-coded values in VAE**: Some dimensions (16, 512) are hard-coded rather than using config values. See TODOs in `models/vae.py`.

2. **Inference pipeline incomplete**: `_add_inference_arguments()` and `inference_command()` have TODO markers for finishing implementation.

3. **Validation incomplete**: `validate_args()` has many commented-out validation checks that should be implemented.

4. **Mixed arg naming**: Some CLI args use `num_samples_beta_vae` while config uses `num_samples_beta_vae`. Inconsistencies exist.

5. **No database migrations**: Schema changes require manual intervention or fresh database.

---

## Critical Warnings

### Potential Deadlocks

**DataHolder lock contention**: If TensorFlow prefetch threads are blocked waiting on `DataHolder._lock` while the main thread calls `clear()`, deadlock can occur. Not yet observed in practice, but monitor if adding new locking.

### Config Mutation

**Never mutate config after initialization** in multi-threaded code. The singleton is shared across threads. If you need per-thread values, copy to local variables first.

### Strategy Scope

**All model operations must happen within strategy scope**:

```python
with self.strategy.scope():
    self.encoder = tf.keras.models.load_model(...)
```

Creating models outside scope causes silent failures in distributed training.

### Memory Leaks

**SharedMemory requires explicit cleanup**: Python's GC won't automatically unlink shared memory segments. Use `ResourceManager.cleanup_all()` or ensure `shm.unlink()` is called.

### Database Writer Thread

**Don't block the writer thread**: The async database uses a background thread for writes. If writes are too slow or the queue fills up, you'll lose data. Monitor `write_buffer_max_size`.

---

## Reference Files

| File | Purpose |
|------|---------|
| `config.py` | All configuration options and defaults |
| `train.py` | Training orchestration, main workflow |
| `models/vae.py` | Beta-VAE architecture and custom loss |
| `db/db.py` | Database schema and async writer |
| `manager/manager.py` | Resource lifecycle management |
| `cli.py` | Argument parsing and validation |

---

## Getting Help

- **Code questions**: Check docstrings and comments first
- **Architecture**: See `README.md` for high-level overview
- **Known issues**: Check `KNOWN_ISSUES.md`
- **Contributing**: See `CONTRIBUTING.md`
