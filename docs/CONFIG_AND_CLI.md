# Configuration and CLI Architecture

This document explains how Aetherscan's runtime configuration is structured, how CLI
flags map onto it, and how the `train` and `inference` subcommands stay isolated from
each other. Read this before adding a new flag or config field — the patterns below
exist precisely so that one mode's parameters can't silently contaminate the other.

## TL;DR

| Layer                                                                            | What it does                                                                                                                                                    | Source of truth                                                    |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `src/aetherscan/config.py`                                                       | Defines the `Config` singleton and its sub-dataclasses (`TrainingConfig`, `InferenceConfig`, `GPUConfig`, ...). Holds **every** parameter with a default value. | `Config` instance returned by `get_config()`                       |
| `_add_train_flags_to(parser)` / `_add_inference_flags_to(parser)` (cli.py)       | Register flags onto a given parser. Reused by the main pipeline and by `utils/find_optimal_configs.py`.                                                         | Single source for flag names, types, help text                     |
| `setup_argument_parser()` (cli.py)                                               | Builds the top-level parser with `train` and `inference` subparsers and delegates flag registration to the helpers above.                                       | argparse `subparsers` object                                       |
| `apply_args_to_config(args)` (cli.py)                                            | Mutates the singleton in place with any non-None overrides on `args`. Three patterns (A/B/C) below.                                                             | The singleton after this call                                      |
| `collect_validation_errors(args, num_replicas)` / `validate_args(args)` (cli.py) | Cross-parameter and bounds validation. Returns structured `ValidationError`s; the wrapper raises `ValueError`.                                                  | The pre-apply args (validation runs before `apply_args_to_config`) |
| `train_command()` / `inference_command()` (main.py)                              | Read their own slice of the singleton and run.                                                                                                                  | `config.training.*`, `config.inference.*`                          |

```
sys.argv
   │
   ▼
parse_args()         ── argparse picks one subparser; args.command set
   │
   ▼
validate_args(args)  ── raises ValueError on any failure (no mutation)
   │
   ▼
apply_args_to_config ── writes non-None args onto the Config singleton
   │
   ▼
init_db / setup_gpu_strategy / ...
   │
   ▼
train_command() | inference_command()
```

## The configuration singleton

`Config` is a dataclass-of-dataclasses with double-checked-locking singleton semantics
(see `Config.__new__`, `_lock`, `_initialized`). Every parameter the pipeline reads at
runtime lives somewhere on this object, with a sensible default.

The sub-dataclasses are grouped by **what subsystem owns the parameter**, not by which
subcommand uses it:

| Sub-dataclass        | Owns                                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `DBConfig`           | SQLite writer timeouts, buffer sizes                                                                                               |
| `ManagerConfig`      | Multiprocessing pool sizing                                                                                                        |
| `MonitorConfig`      | Resource-monitor cadence and timeouts                                                                                              |
| `LoggerConfig`       | Console / file / Slack log routing                                                                                                 |
| `BetaVAEConfig`      | Beta-VAE model hyperparameters                                                                                                     |
| `RandomForestConfig` | RF classifier hyperparameters                                                                                                      |
| `GPUConfig`          | TF strategy: replica count, memory growth, NCCL packs, allocator toggles                                                           |
| `DataConfig`         | Data shape, file lists, chunk sizes                                                                                                |
| `TrainingConfig`     | Anything specific to the `train` command — sample counts, batch sizes, LR schedule, curriculum, latent-viz, retries                |
| `InferenceConfig`    | Anything specific to the `inference` command — encoder/RF paths, classification threshold, energy-detection preprocessing, retries |
| `CheckpointConfig`   | Load/save tags, start round                                                                                                        |

Two important consequences:

1. **Every config field always exists**, regardless of which subcommand the user is
   running. `config.inference.classification_threshold` has its dataclass default during a
   train run; `config.training.effective_batch_size` has its default during inference.
2. **Mode separation lives in the dotted namespace** (`config.training.X` vs
   `config.inference.X`). Code that consumes the config is expected to access only its
   own subsection. There is no runtime gating that prevents a developer from typing
   `config.inference.X` inside `train.py` — that's a code-review concern.

## The CLI surface

`setup_argument_parser()` registers two subparsers (`train`, `inference`) via
`add_subparsers(dest="command")`. The chosen subcommand is stored on the parsed
namespace as `args.command`.

Both subparsers are populated by reusable helpers:

```python
def _add_train_arguments(subparsers):
    train_parser = subparsers.add_parser("train", help="Execute training pipeline")
    _add_train_flags_to(train_parser)

def _add_train_flags_to(parser):
    parser.add_argument("--data-path", type=str, default=None, help="...")
    # ... every train-mode flag, registered on `parser` ...


def _add_inference_arguments(subparsers):
    inf_parser = subparsers.add_parser("inference", help="Execute inference pipeline")
    _add_inference_flags_to(inf_parser)

def _add_inference_flags_to(parser):
    parser.add_argument("--data-path", type=str, default=None, help="...")
    # ... every inference-mode flag, registered on `parser` ...
```

The `_add_*_flags_to(parser)` indirection exists so utility scripts (e.g.
`utils/find_optimal_configs.py`) can expose the exact same flag surface against an
arbitrary parser — they import the helper and call it on their own parser without
re-declaring every argument.

### Flag categories

Each flag falls into one of three categories based on where it appears and where it
routes in `apply_args_to_config`. **Pick the right pattern when adding a new flag.**

#### Pattern A — single-subparser flag

The flag is registered in only one of `_add_train_flags_to` or `_add_inference_flags_to`.
When the other subcommand runs, the attribute simply doesn't exist on `args`.

Application: bare `hasattr` + `None` check.

```python
# cli.py — train-only flag, applies to config.training
if hasattr(args, "num_samples_beta_vae") and args.num_samples_beta_vae is not None:
    config.training.num_samples_beta_vae = args.num_samples_beta_vae
```

Examples: most flags (`--num-samples-beta-vae`, `--curriculum-schedule`,
`--encoder-path`, `--overlap-fraction`, ...). The `hasattr` check is what gives the
guarantee — argparse simply doesn't add inference-only attributes to a train namespace
(and vice versa), so `hasattr` is False for any other mode.

#### Pattern B — shared flag, identical destination

The flag is registered in _both_ subparsers (because either mode might want to override
it), and it routes to the _same_ config field.

Application: same `hasattr` + `None` check; no command guard needed.

```python
# cli.py — shared flag, single destination
if hasattr(args, "num_replicas") and args.num_replicas is not None:
    config.gpu.num_replicas = args.num_replicas
```

Current Pattern B flags:

- `--data-path`, `--model-path`, `--output-path` → `config.data_path` / `model_path` / `output_path`
- `--gpu-memory-limit-mb` → `config.gpu.per_gpu_memory_limit_mb`
- `--async-allocator` → `config.gpu.use_async_allocator`
- `--num-replicas` → `config.gpu.num_replicas`
- `--save-tag` → `config.checkpoint.save_tag`

#### Pattern C — shared flag, divergent destination

The flag is registered in _both_ subparsers but the _destination differs_ by mode. The
same `args.X` value would mean different things for train vs inference.

Application: `hasattr` + `None` + an explicit `args.command` discriminator. The flag
appears **twice** in `apply_args_to_config` — once gated on `command == "train"`, once
on `command == "inference"`.

```python
# train route
if (hasattr(args, "per_replica_batch_size")
    and args.per_replica_batch_size is not None
    and getattr(args, "command", None) == "train"):
    config.training.per_replica_batch_size = args.per_replica_batch_size

# inference route (further down the function)
if (hasattr(args, "per_replica_batch_size")
    and args.per_replica_batch_size is not None
    and getattr(args, "command", None) == "inference"):
    config.inference.per_replica_batch_size = args.per_replica_batch_size
```

Current Pattern C flags (each appears twice in `apply_args_to_config`):

- `--per-replica-batch-size` → `config.training.per_replica_batch_size` |
  `config.inference.per_replica_batch_size`
- `--max-retries` → `config.training.max_retries` | `config.inference.max_retries`
- `--retry-delay` → `config.training.retry_delay` | `config.inference.retry_delay`

If you find yourself wanting a Pattern C flag, double-check that the divergent
destinations are semantically distinct (e.g., training and inference have genuinely
different retry semantics). Otherwise consider promoting the field into a shared
sub-dataclass (Pattern B) instead.

## Cross-mode contamination — the four barriers

| #   | Barrier                                                             | What it prevents                                                                                                         |
| --- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 1   | argparse subparsers                                                 | Flags unique to the unchosen mode never appear as attributes on `args`                                                   |
| 2   | `hasattr(args, "X")` guards in `apply_args_to_config`               | Even if a future refactor changes which mode owns a flag, an absent attribute is a no-op rather than an `AttributeError` |
| 3   | `getattr(args, "command", None) == "train" \| "inference"`          | The three Pattern C flags route to the correct config section based on mode                                              |
| 4   | Dataclass namespacing (`config.training.X` vs `config.inference.X`) | Pipeline code physically can't read an inference value as a training one — they have different fully-qualified names     |

Each layer addresses a different failure mode. Together they make cross-mode
contamination essentially impossible without an explicit code mistake (e.g., a
developer accidentally writing `config.inference.X` inside `train.py`).

## The validation layer

`validate_args` runs **before** `apply_args_to_config`. It reads `args` and merges with
config defaults via the `_resolve(args, arg_name, default)` helper, so checks see the
effective value the pipeline would use even when a flag is omitted.

Three things to know:

1. **`collect_validation_errors(args, num_replicas)`** returns a list of structured
   `ValidationError` objects (with `field`, `current`, `message`, `fix_kind`, ...). The
   `validate_args` wrapper turns the list into a `ValueError`, and uses the proposer
   surface colocated in `cli.py` (`propose_simple_fix`, `_solve_cross_param_constraints`)
   to append a `Suggested fixes:` block to the message. The standalone
   [`utils/find_optimal_configs.py`](#diagnosing-config-issues-with-find_optimal_configspy)
   script exercises the same proposer for ad-hoc "what config would work on N GPUs?"
   queries.

2. **`num_replicas` is determined ahead of time** by `_detect_num_replicas(args)`:
   `args.num_replicas` if the user passed `--num-replicas`, else
   `tf.config.list_physical_devices('GPU')` count if TF reports at least one GPU, else
   `None`. When the value is `None`, `collect_validation_errors` logs a warning and
   skips the cross-replica divisibility section — the runtime will fail later in
   `setup_gpu_strategy` if it really needed GPUs. This lets utility scripts
   (e.g. `find_optimal_configs.py` on a dev box without TF) still produce the
   non-cross-replica part of the report.

3. **The same mode-gating pattern applies inside validation**:

   ```python
   if cmd == "train":
       # ... train-mode checks ...

   if cmd == "inference":
       # ... inference-mode checks ...
   ```

## Runtime dispatch (main.py)

After `validate_args` + `apply_args_to_config` succeed, `main()` branches on
`args.command`:

```python
if args.command == "train":
    train_command()       # reads config.training, config.beta_vae, config.checkpoint, ...
elif args.command == "inference":
    inference_command()   # reads config.inference, config.checkpoint, ...
```

Both functions also read shared sections (`config.gpu`, `config.data`, `config.db`,
`config.manager`, ...). The pattern is consistently dotted access — no `if mode ==
"train"` inside the pipeline body.

## Adding a new flag — checklist

When adding `--my-new-flag`:

1. **Add the config field**, with a default, on the appropriate sub-dataclass in
   `config.py`. Update `Config.to_dict` so the field appears in serialized configs.
2. **Register the flag** on the right helper(s) in `cli.py`:
   - Used only by training → `_add_train_flags_to` (Pattern A)
   - Used only by inference → `_add_inference_flags_to` (Pattern A)
   - Used by both, same destination → both helpers (Pattern B)
   - Used by both, different destinations → both helpers (Pattern C)
3. **Wire `apply_args_to_config`** using the correct pattern:
   - Pattern A or B: one `hasattr + None` block
   - Pattern C: two blocks, each gated on `args.command`
4. **Add validation** in `collect_validation_errors` if the new field has bounds,
   format, or cross-parameter constraints. Use `_resolve` so the check sees the merged
   args+default value. Emit a `ValidationError` with an appropriate `fix_kind` so
   `find_optimal_configs.py` can propose a fix.
5. **Read the field** in the pipeline via `config.<section>.my_new_field`.

> [!NOTE]
> Once your PR merges to `master`, the [`claude-update-docs`](../.github/workflows/claude-update-docs.yml) action picks up CLI changes and regenerates the **CLI Reference** section in `README.md` by running `PYTHONPATH=src python utils/print_cli_help.py all`. You don't need to hand-edit the help blocks.

## Diagnosing config issues with `find_optimal_configs.py`

[`utils/find_optimal_configs.py`](../utils/find_optimal_configs.py) is a standalone
diagnostic that validates the singleton config (defaults plus CLI overrides) against
`collect_validation_errors` and proposes a coordinated fix for each violation. The
proposer surface itself lives in `cli.py` (`propose_simple_fix`,
`_solve_cross_param_constraints`, `_check_cross_constraints`), so the script and
`validate_args` share the same suggestion logic — the script is just the CLI/printing
wrapper around it.

**When to reach for it**:

- `validate_args` rejected your config and the inline `Suggested fixes:` block isn't
  enough — `find_optimal_configs.py` prints the full violation list, the grid-search
  delta table for cross-replica patches, and every proposed CLI flag.
- You're exploring ahead of time: "what would work on 4 _and_ 6 GPUs?" Pass
  `--num-gpus 4,6` and the cross-replica solver finds a six-tuple of batch/sample
  parameters that satisfies the constraints under both replica counts simultaneously.
- You're on a dev box without TensorFlow. The script never touches TF — cross-replica
  divisibility checks run against the `--num-gpus` list directly.

**Why it exists**: many cross-replica violations (e.g. `effective_batch_size` divisible
by `per_replica_batch_size * num_replicas`) can't be fixed by clamping one field — the
six interdependent batch/sample params have to move together. The bounded grid search
minimizes L1 distance to the current values so suggestions stay close to what the user
asked for.

Examples (from the script's docstring):

```bash
# Check default config against a 4-GPU setup
python utils/find_optimal_configs.py --num-gpus 4 train

# Override --effective-batch-size and search for a fix valid on both 4 and 6 GPUs
python utils/find_optimal_configs.py --num-gpus 4,6 train --effective-batch-size 3072

# Check inference defaults with an invalid --overlap-fraction
python utils/find_optimal_configs.py inference --overlap-fraction 1.5
```

## Auditing for drift

Two quick checks to verify no pattern violations have crept in:

```bash
# Every "command == ..." guard must come in matched train+inference pairs (= Pattern C flag).
grep -nE 'command", None\) == "(train|inference)"' src/aetherscan/cli.py

# Every shared flag (appearing in both helpers) is either Pattern B (single
# apply_args block) or Pattern C (two apply_args blocks with command guards).
awk '
/^def _add_train_flags_to/    {mode="TRAIN"; next}
/^def _add_inference_flags_to/{mode="INFER"; next}
/^def [a-zA-Z_]+/             {mode=""; next}
mode && /^ *"--/{
    match($0, /"--[a-z0-9-]+"/)
    if (RSTART) print substr($0, RSTART, RLENGTH)
}
' src/aetherscan/cli.py | sort | uniq -c | awk '$1 == 2 { print $2 }'
```

The second command lists every shared flag (those appearing in both subparsers); each
should match either a single Pattern B `apply_args` block or a pair of Pattern C blocks
with a `command` discriminator. As of this writing it yields 10 shared flags —
7 Pattern B + 3 Pattern C — matching the tables above.
