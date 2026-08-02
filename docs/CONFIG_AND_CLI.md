# Configuration and CLI Architecture

This document explains how Aetherscan's runtime configuration is structured, how CLI
flags map onto it, and how the `train` and `inference` subcommands stay isolated from
each other. Read this before adding a new flag or config field — the patterns below
exist precisely so that one mode's parameters can't silently contaminate the other.
For where config initialization sits in the overall startup sequence (and the singleton
pattern's rules), see [`ARCHITECTURE.md`](ARCHITECTURE.md); for what the individual
training/inference fields *do*, see [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md) and
[`INFERENCE_PIPELINE.md`](INFERENCE_PIPELINE.md).

## TL;DR

| Layer                                                                                    | What it does                                                                                                                                                    |
| ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/aetherscan/config.py`                                                               | Defines the `Config` singleton and its sub-dataclasses (`TrainingConfig`, `InferenceConfig`, `GPUConfig`, ...). Holds **every** parameter with a default value. |
| `src/aetherscan/cli.py:setup_argument_parser()`                                          | Builds the top-level parser with `train` and `inference` subparsers. Delegates flag registration to the helpers below.                                          |
| `src/aetherscan/cli.py:_add_train_flags_to(parser)` / `:_add_inference_flags_to(parser)` | Register flags onto a given parser. Reused by the main pipeline and by `utils/find_optimal_configs.py`.                                                         |
| `src/aetherscan/cli.py:validate_args(args)`                                              | Cross-parameter and bounds validation. Returns structured `ValidationError`s; the wrapper raises `ValueError`.                                                  |
| `src/aetherscan/cli.py:apply_args_to_config(args)`                                       | Mutates the singleton in place with any non-None overrides on `args`. Three patterns (A/B/C) below.                                                             |
| `src/aetherscan/main.py:train_command()` / `:inference_command()`                        | Read their own slice of the singleton and run the main pipeline.                                                                                                |

Startup flow:

```
  ...
   │
   ▼
sys.argv
   │
   ▼
setup_argument_parser()            ── calls helpers for subcommand flag registration
   │
   ▼
parse_args()                       ── argparse picks the specified subparser; args.command set
   │
   ▼
apply_saved_config(config_path)    ── only runs on args.command == inference and args.config_path
   │                                  is not None:
   │                                  layers the saved JSON's allowlisted model-contract fields
   │                                  onto the singleton defaults (#303); every other saved field
   │                                  is ignored (logged when it differs from the resolved value)
   │
   ▼
validate_args(args)                ── raises ValueError on any failure (no mutation)
   │
   ▼
apply_args_to_config(args)         ── writes non-None args onto the Config singleton
   │                                  (CLI overrides saved config)
   │
   ▼
  ...
   │
   ▼
train_command() / inference_command()
```

Priority order, for the fields a saved config is allowed to layer:

```
runtime defaults  <  loaded config  <  CLI args
```

`apply_saved_config` is **allowlist-only** (#303): a saved training config layers exactly
the model-contract / result-affecting fields — the whole `beta_vae` section, the RF's
deployed-representation contract (`latent_variant` / `active_dims` / `calibration_active` /
`calibration_method` — not `n_jobs`, not seeds), the data geometry keys, and precisely the
inference fields the resume fingerprints hash minus the artifact paths. Everything else —
paths (in nested or legacy flat form), `gpu`, `manager`, `db` / `logger` / `monitor`, `hf`,
`training`, `checkpoint`, and `reproducibility` — always comes from env + defaults + CLI,
and every ignored saved field whose value differs from the resolved one is logged as a
startup diff line, so nothing disappears silently. The old inverse design (apply everything
minus a skip-list) left every host-scoped field unsafe by default — the live footgun being
`manager.n_processes`: a config saved on a 96-core host silently 3×-oversubscribed a
32-core host's worker pool, with no CLI rescue before `--n-processes` existed. The
inference and data sub-allowlists are **derived from `run_state.py`'s fingerprint key
sets** and pinned by a drift test, so `cli.py` and `run_state.py` can never silently
disagree about which inference/data fields are result-affecting; the `rf` field allowlist
and the `beta_vae` section allowlist are literal (`rf` is in no fingerprint) and pinned by
their own equality test. One trap this design does **not** remove: a *new* field added to
`InferenceConfig` defaults **into** the allowlist (it is `fields(InferenceConfig)` minus the
exclude set) and — once mirrored into `to_dict` — into both fingerprints, so a new
host-tuning knob on `InferenceConfig` must be added to the run_state denylists explicitly,
exactly as `prune_stamps`/`inference_viz_scope` were. Only a field added to a
non-allowlisted section (`manager`, `gpu`, `db`, …) is ignored by default.

## The configuration singleton

`Config` is a dataclass-of-dataclasses with double-checked-locking singleton semantics
(see `Config.__new__`, `_lock`, `_initialized`). Every parameter the pipeline reads at
runtime lives somewhere on this object, with a sensible default baked in.

The sub-dataclasses are grouped by **what subsystem owns the parameter**, not by which
subcommand uses it:

| Sub-dataclass        | Owns                                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `DBConfig`           | SQLite writer timeouts, buffer sizes                                                                                               |
| `ManagerConfig`      | Multiprocessing pool sizing                                                                                                        |
| `MonitorConfig`      | Resource-monitor cadence and timeouts, stage-band plot annotation toggle, live-dashboard enable (`dashboard_enabled`) and port (`dashboard_port`) |
| `LoggerConfig`       | Console / file / Slack log routing                                                                                                 |
| `ReproducibilityConfig` | The pipeline root seed (`seed`, default 11 — every random stream in **both** modes derives from it; see `seeding.py`) and `tf_deterministic_ops` (default **on** since #298; opt out with `--no-tf-deterministic-ops`, unseed with `--unseeded`) |
| `BetaVAEConfig`      | Beta-VAE model hyperparameters, plus the A/B-gated `mixed_precision` (bf16) opt-in (default off)                                   |
| `RandomForestConfig` | RF classifier hyperparameters, the #282 latent-variant sweep/selection/calibration knobs, and the override-only `seed` (default `None` = derived from the root seed) |
| `GPUConfig`          | TF strategy: replica count, memory growth, NCCL packs, allocator toggles, GPU thread mode (`gpu_thread_mode` / `gpu_thread_count`) |
| `DataConfig`         | Data shape, file lists, chunk sizes                                                                                                |
| `TrainingConfig`     | Anything specific to the `train` command — sample counts, batch sizes, LR schedule, curriculum, round-data layout/dtype, latent-viz, retries |
| `InferenceConfig`    | Anything specific to the `inference` command — encoder/RF paths, classification threshold, energy-detection preprocessing, retries |
| `HFConfig`           | HuggingFace Hub integration — target repo id, opt-in post-training upload, inference revision pin                                  |
| `CheckpointConfig`   | Load/save tags, start round, tag-collision-guard override                                                                          |

Two important consequences:

1. **Every config field always exists**, regardless of which subcommand the user is
   running. `config.inference.classification_threshold` has its dataclass default during a
   train run; `config.training.effective_batch_size` has its default during inference.
2. **Mode separation lives in the dotted namespace** (`config.training.X` vs
   `config.inference.X`). Code that consumes the config is expected to access only its
   own subsection. There is no runtime gating that prevents a developer from typing
   `config.inference.X` inside `train.py` — that's a code-review concern.

Not every config field has a matching CLI flag — most tuning knobs are config-only and
are changed by editing their default in `config.py`. One worth knowing:
`training.min_val_auc` (default `0.0` = disabled) is an opt-in quality floor on the
Random Forest's validation ROC-AUC. When set and unmet after the RF fit, training logs a
loud WARNING (which reaches the Slack summary) rather than failing the run, so a run
that "completes" but learned nothing is caught before its model is promoted.

Four config-only fields from the 2026-07 training-performance pass follow the same
pattern (no CLI flags): `gpu.gpu_thread_mode` (default `"gpu_private"`; also
`"global"`/`"gpu_shared"`) and `gpu.gpu_thread_count` (default 2) set
`TF_GPU_THREAD_MODE`/`TF_GPU_THREAD_COUNT` in `setup_gpu_strategy` before the GPU
runtime initializes; `training.round_array_dtype` (default `"float16"` since
2026-07-29 — passed its val-metric A/B gate; `"float32"` restores the historical input
numerics byte-for-byte) and `beta_vae.mixed_precision` (default `False` — failed its
7-seed gate on a reproducible seed-13 pathology; do not enable without new evidence) are
**A/B-gated numerics levers** — the gates and verdicts are documented in
[`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md#performance-engineering-the-276-follow-up-july-2026). All four are emitted by `Config.to_dict()` and so are part
of the persisted config snapshot.

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
re-declaring every argument. Both helpers additionally call
`_add_reproducibility_flags_to(parser)` and `_add_runtime_flags_to(parser)` first,
registering the shared `--seed` / `--tf-deterministic-ops` (#279) and `--n-processes`
(#303) flags through one function each so the train and inference surfaces cannot drift.

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
`--encoder-path`, `--screening-threshold`, `--mc-draws`, `--overlap-fraction`,
`--prune-stamps`, `--inference-viz-scope`, ...). The
`hasattr` check is what gives the
guarantee — argparse simply doesn't add inference-only attributes to a train namespace
(and vice versa), so `hasattr` is False for any other mode.

One Pattern A flag is a **deprecated alias**: `--rf-seed` (train-only) still routes to
`config.rf.seed` as an explicit override, but the RF seed normally derives from the shared
root `--seed` (#279) — using the alias logs a deprecation warning.

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
- `--hf-repo-id` → `config.hf.repo_id`
- `--force-tag` → `config.checkpoint.force_tag`
- `--dashboard` / `--no-dashboard` (`BooleanOptionalAction`) → `config.monitor.dashboard_enabled`
- `--dashboard-port` → `config.monitor.dashboard_port`
- `--benchmark-report` / `--no-benchmark-report` (`BooleanOptionalAction`) → `config.monitor.benchmark_report_enabled`
- `--seed` / `--unseeded` → `config.reproducibility.seed` (registered via `_add_reproducibility_flags_to`; `--unseeded` sets it to None, mutually exclusive with `--seed`)
- `--tf-deterministic-ops` (`BooleanOptionalAction`) → `config.reproducibility.tf_deterministic_ops` (same helper)
- `--n-processes` → `config.manager.n_processes` (registered via the shared `_add_runtime_flags_to` helper, #303; validated `>= 1` — the operator override for worker-pool sizing, which is never layered from a saved `--config-path`)

> [!NOTE]
> `--save-tag` looks like a Pattern B flag — it's registered in **both** subparsers, so
> the drift check below counts it as shared — but it is the one shared flag **not** wired
> through `apply_args_to_config`. Instead of copying `args.save_tag` onto a config field,
> `main()` resolves it once via `resolve_save_tag` to `{prefix}_{YYYYMMDD_HHMMSS}` (the
> datetime stamped a single time, before `init_logger`, so the log file, config, and
> artifacts all share one timestamp). Its neighbor `--force-tag`, by contrast, **is** a
> genuine Pattern B `apply_args_to_config` block (→ `config.checkpoint.force_tag`).

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
developer accidentally writing `config.inference.X` instead of `config.training.X`).

## The validation layer

`validate_args` runs **before** `apply_args_to_config`. It reads `args` and merges with
config defaults via the `_resolve(args, arg_name, default)` helper, so checks see the
effective value the pipeline would use even when a flag is omitted.

In `inference` mode, `apply_saved_config(args.config_path)` runs immediately
after `parse_args` and before `validate_args` — so by the time validation kicks in,
the singleton's defaults at runtime have already been overridden by the saved JSON's
allowlisted fields (#303 — see the layering rules above), allowing `_resolve` to return
the correct value.

Three things to know:

1. **`collect_validation_errors(args, num_replicas)`** returns a list of structured
   `ValidationError` objects (with `field`, `current`, `message`, `fix_kind`, ...). The
   `validate_args` wrapper turns the list into a single `ValueError`, and uses the
   proposer surface colocated in `cli.py` (`propose_simple_fix`,
   `_solve_cross_param_constraints`, `_check_cross_constraints`) to append a
   `Suggested fixes:` block to the error message. The standalone
   [`utils/find_optimal_configs.py`](#diagnosing-config-issues-with-find_optimal_configspy) script exercises the same proposer for ad-hoc
   "what config would work on N GPUs?" queries. The cross-parameter checks include, since
   #297, the RF train-split floor —
   `num_samples_rf * train_val_split >= effective_batch_size`, a `>=` floor only
   (deliberately not divisibility: the runtime split trims to a multiple of the effective
   batch, and the defaults rely on the trim) — because below one full batch the split
   trims to zero and the run dies at `rf_train` *after* the beta-VAE rounds already
   trained; the same floor is mirrored in `_check_cross_constraints` so the suggested-fix
   solver can never propose a config that dies there.

2. **`num_replicas` is resolved ahead of time** by `_resolve_num_replicas(args)`, in
   priority order: `args.num_replicas` if the user passed `--num-replicas`, else
   `config.gpu.num_replicas` if set on the singleton, else the count returned from
   `tf.config.list_physical_devices('GPU')` if TF reports at least one GPU. When
   the replica count comes from an explicit request (the flag or config), the
   function fails fast with a `ValueError` if the count is `< 1` (checked first,
   without importing TF) or — when TF can confirm the hardware — exceeds the
   host's GPU count, naming whichever knob initially supplied the count, so a bad
   value hard-stops before the cross-replica divisibility checks consume it as
   a divisor. If TF is unavailable or reports zero GPUs, `_resolve_num_replicas`
   returns `None` regardless of whether a replica count was requested — the upper
   bound can't be confirmed, so that check is deferred to `setup_gpu_strategy`.
   When the return value is `None`, `collect_validation_errors` logs a warning
   and skips the cross-replica divisibility checks — the runtime will fail later
   in `setup_gpu_strategy` if GPUs were required. This lets utility scripts (e.g.
   `find_optimal_configs.py` on a dev box without TF) still produce the
   non-cross-replica part of the report.

3. **The same mode-gating pattern applies during validation**:

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
   - Used by both, same destination → both helpers (Pattern B); if the two registrations
     must stay word-for-word identical, register once in a shared helper called by both —
     the `_add_reproducibility_flags_to` precedent
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
six interdependent batch/sample params have to move together. The solver enumerates the
two purely divisibility-bound fields (`effective_batch_size`, `per_replica_val_batch_size`)
exactly from the structure — divisors of the train split, and divisors of the gcd of the
val-side counts (val split, `num_samples_rf`, and `latent_viz_num_cadences_per_type * 4`)
— while the data sizes and per-replica batch are only searched over small neighborhoods
around the current values. Every candidate is confirmed by the latent-aware
`_check_cross_constraints`, and the returned config is the L1-nearest satisfying
combination. `_check_cross_constraints` now accepts an optional `latent_total` argument
(callers in `_build_suggestion_block` and `find_optimal_configs.py` pass
`config.training.latent_viz_num_cadences_per_type * 4`), so the latent-viz divisibility
check is part of the solver's guarantee, not only of `collect_validation_errors`.

Examples (from the script's docstring):

```bash
# Check default config against a 4-GPU setup
python utils/find_optimal_configs.py --num-gpus 4 train

# An override that's invalid for 5 GPUs — the solver proposes the nearest valid combination
python utils/find_optimal_configs.py --num-gpus 5 train --effective-batch-size 3072

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
with a `command` discriminator — with one deliberate exception, `--save-tag`, which is
registered in both subparsers but is resolved in `main()` via `resolve_save_tag` rather
than in `apply_args_to_config` (see the Pattern B note above). Note the awk only sees
flags registered lexically inside the two `_add_*_flags_to` bodies — `--seed` /
`--tf-deterministic-ops` / `--n-processes` live in the shared
`_add_reproducibility_flags_to` / `_add_runtime_flags_to` helpers (called by both) and
must be counted by hand. As of this writing the command yields 15 flags; with the three
helper-registered ones the shared surface is 18 — 14 Pattern B + 3 Pattern C + `--save-tag`
(the special case above).
