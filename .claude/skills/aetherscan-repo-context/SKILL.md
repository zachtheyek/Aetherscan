---
name: aetherscan-repo-context
description: Deep-dive context for working inside the Aetherscan repo — Breakthrough Listen's deep-learning SETI pipeline (Beta-VAE → Random Forest, multi-GPU). Use when developing, debugging, configuring, running, or reviewing changes to this codebase, or when answering questions about its install paths, CLI, config system, conventions, contribution workflow, or security practices.
---

# Working in the Aetherscan Repo

Aetherscan is Breakthrough Listen's first end-to-end production-grade deep-learning pipeline for SETI at scale. It detects anomalies in radio spectrograms with technosignature-like characteristics by pairing a **beta-VAE** (dimensionality reduction / feature extraction) with a **Random Forest** ensemble (candidate detection). It is based on [Ma et al. 2023](https://arxiv.org/abs/2301.12670) and runs single-node data-parallel distributed training/inference.

> **Scope.** `CLAUDE.md` (repo root) holds the lean, always-on rules; this skill is the on-demand deep-dive — read it when a task needs more than the essentials. The authoritative sources are `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `KNOWN_ISSUES.md`, and `docs/`; when this file disagrees with them, they win and this file should be updated. **All paths below are relative to the repository root** (an agent's working directory).

---

## Entry Point & How to Run

`src/aetherscan/main.py` is the **only** designated entry point. Non-development workflows should never call other scripts/modules directly. It dispatches to one of two subcommands via the first positional argument: `train` or `inference`.

There are two install paths off the same source tree:

| Path                                        | Status                                                     | When                          | Launcher                                                                    |
| ------------------------------------------- | ---------------------------------------------------------- | ----------------------------- | --------------------------------------------------------------------------- |
| **NGC container** (Apptainer/SingularityCE) | Canonical; runs on both clusters; only option on Blackwell | Default                       | `./utils/run_container.sh python -m aetherscan.main {train\|inference} ...` |
| **Conda env**                               | Alternative; **Ampere only**                               | When containers aren't usable | `PYTHONPATH=src python -m aetherscan.main {train\|inference} ...`           |

CLI flags are identical between the two paths; only the launcher differs. `PYTHONPATH=src` makes the `aetherscan` package importable from `src/` without a `pip install -e .`; the container sets `PYTHONPATH` automatically.

- **Container build:** `singularity build aetherscan-ngc25.02.sif aetherscan.def` (or `apptainer build ...`) — same `aetherscan.def` recipe builds with either runtime. Build on the cluster you intend to run on.
- **Conda env:** `conda env create -f environment.yml && conda activate aetherscan`
- **`utils/run_container.sh`** auto-detects apptainer vs singularity (Apptainer wins when both present), sets `--nv` for GPU passthrough, auto-loads `<repo>/.env`, and bind-mounts the repo + `AETHERSCAN_{DATA,MODEL,OUTPUT}_PATH` 1:1 so absolute paths persisted in the DB stay valid across host and container.
- **`utils/start_tmux_session.sh`** (optional) spins up a four-window monitoring tmux session (htop/CPU-MEM, `nvidia-smi`, `/dev/shm`, `tree` of models/outputs). Idempotent.
- **`utils/kill_pipeline.sh`** stops a running pipeline (main process + all worker children) from a separate shell on the same machine — works for both run modes, finds the process tree itself, and tries a graceful SIGTERM (lets `ResourceManager` close pools/SHM) before escalating to SIGKILL. Assumes a single running instance. `--force` / `--dry-run` / `--timeout N`.

**Common invocations:**

```bash
# Train (container, canonical)
./utils/run_container.sh python -m aetherscan.main train --save-tag final_v1

# Train (conda source, Ampere)
PYTHONPATH=src python -m aetherscan.main train --save-tag final_v1

# Inference from a pre-processed .npy
./utils/run_container.sh python -m aetherscan.main inference \
    --test-files real_filtered_LARGE_test_HIP15638.npy \
    --encoder-path /path/to/vae_encoder.keras \
    --rf-path /path/to/random_forest.joblib \
    --config-path /path/to/config.json

# Inference from raw .h5 (triggers energy-detection preprocessing)
./utils/run_container.sh python -m aetherscan.main inference \
    --inference-files complete_cadences_catalog.csv \
    --encoder-path /path/to/vae_encoder.keras \
    --rf-path /path/to/random_forest.joblib \
    --config-path /path/to/config.json
```

`--inference-files` (raw `.h5` catalog) triggers the energy-detection preprocessing pipeline and takes precedence over `--test-files` (pre-processed `.npy`).

---

## Configuration & CLI

Hierarchical, dataclass-based config with a thread-safe singleton. Resolution order at command time:

1. **Defaults** — in `src/aetherscan/config.py`
2. **Environment variables** — for paths and secrets
3. **CLI flags** — override both on startup

At runtime, the singleton `Config` is read via `get_config()` and may be modified programmatically. See `docs/CONFIG_AND_CLI.md`.

**Secrets & paths** come from a `.env` file at the repo root (gitignored). Shell `export` takes precedence over `.env`. The container wrapper forwards only `SLACK_*` and `AETHERSCAN_*` via `--env`; the source path loads the **whole** `.env` into `os.environ` at the top of `main.py` via python-dotenv.

```ini
# .env example (Slack integration auto-disables if unset)
SLACK_BOT_TOKEN=your-slack-bot-token
SLACK_CHANNEL=your-slack-channel
# Defaults to /datax/scratch/zachy/{data|models|outputs}/aetherscan; CLI flags override
AETHERSCAN_DATA_PATH=/path/to/data
AETHERSCAN_MODEL_PATH=/path/to/models
AETHERSCAN_OUTPUT_PATH=/path/to/outputs
```

**The CLI Reference in `README.md` is a tight source↔doc contract.** The three code blocks under `## CLI Reference` (Top-Level / Train / Inference Help) are pasted-verbatim argparse output. If `src/aetherscan/cli.py` changes (flags, help strings, subparsers), regenerate them from the repo root with:

```bash
PYTHONPATH=src python utils/print_cli_help.py all
```

`print_cli_help.py` imports only `aetherscan.config` and `aetherscan.cli` (pure stdlib, no TensorFlow/conda needed) and pins `COLUMNS=80` for deterministic wrapping. Replace each block verbatim, preserving each subsection's "Regenerate this output with ..." intro paragraph.

---

## Project Structure

```
src/aetherscan/
├── main.py              # Entry point, command dispatch, GPU strategy setup (NCCL + fallback)
├── cli.py               # Argument parsing, validation, config override
├── config.py            # Configuration dataclasses & defaults (singleton)
├── train.py             # Training orchestration, curriculum learning, checkpointing
├── inference.py         # Inference orchestration, candidate detection
├── preprocessing.py     # Loading / downsampling / log-normalization + energy detection
├── data_generation.py   # Synthetic signal injection (setigen)
├── models/{vae,random_forest}.py
├── db/db.py             # Thread-safe SQLite, async queue-based writes
├── logger/              # Multi-handler logging + Slack integration
├── monitor/monitor.py   # Background resource monitoring (CPU, RAM, GPU)
└── manager/manager.py   # Resource lifecycle management (pools, shared memory)
utils/                   # run_container.sh, kill_pipeline.sh, start_tmux_session.sh,
                         # print_cli_help.py, find_optimal_configs.py,
                         # verify_train_test_files.py, get_system_info.sh
docs/                    # BLACKWELL_MIGRATION.md, CONFIG_AND_CLI.md, README.md, assets/
tests/                   # Placeholder — no test suite yet
```

---

## Architecture Patterns (load-bearing)

- **Distributed training/inference** — Gradients sync via TF `MirroredStrategy` + NCCL AllReduce, with gradient accumulation for larger effective batches under low VRAM. All TensorFlow model ops **must** occur within `strategy.scope()`.
- **Cadence-aware composite loss** — beta-VAE reconstruction + β-weighted KL divergence + α-weighted true/false clustering (ON-ON / OFF-OFF proximity, ON-OFF separation for true signals; uniform for false).
- **Curriculum training** — progressive SNR difficulty with adaptive LR that decays on validation plateaus and resets each round; per-round checkpointing + automatic retry with backoff.
- **Thread-safe singletons** — `Config`, `Database`, `ResourceManager`. Always use the accessors `get_config()`, `get_db()`, `get_manager()`; never instantiate directly.
- **Shared-memory zero-copy parallelism** — worker pools communicate via shared memory (no serialization). Allocate via `manager.create_shared_memory()`; ResourceManager owns cleanup. Only the **creator** may call `shm.unlink()`, never workers.
- **Data holders** — `TrainDataHolder` / `VizDataHolder` (`train.py`) and `InfDataHolder` (`inference.py`) wrap batches with a lock. RF training reuses `TrainDataHolder` via `prepare_distributed_train_dataset`. Call `holder.clear()` after processing completes.
- **Worker cleanup** — custom SIGTERM handlers free resources on interruption. **Never log inside SIGTERM handlers** (deadlock risk).

---

## Code Style & Conventions

Enforced by **ruff** (lint + format) via pre-commit; full config in `pyproject.toml` under `[tool.ruff]`.

- **Target**: Python 3.10 (lowest common denominator across the conda 3.10 and container 3.12 paths). **Line length 100** (formatter wraps; `E501` is intentionally ignored).
- **Modern typing** — every module starts with `from __future__ import annotations` (isort's `required-imports`/`I002` auto-inserts it). Use PEP 604 unions (`X | None`, not `Optional[X]`) and PEP 585 generics (`list[int]`, `dict[str, float]`, not `typing.List`/`Dict`). Annotate args **and** return types. The `UP` family auto-fixes legacy idioms.
- **Docstrings** — short, plain prose. No Sphinx/Google/Numpy section markers. One-liners are fine for self-evident helpers.
- **Logging** — `logger = logging.getLogger(__name__)`, f-strings for messages. `T20` rejects bare `print()` outside one-off `utils/` scripts (and the self-logging `slack_handler.py`); `G001`–`G003` reject `%`/`str.format()`/`+` pre-formatted log messages. The Slack handler attaches automatically when `SLACK_BOT_TOKEN` is set, so anything at `INFO+` may surface in Slack — keep messages information-dense and **free of secrets**.
- **Config access** — `get_config()` returns `Config | None`; the canonical idiom guards `if config is None: raise ValueError(...)` (None only happens if `init_config()` hasn't run — a programming error).
- **Dataclass mutable defaults** — always `field(default_factory=...)`, never a bare `[...]` (shared mutable state; `B`/bugbear flags it).
- **Retry/error-handling** — pipeline retry loops catch `KeyboardInterrupt` separately and re-raise, log with `logger.error`, then either retry after `time.sleep(retry_delay)` or `sys.exit(1)`. Reference: `train_command` / `inference_command`.
- **Naming** — descriptive full words (`num_training_rounds`, not `n`/`bs`). Single letters only in tight loops / math / indexing.

| Element       | Convention  | Example                  |
| ------------- | ----------- | ------------------------ |
| Classes       | PascalCase  | `DataGenerator`          |
| Functions     | snake_case  | `run_training_pipeline`  |
| Constants     | UPPER_SNAKE | `MAX_RETRIES`            |
| Private       | \_prefix    | `_init_worker`           |
| Config fields | snake_case  | `per_replica_batch_size` |

**Grep-friendly inline comment markers** (used consistently): `# TODO:` (actionable work), `# NOTE:` (rationale/question), `# FIX:` (known issue, no time now), `# BUG:` (known bug, often with workaround), `# TEST:` (behavior to verify, no suite yet). Prefer `# NOTE:` over `# TODO:` when there's no obvious action.

---

## Contribution Workflow

> **All issues are actionable, and all PRs must be tied to an existing issue.** Read `AI_POLICY.md` before doing AI-assisted work — the project has strict AI-usage rules.

1. **Discussion first** — check for existing PRs/issues/discussions; otherwise open a [GitHub Discussion](https://github.com/zachtheyek/Aetherscan/discussions) or Slack thread. "Drive-by" issues with no prior discussion may be closed.
2. **Open an issue** via the template; Claude auto-triages and labels it.
3. **Feature branch** — `category/description` with prefix: `feature/` (new functionality), `hotfix/` (bug fixes), `misc/` (housekeeping), `claude/` (reserved for the Claude assistant).
4. **Implement** — focused commits, pass all pre-commit hooks, follow `pyproject.toml` style.
5. **PR** — rebase (not merge) onto `master`; **all commits need verified GPG signatures**; fill the PR template; link the issue via the Development sidebar or `Closes #N` / `Fixes #N` (enables label sync). PRs not tied to an issue may be closed.
6. **Review** — needs passing checks, ≥1 maintainer approval, all conversations resolved, branch up to date. Approvals are voided when new commits are pushed. Claude provides an initial review automatically.

**Pre-commit hooks** (`pre-commit install` to activate): `ruff` (lint, `--fix`), `ruff-format`, general `pre-commit-hooks` (large files >1 MB, case conflict, merge conflict, YAML/TOML syntax, EOF/trailing-whitespace, private-key detection, `no-commit-to-branch` on master), and `gitleaks` (secret detection). Ruff-format auto-reformats on commit — **re-run `git add` after** it modifies files, then commit again. Bypass only sparingly with `git commit --no-verify`.

```bash
pre-commit run                # staged files
pre-commit run --all-files    # everything
pre-commit run ruff --all-files
```

---

## Security

- **Never commit secrets** (tokens, credentials, private data, internal URLs/IPs). Use `.env` (gitignored). `gitleaks` pre-commit hook + GitHub Dependabot back this up but aren't foolproof.
- **Secrets in use**: `SLACK_BOT_TOKEN` (Slack alerts/notifications). Use separate dev/prod tokens; store via a secrets manager or restricted-permission encrypted env files.
- **If a token leaks** — rotate immediately. Slack: revoke in [Slack API](https://api.slack.com/apps) → OAuth & Permissions, reinstall with scopes `channels:read, chat:write, files:write, groups:read, incoming-webhook`, update `SLACK_BOT_TOKEN` everywhere, verify with `PYTHONPATH=src python utils/print_cli_help.py train` (no Slack errors).
- **Incident response**: Contain (revoke creds) → Assess → Notify → Remediate (rotate secrets) → Document → Improve.
- **Reporting**: non-critical → [GitHub Discussion](https://github.com/zachtheyek/Aetherscan/discussions) with the "security" label; critical → contact [@zachtheyek](https://breakthroughlisten.slack.com/archives/D01SJG0L0TE) on Slack directly (do **not** open a public issue), expect a response in 48–72h.
- **Data security**: major outputs (weights, code, search results, training/inference data) are publicly disclosed via HuggingFace / GitHub / publications / [BL Open Data Archive](https://breakthroughinitiatives.org/opendatasearch); intermediate products (DB records, plots) stay on access-controlled HPC servers.
- **Dependency versions**: when bumping a dep, don't chase the latest — target the **newer** of {two minors below the latest stable, the latest stable ≥6 months old}, stable releases only (no alpha/beta/rc/nightly). A known advisory on that target overrides the lag → jump to the minimum patched version. Never cross a documented ceiling (`numpy<2.0`, `setuptools<81`) or the NGC TF 2.17 ABI, and keep `environment.yml` / `requirements-container.txt` / `aetherscan.def` in lockstep for shared deps. Full policy in `SECURITY.md` → Security Scanning → Version Selection Policy.
- False positives: add `file:line` to `.gitleaksignore` or inline `# gitleaks:allow` (less preferred).

---

## Reference Files

Paths relative to the repo root:

- `CLAUDE.md` — condensed always-on agent rules (this skill is its deep-dive companion)
- `README.md` — overview, install matrix, usage examples, full CLI reference
- `CONTRIBUTING.md` — workflow, project structure, code style, pre-commit
- `SECURITY.md` — security policy, secrets management, token rotation
- `KNOWN_ISSUES.md` — known bugs and workarounds
- `AI_POLICY.md` — AI usage policy (read before AI-assisted contributions)
- `docs/CONFIG_AND_CLI.md` — config system deep dive
- `docs/BLACKWELL_MIGRATION.md` — container build/runtime runbook
