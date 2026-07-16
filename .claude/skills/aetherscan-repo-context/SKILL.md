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
- **`utils/fetch_run_outputs.sh`** rsyncs one run's outputs from remote cluster node(s) to the local `outputs/` tree, selecting files by the universal `*_<save_tag>.*` suffix and renaming each to `<machine>_<basename>` (collision-free across nodes). `<train|inference> <save_tag> <machine>...`; `--all` adds train checkpoints/archive, `--db` pulls the SQLite DB into `outputs/data/db/`, `--dry-run`. Assumes tagged log filenames; the inference branch is provisional pending the inference pipeline.
- **`utils/kill_pipeline.sh`** stops a running pipeline (main process + all worker children) from a separate shell on the same machine — works for both run modes, finds the process tree itself, and tries a graceful SIGTERM (lets `ResourceManager` close pools/SHM) before escalating to SIGKILL. Assumes a single running instance. `--force` / `--dry-run` / `--timeout N`.
- **`utils/run_container.sh`** auto-detects apptainer vs singularity (Apptainer wins when both present), sets `--nv` for GPU passthrough, auto-loads `<repo>/.env`, and bind-mounts the repo + `AETHERSCAN_{DATA,MODEL,OUTPUT}_PATH` 1:1 so absolute paths persisted in the DB stay valid across host and container.
- **`utils/start_tmux_session.sh`** (optional) spins up a four-window monitoring tmux session (htop/CPU-MEM, `nvidia-smi`, `/dev/shm`, `tree` of models/outputs). Idempotent.

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

The tree below annotates the **source** layout. For the complete repository structure — root-level build/config files (`pyproject.toml`, `environment.yml`, `aetherscan.def`, `requirements-container.txt`, `.pre-commit-config.yaml`), governance docs (`CLAUDE.md`, `CONTRIBUTING.md`, `SECURITY.md`, `KNOWN_ISSUES.md`, `AI_POLICY.md`), and the `.claude/` and `.github/` directories — see the Project Structure tree in `CONTRIBUTING.md` (the canonical source).

```
src/aetherscan/
├── main.py              # Entry point, command dispatch, GPU strategy setup (NCCL + fallback)
├── cli.py               # Argument parsing, validation, config override
├── config.py            # Configuration dataclasses & defaults (singleton)
├── train.py             # Training orchestration, curriculum learning, checkpointing
├── round_data.py        # Disk-backed (memmap) round datasets + background producer process
├── run_state.py         # Persisted training-run manifest (stage-aware resume)
├── inference.py         # Inference orchestration, candidate detection
├── inference_viz.py     # End-of-run inference visualization suite
├── preprocessing.py     # Loading / downsampling / log-normalization + energy detection
├── pfb.py               # PFB static passband equalization (bandpass flattening)
├── data_generation.py   # Synthetic signal injection (setigen)
├── models/{vae,random_forest}.py
├── db/db.py             # Thread-safe SQLite, async queue-based writes
├── logger/              # Multi-handler logging + Slack integration
├── monitor/monitor.py   # Background resource monitoring (CPU, RAM, GPU)
└── manager/manager.py   # Resource lifecycle management (pools, shared memory)
utils/                   # fetch_run_outputs.sh, find_optimal_configs.py,
                         # get_system_info.sh, kill_pipeline.sh, print_cli_help.py,
                         # run_container.sh, start_tmux_session.sh,
                         # verify_train_test_files.py
docs/                    # Full technical doc suite, one doc per pipeline surface —
                         # indexed in docs/README.md; start at docs/ARCHITECTURE.md
tests/                   # Pytest suite: unit/ (CI surface) + gpu/cluster-marked
                         # integration/ smokes — see the "Testing" section below
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

**Grep-friendly inline comment markers** (used consistently): `# TODO:` (actionable work), `# NOTE:` (rationale/question), `# FIX:` (known issue, no time now), `# BUG:` (known bug, often with workaround), `# TEST:` (behavior to verify — now backed by the `tests/` suite). Prefer `# NOTE:` over `# TODO:` when there's no obvious action.

---

## Testing

The `tests/` suite splits along a hardware axis:

- **`tests/unit/`** — fast, hardware-independent, one `test_<module>.py` per source module. This is the CI surface; everything here must pass on a CPU-only runner.
- **`tests/integration/`** — `gpu`/`cluster`-marked end-to-end smokes (`test_train_smoke.py`, `test_inference_smoke.py`) that launch `python -m aetherscan.main ...` as a real subprocess on a cluster, against cluster-resident data/models. Not run in CI.

**Default selection — exactly what CI runs** (`.github/workflows/tests.yml`, on Python 3.10 and 3.12), no GPUs or cluster data needed:

```bash
pytest -m "not gpu and not cluster" -q
```

`pyproject.toml`'s `[tool.pytest.ini_options]` sets `pythonpath = ["src"]`, so **pytest needs no `PYTHONPATH=src` prefix** (unlike running `main.py` from source); it also sets `testpaths = ["tests"]` and `--strict-markers` (a typo'd marker is a collection error, not a silently-ignored one).

**Markers** (declared in `pyproject.toml`; `--strict-markers` rejects undeclared ones):

| Marker        | Meaning                                          | In default selection?   |
| ------------- | ------------------------------------------------ | ----------------------- |
| `slow`        | Slower CPU tests (e.g. builds real TF graphs)    | **Yes** — CI runs them  |
| `gpu`         | Needs one or more physical GPUs                  | No                      |
| `cluster`     | Needs cluster-resident data/models (blpc3/bla0)  | No                      |
| `integration` | End-to-end subprocess runs; **skips isolation**  | No (also `gpu`+`cluster`) |

**Isolation.** The autouse `aetherscan_isolated_env` fixture in `tests/conftest.py` wraps every non-integration test: it points `AETHERSCAN_{DATA,MODEL,OUTPUT}_PATH` at a fresh `tmp_path` tree, deletes `SLACK_BOT_TOKEN`/`SLACK_CHANNEL` (tests must never talk to Slack), resets all five singletons (`Config`, `Database`, `Logger`, `ResourceManager`, `ResourceMonitor`) via their `_reset()` hooks, then on teardown stops any leaked background threads/pools and restores the snapshotted SIGINT/SIGTERM handlers and stdout/stderr. Net effect: tests never touch real data and can't leak state into one another. Integration tests are exempt — they inherit the real environment and run the pipeline as a subprocess.

**Discipline.** Run the suite (or the subset you can) before claiming a change works, and **ship unit tests with new logic** — every behavior change should land tests under `tests/unit/` in the matching `test_<module>.py` (create it if the module is new).

**Gotcha.** Most unit modules import TensorFlow at collection time, so a bare `pytest` needs the full dependency stack (CI installs `tensorflow-cpu==2.17.*` plus the container requirements). If that stack isn't available locally, run the TF-free subset you can — e.g. `pytest tests/unit/test_config.py -q` — and **say exactly what you ran** rather than claiming the whole suite passed.

Deep dive: `docs/TESTING.md` covers the full layout, the synthetic data factories, the coverage-and-deliberate-gaps notes (`monitor` / `logger` / `slack_handler`), CI specifics, how to run the cluster smokes, and the adding-tests checklist.

---

## Contribution Workflow

> **All issues are actionable, and all PRs must be tied to an existing issue.** Read `AI_POLICY.md` before doing AI-assisted work — the project has strict AI-usage rules.

1. **Discussion first** — check for existing PRs/issues/discussions; otherwise open a [GitHub Discussion](https://github.com/zachtheyek/Aetherscan/discussions) or Slack thread. "Drive-by" issues with no prior discussion may be closed.
2. **Open an issue** via the template; Claude auto-triages and labels it.
3. **Feature branch** — `category/description` with prefix: `feature/` (new functionality), `hotfix/` (bug fixes), `misc/` (housekeeping), `claude/` (reserved for the Claude assistant).
4. **Implement** — focused commits, pass all pre-commit hooks, follow `pyproject.toml` style.
5. **PR** — rebase (not merge) onto `master`; **all commits need verified GPG signatures**; fill the PR template; link the issue via the Development sidebar or `Closes #N` / `Fixes #N` (enables label sync). PRs not tied to an issue may be closed.
6. **Review** — needs passing checks, ≥1 maintainer approval, all conversations resolved, branch up to date. Approvals are voided when new commits are pushed. Claude provides an initial review automatically.

**Invoking vs. mentioning the assistant.** The assistant workflow (`claude.yml`) triggers whenever the assistant handle — an `@` immediately followed by `claude` — appears in the title/body of a Discussion, issue, or PR (or a comment on one). Write it only when you actually want to summon the assistant (e.g. an auto-filed docs issue asking it to open a PR). To refer to the handle as plain text anywhere else — a PR description, issue body, commit message, review comment — write it as `"@ claude"` (a space after the `@`, double quotes on both sides) so the `contains(…, '@claude')` trigger can't match. Tagging it unintentionally spawns a spurious assistant run and follow-up PR (this is what happened around issue #83).

**Responding to the automated review.** Opening (or marking ready) a PR triggers `claude-code-review.yml`, which posts a first-pass review with inline comments (catalogued in `docs/GITHUB_AUTOMATION.md`). Treat it as input, not verdict: wait for the review to land, then work through each comment individually, weighing it against your own understanding of the codebase and the change you actually made — don't assume the reviewer is right. Where a comment exposes a genuine blind spot, fix it in a focused, self-contained commit pushed to the *same* PR; where you're convinced it's wrong, leave the code untouched and be ready to explain concretely why. Then post a single PR comment covering both halves — first the points you addressed (what you changed and the rationale), then the points you think the reviewer got wrong (with your reasoning) — and close that comment by deliberately tagging the assistant handle to kick off a second-pass review. This is precisely the "you actually want to summon it" case from the paragraph above, not a violation of the don't-tag-unintentionally rule. Then repeat the loop — wait, read, validate, address, rebut, comment, re-invoke — until the reviews either come back clean (no further notes / LGTM) or they start drifting out of scope (raising points unrelated to the PR's theme) or turn nonsensical. At that stopping point, post a comment explaining why you're stopping, and do **not** tag the assistant handle again.

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
- `docs/README.md` — index of the technical documentation suite (one doc per surface)
- `docs/ARCHITECTURE.md` — system map: data model, module map, init order, artifact layout
- `docs/TRAINING_PIPELINE.md` — rounds, round data + producer, retries, every training plot
- `docs/INFERENCE_PIPELINE.md` — catalogs, streaming flow, manifest retries, inference figures
- `docs/PREPROCESSING.md` — energy detection math (PFB/spline, k² derivation), signal injection
- `docs/MODELS.md` — Beta-VAE architecture/loss math, RF features + threshold semantics
- `docs/DATABASE.md` — schema, writer thread, flush/supersede protocols, migrations
- `docs/RUNTIME_SERVICES.md` — logger/Slack, ResourceManager lifecycle, resource monitor
- `docs/TESTING.md` — suite layout, markers, isolation fixtures, CI, cluster smokes
- `docs/GITHUB_AUTOMATION.md` — every workflow, dedup guards, assistant-handle rules
- `docs/RELEASE.md` — version-coupling contract, CD gates, release runbook
- `docs/GPU_RUNTIME_GUIDE.md` — container build/runtime runbook
- `docs/CONFIG_AND_CLI.md` — config system deep dive
