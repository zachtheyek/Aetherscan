# CLAUDE.md

Always-on, bare-essentials rules for any coding agent in this repo. The full deep-dive lives in the on-demand skill at [`.claude/skills/aetherscan-repo-context/SKILL.md`](.claude/skills/aetherscan-repo-context/SKILL.md). Canonical sources: [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [KNOWN_ISSUES.md](KNOWN_ISSUES.md), [docs/](docs/). Read [AI_POLICY.md](AI_POLICY.md) before AI-assisted work.

## Project

Aetherscan: Breakthrough Listen's deep-learning SETI pipeline. Two-stage ML (Beta-VAE → Random Forest), single-node multi-GPU distributed train/inference. Sole entry point: `src/aetherscan/main.py`.

## Run / lint

```bash
# Canonical: NGC container (only option on Blackwell)
./utils/run_container.sh python -m aetherscan.main {train|inference} --save-tag train
# Alternative: conda env (Ampere only) — prefix with PYTHONPATH=src
PYTHONPATH=src python -m aetherscan.main {train|inference} --save-tag train
# Lint + format (also enforced via pre-commit)
ruff check src/ && ruff format src/
```

## Testing

Pytest suite in `tests/`: unit (the CI surface) + `gpu`/`cluster`-marked integration smokes. Full layout, fixtures, markers, and coverage notes are in [docs/TESTING.md](docs/TESTING.md) (the contribution-workflow view is in [CONTRIBUTING.md](CONTRIBUTING.md#testing)).

```bash
# Default local-dev selection (no GPUs / cluster data needed); CI runs
# essentially the same, plus `and not integration` as a leak-guard — see docs/TESTING.md:
pytest -m "not gpu and not cluster" -q
# gpu/cluster integration smokes — need a cluster (data + a persisted model under /datax) and the NGC container:
./utils/run_container.sh python -m pytest tests/ -m "gpu or cluster" -q
```

- **Run the suite before claiming a change works**, and **ship unit tests with new logic**.
- Most test modules import TensorFlow at collection — if the full dependency stack isn't available locally, run the subset you can and say exactly what you ran.

## Hard rules (don't break these)

- Create all TensorFlow models **inside `strategy.scope()`**.
- Use singleton accessors `get_config()` / `get_db()` / `get_manager()` — never instantiate directly; never mutate config post-init in multi-threaded code.
- Shared memory: only the **creator** calls `shm.unlink()`, never workers. Register pools/SHM with ResourceManager; call `holder.clear()` when done.
- **Never log inside SIGTERM handlers** (deadlock).
- Dataclass mutable defaults: always `field(default_factory=...)`, never a bare `[...]`.
- **Never commit secrets** — use `.env` (gitignored); the `gitleaks` hook backs this up but isn't foolproof.

## Style

ruff lint+format, 100-char lines, Python 3.10 target ([`pyproject.toml`](pyproject.toml)). Every module starts with `from __future__ import annotations`; PEP 604/585 typing (`X | None`, `list[int]`); module-level f-string loggers, no bare `print()` outside `utils/` (INFO+ may reach Slack → no secrets in logs). Naming: PascalCase classes, snake_case functions/config fields, UPPER_SNAKE constants, `_prefix` private. Comment markers: `# TODO:` / `# NOTE:` / `# FIX:` / `# BUG:` / `# TEST:`.

## Contributing

- Every PR links an existing issue (`Closes #N`); branch prefixes `feature/`/`hotfix/`/`misc/`/`claude/`; rebase (not merge) onto `master`; commits need **verified GPG signatures**; all pre-commit hooks must pass (ruff-format may rewrite files → `git add` again before re-committing).
- **Don't tag the assistant unintentionally.** The assistant handle (an `@` immediately followed by `claude`) in a Discussion/issue/PR title or body triggers the assistant workflow (`claude.yml`) — write it only when you actually want to invoke the assistant. To reference it as plain text, write `"@ claude"` (space after the `@`, double quotes on both sides) so the trigger can't match.
- If you change `cli.py`, regenerate the README CLI Reference: `PYTHONPATH=src python utils/print_cli_help.py all`.
- Bumping a dependency? Don't jump to the latest — target a proven version per [SECURITY.md](SECURITY.md) (the newer of ~2 minors back / latest stable ≥6 months old; a known advisory on that target overrides the lag). Never cross a documented ceiling (`numpy<2.0`, …) or the NGC TF 2.17 ABI, and keep `environment.yml` / `requirements-container.txt` / `aetherscan.def` / `Dockerfile` / `pyproject.toml` in sync (`aetherscan.def` + `Dockerfile` both pin the NGC base digest).
- Releases are SemVer (`vX.Y.Z`): the compatibility contract covers the CLI, artifact/config formats + model contract, and supported runtimes; highest bump wins, and code + weights share one version string (a same-contract retrain is at least a MINOR). Full policy: [docs/RELEASE.md](docs/RELEASE.md#versioning-policy-semver).
- Security: non-critical → GitHub Discussion w/ "security" label; critical → [@zachtheyek](https://breakthroughlisten.slack.com/archives/D01SJG0L0TE) on Slack, no public issue. Rotate any leaked token immediately.

## Code review

Every PR gets an automated `claude-code-review` first pass. **Wait for it to land**, then work each note individually — weigh it against your own read of the code, don't take it on faith.

- Genuine gaps → fix in focused, self-contained commits on the **same** PR. Notes you think are wrong → leave the code, argue concretely why.
- Post **one** reply covering both (what you changed and why, then where you think the review erred and why), and end it by deliberately tagging the assistant handle to trigger a second pass — the sanctioned intentional case of the handle rule above, not a contradiction of it.
- Loop (wait → validate → address/rebut → comment → re-tag) until the review is clean/LGTM, or it drifts off the PR's theme or turns nonsensical — then post why you're stopping and **don't** tag again.

## More detail

On-demand deep-dive skill: [`.claude/skills/aetherscan-repo-context/SKILL.md`](.claude/skills/aetherscan-repo-context/SKILL.md) — install paths, config/CLI system, architecture patterns, full workflow & security. Per-surface technical docs (architecture, training, inference, preprocessing, models, database, runtime services, testing, automation, releases) are indexed in [docs/README.md](docs/README.md) — start at [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
