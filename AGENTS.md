# AGENTS.md

Always-on, bare-essentials rules for any coding agent in this repo. The full deep-dive lives in the on-demand skill at [`.claude/skills/aetherscan-repo-context/SKILL.md`](.claude/skills/aetherscan-repo-context/SKILL.md). Canonical sources: [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [KNOWN_ISSUES.md](KNOWN_ISSUES.md), [docs/](docs/). Read [AI_POLICY.md](AI_POLICY.md) before AI-assisted work.

## Project

Aetherscan: Breakthrough Listen's deep-learning SETI pipeline. Two-stage ML (Beta-VAE → Random Forest), single-node multi-GPU distributed train/inference. Sole entry point: `src/aetherscan/main.py`. `tests/` is a placeholder — **no test suite yet**, so don't claim `pytest` passes; verify changes another way and say how.

## Run / lint

```bash
# Canonical: NGC container (only option on Blackwell)
./utils/run_container.sh python -m aetherscan.main {train|inference} --save-tag final_v1
# Alternative: conda env (Ampere only) — prefix with PYTHONPATH=src
PYTHONPATH=src python -m aetherscan.main {train|inference} --save-tag final_v1
# Lint + format (also enforced via pre-commit)
ruff check src/ && ruff format src/
```

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
- If you change `cli.py`, regenerate the README CLI Reference: `PYTHONPATH=src python utils/print_cli_help.py all`.
- Security: non-critical → GitHub Discussion w/ "security" label; critical → [@zachtheyek](https://breakthroughlisten.slack.com/archives/D01SJG0L0TE) on Slack, no public issue. Rotate any leaked token immediately.

## More detail

On-demand deep-dive skill: [`.claude/skills/aetherscan-repo-context/SKILL.md`](.claude/skills/aetherscan-repo-context/SKILL.md) — install paths, config/CLI system, architecture patterns, full workflow & security.
