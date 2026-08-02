# AGENTS.md

The agent rules for this repo have **one** canonical home: **[`CLAUDE.md`](CLAUDE.md)**. Read that file — it is written for "any coding agent in this repo", so nothing in it is assistant-specific beyond the filename.

- Always-on, bare-essentials rules → [`CLAUDE.md`](CLAUDE.md)
- On-demand deep-dive skill → [`.claude/skills/aetherscan-repo-context/SKILL.md`](.claude/skills/aetherscan-repo-context/SKILL.md)
- Canonical sources → [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [KNOWN_ISSUES.md](KNOWN_ISSUES.md), [docs/](docs/). Read [AI_POLICY.md](AI_POLICY.md) before AI-assisted work.

This file stays a pointer on purpose. The rules encode facts about *this* repo's automation — the assistant handle that triggers `claude.yml`, the reserved `claude/` branch prefix, the `claude-code-review` workflow — so a second copy with the assistant's name find/replaced isn't just duplicated, it's factually wrong (and inverts the don't-tag-unintentionally rule it is trying to state).
