# GitHub Automation

This document catalogs every workflow in
[`.github/workflows/`](../.github/workflows) — trigger, behavior, and the deterministic
dedup guards several of them rely on — plus the issue/PR lifecycle they collectively
implement and the rules for (not) invoking the assistant. The human-side contribution
process is [`CONTRIBUTING.md`](../CONTRIBUTING.md); this is the machine side.

## TL;DR — the lifecycle

```
Discussion → Issue ──────────────► PR ─────────────► merge to master ──► weekly upkeep
             │ auto-assign-author   │ auto-assign      │ claude-release-notes   │ claude-dependency-check
             │ claude-issue-triage  │ sync-pr-labels   │ claude-style-check     │ claude-flaky-test-tracker
             │ claude-contribution- │ claude-code-      │ claude-update-docs
             │   check              │   review
                                    │ pre-commit + tests (required checks)
```

Two workflow families: **deterministic** (pre-commit, tests, auto-assign, label sync) and
**assistant-driven** (`claude-*.yml` — each wraps `anthropics/claude-code-action` with a
task-specific prompt, authenticated via the `CLAUDE_CODE_OAUTH_TOKEN` secret).

## The assistant handle — read this before writing issue/PR text

The general assistant workflow ([`claude.yml`](../.github/workflows/claude.yml)) triggers on
a `contains(...)` match of the assistant handle — an `@` immediately followed by `claude` —
anywhere in the title or body of an issue, PR, or Discussion (or a comment/review on one).
A `contains` match has no notion of quoting or context: *any* occurrence fires a real
assistant run and, typically, a follow-up PR.

- Write the handle **only when you intend to summon the assistant**.
- To mention it as plain text, write it as `"@ claude"` — a space after the `@`, double
  quotes around it — so the substring can never match. This convention is prescribed in
  [`CLAUDE.md`](../CLAUDE.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md); an accidental tag
  is what spawned the spurious run around issue #83.
- The workflow also triggers on **assignment** to the `claude` user and on the `claude`
  **label** — treat those as invocations too.

## Deterministic workflows

### `pre-commit.yml`

**Trigger:** every push and PR to any branch, plus `workflow_dispatch`.
Runs the full [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) suite (ruff lint +
format, hygiene hooks, gitleaks) on Python 3.10 with pip/ruff caches. On master it skips the
`no-commit-to-branch` hook (branch protection already restricts master to merge commits).
This is a required check — the same hooks you run locally, so a green local
`pre-commit run --all-files` predicts a green check.

### `tests.yml`

**Trigger:** every PR, pushes to master, `workflow_dispatch`.
Runs `pytest -m "not gpu and not cluster" -q` on Python 3.10 **and** 3.12 (the conda and NGC
container runtimes; `fail-fast: false`), with `tensorflow-cpu==2.17.*` standing in for the
container's GPU build. Details in [`TESTING.md`](TESTING.md). Its run history is the data
source for the flaky-test tracker below.

### `auto-assign-author.yml`

**Trigger:** issue or PR opened.
Assigns the author to their own issue/PR via `github-script` — keeps the "assignee" field
meaningful without manual bookkeeping.

### `sync-pr-labels.yml`

**Trigger:** PR opened/edited; issue labeled.
Copies labels from the issues a PR formally links (GraphQL `closingIssuesReferences` — i.e.
`Closes #N` / the Development sidebar) onto the PR, and re-syncs when a linked issue is
labeled later (e.g. by triage). `needs-issue`, `needs-discussion`, and `good first issue`
are excluded (they describe issues, not PRs). This is why PRs must link their issue with a
closing keyword: label sync and the contribution check both key on the formal link.

## Assistant-driven workflows

All of these run `anthropics/claude-code-action` (version-pinned by commit SHA) with a
scoped prompt and a `timeout-minutes` cap. Where they create issues/comments, a
**deterministic dedup guard** runs first — a plain shell step that searches for a hidden
HTML marker and skips the assistant step entirely when found, so re-runs can never
double-post.

| Workflow | Trigger | What it does | Dedup marker |
| --- | --- | --- | --- |
| [`claude.yml`](../.github/workflows/claude.yml) | Handle mention / assignment / `claude` label on issues, PRs, Discussions, comments, reviews | The general-purpose assistant: answers, implements, opens PRs on `claude/*` branches. `allowed_bots: "claude,claude[bot]"` — deliberately, so issues *filed by* the other workflows (as the `claude` bot) can invoke it for follow-up PRs. | — |
| [`claude-code-review.yml`](../.github/workflows/claude-code-review.yml) | PR opened / ready_for_review | Automated first-pass code review with inline comments. Every PR gets one on open; address the actionable notes and resolve the conversations before human review. | — |
| [`claude-issue-triage.yml`](../.github/workflows/claude-issue-triage.yml) | Issue opened | Triage: applies labels (type/area/priority) so label sync can propagate them to the eventual PR. | — |
| [`claude-contribution-check.yml`](../.github/workflows/claude-contribution-check.yml) | Issue or PR opened | Verifies workflow compliance (issue linkage, branch-prefix conventions, template use) and comments when something's missing. **Never runs for bot authors** — see the gotcha below. | — |
| [`claude-release-notes.yml`](../.github/workflows/claude-release-notes.yml) | PR **merged** to master | Drafts a release-note entry as a PR comment — the curated raw material for release bodies (see [`RELEASE.md`](RELEASE.md)). | `<!-- claude-release-notes -->` first line of the comment |
| [`claude-style-check.yml`](../.github/workflows/claude-style-check.yml) | PR merged to master | Scans the merged diff's *added* lines against the project style rules ruff can't express (docstring prose style, canonical comment markers, logging idioms); files one consolidated issue when violations exist. | `<!-- aetherscan-style-check pr=<N> -->` |
| [`claude-update-docs.yml`](../.github/workflows/claude-update-docs.yml) | PR merged to master; `workflow_dispatch` with a `pr_number` (re-scan an old PR with the *current* workflow logic) | Detects doc drift caused by the merge. If `cli.py` changed, a **shell step** regenerates the README CLI Reference blocks with `utils/print_cli_help.py` (Python pinned to 3.12 — argparse help formatting changes in 3.13) and embeds the output in the issue, because the follow-up assistant run has no `python` in its tool allowlist. The filed issue contains an intentional handle mention, which triggers `claude.yml` to open the actual docs PR. | `<!-- aetherscan-update-docs pr=<N> -->` |
| [`claude-dependency-check.yml`](../.github/workflows/claude-dependency-check.yml) | Weekly (Mon 01:00 UTC) + `workflow_dispatch` | Audits `environment.yml` / `requirements-container.txt` / `aetherscan.def` against registries and advisories under [`SECURITY.md`](../SECURITY.md)'s version-selection policy; files a weekly report issue. | `<!-- aetherscan-dependency-check week=<WEEK> -->` |
| [`claude-flaky-test-tracker.yml`](../.github/workflows/claude-flaky-test-tracker.yml) | Weekly (Mon 01:00 UTC) + `workflow_dispatch` | Reads the week's `tests.yml` runs, identifies flaky/failing tests, diagnoses the worst offender, files a weekly report issue. | `<!-- aetherscan-flaky-test-tracker week=<WEEK> -->` |

> [!NOTE]
> **`claude-update-docs` is a two-step relay**, which is why its row is dense. Step 1 is a
> deterministic shell step: when `cli.py` changed it regenerates the README CLI Reference with
> `utils/print_cli_help.py` (Python pinned to 3.12 — argparse help formatting changes in 3.13)
> and files an issue embedding that output. Step 2 is the assistant: the filed issue carries an
> intentional handle mention that triggers `claude.yml` to open the actual docs PR. The split
> exists because the follow-up assistant run has no `python` in its tool allowlist, so it
> cannot regenerate the CLI help itself — the shell step must do it first.

The dedup-marker pattern is a convention to preserve in any new workflow that posts content:
the guard step greps existing issues/comments for the marker (`gh ... --json body --jq` +
`grep`), and the prompt instructs the assistant to put the marker as the **first line** of
anything it posts — deterministic idempotence without trusting the model to check.

### The `allowed_bots` gotchas

`claude-code-action` refuses to run for non-human actors unless the triggering actor is in
its `allowed_bots` list. Three configurations coexist here, each deliberate:

- **`claude.yml`: `allowed_bots: "claude,claude[bot]"`** — bot-authored content *can* summon
  the assistant. Required for the update-docs pipeline: the doc issue is filed by the
  `claude` bot and must still trigger the follow-up PR.
- **`claude-code-review.yml` / `claude-issue-triage.yml`: `allowed_bots: "claude[bot]"`** —
  assistant-opened PRs get reviewed and assistant-filed issues get triaged like anyone
  else's.
- **`claude-contribution-check.yml`: `allowed_bots` unset, plus a job-level `if:` that
  excludes bot actors entirely.** Without the `if:`, every bot-filed issue would spin up a
  runner just for the action to abort with a "non-human actor" error — a red ✗ on every
  automated issue (this is the other half of the issue #83 story). Skipping the job at the
  `if:` level avoids both the comment and the noisy failure.

When adding a workflow, decide explicitly which side of this each trigger actor falls on.

## Issue and PR conventions the automation assumes

- **Issue templates**: [`bug_report.md`](../.github/ISSUE_TEMPLATE/bug_report.md) and
  [`feature_request.md`](../.github/ISSUE_TEMPLATE/feature_request.md). Triage expects
  template-shaped issues; drive-by issues may be closed per
  [`CONTRIBUTING.md`](../CONTRIBUTING.md).
- **Every PR links an existing issue** (`Closes #N`) — label sync, the contribution check,
  and the release-notes context all read the formal link.
- **Branch prefixes** `feature/` / `hotfix/` / `misc/` / `claude/` (the last reserved for
  assistant-authored branches).
- **Required checks**: pre-commit + tests (both Python versions) must pass; commits need
  verified GPG signatures; branches rebase (never merge) onto master.
- **Merges fan out**: expect a release-notes comment on your merged PR, and possibly a
  style-check or update-docs issue referencing it — these are normal post-merge automation,
  not review feedback.

## Secrets and permissions

Assistant workflows authenticate with the `CLAUDE_CODE_OAUTH_TOKEN` repository secret and
request `id-token: write` plus the minimal `contents`/`issues`/`pull-requests` permissions
each task needs; deterministic workflows use the default `GITHUB_TOKEN`. Scheduled workflows
only fire from the default branch — after editing one, remember it runs master's copy until
your change merges (use `workflow_dispatch` to test).
