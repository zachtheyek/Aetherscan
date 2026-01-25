# Contributing to Aetherscan

This document describes the process of contributing to Aetherscan.

# TODO:

- (tmp) we encourage contributions from the OSS community, but due to having limited maintainers, all PRs must be tied to an existing issue. if no issue exists yet, open one up & discuss ur changes before submitting a PR. all PRs that don’t follow these rules will be rejected automatically without human review
- describe project structure
  - removed module level READMEs. document module details here
- workflow
  - if you have a question or problem, start a GitHub discussion and/or Slack thread ([#aetherscan-dev](https://breakthroughlisten.slack.com/archives/C0A3CDALQD8))
  - once the maintainers have acknowledged & understood your query, open a GitHub issue with an existing template
  - implement the fix to the issue by creating a feature branch whose name follows the convention `category/description` (e.g. `feature/db_integration`, `hotfix/cpu_sampling_rate`, `release/aetherscan_1.0.3_50m`, `misc/plot_improvements`, etc.)
  - submit a PR when your fixes are complete
  - PRs may be merged after at least one maintainer has reviewed & approved your changes, and your PR passes all existing tests. as well, your feature branch should be up to date with the latest commits in `master` (linear history is strictly preferred. in other words, use git rebase over git merge), and all commits must have verified signatures
    - note that PR approvals are voided when new commits are pushed to the existing feature branch
- when versioned releases come out, update version number in:
  - pyproject.toml
  - src/aetherscan/**init**.py
  - CITATION.cff (also update release date)
- issue/PR authors are automatically designated as assignees. codeowners are automatically designated as reviewers. PRs must be tied to an existing issue. PRs can only be merged after at least one approval from a reviewer. When a PR is merged, the accompanying issue will automatically be closed. claude will automatically triage and label open issues. claude will provide a code review of your PR when it's first set to "ready for review". manual claude triggers are only allowed for users with write access or higher to the repo

## known issues:

```
2026-01-24 03:52:38.996928: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: CANCELLED: GetNextFromShard was cancelled
         [[{{node MultiDeviceIteratorGetNextFromShard}}]]
2026-01-24 03:52:38.997040: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: CANCELLED: GetNextFromShard was cancelled
         [[{{node MultiDeviceIteratorGetNextFromShard}}]]
         [[RemoteCall]] [type.googleapis.com/tensorflow.DerivedStatus='']
...
```

insert snippet about TF prefetch threads
