# Aetherscan docs

This directory holds long-form technical documentation that supplements the project [`README.md`](../README.md). For installation, usage examples, and the full CLI reference, start there; reach for the documents below when you need the underlying detail.

| Document | Summary |
|----------|---------|
| [`BLACKWELL_MIGRATION.md`](BLACKWELL_MIGRATION.md) | Runbook for running Aetherscan on the Blackwell (RTX PRO 6000) workstation alongside the existing Ampere (A4000) cluster. Covers container builds, verification, the four GPU-related CLI flags, debugging recipes, and the fallback escalation ladder. |
| [`CONFIG_AND_CLI.md`](CONFIG_AND_CLI.md) | Deep-dive into Aetherscan's singleton-config + CLI architecture: the three flag-routing patterns (A/B/C), the four cross-mode contamination barriers, the validation + fix-proposal layer, and the checklist for adding a new flag. Read this before touching `config.py` or `cli.py`. |
