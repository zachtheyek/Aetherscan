# Aetherscan docs

This directory holds long-form technical documentation that supplements the project [`README.md`](../README.md). For installation, usage examples, and the full CLI reference, start there; reach for the documents below when you need the underlying detail.

| Document | Summary |
|----------|---------|
| [`GPU_RUNTIME_GUIDE.md`](GPU_RUNTIME_GUIDE.md) | Runbook for building, running, and debugging the pipeline across GPU architectures — the Blackwell (RTX PRO 6000) workstation and the Ampere (A4000) cluster — unified by the NGC container runtime. Covers container builds, image verification, the four GPU-related CLI flags, CUDA/NCCL debugging recipes, the fallback escalation ladder, and cross-machine checkpoint interop. Read this before building the container or running on a new GPU architecture (especially Blackwell/sm_120), or after any run that dies with a `CUDA_ERROR_INVALID_PTX`/NCCL/OOM failure. |
| [`CONFIG_AND_CLI.md`](CONFIG_AND_CLI.md) | Deep-dive into Aetherscan's singleton-config + CLI architecture: the three flag-routing patterns (A/B/C), the four cross-mode contamination barriers, the validation + fix-proposal layer, and the checklist for adding a new flag. Read this before touching `config.py` or `cli.py`. |
