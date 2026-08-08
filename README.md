<h1 align="center">📡 Aetherscan 📡</h1>
<p align="center">
    <img src="docs/assets/aetherscan-banner.png" alt="Aetherscan">
</p>
<p align="center">
    Breakthrough Listen's first end-to-end production-grade deep learning pipeline for SETI @ scale
    <br />
    <br />
    <a href="https://github.com/zachtheyek/Aetherscan/actions/workflows/tests.yml"><img src="https://github.com/zachtheyek/Aetherscan/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
    <a href="https://pypi.org/project/aetherscan/"><img src="https://img.shields.io/pypi/v/aetherscan.svg" alt="PyPI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg" alt="License"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%E2%80%933.12-blue.svg" alt="Python"></a>
    <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.17-orange.svg" alt="TensorFlow"></a>
    <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.4%E2%80%9312.8-green.svg" alt="CUDA"></a>
  </p>
</p>

---

## Overview

Aetherscan is a deep learning pipeline for detecting anomalies in radio spectrograms with technosignature-like characteristics. It combines a beta-VAE (for dimensionality reduction/feature extraction) with a Random Forest ensemble (for candidate detection), trained on ~30m unique cadence snippets using a composite loss that balances reconstruction, KL divergence, and true/false clustering. The pipeline is designed with performance in mind, by default running single-node distributed training & inference, using zero-copy parallelism during pre- and post-processing.

The model architecture is based on [Ma et al. 2023](https://arxiv.org/abs/2301.12670) ("_A deep-learning search for technosignatures from 820 unique stars_"), extending the research prototype into a production-ready system capable of near real-time inference.

---

## Installation

### System Requirements

Aetherscan supports two install paths off the same source tree — the **NGC container** (the canonical runtime on both clusters) and a **conda env** (an alternative on Ampere for those who can't or won't use containers) — plus the published **PyPI package** for off-cluster use (see [Install From PyPI (pip)](#install-from-pypi-pip) for its v1.0.0 caveats).

**NGC container (canonical, runs on both clusters)**

- Ubuntu 24.04
- ≥1x NVIDIA GPU:
  - Blackwell (sm_120, e.g. RTX PRO 6000) — driver ≥570 (native CUDA 12.8)
  - Ampere (sm_86, e.g. RTX A4000) — driver ≥550 (host CUDA 12.3) via CUDA forward compatibility
- VRAM: **≥8 GB per GPU** recommended — measured peaks ~6 GB/GPU (training) and ~2.5 GB/GPU (inference) on the v1.0.0 release runs; gradient accumulation keeps per-GPU VRAM low
- RAM: **≥288 GB** for full-scale training and default catalog-scale inference (measured peaks ~260 GB training / ~200 GB inference, plus headroom — a strict-256 GB host sits too close to the training peak and risks OOM under page-cache pressure). Means are much lower (~150 GB training / ~36 GB inference); inference RAM scales with `--prefetch-depth` × the largest in-flight cadence, so lower `--prefetch-depth` for smaller-RAM hosts or small catalogs
- Disk: full-scale training round data ~147 GB per retained round (float16 default), up to ~7.4 TB (~147 GB × 50) with `--keep-round-data` at the 50-round default; inference stamp caches are retained by default (~1 GB/cadence average — re-scores under the same energy-detection config then skip preprocessing entirely; pass `--prune-stamps` on catalog-scale runs, which keeps ~1 MB/cadence metadata + a transient ~5–20 GB/cadence × `--prefetch-depth` during extraction)
- Apptainer 1.4+ or SingularityCE 4.1+ (Python 3.12 / TF 2.17 / CUDA 12.8 live inside the container)
- Prebuilt image published to GHCR (`ghcr.io/zachtheyek/aetherscan:vX.Y.Z`, `linux/amd64`); `utils/run_container.sh` pulls it automatically, or prints `aetherscan.def` build instructions if the pull fails — see [Run From Container](#run-from-container)
- See [`docs/GPU_RUNTIME_GUIDE.md`](docs/GPU_RUNTIME_GUIDE.md) for the full runbook

**Conda env (alternative, Ampere only)**

- Ubuntu 24.04
- ≥1x NVIDIA GPU:
  - Ampere with CUDA 12.3+ driver
- VRAM / RAM same as above
- Python 3.10 / TF 2.17 + tf_keras 2.17, legacy-Keras mode (managed by conda)

> [!NOTE]
> There are no plans to support non-Nvidia GPUs

> [!NOTE]
> The figures above are measured from the v1.0.0 release runs — training on 6× RTX A4000 (16 GB) + 503 GB RAM (tag `train_20260729_152426`) and inference on 5× RTX PRO 6000 (96 GB) + 503 GB RAM over a 350-cadence `/datag` catalog subset (tag `inf_20260731_182011`) — via the always-on resource instrumentation (`system_resources` DB rows). They characterize full-scale runs; small runs need substantially less.

### Install From PyPI (pip)

For **off-cluster** use, Aetherscan is published on PyPI. The container stays canonical on the clusters (and is **mandatory on Blackwell** — see the caveats):

```bash
pip install aetherscan

# Data roots (default to /datax/scratch/zachy/... on-cluster). Off-cluster, set them to writable
# paths — AETHERSCAN_DATA_PATH must hold your catalog + inputs; the model/output roots are made on demand:
export AETHERSCAN_DATA_PATH=...  AETHERSCAN_MODEL_PATH=...  AETHERSCAN_OUTPUT_PATH=...

# Bare inference (no --encoder-path/--rf-path/--config-path) resolves + downloads the release's
# matching HF weights (revision v<installed version>):
python -m aetherscan.main inference --inference-files catalog.csv --save-tag inf
```

**If the version you resolve is v1.0.0** — whether by pinning `aetherscan==1.0.0` or because it is still the newest published release when you install — that release predates the packaged legacy-Keras fix ([#323](https://github.com/zachtheyek/Aetherscan/issues/323), fixed by [#340](https://github.com/zachtheyek/Aetherscan/pull/340) from v1.1.0): its manifest doesn't pull `tf_keras`, so you must add the two-step workaround yourself — `pip install "tf_keras~=2.17.0"` and `export TF_USE_LEGACY_KERAS=1` — or the released Keras-2 weights fail to load (see [KNOWN_ISSUES.md](KNOWN_ISSUES.md#19-v100-pipconda-install-cannot-load-the-encoder-legacy-keras)).

**Caveats:**

- **Blackwell (sm_120) needs the NGC container, not pip.** pip's `tensorflow[and-cuda]` ships CUDA-12.3 kernels with no sm_120 support, so a Blackwell GPU raises `CUDA_ERROR_INVALID_PTX`. Ampere works on the pip path.
- **No CPU mode.** Both `train` and `inference` hard-exit when no GPU is visible (`"… requires GPU"`).
- **The two end-of-run report PNGs do not render on a pip install.** The wheel ships only `src/aetherscan` (not `utils/`), so `benchmark_report.py` / `perband_report.py` aren't found — those two plots log a warning and skip; the inference viz suite, DB, and results are unaffected. Use the container or source tree if you need them.
- The live dashboard needs the extra: `pip install 'aetherscan[dashboard]'`.
- From **v1.1.0** the manifests declare `tf_keras` and the package sets `TF_USE_LEGACY_KERAS` itself at import time — no manual steps. Only a **v1.0.0** install (pinned, or resolved while it is the newest published release) needs the workaround above.

### Run From Container

> [!NOTE]
> This is the canonical install path, and the only option for Blackwell clusters

**1. Clone the repository**

```bash
git clone https://github.com/zachtheyek/Aetherscan.git
cd Aetherscan
```

**2. Get the `.sif` image**

`utils/run_container.sh` acquires the image on first use, in priority order: **(1)** use a local `aetherscan-ngc25.02.sif` if present; **(2)** else **pull the release-pinned image from GHCR** (`ghcr.io/zachtheyek/aetherscan:v<version>`, derived from `pyproject.toml`) and cache it as that `.sif`; **(3)** else fail loudly with build instructions. So on a release checkout you normally build nothing — the first `run_container.sh` call pulls and caches the image (the runtime converts the OCI image to its own native `.sif`, so the same published image works under both Apptainer and SingularityCE).

> [!NOTE]
> **On a hardened HPC node, set `SINGULARITY_TMPDIR` / `SINGULARITY_CACHEDIR` (or the `APPTAINER_*` equivalents) to scratch _before_ the first pull** — whether that pull comes from `run_container.sh` or the manual commands below. The runtime unpacks the ~9 GB image through them exactly as a build does, so leaving them unset fills `TMPDIR`'s default `/tmp` (the staging area needs ~15 GB) and/or the blob cache under `$HOME` — either of which is likely to fail on a quota'd or hardened node. See [`docs/GPU_RUNTIME_GUIDE.md`](docs/GPU_RUNTIME_GUIDE.md#hardened-hpc-nodes).

To pre-pull explicitly (optional):

```bash
# Apptainer (Ampere) or SingularityCE (Blackwell) — either converts the OCI image to a native .sif
apptainer   pull aetherscan-ngc25.02.sif docker://ghcr.io/zachtheyek/aetherscan:v1.1.0
singularity pull aetherscan-ngc25.02.sif docker://ghcr.io/zachtheyek/aetherscan:v1.1.0
```

A **manual** pull (or build) writes no `<sif>.pulled-tag` sidecar, so the wrapper treats the result like a local build and keeps it across version bumps. Let `run_container.sh` do the pulling if you want it to track the pinned ref (`repo:tag`, so both a version bump and an `AETHERSCAN_IMAGE` change trigger a re-pull) for you; otherwise `rm` the `.sif` when you bump versions.

**Build locally instead** — necessary when the prebuilt image doesn't fit the host:

- **non-x86_64 host** (e.g. aarch64 Grace/GH200): the published image is `linux/amd64` only;
- **host driver below the base's CUDA 12.8 floor** (Blackwell <570 / Ampere <550): a pull succeeds but the container won't see the GPUs — upgrade the driver, or build;
- **you edited `requirements-container.txt` or rebuilt TF from source** locally: a pull fetches the *released* image, not your variant;
- **no matching published tag exists yet** — e.g. a `master`/`.devN` clone before the next release moves `:latest` (the pull has nothing to fetch, so the wrapper prints these build instructions and exits).

A local build placed over the default `.sif` path is safe: `run_container.sh` caches pulled images with a `<sif>.pulled-tag` sidecar and detects a locally-built `.sif` by mtime, so your build is kept and never overwritten by a pull — even across version bumps.

The same [`aetherscan.def`](aetherscan.def) recipe builds with either runtime. Build on the cluster you intend to run on so the resulting `.sif` is produced by that cluster's native runtime:

```bash
# SingularityCE (e.g. Blackwell cluster running 4.1.1)
singularity build aetherscan-ngc25.02.sif aetherscan.def

# Apptainer (e.g. Ampere cluster running v1.4.5)
apptainer build aetherscan-ngc25.02.sif aetherscan.def
```

Build takes ~9 minutes and produces a ~9 GB image. On hardened HPC nodes a build additionally needs the `--fakeroot` flag, and the `noexec /tmp` FATAL it can hit is fixed by the same `TMPDIR`/`CACHEDIR` scratch redirect the pull path needs (see the note above — the symptom is build-only, the remedy is shared); the full troubleshooting walkthrough lives in [`docs/GPU_RUNTIME_GUIDE.md`](docs/GPU_RUNTIME_GUIDE.md#hardened-hpc-nodes).

> [!NOTE]
> The GHCR image is a derivative of NVIDIA's NGC TensorFlow container, so it is governed by the [NVIDIA Deep Learning Container License](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf) — not the repo's BSD-3-Clause, which covers only the (bind-mounted) Aetherscan source.

**3. Set up monitoring dashboards in tmux (optional)**

> [!Tip]
> Subsequent pipeline runs may proceed from the current step (3) onward

The repo ships a convenience script that instantiates a four-window tmux session for monitoring system resources (`htop` + a CPU/MEM ticker), GPU state (`watch nvidia-smi`), shared memory buffers (`watch ls /dev/shm`), and models/outputs dirs (`watch tree`):

```bash
./utils/start_tmux_session.sh
```

Idempotent — re-running attaches to the existing session instead of recreating it.

**4. Configure secrets and paths (optional)**

Aetherscan reads secrets and path overrides from a `.env` file at the repo root. [`utils/run_container.sh`](utils/run_container.sh) auto-loads `<repo>/.env` into its own environment before launching the container and forwards the relevant keys via `--env`, so no `source .env` or inline prefix is needed.

```ini
# .env example

# If none specified, defaults to /datax/scratch/zachy/{data|models|outputs}/aetherscan
# Note, CLI flags (--data-path, --model-path, --output-path) override these
AETHERSCAN_DATA_PATH=/path/to/data
AETHERSCAN_MODEL_PATH=/path/to/models
AETHERSCAN_OUTPUT_PATH=/path/to/outputs

# Optional: comma-separated extra host paths for run_container.sh to bind 1:1, for
# data outside the standard dirs (e.g. parent dir with raw .h5 files for inference)
AETHERSCAN_EXTRA_BINDS=/extra/host/paths

# If none specified, Slack integration is automatically disabled
SLACK_BOT_TOKEN=your-slack-bot-token
SLACK_CHANNEL=your-slack-channel

# Only needed for uploading model weights to the HuggingFace Hub (train --hf-upload);
# downloads (the inference default) hit a public repo and need no token
HF_TOKEN=your-huggingface-write-token

# Optional: redirect the HuggingFace download cache off $HOME — must be an existing absolute
# dir; run_container.sh binds + forwards it (equally settable in ~/.bashrc)
# HF_HOME=/path/to/hf_home
```

> [!TIP]
> See [`SECURITY.md`](SECURITY.md) for best practices on managing `.env` files.

If you'd rather set them directly in your shell (skipping `.env`), `export` works equivalently and takes precedence over `.env` for any keys it sets — useful for one-off overrides:

```bash
export SLACK_BOT_TOKEN="your-slack-bot-token"
export SLACK_CHANNEL="your-slack-channel"
...

./utils/run_container.sh python -m aetherscan.main train ...
```

The `AETHERSCAN_*` paths are bind-mounted 1:1 between host and container, so they must already exist on the host before the pipeline starts. The `utils/run_container.sh` wrapper forwards `SLACK_*`, `AETHERSCAN_*`, `HF_TOKEN`, and (when set) `HF_HOME` into the container explicitly — `HF_HOME` is additionally bound 1:1, so it must be an existing absolute directory; if you need additional env vars on the container side, extend the wrapper's `--env` list.

**5. Run pipeline**

```bash
./utils/run_container.sh python -m aetherscan.main {train|inference} \
  --save-tag train
```

The `utils/run_container.sh` wrapper auto-detects whether `apptainer` or `singularity` is on PATH (Apptainer wins when both are present), sets `--nv` for GPU passthrough, and binds the repo + `AETHERSCAN_{DATA,MODEL,OUTPUT}_PATH` 1:1 between host and container so absolute paths persisted in the DB stay valid across both. `PYTHONPATH` is set automatically inside the container — no inline prefix needed.

See the [Usage Examples](#usage-examples) section below for further ways to invoke the Aetherscan pipeline.

### Run From Source

> [!NOTE]
> This is an alternative install path for Ampere clusters

**1. Clone the repository**

```bash
git clone https://github.com/zachtheyek/Aetherscan.git
cd Aetherscan
```

**2. Create conda environment**

```bash
conda env create -f environment.yml
conda activate aetherscan
```

**3. Set up monitoring dashboards in tmux (optional)**

> [!Tip]
> Subsequent pipeline runs may proceed from the current step (3) onward

The repo ships a convenience script that instantiates a four-window tmux session for monitoring system resources (`htop` + a CPU/MEM ticker), GPU state (`watch nvidia-smi`), shared memory buffers (`watch ls /dev/shm`), and models/outputs dirs (`watch tree`):

```bash
./utils/start_tmux_session.sh
```

Idempotent — re-running attaches to the existing session instead of recreating it.

> [!Note]
> If you skip the tmux helper, it's recommended to run these two exports manually before launching the pipeline — the script's pipeline pane sets them for you, and without them you may hit TF library-loading issues or noisy startup logs:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TF_CPP_MIN_LOG_LEVEL=1
```

**4. Configure secrets and paths (optional)**

Same `.env` file format and precedence rules as [Run From Container](#run-from-container) step 4. Two differences on this path:

- `<repo>/.env` is loaded directly into `os.environ` at the top of `main.py` via [python-dotenv](https://pypi.org/project/python-dotenv/) — no wrapper script in the loop — so **every** key in `.env` is visible to the pipeline, not just the subset the container wrapper forwards via `--env`.
- No host→container bind mounts, so `AETHERSCAN_*` paths only need to exist when the pipeline actually accesses them, not at startup.

Multiprocess worker pools inherit the values via `os.environ` as usual.

**5. Run pipeline**

```bash
PYTHONPATH=src python -m aetherscan.main {train|inference} \
  --save-tag train
```

`PYTHONPATH=src` makes the `aetherscan` package importable from `src/` without a `pip install -e .` step. No inline `KEY=VALUE` prefix is needed for Slack credentials — the `.env` auto-load runs before any worker process is spawned, so `os.environ` inheritance to multiprocess pools is automatic.

See the [Usage Examples](#usage-examples) section below for further ways to invoke the Aetherscan pipeline.

---

## Usage Examples

> [!NOTE]
> `main.py` is the designated pipeline entry point.
> Non-development workflows should avoid directly calling other scripts/modules.

> [!NOTE]
> Each scenario below is shown twice — first with the container wrapper (canonical), then with the conda-env source invocation (alternative). CLI flags are identical between the two; only the launcher differs.

### Training

> [!TIP]
> The examples below cover a small number of scenarios. For the full set of available flags, see [Train Command Help](#train-command-help).

**Default training run**

```bash
# Container (canonical)
./utils/run_container.sh python -m aetherscan.main train

# Source (Ampere conda env)
PYTHONPATH=src python -m aetherscan.main train
```

**Training with custom parameters**

```bash
# Container
./utils/run_container.sh python -m aetherscan.main train \
    --train-files real_filtered_LARGE_HIP110750.npy real_filtered_LARGE_HIP13402.npy real_filtered_LARGE_HIP8497.npy \
    --num-training-rounds 10 \
    --epochs-per-round 100 \
    --curriculum-schedule exponential \
    --save-tag test

# Source
PYTHONPATH=src python -m aetherscan.main train \
    --train-files real_filtered_LARGE_HIP110750.npy real_filtered_LARGE_HIP13402.npy real_filtered_LARGE_HIP8497.npy \
    --num-training-rounds 10 \
    --epochs-per-round 100 \
    --curriculum-schedule exponential \
    --save-tag test
```

**Resume from checkpoint**

```bash
# Container
./utils/run_container.sh python -m aetherscan.main train \
    --load-dir checkpoints \
    --load-tag round_10 \
    --save-tag test

# Source
PYTHONPATH=src python -m aetherscan.main train \
    --load-dir checkpoints \
    --load-tag round_10 \
    --save-tag test
```

> [!WARNING]
> Per-round checkpoints live under `checkpoints/` — `--load-tag round_XX` without `--load-dir checkpoints` is rejected at validation (it used to silently resume from the newest stale model in the models root instead).

**Training with an explicit per-GPU memory cap (e.g. on an older Ampere GPU with lower VRAM)**

```bash
# Container
./utils/run_container.sh python -m aetherscan.main train \
    --gpu-memory-limit-mb 14000 \
    --save-tag test

# Source
PYTHONPATH=src python -m aetherscan.main train \
    --gpu-memory-limit-mb 14000 \
    --save-tag test
```

**Watching the live dashboard from your local browser (SSH port forwarding)**

Each `train`/`inference` run auto-launches a Streamlit dashboard on the cluster node (enabled by
default), served on that node's `localhost:8501` (`config.monitor.dashboard_port`, default `8501`).
It reads the run's live SQLite DB, so it updates as the pipeline progresses. Because it binds to the
node's loopback interface, view it locally by opening an SSH tunnel that forwards the port, then
browsing to the forwarded address:

```bash
# From your local machine — forward the dashboard port from the cluster node running the pipeline:
ssh -L 8501:localhost:8501 <cluster-host>
# ...then open http://localhost:8501 in your local browser. Keep the tunnel open while you watch.
```

If you launched the run on a non-default port, forward that port instead. The pipeline also logs the
exact `ssh -L …` command when it starts the dashboard.

### Inference

> [!TIP]
> The examples below cover a small number of scenarios. For the full set of available flags, see [Inference Command Help](#inference-command-help).

**Default inference run**

```bash
# Container (canonical)
./utils/run_container.sh python -m aetherscan.main inference

# Source (Ampere conda env)
PYTHONPATH=src python -m aetherscan.main inference
```

**Inference on a pre-processed `.npy` file**

```bash
# Container
./utils/run_container.sh python -m aetherscan.main inference \
    --test-files real_filtered_LARGE_test_HIP15638.npy \
    --encoder-path /datax/scratch/zachy/models/aetherscan/vae_encoder_train_20260101_120000.keras \
    --rf-path /datax/scratch/zachy/models/aetherscan/random_forest_train_20260101_120000.joblib \
    --config-path /datax/scratch/zachy/models/aetherscan/config_train_20260101_120000.json \
    --classification-threshold 0.99

# Source
PYTHONPATH=src python -m aetherscan.main inference \
    --test-files real_filtered_LARGE_test_HIP15638.npy \
    --encoder-path /datax/scratch/zachy/models/aetherscan/vae_encoder_train_20260101_120000.keras \
    --rf-path /datax/scratch/zachy/models/aetherscan/random_forest_train_20260101_120000.joblib \
    --config-path /datax/scratch/zachy/models/aetherscan/config_train_20260101_120000.json \
    --classification-threshold 0.99
```

**Inference from raw `.h5` files (invokes energy detection preprocessing)**

```bash
# Container — if the raw .h5 paths in the CSV live outside the standard bind
# mounts (e.g. under /datag), then we bind them via AETHERSCAN_EXTRA_BINDS
AETHERSCAN_EXTRA_BINDS=/datag ./utils/run_container.sh python -m aetherscan.main inference \
    --inference-files complete_cadences_catalog.csv \
    --encoder-path /path/to/vae_encoder.keras \
    --rf-path /path/to/random_forest.joblib \
    --config-path /path/to/config.json \
    --save-tag inf

# Source
PYTHONPATH=src python -m aetherscan.main inference \
    --inference-files complete_cadences_catalog.csv \
    --encoder-path /path/to/vae_encoder.keras \
    --rf-path /path/to/random_forest.joblib \
    --config-path /path/to/config.json \
    --save-tag inf
```

**Inference with async-allocator fallbacks (e.g. on a 5-GPU Blackwell topology)**

```bash
# Container
./utils/run_container.sh python -m aetherscan.main inference \
    --no-async-allocator \
    --save-tag inf

# Source
PYTHONPATH=src python -m aetherscan.main inference \
    --no-async-allocator \
    --save-tag inf
```

---

## CLI Reference

Aetherscan uses a hierarchical configuration system with dataclass-based configs, whose state can be modified both at command time and runtime. At command time, the user can specify values via:

1. **Defaults** - Defined in `src/aetherscan/config.py`
2. **Environment variables** - For paths and secrets
3. **CLI flags** - Override defaults & environment variables on startup

At runtime, the singleton `Config` instance can be accessed via `get_config()` and modified programmatically.

Read [docs/CONFIG_AND_CLI.md](/docs/CONFIG_AND_CLI.md) to learn more.

### Top-Level Help

Aetherscan dispatches to one of two subcommands via the first positional argument. Regenerate this output with `./utils/run_container.sh python utils/print_cli_help.py top` (container) or `PYTHONPATH=src python utils/print_cli_help.py top` (source).

```
usage: [-h] {train,inference} ...

Aetherscan Pipeline -- Breakthrough Listen's first end-to-end production-grade
DL pipeline for SETI @ scale

positional arguments:
  {train,inference}  Command to execute
    train            Execute training pipeline
    inference        Execute inference pipeline

options:
  -h, --help         show this help message and exit
```

### Train Command Help

The Aetherscan training pipeline exposes the following CLI flags to the user. Regenerate this output with `./utils/run_container.sh python utils/print_cli_help.py train` (container) or `PYTHONPATH=src python utils/print_cli_help.py train` (source).

```
usage: train [-h] [--seed SEED] [--unseeded]
             [--tf-deterministic-ops | --no-tf-deterministic-ops]
             [--n-processes N_PROCESSES] [--data-path DATA_PATH]
             [--model-path MODEL_PATH] [--output-path OUTPUT_PATH]
             [--dashboard | --no-dashboard] [--dashboard-port DASHBOARD_PORT]
             [--benchmark-report | --no-benchmark-report]
             [--vae-latent-dim VAE_LATENT_DIM]
             [--vae-dense-layer-size VAE_DENSE_LAYER_SIZE]
             [--vae-kernel-size VAE_KERNEL_SIZE VAE_KERNEL_SIZE]
             [--vae-beta VAE_BETA] [--vae-alpha VAE_ALPHA]
             [--rf-n-estimators RF_N_ESTIMATORS] [--rf-bootstrap RF_BOOTSTRAP]
             [--rf-max-features RF_MAX_FEATURES] [--rf-n-jobs RF_N_JOBS]
             [--rf-seed RF_SEED] [--num-replicas NUM_REPLICAS]
             [--gpu-memory-limit-mb GPU_MEMORY_LIMIT_MB]
             [--nccl-num-packs NCCL_NUM_PACKS]
             [--async-allocator | --no-async-allocator]
             [--num-observations NUM_OBSERVATIONS] [--width-bin WIDTH_BIN]
             [--downsample-factor DOWNSAMPLE_FACTOR] [--time-bins TIME_BINS]
             [--freq-resolution FREQ_RESOLUTION]
             [--time-resolution TIME_RESOLUTION]
             [--num-target-backgrounds NUM_TARGET_BACKGROUNDS]
             [--background-load-chunk-size BACKGROUND_LOAD_CHUNK_SIZE]
             [--max-chunks-per-file MAX_CHUNKS_PER_FILE]
             [--train-files TRAIN_FILES [TRAIN_FILES ...]]
             [--num-training-rounds NUM_TRAINING_ROUNDS]
             [--epochs-per-round EPOCHS_PER_ROUND]
             [--num-samples-beta-vae NUM_SAMPLES_BETA_VAE]
             [--num-samples-rf NUM_SAMPLES_RF]
             [--train-val-split TRAIN_VAL_SPLIT]
             [--per-replica-batch-size PER_REPLICA_BATCH_SIZE]
             [--effective-batch-size EFFECTIVE_BATCH_SIZE]
             [--per-replica-val-batch-size PER_REPLICA_VAL_BATCH_SIZE]
             [--signal-injection-chunk-size SIGNAL_INJECTION_CHUNK_SIZE]
             [--data-gen-task-size DATA_GEN_TASK_SIZE]
             [--round-data-dir ROUND_DATA_DIR]
             [--overlap-data-generation | --no-overlap-data-generation]
             [--keep-round-data | --no-keep-round-data]
             [--plot-injection-subsampling-count PLOT_INJECTION_SUBSAMPLING_COUNT]
             [--plot-injection-outlier-percentile PLOT_INJECTION_OUTLIER_PERCENTILE]
             [--latent-viz-num-cadences-per-type LATENT_VIZ_NUM_CADENCES_PER_TYPE]
             [--latent-viz-step-interval LATENT_VIZ_STEP_INTERVAL]
             [--latent-viz-umap-fit-max-samples LATENT_VIZ_UMAP_FIT_MAX_SAMPLES]
             [--latent-viz-umap-n-neighbors LATENT_VIZ_UMAP_N_NEIGHBORS [LATENT_VIZ_UMAP_N_NEIGHBORS ...]]
             [--latent-viz-umap-min-dist LATENT_VIZ_UMAP_MIN_DIST [LATENT_VIZ_UMAP_MIN_DIST ...]]
             [--latent-viz-gif-max-frames LATENT_VIZ_GIF_MAX_FRAMES]
             [--latent-viz-gif-duration-ms LATENT_VIZ_GIF_DURATION_MS]
             [--latent-traversal-every-round | --no-latent-traversal-every-round]
             [--latent-traversal-num-steps LATENT_TRAVERSAL_NUM_STEPS]
             [--latent-traversal-max-sigma LATENT_TRAVERSAL_MAX_SIGMA]
             [--snr-base SNR_BASE] [--initial-snr-range INITIAL_SNR_RANGE]
             [--final-snr-range FINAL_SNR_RANGE]
             [--curriculum-schedule CURRICULUM_SCHEDULE]
             [--exponential-decay-rate EXPONENTIAL_DECAY_RATE]
             [--step-easy-rounds STEP_EASY_ROUNDS]
             [--step-hard-rounds STEP_HARD_ROUNDS]
             [--base-learning-rate BASE_LEARNING_RATE]
             [--min-learning-rate MIN_LEARNING_RATE]
             [--min-pct-improvement MIN_PCT_IMPROVEMENT]
             [--patience-threshold PATIENCE_THRESHOLD]
             [--lr-reduction-factor LR_REDUCTION_FACTOR]
             [--max-retries MAX_RETRIES] [--retry-delay RETRY_DELAY]
             [--hf-upload | --no-hf-upload] [--hf-repo-id HF_REPO_ID]
             [--load-dir LOAD_DIR] [--load-tag LOAD_TAG]
             [--start-round START_ROUND] [--save-tag SAVE_TAG]
             [--force-tag | --no-force-tag]

options:
  -h, --help            show this help message and exit
  --seed SEED           Root random seed for reproducible runs: every random
                        stream derives from it — data generation, dataset
                        split/shuffles, TF weight init, the VAE sampling layer
                        (training AND inference), the random forest,
                        UMAP/KMeans plot fits, and plot subsampling. Defaults
                        to a concrete value (reproducible out of the box);
                        must be >= 0. To run unseeded, pass --unseeded
  --unseeded            Opt OUT of the seeded default: draw every random
                        stream from OS entropy (non-reproducible). Mutually
                        exclusive with --seed
  --tf-deterministic-ops, --no-tf-deterministic-ops
                        Force deterministic TensorFlow/cuDNN op
                        implementations
                        (tf.config.experimental.enable_op_determinism) for
                        bit-exact GPU reproducibility at some speed cost.
                        Default: enabled — without it, cuDNN autotune noise
                        can flip near-threshold candidates between identical
                        runs; opt out with --no-tf-deterministic-ops
  --n-processes N_PROCESSES
                        Worker-process count for the multiprocessing pools
                        (energy detection + stamp extraction at inference;
                        data generation at training). Default: all cores. Host
                        tuning: never layered from a saved --config-path, so a
                        config recorded on a bigger host cannot oversubscribe
                        this one (must be >= 1)
  --data-path DATA_PATH
                        Path to data directory (overrides AETHERSCAN_DATA_PATH
                        environment variable)
  --model-path MODEL_PATH
                        Path to model directory (overrides
                        AETHERSCAN_MODEL_PATH environment variable)
  --output-path OUTPUT_PATH
                        Path to output directory (overrides
                        AETHERSCAN_OUTPUT_PATH environment variable)
  --dashboard, --no-dashboard
                        Auto-launch the live monitoring Streamlit dashboard
                        for this run; SSH-forward the port to view it
                        (default: on). Use --no-dashboard to disable
  --dashboard-port DASHBOARD_PORT
                        Port for the auto-launched live dashboard (default:
                        8501)
  --benchmark-report, --no-benchmark-report
                        Render the end-of-run benchmark report (stage timeline
                        + bottleneck suggestions) and post it to Slack
                        (default: on). Use --no-benchmark-report to disable
  --vae-latent-dim VAE_LATENT_DIM
                        Dimensionality of the VAE latent space (bottleneck
                        size)
  --vae-dense-layer-size VAE_DENSE_LAYER_SIZE
                        Size of dense layer in VAE architecture (should match
                        frequency bins after downsampling)
  --vae-kernel-size VAE_KERNEL_SIZE VAE_KERNEL_SIZE
                        Kernel size for Conv2D layers as two integers (e.g.,
                        --vae-kernel-size 3 3)
  --vae-beta VAE_BETA   Beta coefficient for KL divergence loss term in beta-
                        VAE (controls disentanglement)
  --vae-alpha VAE_ALPHA
                        Alpha coefficient for clustering loss term in VAE
                        (controls cluster separation)
  --rf-n-estimators RF_N_ESTIMATORS
                        Number of decision trees in the random forest ensemble
  --rf-bootstrap RF_BOOTSTRAP
                        Whether to use bootstrap sampling when building trees
                        (enables bagging)
  --rf-max-features RF_MAX_FEATURES
                        Number of features to consider for splits: 'sqrt',
                        'log2', or a float (fraction of features)
  --rf-n-jobs RF_N_JOBS
                        Number of parallel jobs for random forest training (-1
                        uses all CPU cores)
  --rf-seed RF_SEED     DEPRECATED: explicit random forest seed override. The
                        RF seed now derives from the root --seed (#279); this
                        alias remains for existing scripts and logs a
                        deprecation warning when used.
  --num-replicas NUM_REPLICAS
                        Number of GPUs to use for the distributed strategy. If
                        omitted, the strategy uses every GPU visible to TF;
                        otherwise it is restricted to the first N physical
                        GPUs and the rest are left untouched. Must be >= 1 and
                        <= the number of physical GPUs on your machine.
  --gpu-memory-limit-mb GPU_MEMORY_LIMIT_MB
                        Per-GPU memory cap in MiB. Omit to use memory-growth-
                        only (recommended on Blackwell). Set for TF to
                        allocate a fixed logical device of a given size per
                        physical GPU (e.g. 14000)
  --nccl-num-packs NCCL_NUM_PACKS
                        num_packs for NCCL/HierarchicalCopy all-reduce. Lower
                        values (e.g. 1) reduces tiny-tensor latency; higher
                        values (e.g. >=4) can help bandwidth on >4-GPU
                        topologies.
  --async-allocator, --no-async-allocator
                        Toggle TF_GPU_ALLOCATOR=cuda_malloc_async (default:
                        enabled). Pass --no-async-allocator as a workaround
                        for NGC 25.02 multi-GPU OOM bugs.
  --num-observations NUM_OBSERVATIONS
                        Number of observations per cadence snippet (e.g., 6
                        for 3 ON + 3 OFF)
  --width-bin WIDTH_BIN
                        Number of frequency bins per observation (spectral
                        resolution)
  --downsample-factor DOWNSAMPLE_FACTOR
                        Downsampling factor for frequency bins (reduces
                        spectral dimension)
  --time-bins TIME_BINS
                        Number of time bins per observation (temporal
                        resolution)
  --freq-resolution FREQ_RESOLUTION
                        Frequency resolution in Hz (determined by instrument)
  --time-resolution TIME_RESOLUTION
                        Time resolution in seconds (determined by instrument)
  --num-target-backgrounds NUM_TARGET_BACKGROUNDS
                        Number of background (noise-only) cadences to load for
                        training data generation
  --background-load-chunk-size BACKGROUND_LOAD_CHUNK_SIZE
                        Maximum number of background cadences to process at
                        once during loading (memory management)
  --max-chunks-per-file MAX_CHUNKS_PER_FILE
                        Maximum number of chunks to load from a single data
                        file (limits per-file contribution)
  --train-files TRAIN_FILES [TRAIN_FILES ...]
                        Space-separated list of training data file names
                        (e.g., real_filtered_LARGE_HIP110750.npy)
  --num-training-rounds NUM_TRAINING_ROUNDS
                        Total number of training rounds in curriculum learning
                        schedule
  --epochs-per-round EPOCHS_PER_ROUND
                        Number of epochs to train the VAE per curriculum
                        learning round
  --num-samples-beta-vae NUM_SAMPLES_BETA_VAE
                        Number of training samples to generate for beta-VAE
                        per round (must be divisible by 4)
  --num-samples-rf NUM_SAMPLES_RF
                        Number of training samples to generate for random
                        forest (must be divisible by 4)
  --train-val-split TRAIN_VAL_SPLIT
                        Fraction of data to use for training vs validation
                        (e.g., 0.8 = 80% train, 20% val)
  --per-replica-batch-size PER_REPLICA_BATCH_SIZE
                        Batch size per GPU/device replica during training
  --effective-batch-size EFFECTIVE_BATCH_SIZE
                        Effective batch size for gradient accumulation across
                        all replicas
  --per-replica-val-batch-size PER_REPLICA_VAL_BATCH_SIZE
                        Batch size per GPU/device replica during validation
  --signal-injection-chunk-size SIGNAL_INJECTION_CHUNK_SIZE
                        Maximum cadences to process at once during synthetic
                        signal injection (must be divisible by 4)
  --data-gen-task-size DATA_GEN_TASK_SIZE
                        Cadences per batched signal-injection worker task
                        (workers write results straight into the round's on-
                        disk memmap; must be >= 1)
  --round-data-dir ROUND_DATA_DIR
                        Directory for disk-backed per-round training datasets
                        (defaults to <data-path>/training/round_data; needs
                        ~2.2x one round's size free when data-generation
                        overlap is enabled, ~1.1x otherwise)
  --overlap-data-generation, --no-overlap-data-generation
                        Generate round k+1's training data in a background
                        producer process while round k trains (default:
                        enabled). Pass --no-overlap-data-generation to fall
                        back to sequential in-process generation for debugging
  --keep-round-data, --no-keep-round-data
                        Retain each round's on-disk training data after that
                        round finishes (default: disabled — round k's data
                        directory is deleted as soon as round k's training
                        completes). Enable for debugging
  --plot-injection-subsampling-count PLOT_INJECTION_SUBSAMPLING_COUNT
                        Max points per stat name, per signal type, for A→B
                        intensity bias scatter plots. Outliers are
                        prioritized, with the difference made up from randomly
                        sampling without replacement the remaining points
  --plot-injection-outlier-percentile PLOT_INJECTION_OUTLIER_PERCENTILE
                        Threshold for points to always be included in A→B
                        intensity bias scatter plots
  --latent-viz-num-cadences-per-type LATENT_VIZ_NUM_CADENCES_PER_TYPE
                        Number of cadences per signal type for latent space
                        visualization batch (total points = 4× this value × 6
                        observations per cadence)
  --latent-viz-step-interval LATENT_VIZ_STEP_INTERVAL
                        Capture a latent space snapshot every N training steps
                        (lower = more snapshots, more DB writes, and larger
                        storage costs)
  --latent-viz-umap-fit-max-samples LATENT_VIZ_UMAP_FIT_MAX_SAMPLES
                        Maximum number of pooled latent vectors used to fit
                        the UMAP model (remaining vectors are projected via
                        transform; lower = faster, higher = more faithful
                        embedding)
  --latent-viz-umap-n-neighbors LATENT_VIZ_UMAP_N_NEIGHBORS [LATENT_VIZ_UMAP_N_NEIGHBORS ...]
                        UMAP n_neighbors values to sweep for latent space
                        visualization (e.g., --latent-viz-umap-n-neighbors 5
                        15 30 50)
  --latent-viz-umap-min-dist LATENT_VIZ_UMAP_MIN_DIST [LATENT_VIZ_UMAP_MIN_DIST ...]
                        UMAP min_dist values to sweep for latent space
                        visualization (e.g., --latent-viz-umap-min-dist 0.0
                        0.1 0.5)
  --latent-viz-gif-max-frames LATENT_VIZ_GIF_MAX_FRAMES
                        Maximum number of frames in latent space GIF output
                        (snapshots beyond this limit are log-subsampled,
                        prioritizing earlier training steps)
  --latent-viz-gif-duration-ms LATENT_VIZ_GIF_DURATION_MS
                        Milliseconds per frame in latent space GIF output
  --latent-traversal-every-round, --no-latent-traversal-every-round
                        Render latent-dimension traversal figures at the end
                        of every training round, in addition to the end-of-
                        training set (default: disabled)
  --latent-traversal-num-steps LATENT_TRAVERSAL_NUM_STEPS
                        Number of traversal steps per latent dimension (must
                        be odd and >= 3 so the center column is the
                        unperturbed class-mean decode)
  --latent-traversal-max-sigma LATENT_TRAVERSAL_MAX_SIGMA
                        Latent traversal range in per-dimension standard
                        deviations: steps span [-max_sigma, +max_sigma] (must
                        be > 0)
  --snr-base SNR_BASE   Base signal-to-noise ratio for curriculum learning
                        (minimum SNR difficulty level)
  --initial-snr-range INITIAL_SNR_RANGE
                        SNR range for initial (easiest) training rounds
                        (signals sampled from snr_base to snr_base +
                        initial_snr_range)
  --final-snr-range FINAL_SNR_RANGE
                        SNR range for final (hardest) training rounds (signals
                        sampled from snr_base to snr_base + final_snr_range).
                        Ignored if only training for 1 round
  --curriculum-schedule CURRICULUM_SCHEDULE
                        Curriculum difficulty progression schedule: 'linear',
                        'exponential', or 'step'
  --exponential-decay-rate EXPONENTIAL_DECAY_RATE
                        Decay rate for exponential curriculum schedule (must
                        be negative; more negative = faster difficulty
                        increase)
  --step-easy-rounds STEP_EASY_ROUNDS
                        Number of rounds with easy signals when using step
                        curriculum schedule
  --step-hard-rounds STEP_HARD_ROUNDS
                        Number of rounds with hard signals when using step
                        curriculum schedule
  --base-learning-rate BASE_LEARNING_RATE
                        Initial learning rate for Adam optimizer
  --min-learning-rate MIN_LEARNING_RATE
                        Learning rate floor for adaptive learning rate
                        reduction
  --min-pct-improvement MIN_PCT_IMPROVEMENT
                        Minimum fractional validation loss improvement to
                        avoid LR reduction (e.g., 0.001 = 0.1%)
  --patience-threshold PATIENCE_THRESHOLD
                        Number of consecutive epochs without minimum
                        improvement before reducing learning rate
  --lr-reduction-factor LR_REDUCTION_FACTOR
                        Multiplicative factor for learning rate reduction
                        (e.g., 0.2 reduces LR by 20%)
  --max-retries MAX_RETRIES
                        Maximum number of retry attempts when training fails
                        due to errors
  --retry-delay RETRY_DELAY
                        Delay in seconds between retry attempts after training
                        failure
  --hf-upload, --no-hf-upload
                        Upload the final model artifacts (encoder, decoder,
                        random forest, config) plus a generated model card to
                        the HuggingFace Hub after training completes, tagging
                        the commit with --save-tag (default: disabled = local-
                        only). Requires HF_TOKEN in the environment (via .env)
  --hf-repo-id HF_REPO_ID
                        HuggingFace model repo id (namespace/name) for weight
                        upload/download (default: zachtheyek/aetherscan)
  --load-dir LOAD_DIR   Subdirectory for checkpoint loading (relative to
                        --model-path)
  --load-tag LOAD_TAG   Checkpoint to load. A full run tag
                        ({command}_YYYYMMDD_HHMMSS) resumes that run in place
                        (its tag is adopted, so the resumed attempt writes
                        under the same run). round_XX (requires --load-dir
                        checkpoints) seeds a fresh run from that per-round
                        checkpoint, resuming from round XX+1 unless --start-
                        round is given.
  --start-round START_ROUND
                        Round to begin/resume training from
  --save-tag SAVE_TAG   Run label prefix: one of test, train, inf, bench. The
                        datetime is appended automatically at runtime (e.g.
                        train_20260101_120000). Defaults to the subcommand
                        (train->train, inference->inf) if omitted.
  --force-tag, --no-force-tag
                        Override the fail-early save-tag collision guard:
                        proceed even when an explicitly-provided --save-tag
                        matches existing artifacts, DB rows, or (with --hf-
                        upload) an existing HuggingFace tag (default:
                        disabled)
```

### Inference Command Help

The Aetherscan inference pipeline exposes the following CLI flags to the user. Regenerate this output with `./utils/run_container.sh python utils/print_cli_help.py inference` (container) or `PYTHONPATH=src python utils/print_cli_help.py inference` (source).

```
usage: inference [-h] [--seed SEED] [--unseeded]
                 [--tf-deterministic-ops | --no-tf-deterministic-ops]
                 [--n-processes N_PROCESSES] [--data-path DATA_PATH]
                 [--model-path MODEL_PATH] [--output-path OUTPUT_PATH]
                 [--dashboard | --no-dashboard]
                 [--dashboard-port DASHBOARD_PORT]
                 [--benchmark-report | --no-benchmark-report]
                 [--num-replicas NUM_REPLICAS]
                 [--gpu-memory-limit-mb GPU_MEMORY_LIMIT_MB]
                 [--async-allocator | --no-async-allocator]
                 [--test-files TEST_FILES [TEST_FILES ...]]
                 [--inference-files INFERENCE_FILES [INFERENCE_FILES ...]]
                 [--encoder-path ENCODER_PATH] [--rf-path RF_PATH]
                 [--config-path CONFIG_PATH]
                 [--per-replica-batch-size PER_REPLICA_BATCH_SIZE]
                 [--classification-threshold CLASSIFICATION_THRESHOLD]
                 [--screening-threshold SCREENING_THRESHOLD]
                 [--mc-draws MC_DRAWS]
                 [--reference-cloud-size REFERENCE_CLOUD_SIZE]
                 [--prefetch-depth PREFETCH_DEPTH]
                 [--cadence-group-by-cols CADENCE_GROUP_BY_COLS [CADENCE_GROUP_BY_COLS ...]]
                 [--cadence-h5-path-col CADENCE_H5_PATH_COL]
                 [--cadence-expected-obs CADENCE_EXPECTED_OBS]
                 [--coarse-channel-width COARSE_CHANNEL_WIDTH]
                 [--coarse-channel-log-interval COARSE_CHANNEL_LOG_INTERVAL]
                 [--bandpass-method BANDPASS_METHOD]
                 [--pfb-taps-per-channel PFB_TAPS_PER_CHANNEL]
                 [--bandpass-debug-plot | --no-bandpass-debug-plot]
                 [--spline-order SPLINE_ORDER]
                 [--detection-window-size DETECTION_WINDOW_SIZE]
                 [--detection-step-size DETECTION_STEP_SIZE]
                 [--stat-threshold STAT_THRESHOLD] [--stamp-width STAMP_WIDTH]
                 [--store-downsampled-stamps | --no-store-downsampled-stamps]
                 [--overlap-search | --no-overlap-search]
                 [--overlap-fraction OVERLAP_FRACTION]
                 [--preprocess-output-dir PREPROCESS_OUTPUT_DIR]
                 [--prune-stamps | --no-prune-stamps]
                 [--inference-viz | --no-inference-viz]
                 [--inference-viz-scope {full,new}]
                 [--stamp-gallery-top-k STAMP_GALLERY_TOP_K]
                 [--max-candidate-plots MAX_CANDIDATE_PLOTS]
                 [--max-retries MAX_RETRIES] [--retry-delay RETRY_DELAY]
                 [--hf-repo-id HF_REPO_ID] [--hf-revision HF_REVISION]
                 [--save-tag SAVE_TAG] [--force-tag | --no-force-tag]

options:
  -h, --help            show this help message and exit
  --seed SEED           Root random seed for reproducible runs: every random
                        stream derives from it — data generation, dataset
                        split/shuffles, TF weight init, the VAE sampling layer
                        (training AND inference), the random forest,
                        UMAP/KMeans plot fits, and plot subsampling. Defaults
                        to a concrete value (reproducible out of the box);
                        must be >= 0. To run unseeded, pass --unseeded
  --unseeded            Opt OUT of the seeded default: draw every random
                        stream from OS entropy (non-reproducible). Mutually
                        exclusive with --seed
  --tf-deterministic-ops, --no-tf-deterministic-ops
                        Force deterministic TensorFlow/cuDNN op
                        implementations
                        (tf.config.experimental.enable_op_determinism) for
                        bit-exact GPU reproducibility at some speed cost.
                        Default: enabled — without it, cuDNN autotune noise
                        can flip near-threshold candidates between identical
                        runs; opt out with --no-tf-deterministic-ops
  --n-processes N_PROCESSES
                        Worker-process count for the multiprocessing pools
                        (energy detection + stamp extraction at inference;
                        data generation at training). Default: all cores. Host
                        tuning: never layered from a saved --config-path, so a
                        config recorded on a bigger host cannot oversubscribe
                        this one (must be >= 1)
  --data-path DATA_PATH
                        Path to data directory (overrides AETHERSCAN_DATA_PATH
                        environment variable)
  --model-path MODEL_PATH
                        Path to model directory (overrides
                        AETHERSCAN_MODEL_PATH environment variable)
  --output-path OUTPUT_PATH
                        Path to output directory (overrides
                        AETHERSCAN_OUTPUT_PATH environment variable)
  --dashboard, --no-dashboard
                        Auto-launch the live monitoring Streamlit dashboard
                        for this run; SSH-forward the port to view it
                        (default: on). Use --no-dashboard to disable
  --dashboard-port DASHBOARD_PORT
                        Port for the auto-launched live dashboard (default:
                        8501)
  --benchmark-report, --no-benchmark-report
                        Render the end-of-run benchmark report (stage timeline
                        + bottleneck suggestions) and post it to Slack
                        (default: on). Use --no-benchmark-report to disable
  --num-replicas NUM_REPLICAS
                        Number of GPUs to use for the distributed strategy. If
                        omitted, the strategy uses every GPU visible to TF;
                        otherwise it is restricted to the first N physical
                        GPUs and the rest are left untouched. Must be >= 1 and
                        <= the number of physical GPUs on your machine.
  --gpu-memory-limit-mb GPU_MEMORY_LIMIT_MB
                        Per-GPU memory cap in MiB. Omit to use memory-growth-
                        only (recommended on Blackwell). Set for TF to
                        allocate a fixed logical device of a given size per
                        physical GPU (e.g. 14000)
  --async-allocator, --no-async-allocator
                        Toggle TF_GPU_ALLOCATOR=cuda_malloc_async (default:
                        enabled). Pass --no-async-allocator as a workaround
                        for NGC 25.02 multi-GPU OOM bugs.
  --test-files TEST_FILES [TEST_FILES ...]
                        Space-separated list of testing data file names (e.g.,
                        real_filtered_LARGE_test_HIP15638.npy)
  --inference-files INFERENCE_FILES [INFERENCE_FILES ...]
                        Space-separated list of inference catalog file names
                        (e.g. complete_cadences_catalog.csv). Expects .h5
                        filepaths to individual observations, and sufficient
                        metadata for recovering cadence groupings. If
                        provided, triggers the energy detection preprocessing
                        pipeline and takes precedence over --test-files
  --encoder-path ENCODER_PATH
                        Path to trained VAE encoder model file (.keras).
                        Optional: when none of --encoder-path/--rf-
                        path/--config-path are given, the artifacts are
                        downloaded from the HuggingFace Hub (see --hf-repo-
                        id/--hf-revision); provide either all three local
                        paths or none
  --rf-path RF_PATH     Path to trained Random Forest model file (.joblib).
                        Optional: see --encoder-path for the all-three-or-none
                        rule
  --config-path CONFIG_PATH
                        Path to config file from corresponding training run
                        (.json). Optional: see --encoder-path for the all-
                        three-or-none rule
  --per-replica-batch-size PER_REPLICA_BATCH_SIZE
                        Batch size per GPU/device replica during inference
  --classification-threshold CLASSIFICATION_THRESHOLD
                        Science threshold for candidate detection, applied to
                        the pass-2 MC mean probability (the two-pass cascade's
                        final score)
  --screening-threshold SCREENING_THRESHOLD
                        Permissive pass-1 screening threshold of the two-pass
                        cascade (tuned for recall; must not exceed
                        --classification-threshold). Snippets below it are
                        rejected without MC scoring
  --mc-draws MC_DRAWS   Seeded Monte-Carlo latent draws per pass-2 survivor
                        (mean carries the science threshold; std is the
                        reported uncertainty spread)
  --reference-cloud-size REFERENCE_CLOUD_SIZE
                        Size of the seeded uniform reservoir of pass-1 rejects
                        MC-scored as the candidate uncertainty plot's survey
                        background (0 disables)
  --prefetch-depth PREFETCH_DEPTH
                        Cadences preprocessed+loaded ahead of the GPU stage in
                        the streaming loop (>= 1). Each unit of depth overlaps
                        energy-detection reads with stamp extraction and the
                        serial per-cadence sections, costing one in-flight
                        cadence of RAM (up to ~65 GB for RFI-dense C-band
                        cadences); outputs are identical at any depth
                        (default: 3 per the on-cluster A/B)
  --cadence-group-by-cols CADENCE_GROUP_BY_COLS [CADENCE_GROUP_BY_COLS ...]
                        Space-separated list of CSV column names whose joint
                        value defines cadence membership (e.g., Target Session
                        Band 'Cadence ID' Frequency)
  --cadence-h5-path-col CADENCE_H5_PATH_COL
                        CSV column containing the .h5 file path for each
                        observation (default: '.h5 path')
  --cadence-expected-obs CADENCE_EXPECTED_OBS
                        Required number of observations per cadence (default:
                        6 for ABACAD)
  --coarse-channel-width COARSE_CHANNEL_WIDTH
                        Number of fine channels per coarse channel (default:
                        1048576)
  --coarse-channel-log-interval COARSE_CHANNEL_LOG_INTERVAL
                        Progress-logging cadence for energy detection, in
                        coarse channels per log line. Default: ~25% milestone
                        lines per ON file (the per-channel lines were 62% of a
                        run's Slack-bound log volume); pass an explicit N to
                        restore every-N-channels lines. Parallelism itself
                        comes from the persistent worker pool, not this knob.
  --bandpass-method BANDPASS_METHOD
                        Bandpass flattening method for energy detection: 'pfb'
                        (default) divides each coarse channel by the
                        instrument's static polyphase-filterbank response;
                        'spline' fits and subtracts a per-channel spline
  --pfb-taps-per-channel PFB_TAPS_PER_CHANNEL
                        PFB prototype-filter taps per coarse channel for
                        --bandpass-method pfb (default: 12, the
                        GBT/Breakthrough Listen backend value). INSTRUMENT-
                        DEPENDENT: must match the backend that produced the
                        .h5 files
  --bandpass-debug-plot, --no-bandpass-debug-plot
                        Save a per-cadence bandpass-flattening overlay debug
                        plot (raw vs flattened integrated spectrum for a few
                        sampled coarse channels) under plots/inference/{save-
                        tag}/ (default: off)
  --spline-order SPLINE_ORDER
                        Spline order for bandpass fitting with --bandpass-
                        method spline (default: 16)
  --detection-window-size DETECTION_WINDOW_SIZE
                        Sliding window size in fine channels for normality
                        test (default: 256)
  --detection-step-size DETECTION_STEP_SIZE
                        Step size in fine channels for sliding window
                        (default: 128)
  --stat-threshold STAT_THRESHOLD
                        D'Agostino-Pearson statistic threshold for hit
                        detection (default: 2048.0)
  --stamp-width STAMP_WIDTH
                        Width in fine channels of the extracted stamp around
                        each hit (default: 4096; must equal --width-bin)
  --store-downsampled-stamps, --no-store-downsampled-stamps
                        Downsample stamps along frequency (by --downsample-
                        factor) at extraction time, storing stamp_width //
                        downsample_factor bins per stamp (~8x smaller at
                        defaults; default: enabled). Pass --no-store-
                        downsampled-stamps to archive raw-resolution stamps;
                        loading handles both layouts.
  --overlap-search, --no-overlap-search
                        Additionally extract stamps offset by
                        ±overlap_fraction*stamp_width around each hit. Pass
                        --no-overlap-search to disable when the config default
                        is True.
  --overlap-fraction OVERLAP_FRACTION
                        Fractional offset (relative to stamp_width) for
                        overlap-search stamps (default: 0.5)
  --preprocess-output-dir PREPROCESS_OUTPUT_DIR
                        Directory for per-cadence .npy outputs from
                        preprocessing. Default: a per-CSV directory {data_path
                        }/inference/preprocessed/<csv_stem>_ed<hash>/ keyed on
                        the energy-detection config fingerprint — runs sharing
                        an ED config reuse each other's stamps automatically,
                        and any ED-config change resolves to a fresh
                        directory. Pass a directory explicitly to pin/share
                        one location (reuse is still guarded by the sidecar's
                        recorded h5 paths and ED fingerprint)
  --prune-stamps, --no-prune-stamps
                        Delete each cadence's stamp .npy right after its
                        'inferred' manifest row lands, keeping the metadata
                        .json plus a ~196 KB snippet sidecar per candidate —
                        resume rides the DB row, and only stamps this run
                        freshly extracted are ever pruned. Without pruning a
                        full catalog writes ~30-90 TB of stamps. Default: AUTO
                        — enabled for the fingerprint-scoped default cache
                        directory, disabled when --preprocess-output-dir is
                        set explicitly. Pass --no-prune-stamps to keep every
                        stamp (slice-scale runs wanting the cross-run rerun
                        cache).
  --inference-viz, --no-inference-viz
                        Render the inference visualization suite (energy
                        detection distributions, hit spectrum, bandpass
                        overlay, stamp/candidate galleries, confidence
                        distribution, latent projection, summary card) at the
                        end of a CSV inference run, saved under
                        plots/inference/{save_tag}/ and uploaded to Slack
                        (default: enabled). Pass --no-inference-viz to
                        disable.
  --inference-viz-scope {full,new}
                        Which cadences the metadata-driven viz figures cover:
                        'full' (default) renders the whole accumulated tag
                        every successful pass; 'new' renders only cadences
                        inferred this pass — recommended for resumed multi-
                        pass catalog campaigns, where 'full' re-pays the
                        entire catalog's viz tail on every pass. DB-sourced
                        candidate figures always cover the full tag either
                        way.
  --stamp-gallery-top-k STAMP_GALLERY_TOP_K
                        Number of top-statistic stamps shown in the stamp
                        gallery figure, each as a 6-observation waterfall grid
                        (default: 12)
  --max-candidate-plots MAX_CANDIDATE_PLOTS
                        Maximum number of per-candidate figures rendered per
                        run, highest confidence first (default: 50; the
                        candidate gallery is unaffected)
  --max-retries MAX_RETRIES
                        Maximum number of retry attempts for inference
                        (including preprocessing) on failure
  --retry-delay RETRY_DELAY
                        Delay in seconds between inference retry attempts
  --hf-repo-id HF_REPO_ID
                        HuggingFace model repo id (namespace/name) for weight
                        upload/download (default: zachtheyek/aetherscan)
  --hf-revision HF_REVISION
                        HuggingFace revision (tag, branch, or commit hash) to
                        pin the model download to when no local artifact paths
                        are given (default: v{package version} when running as
                        an installed release, else the repo's latest release
                        tag — highest semver vX.Y.Z tag; a release tag is
                        required for a no-artifact download)
  --save-tag SAVE_TAG   Run label prefix: one of test, train, inf, bench. The
                        datetime is appended automatically at runtime (e.g.
                        inf_20260101_120000). Defaults to the subcommand
                        (train->train, inference->inf) if omitted.
  --force-tag, --no-force-tag
                        Override the fail-early save-tag collision guard:
                        proceed even when an explicitly-provided --save-tag
                        matches a previous run's saved config or DB rows
                        (default: disabled)
```

---

## Known Issues

For a list of known issues, limitations, and workarounds, see [`KNOWN_ISSUES.md`](/KNOWN_ISSUES.md).

---

## Contributing To Aetherscan

Contributions are welcome! Quick start:

```bash
git clone https://github.com/zachtheyek/Aetherscan.git
cd Aetherscan

# The first `utils/run_container.sh` run pulls the prebuilt image from GHCR and caches it as
# aetherscan-ngc25.02.sif. A fresh `master` clone resolves to `:latest`, which doesn't exist until
# the next release ships — so build once now (this is also the fallback for a host the published
# image can't serve). Drop this step on a release-tag checkout.
singularity build aetherscan-ngc25.02.sif aetherscan.def   # or: apptainer build ...

./utils/start_tmux_session.sh

pre-commit install
```

- PRs: Must be linked to an existing issue and pass all hooks
- Commits: Must carry a verified GPG signature — see [Commit Signing (GPG)](CONTRIBUTING.md#commit-signing-gpg)
- Branches: Use `feature/`, `hotfix/`, or `misc/` prefixes
- Code style: PEP-8 with minor relaxations, enforced via [ruff](https://docs.astral.sh/ruff/) (see [pyproject.toml](pyproject.toml))

See [`CONTRIBUTING.md`](/CONTRIBUTING.md) for full guidelines on workflow, project structure, and testing.

---

## Citations

If you use Aetherscan in your research, please cite it using GitHub's citations feature.

<p align="center">
    <img src="docs/assets/github-citation-button.png" alt="Citations">
</p>

See [`CITATION.cff`](CITATION.cff) for details

---

## Security

Aetherscan is committed to responsible disclosure. Quick reference:

- **Report vulnerabilities:** Open a [GitHub Discussion](https://github.com/zachtheyek/Aetherscan/discussions) with the "security" label (non-critical) or contact [@zachtheyek](https://breakthroughlisten.slack.com/archives/D01SJG0L0TE) on Slack (critical; expect a response within 48-72h)
- **Incident response:** Contain compromised credentials immediately, then assess scope, notify affected parties, remediate, and document
- **Secrets:** Never commit tokens; use `.env` files (gitignored). Rotate immediately if compromised
- **Automated scanning:** [gitleaks](https://github.com/gitleaks/gitleaks) pre-commit hook blocks accidental secret commits; GitHub Dependabot monitors for vulnerable dependencies

See [`SECURITY.md`](SECURITY.md) for more details.

---

## License

Aetherscan is distributed under the BSD-3-Clause license, a permissive license that allows commercial use, modification, and distribution with minimal restrictions. See [LICENSE](LICENSE) for details. All contributions to the project are assumed to be licensed under the same terms.
