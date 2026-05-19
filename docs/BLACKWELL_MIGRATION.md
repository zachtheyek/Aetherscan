# Blackwell GPU Migration Runbook

This runbook covers running Aetherscan on the new Blackwell (RTX PRO 6000) workstation alongside the existing Ampere (A4000) cluster. The pipeline source tree is identical on both machines; the difference is the runtime: a conda env on Ampere, an NVIDIA NGC container on Blackwell.

## TL;DR

| Cluster   | GPU                  | Compute capability | Runtime                                                                       |
| --------- | -------------------- | ------------------ | ----------------------------------------------------------------------------- |
| Ampere    | NVIDIA RTX A4000     | sm_86              | `conda env create -f environment.yml` (TF 2.17 + CUDA 12.5 + cuDNN 9.3)       |
| Blackwell | NVIDIA RTX PRO 6000  | sm_120             | NGC container `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` (TF 2.17 + CUDA 12.8) |

Both paths run TF 2.17 with Keras 3 so `@tf.function` tracing, `.keras` checkpoint format, and optimizer state are interchangeable.

## Why a container on Blackwell

TensorFlow's prebuilt CUDA kernels target compute capabilities through sm_90. On Blackwell (sm_120) the JIT compiler tries to lower its baked-in PTX and fails with `CUDA_ERROR_INVALID_PTX` → `CUDA_ERROR_INVALID_HANDLE`. This is reproduced across TF 2.16–2.20 and tf-nightly.

NVIDIA's NGC TensorFlow 25.02 container is the only published path that ships sm_120-ready kernels (and is officially the final NGC TF release). It bundles TF 2.17, CUDA 12.8, cuDNN 9.7.1, Python 3.12, and NCCL 2.25.1.

## One-time setup

### 1. Build the .sif image (per cluster)

Aetherscan ships a single recipe — [`aetherscan.def`](../aetherscan.def) — that builds with either runtime. Build on the cluster you intend to run on so the image is produced by that cluster's native runtime:

```bash
# Blackwell cluster (SingularityCE 4.1.1):
cd /path/to/Aetherscan
singularity build aetherscan-ngc25.02.sif aetherscan.def

# Ampere cluster (Apptainer 1.4.5):
cd /path/to/Aetherscan
apptainer build aetherscan-ngc25.02.sif aetherscan.def
```

Build takes ~15 minutes and produces a ~15 GB `.sif`. The recipe pulls `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` and layers in [`requirements-container.txt`](../requirements-container.txt) (Aetherscan's pip extras).

> [!NOTE]
> A `.sif` built by one runtime is generally readable by the other (both use the SIF format), but rebuilding per cluster avoids any subtle ABI mismatch.

#### Build-time gotchas on hardened HPC nodes

Locked-down clusters typically need three things adjusted before `singularity build` will succeed as an unprivileged user. Work through them in order:

**`FATAL: --remote, --fakeroot, or the proot command are required to build this source as a non-root user`**

SingularityCE 4.x requires fakeroot mappings (or `proot`, or `--remote`) for non-root builds. Have the cluster admin enable fakeroot once:

```bash
sudo singularity config fakeroot --add $USER
```

Then build with `singularity build --fakeroot ...`. If admin help isn't available, build on a machine where you do have fakeroot (e.g. the Apptainer cluster, which usually has it pre-configured), then `scp` the resulting `.sif` over — the SIF format is portable between Apptainer and SingularityCE.

**`FATAL: 'noexec' mount option set on /tmp, temporary root filesystem won't be usable at this location`**

The build's temporary root filesystem needs `exec` permissions, and many HPC nodes harden `/tmp` with `noexec`. Point Singularity at scratch storage instead:

```bash
mkdir -p /datax/scratch/$USER/singularity-tmp /datax/scratch/$USER/singularity-cache
export SINGULARITY_TMPDIR=/datax/scratch/$USER/singularity-tmp
export SINGULARITY_CACHEDIR=/datax/scratch/$USER/singularity-cache
```

`TMPDIR` needs ~15 GB free; `CACHEDIR` caches Docker base-layer blobs (a few GB, persists across rebuilds — keep it). Worth adding to `~/.bashrc` if you rebuild often.

**Build fails in `%post` with `Could not open requirements file: /tmp/...`**

A symptom of Singularity bind-mounting the host's `/tmp` over the container's `/tmp` during `%post`, which hides files placed there by `%files`. Already fixed in [`aetherscan.def`](../aetherscan.def) — we stage to `/opt/` instead. If you hit this, you're on an old revision of the branch; `git pull` and rebuild.

### 2. Verify the image (canary)

```bash
./utils/run_container.sh python -c "
import tensorflow as tf
print('TF:', tf.__version__, '| CUDA built:', tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
with tf.device('/GPU:0'):
    print('Cast sum:', float(tf.reduce_sum(tf.cast(tf.range(1024), tf.float32))))
"
```

Expected on Blackwell: 5 GPUs listed, finite sum, no `PTX` warnings.

### 3. Ampere conda env (unchanged workflow)

```bash
conda env create -f environment.yml
conda activate aetherscan
```

The conda env was bumped to TF 2.17 / numpy 1.26 / cuDNN 9.3 to match the container's API surface. If a stale `aetherscan` env exists, remove it first (`conda remove -n aetherscan --all`) — pip can't downgrade some pinned packages in place.

## Running the pipeline

### Blackwell (container, memory-growth only — recommended default)

```bash
./utils/run_container.sh \
    python -m aetherscan.main train \
    --save-tag final_v1
```

The wrapper auto-detects `apptainer` vs `singularity`, binds the repo and `AETHERSCAN_*` paths into the container, sets `--nv`, and forwards `AETHERSCAN_*` / `SLACK_*` env vars. Environment loading happens at two layers: the wrapper auto-loads `<repo>/.env` at shell time (needed before Python starts so the `AETHERSCAN_*` paths are resolved into the right `--bind` arguments), and `aetherscan.main` calls `python-dotenv`'s `load_dotenv()` at process start (covers Slack credentials inside the container, with `os.environ` then inherited by multiprocess workers). Values already in the wrapper's env — including inline `VAR=val ./utils/run_container.sh ...` or real exports — win at both layers. By default no `--gpu-memory-limit-mb` is passed, so each Blackwell GPU uses memory-growth allocation against its full 96 GB.

### Ampere (conda, 14 GB cap to match prior behavior)

```bash
PYTHONPATH=src python -m aetherscan.main train \
    --gpu-memory-limit-mb 14000 \
    --save-tag final_v1
```

The legacy hardcoded `memory_limit=14000` is now a CLI flag with `GPUConfig` as the source of truth. Setting it preserves the original behavior on the A4000s.

### Inference (either machine)

```bash
# Blackwell
./utils/run_container.sh \
    python -m aetherscan.main inference \
    --inference-files complete_cadences_catalog.csv \
    --encoder-path /datax/scratch/zachy/models/aetherscan/vae_encoder_final_v1.keras \
    --rf-path /datax/scratch/zachy/models/aetherscan/random_forest_final_v1.joblib \
    --config-path /datax/scratch/zachy/models/aetherscan/config_final_v1.json

# Ampere — same args, but invoke without the wrapper and pass --gpu-memory-limit-mb 14000
```

## New CLI flags

Three flags wire onto `GPUConfig` (in [`src/aetherscan/config.py`](../src/aetherscan/config.py)). All accept the dataclass default when omitted:

| Flag                       | Default       | Notes                                                                                                                            |
| -------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `--gpu-memory-limit-mb`    | unset (None)  | Per-GPU memory cap in MiB. Unset = memory growth only. Set to `14000` on Ampere to match legacy behavior.                        |
| `--nccl-num-packs`         | `2`           | num_packs for NCCL/HierarchicalCopy all-reduce. Try `1` or `4` if NCCL is unstable on a 5-GPU Blackwell NVLink topology.         |
| `--async-allocator` / `--no-async-allocator` | enabled       | Toggles `TF_GPU_ALLOCATOR=cuda_malloc_async`. Disable as a workaround for the NGC 25.02 multi-GPU OOM bug.                       |

## Debugging

### `CUDA_ERROR_INVALID_PTX` or `CUDA_ERROR_INVALID_HANDLE` on Blackwell

You're running TF on the host directly, not via the container. The host TF wheel has no sm_120 kernels. Re-run through `./utils/run_container.sh`.

### `nvidia-smi` works but TF sees 0 GPUs inside the container

The `--nv` flag is missing or the host's NVIDIA driver is older than the container's CUDA needs. Driver must be ≥570 for CUDA 12.8 / NGC 25.02. Check with `nvidia-smi` (look at "Driver Version") on the host.

### NCCL hangs or all-reduce errors mid-training

The startup warmup all-reduce in `setup_gpu_strategy` should catch most NCCL failures and fall back to `HierarchicalCopyAllReduce` automatically. If a failure surfaces only mid-epoch (e.g., once gradient sizes grow), try:

```bash
# Re-run with NCCL debug logs
NCCL_DEBUG=INFO ./utils/run_container.sh python -m aetherscan.main train ...

# Or force the hierarchical fallback by lowering num_packs
./utils/run_container.sh python -m aetherscan.main train --nccl-num-packs 1 ...
```

### Multi-GPU OOM on Blackwell

The NGC 25.02 release notes flag a known multi-GPU OOM under the async allocator. Disable it:

```bash
./utils/run_container.sh python -m aetherscan.main train --no-async-allocator ...
```

### Benign noise to ignore

Two warning families fire repeatedly on Blackwell + NGC 25.02 and have no correctness impact. Don't chase them.

- `W gpu_timer.cc:114] Skipping the delay kernel, measurement accuracy will be reduced` — XLA's autotuner normally launches a tiny "delay kernel" before each candidate timing to drain pending GPU work, so measurements are reproducible. TF 2.17's XLA has no delay-kernel implementation registered for sm_120, so the autotuner times without the primer. Picks of fastest kernels become slightly noisier (possibly sub-optimal autotune choices); the kernels themselves still compute correct results. Expect one emission per autotuned fusion — 1000+ lines on a cold start is normal.
- `'+ptxNN' is not a recognized feature for this target (ignoring feature)` (from LLVM NVPTX) — LLVM expresses CUDA capabilities as feature flags. The container's LLVM was built against CUDA 12.8 (PTX ISA ≤8.4); the host driver (580.x, CUDA 13.0) advertises PTX 8.5. When XLA asks LLVM to enable a newer PTX level, LLVM ignores the flag and falls back to a level it knows. Code still compiles and runs; only a handful of PTX 8.5–only instructions are unavailable. Negligible perf delta, zero correctness impact.

Bumping `TF_CPP_MIN_LOG_LEVEL=2` in `aetherscan.def`'s `%environment` silences the first family but also suppresses other potentially useful warnings; the LLVM `+ptxNN` line goes straight to stderr and isn't gated by it. The default is to leave both alone.

## Fallback options

If NGC 25.02 keeps misbehaving, escalate in this order:

1. `--no-async-allocator` (cheapest, often sufficient).
2. `--nccl-num-packs 1` or `4` (5-GPU NVLink topology is unusual).
3. Rebuild against NGC 25.01 — same TF, one minor back; catches 25.02 regressions. Update the `From:` line in `aetherscan.def`.
4. Single-GPU fallback on Blackwell — one 96 GB card still beats 5x A4000 aggregate for correctness validation.
5. Build TF 2.22 from source with sm_86 + sm_120 targets (1–2 days of build-system wrestling; unifies the wheel for both clusters but defers the conda/container split).

## Cross-machine checkpoint interop

`.keras` checkpoints written on Ampere load on Blackwell and vice versa (verified through verification step (f) in the original plan). The `Sampling` layer is now registered with `keras.saving.register_keras_serializable(package="aetherscan")` so `keras.models.load_model(...)` resolves it automatically without `custom_objects=`.

Legacy `.h5` checkpoints from pre-TF-2.16 training runs will NOT load — re-train or re-save to `.keras` first.
