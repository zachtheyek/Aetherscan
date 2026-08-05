# GPU Runtime Guide

This runbook covers running Aetherscan across GPU architectures — the Blackwell (RTX PRO 6000) workstation and the Ampere (A4000) cluster. The pipeline source tree is identical on both machines, and as of [`aetherscan.def`](../aetherscan.def) the NGC container is the canonical runtime on both clusters — one recipe builds and runs on each. The pre-container conda env is kept as an alternative install path on Ampere. For **off-cluster** use there is also a published PyPI package (`pip install aetherscan`) — see the README's [Install From PyPI (pip)](../README.md#install-from-pypi-pip) section (and its v1.0.0 caveats); the container stays canonical on the clusters and is **mandatory on Blackwell**. This document is about the *runtime*; for what the pipeline itself does once it's running, start at [`ARCHITECTURE.md`](ARCHITECTURE.md).

## TL;DR

| Cluster   | GPU                 | Compute capability | Canonical runtime                                                                                  | Alternative                                                 |
| --------- | ------------------- | ------------------ | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Ampere    | NVIDIA RTX A4000    | sm_86              | NGC container `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` (via CUDA forward compat on driver 550.78) | `conda env create -f environment.yml` (TF 2.17 + CUDA 12.3) |
| Blackwell | NVIDIA RTX PRO 6000 | sm_120             | NGC container `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` (TF 2.17 + CUDA 12.8)                      | —                                                           |

All paths run TF 2.17 with Keras 3 so `@tf.function` tracing, `.keras` checkpoint format, and optimizer state are interchangeable.

## Why a container on Blackwell

TensorFlow's prebuilt CUDA kernels target compute capabilities through sm_90. On Blackwell (sm_120) the JIT compiler tries to lower its baked-in PTX and fails with `CUDA_ERROR_INVALID_PTX` → `CUDA_ERROR_INVALID_HANDLE`. This is reproduced across TF 2.16–2.20 and tf-nightly.

NVIDIA's NGC TensorFlow 25.02 container is the only published path that ships sm_120-ready kernels (and is [officially the final NGC TF release](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-25-02.html#:~:text=Deprecation%20notice%3A%20After%20the%2025.02%20release%2C%20NVIDIA%20Optimized%20TensorFlow%20containers%20will%20no%20longer%20be%20released.%20Known%20issues%20may%20be%20resolved%20in%20a%20future%20product%20based%20on%20customer%20demand.)). It bundles TF 2.17, CUDA 12.8, cuDNN 9.7.1, Python 3.12, and NCCL 2.25.1.

## Why the same container also works on Ampere

The container's CUDA 12.8 runtime is newer than the Ampere host's driver supports natively (driver 550.78 caps at CUDA 12.4), but the NGC base ships `cuda-compat-12-8` and `--nv` layers it into the container's library path. TF resolves against the compat libs, the host driver still services GPU ioctls, and sm_86 kernels run unmodified — no host driver upgrade needed, no separate recipe. Verified by `tf.config.list_physical_devices('GPU')` returning all 6 A4000s on the Ampere cluster.

Forward compatibility is a CUDA feature, not a TF feature, so the same trick will keep working as long as Ampere stays within the CUDA 12.x family. If the Ampere driver is ever rolled back below 550, fall back to the conda env (which uses TF's bundled CUDA 12.3 stack and only needs driver ≥525).

## One-time setup

### 1. Get the .sif image (per cluster)

Aetherscan ships a single recipe — [`aetherscan.def`](../aetherscan.def) — that builds with either runtime. Build on the cluster you intend to run on so the image is produced by that cluster's native runtime:

> [!NOTE]
> You usually **don't** need to build. `utils/run_container.sh` pulls the release-pinned image from GHCR (`ghcr.io/zachtheyek/aetherscan:vX.Y.Z`) on first use and caches it as the `.sif` (recording the pulled ref — `repo:tag` — in `<sif>.pulled-tag`). It re-pulls whenever that ref changes, i.e. a new version tag *or* a different repo via `AETHERSCAN_IMAGE` — so a release-tag checkout re-pulls on every version bump, even a digest-identical retag; a `.devN` checkout always asks for the constant `:latest` and keeps whatever it first cached, so `rm` the `.sif` and its sidecar to pick up a moved `:latest`. A local build placed over a pulled `.sif` is detected by mtime and kept, never clobbered. Build from `aetherscan.def` only when the pull can't serve your host — a non-x86_64 host, a driver below the CUDA 12.8 floor, or local `requirements-container.txt` edits. **Either way** (pull or build), on a hardened HPC node with a quota'd `$HOME` first redirect `SINGULARITY_TMPDIR` / `SINGULARITY_CACHEDIR` (or the `APPTAINER_*` equivalents) to scratch — a pull unpacks the ~9 GB image through them exactly as a build does (see the TMPDIR/CACHEDIR note below).

```bash
cd /path/to/Aetherscan

# Blackwell cluster (SingularityCE 4.1.1):
singularity build aetherscan-ngc25.02.sif aetherscan.def

# Ampere cluster (Apptainer 1.4.5):
apptainer build aetherscan-ngc25.02.sif aetherscan.def
```

Build takes ~9 minutes and produces a ~9 GB `.sif`. The recipe pulls `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` and layers in [`requirements-container.txt`](../requirements-container.txt) (Aetherscan's pip extras).

> [!NOTE]
> A `.sif` built by one runtime is generally readable by the other (both use the SIF format), but rebuilding per cluster avoids any subtle ABI mismatch.

#### Pinned base image (digest)

The `From:` line in [`aetherscan.def`](../aetherscan.def) pins the NGC base by **digest**, not just by the `25.02-tf2-py3` tag:

```
From: nvcr.io/nvidia/tensorflow:25.02-tf2-py3@sha256:c83b37d26f19ab00d8a13cf974edd079c3d099918ec3110c304a989d6e2f75d5
```

A tag is a mutable pointer: NVIDIA can re-push `25.02-tf2-py3` (e.g. a security rebuild) and every fresh build would silently pull different bytes. The digest is content-addressed, so it always resolves to the exact image this pipeline was validated against, and a build **fails loudly at the `From:` line** if that image is ever unreachable or altered rather than substituting something else. This matters mostly at rebuild time — an already-built `.sif` is itself immutable — but it keeps rebuilds on both clusters reproducible.

The pinned value is the multi-arch **index** digest (a `manifest.list`), so the same line resolves the `linux/amd64` sub-manifest automatically on both x86_64 clusters — no per-arch digest needed.

25.02 is [officially the final NGC TF release](#why-a-container-on-blackwell), so the tag is unlikely to move. But if NVIDIA ever ships a patched rebuild you want to adopt, updating is a deliberate one-line diff rather than a silent drift:

```bash
# Re-read the tag's current digest (needs `docker login nvcr.io` with an NGC API key).
docker buildx imagetools inspect nvcr.io/nvidia/tensorflow:25.02-tf2-py3 --format '{{.Manifest.Digest}}'
```

Swap the new `sha256:…` into **both** the `From:` line and the `Base` label in [`aetherscan.def`](../aetherscan.def), commit, and rebuild.

#### Build-time gotchas on hardened HPC nodes

Locked-down clusters typically need three things adjusted before `singularity build` will succeed as an unprivileged user. Work through them in order:

**`FATAL: --remote, --fakeroot, or the proot command are required to build this source as a non-root user`**

SingularityCE 4.x requires fakeroot mappings (or `proot`, or `--remote`) for non-root builds. Have the cluster admin enable fakeroot once:

```bash
sudo singularity config fakeroot --add $USER
```

Then build with `singularity build --fakeroot ...`. If admin help isn't available, build on a machine where you do have fakeroot (e.g. the Apptainer cluster, which usually has it pre-configured), then `scp` the resulting `.sif` over — the SIF format is usually portable between Apptainer and SingularityCE.

**`FATAL: 'noexec' mount option set on /tmp, temporary root filesystem won't be usable at this location`**

The build's temporary root filesystem needs `exec` permissions, and many HPC nodes harden `/tmp` with `noexec`. Point Singularity at scratch storage instead:

```bash
mkdir -p /datax/scratch/$USER/singularity-tmp /datax/scratch/$USER/singularity-cache
export SINGULARITY_TMPDIR=/datax/scratch/$USER/singularity-tmp
export SINGULARITY_CACHEDIR=/datax/scratch/$USER/singularity-cache
```

`TMPDIR` needs ~15 GB free; `CACHEDIR` caches Docker base-layer blobs (a few GB, persists across rebuilds — keep it). Worth adding to `~/.bashrc` if you rebuild often. On an **Apptainer** host (e.g. the Ampere cluster) use the `APPTAINER_TMPDIR` / `APPTAINER_CACHEDIR` equivalents instead — Apptainer reads those first and falls back to the `SINGULARITY_*` names.

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

Expected: TF 2.17, CUDA built True, GPUs listed, finite sum, no `PTX` warnings, and the three benign `cuFFT/cuDNN/cuBLAS` "already registered" lines from NGC's CUDA plugin loader (ignore them).

### 3. Ampere conda env (alternative path)

Only needed if the container path is unavailable (e.g. no Apptainer/Singularity on the host, or driver <550 blocks forward compat).

```bash
conda env create -f environment.yml
conda activate aetherscan
```

The conda env was bumped to TF 2.17 / numpy 1.26 to match the container's API surface; under the hood it uses TF's bundled CUDA 12.3 / cuDNN 9.1 stack rather than the container's 12.8 / 9.7.1. If a stale `aetherscan` env exists, remove it first (`conda remove -n aetherscan --all`) — pip can't downgrade some pinned packages in place.

## Running the pipeline

### Container (Blackwell or Ampere — recommended default)

```bash
# Blackwell — memory-growth only, full 96 GB per GPU
./utils/run_container.sh \
    python -m aetherscan.main train \
    --save-tag train

# Ampere — same wrapper, 14 GB cap to match the A4000s' prior behavior
./utils/run_container.sh \
    python -m aetherscan.main train \
    --gpu-memory-limit-mb 14000 \
    --save-tag train
```

The wrapper auto-detects `apptainer` vs `singularity`, binds the repo and `AETHERSCAN_*` paths into the container, sets `--nv` for GPU passthrough, and forwards `AETHERSCAN_*` / `SLACK_*` env vars. Environment loading happens at two layers: the wrapper auto-loads `<repo>/.env` at shell time (needed before Python starts so the `AETHERSCAN_*` paths are resolved into the right `--bind` arguments), and `aetherscan.main` calls `python-dotenv`'s `load_dotenv()` at process start (covers Slack credentials inside the container, with `os.environ` then inherited by multiprocess workers). Values already in the wrapper's env — including inline `VAR=val ./utils/run_container.sh ...` or real exports — win at both layers. By default no `--gpu-memory-limit-mb` is passed, so each Blackwell GPU uses memory-growth allocation against its full 96 GB; on Ampere, pass `--gpu-memory-limit-mb 14000` to preserve the legacy A4000 cap. The wrapper also forwards `HF_TOKEN`, and — when `HF_HOME` is set — binds and forwards it, so the HuggingFace download cache (used when bare inference resolves the released weights from the Hub) lands where you point it. Set `HF_HOME` to scratch (`mkdir -p /datax/scratch/$USER/hf_home` first, then `export HF_HOME=/datax/scratch/$USER/hf_home` in `~/.bashrc` — the wrapper fails fast if `HF_HOME` is missing or relative) so downloaded weights persist across runs and don't fill `$HOME`; left unset off-cluster, HuggingFace falls back to `~/.cache/huggingface` inside the container.

### Conda (Ampere only, alternative path)

```bash
PYTHONPATH=src python -m aetherscan.main train \
    --gpu-memory-limit-mb 14000 \
    --save-tag train
```

The legacy hardcoded `memory_limit=14000` is now a CLI flag with `GPUConfig` as the source of truth. Setting it preserves the original behavior on the A4000s.

### Inference (either cluster)

```bash
./utils/run_container.sh \
    python -m aetherscan.main inference \
    --inference-files complete_cadences_catalog.csv \
    --encoder-path /datax/scratch/zachy/models/aetherscan/vae_encoder_train_20260101_120000.keras \
    --rf-path /datax/scratch/zachy/models/aetherscan/random_forest_train_20260101_120000.joblib \
    --config-path /datax/scratch/zachy/models/aetherscan/config_train_20260101_120000.json

# Add --gpu-memory-limit-mb 14000 when running on Ampere (container or conda).
# For the Ampere conda alternative, drop the run_container.sh wrapper and prepend PYTHONPATH=src.
```

## New CLI flags

Four flags wire onto `GPUConfig` (in [`src/aetherscan/config.py`](../src/aetherscan/config.py)). All accept the dataclass default when omitted:

| Flag                                         | Default      | Notes                                                                                                                    |
| -------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `--num-replicas`                             | unset (None) | Number of GPUs to pin TF to. Unset = use every visible GPU. Must satisfy `1 <= N <= len(physical_devices)` if set.       |
| `--gpu-memory-limit-mb`                      | unset (None) | Per-GPU memory cap in MiB. Unset = memory growth only. Set to `14000` on Ampere to match legacy behavior.                |
| `--nccl-num-packs`                           | `2`          | num_packs for NCCL/HierarchicalCopy all-reduce. Try `1` or `4` if NCCL is unstable on a 5-GPU Blackwell NVLink topology. |
| `--async-allocator` / `--no-async-allocator` | enabled      | Toggles `TF_GPU_ALLOCATOR=cuda_malloc_async`. Disable as a workaround for the NGC 25.02 multi-GPU OOM bug.               |

## Debugging

### `CUDA_ERROR_INVALID_PTX` or `CUDA_ERROR_INVALID_HANDLE` on Blackwell

You're running TF on the host directly, not via the container. The host TF wheel has no sm_120 kernels. Re-run through `./utils/run_container.sh`.

### `nvidia-smi` works but TF sees 0 GPUs inside the container

The `--nv` flag is missing or the host's NVIDIA driver is too old for the container's CUDA stack. Driver requirements:

- **Blackwell**: ≥570 (native CUDA 12.8 support).
- **Ampere**: ≥550 — works via CUDA forward compatibility (NGC ships `cuda-compat-12-8`). Drivers in the 525–549 range will fall back to the conda env path.

Check with `nvidia-smi` (look at "Driver Version") on the host.

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
- `'+ptxNN' is not a recognized feature for this target (ignoring feature)` (from LLVM NVPTX) — LLVM expresses CUDA capabilities as feature flags. The container's LLVM was built against CUDA 12.8 (PTX ISA ≤8.4); the host driver (595.x, CUDA 13.2) advertises PTX 8.5. When XLA asks LLVM to enable a newer PTX level, LLVM ignores the flag and falls back to a level it knows. Code still compiles and runs; only a handful of PTX 8.5–only instructions are unavailable. Negligible perf delta, zero correctness impact.

Bumping `TF_CPP_MIN_LOG_LEVEL=2` in `aetherscan.def`'s `%environment` silences the first family but also suppresses other potentially useful warnings; the LLVM `+ptxNN` line goes straight to stderr and isn't gated by it. The default is to leave both alone.

## Fallback options

If NGC 25.02 keeps misbehaving, escalate in this order:

1. `--no-async-allocator` (cheapest, often sufficient).
2. `--nccl-num-packs 1` or `4` (5-GPU NVLink topology is unusual).
3. Rebuild against NGC 25.01 — same TF, one minor back; catches 25.02 regressions. Update the `From:` line in `aetherscan.def` (re-pin its digest too — see [Pinned base image](#pinned-base-image-digest)).
4. `--num-replicas 1` for single-GPU fallback on Blackwell — one 96 GB card still beats 5x A4000 aggregate for correctness validation.
5. Fall back to the Ampere conda env for any work that doesn't need Blackwell. With the container as the canonical runtime on both clusters, the conda env is now a fallback rather than a parallel workflow.
6. Build TF 2.22 from source with sm_86 + sm_120 targets (1–2 days of build-system wrestling). Mostly obsolete now that one container runs on both clusters via forward compat; keep as a last resort if NGC 25.02 ever becomes unsupportable.

## Cross-machine checkpoint interop

`.keras` checkpoints written on Ampere load on Blackwell and vice versa. The `Sampling` layer is now registered with `keras.utils.register_keras_serializable(package="aetherscan")
` so `keras.models.load_model(...)` resolves it automatically without `custom_objects=`.

Legacy `.h5` checkpoints from pre-TF-2.16 training runs will NOT load — re-train or re-save to `.keras` first.

## Additional resources

- [TensorFlow-CUDA compatibility matrix](https://www.tensorflow.org/install/source#gpu)
