# Known Issues

This document lists known issues, limitations, and workarounds in Aetherscan.

## Reporting New Issues

If you encounter an issue not listed here:

1. Check [GitHub Discussions](https://github.com/zachtheyek/Aetherscan/discussions) for existing reports
2. If new, open a discussion with:
   - Clear description of the problem
   - Steps to reproduce
   - System information (use `utils/get_system_info.sh` and append outputs as attachments)
   - Relevant log outputs
   - Configuration used

See [`CONTRIBUTING.md`](/CONTRIBUTING.md) for more details.

---

## Issue Status Definitions

| Status        | Meaning                                              |
| ------------- | ---------------------------------------------------- |
| **Open**      | Actively investigating or planned for fix            |
| **Mitigated** | Workaround implemented, full fix pending             |
| **Won't fix** | Expected behavior or not worth the complexity to fix |
| **Closed**    | Fixed in a specific version                          |

---

## 1. CUDA Factory Registration Warnings

### Symptom

At pipeline startup, you'll see warnings like:

```
E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:479] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:10575] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1442] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
```

### Cause

This is a known TensorFlow issue where CUDA plugin factories (cuFFT, cuDNN, cuBLAS) attempt to register multiple times during initialization. This typically occurs due to how TensorFlow's plugin system handles dynamic library loading.

### Impact

**None.** Despite being logged, these messages are harmless and do not affect GPU computation or training/inference correctness.

### Workaround

No action required. The warnings can be safely ignored.

### Status

**Won't fix.** Upstream TensorFlow issue. See [tensorflow/tensorflow#62075](https://github.com/tensorflow/tensorflow/issues/62075).

---

## 2. CPU Optimization Warnings

### Symptom

At pipeline startup, you'll see:

```
I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
```

### Cause

TensorFlow detects that the CPU supports additional instruction sets (AVX2, FMA) that could be used for optimization, but the pre-built TensorFlow binary wasn't compiled to use them in all operations.

### Impact

**None.** This is informational only. TensorFlow still uses these optimized instructions for performance-critical operations. The message suggests that rebuilding from source could enable them more broadly, but this provides minimal benefit for GPU-accelerated workloads like Aetherscan.

### Workaround

No action required. For GPU-based training/inference, CPU instruction optimizations have negligible impact.

### Status

**Won't fix.** Informational message, not a warning or error.

---

## 3. TensorRT Warning

### Symptom

At pipeline startup, you'll see:

```
W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
```

### Cause

TensorFlow's TensorRT integration looks for TensorRT libraries at startup. When TensorRT is not installed (which is typical for most setups), this warning is emitted.

### Impact

**None.** TensorRT is an optional optimization library for inference acceleration. Aetherscan does not require or use TensorRT, so this warning has no functional impact.

### Workaround

No action required. The warning can be safely ignored.

### Status

**Won't fix.** Expected behavior when TensorRT is not installed. See [tensorflow/tensorflow#64809](https://github.com/tensorflow/tensorflow/issues/64809).

---

## 4. Blimpy pkg_resources Deprecation Warning

### Symptom

At pipeline startup, you'll see:

```
/home/.../site-packages/blimpy/__init__.py:21: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import get_distribution, DistributionNotFound
```

### Cause

The `blimpy` library (a dependency of `setigen`, which Aetherscan uses for signal injection) uses the deprecated `pkg_resources` API from setuptools for version detection.

### Impact

**None currently.** The warning indicates future incompatibility with newer setuptools versions, but does not affect current functionality.

### Workaround

No action required. A fix has been submitted upstream: [UCBerkeleySETI/blimpy#281](https://github.com/UCBerkeleySETI/blimpy/pull/281), though it may not be merged soon due to the library's maintenance status.

### Status

**Won't fix.** Upstream dependency issue outside Aetherscan's control.

---

## 5. TensorFlow Prefetch Warnings

### Symptom

During training/inference, you may see warnings like:

```
2026-01-24 03:52:38.996928: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: CANCELLED: GetNextFromShard was cancelled
         [[{{node MultiDeviceIteratorGetNextFromShard}}]]
2026-01-24 03:52:38.997040: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: CANCELLED: GetNextFromShard was cancelled
         [[{{node MultiDeviceIteratorGetNextFromShard}}]]
         [[RemoteCall]] [type.googleapis.com/tensorflow.DerivedStatus='']
```

### Cause

Since we use `strategy.experimental_distribute_dataset()`, TensorFlow creates a `MultiDeviceIterator` that coordinates data distribution across GPUs. This iterator has background threads that prefetch data (via `.prefetch(tf.data.AUTOTUNE)`). These warnings occur when TensorFlow's prefetch threads are cancelled mid-operation, which happens when:

1. Iterator isn't fully consumed - Datasets use `.repeat()` making them infinite, so they never "finish" naturally. When the training/inference loop ends after a fixed number of steps, the iterator is abandoned with prefetch operations still pending.
2. Generator exits early - `DataHolder.clear()` pattern causes generators to return when `_cleared` is True, which abruptly stops the data pipeline.
3. Phase transitions - Switching between training and validation (or ending a run) causes pending `GetNextFromShard` operations to be cancelled.

### Impact

**None.** These warnings are benign and do not indicate data loss or training/inference issues. `CANCELLED` status simply means "we stopped asking for more data", which is the normal cleanup path when you don't consume an infinite iterator to completion.

### Workaround

No action required. The warnings can be safely ignored. If they cause log noise, you can suppress TensorFlow warnings:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING
```

However, this also suppresses other potentially useful warnings, so it's not recommended.

### Status

**Won't fix.** This is expected TensorFlow behavior when using infinite datasets with `.repeat()` and cleaning up between epochs.

---

## 6. Resource Monitor CPU Undercounting

### Symptom

Aetherscan CPU usage is systematically lower than system total CPU usage in resource monitoring plots, particularly during multiprocessing workloads.

### Cause

The issue was in the `get_process_tree_stats()` function. Every monitoring interval, `process.children(recursive=True)` created **new `psutil.Process` objects** for all child processes. When `cpu_percent(interval=0.0)` is called on a newly created Process object, psutil has no baseline CPU measurements to compare against, so it returns `0.0`.

With 96+ child processes being recreated every interval, the majority reported `0.0` CPU, leading to severe undercounting.

### Impact

**Minor.** Resource utilization plots showed inaccurate CPU values. Training and inference correctness were not affected.

### Status

**Closed.** `get_process_tree_stats()` now maintains a PID -> `psutil.Process` cache across monitoring intervals, so `cpu_percent(interval=0)` measures against the previous sample's baseline. Each newly spawned process contributes one `0.0` reading the first interval it is seen (accurate thereafter); PIDs that leave the tree are evicted from the cache. See [GitHub Issue #12](https://github.com/zachtheyek/Aetherscan/issues/12).

### Related Code

`src/aetherscan/monitor/monitor.py:get_process_tree_stats()`

---

## 7. Pool Cleanup Hangs

### Symptom

During shutdown or between training rounds, the pipeline may hang when cleaning up multiprocessing pools. Logs may show:

```
Terminating pool 'data_generation'
Pool 'data_generation' terminate() timed out, escalating to SIGKILL
Pool 'data_generation' join() timed out (Pool internal threads may be stuck)
```

### Cause

Python's `multiprocessing.Pool` has internal threads (feeder thread, result handler) that can get stuck in certain edge cases:

1. Workers holding locks when SIGTERM is received
2. Large data in the result queue that can't be pickled/unpickled quickly
3. Database or Slack logger connections not properly closed in child processes

### Impact

**Moderate.** Can cause slow shutdowns or require manual process termination. Training state is checkpointed, so data loss is minimal.

### Workaround

The pipeline uses a timeout-based escalation strategy:

1. `pool.terminate()` with timeout
2. If timeout, SIGKILL individual workers
3. `pool.join()` with timeout
4. Log warning and continue if join times out

If the pipeline hangs completely, run [`utils/kill_pipeline.sh`](utils/kill_pipeline.sh)
from a separate shell on the same machine. It finds the main process and all its
workers on its own (no PID needed), works for both the container and source run
modes, and tries a graceful SIGTERM first (letting the `ResourceManager` close
pools / shared memory) before escalating to SIGKILL:

```bash
# Graceful stop, escalating to SIGKILL after a timeout (default 30s)
./utils/kill_pipeline.sh

# Skip straight to SIGKILL if the process tree is wedged
./utils/kill_pipeline.sh --force

# Preview the process tree without sending any signals
./utils/kill_pipeline.sh --dry-run

# Override the round-data root used for orphan-producer pidfile discovery
./utils/kill_pipeline.sh --round-data-root /path/to/round_data
```

When no main process is found, `kill_pipeline.sh` sweeps
`{round_data_root}/*/producer.pid` for orphaned `RoundDataProducer` process
trees left behind by an ungraceful main-process death, and reaps any that are
still alive.

A forced kill skips `ResourceManager` cleanup, so clean up any orphaned shared
memory afterwards (the script prints a reminder):

```bash
# List shared memory segments, then remove stale ones (be careful!)
ls /dev/shm/
rm /dev/shm/psm_*
```

To do it by hand instead: `ps aux | grep aetherscan`, then `kill -9 <pid>`.

A separate, benign class of leftover can also appear: `/dev/shm/sem.loky-*` POSIX
semaphores. These come from joblib's loky reusable-executor resource-tracker (pulled in
transitively, e.g. via scikit-learn), not from the pipeline's own shared memory — Aetherscan's
`SharedMemory` lifecycle is clean (creator-only `unlink()`, `ResourceManager`-registered, and
verified). Like the blocks above they leak only on a non-clean teardown (SIGKILL / OOM /
container exit); they are harmless and safe to remove:

```bash
rm -f /dev/shm/sem.loky-*
```

### Status

**Open.** Investigating root cause. May be related to Slack logger connections in child processes.

### Related Code

- `src/aetherscan/manager/manager.py:ManagedPool.close()`
- `src/aetherscan/manager/manager.py:1` (BUG comment)

---

## 8. Data Generation Performance Bottleneck

### Symptom

Data generation takes approximately 160 minutes per round on a 96-core system, creating a severe bottleneck that dominates total training time. Additionally, rounds 2+ exhibit a 3-5x slowdown compared to round 1, despite identical workloads.

### Cause

Four independent bottlenecks contribute to this issue:

1. **Excessive task dispatch overhead**: `pool.imap()` called with `chunksize=1` results in 120,000 individual task dispatches per round, with scheduling overhead dominating actual computation.

2. **IPC data transfer bottleneck**: Each worker returns a 192KB numpy array through the multiprocessing result queue. Per round, this is ~23GB of pickle serialization, which requires the GIL.

3. **Sequential synchronization barriers**: `generate_batch()` makes 8 sequential calls to `batch_create_cadence()`, each with its own synchronization barrier, causing workers to sit idle.

4. **TensorFlow thread pool GIL contention**: After round 1, TensorFlow's 192 inter-op and intra-op threads continuously poll for work, periodically acquiring GIL and interrupting main process pickle operations.

### Impact

**High.** Data generation dominates training time, significantly increasing total training duration.

### Fix

Resolved in [PR #117](https://github.com/zachtheyek/Aetherscan/pull/117) (commit 6e89608). The implemented design addresses all four bottlenecks:

- **Memmap-backed rounds** — workers write directly into on-disk memmaps (`round_data.py`), eliminating per-sample IPC pickle overhead
- **Batched tasks** — `--data-gen-task-size` groups multiple cadences per worker task, reducing dispatch overhead from 120k dispatches to a few hundred
- **Unified dispatch** — one `pool.map` barrier replaces 8 sequential `batch_create_cadence()` calls
- **Process-isolated generation** — background producer process (`RoundDataProducer`) is spawn-started and never imports TensorFlow, eliminating GIL contention from TF thread pools

See also [GitHub Issue #114](https://github.com/zachtheyek/Aetherscan/issues/114).

### Status

**Closed.** Fixed in [PR #117](https://github.com/zachtheyek/Aetherscan/pull/117).

### Related Code

- `src/aetherscan/round_data.py` — disk-backed per-round dataset lifecycle (memmap creation, manifests, cleanup)
- `src/aetherscan/data_generation.py` — batched memmap workers + background producer integration

---

## 9. Slack Message Truncation

### Symptom

Some batched Slack messages that are too long get cut off with "..." even after clicking "show more".

### Cause

Slack API has character limits for message blocks. When batched messages exceed this limit, they are truncated without warning.

### Impact

**Minor.** Some log information may be lost in Slack notifications. Full logs are still available in file logs.

### Status

**Open.**

### Related Code

`src/aetherscan/logger/slack_handler.py:4` (BUG comment)

---

## 10. Missing Sections in Resource Monitor Plots

### Symptom

Entire sections (e.g., data generation for round X) don't appear in the resource utilization plot.

### Cause

Unknown. Not verified whether data isn't being written to the database properly or if there's a plotting issue.

### Impact

**Minor.** Incomplete resource utilization visualization. Training correctness is not affected.

### Status

**Open.** Needs investigation.

### Related Code

`src/aetherscan/monitor/monitor.py:1` (BUG comment)

---

## 11. Negative Memory Freed Calculation

### Symptom

The `total_memory_freed_gb` value in cleanup logs sometimes appears negative.

### Cause

Race condition or measurement timing issue when calculating memory before and after cleanup operations.

### Impact

**Minor.** Cosmetic issue in logs. Cleanup still functions correctly.

### Status

**Open.**

### Related Code

`src/aetherscan/manager/manager.py:443` (BUG comment)

---

## 12. sys.exit() Blocked by Non-Daemon Threads

### Symptom

Calling `sys.exit()` in the main process may hang or not exit cleanly.

### Cause

Non-daemon threads (such as database writer threads, resource monitor threads, or Slack logger threads) prevent the process from exiting. `sys.exit()` waits for all non-daemon threads to complete before actually terminating.

### Impact

**Minor.** May require force-killing the process. Workaround exists by calling `manager.cleanup_all()` directly.

### Workaround

Instead of relying on `sys.exit()`, explicitly call `manager.cleanup_all()` before exiting to ensure all threads and pools are properly terminated.

### Status

**Open.** Behavior is inconsistent and needs further testing.

### Related Code

- `src/aetherscan/main.py:491-493` (BUG comment)
- `src/aetherscan/main.py:478`

---

## 13. Matplotlib tight_layout Warning

### Symptom

During plot generation, you may see:

```
UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
```

### Cause

Some plotting functions use axes configurations (e.g., colorbars, insets) that are not fully compatible with `plt.tight_layout()`.

### Impact

**None.** Plots are still generated correctly. This is a cosmetic warning.

### Workaround

No action required. The warning can be safely ignored.

### Status

**Won't fix.** Expected matplotlib behavior for complex figure layouts.

### Related Code

`src/aetherscan/train.py:1974-1976` (WARN comment)

---

## 14. XLA "Skipping the delay kernel" Warnings

### Symptom

During training and inference — on **both** the Ampere (conda) and Blackwell (NGC
container) clusters — the console is flooded with:

```
...u_timer.cc:114] Skipping the delay kernel, measurement accuracy will be reduced
```

(observed ~1,400–1,600 times in a single short run).

### Cause

This comes from XLA's GPU timer (`gpu_timer.cc`). To time GPU ops precisely for
autotuning, XLA normally launches a small "delay kernel"; when it opts to skip it, the
internal timing measurements it feeds to its autotuning heuristics are slightly less
precise. It is emitted per-measurement, hence the high count.

### Impact

**None on correctness.** Only the precision of XLA's internal _timing measurements_ is
reduced, which at worst could nudge autotuning toward a marginally suboptimal kernel
variant — a negligible perf delta in practice. It is not specific to the Blackwell
migration (it appears identically on the Ampere cluster), and runs complete normally.

### Workaround

No action required; safe to ignore. If the log volume is bothersome it is gated by
`TF_CPP_MIN_LOG_LEVEL` along with TF's other INFO/WARNING output (raising it to `2`
silences it but also hides other potentially useful warnings, so it's not recommended).

### Status

**Won't fix.** Benign upstream TensorFlow/XLA behavior. See also
[`docs/GPU_RUNTIME_GUIDE.md`](docs/GPU_RUNTIME_GUIDE.md).

---

## 15. LLVM "+ptxNN is not a recognized feature" Warnings (Blackwell only)

### Symptom

On the **Blackwell** cluster (NGC 25.02 container) only, stderr repeats:

```
'+ptx85' is not a recognized feature for this target (ignoring feature)
```

### Cause

LLVM (via XLA's NVPTX backend) expresses CUDA capabilities as feature flags. The
container's LLVM was built against CUDA 12.8 (PTX ISA ≤ 8.4); the newer host driver
advertises PTX 8.5. When XLA asks LLVM to enable the newer PTX level, LLVM doesn't
recognize the flag and falls back to a level it knows.

### Impact

**None.** Code still compiles and runs; only a handful of PTX 8.5-only instructions are
unavailable. Negligible perf delta, zero correctness impact. Does not appear on the
Ampere cluster.

### Workaround

No action required; safe to ignore. The line goes straight to stderr and is **not** gated
by `TF_CPP_MIN_LOG_LEVEL`, so it can't be suppressed without redirecting stderr.

### Status

**Won't fix.** Expected LLVM/PTX version-skew behavior. See also
[`docs/GPU_RUNTIME_GUIDE.md`](docs/GPU_RUNTIME_GUIDE.md).

---

## 16. Container-Build pip Resolver Warning (pydot/pyparsing)

### Symptom

During `singularity/apptainer build` of `aetherscan.def`, while `%post` runs
`pip install -r requirements-container.txt`, the build log prints:

```
ERROR: pip's dependency resolver does not currently take into account all the packages
that are installed. This behaviour is the source of the following dependency conflicts.
pydot 3.0.4 requires pyparsing>=3.0.9, but you have pyparsing 2.4.7 which is incompatible.
```

### Cause

The conflict is **pre-existing inside the NGC `tensorflow:25.02-tf2-py3` base image**,
which ships `pydot 3.0.4` alongside `pyparsing 2.4.7`. Installing our extras makes pip's
resolver re-inspect the environment and report the already-broken pair; nothing in
`requirements-container.txt` installs or touches either package.

### Impact

**None.** Aetherscan never imports `pydot` (it is only used by
`keras.utils.plot_model`-style graph plotting, which we don't call). The build completes
and the image is fully functional.

### Workaround

None needed; safe to ignore. Do not "fix" by upgrading `pyparsing` in the base image —
that risks disturbing NGC-pinned packages. NGC 25.02 is the final TF container release,
so this won't change upstream.

### Status

**Won't fix.** Pre-existing conflict in the upstream NGC base image; documented here.

---

## 17. RF-Dataset Injection Stats Rows Not Superseded on Retry

### Symptom

After an `rf_train` retry that regenerates the RF dataset, stale RF-generation `injection_stats` rows (written with `round_number=NULL`) remain live in the DB.

### Cause

`mark_superseded(round_ge=k)` can't target rows with `round_number=NULL` (SQL comparisons against NULL never match).

### Impact

**Minor.** Bounded to partial-generation rows from a failed `rf_train` attempt; does not affect training correctness or final model quality.

### Status

**Closed.** Fixed in [PR #211](https://github.com/zachtheyek/Aetherscan/pull/211): RF-generation rows now carry a sentinel round number (`num_training_rounds + 1`, one past the last beta-VAE round) instead of `NULL`, making them reachable by the existing `round_ge` supersede call — no NULL-aware special case needed.

### Related Code

`src/aetherscan/db/db.py` (`mark_superseded`), `src/aetherscan/train.py` (RF dataset generation), `src/aetherscan/data_generation.py` (writes the `round_number` rows)

---

## 18. setuptools sdist MANIFEST.in Bypass (Dependabot Alert)

### Symptom

Dependabot flags [alert #1](https://github.com/zachtheyek/Aetherscan/security/dependabot/1): setuptools `< 83.0.0` is affected by [CVE-2026-59890](https://github.com/advisories/GHSA-h35f-9h28-mq5c) (GHSA-h35f-9h28-mq5c, **medium**) — a `MANIFEST.in` exclusion bypass when building an sdist, via a Unicode normalization (NFC/NFD) filename collision on macOS APFS/HFS+ filesystems. Our pin is `setuptools>=78.1.1,<81`, which is inside the vulnerable range.

### Cause

`setuptools` is a transitive dependency only: it is installed so that `pkg_resources` is importable (`blimpy`, pulled in via `setigen`, imports it at module load — see issue #4). Nothing in Aetherscan imports `setuptools` or `pkg_resources` directly.

### Impact

**None.** The vulnerable code path is sdist construction (`setuptools` reading `MANIFEST.in` while packaging a source distribution). Aetherscan's build backend is **hatchling**, not setuptools (`pyproject.toml` `[build-system]`), so setuptools never builds a distribution here — the affected path is never exercised. The bypass additionally requires a macOS APFS/HFS+ filesystem's Unicode normalization; production runs are on Linux clusters.

### Workaround

No action required. The alert is not exploitable in this project's usage.

### Status

**Won't fix.** Not reachable (setuptools is present only for `pkg_resources`, never used to build), and the fix is unreachable under our constraints anyway: the patched release is `83.0.0`, which crosses the documented `setuptools<81` ceiling. That ceiling exists because setuptools 81 removed the vendored `pkg_resources` submodule that `blimpy` still imports (issue #4); raising the ceiling to reach 83.0.0 would break `blimpy`. Revisit if `blimpy` migrates off `pkg_resources` (letting the `setuptools` dependency be dropped entirely) or if setuptools becomes a direct build-time dependency.

### Related Code

`pyproject.toml` / `requirements-container.txt` / `environment.yml` (the `setuptools>=78.1.1,<81` pin and its rationale comment), issue #4 (the `pkg_resources`/`blimpy` coupling that fixes the `<81` ceiling).

---

## 19. v1.0.0 pip/conda Install Cannot Load the Encoder (legacy Keras)

### Symptom

After a clean `pip install aetherscan==1.0.0` (or a conda env built from the v1.0.0 `environment.yml`), inference fails while loading the encoder:

```
TypeError: Could not locate class 'Functional'. Make sure custom classes are decorated with
`@keras.saving.register_keras_serializable()`. Full object config: {'module': 'tf_keras.src.engine.functional', ...}
```

### Cause

The released `.keras` weights are **Keras-2** (`tf_keras`) artifacts, but no v1.0.0 manifest pulls `tf_keras` and TF 2.17 defaults to **Keras 3** unless `TF_USE_LEGACY_KERAS=1` is set. Keras 3 therefore tries to deserialize a `tf_keras.src.engine.functional` config it has no class for. See issue #323.

### Impact

**pip / conda install paths only.** The NGC container path is unaffected — its base image ships `tf_keras` 2.17 and already sets `TF_USE_LEGACY_KERAS`. Training on a fresh pip install is also unaffected until it reloads a released checkpoint; the failure is specific to loading the v1.0.0 artifacts.

### Workaround

Install `tf_keras` and set the flag before running — the two extra steps in [Install From PyPI (pip)](README.md#install-from-pypi-pip):

```bash
pip install "tf_keras~=2.17.0"
export TF_USE_LEGACY_KERAS=1
```

### Status

**Fixed on `master`** by PR #340: `tf_keras==2.17.*` is a declared dependency in `pyproject.toml` / `environment.yml`, and `src/aetherscan/__init__.py` sets `TF_USE_LEGACY_KERAS=1` via `os.environ.setdefault` at package-import time (`aetherscan.def`'s `%environment` and the `Dockerfile`'s `ENV` export it explicitly too). The workaround is needed only for the published **v1.0.0** wheel and drops away once v1.0.1 ships.

### Related Code

`src/aetherscan/__init__.py` (the `setdefault`), `pyproject.toml` / `environment.yml` (the `tf_keras==2.17.*` pin), `requirements-container.txt` (intentionally omits it — the NGC base provides it), `aetherscan.def` / `Dockerfile` (the explicit exports), `src/aetherscan/models/vae.py` (the `Sampling` back-compat serialization comment).

## Pre-#283 saved configs: `training.seed` / `training.tf_deterministic_ops` are silently skipped on restore

PR #283 (issue #279) moved the root seed to `reproducibility.seed` (new default **11**) and
`tf_deterministic_ops` alongside it. `apply_saved_config` skips unknown keys by design, so
reloading a `config_{tag}.json` written **before** #283 silently ignores the old
`training.seed` / `training.tf_deterministic_ops` values and the run uses the new defaults
instead. **Workaround:** pass the old value explicitly (`--seed N` / `--tf-deterministic-ops`
— both flags now exist on `train` *and* `inference`). Same clean-break precedent as the #272
tag-scheme change.

## Pre-#293 saved configs: `beta_vae.regularization_active` is silently skipped on restore

PR #324 (issue #293) removed the `beta_vae.regularization_active` field along with the L1/L2
declarations themselves, after the #293 sweep found no benefit at any calibrated strength. A
`config_{tag}.json` written **before** #324 still carries the key; `apply_saved_config`'s
`hasattr` guard drops it without even a "fields NOT layered" diff line (that log only covers
fields that still exist but are not allowlisted). **No action required** — unlike the pre-#283
seed case, nothing is lost: the flag defaulted to `False`, and the objective now has no
regularization term at any setting, so loading a pre-#324 config (including the released
v1.0.0 one) reproduces the same numerics it always did.

The same field removal also shifts the **training-resume** config fingerprint. `config_fingerprint`
hashes the whole `beta_vae` section, so a run-state manifest persisted **mid-training** under
v1.0.0 (whose `beta_vae` dict still carried `regularization_active`) hashes differently after the
#324 upgrade → `config_changed()` returns True, and an in-place `--load-tag` resume is treated as a
config change and **restarts from round 1** rather than resuming. This is inherent to removing any
fingerprinted field and is the intended clean-break behavior — the config-drift guard logs the
mismatch loudly (nothing silent) — and it only bites the narrow window of upgrading the codebase
mid-v1.0.0-training and relaunching to resume. A fresh (post-#324) training run is unaffected.
