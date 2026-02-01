# Known Issues

This document tracks known issues, limitations, and workarounds in Aetherscan.

---

## 1. TensorFlow Prefetch Warnings

### Symptom

During training round transitions, you may see warnings like:

```
2026-01-24 03:52:38.996928: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: CANCELLED: GetNextFromShard was cancelled
         [[{{node MultiDeviceIteratorGetNextFromShard}}]]
2026-01-24 03:52:38.997040: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: CANCELLED: GetNextFromShard was cancelled
         [[{{node MultiDeviceIteratorGetNextFromShard}}]]
         [[RemoteCall]] [type.googleapis.com/tensorflow.DerivedStatus='']
```

### Cause

These warnings occur when TensorFlow's `.prefetch(tf.data.AUTOTUNE)` threads are cancelled during dataset cleanup between training rounds. The prefetch mechanism spawns background threads that may still be waiting on data when the main training loop finishes a round and clears the dataset.

### Impact

**None.** These warnings are benign and do not indicate data loss or training issues. The training pipeline handles round transitions correctly.

### Workaround

No action required. The warnings can be safely ignored. If they cause log noise, you can suppress TensorFlow warnings:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING
```

However, this also suppresses other potentially useful warnings, so it's not recommended.

### Status

**Won't fix.** This is expected TensorFlow behavior when using infinite datasets with `.repeat()` and cleaning up between epochs/rounds.

---

## 2. Resource Monitor CPU Normalization

### Symptom

System total CPU usage sometimes appears lower than Aetherscan process CPU usage in resource monitoring plots.

### Cause

The issue stems from how CPU percentages are aggregated across the process tree. When using `psutil` to collect CPU metrics:

1. Individual process CPU can exceed 100% on multi-core systems (e.g., 400% on a 4-core system means 100% utilization of each core)
2. We normalize by dividing by core count to get a 0-100% scale
3. However, child processes spawned by `multiprocessing.Pool()` may report their CPU independently
4. Race conditions during process tree enumeration can miss or double-count short-lived processes

### Impact

**Minor.** Resource utilization plots may show inconsistent CPU values. Training and inference correctness are not affected.

### Workaround

For accurate CPU monitoring, use external tools like `htop`, `nvidia-smi`, or system monitoring dashboards.

### Status

**Open.** See [GitHub Issue #12](https://github.com/zachtheyek/Aetherscan/issues/12).

### Related Code

`src/aetherscan/monitor/monitor.py:get_process_tree_stats()`

---

## 3. Pool Cleanup Hangs

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

If the pipeline hangs completely:

```bash
# Find stuck Python processes
ps aux | grep aetherscan

# Force kill (use PID from above)
kill -9 <pid>

# Clean up any orphaned shared memory
# List shared memory segments
ls /dev/shm/

# Remove orphaned segments (be careful!)
rm /dev/shm/shm_name_*
```

### Status

**Open.** Investigating root cause. May be related to Slack logger connections in child processes.

### Related Code

`src/aetherscan/manager/manager.py:ManagedPool.close()`

---

## 4. GPU Memory Fragmentation

### Symptom

After many training rounds, you may see OOM (Out of Memory) errors even though total GPU memory usage appears normal:

```
ResourceExhaustedError: OOM when allocating tensor with shape [batch, 6, 16, 512]
```

### Cause

TensorFlow's memory allocator can cause fragmentation over time:

1. Tensors of varying sizes are allocated/deallocated
2. Free memory becomes scattered in small chunks
3. Large allocations fail even though total free memory is sufficient

### Impact

**Moderate.** May require restarting training from checkpoint after many rounds.

### Workaround

1. **Enable memory growth** (already default in Aetherscan):
   ```python
   gpus = tf.config.experimental.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   ```

2. **Use TensorFlow's async allocator**:
   ```bash
   export TF_GPU_ALLOCATOR=cuda_malloc_async
   ```

3. **Force garbage collection between rounds** (already implemented):
   ```python
   tf.keras.backend.clear_session()
   gc.collect()
   ```

4. **Restart from checkpoint periodically**: For very long training runs (50+ rounds), consider checkpointing and restarting every 10-20 rounds.

### Status

**Mitigated.** The pipeline includes cleanup between rounds. For extremely long runs, manual restarts may still be needed.

---

## 5. Shared Memory Leaks

### Symptom

After abnormal termination (Ctrl+C, kill, crash), orphaned shared memory segments may remain:

```bash
$ ls /dev/shm/
shm_backgrounds_12345
shm_cadences_12346
```

Over time, these can consume significant system memory.

### Cause

Python's `multiprocessing.shared_memory.SharedMemory` requires explicit cleanup:

1. `shm.close()` releases the handle
2. `shm.unlink()` removes the segment from the filesystem

If the process crashes or is killed before `unlink()` is called, the segment persists until system reboot or manual cleanup.

### Impact

**Moderate.** Orphaned segments consume RAM. On systems with many Aetherscan runs, this can accumulate.

### Workaround

**Automatic cleanup**: The `ResourceManager` tracks all shared memory segments and attempts cleanup on exit (including signal handlers for SIGTERM, SIGINT).

**Manual cleanup**:

```bash
# List shared memory segments
ls /dev/shm/

# Remove Aetherscan segments (check names match your session)
rm /dev/shm/shm_backgrounds_*
rm /dev/shm/shm_cadences_*

# Or remove all (be careful on shared systems!)
# rm /dev/shm/*
```

**Programmatic cleanup**:

```python
from multiprocessing import shared_memory

# List all shared memory (requires knowing names)
# Unfortunately, Python doesn't provide a way to enumerate all segments

# If you know the name:
try:
    shm = shared_memory.SharedMemory(name="shm_backgrounds_12345")
    shm.close()
    shm.unlink()
except FileNotFoundError:
    pass
```

### Status

**Mitigated.** The `ResourceManager` handles normal cleanup. Abnormal termination still requires manual cleanup.

### Related Code

`src/aetherscan/manager/manager.py:ManagedSharedMemory`

---

## 6. Database Lock Contention (Rare)

### Symptom

Slow database writes or occasional `SQLITE_BUSY` errors in logs.

### Cause

Although the database uses queue-based async writes, contention can occur if:

1. Write buffer fills up faster than the writer thread can process
2. Many concurrent reads block the writer
3. File system is slow (network storage, spinning disk)

### Impact

**Minor.** May cause metric data to be dropped if queue overflows.

### Workaround

1. **Increase buffer size** in config:
   ```python
   config.db.write_buffer_max_size = 200  # Default is 100
   ```

2. **Decrease write interval**:
   ```python
   config.db.write_interval = 2.0  # Default is 5.0 seconds
   ```

3. **Use local SSD storage** for the database file

### Status

**Rare.** Only observed under extreme load.

---

## Reporting New Issues

If you encounter an issue not listed here:

1. Check [GitHub Issues](https://github.com/zachtheyek/Aetherscan/issues) for existing reports
2. If new, open an issue with:
   - Clear description of the problem
   - Steps to reproduce
   - System information (OS, GPU, CUDA version, Python version)
   - Relevant log output
   - Configuration used

---

## Issue Status Definitions

| Status | Meaning |
|--------|---------|
| **Open** | Actively investigating or planned for fix |
| **Mitigated** | Workaround implemented, full fix pending |
| **Won't fix** | Expected behavior or not worth the complexity to fix |
| **Closed** | Fixed in a specific version |
