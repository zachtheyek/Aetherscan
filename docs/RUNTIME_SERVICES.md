# Runtime Services

This document covers the three infrastructure singletons every run stands on: the **logger**
([`src/aetherscan/logger/`](../src/aetherscan/logger)) with its queue architecture and Slack
integration, the **ResourceManager**
([`src/aetherscan/manager/manager.py`](../src/aetherscan/manager/manager.py)) that owns
pool/shared-memory/process lifecycles, signal handling, and cleanup ordering, and the
**resource monitor**
([`src/aetherscan/monitor/monitor.py`](../src/aetherscan/monitor/monitor.py)). They
initialize early (see the init order in [`ARCHITECTURE.md`](ARCHITECTURE.md)) and outlive
everything else.

## Logger

### Queue architecture

Naive logging from a process tree — dozens of pool workers plus a handful of main-process
threads — interleaves output and contends on handler locks. Aetherscan funnels **every**
record through one consumer instead:

```
main process:  logging.* → QueueHandler ─┐
pool workers:  logging.* → QueueHandler ─┼→ multiprocessing.Queue → QueueListener thread
producer tree: QueueHandler (own queue,  ─┘        │
               relayed by RoundDataProducer)       ├→ FileHandler   logs/aetherscan_{tag}.log (mode="w")
                                                   ├→ StreamHandler stdout
                                                   └→ SlackHandler  (optional)
```

`Logger.__init__` sets the root logger to DEBUG (handlers do the filtering), attaches a
`QueueHandler` writing to a shared `multiprocessing.Queue`, and starts a `QueueListener`
thread that drains it into the real handlers, each with its own configured level
(`logger.{console,file,slack}_level`, all INFO by default). Consequences:

- Each run's log file is **tag-scoped** (`logs/aetherscan_{tag}.log`, named from the effective
  `--save-tag`); `mode="w"` truncates at startup, so a same-tag rerun overwrites that tag's own
  log while differently-tagged runs no longer clobber each other.
- **Fork-started workers** inherit the queue: `init_worker_logging()` (called from every pool
  initializer) resets the worker's root logger to a single `QueueHandler` — and resets
  `sys.stdout`/`sys.stderr` to the real streams, because the inherited `StreamToLogger`
  objects (below) would otherwise re-enqueue everything twice.
- **Spawn-started processes** (the `RoundDataProducer`) can't inherit the singleton; they log
  into their own spawn-context queue which the parent relays into the same handler set via a
  second `QueueListener` (see [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)).
- TF's logger is stripped of its handlers and set to propagate into the root; Python
  warnings are captured (`logging.captureWarnings(True)`).

### The stderr-redirect gotcha

`Logger.__init__` replaces `sys.stdout` with a `StreamToLogger` at INFO and `sys.stderr` with
one at **ERROR**. That means:

1. Bare `print()` and C-library stdout end up in the log (which is why `print()` is banned
   outside `utils/` — it's not lost, but it's unformatted).
2. **Anything writing to stderr becomes an ERROR record** — and ERROR-level records are
   broadcast to the main Slack channel (below). Progress bars are the classic trap: SHAP's
   tqdm writes its refreshes to stderr, so every refresh would spam Slack as a separate
   ERROR and eventually trip the webhook. `train.py:_silence_stderr()` exists precisely for
   this — it redirects stderr to `/dev/null` around SHAP's explainer calls (exceptions still
   propagate; they don't travel through the stream). Wrap any new dependency that draws
   progress bars the same way, or configure the bar off.

### Slack integration

When `SLACK_BOT_TOKEN` (and a channel via `SLACK_CHANNEL` or `logger.slack_channel`) is set,
a `SlackHandler` ([`slack_handler.py`](../src/aetherscan/logger/slack_handler.py)) joins the
listener's handler list; otherwise Slack quietly disables itself.

- **Per-run thread.** `start_run()` posts one summary message (host, CPU/GPU/RAM inventory,
  the CLI invocation) and caches its thread timestamp; every subsequent record is posted as a
  **reply in that thread**, so one channel can carry many runs without interleaving.
- **Batching.** Records buffer up to `slack_buffer_size` (100) or `slack_flush_interval`
  (60 s) and go out as one combined message, color-coded by the batch's highest severity and
  truncated to Slack's limits.
- **Broadcast escalation.** Records at `slack_broadcast_level`+ (ERROR) are echoed to the
  main channel, not just the thread — failures surface without opening the thread.
- **Image uploads.** Every plot the pipeline saves is pushed via
  `Logger.upload_image_to_slack()` → `SlackHandler.upload_file()`: the file lands in the run
  thread, with a link-back message broadcast to the main channel by default.
- **Failure hygiene.** Send failures retry (`slack_retry_attempts`) and repeated failures
  put the handler in a cooldown so a dead webhook can't stall the run. The handler's own
  diagnostics print to the *real* stderr (`sys.__stderr__`) — logging them would recurse.

> [!WARNING]
> Anything logged at INFO or above may reach Slack. Never log secrets, tokens, or
> internal URLs ([`SECURITY.md`](../SECURITY.md)).

## ResourceManager

The ResourceManager is the registry of everything that must not outlive the run:
multiprocessing pools, spawned processes, POSIX shared memory, and the other service
singletons. Modules never call `Pool(...)`/`SharedMemory(...)` directly for long-lived
resources — they go through the manager so cleanup is centralized and ordered.

### Managed resources

| Wrapper | Created via | Cleanup behavior |
| --- | --- | --- |
| `ManagedPool` | `create_pool(n_processes, name, initializer, initargs)` | `terminate()` → `join()` with `manager.pool_terminate_timeout`; workers still alive after the timeout are **SIGKILL-escalated** individually (`_force_kill_workers`). Termination sends SIGTERM first, which is what triggers the workers' cleanup handlers. |
| `ManagedProcess` | `register_process(process, name)` (e.g. the RoundDataProducer) | `terminate()` → `join(timeout)` → `kill()` escalation (`close_process`). `_reap_process_subtree()` (best-effort recursive-children kill) now runs in **all three** `close()` paths — survived-SIGTERM, exception fallback, and dead-on-entry — not just the SIGKILL escalation. The dead-on-entry path is best-effort (children have already been reparented); on the producer side this is covered by the ppid watch. |
| `ManagedSharedMemory` | `create_shared_memory(size, name)` | `close()` + `unlink()` in the creator, then a verification probe (`_check_unlinked`) that re-attaches by name to confirm the segment is really gone — leaked `/dev/shm` blocks survive process death and eat node RAM. |

The **creator-unlinks rule** ([`CLAUDE.md`](../CLAUDE.md)): workers attaching to a shared
block only ever `close()` their own mapping (their SIGTERM handlers do exactly that — and
**never log**, because the queue handler's feeder thread needs the GIL and a blocked handler
deadlocks termination; see the canonical worker handler in
[`data_generation.py`](../src/aetherscan/data_generation.py)`:_init_worker`). Only the main
process, via the manager, unlinks.

### Signal handling

`_register_cleanup_handlers()` registers `cleanup_all` with `atexit` and installs a handler
for SIGINT/SIGTERM:

- **First signal**: run `cleanup_all()` once, then `sys.exit(0)`.
- **Second signal** while cleanup is still running: reset both handlers to `SIG_IGN` and
  `os._exit(130)` — a hard exit. Re-entering `cleanup_all` from the same (main) thread would
  self-deadlock on the non-reentrant `_cleanup_lock`, so the force-quit is the only safe
  response to an impatient double Ctrl-C.
- Signals delivered to forked workers are ignored by the handler (PID check) — workers have
  their own SIGINT-ignore + SIGTERM-cleanup handlers installed by their pool initializers.

### Cleanup order

`cleanup_all()` (idempotent, lock-guarded) tears down in a strict order — each stage may
still need the ones after it:

1. **Processes** — before shared memory: the producer's workers hold attachments to the
   background-plate block.
2. **Pools** — same reasoning.
3. **Shared memory** — nothing is attached anymore; unlink + verify.
4. **Monitor** — stopping it triggers the final resource plot, which *queries the DB*.
5. **Database** — final buffer flush; nothing writes after this.
6. **Logger** — last, so every earlier stage could still log. Nothing can log after it stops.

This is also why `main()` calls `manager.cleanup_all()` in its `finally` block explicitly:
the DB writer, listener, and monitor threads are non-daemon, and without an explicit cleanup
they would block interpreter exit before `atexit` ever ran.

## Resource monitor

A 1 Hz background thread (`monitor.monitor_interval`) sampling into the `system_resources`
table ([`DATABASE.md`](DATABASE.md)) via the normal write queue:

| Metric (`resource_type` / `resource_name`) | Source |
| --- | --- |
| `cpu / system_total`, `ram / system_total` | `psutil.cpu_percent()`, `psutil.virtual_memory().percent` |
| `cpu / process_tree`, `ram / process_tree` | `get_process_tree_stats()` over the main process + all descendants |
| `gpu / {name}_utilization`, `gpu / {name}_memory` | `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total` per device |

### Why PSS, not RSS

`get_process_tree_stats()` sums memory across the whole process tree using **PSS**
(Proportional Set Size, `psutil.Process.memory_full_info().pss`) rather than RSS. RSS counts
shared pages once *per process*: with the multi-GB background plates mapped into dozens of
injection workers, summing RSS across the tree would multiply that memory by the worker count
and happily report more than the machine has. PSS divides each shared page by the number of
processes sharing it, making the per-process values **additive** — the tree sum is the tree's
actual footprint. (PSS is Linux-specific; the corresponding tests self-skip elsewhere, and
known accounting quirks are tracked in
[`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) (#6, now resolved; #11).) CPU is normalized against the system
core count, so 100 % means "all cores busy".

### The shutdown plot

When the monitor stops (during cleanup), `_save_plot()` queries the run's samples and renders
`plots/resource_utilization_{tag}.png` — three stacked, time-aligned panels:

1. **CPU** — Aetherscan process tree (filled) vs system total.
2. **RAM** — same pair; the gap between them is other users/jobs on the node.
3. **GPU** — per-device utilization (solid, left axis) and memory (dashed, right axis) on a
   shared twin-axis panel with a combined legend.

The x-axis is minutes since monitor start, so the panels line up with the log timeline: data
generation reads as CPU plateaus, epochs as GPU bands, per-cadence inference as alternating
CPU/GPU activity. The figure is uploaded to the run's Slack thread like every other plot.
Missing stretches in the panels are a known symptom
([`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) #10).

### Stage annotations

When `monitor.annotate_stages` is enabled (config default `True`), `_save_plot()` overlays the
run's pipeline-stage spans as solid `dimgray` vertical boundary lines at each span's right edge
on **every panel** (CPU, RAM, GPU — they share an x-axis) via `_annotate_stage_spans`; the leaf
stage name is labeled once, just left of each line and angled 30° from horizontal, on the
**CPU (top) panel only** so the other panels stay uncluttered. This turns the utilization curves
into a self-explaining timeline: the region ending at a `round_03` line is round 3 data
generation, the region ending at an `epochs` line is training, and so on. The spans come from
the `pipeline_stages` table written by the always-on stage timers — see
[`BENCHMARKING.md`](BENCHMARKING.md) for the timers and [`DATABASE.md`](DATABASE.md#pipeline_stages-stage-timers-schema-v4)
for the table.

Two details keep it readable and correct:

- **Depth ≤ 2 only.** `select_annotation_spans` keeps spans whose dot-name has at most two
  components (`train.round_03`, not `train.round_03.epochs`) — the deep per-ON-file and
  encode/rf sub-stages stay report-tool-only so the panels don't drown in divider lines. Every
  line uses the same `dimgray` color (`_ANNOTATION_COLOR`) and is labeled (once, on the CPU
  panel) with the leaf component (`round_03`).
- **Flush first.** The method calls `db.flush()` before querying, so spans recorded moments
  before shutdown (`final_save`, `inference.viz`) make it onto the plot. This is safe because
  the writer thread outlives the monitor in the cleanup order (monitor stops before db); a
  flush timeout just means the very newest spans are missing, never an error. Annotation
  failures are caught and logged — a benchmarking overlay must never break the resource plot.

## Live monitoring dashboard

An auto-launched Streamlit app ([`src/aetherscan/dashboard.py`](../src/aetherscan/dashboard.py))
gives a live view of a run without waiting for the shutdown plot. `launch_dashboard()`
([`src/aetherscan/dashboard_launcher.py`](../src/aetherscan/dashboard_launcher.py)) spawns it
from `main.py` at run start, gated on `config.monitor.dashboard_enabled` (default `True`) and
served on `config.monitor.dashboard_port` (default `8501`). It reads every plot the DB can
reconstruct plus a PNG gallery.

- **First-class RF tab.** Alongside the beta-VAE views, the dashboard surfaces the RF stage's
  eval metrics as a dedicated **RF** tab: metric tiles (accuracy, ROC-AUC, average precision,
  Brier score), binary + sub-type confusion heatmaps, per-sub-type accuracy bars, the
  ensemble-accuracy-vs-tree-count curve, and val P(true) confidence quantiles. These are
  driven by the scalars and `ensemble_val_accuracy` series that `train.py` writes to
  `training_stats` under `model_name='rf'` (see [`DATABASE.md`](DATABASE.md) and
  [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)); the SHAP, decision-boundary, and
  calibration figures stay available under the PNG gallery.
- **Served headless.** The subprocess runs `--server.headless`, so on a cluster you
  SSH-forward the port to view it locally (`ssh -L 8501:localhost:8501 ...`); the launcher logs
  the exact forward instructions.
- **Opt out with `--no-dashboard`.** The `--dashboard` / `--no-dashboard` flag (and
  `--dashboard-port`) is on both subcommands.
- **Fully guarded.** A missing `streamlit` (it lives in the optional `dashboard` extra) or any
  spawn failure only warns and never aborts the run — the dashboard is optional observability.
  Teardown is registered via `atexit` and the `SIGTERM`/`SIGINT` handlers so the server is
  reaped with the run.
- **Manual runs against a saved DB.** The `aetherscan-dashboard` console script
  ([`src/aetherscan/dashboard_cli.py`](../src/aetherscan/dashboard_cli.py), registered under
  `[project.scripts]`) forwards its args verbatim to `dashboard.py`'s argparse:
  `aetherscan-dashboard --db-path /path/to/aetherscan.db --tag final_v1`. In a source checkout /
  the container (no installed entry point), run the same shim as
  `PYTHONPATH=src python -m aetherscan.dashboard_cli <args>`. Either way it re-execs
  `python -m streamlit run <packaged dashboard.py> -- <args>` — Streamlit must own the process,
  so `python -m aetherscan.dashboard` does **not** work (`st.*` calls outside a
  ScriptRunContext render nothing). Everything after the script is forwarded to `dashboard.py`'s
  argparse (`--db-path` / `--tag` / `--plots-dir` / `--refresh`), so `aetherscan-dashboard` does
  **not** expose Streamlit's own `--server.*` flags; to set those (e.g. a custom `--server.port`),
  run the verbose form directly:
  `python -m streamlit run <packaged dashboard.py> --server.port 9000 -- --db-path … --tag …`.
