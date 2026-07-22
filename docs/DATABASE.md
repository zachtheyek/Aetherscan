# Database

This document covers Aetherscan's persistence layer
([`src/aetherscan/db/db.py`](../src/aetherscan/db/db.py)): the schema of every table, the
single-writer-thread design and its flush / mark-superseded protocols, the migration
mechanism, the query API, and how big to expect the database to get.

## TL;DR

One SQLite file — `{output_path}/db/aetherscan.db`, WAL mode — behind a thread-safe
`Database` singleton (`get_db()` accessor). All writes go through an in-process
`queue.Queue` drained by **one background writer thread** that batches rows and commits with
`executemany()`; reads open short-lived connections directly. Failed-attempt rows are never
deleted — they're flagged `superseded = 1`, and every query filters them out by default.
Schema evolution is a minimal `PRAGMA user_version` gate (currently version 4).

> [!IMPORTANT]
> The write queue is a **thread** queue, not process-safe. Worker *processes* must never call
> `db.write_*` — stats produced in workers travel back over multiprocessing channels and are
> written by main-process threads (see the producer/drainer design in
> [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)).

## Writer-thread design

**Why:** SQLite allows one writer at a time; concurrent writers from the monitor thread, the
training loop, and the round-data drainer would contend on `SQLITE_BUSY`. Serializing all
writes through a single consumer eliminates that class of failure and makes ordering
guarantees possible.

`write_*` methods are non-blocking: they validate/sanitize their arguments and `put()` a
`(table, values_tuple)` record on the queue. The writer loop (`_writer_loop`) accumulates
records into a buffer and flushes when either the buffer reaches
`db.write_buffer_max_size` (100 records) or `db.write_interval` (5 s) elapses. A flush
(`_flush_buffer`) groups the buffer by table and bulk-inserts each group with a single
`executemany()` per table — SQL parsed once, one commit per flush; the batch is
all-or-nothing (errors are logged and the loop continues; a failed write never kills the
thread). On shutdown (`stop()`), the loop drains and performs a final flush.

Lifecycle: `init_db()` constructs the singleton (schema init + migration) and `start()`s the
writer thread; the ResourceManager stops it during `cleanup_all()` — after the monitor (which
still writes samples) and before the logger (see
[`RUNTIME_SERVICES.md`](RUNTIME_SERVICES.md)).

### Flush protocol

Reads that must observe queued-but-unwritten rows (every plot function does this) call
`db.flush(timeout=...)`: a `(_FLUSH_SENTINEL, event)` tuple is queued; when the writer
dequeues it, it flushes the buffer immediately and sets the event. Queue FIFO ordering makes
the semantics exact: everything queued *before* the sentinel is on disk when `flush()`
returns True. Returns False on timeout or if shutdown began mid-wait — callers treat that as
"plot may be missing the newest rows", not as fatal.

### Mark-superseded protocol

`mark_superseded(table, tag, *, round_ge=None, npy_path=None)` implements the stale-data
semantics both retry systems rely on. It ships a command tuple through the same queue
(`_MARK_SUPERSEDED_SENTINEL`), so single-writer semantics are preserved and ordering is
airtight: the writer first flushes its buffer (every row queued before the command is
covered), then runs

```sql
UPDATE {table} SET superseded = 1
WHERE superseded = 0 AND tag = ?
  [AND round_number >= ?]      -- round_ge (training tables)
  [AND npy_path = ?]           -- npy_path (inference tables)
```

while rows queued *after* the command keep `superseded = 0`. The call blocks until the mark
lands (same timeout/shutdown semantics as `flush`); with no writer thread running it executes
inline. Table/filter combinations are whitelisted (`_SUPERSEDE_TABLES`): `round_ge` applies
to `training_stats` / `injection_stats` / `latent_snapshots`, `npy_path` to
`inference_results` / `inference_cadences`.

Who calls it:

- Training resume (`TrainingPipeline._init_run_state`): rows for `(tag, round >= resume)` in
  the three training tables — a dead attempt's partial epochs would otherwise appear twice in
  every curve.
- Inference retry (`main._infer_cadence`): `inference_results` rows for `(tag, npy_path)`
  before fresh positives land, and the cadence's old manifest rows before the new
  `status='inferred'` row.

Rows are **never deleted** — pass `include_superseded=True` to any query to audit what a
failed attempt wrote.

## Schema

Seven tables, created idempotently (`CREATE TABLE IF NOT EXISTS`) in `_init_database()`, each
with a composite index matched to its dominant filter pattern. All rows carry `tag` (the run's
save tag — the primary provenance key), and all but `pipeline_stages` carry `timestamp`
(write time, `REAL` Unix seconds); `pipeline_stages` records explicit `start_time`/`end_time`
span bounds instead.

| Table | Rows | `superseded`? | Added in |
| --- | --- | --- | --- |
| `system_resources` | 1 Hz monitor samples | no (attempt-agnostic history) | v0 |
| `injection_stats` | per generated cadence | yes | v0 (+ column v1) |
| `training_stats` | per training epoch | yes | v0 (+ column v1) |
| `latent_snapshots` | per viz cadence per capture | yes | v0 (+ column v1) |
| `inference_results` | positives only | yes | v0 (+ column v1) |
| `inference_cadences` | per-cadence run manifest | yes | v2 |
| `pipeline_stages` | per timed stage span | no (attempt-agnostic history) | v4 |

The `superseded` column and its default-filtering are what make same-tag retries safe; see
[the migration section](#schema-migration) for how the `user_version` stamp maps to these.

### `system_resources`

1 Hz samples from the resource monitor ([`RUNTIME_SERVICES.md`](RUNTIME_SERVICES.md)).

| Column | Type | Notes |
| --- | --- | --- |
| `resource_type` | TEXT | `cpu` / `ram` / `gpu` |
| `resource_name` | TEXT | `system_total`, `process_tree`, or `{gpu_name}_{utilization,memory}` |
| `value`, `unit` | REAL, TEXT | Percentages throughout |
| `metadata` | TEXT | Optional JSON |

Index: `(tag, timestamp)`. No `superseded` column — resource samples are attempt-agnostic
history.

### `injection_stats`

Signal-injection provenance, written per generated cadence by the round-data drainer
(`data_generation.write_segment_stats`).

| Column | Type | Notes |
| --- | --- | --- |
| `stat_name`, `value` | TEXT, REAL | Intensity statistics (`global_{mean,median,std,mad,skew,kurtosis}`), signal characteristics (`eti_*` / `rfi_*`: snr, drift_rate, signal_width, starting_bin, slope_pixel, y_intercept), intersection-retry telemetry for `true_eti_rfi` samples (`intersection_retries`, `intersection_retry_capped`; `injection_stage=NULL`), or segment metadata (`snr_range_floor/ceil`, `num_samples`, `inject_duration`) |
| `round_number`, `chunk_number`, `sample_index`, `background_index` | INTEGER | Position of the sample in the generation layout (`sample_index`/`background_index` NULL for segment metadata) |
| `signal_class`, `signal_type` | TEXT | `main`/`true`/`false`; `false_no_signal`/`false_with_rfi`/`true_only_eti`/`true_eti_rfi` |
| `injection_stage` | TEXT | `A` (raw background) / `B` (post-injection) / `C` (post-normalization); NULL for signal characteristics and metadata |
| `is_finite` | INTEGER | 0 when the value was NaN/Inf at write time (sanitized; filterable) |
| `slope_clamped` | INTEGER | 1 when the injection's drift slope hit the near-zero clamp |
| `superseded` | INTEGER | Default 0 |

Index: `(tag, timestamp, stat_name, signal_type, injection_stage)`.

### `training_stats`

Per-epoch training telemetry (~21 rows per epoch): losses (`total_loss`,
`reconstruction_loss`, `kl_loss`, `true_loss`, `false_loss` + `val_` variants), gradient
statistics (`gradient_norm_{mean,max,std}`, `clipping_rate`), `learning_rate`, durations,
step counts, and the round's SNR floor/ceiling.

| Column | Type | Notes |
| --- | --- | --- |
| `model_name` | TEXT | Currently always `beta_vae` |
| `stat_name`, `value` | TEXT, REAL | |
| `round_number`, `epoch_number` | INTEGER | 1-based |
| `superseded` | INTEGER | Default 0 |

Index: `(tag, timestamp, model_name, stat_name)`.

### `latent_snapshots`

The withheld viz batch's latent vectors, captured every `latent_viz_step_interval` training
steps — the raw material of the latent-space GIFs.

| Column | Type | Notes |
| --- | --- | --- |
| `round_number`, `epoch_number`, `step_number` | INTEGER | Capture coordinates |
| `cadence_index` | INTEGER | Position in the (persistent) viz batch |
| `signal_type` | TEXT | The cadence's class |
| `latent_vector` | TEXT | JSON: `(6, latent_dim)` `z_mean`, rounded to 8 decimals |
| `snr_base`, `snr_range` | INTEGER | Curriculum stage at capture time |
| `superseded` | INTEGER | Default 0 |

Index: `(tag, timestamp, model_name, round_number, epoch_number, step_number)`.

### `inference_results`

**Positives only**: one row per snippet whose `P(true)` cleared the classification threshold.
This is a deliberate size trade — at threshold 0.99 the negatives are ~everything, and the
per-cadence aggregates that summaries need live in the manifest table instead.

| Column | Type | Notes |
| --- | --- | --- |
| `npy_path`, `snippet_index` | TEXT, INTEGER | The stamp file and row — enough to reload the exact waterfall |
| `prediction`, `confidence` | INTEGER, REAL | 1; probability of the predicted class (see [`MODELS.md`](MODELS.md)) |
| `latent_vector` | TEXT | JSON, flattened `(6 · latent_dim,)` |
| `target`, `session`, `cadence_id`, `band` | TEXT/INTEGER | From the CSV group key |
| `frequency_mhz` | REAL | The snippet's stamp center frequency |
| `timestamp_observed` | REAL | The `.h5` header's `tstart` (MJD) |
| `h5_path` | TEXT | First observation of the cadence |
| `superseded` | INTEGER | Default 0 |

Index: `(tag, timestamp, confidence, prediction)`.

### `inference_cadences` (run manifest, schema v2)

One row per (cadence, stage transition); the newest live row per `(tag, npy_path)` is the
cadence's current state, older rows having been superseded. Drives the stage-aware inference
resume ([`INFERENCE_PIPELINE.md`](INFERENCE_PIPELINE.md)).

| Column | Type | Notes |
| --- | --- | --- |
| `csv_path`, `cadence_key` | TEXT | Source catalog; the group key as JSON |
| `npy_path` | TEXT | The cadence's stamp file (the resume key, with `tag`) |
| `status` | TEXT | `preprocessed` → `inferred` (or `failed`) |
| `n_stamps`, `n_candidates` | INTEGER | Aggregates (candidates only on `inferred` rows) |
| `confidence_summary` | TEXT | JSON from `inference.summarize_confidences`: n, threshold, above-threshold count, mean/min/max, quantiles p01–p99 |
| `duration_s` | REAL | Stage wall time |
| `config_fingerprint` | TEXT | Fingerprint of the inference-result-affecting config; the stage-aware resume reuses an `inferred` row only if its stored fingerprint matches the current run (guards a reused `--save-tag` with a changed inference config) |
| `superseded` | INTEGER | Default 0 |

Index: `(tag, npy_path, status)` — the resume lookup.

### `pipeline_stages` (stage timers, schema v4)

One row per timed pipeline-stage span, written by the always-on stage timers in
[`aetherscan.benchmark`](../src/aetherscan/benchmark.py) (`stage_timer` / `record_stage`) via
`db.write_pipeline_stage()`. Read back by [`utils/benchmark_report.py`](../utils/benchmark_report.py)
and the monitor's stage-band overlay. Full context in [`BENCHMARKING.md`](BENCHMARKING.md).

| Column | Type | Notes |
| --- | --- | --- |
| `stage` | TEXT | Hierarchical dot-name (`train.round_02.data_generation`) |
| `start_time`, `end_time` | REAL | Unix timestamps bounding the span |
| `duration_s` | REAL | Derived (`end_time - start_time`) at write time so the table stays internally consistent |
| `metadata` | TEXT | Optional pre-serialized JSON (e.g. `{"source": "producer"}`, or `{"status": "failed", ...}` on a span whose block raised) |

Index: `(tag, start_time)`. **No `superseded` column** — timing spans are attempt-agnostic
history like `system_resources`. Retried stages simply append new rows, so consumers see every
attempt, each with its own span.

## Schema migration

`_migrate_schema()` runs on every startup, gated on `PRAGMA user_version`
(`_SCHEMA_VERSION = 4`). The stamp maps to schema features as:

| `user_version` | What it added | Migration work |
| --- | --- | --- |
| v0 | pre-versioning baseline (any db with no stamp) | — |
| v1 | `superseded INTEGER DEFAULT 0` on `training_stats`, `injection_stats`, `latent_snapshots`, `inference_results` | additive `ALTER TABLE ... ADD COLUMN` |
| v2 | the `inference_cadences` run-manifest table | none (whole-table `CREATE TABLE IF NOT EXISTS`) |
| v3 | `config_fingerprint TEXT` on `inference_cadences` | additive `ALTER TABLE ... ADD COLUMN` |
| v4 | the `pipeline_stages` stage-timing table | none (whole-table `CREATE TABLE IF NOT EXISTS`) |

- **v0 → v1**: `ALTER TABLE ... ADD COLUMN superseded INTEGER DEFAULT 0` on the four tables
  above — the only in-place change SQLite supports is additive `ADD COLUMN`, which is exactly
  what supersede semantics needed. A per-table column-existence check (`PRAGMA table_info`)
  keeps the step idempotent even if `user_version` was lost (a db file copied without its
  journal).
- **v1 → v2** (`inference_cadences`) and **v3 → v4** (`pipeline_stages`): no migration step
  needed — `CREATE TABLE IF NOT EXISTS` in `_init_database()` creates each new table for old
  and new databases alike *before* `_migrate_schema()` runs; only the version stamp advances.
- **v2 → v3** (`inference_cadences.config_fingerprint`): additive
  `ALTER TABLE inference_cadences ADD COLUMN config_fingerprint TEXT`, made idempotent by the
  same `PRAGMA table_info` column-existence check as the v0 → v1 step (a fresh db already has
  the column from `CREATE TABLE`, so the ALTER only patches a db that created the v2 table
  before the column existed).

Fresh databases get the full current schema from the CREATE statements and are just stamped.
The pattern to follow for future changes: bump `_SCHEMA_VERSION`, add a
`if version < N:` block with additive, idempotent statements, and rely on
`CREATE TABLE IF NOT EXISTS` for whole new tables.

## Query API

Each table has a `query_*` method returning `list[dict]`
(`query_system_resource`, `query_injection_stat`, `query_injection_stat_stability`,
`query_training_stat`, `query_latent_snapshots`, `query_latent_snapshot_keys`,
`query_inference_result`, `query_inference_cadences`, `query_pipeline_stages`). Shared
conventions:

- **String filters** (`tag`, `stat_name`, `status`, ...) accept a single value (`=`) or a
  list (`IN`); **range filters** come as `start_*`/`end_*` pairs (inclusive);
  `start_time`/`end_time` bound `timestamp` — plots pass the run's `start_time` so
  multi-attempt runs query their whole history.
- **`columns`** selects a subset, validated against a per-table whitelist
  (`_build_select`) — column names never reach SQL unchecked, and everything is
  parameter-bound.
- **`include_superseded=False`** by default on every table that has the column: stale rows
  from failed attempts are invisible unless you ask. The inference resume *relies* on the
  default — a superseded `inferred` row must read as "not done".
- JSON-typed columns (`latent_vector`, `cadence_key`, `confidence_summary`) come back as
  strings; callers `json.loads` them.

`get_db_stats()` returns row counts per table, the covered time range, and the database file
size — logged at startup.

## Growth expectations

Rules of thumb at full-scale defaults (dominant terms only):

- **`injection_stats` is the giant.** Every generated cadence writes 18 intensity rows
  (6 statistics × 3 stages A/B/C) plus 0–14 signal/telemetry rows depending on its type
  (0 / 6 / 6 / 14 for the four equal-weighted `main`-class types — 0 for the no-signal type,
  6 per injected signal, and an extra 2 intersection-retry telemetry rows on top of the 12
  signal-characteristic rows for `true_eti_rfi`, so their mean is 6.5). That is
  **~24.5 rows per cadence** (18 + 6.5). A training round generates `3 × num_samples_beta_vae`
  cadences (main + true + false), so at defaults:
  `3 × 499 200 × ~24.5 ≈ 37 M rows per round`, times 20 rounds plus the RF dataset. This is
  why writes are batched, why the drainer runs off the training critical
  path, and why the injection plots subsample (`plot_injection_subsampling_count`). If the
  database size becomes a problem, this table is where the budget goes — smoke-scale runs
  (`--num-samples-beta-vae 3072`) keep it trivial.
- **`training_stats`**: ~21 rows/epoch → ~42 k rows for 20 × 100 epochs. Negligible.
- **`latent_snapshots`**: one row per viz cadence per capture — 960 cadences × one capture
  every `latent_viz_step_interval` steps (plus the final step) × epochs. At full scale
  (130 steps/epoch → 13 captures/epoch): ~25 M rows over 20 × 100 epochs, each carrying a
  48-float JSON vector. The second heaviest table; `latent_viz_step_interval` and
  `latent_viz_num_cadences_per_type` are the knobs.
- **`system_resources`**: (4 + 2 × n_GPUs) rows/second — ~1.2 M rows/day on a 6-GPU node.
- **`inference_results`**: positives only; at a 0.99 threshold this stays small by
  construction. `inference_cadences`: a handful of rows per cadence.
- **`pipeline_stages`**: one row per timed stage span — tens per training run, a few per
  inference cadence. Negligible.

WAL mode keeps readers (plots, resume queries) unblocked during heavy write phases; the WAL
file is checkpointed back into the main db periodically by SQLite itself.
