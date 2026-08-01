# Database

This document covers Aetherscan's persistence layer
([`src/aetherscan/db/db.py`](../src/aetherscan/db/db.py)): the schema of every table, the
single-writer-thread design and its flush / mark-superseded protocols, the migration
mechanism, the query API, and how big to expect the database to get.

## TL;DR

One SQLite file — `{output_path}/db/aetherscan.db`, WAL mode — behind a thread-safe
`Database` singleton (`get_db()` accessor). All writes go through in-process
`queue.Queue`s — a foreground lane for everything, plus a bounded **bulk lane** for
high-volume injection-stat chunks (#277) — drained by **one background writer thread** that
batches rows and commits with `executemany()`; reads open short-lived connections directly.
Failed-attempt rows are never deleted — they're flagged `superseded = 1`, and every query
filters them out by default. Schema evolution is a minimal `PRAGMA user_version` gate
(currently version 8).

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
`db.write_buffer_max_size` (5000 records — raised from 100 in #277: a commit-and-fsync every
100 rows was one driver of the ~590 rows/s writer that let multi-hour backlogs build) or
`db.write_interval` (5 s) elapses. A flush
(`_flush_buffer`) groups the buffer by table and bulk-inserts each group with a single
`executemany()` per table — SQL parsed once, one commit per flush. A failed batch falls
back to per-row inserts (`_executemany_resilient`, #289): only the unbindable row(s) are
dropped, with an exact count logged — before v6 a single bad row (a NaN stat binding as
SQL NULL against a NOT NULL column) silently discarded every row in the flush. Errors are
logged and the loop continues; a failed write never kills the thread. Connections set `PRAGMA synchronous=NORMAL`: under WAL that only skips the
per-commit WAL fsync (the WAL is still synced at checkpoints) — a crash can lose the newest
commits but never corrupts the database, ample durability for diagnostic telemetry and the
removal of the dominant per-transaction fsync stall.

### The bulk lane (#277)

High-volume injection stats ride a separate, **bounded** queue: `write_injection_stats_bulk`
takes a whole batch of rows (per-row semantics identical to `write_injection_stat` —
NaN/Inf coercion with `is_finite=0`, one shared system-metadata lookup, which is also cached
per process now) and enqueues it in `db.bulk_chunk_rows`-sized chunks (50 000, also the bulk
transaction size), so a ~300 K-row segment costs a handful of queue operations instead of
~300 K. The lane is capped at `db.bulk_queue_max_items` chunks (32 — ~1.6 M rows in memory
at defaults; the old single unbounded queue grew to ~35 GB of RSS on a release run):
a full lane **blocks the enqueuer**, which is deliberately the round-data drainer thread
(background work that can afford to wait), never the training path. The writer services the
foreground lane with strict priority and consumes bulk chunks whenever it is idle.
`data_generation.write_segment_stats` batches each generated class-segment's rows into one
bulk call. With no writer thread running, bulk rows are written inline (mirroring
`mark_superseded`'s no-writer path).

Lifecycle: `init_db()` constructs the singleton (schema init + migration) and `start()`s the
writer thread; the ResourceManager stops it during `cleanup_all()` — after the monitor (which
still writes samples) and before the logger (see
[`RUNTIME_SERVICES.md`](RUNTIME_SERVICES.md)). `stop()` **drains both lanes to disk** before
the writer exits (the old behavior silently dropped everything still queued — up to tens of
millions of rows on a release run), with progress heartbeats and a `db.stop_drain_timeout`
cap (600 s); if the cap is hit the writer is force-stopped and the exact number of dropped
rows is logged at ERROR — never silently.

### Flush protocol

Reads that must observe queued-but-unwritten rows (every plot function does this) call
`db.flush(timeout=...)`: a `(_FLUSH_SENTINEL, event)` tuple is queued; when the writer
dequeues it, it flushes the buffer immediately and sets the event. Queue FIFO ordering makes
the semantics exact: everything queued *before* the sentinel is on disk when `flush()`
returns True. Returns False on timeout or if shutdown began mid-wait; since #277 the
training plot functions treat a False return by **skipping the figure** (a non-critical
failure) rather than rendering a partial result set.

`flush()` covers the **foreground lane only** — it must never queue behind a round's worth
of bulk injection rows. Readers of `injection_stats` gate on
`db.injection_backlog_rows(max_round=...)` instead: the count of bulk-lane rows enqueued but
not yet committed for rounds ≤ `max_round` (rows with `round_number=NULL` count against
every round, conservatively). `plot_injection_stats` skips its figures while the backlog for
the rounds being plotted is nonzero.

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
| `inference_results` | positives only | yes | v0 (+ column v1, + columns v5) |
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
(`data_generation.write_segment_stats` — one `write_injection_stats_bulk` batch per
class-segment, on the bounded bulk lane).

| Column | Type | Notes |
| --- | --- | --- |
| `stat_name`, `value` | TEXT, REAL | Intensity statistics (`global_{mean,median,std,mad,skew,kurtosis}`), signal characteristics (`eti_*` / `rfi_*`: snr, drift_rate, signal_width, starting_bin, slope_pixel, y_intercept), intersection-retry telemetry for `true_eti_rfi` samples (`intersection_retries`, `intersection_retry_capped`; `injection_stage=NULL`), or segment metadata (`snr_range_floor/ceil`, `num_samples`, `inject_duration`) |
| `round_number`, `chunk_number`, `sample_index`, `background_index` | INTEGER | Position of the sample in the generation layout (`sample_index`/`background_index` NULL for segment metadata) |
| `signal_class`, `signal_type` | TEXT | `main`/`true`/`false`; `false_no_signal`/`false_with_rfi`/`true_only_eti`/`true_eti_rfi` |
| `injection_stage` | TEXT | `A` (raw background) / `B` (post-injection) / `C` (post-normalization); NULL for signal characteristics and metadata |
| `is_finite` | INTEGER | 0 when the value was NaN/Inf at write time (sanitized; filterable) |
| `slope_clamped` | INTEGER | 1 when the injection's drift slope hit the near-zero clamp |
| `superseded` | INTEGER | Default 0 |

Indexes: `(tag, timestamp, stat_name, signal_type, injection_stage)`
(`idx_injection_stats_filter`) plus, since v7, the stat-scoped secondary
`(tag, stat_name, signal_type, injection_stage, timestamp)`
(`idx_injection_stats_by_stat`) — shaped for the end-of-run plot queries, whose equality
filters with run-wide timestamp bounds the first index can only answer by scanning the tag's
whole timestamp range. The trailing column is deliberately `timestamp`: that shape is chosen
by SQLite's default cost model with no `ANALYZE` stats (a `round_number`-trailing variant
measured as never chosen without `sqlite_stat1`), and it serves the round-scoped per-round
queries too via their span-tightened timestamp windows. No query changes: the planner picks
the better index per query.

### `training_stats`

Per-epoch beta-VAE training telemetry (~21 rows per epoch): losses (`total_loss`,
`reconstruction_loss`, `kl_loss`, `true_loss`, `false_loss` + `val_` variants), gradient
statistics (`gradient_norm_{mean,max,std}`, `clipping_rate`), `learning_rate`, durations,
step counts, and the round's SNR floor/ceiling — plus, since #282, `latent_dim` per-dimension
KL rows per epoch (`kl_dim_00` … `kl_dim_NN`, feeding the posterior-collapse diagnostics). The RF stage also writes here: at the tail
of `rf_train`, `train_random_forest()` persists ~25 scalar eval metrics (accuracy, ROC-AUC,
average precision, Brier score, per-sub-type accuracies, binary + sub-type × prediction
confusion cell counts, val P(true) quantiles) plus a `classification_threshold` row; the
`rf_plots` stage additionally writes the per-tree `ensemble_val_accuracy` series
(`epoch_number` = tree count) from inside `plot_rf_ensemble_accuracy_curve()`. All RF rows
use `model_name='rf'`; the dashboard reads them last-write-wins so `rf_plots` retries and
reused-tag stale scalars are absorbed.

| Column | Type | Notes |
| --- | --- | --- |
| `model_name` | TEXT | `beta_vae` for per-epoch training telemetry; `rf` for the RF stage's eval metrics (scalars + the `ensemble_val_accuracy` per-tree series) |
| `stat_name`, `value` | TEXT, REAL | |
| `round_number`, `epoch_number` | INTEGER | 1-based |
| `superseded` | INTEGER | Default 0 |
| `is_finite` | INTEGER | 0 when the value was NaN/Inf/None at write time (stored as 0.0; v6, #289). `query_training_stat`'s `only_finite` default drops these — sqlite binds NaN as SQL NULL, and before v6 the resulting NOT NULL violation silently discarded the *entire* flush batch |

Index: `(tag, timestamp, model_name, stat_name)` (`idx_training_stats_filter`).
Deliberately kept through the v7 index sweep: an equality-first reshape
(`tag, model_name, stat_name, timestamp`) was benchmarked and rejected — it regressed the
dominant run-window loss-curve fetch 1.8× (stat-sorted index order scatters the table-row
lookups timestamp order visits sequentially) and cost ~20% insert throughput, to speed only
the minor single-stat shape on tag partitions that stay ~58 k rows
(`benchmarks/bench_db_index_shapes.py`).

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

Index: `(tag, round_number, epoch_number, step_number, model_name, timestamp)`
(`idx_latent_snapshots_by_key`, reshaped in v7 from the original
`(tag, timestamp, model_name, round_number, epoch_number, step_number)`). The GIF pass
loads one capture per frame — up to `latent_viz_gif_max_frames` (500) queries with equality
on the capture key and the run window trailing — and the old timestamp-second shape
answered each by scanning the tag's whole window (measured 67× slower per frame at 2 M
rows; ~1.5 h → ~2 s per GIF pass at the ~100 M-row release scale). Key columns lead so the
dashboard's model-less latest-capture lookup walks the index backward instead of sorting
the partition; write cost is unchanged because `round/epoch/step` grow monotonically like
`timestamp`, so both shapes append at the B-tree's right edge
(`benchmarks/bench_db_index_shapes.py`).

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
| `screening_proba` | REAL | Deterministic pass-1 score of the #282 two-pass cascade (v5) |
| `mc_mean`, `mc_std` | REAL | Seeded MC mean/spread for pass-2 survivors — the mean carries the science threshold; NULL for snippets that never reached pass 2 (v5) |

Indexes: `(tag, timestamp, confidence, prediction)` plus, since v8, the partial
`idx_inference_results_supersede` on `(tag, npy_path) WHERE superseded = 0` — matching
`_execute_mark_superseded`'s exact predicate. The once-per-cadence supersede UPDATE blocks
the inference thread via the writer-queue sentinel, and the timestamp-led index only
narrowed it to the tag partition — a quadratic-in-catalog term on RFI-dense 6k-cadence
runs (#301).

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
(`_SCHEMA_VERSION = 8`). The stamp maps to schema features as:

| `user_version` | What it added | Migration work |
| --- | --- | --- |
| v0 | pre-versioning baseline (any db with no stamp) | — |
| v1 | `superseded INTEGER DEFAULT 0` on `training_stats`, `injection_stats`, `latent_snapshots`, `inference_results` | additive `ALTER TABLE ... ADD COLUMN` |
| v2 | the `inference_cadences` run-manifest table | none (whole-table `CREATE TABLE IF NOT EXISTS`) |
| v3 | `config_fingerprint TEXT` on `inference_cadences` | additive `ALTER TABLE ... ADD COLUMN` |
| v4 | the `pipeline_stages` stage-timing table | none (whole-table `CREATE TABLE IF NOT EXISTS`) |
| v5 | `screening_proba` / `mc_mean` / `mc_std` on `inference_results` (#282 two-pass inference) | additive `ALTER TABLE ... ADD COLUMN` |
| v6 | `is_finite INTEGER DEFAULT 1` on `training_stats` (#289 NaN-write hardening) | additive `ALTER TABLE ... ADD COLUMN` |
| v7 | the index sweep: the `idx_injection_stats_by_stat` secondary index on `injection_stats`; `latent_snapshots`' index reshaped to `idx_latent_snapshots_by_key` | `DROP INDEX IF EXISTS idx_latent_snapshots_filter` in the migration block; the CREATEs run in `_init_database()` and are re-executed there |
| v8 | the `idx_inference_results_supersede` partial index on `inference_results` (`(tag, npy_path) WHERE superseded = 0`, #301) | the `if version < 8` block creates it — deliberately NOT `_init_database()`: the partial predicate needs the `superseded` column the v1 ALTER adds, so an init-time CREATE would fail on a pre-v1 file. Fresh databases reach the block too (version 0 → current) |

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
- **v4 → v5** (`inference_results.{screening_proba,mc_mean,mc_std}`): additive
  `ALTER TABLE ... ADD COLUMN ... REAL` per column, idempotent via the same
  `PRAGMA table_info` existence check.
- **v5 → v6** (`training_stats.is_finite`, #289): additive
  `ALTER TABLE training_stats ADD COLUMN is_finite INTEGER DEFAULT 1`, same idempotence
  check. The `DEFAULT 1` backfill is exact, not approximate: a non-finite value could never
  have been written before v6 (it bound as NULL and the NOT NULL constraint rejected the
  whole batch), so every pre-v6 row is finite by construction.
- **v6 → v7** (the index sweep — every table's indexes audited against the production
  query shapes; per-table cost/benefit in `benchmarks/bench_injection_index.py` and
  `benchmarks/bench_db_index_shapes.py`):
  - `injection_stats` gains the secondary `idx_injection_stats_by_stat`: the end-of-run
    `plot_injection_stats` pass issues ~165 queries whose equality filters
    (tag/stat_name/signal_type/injection_stage) ride run-wide timestamp bounds, so
    `idx_injection_stats_filter` scanned the whole tag partition per query (~6 h projected
    at release scale).
  - `latent_snapshots`' index is reshaped to `idx_latent_snapshots_by_key`
    (`tag, round_number, epoch_number, step_number, model_name, timestamp`): the GIF pass's
    up-to-500 per-frame capture fetches each scanned the tag's whole window under the old
    timestamp-second shape (~1.5 h → ~2 s per pass at release scale, at no measurable
    write cost). The old `idx_latent_snapshots_filter` is dropped
    (`DROP INDEX IF EXISTS`) — the only shape it served better was already run-window ≈
    whole-partition.
  - Everything else was audited and deliberately kept — notably a `training_stats` reshape
    was benchmarked and rejected (see [that table's index note](#training_stats)).

  Like v2/v4, `_init_database()` does the CREATE work — `CREATE INDEX IF NOT EXISTS` runs
  for old and new databases alike before migration; the `if version < 7` block re-executes
  the (themselves idempotent) statements, performs the DROP, and advances the stamp.
- **v7 → v8** (`idx_inference_results_supersede`, #301): the supersede partial index on
  `inference_results` — see [that table's index note](#inference_results). Unlike v7, the
  `CREATE INDEX IF NOT EXISTS` lives **only** in the `if version < 8` block, never in
  `_init_database()`: its `WHERE superseded = 0` predicate needs the column the v1 ALTER
  adds, so an init-time CREATE would precede the ALTER and fail on a pre-v1 file. Fresh
  databases reach the block too (version 0 → current), so both paths land the same final
  index set.

Fresh databases get the full current schema from the CREATE statements and are just stamped
(the one v8 exception above lands via the migration block either way).
The pattern to follow for future changes: bump `_SCHEMA_VERSION`, add a
`if version < N:` block with additive, idempotent statements, and rely on
`CREATE TABLE IF NOT EXISTS` for whole new tables.

## Query API

Each table has a `query_*` method returning `list[dict]`
(`query_system_resource`, `query_system_resource_decimated`, `query_injection_stat`,
`query_injection_stat_stability`, `query_training_stat`, `query_latent_snapshots`,
`query_latent_snapshot_keys`, `query_inference_result`, `query_inference_cadences`,
`query_pipeline_stages`). Shared conventions:

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

One deliberate exception to the list-of-rows shape:
**`query_injection_stat_time_span(tag, start_round_number=None, end_round_number=None)`**
returns the `(MIN, MAX)` timestamp pair over a tag's `injection_stats` rows (optionally
bounded to a round range), or `None` when no rows match. It is a single whole-partition
aggregate with **no timestamp filter and no superseded/`is_finite` filtering** — a deliberate
**superset bound**: the span covers every row a filtered query could return for those rounds,
so a caller that intersects its own time window with the span can only narrow index scans,
never change a result set. It exists because `idx_injection_stats_filter` leads with
`(tag, timestamp)` and `round_number` is not in the index, so a round-scoped query with a
run-wide window re-scans the tag's entire row history (measured 10.5× slower at 12M rows,
quadratic over a campaign as history accumulates) — `plot_injection_stats` issues ~165 such
queries per call and now pays this one aggregate up front, tightening every window to the
plotted rounds' actual span (±1 s).

**`query_system_resource_decimated(tag, start_time, end_time, max_points_per_series)`**
(#301) returns a per-series uniformly-strided subset of `system_resources` rows (same dict
shape as `query_system_resource`): a
`ROW_NUMBER() OVER (PARTITION BY resource_type, resource_name ORDER BY timestamp)` window
keeps at most `max_points_per_series` uniformly-spaced points per
`(resource_type, resource_name)` line, with the stride computed from the largest series
(stride 1 degenerates to the plain query). It exists for the monitor's teardown resource
plot ([`RUNTIME_SERVICES.md`](RUNTIME_SERVICES.md)): a multi-week catalog run accumulates
tens of millions of samples while the plot renders ~2 k px wide, so materializing every row
cost a multi-GB teardown RAM spike for invisible detail.

`get_db_stats()` returns row counts per table, the covered time range, and the database file
size — **on-demand diagnostics only** since #301: its per-table `COUNT(*)` full scans
measured ~13 min per cold-cache launch on the 80 GB production DB (re-paid on every
retry-loop relaunch), so `Database.__init__` no longer calls it and startup logs only the
O(1) `db_size_bytes` pragma.

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
  why these rows ride the bounded bulk lane in per-segment batches (#277), why the drainer
  runs off the training critical
  path, and why the injection plots subsample (`plot_injection_subsampling_count`). If the
  database size becomes a problem, this table is where the budget goes — smoke-scale runs
  (`--num-samples-beta-vae 3072`) keep it trivial.
- **`training_stats`**: ~21 + `latent_dim` rows/epoch → ~58 k rows for 20 × 100 epochs at
  the default `latent_dim` 8; the RF stage adds
  a negligible tail (~25 scalars + `classification_threshold` + the #282 sweep/test/screen
  metrics + the per-tree
  `ensemble_val_accuracy` series ≈ `rf.n_estimators` rows). Still negligible.
- **`latent_snapshots`**: one row per viz cadence per capture — 3840 cadences
  (`latent_viz_num_cadences_per_type=960` × 4 signal types) × one capture every
  `latent_viz_step_interval` steps (plus the final step) × epochs. At full scale
  (130 steps/epoch → 13 captures/epoch): ~100 M rows over 20 × 100 epochs, each carrying a
  48-float JSON vector. Still the second heaviest table (behind `signal_characteristics`);
  `latent_viz_step_interval` and `latent_viz_num_cadences_per_type` are the knobs.
- **`system_resources`**: (4 + 2 × n_GPUs) rows/second — ~1.2 M rows/day on a 6-GPU node.
- **`inference_results`**: positives only; at a 0.99 threshold this stays small by
  construction. `inference_cadences`: a handful of rows per cadence.
- **`pipeline_stages`**: one row per timed stage span — tens per training run, a few per
  inference cadence. Negligible.

WAL mode keeps readers (plots, resume queries) unblocked during heavy write phases; the WAL
file is checkpointed back into the main db periodically by SQLite itself.
