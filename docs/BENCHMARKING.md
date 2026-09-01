# Benchmarking

Aetherscan carries always-on stage timing plus a set of offline tools to read it. This
document covers the six pieces: the `stage_timer` instrumentation
([`src/aetherscan/benchmark.py`](../src/aetherscan/benchmark.py)), the `pipeline_stages` DB
table it writes to, the annotated resource plot the monitor renders, the report tool
([`utils/benchmark_report.py`](../utils/benchmark_report.py)), the per-band inference plot
([`utils/perband_report.py`](../utils/perband_report.py)), and the standalone benchmarks
([`benchmarks/`](../benchmarks/)). It closes with the current baseline numbers and how to
read the annotated resource plot.

## TL;DR

- Every pipeline stage is wrapped in `with stage_timer("dotted.name"):`, which queues one
  `pipeline_stages` row (start, end, duration, tag) on the DB writer thread — two
  `time.time()` calls and one queue put, so the timers stay on in production runs.
- Stage names are **hierarchical dot-names** and nest automatically per-thread: a
  `stage_timer` entered while another is active on the same thread records its name relative
  to the parent, so instrumented library code inherits whatever umbrella span the caller
  opened without name plumbing.
- `python utils/benchmark_report.py --save-tag <tag>` prints the stage tree, writes a
  flame-style timeline PNG, and flags likely bottlenecks from `pipeline_stages` joined with
  `system_resources`. The report PNG is also rendered and posted to Slack automatically at
  the end of every `train`/`inference` run (`--no-benchmark-report` opts out).
- `python utils/perband_report.py --save-tag <tag> --catalog <csv>` writes
  `plots/perband_inference_perf_{tag}.png` — per-cadence energy-detection wall-clock split by
  observing band (boxplot + strip) and against catalog frequency — the question the flame
  timeline can't answer: is a whole band systematically slower? It gets the same
  auto-render-and-post-to-Slack treatment at the tail of every streaming-CSV `inference` run,
  under the same opt-out, and skips (never crashes) on the legacy `--test-files` path or when
  the catalog → cadence join guard trips.
- The 1 Hz resource plot overlays the top-level stages as `dimgray` vertical boundary lines
  at each span's right edge on all three (CPU/RAM/GPU) panels — labeled once on the CPU panel
  (angled 30°, left of the line) — via `monitor.annotate_stages`, so a CPU plateau reads as
  "round 3 data generation" at a glance.
- `benchmarks/*.py` time individual pipeline kernels in isolation, in seconds instead of a
  full run. Most are CPU (`bench_normality`, `bench_injection`, `bench_lognorm_downsample`,
  `bench_pfb_vs_spline`, `bench_rf`); `bench_gpu` is the exception — a real-GPU profiler for
  the Beta-VAE (throughput + peak VRAM across a per-replica batch-size sweep), NGC-container-only.
  See [`benchmarks/README.md`](../benchmarks/README.md).

## Stage timers (`benchmark.py`)

Two entry points, both **failure-safe by contract** — benchmarking must never be able to
fail the pipeline, so a missing DB (unit tests, dev scripts) or a serialization hiccup
downgrades to a debug/warning log and the block's result is untouched:

- **`stage_timer(stage, tag=None, metadata=None)`** — a `ContextDecorator` usable as either
  `with stage_timer("train.round_02.data_generation"):` or `@stage_timer("inference.viz")`.
  On `__enter__` it resolves its full name against the thread's active-stage stack and pushes
  itself; on `__exit__` it pops and calls `record_stage()`. On exception the span is **still
  recorded** (metadata gains `status="failed"` plus the error string) and the exception
  propagates — the timer never suppresses it.
- **`record_stage(stage, start_time, end_time, tag=None, metadata=None)`** — writes a span
  measured elsewhere with explicit timestamps and **no nesting resolution** (the name is used
  as-is). This is the seam for work that happens in another process: the round-data producer
  generates data in a separate process that can't touch the DB writer queue (a thread-only
  `queue.Queue`), so it reports its `(start, end)` over its result-message channel and the
  main-process drainer calls `record_stage()` (the `"timing"` branch of the `RoundDataDrainer`
  thread's message handler, [`round_data.py`](../src/aetherscan/round_data.py)).

### Automatic nesting

The active-stage stack is **thread-local** (`_ACTIVE_STAGES`), which matters because training,
prefetch preprocessing, and the producer drainer all time stages concurrently and must not
interleave names. Within one thread, entering `stage_timer("encode")` while
`"inference.infer_cadence_001"` is active records `inference.infer_cadence_001.encode`; with
no active parent it records `encode` as-is. `current_stage()` exposes the innermost active
full name.

`tag` defaults to `config.checkpoint.save_tag` (the run's provenance key). `metadata`, when
given, is a small JSON-serializable dict stored on the row (e.g. `{"source": "producer"}` on
the data-generation span to distinguish the overlapping-producer path from the in-process
one, which the report tool keys its first suggestion off).

### What is instrumented

Every span below is a real `stage_timer`/`record_stage` call site in the current code. Names
are shown at full depth; the leaf component is what the report tool and resource plot label.

| Stage (dotted name) | Where | Notes |
| --- | --- | --- |
| `train.load_backgrounds` | `main.py` | Background plate load before the round loop |
| `train.round_{k:02d}` | `train.py` | Umbrella span for one round |
| `train.round_{k:02d}.data_generation` | `train.py` (in-process) / `round_data.py` (producer) | `metadata.source` = `in-process` or `producer` |
| `train.round_{k:02d}.epochs` | `train.py` | Round-level epoch span (`stage_timer`, nested under the round; per-epoch detail lives in `training_stats`) |
| `train.round_{k:02d}.plots` | `train.py` | Per-round plots |
| `train.round_{k:02d}.checkpoint_save` | `train.py` | `save_models(round_XX)` |
| `train.vae_plots` | `train.py` | End-of-VAE plot stage |
| `train.rf` (`.data_generation`, `.encode`, `.fit`) | `train.py` | RF training and its sub-stages |
| `train.rf_plots` | `train.py` | RF plot stage |
| `train.final_save` | `train.py` | Final model + config save (+ HF upload) |
| `inference.infer` | `inference.py` | Legacy single-array `run_inference` span |
| `inference.load_lognorm` | `main.py` | Legacy `--test-files` load span (top-level; distinct from the per-cadence `.load_lognorm` child below) |
| `inference.preprocess_cadence_{i:03d}` (`.read_ed`, `.dedup`, `.extract`) | `preprocessing.py` | Per-cadence energy detection |
| `inference.infer_cadence_{i:03d}` (`.load_lognorm`, `.encode`, `.rf`, `.db_write`) | `main.py` / `inference.py` | Per-cadence GPU inference + write |
| `inference.reference_cloud` | `main.py` | #282 MC reference-cloud finalization (best-effort, once per successful pass) |
| `inference.viz` | `main.py` | Visualization suite |

## The `pipeline_stages` table (schema v4)

`record_stage()` calls `db.write_pipeline_stage()`, which queues a
`("pipeline_stages", values)` record on the writer thread like every other write. The table
(columns `stage`, `start_time`, `end_time`, `duration_s`, `tag`, `metadata`; index
`(tag, start_time)`) is documented in full in [`DATABASE.md`](DATABASE.md#pipeline_stages-stage-timers-schema-v4).
Retried stages simply append new rows — every attempt keeps its own span, so the report tool
and resource plot see the full history. There is **no `superseded` column**: timing spans are
attempt-agnostic history, like `system_resources`.

## Annotated resource plot (`monitor.annotate_stages`)

When the resource monitor renders its shutdown plot, it optionally overlays this run's
top-level `pipeline_stages` spans as `dimgray` vertical boundary lines at each span's right
edge on every panel (CPU, RAM, GPU — they share an x-axis), labeled once on the CPU panel
(left of the line, angled 30°), turning the utilization curves into a self-explaining timeline.
This is documented alongside the plot in
[`RUNTIME_SERVICES.md`](RUNTIME_SERVICES.md#stage-annotations). The essentials:

- Config knob **`monitor.annotate_stages`** (default `True`) gates it.
- Only **depth ≤ 2** dot-names are drawn (`train.round_03`, not
  `train.round_03.epochs`) — deep spans stay report-tool-only so the panels don't drown in
  divider lines (`select_annotation_spans` / `_ANNOTATION_MAX_DEPTH`).
- The monitor flushes the DB first so spans recorded moments before shutdown (`final_save`,
  `inference.viz`) make it onto the plot.

## The report tool (`utils/benchmark_report.py`)

A dev tool (it may `print` — `utils/` is exempt from the no-`print` rule) that reads the
SQLite file directly with the stdlib `sqlite3` + numpy + matplotlib only (no `aetherscan`
imports), so it also runs against a database fetched from a cluster with
`utils/fetch_run_outputs.sh`:

```bash
# Default db path: {AETHERSCAN_OUTPUT_PATH}/db/aetherscan.db
python utils/benchmark_report.py --save-tag train
# Explicit db (e.g. a copy pulled off the cluster)
python utils/benchmark_report.py --save-tag test --db-path /path/to/aetherscan.db
```

The same report is also generated **automatically at the tail of every `train`/`inference`
run** (`_post_benchmark_report` in [`main.py`](../src/aetherscan/main.py)): the hook flushes
the DB write queue (stage spans land through it asynchronously), loads this tool by file path
(preserving its no-`aetherscan`-imports contract), renders
`{output_path}/plots/benchmark_report_{tag}.png`, and uploads it to Slack with the
suggestions as the image comment. It is gated by `monitor.benchmark_report_enabled`
(`--no-benchmark-report` to opt out) and fully guarded — any failure logs an error and never
fails the run.

It produces three things for one `--save-tag`:

1. **Stage tree** (stdout) — the `pipeline_stages` spans folded into a tree by dot-name
   component (`build_stage_tree`), each node showing its span count `n`, total duration, and
   percentage of its parent. A node's **total** is its own summed span time when it has spans
   (an umbrella span already covers its children's time), else the sum over children (pure
   grouping nodes like a bare `train`). **Self time** is total minus child totals, floored at
   0 — children that run concurrently with a parent (producer data generation overlapping a
   round) can sum past the parent's wall-clock, and the floor keeps that sane.
2. **Report PNG** — `plots/benchmark_report_{tag}.png`: a flame-style timeline with one lane
   per dot-depth (bars colored by top-level family, overlaps = concurrent stages) above a
   "top 10 slowest stages" table ranked by **self time** (where wall-clock actually went,
   rather than umbrella spans).
3. **Suggestions** (stdout) — a rules-driven bottleneck section, sourced by joining
   `pipeline_stages` with `system_resources`.

### Suggestion rules

| Rule | Trigger | Suggestion |
| --- | --- | --- |
| Data-gen dominates a round | `round.data_generation / round.total ≥ 0.30` | In-process (`metadata.source`) → enable `--overlap-data-generation`. Producer path → only fires when the round actually *waited* on generation (generation ended after the round started, 1 s tolerance) → raise `manager.n_processes` / `--data-gen-task-size` or lower `--num-samples-beta-vae`. Fully-overlapped generation is free and produces no suggestion. |
| Training input-bound | mean GPU % during an `epochs` span `< 40%` | Check the `tf.data` feed (batch sizes, host-side preprocessing). |
| ED dominates inference | summed `preprocess_cadence_* / inference wall ≥ 0.60` | Energy detection is the bottleneck; raise ED worker count (`manager.n_processes`) before anything GPU-side. |
| Encode input-bound | mean GPU % across `encode` spans `< 40%` | Encoding is input-bound; consider a larger inference batch (`--per-replica-batch-size`). |
| RAM pressure | peak `ram/system_total ≥ 90%` | Names the stage holding the peak; reduce chunk/sample sizes or disable `keep_round_data`. |

Thresholds are module constants (`DATA_GEN_ROUND_FRACTION`, `GPU_UTIL_INPUT_BOUND`,
`PREPROCESS_WALL_FRACTION`, `RAM_PEAK_WARN`) — tune them there if a rule is too eager for your
hardware.

The RAM-pressure rule is post-run; its pre-run complement is the catalog-derived RAM
preflight (#408) in `main.py`, which warns at inference startup — with a suggested
`--prefetch-depth` — when the pending catalog's per-band worst case exceeds the host
budget (see the streaming-loop section of
[`INFERENCE_PIPELINE.md`](INFERENCE_PIPELINE.md)).

## The per-band inference plot (`utils/perband_report.py`)

A sibling tool (same stdlib `sqlite3` + `csv` + numpy + matplotlib, no `aetherscan` imports) that
answers a question the flame timeline can't: **is a whole observing band or frequency region
systematically slower to preprocess?** It writes
`{output_path}/plots/perband_inference_perf_{tag}.png` — a two-panel, log-y figure of per-cadence
energy-detection preprocessing wall-clock: Panel A is a per-band boxplot + jittered strip (bands
ordered L, S, C, X, each annotated with median/max/n), Panel B scatters the same walls against the
catalog `Frequency` (MHz), colored by band. On the automatic end-of-run fire that `{tag}` is the
machine-scoped *display* tag `{cmd}_{machine}_{datetime}` (matching the other auto-fired report
PNGs); run standalone, it's the plain `--save-tag` you pass. The preprocessing wall per cadence is
the umbrella `inference.preprocess_cadence_<N>` span (its `.read_ed`/`.dedup`/`.extract` children
are excluded),
and the band/frequency come from the run's inference catalog CSV. It fires **automatically at the
tail of every streaming-CSV `inference` run** right after the benchmark report
(`_post_perband_report` in [`main.py`](../src/aetherscan/main.py), same
`monitor.benchmark_report_enabled` gate, same Slack upload, fully guarded — a plot failure never
fails the run), and is also runnable standalone:

```bash
python utils/perband_report.py --save-tag test --catalog /path/to/catalog.csv \
    --db-path /path/to/aetherscan.db
```

**Join caveat.** The umbrella span carries only the 1-based planner index `N`, and no DB table maps
`N` to a band, so the cadence → band/frequency map is reconstructed from the catalog by mirroring
`preprocessing.group_observations_from_csv` (group rows by the default
`cadence_group_by_cols` = `["Target", "Session", "Band", "Cadence ID", "Frequency"]`, keep the
6-obs valid groups in first-appearance order; the i-th is cadence `N=i`). This **assumes the run
used the default group-by columns / expected-obs**. A runtime guard compares the mapped catalog
cadence count to the umbrella-span count and skips the plot (logged warning, never a crash) when
they disagree — so a resumed run (fewer fresh preprocess spans than catalog cadences) or a
non-default grouping degrades to no plot rather than a misleading one. The legacy `--test-files`
path (no catalog CSV) is skipped for the same reason.

## The cascade location probe (`utils/probe_candidate_location.py`)

A standalone diagnostic (container/source path — it drives the real preprocessing helpers and
the encoder, so it needs the TF stack; `--help` alone is stdlib-only) that answers the question
catalog runs deliberately don't persist: **where does a specific (cadence, frequency) location
land at every stage of the scoring cascade?** Given six ordered `--h5-files` (or a `--catalog`
plus exact `--target`/`--band`[/`--session`/`--cadence-id`] group resolution) and one or more
`--frequency-mhz` values — with the standard `--encoder-path`/`--rf-path`/`--config-path` trio —
it reports, per location: the max k² statistic vs the ED threshold ("would energy detection
propose this?", a window-start upper bound; dedup/overlap placement is not replayed), the
production-preprocessed stamp, the pass-1 screening probability, the deterministic RF score, and
the seeded MC mean ± std vs the science threshold, with per-location verdict lines, optional
`--csv`, and optional `--plot-dir` six-panel waterfalls. Built for benchmark comparisons
(scoring published turboSETI event locations; localizing the stage at which a non-recovered
candidate from a prior search drops out) — see [#436](https://github.com/zachtheyek/Aetherscan/issues/436).

Read-only by contract: no DB rows, manifests, or caches are touched (a missing/mismatched PFB
response falls back to spline with a warning instead of being generated). **Documented deltas
from a production catalog run** — the reason probe numbers can differ in the low digits from a
recorded candidate's: each location is encoded *and* MC-scored alone (production batches whole
cadences and draws one MC noise block per cadence, so scores there depend on batch composition;
the probe's per-location seeding — root `--seed`, `--cadence-seed-key`, then the location's
absolute frequency bin — is reproducible under any invocation); screen-rejected locations get a
forced diagnostic MC pass production would never run; edge stamps clamp instead of being
skipped; and `apply_saved_config` never layers the saved run's `reproducibility` section, so
pass the run's `--seed` explicitly to mirror it.

## Standalone benchmarks (`benchmarks/`)

Small standalone scripts that time individual pipeline kernels in isolation, so a change to
one can be measured in seconds instead of via a full run. Most are CPU micro-benchmarks
(`bench_normality`, `bench_injection`, `bench_lognorm_downsample`, `bench_pfb_vs_spline`,
`bench_rf`) that print ops/s; `bench_gpu` is a real-GPU profiler for the Beta-VAE that reports
throughput + peak VRAM across a per-replica batch-size sweep (NGC-container-only). **Not**
collected by pytest (`testpaths = ["tests"]`); run on demand. Each writes a JSON result to
`benchmarks/results/` (gitignored); the two benchmarks that need bulk data on disk
(`bench_input_pipeline.py`, `bench_datagen.py`) default their `--data-dir` to
`{AETHERSCAN_DATA_PATH}/bench/{input,datagen}` — the same data root the pipeline uses — and
accept `--data-dir` to override. See [`benchmarks/README.md`](../benchmarks/README.md) for
the maintained baseline tables and per-flag detail.

```bash
python benchmarks/bench_normality.py            # sliding-window normality test (energy detection)
python benchmarks/bench_injection.py            # setigen signal injection (training data gen)
python benchmarks/bench_lognorm_downsample.py   # per-cadence downsample + log-norm (load path)
python benchmarks/bench_pfb_vs_spline.py        # PFB static equalization vs per-channel spline fit
python benchmarks/bench_rf.py                   # Random Forest stage: latent prep + fit + predict
# Container-only additions (#276/#278 audits + the generation-path A/B; bench_input_pipeline's
# step mode needs GPUs):
# ./utils/run_container.sh python benchmarks/bench_input_pipeline.py --mode step --variant current --num-gpus 5
# ./utils/run_container.sh python benchmarks/bench_latent_gif.py --mode all
# ./utils/run_container.sh python benchmarks/bench_datagen.py --preload-tf --data-dir /datax/scratch/$USER/data/aetherscan/bench/datagen
# On the clusters, through the container:
./utils/run_container.sh python benchmarks/bench_normality.py
# GPU-only, container required — real Beta-VAE profiler on a cluster GPU:
./utils/run_container.sh python benchmarks/bench_gpu.py --mode train --find-max
```

See the [GPU benchmark section of `benchmarks/README.md`](../benchmarks/README.md#gpu-benchmark)
for `bench_gpu.py` flags (`--mode`, `--num-gpus`, `--batch-sizes`, `--accumulation-steps`) and
the maintained per-host baseline tables.

| Script | Kernel it isolates | Pipeline stage it models |
| --- | --- | --- |
| `bench_normality.py` | vectorized `_sliding_normality_k2` vs the historical per-window `scipy.stats.normaltest` loop | ED thresholding (`inference.*.read_ed`) |
| `bench_injection.py` | `data_generation.new_cadence` narrowband injection | `train.round_XX.data_generation` |
| `bench_lognorm_downsample.py` | per-obs `downscale_local_mean` (×8) + per-cadence `log_norm` | stamp downsample + `inference.*.load_lognorm` |
| `bench_pfb_vs_spline.py` | `pfb.equalize_passband` vs `_spline_flatten_bandpass` on one 1M-bin coarse channel | bandpass flattening inside ED |
| `bench_rf.py` | `RandomForestClassifier.fit` + `predict_proba` and the `prepare_latent_features` reshape (sklearn, CPU) | Second-stage RF training + inference (`train.train_random_forest`) |
| `bench_gpu.py` | Beta-VAE training step (`compute_total_loss` + gradients + clipped Adam) and encoder forward on one or more GPUs | VAE training (`train.round_XX`) and encoder inference — **GPU-only, container required** |
| `bench_input_pipeline.py` | The real memmap → tf.data → distribute → train-step input path, legacy vs current builder (current = the zero-copy pure-TF gather AND the graph-side accumulated step, mirroring train.py as checked out), with a `--gil-load` contention knob and `--profile` TF-profiler hook | The training input pipeline `bench_gpu.py` deliberately excludes — the #276 audit harness. Its profiler traces are what established that the GPUs sit idle >90% of the wall clock while doing exactly the kernel work `bench_gpu` predicts; the follow-up section of `benchmarks/README.md` carries the before/after ladder for the Python-free rewrite |
| `bench_latent_gif.py` | Latent-GIF stage decomposition (UMAP fit / transform / render / assemble) with output-equality gates on every candidate optimization | The `vae_plots` latent-GIF tail — the #278 audit harness (numbers in `benchmarks/README.md`) |
| `bench_datagen.py` | Seeded round generation through the real pooled `generate_round_to_memmap` path (shared-memory plates, per-task seed derivation, batched memmap tasks), sha256-ing every output array as the byte-compatibility gate for generation-path changes; `--preload-tf` mirrors the producer workers' TF import graph | The producer wall (`train.round_XX.data_generation`) that `bench_injection.py`'s single-process kernel can't reach — the generation-path A/B harness (ladder + mechanisms in `benchmarks/README.md`) |
| `bench_db_index_shapes.py` | The schema-v7 `training_stats` / `latent_snapshots` index reshapes — old `(tag, timestamp, ...)` filter index vs the equality-first replacement — timed across every production query shape plus each table's real write pattern, no `ANALYZE` (matching `db.py`) | DB read/write cost behind the loss-curve / posterior-collapse / latent-GIF-frame fetches — the schema-v7 DB-index audit (verdict baked into the schema: latent reshape shipped, `training_stats` rejected; numbers in the docstring) |
| `bench_injection_index.py` | The schema-v7 secondary `injection_stats` index (`idx_injection_stats_by_stat`) vs the original filter index — bulk-insert write cost and the end-of-run plot query shape, benched with and without `ANALYZE` | DB read/write cost behind the injection-stats end-of-run plot pass — the schema-v7 DB-index audit (companion to `bench_db_index_shapes.py`) |

Only the four CPU micro-benchmarks (`bench_normality`, `bench_injection`,
`bench_lognorm_downsample`, `bench_pfb_vs_spline`) are single-process; the pipeline
parallelizes each of those across `manager.n_processes` workers, so whole-stage throughput
scales roughly with core count. `bench_rf` uses sklearn `n_jobs=-1`, matching production.
`bench_gpu` reports aggregate throughput across `--num-gpus` MirroredStrategy replicas plus
per-GPU peak VRAM.

## Baseline numbers

Micro-benchmark speedups below are best-of-3 at production shapes on **blpc3 (EPYC 7313,
32 cores, NGC 25.02 container)** — the maintained table (plus a MacBook Air M3 column) is in
[`benchmarks/README.md`](../benchmarks/README.md#baseline-numbers). Higher is better; expect
~10% run-to-run noise.

| Kernel | Vectorized / new | Baseline / old | Speedup |
| --- | --- | --- | --- |
| Sliding-window normality (`bench_normality`) | 83,667 windows/s | 679 windows/s (scipy loop) | **123×** |
| Bandpass flattening (`bench_pfb_vs_spline`) | 59.6 channels/s (PFB) | 5.0 channels/s (spline fit) | **11.9×** |

End-to-end numbers, measured with the stage timers and resource plot on real runs:

- **Inference (blpc3, 5× RTX PRO 6000, subset CSV run).** ~6.2× faster end-to-end after the
  energy-detection overhaul. Energy detection fell from ~45 min/cadence to ~2 min/cadence
  (the vectorized D'Agostino–Pearson test above is the main lever), and per-cadence stamp
  storage dropped ~8× via downsample-at-extraction (the frequency axis is reduced ×8 in the
  extraction worker instead of storing full-width stamps — see
  [`PREPROCESSING.md`](PREPROCESSING.md)).
- **Bandpass method (blpc3, subset CSV run).** PFB and spline flattening come out at roughly
  parity in **whole-pipeline** wall-clock and produce near-identical hit sets, even though the
  micro-benchmark shows PFB is 11.9× cheaper on the isolated per-channel kernel. That is the
  expected outcome of the report tool's finding that, once the normality test is vectorized,
  **stamp extraction — not bandpass flattening or thresholding — is the inference
  bottleneck**; shaving an already-small stage barely moves the total. PFB stays the default
  for its determinism (no per-channel fit); spline remains available via
  `--bandpass-method spline`.
- **Training (multi-GPU Ampere node, 6× RTX A4000 smoke).** The disk-backed round data +
  producer redesign holds steady-state RSS at ~26% of the 503 GB node, versus the old
  in-RAM design that saw-toothed to ~99% and SIGKILL-ed the run in round 2. Data-generation
  wall-clock in rounds 2+ matches round 1 (the GIL-contention regression that made later
  rounds slower is gone), and round *k+1*'s data is generated by the producer process while
  round *k* trains (visible on the annotated resource plot as the CPU `data_generation`
  plateau of round *k+1* starting *before* the boundary line that closes round *k*'s `epochs`
  stage on the GPU panel).

## Reading the annotated resource plot

The shutdown resource plot (`plots/resource_utilization_{tag}.png`,
[`RUNTIME_SERVICES.md`](RUNTIME_SERVICES.md#the-shutdown-plot)) has three time-aligned panels
(CPU, RAM, GPU) with the x-axis in minutes since monitor start. With `annotate_stages` on,
`dimgray` boundary lines drop through all three panels at each top-level stage's end time
(labeled once on the CPU panel, left of the line, angled 30°). Read a stage as the region
ending at its labeled line:

- **CPU plateau ending at a `data_generation` line** — worker pool was saturating cores
  generating a round. If the CPU plateau of round *k+1*'s `data_generation` starts *before*
  round *k*'s `epochs` line, the producer is overlapping correctly (good); if `epochs` on the
  GPU panel starts only *after* the round's `data_generation` line, that round waited on
  generation — the report tool's data-gen rule will flag it.
- **Low GPU utilization in the region ending at an `epochs` line** — input-bound training;
  cross-check the report tool's GPU-util rule.
- **Alternating CPU/GPU activity across successive `preprocess_cadence_*` / `infer_cadence_*`
  lines** — per-cadence streaming inference, CPU-heavy energy detection overlapping the
  previous cadence's GPU encode.
- **RAM approaching the top of its panel** — memory pressure; the report tool's RAM rule
  names the stage holding the peak. The training redesign should keep this well clear of the
  node ceiling.

Pair the plot (which region) with `benchmark_report.py` (how long, and the suggestion) to go
from "this run felt slow" to a specific stage and knob.
