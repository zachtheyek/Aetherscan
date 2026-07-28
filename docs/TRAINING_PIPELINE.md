# Training Pipeline

This document covers the `train` command end to end: the curriculum round lifecycle, the
disk-backed round-data pipeline with background generation, the distributed training loop,
checkpointing, the run-state manifest that makes retries seamless, and every diagnostic plot
the pipeline produces (and what to look for in each). The orchestration lives in
[`src/aetherscan/train.py`](../src/aetherscan/train.py) with two supporting modules:
[`round_data.py`](../src/aetherscan/round_data.py) (memmap datasets + producer process) and
[`run_state.py`](../src/aetherscan/run_state.py) (persisted run manifest). For the model math
see [`MODELS.md`](MODELS.md); for signal injection internals see
[`PREPROCESSING.md`](PREPROCESSING.md).

## TL;DR

```
train_command() (main.py)
└── retry loop (training.max_retries, training.retry_delay)
    └── run_training_pipeline() → _execute_training_stages():
        1. vae_rounds   — for each round: get data → epochs → per-round plots → checkpoint
        2. vae_plots    — final loss/stability/collapse/injection plots, latent GIF, latent traversal
        3. rf_train     — generate RF data → encode latents → 8-variant RF sweep → calibrate →
                          persist winner+artifacts
        4. rf_plots     — the ten RF diagnostics
        5. final_save   — final models + config_{tag}.json
        6. hf_upload    — publish artifacts + model card to the HuggingFace Hub (opt-in)
```

Stages 1/3/5 are critical (failures raise → retry); 2/4/6 are non-critical (failures are
recorded in the manifest and retried on the next run, but never cost a data regeneration).
Stage 6 (`hf_upload`) runs only when `config.hf.upload_after_training` is set — a failed
upload is recorded but never fails the run, since the weights are already safe locally.
Every stage is skipped if the persisted manifest (`run_state_{tag}.json`) already records it
as done, so the in-process retry loop resumes exactly where the last attempt died. Resuming
across a full process relaunch is explicit: pass `--load-tag {full-tag}` (a bare re-run mints a
fresh datetime tag and starts a new run).

## Round lifecycle

`TrainingPipeline.train_beta_vae()` runs `training.num_training_rounds` rounds (default 20) of
`training.epochs_per_round` epochs (default 100). Each round (`train_round()`):

1. **Obtain data** — reuse a validated on-disk round dataset if one exists, else wait on the
   background producer (or generate in-process when overlap is disabled).
2. **Queue the next round** — generation of round *k+1* is requested immediately, so it runs
   in the producer process while round *k* trains.
3. **Build datasets** — `prepare_distributed_train_dataset()` over the round's memmaps
   (stratified 80/20 train/val split on the labels array).
4. **Prepare the latent-viz batch** (first round only) — 960 held-out val cadences per signal
   type, persisted across rounds so latent-space snapshots aren't confounded by curriculum
   distribution shift.
5. **Epoch loop** — `_train_epoch()` + `_validate_epoch()`, ~23 `training_stats` rows per
   epoch (losses, gradient norms, LR, durations, SNR range) plus `latent_dim` per-dimension
   KL rows (`kl_dim_NN`), adaptive LR update. At round end `check_posterior_collapse()`
   WARNs loudly (never fails) when latent dims are going dark — see
   [the playbook](#posterior_collapse_tagpng--the-collapse-playbook).
6. **Per-round plots** — loss curves, training stability, injection stats (tagged
   `round_XX`, saved under `plots/training/{save_tag}/checkpoints/`; the injection queries
   are scoped to this round only), plus the latent traversal when
   `--latent-traversal-every-round` is set. Each per-round plot is best-effort: a failure
   (including a skipped plot whose DB flush timed out) logs an error and the round
   continues — a retry here would cost a full data regeneration.
7. **Checkpoint** — `save_models(tag="round_XX", dir="checkpoints")`, then the round is
   recorded in the run manifest (`completed_rounds`).
8. **Cleanup** — holders cleared, `tf.keras.backend.clear_session()`, pools reset, and the
   round's data directory deleted (unless `--keep-round-data`).

At the start of each round the learning rate resets to `training.base_learning_rate` and the
adaptive-LR state (`best_val_loss`, `patience_counter`) is dropped — each curriculum stage is
a fresh optimization problem. Adam moments are curriculum-stage-local by the same reasoning.

### Curriculum schedules

`_calculate_curriculum_snr(round_idx)` narrows the injection SNR range from
`initial_snr_range` (40) down to `final_snr_range` (10) above `snr_base` (10) across the
rounds — early rounds see bright, easy signals; late rounds see predominantly faint ones.
Three schedules (`--curriculum-schedule`):

| Schedule | Behavior | Knobs |
| --- | --- | --- |
| `linear` | Uniform narrowing: `range = initial - progress·(initial - final)` with `progress = round_idx / (total - 1)`. | — |
| `exponential` (default) | Fast-then-slow decay, normalized so progress 0 and 1 hit the exact endpoints: `range = final + (initial - final) · (e^{r·p} - e^{r}) / (1 - e^{r})`. More negative `r` = fewer easy rounds. | `exponential_decay_rate` (must be < 0; default −3.0) |
| `step` | `initial_snr_range` for the first `step_easy_rounds`, then `final_snr_range` for `step_hard_rounds`. The two must sum to `num_training_rounds`. | `step_easy_rounds`, `step_hard_rounds` |

Each injected signal draws `snr = snr_base + U(0,1) · snr_range`, so the *floor* stays fixed
while the ceiling tightens. The per-round floor/ceiling is written to both `training_stats`
and `injection_stats` and shows up as background shading on the training plots.

## Round data: memmaps + background producer

A full-scale round is three arrays (`main`, `true`, `false`) of shape
`(499200, 6, 16, 512)` — float32 at the `training.round_array_dtype` default ≈ 98 GB each,
~294 GB per round (the A/B-gated `"float16"` setting halves all of that; see the
[performance-engineering section](#performance-engineering-the-276-follow-up-july-2026)).
Holding that in RAM is what used to OOM-kill 503 GB training nodes; instead each round lives
on disk under `{round_data_dir}/{save_tag}/round_{k:02d}/` (default root
`{data_path}/training/round_data`):

```
round_02/
├── main.npy  true.npy  false.npy        # (n, 6, 16, 512) round_array_dtype memmaps
├── main_lognorm.npy  true_lognorm.npy  false_lognorm.npy   # (n, 6, 2) log-norm params
├── labels.npy                            # (n,) signal-type strings
└── round_02.done                         # atomic JSON manifest
```

Key properties (all in [`round_data.py`](../src/aetherscan/round_data.py) /
[`data_generation.py`](../src/aetherscan/data_generation.py)):

- **Workers write straight into the memmaps.** `generate_round_to_memmap()` dispatches
  batched tasks (`training.data_gen_task_size` cadences each, default 64) covering disjoint
  row ranges; each worker opens the `.npy` in `r+` mode, writes its rows in place, and
  returns only small stats dicts. No per-sample IPC pickling, one `pool.map` barrier per
  chunk.
- **The `.done` manifest is the completion contract.** Written atomically
  (`.tmp` → `os.replace`) only after every chunk finishes and every array is msync'd; it
  records shapes, per-array dtypes (plus the requested `array_dtype`), SNR params, and cheap
  sampled checksums. `validate_done_manifest()` re-checks all of it — a directory without a
  valid manifest is garbage and gets regenerated, and every reuse/resume path (producer
  short-circuit, round reuse, RF-dataset reuse, `prepare_round_data_dir`) also gates on the
  dtype, so a round generated under a different `round_array_dtype` is regenerated rather than
  silently changing input numerics mid-run (manifests predating the dtype keys validate as
  float32).
- **Page-cache-backed reads.** Training opens the arrays with `np.load(mmap_mode="r")`;
  after the first epoch the OS caches the round in otherwise-free RAM, so steady-state reads
  run at RAM speed — but under memory pressure the kernel evicts pages instead of OOM-killing
  the process.
- **Disk budget.** ~295 GB per round at defaults, ~2 rounds on disk at once with overlap
  (~590 GB peak). `cli.py:collect_validation_errors` checks free space at startup
  (`_estimate_round_data_nbytes`: 2.2× one round with overlap, 1.1× without) and hard-fails
  with the computed numbers. Round *k*'s directory is deleted as soon as round *k* finishes
  training (`--keep-round-data` retains it for debugging).

> [!TIP]
> **For official tagged training releases, pass `--keep-round-data`.** By default each round's
> memmaps are deleted the moment that round finishes training (delete-as-you-go keeps the disk
> footprint at ~590 GB). `--keep-round-data` retains every round's exact on-disk dataset (plus the
> RF training set) under `{data_path}/training/round_data/{save_tag}/{round_XX,rf}/`, so a release
> model's training data is reproducible/inspectable after the fact — at the cost of holding the full
> run on disk (~295 GB × num_training_rounds, e.g. ~6 TB for a 20-round run). Nothing in the pipeline
> *reads* an earlier round once it has trained, so this flag is purely for post-hoc retention.

### The producer process

`RoundDataProducer` generates round *k+1* while round *k* trains, and isolates generation from
the trainer's GIL (TF's prefetch/callback threads used to make round-2+ generation far slower
than round 1's):

- A **spawn**-started `multiprocessing.Process` (never fork — the TF/NCCL/CUDA-laden parent
  holds locks a forked child can inherit mid-acquisition and deadlock on). The producer owns
  a private fork-started worker pool whose workers attach to the background-plate shared
  memory created by the main process.
- Protocol over two spawn-context queues: main sends `("generate", round_idx, snr_base,
  snr_range)` / `("shutdown",)`; the producer streams back `stats` (per class-segment
  injection statistics), `progress`, and terminal `done`/`error` messages.
- **DB writes stay in the main process**: a drainer thread consumes the `stats` messages and
  calls `data_generation.write_segment_stats()` — the DB writer queue is a thread
  `queue.Queue`, not process-safe. The drainer runs while the GPUs compute, so injection-stat
  writes are off the training critical path.
- The producer logs into its own spawn-context queue, relayed into the main process's
  handlers by a `QueueListener`; `CUDA_VISIBLE_DEVICES` is blanked during the spawn so the
  child's TF import can never initialize CUDA.
- Registered with the ResourceManager (`ManagedProcess`), so cleanup escalates
  terminate → join → kill.
- **Parent-death watch.** The request loop's `get(timeout=5)` doubles as a
  heartbeat: each timeout re-checks `os.getppid()`, and if the parent PID has
  changed (reparented to init/systemd after an ungraceful main-process death),
  the producer terminates its pool and exits — no `shutdown_ack` is sent. On
  Linux, `prctl(PR_SET_PDEATHSIG, SIGTERM)` provides immediate coverage for
  mid-generation parent death via the existing SIGTERM handler.
- **Pidfile (`producer.pid`).** `start()` writes
  `{round_data_root}/{tag}/producer.pid`; `shutdown()` removes it on graceful
  exit. The pidfile enables post-mortem discovery by `kill_pipeline.sh` and
  `_reap_stale_producer()`.
- **Restart-race guard.** `prepare_round_data_dir()` calls
  `_reap_stale_producer()` before any `rmtree`, terminating a live orphan
  recorded in the pidfile (with a PID-reuse guard via `create_time()` vs
  pidfile mtime) so a new run cannot race an orphan's live writes.

`--no-overlap-data-generation` falls back to sequential in-process generation (the debugging
path, also used automatically when `manager.n_processes == 1`).

## Distributed training

### Datasets

`prepare_distributed_train_dataset()` builds infinite `tf.data` datasets from cheap
**index generators** followed by a parallel, deterministic `.map()` of **pure `tf.gather`
ops** over zero-copy tensor views of the round memmaps (`_as_cpu_tensor`: dlpack export of
the `mmap_mode="c"` arrays — no allocation, no copy, and gathers stream through the OS page
cache exactly as before, but inside tf.data's C++ threadpool with no Python and no GIL in
the hot path). This is the durable #276 fix: the historical single generator starved the
GPUs on per-sample Python crossings, and the interim `tf.numpy_function` gather re-entered
the interpreter from every map worker, so its benefit inverted whenever another Python
thread (DB writer, producer relay) competed for the GIL — see the follow-up section of
[`benchmarks/README.md`](../benchmarks/README.md) for the measured ladder.

Each index-stream element is one **per-replica batch** (replica *r* of global batch *g* is
element `g·R + r`), distributed via `strategy.distribute_datasets_from_function`, which
hands consecutive elements to consecutive replicas — reproducing exactly the contiguous
split `experimental_distribute_dataset` used to apply to whole global batches, while
skipping the split op and prefetching each replica's batches straight to its GPU
(`InputOptions(experimental_fetch_to_device=True)`), so host→device copies overlap compute.
`deterministic=True` preserves the exact batch order of the index stream, so parallelism
changes throughput only — epoch composition and ordering (and therefore the seeding
contract below) are byte-identical to the previous implementations. Randomness lives at the
epoch level (train indices reshuffled per pass); within a batch, indices are sorted for
memmap read locality (the model is order-invariant within a batch).

The train/val split is **stratified** over the four signal types (generation lays labels out
contiguously per chunk, so a positional split would skew val), then trimmed to exact multiples
of `effective_batch_size` (train) and the global val batch size. With `shuffle=False`
(used by RF training) the yield order is pinned to the returned `train_indices`/`val_indices`
— the alignment contract that lets encoded latents be matched back to labels.

### Gradient accumulation

Each training step accumulates `accumulation_steps = effective_batch_size /
(per_replica_batch_size × num_replicas)` micro-batch gradients before applying, giving an
effective batch of 7680 at defaults (chosen so it divides evenly on 4-, 5-, or 6-GPU hosts)
regardless of per-GPU memory. **Note:** 7680 is ~2.5× the previous 3072, so there are ~2.5×
fewer optimizer updates per epoch; LR-schedule behavior calibrated to the old cadence may
differ. On a fixed 4- or 6-GPU host you can pass `--effective-batch-size 3072` to restore the
old cadence (it stays valid there).

The whole optimizer step runs inside **one `tf.function`**
(`_train_epoch()` → `_build_accumulated_train_step()`): the K micro-batches accumulate into
per-replica `ON_READ` accumulator variables inside a `tf.range` loop (sequential graph loop —
bounded activation memory, which is what keeps peak VRAM at ~8.4 GB/GPU instead of the
~23 GB/GPU an unrolled K=12 loop measures, and what lets the 16 GB A4000 release host run it),
then ONE cross-replica reduction per variable, an in-graph NaN/Inf guard (a bad step skips the
apply so weights are never corrupted, then the Python caller raises exactly as before), a
global-norm clip at 1.0 with the pre-clip norm recorded per step (the `clipping_rate`
statistic), and one apply. The interpreter is re-entered once per optimizer step — the
previous loop re-entered per micro-batch and additionally ran one `strategy.reduce` per
variable per micro-batch plus per-variable eager NaN checks, which the #276 profiler traces
showed dominated the host-side wall. Validation is likewise one traced graph call per epoch
(`_build_val_loop`).

The divisibility preconditions (`effective_batch_size % (per_replica × replicas) == 0`, sample
counts divisible by batch sizes, etc.) are validated up front by
`cli.py:collect_validation_errors` — see [`CONFIG_AND_CLI.md`](CONFIG_AND_CLI.md) for the
cross-replica constraint system and the fix proposer.

### Performance engineering (the #276 follow-up, July 2026)

The 2026-07 release-rehearsal runs showed the GPUs idle >90% of every epoch (profiler-verified;
`nvidia-smi`'s sampled "utilization" reads higher — see `benchmarks/parse_xplane_occupancy.py`).
Three successive host-side walls were identified and removed; the full measured evidence chain
lives in the follow-up section of [`benchmarks/README.md`](../benchmarks/README.md), and every
number below is from blpc3 (5× RTX PRO 6000, NGC 25.02).

1. **Python in the gather** — first a single generator thread doing the memmap gathers, then
   (post-#283) a `tf.numpy_function` gather whose map workers re-entered the interpreter and
   convoyed on the GIL with the DB writer / producer-relay threads (+38% idle turned into
   −11% under two competing Python threads). *Fix:* pure `tf.gather` over zero-copy dlpack
   tensor views of the `mmap_mode="c"` round arrays; index generators yield one epoch table
   per epoch, streamed by `.unbatch()`. Python touches the input path once per epoch.
2. **Python in the training loop** — per micro-batch: one interpreter re-entry, one
   `strategy.reduce` per variable (an all-reduce launch per variable per micro-batch), eager
   gradient sums, and per-variable NaN checks that forced device syncs. *Fix:* the
   accumulated-step graph described above (one re-entry and one reduction per variable per
   optimizer step).
3. **Per-epoch iterator lifecycle** — `iter(distributed_dataset)` costs ~9 s per call on a
   5-GPU MirroredStrategy (measured; stable across calls, it is construction cost, not
   tracing — trace counts were probed and stay flat), and the old loop paid it twice per
   epoch (train + val) while also discarding whatever the pipeline had prefetched across the
   boundary. *Fix:* iterators are created once per round in `train_round` and passed to every
   epoch; the infinite datasets make epoch boundaries purely step-counted, so batch
   composition is unchanged. A fresh-eyes audit then found the SAME disease in the
   latent-snapshot path (a fresh viz-dataset iterator plus a gc pass per capture: ~1.2 s of
   the ~1.5 s per-snapshot cost, at up to 6 captures per epoch at full scale) — fixed the
   same way (round-scoped viz iterator, `_distributed_encode(iterator=...)`), alongside
   batching the snapshot's ~3,840 per-row DB writes into one bulk call.

Design decisions worth knowing before touching this code:

- **`tf.range`, not an unrolled micro-batch loop.** Unrolling K=12 overlaps enough in-flight
  activations to peak at 23–26 GB/GPU — it does not fit the 16 GB A4000 release host. The
  `tf.while_loop` form peaks at ~8.4 GB and still pipelines (autograph's default
  `parallel_iterations=10`); at the step level it costs nothing (a fresh-eyes audit measured
  the production tf.range step FASTER than the unrolled single-GPU benchmark, 3,015 vs
  2,907 cad/s — the shipped configuration sits at ~85% of the measured 15,112 cad/s
  zero-communication bound, with the residual split ~7.4% MirroredStrategy lockstep +
  ~7.6% input h2d interference, both probed for further levers with null results; see
  `benchmarks/README.md`, "Corrected ceiling decomposition").
- **Zero-copy has an alignment gate.** TF CHECK-aborts (uncatchable SIGABRT, not an
  exception) on CPU tensors whose buffers are under 64-byte aligned; `_as_cpu_tensor` only
  dlpack-wraps 64-aligned writable arrays (`.npy` memmaps always qualify) and falls back to a
  copying `tf.convert_to_tensor` otherwise.
- **Avoid replica-context collective ops on blpc3.** `CollectiveReduceV2` (e.g. a
  replica-context `all_reduce`, or tf_keras' own aggregation when `apply_gradients` runs in
  replica context) aborts with an NCCL "unhandled cuda error" on the 5-GPU Blackwell host;
  the pipeline keeps all cross-replica communication on the `strategy.reduce` /
  cross-replica-apply path, which is proven there.
- **Same-seed results are deterministic but not bit-compatible** with the pre-follow-up
  implementation (gather backend and float summation order changed). The #49 contract —
  same seed ⇒ same split, same per-epoch batch order, byte-identical reruns — is preserved
  and pinned by `test_train_datasets.py` / `test_train_accumulation.py` /
  `test_train_distribution.py` (the last one pins the cross-replica accumulator semantics at
  2 CPU replicas, where a silent gradient mis-scale would otherwise hide).

End-to-end effect (5 GPUs, accumulation 12): optimizer-step throughput 1,453 → 12,317 cad/s
(8.5×), GPU occupancy 8.3% → 73.7%, the step loop is insensitive to the production DB
writer at full flood (−0.3%), and real-run training epochs went 48.5 → 5.8 s quiet /
56.5 → 10.9 s with generation overlapped (8.4× / 5.2×), validation 16.7 → 1.7 s. The full
A/B and the measured ladder live in `benchmarks/README.md`.

**The second pass (late July 2026)** attacked the walls left standing once training epochs got
fast, with the same byte-identity discipline:

1. **Data generation** — with training 8.5× faster, generation became the run's dominant wall
   (a killed production-scale blpc3 run measured ~6.2 h/round, GPUs at 0%, 32 workers at
   ~94.5% CPU / ~97% user-mode). Three producer-path fixes, all byte-identical and gated by
   `benchmarks/bench_datagen.py`'s per-array sha256 A/B: the per-injection `gc.collect()` in
   `new_cadence` is gone (~2.5M calls per round in TF-laden pool workers; measured ~23 ms per
   call against ~4.5 ms for the entire rest of the function), `create_true_double` draws both
   signals' geometry first and materializes setigen injections only for the accepted pair (at
   the measured p≈0.42 acceptance, ~41% of injections were computed and discarded), and the
   per-task memmap msync is one flush per array before the `.done` manifest (same durability
   contract). Measured on blpc3 (8192 samples × 3 arrays, 32 workers, TF preloaded):
   282.0 → 13.1 s (**21.5×**), checksums identical — projecting a production round from
   ~6.2 h to ~20 min on the same 32-core host (a projection, not a measured production
   round). Ladder + mechanisms in `benchmarks/README.md`.
2. **Injection-plot query windows** — `plot_injection_stats` issues ~165 round-scoped queries
   per call, but `idx_injection_stats_filter` leads with `(tag, timestamp)` and
   `round_number` is not in it, so the run-wide `[run-start, now]` window re-scanned the
   tag's entire row history per query (measured 10.5× slower at 12M rows).
   `Database.query_injection_stat_time_span` — one whole-partition MIN/MAX aggregate, a
   deliberate superset bound — now tightens the window to the plotted rounds' actual row
   span (±1 s); intersection can only narrow scans, never change a result set (see
   [`DATABASE.md`](DATABASE.md#query-api)).
3. **The UMAP GIF sweep** — the 24-combo sweep in `plot_latent_space_gif` ran strictly
   serially at ~95% single-core (~1.7–1.9 h per run) even after #278 parallelized frame
   rendering; whole combos now run across forkserver workers. First parallel measurement
   (reduced 60-frame shape, 24 workers on blpc3): ~93 min — 24 concurrent single-threaded
   UMAP fits/transforms contend for memory bandwidth, so the wall is well below the naive
   per-combo × 24 serial bound but far from core-count scaling; the production-scale number
   and the fit-vs-transform-vs-JIT attribution (plus whether a smaller worker cap beats 24
   under contention) are open follow-ups. Byte-identity pinned by test. Details in
   [the latent-GIF plot section](#latent_space_obscadence_nnn_mdm_taggif) below.
4. **GPU thread mode** — `gpu.gpu_thread_mode` (default `"gpu_private"`, also
   `"global"`/`"gpu_shared"`) and `gpu.gpu_thread_count` (default 2) set
   `TF_GPU_THREAD_MODE`/`TF_GPU_THREAD_COUNT` in `setup_gpu_strategy` before the GPU runtime
   initializes: dedicated per-GPU kernel-launch threads that tf.data host work cannot steal,
   aimed at the measured ~7.6% input h2d/scheduling interference (the one residual with no
   pinned null result). The flipped default is provisional pending the bench A/B on blpc3
   (validation in flight) — flip back to `"global"` if the step ladder regresses.

Two further levers landed **default-off behind an A/B gate**, because flipping either changes
numerics. The gate is a val-metric A/B (3 seeds × 2 arms): val AUC within max(2σ, 0.002),
losses and recalls within 2σ, the same active-dimension count, and zero NaN-guard trips.
Until it passes on the target host, both flags stay at their defaults — which reproduce the
pre-flag pipeline byte-for-byte (neither makes so much as a policy call when off):

- **`training.round_array_dtype`** (`"float32"` default). `"float16"` halves the ~294.5 GB
  round footprint to ~147 GB — and with it the gather volume and the page-cache working set,
  the lever that keeps overlapped epochs at page-cache speed once two rounds no longer fit in
  RAM at full scale. Quantization is ≤ 2⁻¹² on the [0, 1] log-normed inputs; the gather map's
  existing `tf.cast` becomes the host-side upcast (the viz fancy-index path upcasts
  identically), so the training graph and loss math see float32 unchanged either way. Labels
  and lognorm sidecars stay float32; `.done` manifests record the dtypes and every
  reuse/resume path gates on them (legacy manifests read as float32).
- **`beta_vae.mixed_precision`** (`False` default). `True` sets the keras `mixed_bfloat16`
  global policy before the model build, with fp32 islands pinned in `models/vae.py` — the
  z_mean/z_log_var heads, `Sampling`, and the decoder's sigmoid output — so everything
  reaching `compute_total_loss` stays fp32. bf16 needs no loss scaling; variables, Adam
  state, and the step's gradient/loss accumulators stay fp32. Saved `.keras` files carry the
  per-layer dtypes, so inference follows automatically. `bench_gpu.py --mixed-precision` is
  the Phase-0 throughput A/B.

Deferred with rationale (recorded in `benchmarks/README.md` so they are not blindly retried):
RF-dataset pre-generation on the producer, a direct-numpy injection bundle, SHAP-stage
overlap, pool thread-pinning, and fused moments — the 21.5× generation result collapsed their
absolute value.

### Adaptive learning rate

`_update_learning_rate()` tracks validation total loss: if it fails to improve by
`min_pct_improvement` (0.1 %) for `patience_threshold` (3) consecutive epochs, the LR is
scaled by `1 − reduction_factor` (×0.8), floored at `min_learning_rate` (1e-6). The LR resets
to `base_learning_rate` (1e-3) at each round start. Rule of thumb from the docstring: the LR
can only bottom out within a round if
`base_learning_rate · (1 − reduction_factor)^(epochs_per_round / patience_threshold)`
reaches `min_learning_rate`.

## Reproducibility

Runs are reproducible **out of the box**: one root seed — `config.reproducibility.seed`,
default **11** (#279 flipped this from the historical `None`) — drives every random stream in
*both* pipelines. Setting it to `None` restores the OS-entropy behavior (non-reproducible,
warned once per process).

- **`--seed`** (mirrors `config.reproducibility.seed`; `int | None`, must be `>= 0`; on
  **both** subcommands). Every consumer derives an independent stream from it: synthetic data
  generation (per-round worker-task seeds derived from `(seed, round_number)`, identical on the
  background-producer and the sequential in-process paths), the dataset split/trim/per-epoch
  train shuffles (`prepare_distributed_train_dataset`, one stream per round),
  latent-visualization batch selection and padding, plot subsampling (injection-bias figures,
  SHAP sample selection, RF learning-curve/decision-boundary points), every UMAP/KMeans
  `random_state`, the Random Forest (below), and the TensorFlow global RNG (fixing
  `HeNormal`/`GlorotNormal` weight init and the VAE `Sampling` layer) via
  `seeding.seed_tensorflow` — called by **both** pipeline constructors so training and
  inference can't drift, and again at each round boundary (sub-keyed by round number) and at
  the RF stage, so a resumed run reproduces an uninterrupted one (a single `__init__`-time
  `set_seed` is not resume-safe: skipping rounds shifts the stream position). Inference
  additionally re-seeds per cadence — see [`INFERENCE_PIPELINE.md`](INFERENCE_PIPELINE.md).

  > **Why `seed_tensorflow` also seeds Python's `random`.** `tf.random.set_seed()` alone does
  > **not** pin weight initialization on this stack: tf_keras's initializers build a
  > `backend.RandomGenerator(seed=None, rng_type="stateless")` whose `_create_seed()` falls
  > back to `random.randint(...)` on Python's *global* `random` module. Before this was fixed,
  > every VAE initialized from OS entropy regardless of `--seed`. The `Sampling` layer calls
  > `tf.random.normal` directly and was always covered — only initialization was affected.
  > **Do not replace this with `tf_keras.utils.set_random_seed()`**, the canonical Keras API:
  > it populates the thread-local `_SEED_GENERATOR`, after which `_create_seed()` calls
  > `randint(1, 1e9)` with a *float* bound that Python 3.12 rejects, and every subsequent
  > initializer raises `TypeError`. Verified on the NGC 2.17 image; guarded by a regression
  > test in `tests/unit/test_models.py`.
- **The Random Forest seed derives from the root** (`STREAM_RF`). `config.rf.seed` is now an
  explicit-override-only field (default `None`); the **deprecated** `--rf-seed` alias still
  sets it for existing scripts but logs a deprecation warning.
  `Config.resolved_rf_seed()` reports the value actually used. The RF *dataset* borrows the
  round-`0` `STREAM_DATASET` key while curriculum (beta-VAE) rounds are 1-based, so their
  streams never collide.
- **`--tf-deterministic-ops`** (`config.reproducibility.tf_deterministic_ops`, off by default;
  on both subcommands) forces deterministic TF/cuDNN kernels via
  `tf.config.experimental.enable_op_determinism()`. It costs some speed and is only meaningful
  alongside a seed — enabling it without one logs a warning and buys nothing.
- **Approximate vs. bit-exact.** Seeding alone gives *approximate* run-to-run reproducibility;
  *bit-exact* GPU reproducibility additionally requires `--tf-deterministic-ops` plus identical
  hardware and software.

Stream derivation lives in [`seeding.py`](../src/aetherscan/seeding.py): `derive_rng(root_seed,
*stream_key)` builds an independent NumPy `Generator` per consumer from
`SeedSequence([root_seed, *stream_key])` — `derive_seed` is its int-valued sibling for APIs
that take an integer `random_state` (sklearn, umap, `tf.random.set_seed`) — so distinct keys
are statistically independent and each consumer's stream is stable regardless of what the
others draw. The deliberately *unseeded* sites (uuid4 temp names, content-keyed checksums,
per-PID worker init) are catalogued in the module docstring; don't "fix" them.

Both `seed` and `tf_deterministic_ops` are emitted by `Config.to_dict()["reproducibility"]`
(alongside a provenance-only `derived_rf_seed`), so they are part of the run-manifest config
fingerprint: a tag started before these fields existed — or under a different seed — cannot
resume across the change under the same `--save-tag`; the guard downgrades to a fresh run with
a loud warning. See [`CONFIG_AND_CLI.md`](CONFIG_AND_CLI.md) for the config/CLI plumbing and
the [CLI Reference](../README.md#cli-reference) for the exact flag help.

## Checkpointing, the run manifest, and retries

### What gets saved when

| Artifact | When | Where |
| --- | --- | --- |
| `vae_{encoder,decoder}_round_XX.keras` | End of every round | `{model_path}/checkpoints/` |
| `random_forest_{tag}.joblib` (the winning variant) + `random_forest_{tag}_{variant}.joblib` (all 8 variants) + `rf_calibrator_{tag}.joblib` (only when calibration is kept) + `rf_eval_artifacts_{tag}.joblib` | End of `rf_train` | `{model_path}/` |
| `vae_{encoder,decoder}_{tag}.keras`, `random_forest_{tag}.joblib` | `final_save` stage | `{model_path}/` |
| `config_{tag}.json` (resolved config snapshot) | `final_save` stage | `{output_path}/` |
| `run_state_{tag}.json` | Updated after every stage/round transition | `{output_path}/` |

### The run-state manifest

[`run_state.py`](../src/aetherscan/run_state.py)`:TrainingRunState` persists (atomically,
`.tmp` → `os.replace`):

- `run_start_time` — wall clock of **attempt 1**. `TrainingPipeline.__init__` seeds
  `self.start_time` from it, so every DB query and plot spans the whole run, not just the
  current attempt (earlier attempts' epochs stay in the loss curves).
- `attempt` — incremented per pipeline rebuild.
- `completed_rounds` — rounds whose checkpoint landed; the round loop resumes at
  `max(completed_rounds) + 1`, reloading `round_{k:02d}` weights from `checkpoints/`.
- `stages_done` / `stages_failed` — drive the stage machine
  (`train.py:_execute_training_stages`).

On resume, `_init_run_state()` also calls
`db.mark_superseded(table, tag, round_ge=resume_round)` for `training_stats`,
`injection_stats`, and `latent_snapshots`: partial rows written by the dead attempt are
flagged so default queries (and therefore the plots) ignore them — otherwise re-run epochs
would appear twice and corrupt every curve. Rows from completed rounds stay live; they are
valid history. See [`DATABASE.md`](DATABASE.md) for the supersede mechanics.

Explicit checkpoint flags (`--load-tag` / `--load-dir` [+ `--start-round`]) are the escape
hatch: they override the manifest, trim `completed_rounds` below the forced start round, and
clear `stages_done` so downstream stages re-run against the re-trained rounds.

### Retry semantics

`main.py:train_command` retries up to `training.max_retries` times with
`training.retry_delay` between attempts. Each attempt rebuilds the `TrainingPipeline` from
scratch (no corrupted in-memory state survives); the manifest tells the new pipeline where to
resume. Background plates are loaded once in `train_command` and reused across attempts.
The manifest is on disk, so the in-process retry loop resumes off it automatically; a
crash-and-relaunch resumes identically **only when re-invoked with `--load-tag {full-tag}`** (a
bare relaunch mints a fresh datetime tag and starts a new run).

Non-critical plot stages (`vae_plots`, `rf_plots`) never trigger a retry: each plot in the
group is attempted even if a sibling fails (`_run_plot_group`), failures are recorded in
`stages_failed`, and `train_command` exits **nonzero at the very end** if any recorded failure
never recovered — artifacts can be lost loudly, but a broken plot can't cost a
data-regeneration cycle. `rf_train` resumes cheaply too: if the tag's RF joblib + eval
artifacts already exist from a previous attempt, they are loaded instead of regenerating
~`num_samples_rf` cadences and retraining (`try_load_rf_for_resume`).

## Training plots — what each one shows

Per-round copies (tagged `round_XX`) land in `{output_path}/plots/training/{save_tag}/checkpoints/`;
the end-of-training set (tagged with the run tag) in `{output_path}/plots/training/{save_tag}/`. Every
figure is also
uploaded to the run's Slack thread. All of them query the DB with `start_time =
run_start_time`, so multi-attempt runs plot complete histories with superseded rows filtered
out.

### `beta_vae_loss_curves_{tag}.png`

Total loss (full-width top panel) plus reconstruction / KL / true-clustering /
false-clustering / regularization components (bottom row; the regularization panel is empty
for runs predating the 2026-07 L1/L2 activation — see MODELS.md), train and val overlaid,
epochs on the x-axis with
per-round SNR-range shading in the background. Since #277 the x-axis is the **real**
global-epoch position (`(round − 1) · epochs_per_round + epoch`, via `build_epoch_history`):
epochs with no committed row render as visible NaN gaps instead of silently shifting later
epochs left, and a failed pre-plot DB flush now **skips the figure** (raised as a
non-critical plot failure) rather than rendering a partial result set as if it were
complete. What to look for:

- Both curves trending down within each round; **val tracking train** (a widening gap =
  overfitting the current curriculum stage).
- Small upward steps at round boundaries are expected — the data gets harder; a *large*
  sustained jump means the curriculum narrowed too fast (`exponential_decay_rate` too
  negative).
- KL should settle to a moderate plateau: collapsing toward 0 means the posterior ignores the
  input (posterior collapse — see the [per-dimension diagnostics and
  playbook](#posterior_collapse_tagpng--the-collapse-playbook)); growing without bound means
  the latent space isn't regularizing.
- True/false clustering losses should decay and stay low; if `true_loss` dominates late
  rounds, the ON/OFF separation is failing on faint signals.
- Doubled/serrated series are the signature of stale rows from a failed attempt leaking in —
  they should never appear now that resumes mark old rows superseded; if you see them, check
  the `mark_superseded` warnings in the log.

### `beta_vae_training_stability_{tag}.png`

2×3 grid: gradient **clipping rate** across the top, gradient-norm mean/std/max across the
bottom, same SNR shading; same real-epoch axis / NaN-gap / skip-on-failed-flush contract as
the loss curves. What to look for: clipping rate near zero after the first epochs
(sustained clipping = LR too high for the stage); norm mean smooth and slowly decaying;
isolated max spikes are fine, but spikes that coincide with loss cliffs point at bad batches
or an injection bug. NaN/Inf gradients abort the epoch outright, so anything you see here was
at least finite.

### `posterior_collapse_{tag}.png` + the collapse playbook

Per-dimension KL diagnostics (#282), from the `kl_dim_NN` `training_stats` rows written each
epoch: a `latent_dim × global-epoch` **KL heatmap** (collapsing dims visibly go dark) over an
**active-units curve** (dims with KL > `training.posterior_collapse_kl_epsilon` per epoch,
with the `min_active_units_fraction` alarm line). Same flush-skip contract as the loss
curves. The plot is the offline view of the per-round `check_posterior_collapse()` guard,
which WARNs (reaching Slack, never failing the run) when the active fraction drops below
`training.min_active_units_fraction` or any dim's KL sits under epsilon for
`training.posterior_collapse_patience` consecutive epochs.

What to look for — and what to do about it:

- With `beta > 1` some pruning is **expected and even desirable**: 6–8 active dims of 8 is
  healthy; 1–2 is pathological (the VAE is ignoring its latent capacity and the RF has almost
  nothing to work with).
- Remedies, least → most intrusive: **KL warm-up** (anneal `beta` in from 0 over early
  epochs), **free bits** (exempt a per-dim KL floor from the loss), **lower `vae_beta`**,
  **shrink `latent_dim`** (if dims stay dead across remedies, the capacity was never needed).
- If the check *passes*, read it as evidence the `log_var`-carrying RF variants (below)
  deserve their weight — and confirm the active dims are actually discriminative via the
  SHAP summary plot (active-but-uninformative dims are possible).

### Injection-stats figures (from `plot_injection_stats`, 8 PNGs)

Bias/leakage analysis of the synthetic data itself, sourced from the `injection_stats` table
(intensity statistics captured at stage **A** = raw background, **B** = post-injection,
**C** = post-normalization; see [`PREPROCESSING.md`](PREPROCESSING.md)). Since #277 every
query is **round-scoped**: the per-round call passes its own `round_number`, and the
end-of-run call spans rounds `1..num_training_rounds` — so the background producer's
pre-generated next-round rows and the RF stage's sentinel round
(`num_training_rounds + 1`) can no longer bleed into the figures. Injection rows ride the
DB's bulk lane (which `flush()` deliberately does not cover), so before rendering the
function gates on `db.injection_backlog_rows(max_round=...)` and skips (non-critically)
while any row for the plotted rounds is still queued — never a partial result set:

| File | Contents | What to look for |
| --- | --- | --- |
| `injected_signal_characteristics_{tag}.png` | Distributions of realized SNR, drift rate, signal width, starting bin, slope, intercept for ETI vs RFI injections, plus background-index usage. | ETI and RFI parameter distributions should match (the classifier must not be able to tell them apart from injection parameters alone); background usage should be uniform. |
| `injection_stability_{tag}.png` | Per-round NaN/Inf sanitization rate per statistic + slope-clamping rate. | Both ≈ 0. A rising sanitization rate means numerically degenerate cadences; clamping spikes mean the drift-slope edge case is being hit unusually often. |
| `{signal_type}_global_intensity_distributions_{tag}.png` (×4) | 2×3 histograms of mean/median/std/MAD/skew/kurtosis at stages A/B/C for one signal type. | Stage C distributions should be near-identical **across** the four types — any statistic that separates the types at stage C is leakage the models could shortcut on instead of learning morphology. |
| `a_b_global_intensity_biases_{tag}.png` | A→B scatter (pre- vs post-injection) per statistic, colored by signal type, outliers always kept. | Points should hug the diagonal with a modest, SNR-consistent offset for injected classes. Big vertical excursions = injections that dominate the background (dynamic-range bug). |
| `final_global_intensity_biases_{tag}.png` | Stage-C box plots per statistic, compared across signal types. | Boxes should overlap heavily. Separated medians = the normalization didn't erase injection-strength cues. |

### `latent_space_{obs,cadence}_nn{n}_md{m}_{tag}.gif`

UMAP animations of latent-space evolution over training, one GIF per
(n_neighbors, min_dist) combination in the configured sweep, built from the
`latent_snapshots` table (the withheld viz batch is re-encoded every
`latent_viz_step_interval` steps):

- **obs-level**: each point is one observation's 8-dim latent, 8 classes (4 signal types ×
  ON/OFF) — the VAE's view.
- **cadence-level**: each point is a cadence's 48-dim concatenated latent, 4 classes — the
  RF's view.

What to look for: over the animation, true-class ON points should drift away from OFF/false
points (the clustering loss doing its job); by the final frames the cadence-level view should
show 4 separable — not necessarily linearly — clusters. Classes collapsing back together in
late rounds mean the faint-SNR curriculum is destroying earlier structure. The fitted UMAP
models are persisted (`umap_*.joblib`) and reused by the RF decision-boundary plot and by
inference's latent-projection figure.

The sweep is combo-parallel (the #278 follow-up): each (n_neighbors × min_dist ×
obs/cadence) combo — 18 at the current defaults (3 × 3 × 2; the sweep was 24 until
`n_neighbors=30` was dropped as redundant between 15 and 50) — is an independent UMAP fit
with its own derived `random_state` (sub-keyed `(level, nn, md)`, #279), reads the shared
inputs read-only (shipped to workers through one on-disk joblib bundle), and writes
distinct files — so
[`latent_gif.py`](../src/aetherscan/latent_gif.py)`:run_umap_gif_sweep` farms WHOLE combos
(fit + joblib persist + per-snapshot transforms + frame render + GIF assembly) to forkserver
workers (empty preload; BLAS-family thread pools pinned per the `shap_parallel.py` isolation
pattern — but deliberately NOT numba's or OMP's: UMAP grows numba's pool itself on large-N
paths and numba hard-errors past a capped launch, see `_sweep_worker_init`). At production
shape the serial tail is ~1.7–1.9 h (~95% single-core, even after #278 parallelized frame
rendering); the first parallel measurement, at a reduced 60-frame shape with 24 workers on
blpc3, is ~93 min (memory-bandwidth contention between concurrent single-threaded fits), so
no production-shape speedup can be quoted yet. Logging and Slack uploads stay in the parent
process, in the serial loop's order; byte-identity of the GIFs is pinned by a slow-marked unit test
comparing serial vs pooled output. This is disjoint from the #278-**rejected** within-fit
ideas: batching the per-snapshot `.transform()` calls and reusing a precomputed kNN graph
across the sweep remain rejected because they change how UMAP consumes its `random_state`
stream (per-snapshot transforms stay serial *within* each combo) — process isolation between
combos changes no stream at all.

### `latent_traversal_{signal_type}_{tag}.png` + `latent_traversal_spectra_{signal_type}_{tag}.png`

Decoder-based interpretation of the latent dimensions (`plot_latent_traversal`, helpers
`build_traversal_latents` / `compute_traversal_panels` / `unpreprocess_traversal_panels`).
For each signal type: the class-mean latent `z_t` (mean encoder `z_mean` over that type's ON
observations) is nudged one dimension at a time, `z_t + s·σ_d·e_d` for steps
`s ∈ linspace(−max_sigma, +max_sigma, num_steps)` (defaults 3.0, 7; `num_steps` validated odd
so the center column is the exact unperturbed decode), and decoded:

- The **waterfall grid** (`latent_dim × num_steps` panels, shared per-row color scale) shows
  what each dimension *does*: scan a row and watch the reconstruction morph.
- The **spectra figure** (per-dim time-integrated spectra, one line per step) makes
  brightness/width/position shifts quantitative at a glance.

What to look for: each row should vary one interpretable property (signal brightness, drift,
width, position...) — that's the disentanglement `beta` buys. Rows that do nothing are dead
dimensions (latent capacity to spare); rows that change everything at once suggest an
entangled space. Display inversion is an honest approximation (stated on the figure):
downsampling is undone by ×8 nearest-neighbor repetition, and intensities are un-log-normed
only where per-observation parameters were recorded at generation time (the `*_lognorm.npy`
sidecars). Runs once at end of training (`vae_plots` stage); `--latent-traversal-every-round`
adds per-round copies. On a resumed run whose rounds all completed before the resume, the
in-memory viz batch never existed, so the plot skips with a warning.

### RF diagnostics (`rf_plots` stage, 10 PNGs)

All consume `rf_eval_artifacts_{tag}.joblib` (val features/labels/probas thresholded at the
**deployment** `classification_threshold`, not sklearn's 0.5 default); the five SHAP figures
share `rf_shap_values_{tag}.joblib` (computed once, cached). Since #282 the artifacts carry
the *winning variant's* features, both raw (`val_probas` — rank plots are
calibration-invariant) and deployment-scored (`val_probas_deployed`, calibrated when a
calibrator is active) probabilities, plus the sweep record (variant metrics, calibration
outcome, val partition indices).

| File | Contents | What to look for |
| --- | --- | --- |
| `rf_confusion_matrices_{tag}.png` | Binary (2×2) and per-subtype (4×2) confusion matrices at the deployment threshold. | With the default 0.99 threshold expect conservative behavior: near-zero false positives at the cost of true-class recall. Check the subtype panel for *which* true class carries the misses (usually `true_eti_rfi`). |
| `rf_classification_curves_{tag}.png` | ROC + AUC, PR + AP, confidence histograms (overall and per subtype). | AUC/AP near 1 on synthetic val is normal; the interesting part is the confidence histograms — a clean bimodal split means the threshold placement is easy, mass near the threshold means candidate counts will be sensitive to it. |
| `rf_shap_summary_{tag}.png` | Beeswarm of top features driving P(true). | Features are `obs{i}_z{d}` (observation × latent dim). ON-observation features (obs 0/2/4) should dominate — that's the physics. OFF features ranking high means the RF keys on OFF-source structure (leakage or RFI shortcuts). |
| `rf_shap_dependence_{tag}.png` | Dependence panels for the top-K features, colored by the strongest interacting feature. | Smooth monotone-ish trends = healthy; vertical striping = the RF memorizing discrete latent values. |
| `rf_shap_interactions_{tag}.png` | Pairwise interaction matrix (diagonal = main effects). | Strong off-diagonal ON×OFF blocks mean the forest genuinely compares ON against OFF within a cadence (good — that's the ABACAD logic); a purely diagonal matrix means per-observation features alone are being used. |
| `rf_shap_loss_monitoring_{tag}.png` | Per-sample log-loss histogram by class + per-feature loss-increasing/decreasing decomposition. | The high-loss tail is your inspection queue; any feature whose net contribution *increases* loss is actively harmful. |
| `rf_shap_explanation_clustering_{tag}.png` | UMAP of SHAP explanation vectors, colored by subtype, markers for correct/incorrect. | Errors concentrated in one explanation cluster = a single confusable mode (fixable with targeted data); errors scattered everywhere = noise-floor performance. |
| `rf_calibration_curve_{tag}.png` | Reliability diagram (quantile-binned) + Brier/ECE + probability histogram. | With a 0.99 threshold, calibration in the top bins is what matters: if the top-bin empirical frequency is well below its predicted probability, the threshold is less conservative than it looks. |
| `rf_oob_accuracy_curve_{tag}.png` (from `plot_rf_ensemble_accuracy_curve`) | Cumulative accuracy vs number of trees (val + train-subsample baseline), elbow annotated. Also persists the per-tree `ensemble_val_accuracy` series to `training_stats` (`model_name='rf'`, `epoch_number` = tree count) for the dashboard RF tab — so the DB series only lands when `rf_plots` succeeds. | Should saturate well before 1000 trees; if it's still climbing at the end, raise `rf.n_estimators`. |
| `rf_latent_decision_boundary_nn{n}_md{m}_{tag}.png` | RF P(true) contour over each persisted cadence-level UMAP plane, val points + 0.5 contour. The UMAP lives in the 48-dim `z_mean` space; wide #282 variants project their lead columns only, and grid features the inverse transform can't reconstruct (the uncertainty extras) are held at their training-set means — the boundary is then the slice through typical uncertainty (a stated approximation). | A coherent boundary separating the true classes; ragged islands = the forest partitioning noise. Depends on the UMAPs from `plot_latent_space_gif`, so `vae_plots` must have succeeded. |

### SHAP explainability performance (CPU multiprocessing; GPU is a documented alternative)

The five SHAP figures share `rf_shap_values_{tag}.joblib`, computed once by
`_compute_or_load_shap_values`. shap's TreeSHAP C extension is **single-threaded** (no OpenMP, no
`n_jobs`; the RF's own `n_jobs` does not apply — shap re-walks the trees itself), so on a 1000-tree
forest the step is dominated by the **interaction** pass and runs for hours-to-days if left serial
(measured ~183 s/sample on a depth-53 RF → ~76 h for 1500 interaction samples; the whole tail is
~95% interaction).

SHAP values are per-sample independent, so we **chunk the samples across all cores**
(`aetherscan.shap_parallel`, driven by `manager.n_processes` = `cpu_count()` by default): each worker
rebuilds a *stock* `TreeExplainer` and explains its chunk, and the results are byte-identical to the
serial computation (measured ~40-45x on a 96-core node). This is the shipped path for all three
passes (summary, interaction, log-loss).

#### GPU is faster on interaction, but we don't use it — here's why, and how to switch

shap's `GPUTreeExplainer` (GPUTreeShap) runs each pass in ~seconds regardless of sample count (~1000x
on interaction). We benchmarked it on both clusters; it works and is **correct** for the summary and
interaction passes (`np.allclose` vs CPU). We still ship CPU multiprocessing, because the GPU route
buys only a few minutes at the very end of a multi-day/-week run in exchange for maintaining an extra
CUDA + from-source dependency on an experimental shap code path:

- **Not in the stock wheel.** `GPUTreeExplainer` needs a `_cext_gpu` CUDA extension that exists only
  if shap is built from source with `SHAP_ENABLE_CUDA=1` + a CUDA toolkit — a bespoke build baked into
  the container/conda and re-verified on every container/shap/CUDA bump.
- **Hard depth limit (fixed lane overflow).** GPUTreeShap maps one path element per CUDA warp *lane*
  (`warpSize == 32`, fixed on every NVIDIA arch), so an over-long root-to-leaf path overflows and the
  kernel aborts (`Tree depth must be < 32`, core dump). Precisely, it caps the number of **distinct
  features per root-to-leaf path at ≤ 31** (paths are de-duplicated by feature before the length
  check), so `max_depth ≤ 31` is a *sufficient*, guaranteed-safe setting; a deeper forest can still
  run if no path uses > 31 features, but that is not guaranteed. It is **not** fixable by a different
  build.
- **Log-loss is broken on GPU.** The interventional `model_output="log_loss"` path silently returns
  raw-margin numbers (the GPU kernel drops the output-transform pointer — shap #4270/#3936/#1726,
  unfixed as of 0.46.0) and fails the additivity axiom, so log-loss would have to stay on CPU anyway.

**To switch to GPU later** (if interaction runtime ever becomes the bottleneck): (1) cap the RF at
`max_depth = 31` in `RandomForestConfig` and confirm val-AUC is unaffected with an A/B; (2) bake a
CUDA-built shap into `aetherscan.def` / `environment.yml`
(`SHAP_ENABLE_CUDA=1 CUDA_PATH=/usr/local/cuda pip install --no-binary shap shap==<pinned>`);
(3) route **summary + interaction** through `shap.explainers.GPUTree`, keeping **log-loss** on the CPU
multiprocessing path; (4) CPU-validate the GPU output with `np.allclose` before trusting it.

**On the depth cap:** we keep `max_depth` **unbounded** (its sklearn default; not set in
`RandomForestConfig`) — the `< 32` limit only matters *if* we adopt GPU SHAP. A cap would be low-risk
in practice: the Beta-VAE's objective is to make the classes separable in latent space, so a
well-trained VAE yields simple decision boundaries and naturally shallow trees, and the cap then just
acts as a mild regularizer aligned with the upstream objective. It could bite an **undertrained** VAE
(muddy latents → deep trees) or a future extension to more complex signal morphologies whose latents
genuinely need deeper trees for accuracy — which the val-AUC A/B in step (1) would catch.

### `resource_utilization_{tag}.png`

Written by the resource monitor at shutdown, not by `train.py` — see
[`RUNTIME_SERVICES.md`](RUNTIME_SERVICES.md). Read it alongside the log timeline: data
generation shows as CPU-saturated plateaus, epochs as GPU-utilization bands, and (with
overlap enabled) the two should visibly coincide from round 2 onward.

## Random Forest training (`rf_train` stage)

`train_random_forest()` generates a fresh dataset (`num_samples_rf`, default 99 840; SNR range
= `initial_snr_range` — the wide range, so the RF sees the full difficulty spectrum) into
`round_data/{tag}/rf/` using the same memmap machinery (in-process; the producer has already
shut down). It reuses `prepare_distributed_train_dataset(shuffle=False)` and encodes train and
val cadences through the (frozen) encoder with `_distributed_encode` — note the
`train_steps × accumulation_steps` step-count correction, guarded by an exact-count assertion.
Since #282 the encode keeps **all three** encoder outputs (`z_mean`, `z_log_var`, `z` — the
posterior parameters used to be computed and thrown away here), which makes the
representation sweep below free on the GPU side. A `check_encoder_trained()`
heuristic (weight-std deviation from initializer expectations) guards against accidentally
encoding with untrained weights and falls back to loading the newest checkpoint.

### The latent-representation sweep (#282)

The RF historically consumed the stochastic sampled `z`; nobody had ever tested whether that
is the best representation. `train_random_forest()` now trains **all 8 variants** of the
catalogue in [`latent_variants.py`](../src/aetherscan/latent_variants.py) (see
[`MODELS.md`](MODELS.md) for the table) on the same generated data and split, and picks the
winner empirically:

1. **Active units** are measured first (Burda et al.: dims whose `z_mean` variance exceeds
   `rf.active_units_threshold`) — they gate the `z_mean_logvar_active` variant and size any
   collapse problem retroactively.
2. **The val split is partitioned** (seeded) into *selection* / *calibration* / *test* via
   `rf.val_selection_fraction` / `rf.val_calibration_fraction` (remainder = test). Release
   metrics are reported on the held-out test split — best-of-8 on the selection split alone
   would be optimistically biased.
3. **Every variant is fit and evaluated under its deterministic inference-time form**
   (`z_mean` in the lead feature slot — for the `z`/`z_aug` variants that is deliberately the
   deployed configuration, not the training one) and saved as
   `random_forest_{tag}_{variant}.joblib`.
4. **Selection**: the primary metric is recall at `rf.selection_max_fpr` on the selection
   split (AUC averages over operating points the pipeline never uses). The best variant must
   beat every *simpler* (fewer-feature) variant by more than a bootstrap CI of the recall
   difference (`rf.selection_bootstrap_rounds`), else the simpler variant wins the tie —
   `select_winner()`'s minimum-margin rule.
5. **The winner becomes THE model** — canonical `random_forest_{tag}.joblib` filename, HF
   upload, and release tagging all pick it up unchanged — and the sweep outcome is recorded
   on the config (`rf.latent_variant`, `rf.active_dims`) so `config_{tag}.json` tells
   inference exactly how to rebuild features (never hardcoded). A resumed attempt restores
   these fields from the eval artifacts instead of re-sweeping.
6. **ECE-gated calibration**: the winner's ECE is measured on the calibration split; only
   when it exceeds `rf.max_ece` is a calibrator fit (isotonic with ≥
   `rf.calibration_min_isotonic` rows, else sigmoid/Platt — isotonic overfits small sets),
   and it is **kept only if** it improves ECE without worsening Brier on the held-out test
   split. A kept calibrator is persisted as `rf_calibrator_{tag}.joblib` and recorded as
   `rf.calibration_active` / `rf.calibration_method`; inference applies it identically
   (an unapplied calibrator would be a silent train/serve mismatch).
7. **Screening-threshold validation**: `check_screening_threshold()` replays the two-pass
   inference cascade on the test split and WARNs (advisory, like `check_val_auc_floor`) if it
   loses more than `rf.screen_recall_tolerance` recall versus MC-scoring *everything* at the
   science threshold — anything pass 1 rejects never gets a second look, so the screen must
   be demonstrably safe before a science run. The log also reports the largest zero-loss
   screening threshold on that split. See [`INFERENCE_PIPELINE.md`](INFERENCE_PIPELINE.md)
   for the cascade itself.

At the tail of the stage, `train_random_forest()` also persists scalar RF eval metrics
(accuracy, ROC-AUC, average precision, Brier score, per-sub-type accuracies, binary +
sub-type × prediction confusion cell counts, val P(true) quantiles) to `training_stats`
under `model_name='rf'` via the pure (TF-free) helper
[`compute_rf_eval_metrics()`](../src/aetherscan/rf_metrics.py); the deployment
`inference.classification_threshold` used to derive `val_accuracy` is written alongside as
its own `classification_threshold` row, and since #282 so are the held-out test metrics
(`test_*`), the screening-validation numbers (`screen_*`), the active-unit count, and the
per-variant selection metrics (`variant_{name}_{metric}`). `plot_rf_ensemble_accuracy_curve()` (in the
downstream `rf_plots` stage) then writes the per-tree `ensemble_val_accuracy` series
(`epoch_number` = tree count) so the dashboard's RF tab is live end-to-end. The ensemble
curve keeps its pre-existing hard-coded 0.5 threshold (the dashboard shows a caption to
disambiguate it from the deployment-threshold scalar). Metric persistence is best-effort:
an sklearn edge case (e.g. a single-class val split) logs a warning and never fails the
training run.

## Configuration quick reference

Training-specific fields live on `TrainingConfig`
([`config.py`](../src/aetherscan/config.py)); flag routing and validation are documented in
[`CONFIG_AND_CLI.md`](CONFIG_AND_CLI.md). The load-bearing groups:

| Group | Fields |
| --- | --- |
| Scale | `num_training_rounds`, `epochs_per_round`, `num_samples_beta_vae`, `num_samples_rf`, `train_val_split` |
| Posterior-collapse guard | `posterior_collapse_kl_epsilon`, `min_active_units_fraction`, `posterior_collapse_patience` |
| Batching | `per_replica_batch_size`, `effective_batch_size`, `per_replica_val_batch_size` |
| Round data | `round_data_dir`, `overlap_data_generation`, `keep_round_data`, `signal_injection_chunk_size`, `data_gen_task_size`, `round_array_dtype` (A/B-gated) |
| Curriculum | `snr_base`, `initial_snr_range`, `final_snr_range`, `curriculum_schedule`, `exponential_decay_rate`, `step_easy_rounds`, `step_hard_rounds` |
| Adaptive LR | `base_learning_rate`, `min_learning_rate`, `min_pct_improvement`, `patience_threshold`, `reduction_factor` |
| Latent viz / traversal | `latent_viz_*`, `latent_traversal_every_round`, `latent_traversal_num_steps`, `latent_traversal_max_sigma` |
| RF plots | `shap_max_samples_*`, `shap_top_k_features_dependence`, `rf_decision_boundary_*` |
| Fault tolerance | `max_retries`, `retry_delay` |
