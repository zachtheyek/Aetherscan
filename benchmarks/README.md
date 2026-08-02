# Micro-benchmarks

Small standalone scripts that time the pipeline's hot CPU kernels in isolation, so a change
to any of them can be measured in seconds instead of via a full training/inference run. No
framework — each script prints ops/s and writes a JSON result to `benchmarks/results/`
(gitignored; per-machine baselines live in the table below).

These are **not** collected by pytest (`testpaths = ["tests"]` in `pyproject.toml`) and are
run on demand:

```bash
# From the repo root (each script inserts src/ into sys.path itself)
python benchmarks/bench_normality.py            # sliding-window normality test (energy detection)
python benchmarks/bench_injection.py            # setigen signal injection (training data gen)
python benchmarks/bench_lognorm_downsample.py   # per-cadence downsample + log-norm (load path)
python benchmarks/bench_pfb_vs_spline.py        # PFB static equalization vs per-channel spline fit
python benchmarks/bench_rf.py                    # Random Forest stage: latent prep + fit + predict
# On the clusters, run through the container:
./utils/run_container.sh python benchmarks/bench_normality.py
# Container-only (need the full TF/umap stack; bench_input_pipeline's step mode needs GPUs):
./utils/run_container.sh python benchmarks/bench_input_pipeline.py --mode iterate --variant current
./utils/run_container.sh python benchmarks/bench_latent_gif.py --mode all
# Generation-path A/B (real pooled producer path; --preload-tf needs TF, so run containerized):
./utils/run_container.sh python benchmarks/bench_datagen.py --preload-tf \
    --data-dir /datax/scratch/$USER/data/aetherscan/bench/datagen
```

Bulk bench data (the synthetic round `bench_input_pipeline.py` writes and the generated round
`bench_datagen.py` drives) defaults to `{AETHERSCAN_DATA_PATH}/bench/input` and
`{AETHERSCAN_DATA_PATH}/bench/datagen` respectively — the same data root the pipeline uses
(falling back to `/datax/scratch/zachy/data/aetherscan` when the env var is unset). Pass
`--data-dir` to override. Result JSONs are unaffected and still land in `benchmarks/results/`.
`bench_input_pipeline.py`'s round is ~12 GB at the default `--n-samples` and is reused across
invocations (`--regen` to rewrite); `bench_datagen.py` deletes its round on exit unless `--keep`
is passed.

`bench_gpu.py` is a different animal — it profiles the Beta-VAE on a real GPU (throughput +
peak VRAM, with a batch-size sweep) rather than a CPU kernel, so it only runs inside the
container on a cluster. See [GPU benchmark](#gpu-benchmark) below.

Common flags: `--repeats N` (default 3; ops/s is reported from the best repeat) and
`--output PATH` for the JSON result. Each script also exposes size knobs (`--width`,
`--injections`, `--cadences`) — the defaults match production shapes, so stick to them when
comparing against the baselines.

## What each script measures

| Script | Kernel | Pipeline stage it models |
|---|---|---|
| `bench_normality.py` | `preprocessing._sliding_normality_k2` vs the historical per-window `scipy.stats.normaltest` loop | Energy detection thresholding (`inference.*.read_ed`) |
| `bench_injection.py` | `data_generation.new_cadence` (setigen narrowband injection into a stacked 96x4096 cadence) | Training data generation (`train.round_XX.data_generation`) |
| `bench_lognorm_downsample.py` | per-observation `downscale_local_mean` (x8) and per-cadence `log_norm` | Stamp extraction downsample + inference load (`inference.*.load_lognorm`) |
| `bench_pfb_vs_spline.py` | `pfb.equalize_passband` (static response divide) vs `preprocessing._spline_flatten_bandpass` (order-16 fit) on one 1M-bin coarse channel | Bandpass flattening inside energy detection |
| `bench_gpu.py` | Beta-VAE training step (`compute_total_loss` + gradients + clipped Adam) and encoder forward on one or more GPUs | VAE training (`train.round_XX`) and encoder inference — **GPU-only, see below** |
| `bench_rf.py` | `RandomForestClassifier` fit + `predict_proba` and the `prepare_latent_features` reshape (sklearn, CPU) | Second-stage RF training + inference (`train.train_random_forest`) |
| `bench_input_pipeline.py` | The REAL memmap → tf.data → distribute → train-step input path (gather / iterate / step modes; legacy vs current builder; `--gil-load` contention knob) — **container-only; step mode needs GPUs** | The training input pipeline `bench_gpu.py` deliberately excludes (`train.prepare_distributed_train_dataset`) — see [#276 audit](#input-pipeline-audit-276) |
| `bench_latent_gif.py` | Latent-GIF stage decomposition (UMAP fit / transform / frame render / GIF assembly) with output-equality checks on every candidate optimization — **container-only** | The `vae_plots` latent-GIF tail (`train.plot_latent_space_gif`) — see [#278 audit](#latent-gif-audit-278) |
| `bench_datagen.py` | Seeded round generation through the REAL pooled `generate_round_to_memmap` path (shared-memory plates, `_init_worker`, per-task seed derivation, batched memmap tasks) with per-array sha256 checksums — the byte-compatibility gate for generation-path changes; `--preload-tf` mirrors the producer workers' TF import graph | The producer wall (`train.round_XX.data_generation`) that `bench_injection.py`'s single-process kernel can't reach — see [Producer/data-generation follow-up](#producerdata-generation-follow-up) |
| `bench_injection_index.py` | The schema-v7 secondary `injection_stats` index (`idx_injection_stats_by_stat` vs the original `idx_injection_stats_filter`): bulk-insert write cost and the end-of-run plot query shape (equality on stat/type/stage + run-wide time bounds), timed with and without `ANALYZE`, with an `EXPLAIN QUERY PLAN` per setup — SQLite, no TF/GPU | The `plot_injection_stats` end-of-run query pass — the ~165-query tag-partition scan the second index targets — see [Producer/data-generation follow-up](#producerdata-generation-follow-up) |
| `bench_db_index_shapes.py` | The schema-v7 index reshapes on `training_stats` and `latent_snapshots` (old filter index vs the equality-first replacement): every production query shape + insert throughput against production-shaped synthetic DBs, `EXPLAIN QUERY PLAN` per shape (companion to `bench_injection_index.py`) — SQLite, no TF/GPU | The latent-GIF frame fetch (`train.plot_latent_space_gif`) plus the training-stats/dashboard reads — the schema-v7 DB-index audit |
| `parse_xplane_occupancy.py` | Post-processor, not a benchmark: per-GPU busy time / occupancy from a `--profile` run's XPlane trace (merged event intervals, strict kernel-time measure) | The "GPUs idle >90%" profiler evidence in the [#276 audit](#input-pipeline-audit-276) and its follow-up |

## Baseline numbers

Higher is better everywhere. Numbers are the best-of-3 repeats at default parameters;
expect ~10% run-to-run noise. Regenerate after touching any of the measured kernels and
update this table in the same PR.

### MacBook Air M3 (arm64, macOS, Python 3.12, July 2026)

| Benchmark | Result |
|---|---|
| `bench_normality` — vectorized | 44,442 windows/s (0.18 s per 1M-bin channel) |
| `bench_normality` — scipy loop | 689 windows/s (11.9 s per channel) → **64x speedup** |
| `bench_injection` | 6.6 injections/s |
| `bench_lognorm_downsample` — downsample | 1,900 cadences/s |
| `bench_lognorm_downsample` — lognorm | 4,861 cadences/s |
| `bench_pfb_vs_spline` — pfb | 46.3 channels/s (plus a 7.7 s one-time response FFT) |
| `bench_pfb_vs_spline` — spline | 6.0 channels/s → **7.7x speedup** |
| `bench_rf` — prep (latent reshape) | 1,936,313 cadences/s (0.05 s) |
| `bench_rf` — fit (1000 trees) | 340 cadences/s (235 s) — random-label upper bound¹ |
| `bench_rf` — predict | 15,989 cadences/s |

¹ `bench_rf` uses random binary labels, so the trees grow until pure — a conservative *upper*
bound on fit time; real (separable) latents yield shallower trees that fit faster. The prep
reshape is negligible (~0.05 s for the full 99,840-cadence set).

### blpc3 (EPYC 7313, 32 cores, NGC 25.02 container, July 2026)

| Benchmark | Result |
|---|---|
| `bench_normality` — vectorized | 83,667 windows/s (0.10 s per 1M-bin channel) |
| `bench_normality` — scipy loop | 679 windows/s (12.1 s per channel) → **123x speedup** |
| `bench_injection` | 5.8 injections/s |
| `bench_lognorm_downsample` — downsample | 798 cadences/s |
| `bench_lognorm_downsample` — lognorm | 8,844 cadences/s |
| `bench_pfb_vs_spline` — pfb | 59.6 channels/s (plus an 11.7 s one-time response FFT) |
| `bench_pfb_vs_spline` — spline | 5.0 channels/s → **11.9x speedup** |

Notes:

- All numbers are single-process: the pipeline parallelizes these kernels across
  `manager.n_processes` workers, so whole-stage throughput scales roughly with core count
  (e.g. energy detection runs one fused task per coarse channel on the persistent pool).
- `bench_injection` is dominated by setigen frame construction, which is why data
  generation gets a dedicated producer process and worker pool in training.

## GPU benchmark

`bench_gpu.py` profiles the part of the pipeline the CPU micro-benchmarks above can't reach: the
Beta-VAE on the GPU. It needs a physical GPU and the full TensorFlow / aetherscan stack, so it
only runs inside the NGC container on a cluster. It builds the real model
(`create_beta_vae_model`) under a `MirroredStrategy` over the first `--num-gpus` GPUs (default 1)
and drives synthetic batches of the true pipeline shapes, measuring aggregate throughput and
per-GPU peak VRAM and sweeping the per-replica batch size to find the largest that fits.

```bash
# Largest per-replica batch that fits, with throughput + VRAM at each size (stops at the first OOM):
./utils/run_container.sh python benchmarks/bench_gpu.py --mode train  --find-max
./utils/run_container.sh python benchmarks/bench_gpu.py --mode encode --find-max
# Or measure specific sizes:
./utils/run_container.sh python benchmarks/bench_gpu.py --mode train --batch-sizes 128,256,512
# Multi-GPU scaling (MirroredStrategy all-reduce across all replicas):
./utils/run_container.sh python benchmarks/bench_gpu.py --mode train --num-gpus 6 --batch-sizes 128
# Model the accumulate-then-apply cadence of the real training loop (one apply per K micro-batches):
./utils/run_container.sh python benchmarks/bench_gpu.py --mode train --num-gpus 6 --batch-sizes 128 --accumulation-steps 4
```

A training example is a *cadence* `(6, 16, 512)`, so one step runs `18*B` encoder + `6*B` decoder
passes (`B` = per-replica batch of cadences, matching `_distributed_train_step`); `encode` mode
drives single observations `(16, 512, 1)` like inference. Peak VRAM is reported **per GPU** — each
replica holds a full copy of the model, so combined VRAM is roughly this figure times the replica
count. With `--accumulation-steps K`, one optimizer step accumulates the all-reduced grads over K
micro-batches and applies once (as `_train_epoch`), so peak VRAM then includes the persistent
accumulator and throughput reflects the once-per-K apply cadence. `--mixed-precision` sets the
keras `mixed_bfloat16` global policy before the model is built (the same policy as the pipeline's
`beta_vae.mixed_precision`, fp32 islands in `models/vae.py` included); default off = fp32, matching
the pipeline default. `--find-max` is capped at
`--max-batch 4096`: a single encoder forward whose conv feature
maps exceed 2^31 elements (batch ≳ 8192) trips an uncatchable TensorFlow int32 launch-config abort
rather than a clean OOM, and 4096 already covers the training VRAM ceiling and a generous inference
range (the pipeline's inference default is 256 SNIPPETS per replica = 1,536 obs forwards —
see the takeaway below).

### Baseline: blpc3 (1× RTX PRO 6000 Blackwell, 96 GB, NGC 25.02, July 2026)

Training step — per-replica batch of cadences:

| per-replica batch | cadences/s | peak VRAM |
|---|---|---|
| 128 | 2,986 | 2.81 GB |
| 256 | 2,892 | 5.21 GB |
| 512 | 2,852 | 10.15 GB |
| 1024 | 2,826 | 19.79 GB |
| 2048 | 2,786 | 38.76 GB |
| 4096 | 2,742 | 76.52 GB |
| 8192 | OOM | — |

Encoder inference — per-replica batch of observations:

| per-replica batch | obs/s | peak VRAM |
|---|---|---|
| 512 | 198,963 | 0.52 GB |
| 1024 | 231,065 | 1.01 GB |
| 2048 | 197,361 | 1.72 GB |
| 4096 | 192,104 | 3.39 GB |

**Takeaway — the Beta-VAE is compute-bound, not VRAM-bound.** Training throughput peaks at
per-replica batch ~128 (2.81 GB) and does *not* improve — it slightly declines — as the batch grows
toward the 76 GB VRAM ceiling; inference throughput peaks near batch ~1024 using under 2 GB. On a
large-VRAM card (e.g. the 96 GB Blackwell) the spare memory cannot be converted into throughput for
this compact model, so the defaults are already near-optimal — size the per-replica batch for
constraint-divisibility and convenience, not to fill VRAM. ⚠ Units: this table sweeps
OBSERVATIONS per forward; `inference.per_replica_batch_size` counts SNIPPETS (= 6
observations each). The old 2048-snippet default was exactly that conflation — a
12,288-obs per-replica forward, past the ~1024-obs peak above and above the int32 abort
cliff — which is why #298 set it to 256 snippets (1,536 obs).

### Baseline: bla0 (6× RTX A4000 16 GB, NGC 25.02, July 2026)

bla0 is the release-training host. Single-GPU training step and encoder inference:

| per-replica batch | train cadences/s | train VRAM | encode obs/s | encode VRAM |
|---|---|---|---|---|
| 128 | 795 | 2.76 GB | 47,868 | 0.12 GB |
| 256 | 819 | 5.26 GB | 50,364 | 0.19 GB |
| 512 | 820 | 9.99 GB | 51,016 | 0.36 GB |
| 1024 | OOM | — | 52,462 | 0.61 GB |
| 2048 | — | — | 53,077 | 1.18 GB |
| 4096 | — | — | 53,001 | 2.32 GB |

Multi-GPU (all 6 A4000s, per-replica batch 128, **aggregate** throughput):

| config | throughput | scaling | VRAM/GPU |
|---|---|---|---|
| train, 1 GPU | 795 cad/s | 1.0× | 2.76 GB |
| train, 6 GPUs | 3,087 cad/s | 3.9× | 3.21 GB |
| train, 6 GPUs, `--accumulation-steps 4` | 4,272 cad/s | — | 9.91 GB |
| encode, 6 GPUs (batch 1024) | 214,451 obs/s | 4.1× | 0.78 GB |

The A4000 is ~3.8× slower per GPU than the Blackwell (795 vs 2,986 cad/s at batch 128), consistent
with its lower compute; the model is still compute-bound (throughput flat as batch grows toward the
16 GB ceiling — training OOMs above batch 512). MirroredStrategy scales ~3.9–4.1× across the 6
replicas (all-reduce overhead eats the rest). `--accumulation-steps 4` raises throughput (4,272 vs
3,087 cad/s) because it runs one optimizer apply — and therefore one clip + one cross-replica
all-reduce — per 4 micro-batches instead of every step, amortizing that fixed per-apply overhead over
4× as many cadences; the cost is the persistent gradient accumulator's VRAM (3.21 → 9.91 GB/GPU). At
the default batch 128, training uses ~2.8 GB/GPU — comfortably within the 16 GB budget.


## Input-pipeline audit (#276)

`bench_input_pipeline.py` closes the gap `bench_gpu.py` documents (synthetic in-memory tensors,
no input path): it synthesizes an on-disk round of the production layout and measures the real
memmap -> tf.data -> distribute -> train-step path, with a verbatim replica of the pre-#276
single-generator builder (`--variant legacy`) for before/after, a `--gil-load N` knob running N
background threads doing `write_injection_stat`-shaped pure-Python work (the #277 drainer flood,
in proxy form), and a `--profile` TF-profiler hook. Numbers below from blpc3 (5x RTX PRO 6000,
32 cores, NGC 25.02, July 2026; 20k-sample synthetic round, page-cache resident -- the same
regime the live-run audit measured: `wa=0`, `bi=0`, round resident in RAM).

### Throughput ladder (idle host)

| measurement | legacy | current (#276) |
|---|---|---|
| raw numpy gather (single thread, no tf.data) | 11,815 cad/s (**6.97 GB/s**)¹ | — |
| iterate, 1 replica, batch 128 (tf.data only, no GPU) | 4,790 cad/s | 3,031 cad/s |
| **step: real VAE train step, 5 GPUs, accum 12** | **1,022 cad/s** | **1,409 cad/s (+38%)** |
| step, current + `--tf-deterministic-ops` | — | 1,307 cad/s (−7%) |

¹ **Per-cadence gather volume is 3× the memmap element**: every training sample pulls one
`main` + one `true` + one `false` cadence out of the three round memmaps, so one cadence costs
`3 × 6 × 16 × 512 × 4 B = 590 KB`, not 197 KB. Every GB/s figure in this section uses the 3×
volume — miss it and the throughput numbers come out 3× low (this is easy to do; a reviewer of
this very section did it one paragraph after flagging the risk).

### TF profiler: the GPUs are idle >90% of the time

XPlane compute-stream occupancy over the timed steps (the authoritative host-vs-device
measurement issue #276 asked for):

| | legacy | current (#276) |
|---|---|---|
| wall window for identical GPU work | 75.66 s | **57.25 s (−24%)** |
| GPU compute time (mean per GPU) | 4.72 s | 4.72 s |
| **mean GPU compute occupancy** | **6.3%** | **8.3%** |
| GPUs idle | 93.7% | 91.7% |

Two things fall out. First, **the per-GPU kernel time is identical between variants** (4.72 s for
the same 120 micro-batches) — the fix changes only how long the GPUs wait, not the work they do.
Second, that kernel time implies **3,253 cad/s per GPU** of pure compute, within 9% of
`bench_gpu`'s independently measured 2,986 cad/s synthetic figure. The GPUs are doing exactly the
work the synthetic benchmark predicts; they simply sit idle for >90% of the wall clock waiting on
the host. (Live-run `nvidia-smi` sampling reported 20–25% "utilization"; that counter registers a
sample as busy if *any* kernel ran during it, so it reads high against true kernel occupancy —
the two observations agree.)

### GIL dose-response (end-to-end 5-GPU step throughput)

| background Python threads | legacy | current (#276) | current advantage |
|---|---|---|---|
| 0 | 1,022 cad/s | 1,409 cad/s | **+38%** |
| 1 | 468 cad/s (−54%) | 526 cad/s (−63%) | +12% |
| 2 | 92 cad/s (−91%) | 82 cad/s (−94%) | **−11% (crossover)** |

### Conclusions

Graded by what the data actually supports.

1. **GIL contention is the dominant mechanism — measured end-to-end, not inferred.** One
   background thread doing drainer-shaped Python work costs **54%** of legacy end-to-end training
   throughput; two threads cost **91%** (an 11x collapse). This is the #276 x #277 interaction,
   and it is the strongest signal in the audit. *Caveat, stated plainly:* the hog's intensity is
   synthetic and was not calibrated against the real drainer's burst rate, so this establishes
   the mechanism and the pipeline's sensitivity to it — not the exact magnitude attributable to
   the live run. #277's bulk write API removes ~all of that per-row Python call volume, which is
   why the two fixes belong together.
2. **The single-threaded memmap copy is the NEXT wall, not the current one.** To be precise
   about the order of limits: at the throughput actually observed (1,409 cad/s = 0.83 GB/s of
   gather) the copy is running **8.4x under** its own 6.97 GB/s capability — it is emphatically
   *not* what limits the pipeline today; the GIL is. But the trace's own kernel timings put a
   *fully fed* 5-GPU consumer at 16,265 cad/s = **9.59 GB/s** of demand, and one thread supplies
   only **0.73x** of that. So the copy is co-limiting **at saturation**: it is what you hit after
   removing the GIL ceiling, not before. Both statements matter — the first says don't optimize
   the copy now, the second says don't assume it will scale when you do.
   > **Correction.** An earlier revision of this section claimed the copy had "~2.5x the volume
   > the 5-GPU consumer needs" and was therefore "not the bottleneck". That figure divided the
   > raw gather rate by the *legacy pipeline's own delivered rate* — circular, since that rate is
   > the thing under evaluation — and the conclusion did not follow. Corrected above against the
   > profiler-derived ceiling.
3. **The #276 fix's benefit is contention-dependent, and it inverts under heavy load.** +38% on
   an idle host, +12% with one competing Python thread, and **−11% with two** — because
   `tf.numpy_function` re-enters the interpreter, so the parallel-map workers contend for the very
   GIL they are meant to route around. It is still the right change to land: with #277 in the same
   PR the drainer flood is gone, which puts production near the 0–1 thread regime where the fix
   wins. But the durable lever is removing Python from the hot path, and a future gather that
   never returns to Python (pure `tf.data` ops or a C-level reader) would be immune to this
   entirely. **Do not port the parallel map into a Python-heavy process without re-measuring.**
4. **`--tf-deterministic-ops` costs ~7%** end-to-end (deterministic kernels + ordered `tf.data`)
   — the price of bit-exact reproducibility, now quantified.

**Not measured** (honest gaps, for whoever picks this up): `py-spy` attribution on a live training
process, and the ~700k context-switches/s the issue reported — the proxy hog reproduces the
*effect*, but neither was attributed on the real process. Both want a real (small) training run on
an otherwise idle cluster.

## Input-pipeline follow-up: removing Python from the hot path

The audit above ends by naming the durable lever — "a gather that never returns to Python" —
and this follow-up lands it, together with the other host-side wall the profiler exposed: the
training loop itself. Everything below was measured on blpc3 (5x RTX PRO 6000, NGC 25.02, July
2026) with the same synthetic round, `--num-gpus 5 --accumulation-steps 12`, so the numbers are
directly comparable to the tables above.

What changed (see `prepare_distributed_train_dataset` and `_build_accumulated_train_step`):

1. **Graph-side gather.** The index generators stay in Python (tiny, and they carry the #49
   rng contract), but the gather is now a deterministic parallel `.map()` of pure `tf.gather`
   ops over zero-copy dlpack tensor views of the round memmaps (`_as_cpu_tensor`;
   `load_round_arrays` opens mmap_mode="c" so the mapping is dlpack-exportable while the files
   stay write-protected). No `tf.numpy_function`, no interpreter re-entry, no GIL.
2. **Per-replica elements + device prefetch.** Each index-stream element is one per-replica
   batch handed out via `distribute_datasets_from_function` (consecutive elements -> consecutive
   replicas, reproducing the old contiguous global-batch split exactly), with
   `InputOptions(experimental_fetch_to_device=True, experimental_per_replica_buffer_size=2)` so
   host->device copies overlap compute instead of serializing inside the step.
3. **Graph-side accumulation loop.** One `tf.function` per optimizer step: K micro-batches
   accumulated into per-replica ON_READ accumulators inside a `tf.range` loop, ONE cross-replica
   reduction per variable, in-graph NaN/Inf guard, clip, apply, reset. The interpreter is
   re-entered once per optimizer step instead of once per micro-batch plus one eager op per
   variable per micro-batch (the previous loop also launched one all-reduce per variable per
   micro-batch and per-variable NaN checks that forced device syncs).

### Step-throughput ladder (5 GPUs, accum 12, idle host)

| configuration | cad/s | vs pre-follow-up |
|---|---|---|
| numpy_function gather + Python loop (the audited #283 state) | 1,453 | 1.0x |
| + graph accumulation loop only | 1,781 | 1.2x |
| + pure-TF gather (global-batch split) | 9,262 | 6.4x |
| + per-replica elements & fetch-to-device (unrolled K) | 12,618 | 8.7x |
| **as shipped: tf.range loop, epoch-table index yields (bounded VRAM)** | **12,317** | **8.5x** |
| bench_gpu synthetic ceiling x 5 (no input, no all-reduce)¹ | 14,536 | 10.0x |

¹ `bench_gpu --mode train --num-gpus 1 --accumulation-steps 12` measured 2,907 cad/s/GPU;
multi-GPU bench_gpu numbers are unavailable on blpc3 (its collective-ops path aborts with an
NCCL "unhandled cuda error"; the pipeline's `strategy.reduce` path is unaffected). A later
fresh-eyes audit measured tighter bounds — see "Corrected ceiling decomposition" below: the
true zero-communication bound is 15,112 cad/s and the shipped configuration sits at ~85% of
it, with the residual attributed to MirroredStrategy lockstep (7.4%) and input h2d/scheduling
interference (7.6%), NOT to the `tf.range` loop (which at 1 GPU actually beats bench_gpu's
unrolled step, 3,015 vs 2,907 cad/s). The unrolled-vs-range VRAM tradeoff stands on its own:
23.4 GB/GPU at K=12 unrolled does not fit the 16 GB A4000 release host; the tf.range loop
peaks at **8.4 GB/GPU** (autograph's while_loop still overlaps up to `parallel_iterations=10`
iterations). On bla0 the distinction is moot — its 6-GPU ceiling (~4,770 cad/s) sits far
below either variant, so the release host is GPU-bound both ways.

### GIL immunity (the #276 x #277 interaction, revisited)

| background Python threads | pre-follow-up | shipped |
|---|---|---|
| 0 | 1,453 cad/s | 12,317 cad/s |
| 1 | 526 cad/s (−63%) | 12,627 cad/s (−0%) |
| 2 | 82 cad/s (−94%) | 9,571 cad/s (−22%) |

An earlier iteration that yielded one small index array per MICRO-BATCH (instead of one
epoch table per epoch) measured a further 35% off under the two-hog regime — the
per-micro-batch generator GIL acquisitions convoy exactly when the interpreter is busiest.
The shipped epoch-table yield touches Python once per epoch, which is why the two-hog cost
collapses from −94% to −22% (and the −22% itself is against a deliberately hostile synthetic
load — see the real-writer row below).

With the REAL post-#277 DB writer flooded at maximum rate through the bulk lane
(`write_injection_stats_bulk`, segment-shaped batches; ~940K genuine rows committed during
the timed window) instead of the synthetic hog, the design measured **within 0.3% of its
idle throughput** (10,742 vs 10,776 cad/s on the prototype harness at that iteration; the
delta, not the absolute, is the result). The same real-writer flood against the new gather
paired with the OLD Python training loop cost ~3% (9,076 vs 9,353), which bounds the
production writer's whole GIL footprint at a few percent even without the graph-side loop —
the writer spends most of its time inside sqlite's C code with the GIL released, exactly as
#277 intended.

### Profiler evidence

XPlane trace over the timed steps of the shipped configuration
(`benchmarks/parse_xplane_occupancy.py`, merged per-GPU event intervals):

| | legacy | #283 as audited | shipped follow-up |
|---|---|---|---|
| mean GPU occupancy | 6.3% | 8.3% | **73.7%** |

Method note: the audited-state figures counted compute-stream kernel time; the follow-up
figure merges all GPU-plane events (kernels + the copies that now overlap them), so it reads
marginally high in comparison — the honest cross-check is the throughput ratio, and
12,317 / 1,446 = 8.5x against 73.7 / 8.3 = 8.9x agrees to within the method delta.

### Corrected ceiling decomposition (fresh-eyes audit)

An independent audit of the shipped state re-measured the bounds and corrected two
attributions an earlier revision of this section made:

- **True compute bound: 15,112 cad/s** — five concurrent INDEPENDENT single-GPU processes
  running the production-shaped tf.range step with zero input (not 14,536 = 5x bench_gpu,
  whose unrolled single-GPU step is itself 3.7% SLOWER than the shipped graph loop:
  2,907 vs 3,015 cad/s). The "~15% tf.range penalty" a prior revision claimed does not
  exist at the step level.
- **Residual split, both measured**: single-process MirroredStrategy lockstep + reduce/apply
  costs 7.4% (zero-input in-process bound 13,997), and the real input pipeline a further
  7.6% (12,850 mean) — h2d/scheduling interference, not gather bandwidth (the producer alone
  sustains 4x the demand). 12,850 / 15,112 = **85% of ceiling at the step level.**
- **Null results, pinned so they are not retried**: packing all 46 gradients into one flat
  37.2 MB collective — no change; HierarchicalCopyAllReduce and ReductionToOneDevice — both
  lose to NCCL; per-replica device buffer 2→6 — within run noise; XLA jit of the micro-batch
  fn — −11% AND 26.5 GB/GPU (breaks the A4000 constraint); no thermal/concurrency penalty
  (the 5-process bound proves it).
- The remaining step-level headroom within the fp32 + MirroredStrategy + 16 GB constraints is
  ~15% with no identified software lever; the levers that WOULD move the ceiling (bf16
  mixed precision, fp16 round arrays halving gather volume) change numerics and are recorded
  as candidate follow-up issues, not taken here.
- **Provenance caveat**: JSONs under `benchmarks/results/` from different working-tree
  states carry identical params blocks (e.g. `audit2_step_current.json` at ~1,400 cad/s is
  the PRE-follow-up tree; `final_step_current.json` at 12,317 is the shipped tree) — check
  file dates against the narrative here before comparing.

### Real-run A/B (the measurement the original audit called "the single most useful missing number")

Three identical scaled-down real training runs on blpc3 (2 rounds x 30 epochs,
`--num-samples-beta-vae 48000 --num-samples-rf 9600 --seed 11`, full producer + DB writer +
monitor stack), differing only in the checked-out training code:

| per-epoch wall (s) | baseline (#283 tip) | follow-up, per-epoch iterators | follow-up, round-scoped iterators |
|---|---|---|---|
| train, round 1 (overlapped with round-2 generation) | 56.5 | 35.9 | **10.9** |
| train, round 2 (quiet) | 48.5 | 27.7 | **5.8** |
| validation | 16.7 | 24.3 | **1.7** |

Two findings beyond the headline **8.4x quiet / 5.2x contended** epoch speedup:

1. **The middle column is the trap this table exists to record.** The first follow-up run
   kept the old one-`iter()`-per-epoch structure and captured only 1.75x — because
   `iter()` on a 5-GPU distributed dataset costs **~9.1 s per call** (measured directly;
   stable across calls, and tf.function trace counts were probed and stay flat, so it is
   iterator/device-pipeline construction, not retracing), and the old loop paid it twice per
   epoch while also discarding cross-boundary prefetch. The fused val loop even regressed
   (16.7 -> 24.3 s) since a fresh iterator dominated its now-tiny compute. Round-scoped
   iterators (one per dataset per round, epoch boundaries purely step-counted) removed it.
2. **At this test scale the producer becomes the wall**: round-1 training (5.5 min) now
   finishes long before round-2 generation (~38 min), so the run idles in `await_round`.
   At production scale (100 epochs x 52 steps) training still dominates and the overlap
   behaves as designed — but any future scale-down benchmarking should expect the wait.

The quiet-round number closes the loop against the synthetic benchmark: 38,400 cadences in
5.8 s minus the once-per-epoch latent snapshot puts the real pipeline at the bench's
~10.7k cad/s steady state — the harness and the production loop now agree.

**Post-A/B addendum (fresh-eyes audit follow-through).** After the third leg, the audit's
decomposition of the remaining 5.8 s showed the once-per-epoch latent snapshot costing
~1.5 s, of which only ~0.15 s is encode: ~0.9 s was a fresh `iter()` on the distributed viz
dataset per capture (the same per-epoch-iterator disease fixed above for train/val), ~0.33 s
a `gc.collect()` per capture, and ~0.12 s per-row DB payload serialization (3,840 rows). All
three are now fixed (round-scoped viz iterator, gc skipped for persistent-iterator callers,
one bulk DB call per capture) — semantics pinned by unit tests. The epoch-wall effect
(predicted from the isolated measurements, not re-run at A/B scale: quiet train epoch
5.8 → ~4.4 s, and ~7-9 s saved per production-scale epoch where the snapshot fires up to
6x) should be treated as an estimate until the next real run reports its `train_duration`
stats.

### Caveats

- The synthetic-hog dose-response is the same deliberately-hostile proxy as the audit's; the
  real writer's bulk lane spends most of its time in sqlite's C code with the GIL released, so
  the hog rows overstate production contention (that is what makes the immunity result strong).
- Tracing the accumulated-step graph is itself Python and competes for the GIL once per run;
  the tf.range loop keeps that trace small (~seconds). An unrolled K=12 variant traced for
  minutes under GIL load — a second reason it was rejected.
- Same-seed training results are NOT bit-identical to the previous implementation (gather
  backend and float summation order changed); determinism itself is preserved — same seed, same
  split, same per-epoch batch order, byte-identical reruns — and is pinned by unit tests
  (`test_train_datasets.py`, `test_train_accumulation.py`, `test_train_distribution.py`).

## Producer/data-generation follow-up

With the training hot path Python-free, the wall moved. The real-run A/B above already showed
it at test scale (round-1 training finished long before round-2 generation), and a killed
production-scale run on blpc3 confirmed it live: ~6.2 h of data generation per round with the
GPUs at 0% and all 32 pool workers at ~94.5% CPU, ~97% user-mode — compute inside the workers,
not IO or scheduling. `bench_datagen.py` is the harness this follow-up was gated on: it drives
the real pooled `generate_round_to_memmap` path (shared-memory plates, `_init_worker`, per-task
seed derivation, batched memmap tasks) at reduced scale and sha256s every output array, so a
generation-path change lands only if its checksums match master's byte-for-byte. `--preload-tf`
imports TensorFlow (CUDA-blanked) into the parent before the pool forks, mirroring the import
graph the production producer's workers inherit — without it the gc/interpreter costs read
optimistically low against exactly the regime the live run measured.

Three mechanisms shipped, all byte-identical by construction and by checksum
(`data_generation.py`):

1. **The per-injection `gc.collect()` in `new_cadence` is gone.** A full generational
   collection ran once per injection — ~2.5M times per production round — inside pool workers
   carrying the full TF import graph, and measured ~23 ms per call against ~4.5 ms for the
   entire rest of the function: the dominant term of the generation wall. Refcounting already
   frees the setigen Frame's arrays immediately; the per-chunk collect in
   `generate_round_to_memmap` stays for cycle cleanup.
2. **Draw-first `create_true_double`.** The intersection-acceptance test is a pure function of
   the pre-array RNG draws, but the retry loop materialized BOTH full setigen injections before
   testing — at the measured p≈0.42 acceptance, ~41% of all injections in a production round
   were computed and discarded. `new_cadence` is split into `_draw_signal_params` (consumes the
   exact 4-draw RNG sequence) and `_inject_drawn_signal` (consumes no RNG); the retry loop now
   replays the identical per-attempt draws and materializes only the accepted pair — same
   order, same values, pinned by byte-compat tests in `test_data_generation.py`.
3. **One msync per array, not per task.** `_run_memmap_task`'s finally flushed its whole
   mapping once per task — ~23,400 concurrent full-mapping msyncs per production round, a
   driver of the #117/#118 chunk-tail stragglers. Replaced by one flush per output array
   immediately before the `.done` manifest. The durability contract is unchanged: no array
   bytes are trusted until the manifest exists, and the manifest is written strictly after
   every array is msync'd.

### A/B ladder (blpc3, 8192 samples × 3 arrays, 32 workers, `--preload-tf`, seed 11)

| arm | wall | speedup | sha256s (all 7 outputs) |
|---|---|---|---|
| A: master | 282.0 s | 1.0× | baseline |
| B: + gc removal | 18.6 s | 15.2× | identical to A |
| C: + draw-first (as shipped) | 13.1 s | **21.5×** | identical to A |

The msync change lands with arm C — it alters no bytes, only when they are flushed, and is
covered by the same checksum gate. Projected to production scale on the same 32-core host:
~6.2 h/round → **~20 min/round**. That is a projection from the benchmark ratio, not a measured
production round — the next full-scale run's `data_generation` stage timers are the check.

**Deferred with rationale (do not silently resurrect):** RF-dataset pre-generation on the
producer, a direct-numpy injection bundle (bypassing setigen Frames), SHAP-stage overlap, pool
thread-pinning, and fused moment computation — the 21.5× result collapsed their absolute value
to ~minutes each.

**Related fix from the same audit, outside the generation path:** `plot_injection_stats` issues
~165 round-scoped queries per call against an index that leads with `(tag, timestamp)` and does
not contain `round_number`, so its run-wide time window re-scanned the tag's whole row history
on every query — measured **10.5× slower at 12M rows**, and quadratic over a campaign as
history accumulates. `Database.query_injection_stat_time_span` (one whole-partition MIN/MAX
aggregate; a superset bound, so intersecting it with a query window can never change a result
set) now tightens the window to the plotted rounds' actual span — see `docs/DATABASE.md`.

## Latent-GIF audit (#278)

`bench_latent_gif.py` decomposes the ~24–29 h `vae_plots` GIF tail and gates every candidate
optimization on output equality (issue #278 requires the produced GIFs unchanged). blpc3
numbers (48 frames × 23,040 obs-level points, fit pool 100k, n_neighbors 15):

| phase | baseline | optimized | output identical? |
|---|---|---|---|
| render (pre-#278 per-frame figure loop) | 7.25 s (0.15 s/frame) | pool 1 worker: 4.81 s; **32 workers: 1.01 s (7.2×)** | **yes — byte-identical PNGs** |
| UMAP fit (direct) | 62.6 s | precomputed-knn reuse: 3.3 + 38.6 s | **NO** (max abs diff 13.8) |
| transforms (500-call serial) | 213 s | one batched call: 193 s (−9%) | **NO** (max abs diff 11.6) |
| GIF assembly (imageio) | 1.6 s | — | — |

Only the render parallelization shipped (`aetherscan.latent_gif`, byte-identical by
construction and pinned by test). Precomputed-knn reuse and batched transforms are **rejected
with numbers**: both change how UMAP consumes its `random_state` stream, so they produce a
*different* (equally deterministic) embedding — a violation of #278's output-identity
constraint for a ~9% / ~24 s-per-fit saving. The two other UMAP sites #278 flags land at
benchmark scale: decision-boundary fit 10.2 s + inverse-transform grid 29.4 s per combo, and
SHAP-space UMAP 9.4 s + KMeans 4.1 s — minutes per run, not hours; deferred.
