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
```

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
| `bench_normality.py` | `preprocessing._sliding_normality_k2` vs the historical per-window `scipy.stats.normaltest` loop | Energy detection thresholding (`inference.*.read_ed_on*`) |
| `bench_injection.py` | `data_generation.new_cadence` (setigen narrowband injection into a stacked 96x4096 cadence) | Training data generation (`train.round_XX.data_generation`) |
| `bench_lognorm_downsample.py` | per-observation `downscale_local_mean` (x8) and per-cadence `log_norm` | Stamp extraction downsample + inference load (`inference.*.load_lognorm`) |
| `bench_pfb_vs_spline.py` | `pfb.equalize_passband` (static response divide) vs `preprocessing._spline_flatten_bandpass` (order-16 fit) on one 1M-bin coarse channel | Bandpass flattening inside energy detection |
| `bench_gpu.py` | Beta-VAE training step (`compute_total_loss` + gradients + clipped Adam) and encoder forward on one or more GPUs | VAE training (`train.round_XX`) and encoder inference — **GPU-only, see below** |
| `bench_rf.py` | `RandomForestClassifier` fit + `predict_proba` and the `prepare_latent_features` reshape (sklearn, CPU) | Second-stage RF training + inference (`train.train_random_forest`) |
| `bench_input_pipeline.py` | The REAL memmap → tf.data → distribute → train-step input path (gather / iterate / step modes; legacy vs current builder; `--gil-load` contention knob) — **container-only; step mode needs GPUs** | The training input pipeline `bench_gpu.py` deliberately excludes (`train.prepare_distributed_train_dataset`) — see [#276 audit](#input-pipeline-audit-276) |
| `bench_latent_gif.py` | Latent-GIF stage decomposition (UMAP fit / transform / frame render / GIF assembly) with output-equality checks on every candidate optimization — **container-only** | The `vae_plots` latent-GIF tail (`train.plot_latent_space_gif`) — see [#278 audit](#latent-gif-audit-278) |

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
accumulator and throughput reflects the once-per-K apply cadence. `--find-max` is capped at
`--max-batch 4096`: a single encoder forward whose conv feature
maps exceed 2^31 elements (batch ≳ 8192) trips an uncatchable TensorFlow int32 launch-config abort
rather than a clean OOM, and 4096 already covers the training VRAM ceiling and a generous inference
range (the pipeline's inference default is 2048).

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
this compact model, so the defaults (train 128, inference 2048) are already near-optimal — size the
per-replica batch for constraint-divisibility and convenience, not to fill VRAM.

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
