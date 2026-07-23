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

The numbers above are single-GPU on blpc3. Multi-GPU scaling (`--num-gpus`), the
`--accumulation-steps` cadence, and the bla0 (6× A4000 16 GB) baselines are captured on the
release-benchmark runs and land with the System-Requirements update (#183).
