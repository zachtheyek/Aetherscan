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
# On the clusters, run through the container:
./utils/run_container.sh python benchmarks/bench_normality.py
```

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
