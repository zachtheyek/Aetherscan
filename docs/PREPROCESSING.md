# Preprocessing

This document covers everything that happens to data before a model sees it, on both sides of
the pipeline: **energy detection** (inference: raw `.h5` filterbank files → stamp `.npy`
arrays), including the bandpass-flattening math (spline vs PFB) and the vectorized
D'Agostino–Pearson normality test, and **signal injection** (training: real backgrounds →
labeled synthetic cadences), plus the background/stamp loading paths. Code:
[`src/aetherscan/preprocessing.py`](../src/aetherscan/preprocessing.py),
[`src/aetherscan/pfb.py`](../src/aetherscan/pfb.py),
[`src/aetherscan/data_generation.py`](../src/aetherscan/data_generation.py).

## Energy detection (inference side)

Goal: reduce a cadence of six ~1-billion-bin observations to a few thousand
`stamp_width`-wide frequency windows worth encoding. Detection runs on the **ON-source**
observations only (ABACAD positions 0/2/4); the hits found there define the slices extracted
from all six observations, so the models see the full ON/OFF context at each hit frequency.

Per cadence (`DataPreprocessor._process_cadence`), the chain is:

```
for each ON file:                      (one fused pool task per coarse channel)
    read h5 slice → remove DC spike → flatten bandpass (pfb | spline)
    → vectorized k² over sliding windows → threshold → per-channel hit list
aggregate hits → deduplicate → build stamp centers (± overlap offsets)
→ parallel memmap stamp extraction (downsample-at-extraction) → .npy + metadata .json
```

One **persistent worker pool** (started once per run by `start_energy_detection_pool()`)
serves every channel of every file of every cadence, and the extraction stage reuses it. Each
task returns only a small hit list plus a fixed-size statistic histogram — the bulky
`(time_bins, coarse_channel_width)` intermediates never leave the worker, so there is no
shared memory and no block-sized array in the parent at all
(`_energy_detect_channel_worker`).

### Coarse channels and the DC spike

Filterbank files are processed in **coarse channels** of `inference.coarse_channel_width`
(default 1 048 576) fine bins — the natural unit of the telescope backend's channelization
(`n_coarse = nchans // coarse_channel_width`; every complete coarse channel is processed).
Each coarse channel carries a 2-bin **DC spike** at its center, an FFT artifact of the
backend. `_remove_dc_spike` interpolates over it in place, pulling replacement values from
bins ±2/±3 away (the immediate neighbors are themselves contaminated; the asymmetric offsets
match the reference implementation byte-for-byte). The interpolation reaches at most ±3 bins
around the channel center, so per-channel processing touches exactly the same bins the
historical block-based implementation did.

### Bandpass flattening — spline vs PFB

The backend's polyphase filterbank imprints a scalloped passband shape onto every coarse
channel. Left in place, that shape — not astrophysics — would dominate the normality
statistic, so each channel is flattened first. The stage is pluggable
(`_get_bandpass_flattener()` returns a picklable callable shipped to the workers); two
implementations exist, selected by `--bandpass-method`:

**Spline (`spline`, the historical data-driven method).** For a channel
`X ∈ ℝ^{time_bins × W}`:

1. Time-integrate: `x̄ = mean_t X[t, :]` (shape `(W,)`).
2. Fit a cubic smoothing spline through `x̄` with interior knots every
   `W // spline_order + 1` bins (`spline_order` default 16, i.e. ~16 knots across the
   channel — smooth enough to follow the passband, too stiff to follow narrowband signals):
   `scipy.interpolate.splrep(x, x̄, t=knots[1:])`.
3. Evaluate the spline at every bin and **subtract** it from every time row:
   `X_flat = X − spl(x)`.

Cost: one spline fit per coarse channel per file, every file.

**PFB static equalization (`pfb`, the default).** The passband shape is a property of the
instrument's prototype filter, not of the data — so compute it once from the filter design
and **divide** it out ([`pfb.py`](../src/aetherscan/pfb.py), a native NumPy port of the bliss
reference implementation):

1. **Prototype filter** (`firdes`): a windowed-sinc lowpass with
   `N = taps_per_channel × num_coarse_channels` taps and cutoff `f_c = 1/num_coarse_channels`
   (critically sampled PFB):
   `h[n] = sinc(f_c · (n − (N−1)/2)) · w[n]`, `w` the Hamming window
   `0.54 − 0.46·cos(2πn/(N−1))`.
2. **Response** (`gen_coarse_channel_response`): zero-pad `h` to
   `num_coarse × fine_per_coarse` points; take the power spectrum
   `|fftshift(FFT(h_pad))|²`; drop half a fine channel from each end so the remaining band is
   an integer number of coarse-channel spans centered on channel boundaries; reshape to
   `(num_coarse − 1, fine_per_coarse)` and **sum across spans** — the fold adds in
   adjacent-channel leakage, which is why the method needs ≥ 2 coarse channels (files with one
   fall back to spline with a warning); normalize to peak 1.0. The result `H` (shape
   `(fine_per_coarse,)`) is `lru_cache`d per parameter tuple.
3. **Equalize** (`equalize_passband`): `X_flat = X / max(H, 10⁻¹⁰)` broadcast over time (the
   floor guards against a pathologically sharp filter design driving edge bins to zero).

At GBT scale the one-time FFT is an ~`n_chans`-point transform with a tens-of-GB transient,
so it runs **once in the parent**, is persisted to a content-addressed sidecar
(`{output_path}/pfb_cache/pfb_response_w{W}_c{C}_t{T}.npy`, atomic write), and workers just
read the ~8 MB file (cached per process by `_load_pfb_response`). Afterwards, flattening a
channel is a single vectorized divide — versus a fresh spline fit per channel per file.

**Why subtract-vs-divide doesn't change detection.** The spline path subtracts its fit; the
PFB path divides by `H`, leaving each channel its DC offset and a bin-dependent scale. The
detection statistic below is built from skewness and kurtosis — *central* moments (location
cancels) normalized by powers of the variance (scale cancels) — so only the bandpass *shape*
matters, and both methods remove exactly that. `taps_per_channel` (default 12, the
GBT/Breakthrough Listen value) is **instrument-dependent** and must match the backend that
produced the `.h5`; a cheap residual-flatness sanity check (`_warn_on_pfb_response_mismatch`)
flattens several sampled channels of each cadence's primary ON file with the active response
and compares the median flattened edge/mid power ratio (`pfb.edge_mid_power_ratio`) against
1.0 — i.e. it asks directly whether dividing by `H` actually flattens the data — warning once
per file when the deviation exceeds ~10 %. The warning is **informational**: the residual
still carries analog-frontend tilt and edge RFI the response doesn't model, so only a large
or consistent deviation across files points at a wrong tap count (fallback:
`--bandpass-method spline`); the threshold stays provisional until the pfb_taps-vs-backend
characterization fixes the legitimate baseline. Only the edge/mid bands of each sampled
channel are read (float32, time-integrated on read), keeping the check cheap at GBT scale.
`--bandpass-debug-plot` saves a raw-vs-flattened overlay for visual confirmation.

**Provenance of the GBT/BL parameters.** The defaults `pfb_taps_per_channel = 12`, the Hamming
window, and the sinc prototype aren't guesses: the GBT Breakthrough Listen backend PFB was
confirmed as the CASPER `GBT512` configuration — `nchan=1024, ntaps=12, width=1.0,
window=hamming, lpf=sinc` (from
[PFBPassband.jl](https://github.com/david-macmahon/PFBPassband.jl)) — which is exactly what
[`pfb.py`](../src/aetherscan/pfb.py) models. That reference config also carries a `bug=true`
flag: the CASPER coefficient generator omits the half-sample offset from the sinc argument
(`sinc((n − N/2) / nchan)` instead of the bug-free `sinc((n + 0.5 − N/2) / nchan)`,
`N = ntaps × nchan`), so the real hardware runs a sinc mis-centered by half a sample relative
to its Hamming window while `pfb.py` models the bug-free design. That deviation is quantified
and negligible (issue #180): the folded per-channel power responses agree to ~10⁻⁵ (relative)
at a typical GBT geometry (64 coarse channels, 12 taps) — bounded below 10⁻⁴ by a unit test in
[`test_pfb.py`](../tests/unit/test_pfb.py) and 3–4 orders of magnitude under the 5 %
mismatch-warning tolerance — so `pfb.py` keeps the exact bug-free form.

### The detection statistic: vectorized D'Agostino–Pearson

Narrowband signals make the flattened residuals *non-Gaussian* within a small frequency
window. Detection therefore slides a window of `detection_window_size` (256) bins with step
`detection_step_size` (128) across each flattened channel — window *j* covering columns
`[j·step, j·step + window)` — and computes D'Agostino–Pearson's k² over each window's
flattened `(time_bins × window)` sample, flagging windows with
`k² > stat_threshold` (default 2048).

The reference statistic is `scipy.stats.normaltest`:

```
k² = Z₁(g₁)² + Z₂(b₂)²   ~  χ²(df=2) under normality
```

where `g₁ = m₃/m₂^{3/2}` (sample skewness), `b₂ = m₄/m₂²` (sample kurtosis), and `Z₁`/`Z₂`
are D'Agostino's and Anscombe–Glynn's normalizing transforms. Calling `normaltest` per window
in Python — ~8 190 windows per channel × 3 ON files × hundreds of channels, several array
allocations each — was the single largest cost of inference. `_sliding_normality_k2`
computes the identical statistic in closed form:

1. **Constant n.** Every window has `n = time_bins × window_size` samples (4 096 at
   defaults), so every n-dependent constant in `Z₁`/`Z₂` is a scalar computed once.
2. **Conditioning shift.** The channel mean is subtracted up front — central moments are
   shift-invariant, and the `S₂/n − mean²`-style differencing below loses precision when
   `|mean| ≫ std` (flattened residuals are near-zero-mean anyway; the shift makes it
   unconditional).
3. **Per-column power sums.** `p_k[c] = Σ_t X[t,c]^k` for k = 1..4, accumulated row-by-row in
   float64 (temporaries stay at `(W,)`).
4. **Block → window sums.** Columns are aggregated into blocks of size `step` (fast path,
   when `step | window` — the default 128 | 256) or `gcd(window, step)` (general path), via
   `np.add.reduceat`; window sums `S₁..S₄` are then length-`window/block` moving sums over
   the block sums, sampled every `step/block` blocks — one entry per window, no long
   cumulative accumulation.
5. **Raw → central moments.** With `μ = S₁/n`:
   `m₂ = S₂/n − μ²`, `m₃ = S₃/n − 3μ(S₂/n) + 2μ³`,
   `m₄ = S₄/n − 4μ(S₃/n) + 6μ²(S₂/n) − 3μ⁴`.
6. **Z transforms.** Elementwise transcriptions of
   `scipy.stats._stats_py::skewtest/kurtosistest` (variable names follow scipy), then
   `k² = Z₁² + Z₂²`. Zero-variance windows are forced to NaN, matching scipy (and preventing
   a tiny negative `m₂` from float cancellation fabricating a huge k²).

The per-window Python/scipy overhead collapses into a handful of large ndarray operations.
Equivalence is pinned by unit tests (`tests/unit/test_preprocessing.py`) to
`rtol = 1e-9` against `scipy.stats.normaltest` across several input distributions — since the
math is exact, hit sets on identical inputs match the historical loop **exactly**. p-values
(`chi2.sf(k², 2)`) are computed only for the hit subset (they're metadata, not selection).
Each worker also folds *all* finite window statistics into a fixed 121-edge log-spaced
histogram (`ED_STAT_HIST_EDGES`, 10⁻³–10⁹) — per-channel counts add, giving the run's full
statistic distribution for `ed_stat_distributions_{tag}.png` at negligible cost.

### Deduplication and overlap stamps

Hits from all three ON files are pooled and **greedily merged**
(`_deduplicate_hits`): hits sorted by fine-channel index; any hit within
`stamp_width // 2` bins of the previous survivor is merged into it, keeping the higher
statistic. One drifting signal (or one RFI comb tooth) therefore yields one stamp instead of
dozens of step-offset duplicates.

Each surviving hit becomes a stamp center; with `overlap_search` (default on), two extra
copies offset by `±overlap_fraction × stamp_width` (±2 048 bins at defaults) are added so a
signal drifting out of one stamp's frame is centered in a neighbor. Out-of-band stamps are
dropped; centers are sorted by start index before extraction (sequential reads let the HDF5
chunk cache reuse decompressed bitshuffle chunks across adjacent stamps).

### Stamp extraction and storage math

Extraction writes straight into a memmap-backed `.npy` of shape
`(n_stamps, 6, time_bins, stored_width)` float32, filled by pool workers over disjoint
(observation, stamp-range) slices (`_extract_stamps_worker`), with `.tmp` → `os.replace`
atomicity — the resume path treats the `.npy`'s existence as proof of a complete write.

**Downsample-at-extraction** (`store_downsampled_stamps`, default on): each stamp is reduced
along frequency with `skimage.transform.downscale_local_mean(stamp, (1, downsample_factor))`
*before* writing, so `stored_width = stamp_width // downsample_factor` (512 at defaults).
Per stamp:

```
full width:   6 obs × 16 t × 4096 f × 4 B = 1.5   MiB
downsampled:  6 obs × 16 t ×  512 f × 4 B = 0.1875 MiB   (÷ 8)
```

A cadence yielding ~10⁴–10⁵ stamps (RFI-heavy bands, with overlap ×3) thus costs ~2–20 GB
instead of ~15–150 GB, and the separate downsample pass at load time disappears. The sibling
metadata JSON records `stored_width` and `downsample_factor_applied`; the `.npy` shape
self-describes, and the loader handles both layouts (below). Disable to archive raw-resolution
stamps.

The metadata JSON also carries the full ED provenance for the visualization suite: per-stamp
starts/frequencies/statistics/p-values, raw and merged hit frequency lists, the per-ON-file
statistic histograms, and the `.h5` header.

## Loading paths

**Training backgrounds** (`load_train_data`): each file in `data.train_files`
(`{data_path}/training/*.npy`, raw width 4096) is memory-mapped and processed in chunks of
`background_load_chunk_size` (15 000 cadences): the chunk is copied into a shared-memory
block, and a pool downsamples each cadence ×8 per observation (`_downsample_worker`).
Cadences containing NaN/Inf or with non-positive max are dropped. **Log-normalization is
deferred** — training-side log-norm happens per sample *after* signal injection (injection
must operate on linear intensities). Loading stops at `num_target_backgrounds` (45 000)
cadences — about 9 GB at model resolution (45 000 × 6 × 16 × 512 × 4 B) — held in RAM for
the whole run and shared with injection workers via shared memory.

**Inference snippets** (`load_inference_data`): same chunked SHM+pool pattern, branching on
the stored width: files already at `width_bin // downsample_factor` (512 — written by
downsample-at-extraction) need only per-cadence log-norm, run vectorized in the workers
(`_lognorm_worker`); legacy full-width files (4096) keep the historical
downsample-then-log-norm path. Any other width is an error (skipped with a log). The loaded
array must survive strict [0, 1] / NaN / Inf checks before inference proceeds.
`parallel=False` forces the sequential in-process branch (no chunk pool / SHM) — the
streaming per-cadence path uses it so the loader never competes with the persistent
energy-detection pool for cores.

**Log-normalization** (`data_generation.log_norm`): `y = log(x + 10⁻¹⁰)` shifted by its
minimum and scaled by its range into [0, 1], per observation. With `return_params=True` it
also returns `(min_log, range_log)`, enabling the approximate inversion
`x ≈ exp(y · range_log + min_log)` used by the latent-traversal display
([`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)).

## Signal injection (training side)

Training data is synthetic: setigen injects drifting narrowband signals into real observed
backgrounds ([`data_generation.py`](../src/aetherscan/data_generation.py)). The atom is
`new_cadence(data, snr, ...)`, which injects **one** signal into a vertically stacked cadence
(`(96, 4096)` = 6 obs × 16 time rows):

- `starting_bin ~ U(1, width_bin−1)`; drift direction ±1 with equal probability.
- The pixel-space slope runs the signal across the full stacked height:
  `slope_pixel = total_time / starting_bin` (positive drift) or
  `total_time / (starting_bin − width_bin)` (negative), converted to physical units via
  `time_resolution / freq_resolution` plus `U(0, 3)` noise; near-zero physical slopes are
  clamped to ±10⁻⁶ (recorded via `slope_clamped` so queries can filter them) and
  `drift_rate = −1 / slope_physical`.
- `signal_width = U(0, 50) + 18·|drift_rate|` Hz — the drift-proportional term keeps fast
  drifters coherent across a time bin.
- setigen frame: `constant_path` (linear drift), `constant_t_profile` at
  `frame.get_intensity(snr=snr)` (intensity calibrated to the requested SNR against the
  actual background noise), `gaussian_f_profile(signal_width)`, flat bandpass profile.
- Realized parameters (`snr`, `drift_rate`, `signal_width`, `starting_bin`, `slope_pixel`,
  `y_intercept`) are returned as `signal_info` and written to `injection_stats`.

Signals are injected into the *stacked* array, so a drifting line naturally traverses
observation boundaries — appearing at shifted frequencies in successive observations exactly
as a real drifting source would.

### The four signal types

| `signal_type` | Builder | Construction | Semantics |
| --- | --- | --- | --- |
| `false_no_signal` | `create_false(inject=False)` | Background, log-normed as-is | Pure noise |
| `false_with_rfi` | `create_false(inject=True)` | One signal in **all 6** observations | RFI: persists in OFFs |
| `true_only_eti` | `create_true_single` | One signal; ONs take the injected rows, OFFs keep the original background | Technosignature: ON-only |
| `true_eti_rfi` | `create_true_double` | RFI signal everywhere + ETI signal in ONs, re-drawn until the two tracks don't intersect inside any ON region (`check_valid_intersection`), up to `MAX_INTERSECTION_RETRIES` (100) attempts; if the cap is exhausted the last drawn pair is kept and the sample is flagged (`intersection_retry_capped`) | Technosignature under RFI |

Each builder picks a random background plate, records **intensity statistics at three
stages** — **A** (raw background, pre-injection), **B** (post-injection, pre-normalization),
**C** (post-normalization, exactly what the model sees) — six global statistics each (mean,
median, std, MAD, skew, kurtosis, float64 to survive higher-order moments), plus the
per-observation log-norm parameters. The A/B/C staging is what powers the injection-bias
plots: A→B isolates what injection adds, B→C what normalization erases, and any statistic
that still separates the classes at stage C is leakage a model could shortcut on
(see [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)).

### Round generation layout

`generate_round_to_memmap()` fills three arrays per round — `main` (4-way balanced across the
types, the reconstruction input), `false` and `true` (2-way balanced, feeding the clustering
losses; see [`MODELS.md`](MODELS.md)) — in chunks of `signal_injection_chunk_size` (50 000,
validated divisible by 4). Each chunk decomposes into 8 contiguous class segments
(`build_chunk_segments`: 4 quarters for `main`, 2 halves each for `false`/`true`), and each
segment into batched tasks of `data_gen_task_size` cadences (`build_segment_tasks`) submitted
as **one unified `pool.map` per chunk**. Each task carries a fresh RNG seed drawn in the
parent, so results don't depend on which persistent worker picks it up. Workers write rows
directly into the round memmaps and return only stats dicts; per-segment statistics flow back
through the producer's `stats` channel and are written to `injection_stats` by the
main-process drainer ([`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)). The labels array
mirrors `main`'s contiguous per-chunk layout, which is what the stratified train/val split
keys on.
