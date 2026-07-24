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
        2. vae_plots    — final loss/stability/injection plots, latent GIF, latent traversal
        3. rf_train     — generate RF data → encode latents → fit RF → persist model+artifacts
        4. rf_plots     — the ten RF diagnostics
        5. final_save   — final models + config_{tag}.json
        6. hf_upload    — publish artifacts + model card to the HuggingFace Hub (opt-in)
```

Stages 1/3/5 are critical (failures raise → retry); 2/4/6 are non-critical (failures are
recorded in the manifest and retried on the next run, but never cost a data regeneration).
Stage 6 (`hf_upload`) runs only when `config.hf.upload_after_training` is set — a failed
upload is recorded but never fails the run, since the weights are already safe locally.
Every stage is skipped if the persisted manifest (`run_state_{tag}.json`) already records it
as done, so re-running the identical command resumes exactly where the last attempt died.

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
5. **Epoch loop** — `_train_epoch()` + `_validate_epoch()`, ~21 `training_stats` rows per
   epoch (losses, gradient norms, LR, durations, SNR range), adaptive LR update.
6. **Per-round plots** — loss curves, training stability, injection stats (tagged
   `round_XX`, saved under `plots/training/{save_tag}/checkpoints/`), plus the latent traversal when
   `--latent-traversal-every-round` is set.
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
`(499200, 6, 16, 512)` float32 ≈ 98 GB each — ~294 GB per round. Holding that in RAM is what
used to OOM-kill 503 GB training nodes; instead each round lives on disk under
`{round_data_dir}/{save_tag}/round_{k:02d}/` (default root `{data_path}/training/round_data`):

```
round_02/
├── main.npy  true.npy  false.npy        # (n, 6, 16, 512) float32 memmaps
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
  (`.tmp` → `os.replace`) only after every chunk finishes; it records shapes, SNR params, and
  cheap sampled checksums. `validate_done_manifest()` re-checks all of it — a directory
  without a valid manifest is garbage and gets regenerated.
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

`prepare_distributed_train_dataset()` builds infinite generator-backed `tf.data` datasets that
yield **whole global batches** (`per_replica_batch_size × num_replicas` rows gathered from the
memmaps by fancy indexing), with a leading batch dimension in the output signature and no
`.batch()` call — cutting per-sample Python boundary crossings by the global batch size,
which is what fixed the historical 0–14 % GPU utilization. Randomness lives at the epoch
level (train indices reshuffled per pass); within a batch, indices are sorted for memmap read
locality (the model is order-invariant within a batch).

The train/val split is **stratified** over the four signal types (generation lays labels out
contiguously per chunk, so a positional split would skew val), then trimmed to exact multiples
of `effective_batch_size` (train) and the global val batch size. With `shuffle=False`
(used by RF training) the yield order is pinned to the returned `train_indices`/`val_indices`
— the alignment contract that lets encoded latents be matched back to labels.

### Gradient accumulation

Each training step accumulates `accumulation_steps = effective_batch_size /
(per_replica_batch_size × num_replicas)` micro-batch gradients before applying
(`_train_epoch()` → `_distributed_train_step()` → `_apply_gradients()`), giving an effective
batch of 7680 at defaults (chosen so it divides evenly on 4-, 5-, or 6-GPU hosts) regardless of
per-GPU memory. **Note:** 7680 is ~2.5× the previous 3072, so there are ~2.5× fewer optimizer
updates per epoch; LR-schedule behavior calibrated to the old cadence may differ. On a fixed 4- or
6-GPU host you can pass `--effective-batch-size 3072` to restore the old cadence (it stays valid
there). Guards along the way: all-None
gradient micro-batches are skipped, accumulated gradients are averaged over successful
micro-steps, NaN/Inf gradients raise immediately, and the global gradient norm is clipped at
1.0 with the pre-clip norm recorded per step (that's the `clipping_rate` statistic).

The divisibility preconditions (`effective_batch_size % (per_replica × replicas) == 0`, sample
counts divisible by batch sizes, etc.) are validated up front by
`cli.py:collect_validation_errors` — see [`CONFIG_AND_CLI.md`](CONFIG_AND_CLI.md) for the
cross-replica constraint system and the fix proposer.

### Adaptive learning rate

`_update_learning_rate()` tracks validation total loss: if it fails to improve by
`min_pct_improvement` (0.1 %) for `patience_threshold` (3) consecutive epochs, the LR is
scaled by `1 − reduction_factor` (×0.8), floored at `min_learning_rate` (1e-6). The LR resets
to `base_learning_rate` (1e-3) at each round start. Rule of thumb from the docstring: the LR
can only bottom out within a round if
`base_learning_rate · (1 − reduction_factor)^(epochs_per_round / patience_threshold)`
reaches `min_learning_rate`.

## Reproducibility

Training randomness is reproducible only behind an opt-in root seed; with no seed the pipeline
draws from OS entropy and runs are non-reproducible (the historical default).

- **`--seed`** (mirrors `config.training.seed`; `int | None`, default `None`, must be `>= 0`).
  When set it seeds every training-side source of randomness: synthetic data generation
  (per-round worker-task seeds derived from `(seed, round_number)`, identical on the
  background-producer and the sequential in-process paths), the dataset split/trim/per-epoch
  train shuffles (`prepare_distributed_train_dataset`, one stream per round), latent-visualization
  batch selection and padding, injection-bias plot subsampling, and the TensorFlow global RNG via
  `tf.random.set_seed` (fixing `HeNormal`/`GlorotNormal` weight init and the VAE `Sampling`
  layer). `tf.random.set_seed` runs in `TrainingPipeline.__init__`, before any model is built.
- **Independently seeded surfaces** are unaffected by `--seed`: the Random Forest keeps its own
  `config.rf.seed` / `--rf-seed`, and the latent-viz UMAP/KMeans fits use fixed `random_state`s.
  The RF dataset borrows the round-`0` stream key while curriculum (beta-VAE) rounds are 1-based,
  so their streams never collide.
- **`--tf-deterministic-ops`** (`config.training.tf_deterministic_ops`, off by default) forces
  deterministic TF/cuDNN kernels via `tf.config.experimental.enable_op_determinism()`. It costs
  some training speed and is only meaningful alongside `--seed` — enabling it without a seed logs
  a warning and buys nothing.
- **Approximate vs. bit-exact.** Seeding alone gives *approximate* run-to-run reproducibility;
  *bit-exact* GPU reproducibility additionally requires `--tf-deterministic-ops` plus identical
  hardware and software.

Stream derivation lives in [`seeding.py`](../src/aetherscan/seeding.py): `derive_rng(root_seed,
*stream_key)` builds an independent NumPy `Generator` per consumer from
`SeedSequence([root_seed, *stream_key])`, so distinct keys are statistically independent and each
consumer's stream is stable regardless of what the others draw.

Both `seed` and `tf_deterministic_ops` are emitted by `Config.to_dict()["training"]`, so they are
part of the run-manifest config fingerprint: a tag started before these fields existed — or under
a different seed — cannot resume across the change under the same `--save-tag`; the guard
downgrades to a fresh run with a loud warning. See [`CONFIG_AND_CLI.md`](CONFIG_AND_CLI.md) for the
config/CLI plumbing and the [CLI Reference](../README.md#cli-reference) for the exact flag help.

## Checkpointing, the run manifest, and retries

### What gets saved when

| Artifact | When | Where |
| --- | --- | --- |
| `vae_{encoder,decoder}_round_XX.keras` | End of every round | `{model_path}/checkpoints/` |
| `random_forest_{tag}.joblib` + `rf_eval_artifacts_{tag}.joblib` | End of `rf_train` | `{model_path}/` |
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
Because the manifest is on disk, a **full process relaunch of the identical command resumes
identically** — the in-process loop and a crash-and-relaunch are the same code path.

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
false-clustering components (bottom row), train and val overlaid, epochs on the x-axis with
per-round SNR-range shading in the background. What to look for:

- Both curves trending down within each round; **val tracking train** (a widening gap =
  overfitting the current curriculum stage).
- Small upward steps at round boundaries are expected — the data gets harder; a *large*
  sustained jump means the curriculum narrowed too fast (`exponential_decay_rate` too
  negative).
- KL should settle to a moderate plateau: collapsing toward 0 means the posterior ignores the
  input (posterior collapse — consider lowering `beta`); growing without bound means the
  latent space isn't regularizing.
- True/false clustering losses should decay and stay low; if `true_loss` dominates late
  rounds, the ON/OFF separation is failing on faint signals.
- Doubled/serrated series are the signature of stale rows from a failed attempt leaking in —
  they should never appear now that resumes mark old rows superseded; if you see them, check
  the `mark_superseded` warnings in the log.

### `beta_vae_training_stability_{tag}.png`

2×3 grid: gradient **clipping rate** across the top, gradient-norm mean/std/max across the
bottom, same SNR shading. What to look for: clipping rate near zero after the first epochs
(sustained clipping = LR too high for the stage); norm mean smooth and slowly decaying;
isolated max spikes are fine, but spikes that coincide with loss cliffs point at bad batches
or an injection bug. NaN/Inf gradients abort the epoch outright, so anything you see here was
at least finite.

### Injection-stats figures (from `plot_injection_stats`, 8 PNGs)

Bias/leakage analysis of the synthetic data itself, sourced from the `injection_stats` table
(intensity statistics captured at stage **A** = raw background, **B** = post-injection,
**C** = post-normalization; see [`PREPROCESSING.md`](PREPROCESSING.md)):

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
share `rf_shap_values_{tag}.joblib` (computed once, cached).

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
| `rf_latent_decision_boundary_nn{n}_md{m}_{tag}.png` | RF P(true) contour over each persisted cadence-level UMAP plane, val points + 0.5 contour. | A coherent boundary separating the true classes; ragged islands = the forest partitioning noise. Depends on the UMAPs from `plot_latent_space_gif`, so `vae_plots` must have succeeded. |

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
shut down). It reuses `prepare_distributed_train_dataset(shuffle=False)`, encodes train and
val cadences through the (frozen) encoder with `_distributed_encode` — note the
`train_steps × accumulation_steps` step-count correction, guarded by an exact-count assertion
— fits the RF on binary labels (`true_*` vs `false_*`), and persists the model plus the eval
artifacts immediately so a retry can skip straight to plots. A `check_encoder_trained()`
heuristic (weight-std deviation from initializer expectations) guards against accidentally
encoding with untrained weights and falls back to loading the newest checkpoint.

At the tail of the stage, `train_random_forest()` also persists scalar RF eval metrics
(accuracy, ROC-AUC, average precision, Brier score, per-sub-type accuracies, binary +
sub-type × prediction confusion cell counts, val P(true) quantiles) to `training_stats`
under `model_name='rf'` via the pure (TF-free) helper
[`compute_rf_eval_metrics()`](../src/aetherscan/rf_metrics.py); the deployment
`inference.classification_threshold` used to derive `val_accuracy` is written alongside as
its own `classification_threshold` row. `plot_rf_ensemble_accuracy_curve()` (in the
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
| Batching | `per_replica_batch_size`, `effective_batch_size`, `per_replica_val_batch_size` |
| Round data | `round_data_dir`, `overlap_data_generation`, `keep_round_data`, `signal_injection_chunk_size`, `data_gen_task_size` |
| Curriculum | `snr_base`, `initial_snr_range`, `final_snr_range`, `curriculum_schedule`, `exponential_decay_rate`, `step_easy_rounds`, `step_hard_rounds` |
| Adaptive LR | `base_learning_rate`, `min_learning_rate`, `min_pct_improvement`, `patience_threshold`, `reduction_factor` |
| Latent viz / traversal | `latent_viz_*`, `latent_traversal_every_round`, `latent_traversal_num_steps`, `latent_traversal_max_sigma` |
| RF plots | `shap_max_samples_*`, `shap_top_k_features_dependence`, `rf_decision_boundary_*` |
| Fault tolerance | `max_retries`, `retry_delay` |
