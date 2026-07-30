# Inference Pipeline

This document covers the `inference` command: the CSV catalog format and cadence grouping,
the per-cadence streaming architecture, model loading, classification-threshold semantics,
the per-cadence run manifest that makes retries seamless, and every artifact/figure an
inference run produces. Orchestration lives in
[`src/aetherscan/main.py`](../src/aetherscan/main.py)`:inference_command()` /
`_run_streaming_csv_inference()`, the GPU stage in
[`src/aetherscan/inference.py`](../src/aetherscan/inference.py), preprocessing in
[`src/aetherscan/preprocessing.py`](../src/aetherscan/preprocessing.py) (see
[`PREPROCESSING.md`](PREPROCESSING.md)), and the visualization suite in
[`src/aetherscan/inference_viz.py`](../src/aetherscan/inference_viz.py).

## TL;DR

```
inference_command() (main.py)
└── retry loop (inference.max_retries, inference.retry_delay)
    └── _run_streaming_csv_inference():
        units = plan_cadences()                     # CSV rows → cadence work units, no work yet
        skip units with a live 'inferred' manifest row for this tag
        start ED pool + first prefetch, then InferencePipeline(strategy)
                                                    # encoder + RF load once, hidden under
                                                    # the first cadence's energy detection
        for each pending cadence (prefetch depth = inference.prefetch_depth):
            [prefetch thread(s)] preprocess + load/log-norm upcoming cadences
                                 (energy detection → stamp .npy → snippet array)
            [main thread]        encode cadence i on GPUs → RF → write
                                 inference_results (positives) + inference_cadences row
        render_inference_visualizations()           # on a fully successful pass
```

Peak memory is up to `inference.prefetch_depth` in-flight cadences (stamps + loaded
arrays) plus the one being inferred — **independent of catalog size**. Two input modes:

- `--inference-files <catalog.csv>` — the production path described here: raw `.h5`
  observations grouped into cadences, energy-detection preprocessing, streaming inference.
- `--test-files <file.npy>` — legacy path for already-preprocessed arrays: one
  `run_inference_pipeline()` shot over the whole file, no manifest/streaming machinery.

## Input: CSV catalogs and cadence grouping

Each entry in `config.data.inference_files` is a CSV (resolved via
`config.get_inference_file_path`, i.e. `{data_path}/inference/<name>`) whose rows describe
individual `.h5` observations. `preprocessing.group_observations_from_csv()` groups rows by
the joint value of `inference.cadence_group_by_cols` (default
`["Target", "Session", "Band", "Cadence ID", "Frequency"]`) and reads each row's file path
from `inference.cadence_h5_path_col` (default `".h5 path"`):

- Rows are assumed **already ordered** within each group (ABACAD order in the source CSV).
- Groups with exactly `inference.cadence_expected_obs` (6) rows become valid
  `CadenceGroup`s; any other count is **flagged and skipped** with a warning — a cadence with
  missing or extra observations can't be scored.
- The function is column-agnostic beyond the configured names; a missing required column
  raises `KeyError` for that CSV (logged, and the CSV is skipped).

`DataPreprocessor.plan_cadences()` turns the valid groups into `PendingCadence` work units,
each paired with a deterministic output path
`{output_dir}/{csv_stem}_{sanitized_group_key}.npy`. The output directory default is
**ED-fingerprint-scoped per CSV** (#298) —
`{data_path}/inference/preprocessed/<csv_stem>_ed<hash12>/`, where the hash is
`run_state.preprocessing_config_fingerprint`: a fail-safe **denylist** hash of the inference
config minus its scoring/model/viz/retry/batching keys, plus the data-section geometry.
Energy detection is deterministic given (csv, h5 files, ED config), so **runs sharing an ED
config share stamps** — a threshold sweep or a re-inference with new weights skips
preprocessing entirely — while any ED-config change lands in a different directory by
construction (an unknown new config key over-invalidates rather than ever reusing wrong
stamps). Published `.npy` files can't be stale (atomic per-run-unique tmp → `os.replace`
publication), and the resume guard additionally verifies each sidecar's `h5_paths` and
recorded `ed_config_fingerprint` before reuse; only scores stay tag-scoped (in the DB).
Pass `--preprocess-output-dir` explicitly to pin one directory across runs and CSVs (reuse
is still guarded). Because both the output directory and per-cadence stamp filenames are
keyed on the CSV basename stem, all `inference_files` entries must have **unique basename
stems** — `plan_cadences()` raises `ValueError` naming both colliding entries if two share
a stem (e.g. `runA/x.csv` and `runB/x.csv`). The fix is to rename one CSV so basenames are
distinct.

## Model loading

### Artifact resolution: local paths or the HuggingFace Hub

Inference needs three artifacts — the encoder, the Random Forest, and the training run's
`config.json` — plus an optional fourth, the probability calibrator
`rf_calibrator.joblib`, present only for runs whose training kept one (the downloaded
config's `rf.calibration_active` records which; a Hub download fetches it best-effort
alongside the RF). They come from local disk **or** the HuggingFace Hub, resolved *before*
validation by `hf_hub.resolve_inference_artifacts(args)` (called from `main()` immediately
after argument parsing). The trio is **all-or-none**
([`src/aetherscan/hf_hub.py`](../src/aetherscan/hf_hub.py)):

- **All three local paths given** (`--encoder-path` / `--rf-path` / `--config-path`) → used
  as-is; any `--hf-revision` is ignored (logged). The offline / cluster path.
- **None given** → the artifacts are downloaded from the Hub and their cache paths are written
  onto `args`, exactly as if passed on the CLI, so validation / `apply_saved_config` / model
  load run unchanged. The resolved revision is also written to `config.hf.revision` for
  provenance in the saved inference config.
- **A partial set (one or two paths)** → left untouched; `collect_validation_errors` reports
  the missing ones. Mixing local and Hub-sourced artifacts would silently pair mismatched
  models, so it is rejected rather than half-resolved.

When downloading, the **revision** is chosen by `resolve_hf_revision` in precedence order:

1. **`--hf-revision <tag>`** — an explicit revision (a training tag `train_20260101_120000`, a
   release tag `v1.0.0`, or a commit); returned as-is, existence checked by the download itself.
2. **`v{__version__}`** — the version-coupled default. When the package is an **installed
   release**, `version_default_revision()` returns `f"v{__version__}"`, so
   `pip install aetherscan==1.0.0` + bare inference pulls exactly the `v1.0.0` weights. The
   guard is strict — only a `vX.Y.Z` version matches (`_SEMVER_TAG_PATTERN`). Source-tree /
   container runs (`PYTHONPATH=src`, the NGC image) have no installed distribution, so
   `__version__` is the `"0.0.0.dev0"` fallback
   ([`__init__.py`](../src/aetherscan/__init__.py)); `.dev` / `rc` / `post` pre-releases also
   fail the match — all fall through to step 3. This is deliberately **not** existence-checked:
   an installed release whose weights tag is missing must fail loudly (the release blessing
   step was skipped), never silently pull some other version.
3. **Latest `vX.Y.Z` release tag** on the repo (`select_default_revision`; numeric comparison,
   so `v1.10.0 > v1.9.9`). Training tags never name the default download — a no-artifact
   inference download requires a blessed release tag. Raises `RuntimeError` with guidance when
   nothing resolves.

Downloads go through `hf_hub_download` (revision-pinned, cached under `HF_HOME` /
`~/.cache/huggingface`; repeated runs hit the cache); the public repo needs no token. The repo
defaults to `config.hf.repo_id` (`zachtheyek/aetherscan`), overridable with `--hf-repo-id`.
[`RELEASE.md`](RELEASE.md) covers how this revision couples releases to weights.

### Loading the resolved artifacts

`InferencePipeline.__init__` → `init_models()` loads the two models from the (now populated)
paths:

- `--encoder-path` → `tf.keras.models.load_model` **inside `strategy.scope()`** (the hard
  rule: all TF model creation/loading happens in scope so variables are mirrored across
  replicas). The `Sampling` layer is registered serializable, so no `custom_objects` needed.
- `--rf-path` → `RandomForestModel.load()` (joblib). When the saved config records an active
  calibrator (`rf.calibration_active`), `rf_calibrator_{tag}.joblib` is loaded from the same
  directory — and a **missing calibrator artifact is a hard error**: scoring uncalibrated
  when training calibrated would be a silent train/serve mismatch. The saved
  `rf.latent_variant` / `rf.active_dims` drive how features are rebuilt from the encoder
  outputs (never hardcoded — see the two-pass cascade below).
- `--config-path` → the training run's `config_{tag}.json`, layered onto the singleton by
  `cli.apply_saved_config()` **before validation** so shape-critical fields
  (`width_bin`, `stamp_width`, `latent_dim`, `dense_layer_size`, ...) match what the encoder
  was trained with. The saved `checkpoint` section is deliberately skipped — most damagingly
  `save_tag`: without the skip, an inference run would masquerade under the training run's
  tag, corrupting DB provenance and output paths. This run's resolved save_tag (the
  `{command}_{datetime}` tag set once in `main()`) stays authoritative.

`collect_validation_errors` enforces the trio all-or-none (a partial set is the error above);
every path that *is* set must exist on disk. The three artifacts should carry the same training
tag; nothing enforces it yet (`# TODO` in `inference_command`), so mismatched encoder/RF pairs
are on the operator.

### Tag-dedup guards

Immediately before command dispatch (post-validation, post-DB-init, pre-any-work),
`tag_guards.enforce_tag_guards(args)` hard-stops a run whose **explicitly-provided**
`--save-tag` collides with a previous run's state
([`src/aetherscan/tag_guards.py`](../src/aetherscan/tag_guards.py)) — the stale-artifact
confusion that used to force manual `test_vNN` incrementing. For inference the collision
markers (`find_inference_tag_collisions`) are:

- the saved `config_{tag}.json` — written only at the very end of a successful pass, so it
  marks a *completed* run and is always a collision; and
- on the **legacy `--test-files` path only** (no `inference_cadences` manifest rows),
  non-superseded `inference_results` rows for the tag.

Manifest rows are deliberately **not** a collision: they mark an in-progress streaming run that
the resume flow below consumes, so same-tag DB state there is expected. Every resolved tag
carries a fresh second-resolution `{command}_{datetime}` stamp, so a fresh inference run can't
collide; `--force-tag` overrides the guard if it ever fires. (The same module also guards
training tags and, under `--hf-upload`, checks the Hub for the tag at startup rather than after
~30 h of training.)

## The streaming loop

`_run_streaming_csv_inference()` (main.py) is the production driver:

1. **Resume filter.** `db.flush()` (so rows queued by an in-process retry are visible), then
   every unit with a live `status='inferred'` row in `inference_cadences` for
   `(tag, npy_path)` is skipped outright — **but only if that row's stored
   `config_fingerprint` matches the current run's
   `run_state.inference_config_fingerprint(config)`** (#129). On a match, its stored aggregates
   (`n_stamps`, `n_candidates`, confidence summary) fold into the run totals and the viz
   collector; on a mismatch the cadence is re-inferred instead of reused, so a reused
   `--save-tag` with a changed inference config (threshold / model / geometry) never serves
   stale results (the supersede step retires the old row).
2. **Persistent worker pool first.** `preprocessor.start_energy_detection_pool()` starts the
   single pool that serves energy detection *and* stamp extraction for the whole run —
   started from the main thread before any background thread exists (forking after threads
   spin up risks children inheriting mid-operation locks).
3. **Models load once** (`InferencePipeline`), reused by every cadence — constructed AFTER
   the first prefetch is submitted (#298), so the 10–60 s encoder+RF+calibrator load hides
   under the first cadence's energy detection. The distributed encode step is a
   lazily-built, cached `tf.function` — repeated `run_inference` calls reuse a bounded set
   of traces (one per batch-shape bucket) instead of retracing per cadence.
4. **Prefetch depth = `inference.prefetch_depth`** (#298, default 1). A
   `ThreadPoolExecutor` keeps that many `_prefetch_cadence` futures in flight — each
   preprocesses AND loads/log-norms its cadence (`load_inference_data(parallel=False)`:
   the sequential vectorized branch, since the persistent energy-detection pool already
   owns the CPU) — consumed strictly in catalog order, so results, manifest ordering, and
   per-cadence seeding are identical at any depth. Depth 2 overlaps one cadence's
   disk-bound energy detection with the previous one's decompression-bound extraction, at
   the cost of one extra in-flight cadence of RAM. A prefetch-side load failure degrades to
   loading on the inference thread under the per-cadence containment below.
5. **Per-cadence inference** (`main.py:_infer_cadence`):
   - provenance derived from the group key + metadata JSON
     (`preprocessing.derive_cadence_provenance`): target, session, band, cadence id, header
     `tstart`, first `.h5` path, and the per-stamp center frequencies;
   - the prefetched snippet array is consumed directly (see step 4; log-norm details in
     [`PREPROCESSING.md`](PREPROCESSING.md));
   - `mark_superseded("inference_results", tag, npy_path=...)` — partial positives from a
     dead attempt are retired *before* fresh rows land;
   - `run_inference()` (below), then a superseding `inference_cadences` row with
     `status='inferred'` carrying the aggregate stats.
6. **Failure containment.** A cadence whose inference stage throws is logged, recorded as
   `status='failed'` in the manifest, and the loop moves on — one bad cadence never aborts
   the catalog. After the loop, the pass raises so the retry loop re-attempts, and the
   manifest skip means **only the failed cadences re-run**. Permanent conditions (no work
   units from the CSVs, no cadence produced stamps) raise `NonRetryableInferenceError`, which
   fails fast instead of burning retries.
7. **Finalization.** On a fully successful pass, the reference cloud is MC-scored and
   persisted (`finalize_reference_cloud()`, best-effort — see below), then (with
   `inference.inference_viz_enabled`) `render_inference_visualizations()` draws the whole
   suite, every figure individually exception-guarded.

The retry loop wraps all of it: transient failures (I/O hiccups, GPU errors) retry up to
`inference.max_retries` with `inference.retry_delay` between passes; `KeyboardInterrupt`
propagates; state-based resume (stamp `.npy` for preprocessing, manifest rows for inference)
lets the in-process retry loop pick up where the last pass died. A full relaunch mints a
fresh datetime tag and starts clean on the scoring side — its preprocessing reuses the
fingerprint-scoped stamp cache automatically (see the output-directory paragraph above).

## The GPU stage: `InferencePipeline.run_inference()`

For one cadence's snippet array `(n, 6, 16, 512)`:

1. `_distributed_encode()` (#298) feeds the replicas **directly from numpy slices** via
   `experimental_distribute_values_from_function` — no per-cadence `tf.data` dataset,
   distribution, or iterator rebuild. Full steps run at `per_replica_batch_size` snippets
   per replica (default 256 — each snippet is 6 encoder inputs, so 1,536 observation
   forwards, near the measured single-GPU encode throughput peak); the final partial step
   drops to a power-of-two bucket (floor 16), its tail padded with duplicate rows cycled
   from the cadence front so the remainder is **encoded rather than silently dropped**, and
   the padded outputs never leave the encode (only real rows are written out). Order is
   preserved (no shuffle), and the traced-shape count is bounded by the bucket ladder for
   any catalog.
2. Each snippet's 6 observations go through the encoder as independent `(16, 512, 1)`
   inputs and come back as the **deterministic posterior parameters** `z_mean` /
   `z_log_var` (#282 — no stochastic `z` ever crosses the GPU boundary; the MC draws below
   are reparameterized in NumPy from these, seeded per cadence). Per-replica results are
   gathered with `experimental_local_results` + `np.concatenate` (cheaper than an NCCL
   gather for the tiny latent payload).
3. **Two-pass cascade** (#282). Features are rebuilt per the saved config's winning
   representation (`rf.latent_variant` / `rf.active_dims` — see [`MODELS.md`](MODELS.md)),
   and the calibrator is applied to every probability when active:
   - **Pass 1** scores *every* snippet deterministically (`z_mean` in the lead feature slot)
     against the permissive `inference.screening_threshold` (default 0.5, tuned for recall;
     validated `<= classification_threshold` at startup). Its only job is to say "definitely
     not a candidate" cheaply — snippets below it are rejected without MC scoring.
   - **Pass 2** re-scores the survivors with `inference.mc_draws` (default 32) seeded
     reparameterized latent samples; the **MC mean replaces their score** and carries the
     science threshold, while the MC std is the reported uncertainty spread. Draws are keyed
     on (root seed, cadence), so a cadence's MC statistics are independent of the rest of
     the catalog.
   The two passes are a **cascade, not two ANDed criteria**: pass 1 can only ever *remove*
   candidates (training's `check_screening_threshold` verifies it loses ~zero recall vs
   MC-scoring everything — see [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)).
4. Threshold semantics: `predictions = proba_true > classification_threshold` (default
   **0.99** — deliberately conservative: at BL scale, false positives are expensive and true
   positives are expected to be vanishingly rare), where `proba_true` is the MC mean for
   pass-2 survivors and the pass-1 score for rejects. The stored `confidence` is the
   probability of the **predicted** class — `proba_true` for candidates, `1 − proba_true`
   for rejections — so a confident rejection also reads as ~1.0. Note the semantic shift
   from pre-#282 rows: under an active calibrator the negative-class confidence is the
   complement of the *calibrated* positive score, not the raw RF's class-0 probability
   (`1 − cal(p₁) ≠ cal(p₀)` in general) — every persisted probability now lives on the one
   deployed scale.
5. `_write_inference_results()` writes **positives only** to `inference_results` (a deliberate
   size trade — see [`DATABASE.md`](DATABASE.md)), each row carrying: snippet index (== row
   in the `.npy`), confidence, the flattened 48-dim latent vector (the deterministic `z_mean`
   rows), the schema-v5 score columns (`screening_proba`, `mc_mean`, `mc_std`), and full
   provenance (target/session/band/cadence id/per-stamp frequency/observation
   timestamp/`h5` path/tag). Aggregates for *all* snippets (count, threshold, above-threshold
   count, mean/min/max, quantiles p01–p99 via `summarize_confidences`) go into the cadence's
   manifest row, so run-level summaries never depend on the positives-only table.

### Interpreting the MC mean and spread

What the MC average *means* depends on the representation the RF was trained on (recorded in
`rf.latent_variant`): trained on sampled `z`, the MC mean is a true **posterior-predictive
expectation** (averaging the classifier over the encoder's posterior); trained on `z_mean`
(+ uncertainty aggregates), MC scoring is a **sensitivity/robustness probe** — how stable the
verdict is under the encoder's own uncertainty. Either way the (mean, spread) pair reads the
same:

| MC mean (p) | MC spread | Reading |
| --- | --- | --- |
| high | low | Strong candidate — confident and stable under posterior perturbation. |
| high | high | The dangerous false positive `p` alone cannot flag: the mean looks confident while individual draws swing wildly. Inspect these first. |
| ~0.5 | low | Irreducibly ambiguous — the model is *certain* it can't tell. |
| ~0.5 | high | Lowest confidence — ambiguous *and* unstable. |

Note that tree disagreement adds no independent signal here — for a forest of binary votes
the vote variance is `p(1 − p)`, a deterministic function of the probability itself; the MC
spread over latent draws is the informative axis.

### The reference cloud

A `(p, spread)` pair is only interpretable against where the *survey* sits, so pass 1 also
feeds a seeded uniform reservoir (algorithm R, `inference.reference_cloud_size`, default
10 000; `0` disables) over the **rejects'** posterior parameters — deliberately uniform, not
near-threshold, which would bias the cloud toward the boundary and make every candidate look
ordinary. After the last cadence, `finalize_reference_cloud()` MC-scores the reservoir once
and persists `{output_path}/inference_reference_cloud_{tag}.npz` (screening/mc_mean/mc_std
arrays plus subsample size, rejects seen, root seed, and draw count), so the candidate
uncertainty figure can be regenerated without re-running inference. Best-effort: a failure
degrades the plot, never the science. On a resumed run only the final attempt's cadences
feed the reservoir (manifest-skipped cadences never re-offer their rejects).

## Reproducibility

Inference used to be entirely unseeded: the encoder's `Sampling` layer drew fresh entropy
every run, so the same encoder + RF + stamps could yield a different candidate set each time.
Since #279 the `InferencePipeline` constructor seeds TF's global RNG from the shared root
seed (`config.reproducibility.seed`, default 11; `--seed`/`--unseeded` and
`--tf-deterministic-ops`/`--no-tf-deterministic-ops` are on the inference subparser too),
and `run_inference` re-seeds per cadence keyed on the stable catalog index — so a cadence's
results depend only on (root seed, cadence), reproducible even when the catalog is subset
or a run resumes partway. Deterministic ops are **ON by default** (#298): without them,
cuDNN autotune noise between otherwise identical runs is enough to flip candidates sitting
at the 0.99 threshold (measured live: 3 flips between two identical unflagged runs; two
flagged runs were bit-identical). Neither reproducibility field is layered from the saved
`--config-path` — opting out is always an explicit CLI act. The pass-2 MC draws and the
reference-cloud reservoir derive their own NumPy streams from the same root (see
[`seeding.py`](../src/aetherscan/seeding.py) and the Reproducibility section of
[`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)).

## The run manifest: `inference_cadences`

One row per (cadence, stage transition), superseding predecessors
(see [`DATABASE.md`](DATABASE.md) for the schema):

| `status` | Written when | Meaning on retry |
| --- | --- | --- |
| `preprocessed` | Stamp `.npy` + metadata land on disk | Preprocessing is done; skip to inference (the `.npy`'s existence is the actual resume key). |
| `inferred` | Inference stage completes (aggregates attached) | Cadence fully done; skipped entirely, aggregates reused. |
| `failed` | Inference stage threw for this cadence | Re-attempted on the next pass. |

Ordering makes each transition safe: within the DB's single-writer FIFO queue, the
`mark_superseded` command flushes buffered rows first, so every row queued before it gets
flagged while later writes stay live (`Database.mark_superseded`).

## Artifacts of an inference run

| Artifact | Where | Notes |
| --- | --- | --- |
| Stamp arrays + metadata | `{data_path}/inference/preprocessed/<csv_stem>_ed<hash12>/*.npy` + `.json` | Shared across runs with the same ED config (#298). The `.json` carries hit provenance: stamp starts/frequencies/statistics/p-values, ED statistic histograms, raw/merged hit lists, the `.h5` header, and the `ed_config_fingerprint` the resume guard checks. |
| Candidate rows | `inference_results` table | Positives only, with latents + provenance + the two-pass scores (`screening_proba`/`mc_mean`/`mc_std`, schema v5). |
| Run manifest | `inference_cadences` table | Per-cadence stages, aggregates, durations. |
| Reference cloud | `{output_path}/inference_reference_cloud_{tag}.npz` | MC scores for the seeded uniform reservoir of pass-1 rejects — the survey background of the candidate uncertainty figure (regenerable without re-running inference). |
| Config snapshot | `{output_path}/config_{tag}.json` | The resolved (saved-config + CLI) view this run actually used. |
| Figures | `{output_path}/plots/inference/{tag}/` | The visualization suite below; also uploaded to the run's Slack thread. |
| Resource plot | `{output_path}/plots/resource_utilization_{tag}.png` | Written by the monitor at shutdown. |
| PFB response cache | `{output_path}/cache/pfb/pfb_response_*.npy` | Content-addressed by channelization geometry; one file per `(width, coarse count, taps)`, shared across all `.h5` and all runs. See [The PFB response cache](#the-pfb-response-cache). |

### The PFB response cache

`{output_path}/cache/pfb/pfb_response_w{W}_c{C}_t{T}.npy` caches the **static PFB coarse-channel
passband response** — the filter shape that `pfb.equalize_passband` divides each coarse channel by
during bandpass flattening. It is **not** a cache of preprocessed `.h5` output: it holds a single
~8 MB array, computed once per channelization *geometry* by an ~`n_chans`-point FFT
(`gen_coarse_channel_response`) and content-addressed by its parameters — `W` = fine channels per
coarse, `C` = coarse-channel count, `T` = `pfb_taps_per_channel`.

- **What hits the cache:** every `.h5` that shares the same `(W, C, T)` channelization — which, for a
  given telescope/receiver/product, is all of them. The first cadence of a run computes the response
  (the one-time FFT — ~7–12 s at GBT scale) and writes the sidecar; every later cadence *in that run*,
  and every *future run* on any `.h5` of the same geometry, reads the cached array instead of
  recomputing.
- **What misses:** an `.h5` with a *different* channelization (different fine-per-coarse, coarse count,
  or `--pfb-taps-per-channel`) is a distinct key and computes + caches its own response. Mixing
  geometries in one campaign simply yields one cached file per geometry — none of them ever collide.
- **What it does and doesn't speed up:** it removes the one-time response FFT from all but the first
  cadence of each geometry. It does **not** change the per-channel flattening cost — `equalize_passband`
  (a vectorized divide) still runs for every coarse channel of every file. So the win is "compute the
  response once, not once per file," not "preprocess each `.h5` once."
- **Tag-independent by design:** the cache key is the geometry, never `--save-tag`, so it is shared
  across runs and never invalidated by a new tag. Because it is content-addressed, a corrupt or
  mismatched sidecar is transparently rewritten — stale-run leftovers are impossible, and the cache is
  safe to delete at any time (it just rebuilds on next use).

## The visualization suite — what each figure shows

[`inference_viz.py`](../src/aetherscan/inference_viz.py) renders at end of run from three
sources: the bounded in-memory `InferenceVizCollector` (per-cadence aggregates + a
reservoir-sampled latent pool, memory O(#cadences), candidates always kept), the durable
per-cadence metadata JSONs, and the DB tables — so figures also cover cadences that were
skipped by the resume. Every figure is wrapped in `_viz_safe` (log-and-swallow): a plot bug
can never kill a science run. Slack uploads go through a single-worker FIFO background
uploader (#298 — figure ordering and API rate unchanged; drained before teardown), and the
per-candidate figures render across a forkserver process pool in the TF-free
[`candidate_figures.py`](../src/aetherscan/candidate_figures.py) (empty preload — the
`shap_parallel`/`latent_gif` isolation pattern; per-figure failures degrade the suite
exactly like `_viz_safe`).

| File | Contents | What to look for |
| --- | --- | --- |
| `ed_stat_distributions_{tag}.png` | Log-log histogram of the D'Agostino–Pearson k² statistic over **all** windows (not just hits), per-ON-file overlay + total, threshold line. | The bulk should be a compact low-k² mass (noise ≈ χ², df=2) with a long RFI tail. The threshold should sit far into the tail: if the noise bulk crosses it, the stamp count explodes; if one ON file's curve is shifted, that file has a bandpass/level problem. |
| `ed_hit_spectrum_{tag}.png` | Hit density vs frequency (MHz), pre- vs post-deduplication. | Instantly shows RFI comb structure (regular spikes) and band edges. Dedup should collapse combs dramatically; a band where post-dedup density is still high dominates your stamp budget. |
| `bandpass_flattening_{tag}.png` | Raw vs flattened integrated spectrum for a few sampled coarse channels, with the removed model (scaled PFB response H or spline fit) overlaid. | The flattened spectrum should be level across the channel. Residual scalloping under PFB means the static response doesn't match the recording — check `--pfb-taps-per-channel` or fall back to `--bandpass-method spline` (the log's edge/mid-ratio warning fires on the same condition). |
| `stamp_gallery_{tag}.png` | Top-K stamps by detection statistic (`stamp_gallery_top_k`, default 12), each a 6-observation waterfall strip; overlap-offset copies collapsed first. | The cadence layout scientists actually inspect: a real technosignature shows in ONs (rows 0/2/4) and vanishes in OFFs; the top of this gallery is virtually always bright RFI present in all six — that's expected. |
| `preproc_funnel_{tag}.png` | Per-cadence bar funnel: raw hits → merged hits → stamps (incl. overlap copies) → snippets inferred, plus storage per cadence. | Where the volume goes. A weak merge step (raw ≈ merged) means hits are spread out rather than comb-like; snippets ≪ stamps indicates load-time validity rejections. |
| `confidence_distribution_{tag}.png` | P(true) histogram over all snippets inferred this pass (log-y), threshold line, per-cadence overlay when ≤ 10 cadences. | Mass should hug 0 with a thin bridge toward 1. Any mass just *below* threshold is worth manual inspection; a large mass above it usually means model/data mismatch (e.g. wrong config JSON) rather than a sky full of signals. |
| `candidate_gallery_{tag}.png` + `candidate_{i}_{tag}.png` | Gallery of top candidates by confidence + up to `max_candidate_plots` (50) per-candidate figures: 6-panel waterfall annotated with confidence, frequency, target/session/band, and the latent bar chart. Sourced from `inference_results`, so resumed cadences are included. | The human veto stage. Check the ON/OFF pattern by eye, the frequency against known RFI allocations, and whether the latent vector resembles the true-class latents from training. |
| `candidate_uncertainty_{tag}.png` | Each candidate (red star) at x = final RF probability (MC mean), y = MC spread, over a hexbin density background of the reference cloud (the survey's pass-1 rejects), with the science threshold as a vertical line. | Population context is the whole point: "p = 0.97, spread = 0.05" is only interpretable against where the survey sits. The dangerous quadrant is **high p + high spread** — a mean that looks confident while draws swing — exactly what `p` alone cannot flag (see the interpretation table above). Candidates hugging the survey cloud are threshold noise. |
| `inference_latent_projection_{tag}.png` | This run's cadence-level latents projected through the **training run's persisted UMAP** (`umap_cadence_nn*_md*_*.joblib`, located via the training config JSON's `model_path` + tag), over the training embedding as backdrop; candidates highlighted. Skips gracefully if the UMAP is absent. | "Where does real data live relative to the synthetic classes?" Real snippets clustering onto the training false-class region = healthy. Candidates far from *any* training class are the interesting anomalies; candidates inside the false-class cloud are threshold noise. |
| `inference_summary_{tag}.png` | Table-style run card: cadence/snippet/candidate counts, per-stage durations and throughput from the manifest, per-target/band candidate counts. | The one-glance run report — read it before opening anything else. |

## Legacy `--test-files` path

When `--inference-files` is not set, `inference_command` falls back to loading
`config.data.test_files[0]` (a preprocessed `.npy` under `{data_path}/testing/`) in one shot
and calling `inference.run_inference_pipeline()` once. No manifest, no streaming, no viz
suite; the load repeats on retry. This path exists for model smoke-testing against curated
arrays; the CSV path is the production surface.

## Configuration quick reference

Inference-specific fields live on `InferenceConfig`
([`config.py`](../src/aetherscan/config.py)); routing details in
[`CONFIG_AND_CLI.md`](CONFIG_AND_CLI.md):

| Group | Fields |
| --- | --- |
| Models | `encoder_path`, `rf_path`, `config_path` |
| Classification | `per_replica_batch_size`, `classification_threshold`, `screening_threshold`, `mc_draws`, `reference_cloud_size` |
| Streaming | `prefetch_depth` |
| Cadence grouping | `cadence_group_by_cols`, `cadence_h5_path_col`, `cadence_expected_obs` |
| Energy detection | `coarse_channel_width`, `coarse_channel_log_interval`, `bandpass_method`, `pfb_taps_per_channel`, `bandpass_debug_plot`, `spline_order`, `detection_window_size`, `detection_step_size`, `stat_threshold` |
| Stamps | `stamp_width`, `store_downsampled_stamps`, `overlap_search`, `overlap_fraction`, `preprocess_output_dir` |
| Visualization | `inference_viz_enabled`, `stamp_gallery_top_k`, `max_candidate_plots` |
| Fault tolerance | `max_retries`, `retry_delay` |
