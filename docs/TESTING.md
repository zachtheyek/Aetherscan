# Testing

This document covers the pytest suite: layout, markers, the singleton-isolation machinery in
`conftest.py`, the synthetic data factories, what CI runs, how to run the cluster-marked
integration smokes, and the checklist for adding tests. The quick-start version lives in
[`CONTRIBUTING.md`](../CONTRIBUTING.md#testing); this is the deeper reference.

## TL;DR

```bash
# Default selection — matches what CI runs (no GPUs, no cluster data needed).
# CI additionally appends `and not integration` as a leak-guard; see the `## CI`
# section below. The two expressions select the same set today.
pytest -m "not gpu and not cluster" -q

# One file / one test while iterating:
pytest tests/unit/test_preprocessing.py -q
pytest tests/unit/test_db.py -k supersede -q

# Cluster integration smokes (end-to-end train + CSV inference), on a cluster,
# inside the NGC container:
./utils/run_container.sh python -m pytest tests/ -m "gpu or cluster" -q
```

Configuration lives in [`pyproject.toml`](../pyproject.toml) under
`[tool.pytest.ini_options]`: `pythonpath = ["src"]` (no `PYTHONPATH` prefix needed),
`testpaths = ["tests"]`, and `--strict-markers` (a typo'd marker is a collection error, not a
silently-ignored one).

## Layout

```
tests/
├── conftest.py                  # isolation fixture + data factories (below)
├── fixtures/
│   └── ed_real_slice.npz        # recorded real coarse-channel slice — the energy-detection
│                                #   equivalence gate runs against real data, not just synthetic
├── unit/                        # fast, hardware-independent; the CI surface
│   ├── test_config.py               # singleton semantics, to_dict round-trip
│   ├── test_cli_validation.py       # tag pattern, divisibility matrix, saved-config precedence
│   ├── test_cli_help_sync.py        # README CLI Reference vs generated help drift guard
│   ├── test_tag_guards.py           # save-tag dedup matrix: local/HF collisions, --force-tag
│   ├── test_data_generation.py      # log_norm, create_* shapes/labels, intersection checks,
│   │                                #   chunk segments / batched task partitioning
│   ├── test_round_data.py           # paths/manifest protocol, reuse/delete semantics, producer
│   ├── test_seeding.py              # root-seed stream derivation (reproducible runs)
│   ├── test_run_state.py            # manifest round-trip, stage machine transitions
│   ├── test_train_utils.py          # load-tag resolution, curriculum schedules, archiving
│   ├── test_train_datasets.py       # batched generators: coverage, stratification, alignment,
│   │                                #   zero-copy tensor wrapping (alignment gate + fallback)
│   ├── test_train_accumulation.py   # graph-side accumulated train step vs eager reference,
│   │                                #   accumulator reset, NaN-guard apply skip, val loop
│   ├── test_train_distribution.py   # 2-logical-CPU MirroredStrategy subprocess: per-replica
│   │                                #   element split == global batch order (RF alignment)
│   ├── test_latent_traversal.py     # traversal grid math with a stub decoder
│   ├── test_latent_gif.py           # parallel GIF frame renderer: byte-identity across worker
│   │                                #   counts, batched-transform split preservation
│   ├── test_models.py               # feature layout, RF train/predict, encoder/decoder symmetry
│   ├── test_latent_variants.py      # variant feature builders, active-unit detection, recall@FPR/
│   │                                #   ECE/min-margin selection, probability calibrator, variant/active-dims stamp joblib round-trip (TF-free)
│   ├── test_rf_metrics.py           # pure RF eval-metric helper: AUC/AP/Brier/confusion/quantile hand-computed cases, degenerate single-class split
│   ├── test_shap_parallel.py        # positive-class shape normalization + MP-vs-single-process byte-identity (all three passes)
│   ├── test_preprocessing.py        # k² equivalence gates, dedup, grouping, DC spike, spline
│   ├── test_pfb.py                  # response shape/symmetry/flatness, C++ sinc cross-check
│   ├── test_inference.py            # padding fix, provenance mapping, confidence summaries
│   ├── test_inference_viz.py        # every figure smoke-tested against tiny synthetic inputs
│   ├── test_main.py                 # streaming-loop resume/containment with stubbed stages
│   ├── test_hf_hub.py               # revision resolution, artifact download, upload staging, card
│   ├── test_db.py                   # writer thread, flush/supersede sentinels, migrations, queries
│   ├── test_manager.py              # pool/SHM tracking and cleanup idempotence
│   ├── test_monitor.py              # get_process_tree_stats cache: reuse/eviction/PSS, RAM-only path, outer guard
│   ├── test_benchmark.py            # stage_timer nesting/failures, report tree math + suggestions
│   ├── test_perband_report.py       # per-band plot: umbrella-vs-child span isolation, catalog
│   │                                #   join + count-guard skip, per-band stats, non-empty PNG
│   ├── test_candidate_figures.py    # TF-free candidate renderer: pool/serial parity, containment
│   ├── test_dashboard.py            # dashboard pure data layer (DB-driven plot data)
│   ├── test_dashboard_cli.py        # aetherscan-dashboard console entry: exec-argv builder + streamlit-missing guard
│   ├── test_dashboard_launcher.py   # dashboard launcher argv builder + guard paths
│   └── test_logger.py               # StreamToLogger redirect probes + log_path_for_tag / tagged FileHandler
└── integration/                 # marked integration+gpu+cluster: needs real GPUs + cluster data
    ├── conftest.py                  # repo-root launcher + cluster path resolution
    ├── test_train_smoke.py          # known-good training smoke config, end to end
    ├── test_inference_smoke.py      # subset CSV inference against cluster-resident .h5 data
    ├── test_model_behavior.py       # SNR→confidence monotonicity gate vs the persisted VAE+RF
    └── test_seeding_determinism.py  # end-to-end determinism smoke: seed + op-determinism → byte-identical weights
```

## Coverage and deliberate gaps

A couple of modules are unit-tested lightly or not at all, by design rather than oversight:

- **`monitor`** now has a dedicated unit-test module (`tests/unit/test_monitor.py`) covering
  the `get_process_tree_stats()` cache logic — process-object reuse, warm-up, eviction,
  access-denied retention, PSS aggregation, the no-cache RAM-only path, and the outer guard.
  The 1 Hz background sampling thread and the matplotlib rendering of the resource plot remain
  exercised only by the integration smokes (real runs) and verified by manual inspection of
  the resource-utilization plot uploaded to Slack. (The pure helper `select_annotation_spans`
  is also covered — in `test_benchmark.py`.)
- **`logger`** is unit-tested in `test_logger.py` for the `StreamToLogger` stdout/stderr-redirect
  probes (`isatty`/`writable`/`readable`/`fileno`), the pure `log_path_for_tag` tag→path
  derivation, and that a `Logger` builds its `FileHandler` from the tagged path; the QueueListener,
  SlackHandler, and stderr-to-logger redirect are exercised by the integration smokes rather
  than unit tests. In particular `src/aetherscan/logger/slack_handler.py` — a `logging.Handler`
  that batches records, posts them as threaded replies to a per-run summary message, color-codes
  by level, retries with exponential backoff, throttles via a consecutive-failure cooldown, and
  uploads images — has no unit tests at all; its batching / level-to-color coding / backoff /
  throttling are only ever driven by real runs. It also carries two known `# BUG:` markers at the
  top of the module (batch messages colored by the wrong priority level, and over-long batched
  messages truncated to a trailing `...`), both observability-only cosmetics that do not affect
  pipeline results.

The inference pipeline follows the suite's usual shape: its logic is unit-tested at the
function level (`test_inference.py`, `test_main.py`, `test_inference_viz.py`), with full
end-to-end behavior covered by `test_inference_smoke.py` — there is no unit-level end-to-end
inference test by nature.

A few narrower gaps in the training-data pipeline were identified during that PR's
verification pass and left as low-risk follow-ups (each is safe by construction or covered
indirectly today; a direct test would only harden against future refactors):

- **`_distributed_encode`'s step-count drift guard** (`train.py`) has no direct unit test. It
  fires loudly (`RuntimeError`) when the caller's `train_steps × accumulation_steps` disagrees
  with what `prepare_distributed_train_dataset` will yield; the alignment it protects *is*
  pinned by `test_train_datasets.py`, but the guard itself is only reached via the full encode
  path. A CPU-strategy test that drives a deliberately mismatched geometry would pin it.
- **Per-task RNG determinism** in `data_generation._run_memmap_task` — that the same
  `(task, seed)` produces byte-identical output regardless of worker scheduling or prior global
  RNG state — is asserted by `test_round_data.py`'s same-seed-twice test
  (`test_same_seed_regenerates_byte_identical_data`, which perturbs the global RNGs between
  runs), but only through the sequential in-process path; worker-scheduling independence under
  a real pool is still only exercised implicitly.
- **Round-data manifest corruption cases**: `test_round_data.py` covers missing/mismatched
  manifests, shape mismatch, and whole-array value corruption, but truncation and array-swap
  are only caught *implicitly* (via the broad `except` and the sampled checksum). The checksum
  is a probabilistic smoke test by design (see `round_data._array_checksum`), so these aren't
  integrity guarantees; a truncation test would at least pin the current behavior.
- **Stage-timing *wiring* in the real train/inference paths** (issue #167 item 4): the
  `stage_timer`/`record_stage` machinery and the report consumers are unit-tested against
  synthetic `pipeline_stages` rows, but nothing at unit level asserts that the real pipeline
  code paths emit the expected span names at the expected places (e.g. `train.round_NN`,
  `train.round_NN.data_generation`, `inference.infer_cadence_NNN`). Driving the real
  `train_round`/`_infer_cadence` in a unit test would need heavy TF-stack mocking for modest
  signal, so this stays a cluster-smoke-only concern — mitigated by the shared
  `round_stage_name()` helper (pins producer/trainer name agreement) and by
  `test_round_data.py`'s timing test, which drives the real drainer-to-`record_stage` path
  with a real span name.

**Model-behavior gates** (issue #139): validation gates on model *quality*, as opposed to
function-level correctness, need a trained model and therefore live outside the CI selection.
Gate 1 — the opt-in `training.min_val_auc` floor on the RF's validation ROC-AUC (a loud
WARNING at train time when unmet; see [`CONFIG_AND_CLI.md`](CONFIG_AND_CLI.md)) — has its
guard logic unit-tested in `test_train_utils.py`, but whether a given run *clears* it is only
ever observed on real training runs. Gate 2 — the SNR→confidence monotonicity check in
`tests/integration/test_model_behavior.py` — generates cadences at controlled SNRs with the
training pipeline's own injection code and scores them in-process against the persisted
VAE+RF, so it is `gpu`+`cluster`-only by nature (it can never run in CI).

Two further gaps are accepted as-is (neither affects pipeline correctness):

- **`slack_handler.py`** carries no unit tests — it is observability-only (a `logging.Handler`
  that posts records to Slack), so its absence never affects pipeline results. Its network-free
  helpers — the consecutive-failure cooldown, exponential backoff, and level-to-color coding —
  would be cheap to unit-test in isolation if the module is ever hardened.
- **End-to-end training-loop resume and clean SIGTERM/SIGINT shutdown mid-run** are exercised
  only by real cluster runs: nothing today asserts that a mid-run interrupt tears down resources
  cleanly and resumes from the persisted checkpoint/manifest. This promotes the standing
  [`tests/conftest.py`](../tests/conftest.py) TODO into the visible gaps list.

## Markers

| Marker | Meaning | In default selection? |
| --- | --- | --- |
| `slow` | Slower CPU tests (e.g. builds real TF graphs) | **Yes** — CI runs them |
| `gpu` | Needs one or more physical GPUs | No |
| `cluster` | Needs cluster-resident data/models (blpc3/bla0 paths) | No |
| `integration` | End-to-end subprocess runs; **skips the isolation fixture** | No — also `gpu`+`cluster`; CI excludes by marker too as a leak-guard |

`--strict-markers` rejects anything not declared in `pyproject.toml` — add new markers there
first.

## Isolation: singletons, env, and teardown

Every singleton class (`Config`, `Database`, `Logger`, `ResourceManager`, `ResourceMonitor`)
carries a `_reset()` classmethod designed for tests. The autouse fixture
`aetherscan_isolated_env` in [`tests/conftest.py`](../tests/conftest.py) wraps every
non-integration test:

**Setup** — point `AETHERSCAN_{DATA,MODEL,OUTPUT}_PATH` at a fresh `tmp_path` tree (with
`training/`/`testing/`/`inference/` subdirs), delete `SLACK_BOT_TOKEN`/`SLACK_CHANNEL` (tests
must never talk to Slack), snapshot the current SIGINT/SIGTERM handlers and
stdout/stderr, reset all singletons, and `init_config()`.

**Teardown** — `_teardown_singletons()` mirrors `ResourceManager.cleanup_all()`'s ordering
(processes → pools → shared memory → monitor → DB → logger) without its logging side effects,
**unregisters the manager's `atexit` callback** (otherwise every test's manager instance
would pile up a stale cleanup hook for interpreter exit), resets all singletons again, and
restores the signal handlers and streams.

Consequences worth knowing:

- Constructing a `ResourceManager` in a test installs real signal handlers — the fixture
  restores them, but don't spawn threads that outlive the test.
- `MPLBACKEND=Agg` and `TF_CPP_MIN_LOG_LEVEL=2` are set before any aetherscan import
  (train.py imports pyplot at module level; CI runners are headless).
- Singleton imports inside the fixture are deferred so **integration runs never import
  TensorFlow into the pytest parent process** — the integration tests exercise the pipeline
  as a subprocess and inherit the real environment instead.

## Synthetic data factories

Three factory fixtures produce inputs shaped like the real thing but sized for speed:

| Fixture | Produces |
| --- | --- |
| `make_background_npy(filename, n_cadences=8, width_bin=512)` | Positive chi-squared-ish float32 plates `(n, 6, 16, width_bin)` under the tmp `data/training/` dir. |
| `make_h5_observation(filename, n_chans=2048, time_bins=16)` | A filterbank-style `.h5` with a `data` dset `(time_bins, 1, n_chans)` + `fch1`/`foff`/`nchans` attrs (plain h5py — no bitshuffle needed for tests). |
| `make_inference_csv(filename, groups)` | A cadence-grouping CSV whose column names come from the live `InferenceConfig` (`cadence_group_by_cols` / `cadence_h5_path_col`). |

Prefer these over hand-rolling arrays — they stay in sync with config defaults, and they're
the reason a preprocessing test can run the *real* grouping/detection code paths in
milliseconds.

For numerical-equivalence tests, `tests/fixtures/ed_real_slice.npz` is a small recorded slice
of a real coarse channel: the vectorized D'Agostino–Pearson implementation is pinned against
`scipy.stats.normaltest` at `rtol=1e-9` on synthetic distributions *and* asserted to produce
identical hit sets on real data (see [`PREPROCESSING.md`](PREPROCESSING.md)).

## CI

[`.github/workflows/tests.yml`](../.github/workflows/tests.yml) runs on every PR and on
pushes to master, on Python **3.10, 3.11, and 3.12** (the full `requires-python` range —
3.10 and 3.12 are the conda and NGC-container runtimes; `fail-fast: false` so all report):

```
pip install "tensorflow-cpu==2.17.*" -r requirements-container.txt h5py hdf5plugin pandas psutil pytest
pytest -m "not gpu and not cluster and not integration" -q
```

The trailing `and not integration` is a defense-in-depth leak-guard: it catches
any future test marked *only* `integration` (which skips the singleton-isolation
fixture) that would otherwise leak into CI without a `gpu` or `cluster`
co-marker. Today every `integration` test is also `gpu`+`cluster`, so the
`and not integration` clause is a no-op on the current suite.

`tensorflow-cpu` stands in for the container's GPU TF 2.17 build; `h5py`/`hdf5plugin`/
`pandas`/`psutil` are installed explicitly because the NGC base image ships them (so
`requirements-container.txt` intentionally omits them). A few tests assert Linux-only
behavior (PSS memory accounting) and self-skip elsewhere — the suite is green on macOS
locally, with skips. See [`GITHUB_AUTOMATION.md`](GITHUB_AUTOMATION.md) for how the test
workflow feeds the weekly flaky-test tracker.

## Running the cluster smokes

The two end-to-end smokes launch `python -m aetherscan.main ...` as a subprocess from the
repo root (their conftest prepends `<repo>/src` to `PYTHONPATH`) with the known-good smoke
configs, and assert on return codes and produced artifacts; `test_model_behavior.py` instead
generates and scores synthetic cadences in-process against the persisted VAE+RF (minutes,
not hours). The suite needs real GPUs, the cluster-resident data/models, and hours of wall
time for the smokes (subprocess timeout: 2 h each):

```bash
# On the cluster, inside the container:
./utils/run_container.sh python -m pytest tests/integration -m "gpu or cluster" -q

# AETHERSCAN_* env vars override the baked-in cluster defaults if your paths differ.
```

Be a good citizen: check `nvidia-smi`/`htop` for other users' jobs first, and clean up
`/dev/shm` and scratch artifacts afterwards.

## Adding tests — checklist

1. **Ship unit tests with new logic.** Every PR that adds or changes behavior lands tests for
   it under `tests/unit/` in the matching `test_<module>.py` (create it if the module is
   new).
2. **Use the fixtures.** Never instantiate singletons directly — the autouse fixture already
   gave you a clean `Config` against tmp paths; use `get_config()` and mutate it in the test
   if you need non-defaults. Use the data factories for inputs.
3. **Mark honestly.** `slow` for anything that builds real TF graphs; `gpu`/`cluster` for
   anything the default selection couldn't run on a laptop. Everything unmarked must pass on
   a CPU-only CI runner.
4. **Test the seam, not the world.** The codebase exposes deliberate test seams — e.g. the
   producer's `generate_fn` stub, duck-typed pipelines in `_execute_training_stages`, and
   module-level pure functions (`_sliding_normality_k2`, `build_chunk_segments`,
   `build_traversal_latents`) — prefer them over patching internals.
5. **Numerical claims get gates.** If you vectorize or port math, pin equivalence against the
   reference implementation with an explicit tolerance, on synthetic *and* (where feasible) a
   small recorded real fixture.
6. **Style applies.** Ruff lint+format runs on `tests/` too; same conventions as `src/`
   ([`CONTRIBUTING.md`](../CONTRIBUTING.md#code-style)).
