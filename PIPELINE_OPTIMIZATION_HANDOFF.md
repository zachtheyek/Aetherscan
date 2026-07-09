# Pipeline Optimization Handoff

> **TEMPORARY / UNCOMMITTED** working note for a future agent session. Not part of the repo
> docs — delete or relocate when done. Written 2026-07-09 on branch
> `feature/blackwell_migration` (HEAD `8d6e6bc`). Goal for the next session: **continue
> optimizing the training and inference pipelines** (both are correct but slow).

---

## 1. What this branch is

`feature/blackwell_migration` adds a dual-runtime deployment (Ampere conda env + Blackwell
NGC container) plus a large CLI-validation/config system, docs, CI, and utils. See PR #82
(closes #75, #77–#81). The pipeline is two-stage ML: **Beta-VAE → Random Forest**,
single-node multi-GPU. Sole entry point: `src/aetherscan/main.py {train|inference}`.

We have been **cluster-testing** the pipeline end-to-end and fixing what surfaced. The last
task was performance: both training and (CSV) inference run correctly but are far too slow
to complete at full scale.

## 2. Clusters & how to run / observe them

| | **blpc3** (Blackwell) | **bla0** (Ampere) |
|---|---|---|
| SSH | `ssh blpc3` (direct) | `ssh bla0` (ProxyJump via NRAO gateway → **interactive password**) |
| GPUs | 5× RTX PRO 6000 Blackwell (sm_120) | 6× RTX A4000 |
| CPU | 32 cores (AMD EPYC 7313, 16c/32t) | (unchecked) |
| Repo | `/mnt_home/zachy/Aetherscan` | `/home/zachy/Aetherscan` |
| Runtime | **container only** (singularity) | container (apptainer/singularity) **or** conda |
| `.sif` | `<repo>/aetherscan-ngc25.02.sif` present | present |

- **bla0 SSH needs an interactive password** (gateway `zyek@ssh.gb.nrao.edu`). There's no
  `sshpass`; `expect` works. This session used an `expect` wrapper + an SSH `ControlMaster`
  socket. A fresh session must re-establish auth (ask the user for the password).
- **tmux**: session `aetherscan`, window 0, **top pane = `aetherscan:0.0`** is where the
  pipeline is run. Drive it with `tmux send-keys -t aetherscan:0.0 "<cmd>" Enter`. This
  keeps long runs alive across SSH drops. `pane_current_command` tells you if it's idle
  (`bash`) vs running (`Singularity`/`Apptainer`/`python`).
- **Shared data paths (identical on both machines):**
  - data `/datax/scratch/zachy/data/aetherscan/{training,testing,inference}/`
  - models `/datax/scratch/zachy/models/aetherscan/` (weights only, after `578b086`)
  - outputs `/datax/scratch/zachy/outputs/aetherscan/` (configs, `plots/`, `logs/`, `db/`, `preprocessed/`)
  - DB: `outputs/aetherscan/db/aetherscan.db` (SQLite). Log: `outputs/aetherscan/logs/aetherscan.log`
    (opened `mode="w"` — **only the current run's** log; grep it for progress/errors).
- **Container wrapper:** `./utils/run_container.sh python -m aetherscan.main …`. It binds
  REPO + DATA/MODEL/OUTPUT (1:1). Code changes take effect via bind-mount — **no rebuild
  needed after `git pull`**.
- **Raw .h5 archive for CSV inference lives at `/datag/pipeline/…` (blpc3 only) and is NOT
  bound by run_container.sh** → must launch CSV inference with `SINGULARITY_BIND=/datag`
  (singularity honors it; run_container.sh doesn't clear env). *Proper fix pending:* add an
  extra-bind knob to run_container.sh.

### Launch recipes actually used (both container, `--max-retries 1`)
```bash
# bla0 full training (6 GPUs, default config valid at 6):
./utils/run_container.sh python -m aetherscan.main train --save-tag final_v1 --max-retries 1

# blpc3 CSV inference on a SUBSET (dummy model = tag test_v17; datetime save-tag):
SINGULARITY_BIND=/datag ./utils/run_container.sh python -m aetherscan.main inference \
  --encoder-path /datax/scratch/zachy/models/aetherscan/vae_encoder_test_v17.keras \
  --rf-path      /datax/scratch/zachy/models/aetherscan/random_forest_test_v17.joblib \
  --config-path  /datax/scratch/zachy/models/aetherscan/config_test_v17.json \
  --inference-files subset_test.csv --save-tag $(date +%Y%m%d_%H%M%S) --max-retries 1
```
Fast-test **subset CSV** (2 complete 6-obs cadences) was built by grouping
`inference/complete_cadences_catalog.csv` on cols `(Target,Session,Band,CadenceID,Frequency)`
and writing the first two 6-row groups to `inference/subset_test.csv`.

## 3. Current run state (as of this handoff)

- **bla0**: full training `final_v1` **still running** in-container. Was ~round 1 / epoch 56
  of 100. **~6.3 min/epoch → full 20×100 ≈ ~9 days.** GPU util **0–14%** (data-feed bound).
- **blpc3**: subset CSV inference **completed successfully** end-to-end (tag was a datetime).
  Dummy `test_v17` model preserved in `models/`. `preprocessed/subset_test_*.npy` exist
  (114 GB + 34 GB). Pane idle.

## 4. Performance findings (measured)

### Inference (blpc3, 5× Blackwell, container) — 2-cadence subset ≈ 2h13m
Stage breakdown, GPUs idle ~98% of the run:
- **Energy detection: ~45 min/cadence** — now the dominant cost. 18 blocks × 3 ON-source
  files; per-block work is parallelized to `parallel_coarse_chans=28` procs, but each block
  runs a **Python `scipy.stats.normaltest` sliding-window loop** (~8190 windows/coarse-chan).
- **Stamp extraction: FIXED this session** — was single-threaded (~tens of min, GPU-idle);
  now parallel (see §5). Measured: 72,947 stamps in ~11.5 min / 21,642 in ~4.75 min (32 procs).
- **Downsample 4096→512 + log-norm: ~24 min** for 94,589 stamps — a *secondary* CPU cost now
  visible (was hidden behind the slower stamp phase).
- **GPU stages: ~32 s total** — VAE latents for 94,589 snippets on 5 GPUs; RF <1 s.
- **Storage blowup:** stamps are stored at full 4096 width → **~114 GB per cadence** (72,947
  stamps × 6 × 16 × 4096 × f32). 4,411 cadences ⇒ hundreds of TB — infeasible on storage
  alone, independent of time.
- **Candidates found: 0** (expected: dummy 1-epoch model + 0.99 threshold + real data). Only
  positive predictions are written to `inference_results`, so 0 rows is correct.

### Training (bla0, 6× A4000, container)
- GPU util **0–14%** → GPUs starved. Cause: `tf.data.Dataset.from_generator` yields **one
  sample at a time** (GIL-bound, single-threaded), then `.batch(768)`; ~1 ms/sample of
  Python/marshal overhead → 6 GPUs can't be fed. A comment at `train.py:351` notes the
  generator's per-iteration **lock contends with the `.prefetch(AUTOTUNE)` threads**.

## 5. Optimization already implemented this session

**`8d6e6bc perf(preprocessing): parallelize inference stamp extraction`**
- `src/aetherscan/preprocessing.py`: the post-dedup stamp loop (was "sequential per file, no
  pool") now writes a **memmap-backed `.npy`** while a worker pool fills **disjoint
  `(obs_file, contiguous stamp-range)` slices** — up to `config.manager.n_processes` workers
  (`ceil(n_processes / n_obs)` contiguous chunks per file, preserving hdf5 chunk-cache reuse).
- New module-level worker `_extract_stamps_worker` (opens its own h5 handle + r+ memmap view).
- No giant SHM, no array pickling; atomicity/resume-safety preserved (`.tmp` → `os.replace`
  after all workers finish); sequential fallback when `n_processes<=1`.
- **Validated**: correct stamp counts (`hits×3`), 114 GB memmap written flawlessly by 32
  concurrent workers, ~6–8× faster than the old sequential path, 0 leaked SHM.

## 6. Prioritized optimization opportunities (next session)

### Inference (biggest wins first)
1. **Vectorize the energy-detection normality test** (the #1 remaining cost). See the TODO at
   `preprocessing.py:254` in `_threshold_hits_worker` (~line 226): replace the Python window
   loop calling `scipy.stats.normaltest` with a `stride_tricks` view + vectorized
   `scipy.stats.skew`/`kurtosis` (axis arg) → few large ndarray ops. Could be a big multiplier.
2. **Parallelize / fuse the downsample step** (~24 min). Downsampling of the 94,589 stamps
   (`downscale_local_mean` 4096→512) appears CPU-bound and single-threaded on the inference
   path; parallelize like the stamp extraction, or **downsample-at-extraction** so stamps are
   stored at 512 width (also cuts the ~114 GB/cadence storage ~8×).
3. **CPU–GPU pipelining** — GPUs idle ~98%. Overlap preprocessing of cadence N+1 with GPU
   inference of cadence N (matches the existing `# NOTE` at top of `main.py`).
4. Minor: bump `parallel_coarse_chans` 28→32 to use all blpc3 cores in energy detection.

### Training (bla0 GPU starvation)
1. **Yield pre-batched slices from the generator** (numpy `arr[i:i+B]`) instead of per-sample
   → ~768× fewer Python calls/marshals. Requires reshaping `output_signature` and dropping the
   `.batch()`. Biggest win. (`train.py:526`.)
2. `interleave` several generator shards with `num_parallel_calls` for parallel feeding.
3. Shrink the generator's lock scope (lock-free reads from the shared-memory `TrainDataHolder`)
   to unblock prefetch (`train.py:351`).
4. `from_tensor_slices` for the smaller **val/viz** datasets (full train data ~300 GB won't fit).

## 7. Config knobs relevant to perf (`src/aetherscan/config.py`)
- Training: `num_training_rounds=20`, `epochs_per_round=100`, `num_samples_beta_vae=499200`,
  `per_replica_batch_size=128`, `effective_batch_size=3072`, `per_replica_val_batch_size=80`,
  `num_samples_rf=99840`, `train_val_split=0.8`, `curriculum_schedule="exponential"`.
- Inference: `coarse_channel_width=1048576`, `parallel_coarse_chans=28`, `spline_order=16`,
  `detection_window_size=256`, `detection_step_size=128`, `stat_threshold=2048.0`,
  `overlap_fraction=0.5` (⇒ 3 stamps/hit), inference `per_replica_batch_size=2048`.
- `manager.n_processes = cpu_count()`; `manager.chunks_per_worker=4`.

## 8. Gotchas / known issues (context; some are latent bugs, not yet fixed)
- **Divisibility**: default batch config is valid only for **4 or 6** replicas, **not 5**.
  blpc3 has 5 GPUs → training there needs `--num-replicas 4` or a hand-tuned config
  (e.g. `--per-replica-batch-size 4 --per-replica-val-batch-size 4 --effective-batch-size 20
  --num-samples-beta-vae 200 --num-samples-rf 200 --latent-viz-num-cadences-per-type 5`).
  Inference is fine at 5. Use `utils/find_optimal_configs.py --num-gpus N train|inference …`
  to validate any config locally without TF.
- **Save-tags must match** `test_vX | final_vX | round_XX | YYYYMMDD_HHMMSS` (`_TAG_PATTERN`,
  cli.py:25). Free-form slugs are rejected (there's a `# NOTE` to broaden this).
- **`--config-path` clobbers `save_tag`** (and other checkpoint fields): `apply_saved_config`
  loads *all* JSON fields, so an inference run using a training config inherits that config's
  `save_tag` unless you override with `--save-tag`.
- **`start_time` resume bug** (documented only, `# BUG:` at `train.py:~908`): a resume/retry
  that finds the VAE already trained (`start_round > n_rounds`) returns before
  `self.start_time` is set → later plotting raises `AttributeError`, masking the real error
  across retries. Fix: set `start_time` in `__init__`. Low risk with `--num-training-rounds 1`
  runs that succeed on attempt 1.
- **Ctrl-C**: fixed this session (`2bff88b`) — 1st = graceful cleanup, 2nd = force-quit +
  ignore further (was a re-entrant `_cleanup_lock` deadlock on double Ctrl-C).
- **Benign warnings** (documented in KNOWN_ISSUES #14/#15): XLA "Skipping the delay kernel"
  (both clusters) and LLVM "+ptx85 … ignoring feature" (Blackwell only). NCCL works
  (NcclAllReduce, no HierarchicalCopy fallback); no PTX/CUDA errors.

## 9. Key code locations
- Inference preprocessing: `src/aetherscan/preprocessing.py` — `find_hits`/cadence loop
  (~824+), `_threshold_block_hits` (~1250) + `_threshold_hits_worker` (~226, normality loop
  ~262, TODO ~254), **new** `_extract_stamps_worker` (~272) + memmap extraction (~1137).
- Inference GPU stage: `src/aetherscan/inference.py` — latents + RF classify; positive-only
  writes to `inference_results` (`prediction == 1`, ~line 430).
- Training loop + data pipeline: `src/aetherscan/train.py` — `from_generator` datasets (~526),
  lock-contention note (~351), stratified RF split (~424), `start_time` (~915).
- Entry/dispatch + config save locations: `src/aetherscan/main.py`
  (`setup_gpu_strategy`, train/inference commands, config→`output_path`).
- Resource/signal management: `src/aetherscan/manager/manager.py`
  (`create_pool`/`close_pool`, `_signal_handler`, `cleanup_all`, non-reentrant `_cleanup_lock`).

## 10. This session's commits (feature/blackwell_migration)
`8d6e6bc` parallel stamp extraction · `578b086` config→output_path · `2bff88b` Ctrl-C fix ·
`abab866` (user) config docstrings · `b45e7b8` KNOWN_ISSUES #14/#15 · `239b401` tag NOTE +
start_time `# BUG:` · `f787cb6` RF-val + curriculum-schedule validation fixes · `32cab08`
data-file path-helper fix · plus earlier `_resolve_num_replicas` merge/rename work.
