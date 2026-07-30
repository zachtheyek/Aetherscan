# Note, we avoid logging anything in config.py to prevent coupling with the logger module
"""
Configuration module for Aetherscan Pipeline
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from multiprocessing import cpu_count


@dataclass
class DBConfig:
    """SQLite database configuration"""

    get_connection_timeout: float = 60.0  # seconds
    stop_writer_timeout: float = 10.0  # seconds
    write_interval: float = 5.0  # seconds
    # Rows per foreground transaction. 100 meant a commit (and fsync) every 100 rows — one of
    # the drivers of the ~590 rows/s writer that let a multi-hour backlog build up (#277)
    write_buffer_max_size: int = 5000  # records
    write_retry_delay: float = 1.0  # seconds
    flush_timeout: float = 10.0  # seconds
    # Bulk lane (#277): high-volume injection stats ride a separate bounded queue so a plot
    # flush only has to drain the (small) foreground lane, and queue growth is capped —
    # backpressure blocks the background enqueuer (the round-data drainer thread), never the
    # training path. bulk_chunk_rows is both the enqueue granularity and the bulk transaction
    # size; the cap is bulk_queue_max_items * bulk_chunk_rows rows in memory (~1.6M at defaults).
    bulk_chunk_rows: int = 50_000  # rows per bulk queue item / transaction
    bulk_queue_max_items: int = 32  # bounded bulk-lane depth (items)
    # Shutdown drain (#277): stop() drains BOTH lanes to disk before the writer exits (the old
    # behavior silently dropped everything still queued — up to ~26M rows on the release run).
    # If the drain exceeds this cap the remaining rows are dropped with an exact ERROR count.
    stop_drain_timeout: float = 600.0  # seconds


@dataclass
class ManagerConfig:
    """Resource manager configuration"""

    n_processes: int = cpu_count()  # use all available cores
    # TODO: experiment with larger chunk sizes (how to track chunk processing efficiency)
    # NOTE: should we move chunks_per_worker to DataConfig() or TrainingConfig() and make it specific to preproc/data_gen?
    chunks_per_worker: int = 4  # for balancing overhead vs parallelism
    pool_terminate_timeout: float = (
        10.0  # seconds (actual timeout may be 2x this value -- from terminate + join threads)
    )


@dataclass
class MonitorConfig:
    """Resource monitor configuration"""

    get_gpu_timeout: float = 5.0  # seconds
    stop_monitor_timeout: float = 10.0  # seconds
    monitor_interval: float = 1.0  # seconds
    monitor_retry_delay: float = 1.0  # seconds
    # Overlay top-level pipeline_stages spans (depth <= 2 dot-names, e.g. "train.round_03")
    # as labeled translucent bands on the resource plot's CPU panel, so utilization
    # plateaus are attributable to pipeline stages at a glance
    annotate_stages: bool = True
    # Live monitoring dashboard (aetherscan/dashboard.py) auto-launched by main.py at run start;
    # --no-dashboard opts out. Served headless on dashboard_port (SSH-forward to reach it).
    dashboard_enabled: bool = True
    dashboard_port: int = 8501
    # End-of-run benchmark report (utils/benchmark_report.py) rendered at the tail of
    # train/inference and posted to Slack; --no-benchmark-report opts out.
    benchmark_report_enabled: bool = True


@dataclass
class LoggerConfig:
    """Logger configuration"""

    # Set log levels
    console_level: str = "INFO"
    file_level: str = "INFO"
    slack_level: str = "INFO"

    # Slack configuration
    slack_enabled: bool = True
    slack_channel: str | None = None  # Override with SLACK_CHANNEL env var
    slack_username: str = "Aetherscan"
    slack_timeout: float = 15.0
    slack_retry_attempts: int = 3
    slack_buffer_size: int = 100  # Max messages to buffer before flushing
    slack_flush_interval: float = 60.0  # Seconds between automatic flushes
    slack_broadcast_level: str = "ERROR"  # Messages at this level+ are broadcast to main channel


@dataclass
class BetaVAEConfig:
    """Beta-VAE model configuration"""

    latent_dim: int = 8  # Bottleneck size
    dense_layer_size: int = 512  # Should match num frequency bins after downsampling
    kernel_size: tuple[int, int] = (3, 3)  # For Conv2D & Conv2DTranspose layers
    beta: float = 1.5  # KL divergence weight
    alpha: float = 10.0  # Clustering loss weight
    # Opt-in keras mixed_bfloat16 policy for the Beta-VAE (train.py sets the global policy
    # before model build). A/B-gated: changes training numerics — bf16 needs no loss scaling,
    # variables/optimizer state stay fp32 under keras mixed precision, and the numerically
    # sensitive layers (z_mean/z_log_var/Sampling, decoder sigmoid output → loss math) are
    # pinned fp32 in models/vae.py. Default False = fp32 end-to-end, byte-identical to before
    # this flag existed (no policy call is made at all). STAYS OFF after the 7-seed A/B:
    # 6 seeds clean, but seed 13 reproducibly fails (recall .8432 / val AUC .9807, far below
    # the 6-seed control envelope, and the only runs to trip the ECE->calibrator gate were
    # bf16-seed-13, twice, across configs) — a bf16-specific trajectory pathology the gate
    # exists to catch. Revisit only with that pathology understood (+1.15x step throughput
    # and ~halved activation VRAM are the upside on the table).
    mixed_precision: bool = False
    # Whether the conv layers' declared L1/L2 penalties are ADDED to the training objective.
    # v1 default False: the declarations were dead code since inception (a custom loop only
    # applies them by consuming model.losses, which nothing did), and activating them at the
    # declared, never-calibrated coefficients measurably degraded the model in a 5-seed A/B
    # (recall@0.01FPR median .984 -> .954, worst seed .72; 1-4 latent dims per seed pushed
    # below the active-units threshold). reg_loss is computed and recorded regardless, so
    # every run observes what the penalties WOULD be; coefficient calibration is a tracked
    # follow-up issue. Flipping this changes training numerics — A/B-gate like the other
    # numerics flags.
    regularization_active: bool = False


@dataclass
class ReproducibilityConfig:
    """Random-state configuration (#279): ONE root seed for the whole pipeline"""

    # Root seed for every random stream in BOTH pipelines: synthetic data generation, dataset
    # split/shuffles, TF weight init, the VAE sampling layer (training AND inference), the
    # Random Forest, UMAP/KMeans plot fits, and plot subsampling — each consumer derives an
    # independent stream (see aetherscan.seeding). Defaults to a concrete value so runs are
    # reproducible out of the box (#279 flipped this from the historical None); None restores
    # OS entropy (non-reproducible, warned once). Bit-exact GPU reproducibility additionally
    # needs tf_deterministic_ops and identical hardware/software (GPU count, TF version).
    seed: int | None = 11
    # Force deterministic TF/cuDNN op implementations (tf.config.experimental
    # .enable_op_determinism) at some speed cost. Only meaningful alongside `seed`.
    tf_deterministic_ops: bool = False


@dataclass
class RandomForestConfig:
    """Random Forest configuration"""

    n_estimators: int = 1000  # Number of trees
    bootstrap: bool = (
        True  # Whether to use bootstrap sampling when building each tree (True = bagging)
    )
    max_features: str = "sqrt"  # Random feature selection (sqrt, log2, float)
    n_jobs: int = -1  # Number of parallel jobs to run (-1 = use all available CPU cores)
    # Explicit override for the RF random_state. None (the default) derives the seed from
    # reproducibility.seed via STREAM_RF (#279); the deprecated --rf-seed flag sets this.
    seed: int | None = None

    # Latent-representation variant the RF consumes (#282). Training sweeps every variant on
    # one shared dataset/split and records the empirical winner here before final_save, so
    # config_{tag}.json tells inference exactly how to rebuild features (never hardcoded).
    # See aetherscan.latent_variants.VARIANT_ORDER for the catalogue; "z" is the legacy baseline.
    latent_variant: str = "z"
    # ACTIVE latent dims measured at training (Burda et al.; None until a sweep ran) —
    # persisted so inference rebuilds the z_mean_logvar_active layout identically
    active_dims: list[int] | None = None
    # Extra sampled-z draws per cadence for the z_aug variant's training rows
    z_aug_draws: int = 4
    # Variance floor for a latent dim to count as ACTIVE (Burda et al. AU with z_mean
    # variance); gates the active-dims variant + the posterior-collapse guard (#282)
    active_units_threshold: float = 0.01
    # Variant selection (#282): primary metric is recall at this false-positive rate on the
    # selection split; the winner must beat every simpler (fewer-feature) variant by more
    # than a bootstrap CI of the recall difference, else the simpler variant wins the tie
    selection_max_fpr: float = 0.01
    selection_bootstrap_rounds: int = 500
    # Probability calibration (#282): auto-fit a calibrator on the held-out calibration
    # split when the winner's measured ECE exceeds max_ece; isotonic when the calibration
    # split has at least calibration_min_isotonic rows, else sigmoid/Platt (isotonic
    # overfits small sets); kept only if it improves ECE (and does not worsen Brier) on the
    # held-out test split. calibration_active/method are RECORDED BY TRAINING for the saved
    # config — inference reads them to decide whether to load rf_calibrator_{tag}.joblib.
    max_ece: float = 0.05
    calibration_min_isotonic: int = 1000
    calibration_active: bool = False
    calibration_method: str | None = None
    # Fractions of the val split carved out for selection / calibration (remainder = the
    # held-out test split that release metrics are reported on — best-of-N on the selection
    # split alone would be optimistically biased)
    val_selection_fraction: float = 0.5
    val_calibration_fraction: float = 0.25
    # Screening-threshold validation (#282): warn if the two-pass cascade loses more than
    # this much recall vs MC-scoring everything at the science threshold on the test split
    screen_recall_tolerance: float = 0.0


# NOTE: verify that our current GPU config gracefully handles cases where the node has a single GPU (vs multiple)
# TODO: add a way to specify (either number or name) the specific GPUs on a system we wish to use (currently defaults to all available). extend to cli.py too
# TODO: run performance benchmarks using different num_gpus on a single node (and in future, multi-node as well)
@dataclass
class GPUConfig:
    """GPU runtime configuration"""

    # If None, the strategy uses every GPU visible to TF. If set to a positive int N, the
    # strategy is restricted to the first N physical GPUs (the rest are left untouched for
    # other workloads on the node). Validated against batch/sample divisibility in cli.py
    # before being applied; runtime mismatch (N > available GPUs) aborts in
    # setup_gpu_strategy rather than silently downgrading, so we never propagate batch
    # sizes that were validated against a different replica count.
    num_replicas: int | None = None
    per_gpu_memory_limit_mb: int | None = None
    nccl_num_packs: int = 2
    use_async_allocator: bool = True
    # TF_GPU_THREAD_MODE: "gpu_private" gives each GPU dedicated kernel-launch threads that
    # tf.data host work cannot steal — the standard NGC lever for input-pipeline h2d/scheduling
    # interference (measured at ~7.6% of step throughput on blpc3, benchmarks/README.md
    # "Corrected ceiling decomposition"). "global" restores TF's default shared pool;
    # "gpu_shared" is the third TF-supported value. Applied in setup_gpu_strategy before the
    # GPU runtime initializes; inert in the producer tree (CUDA-blanked). NOTE: this config is
    # the source of truth — "global" actively CLEARS any shell-set TF_GPU_THREAD_MODE/COUNT
    # (same semantics as use_async_allocator vs TF_GPU_ALLOCATOR above), so an env-only
    # override will not survive; set the config field instead.
    gpu_thread_mode: str = "gpu_private"
    # TF_GPU_THREAD_COUNT: threads per GPU when gpu_thread_mode="gpu_private" (TF default 2)
    gpu_thread_count: int = 2


# TODO: make sure the entire pipeline respects DataConfig() values, instead of hard coding
@dataclass
class DataConfig:
    """Data processing configuration"""

    num_observations: int = 6  # Per cadence snippet (3 ON, 3 OFF)
    width_bin: int = 4096  # Frequency bins per observation
    downsample_factor: int = 8  # Frequency bins downsampling factor
    time_bins: int = 16  # Time bins per observation
    freq_resolution: float = 2.7939677238464355  # Hz
    time_resolution: float = 18.25361108  # seconds

    # TODO: experiment with larger chunk sizes (how to track chunk processing efficiency)
    # Note the following heuristics/constraints, which only apply to training
    # max_backgrounds_per_file = max_chunks_per_file * background_load_chunk_size
    # max_backgrounds_total = min(max_backgrounds_per_file * num_files, num_target_backgrounds)
    num_target_backgrounds: int = 45000  # Number of background cadences to load
    background_load_chunk_size: int = (
        15000  # Maximum cadences to process at once during load_train_data()
    )
    max_chunks_per_file: int = 1  # Maximum chunks to load from a single file

    # NOTE: should this be in DataConfig, InferenceConfig, or a new dataclass entirely (e.g. PreprocessingConfig)
    # NOTE: should be renamed test_background_load_chunk_size? or no because inference_files still generate .npy which uses this same chunk size?
    # TODO: experiment with larger chunk sizes (how to track chunk processing efficiency)
    # We only specify chunk size during inference, since we assume all backgrounds from all files must be loaded
    inference_background_load_chunk_size: int = (
        50000  # Maximum cadences to process at once during load_inference_data()
    )

    # Data files
    # Note, Python dataclasses don't allow mutable objects (e.g. lists) to be used as defaults,
    # since Python will create that object once when the class is defined, rather than each time
    # a new object of that class is instantiated. This means that all instances of that class
    # would share the same mutable object in memory (i.e. if we modified train_files in one
    # instance, it would affect all other instances -- a dangerous bug).
    # The default_factory parameter takes a callable (lambda function) that's called each time a
    # new instance is created, ensuring each instance gets its own independent list, preventing
    # the shared-state bug. Note that once created, the list behaves identical to any other list
    train_files: list[str] = field(
        default_factory=lambda: [
            "real_filtered_LARGE_HIP110750.npy",
            "real_filtered_LARGE_HIP13402.npy",
            "real_filtered_LARGE_HIP8497.npy",
        ]
    )
    # TODO: add comment specifying test_files requirements & behavior (.npy instead of .csv containing individual .h5 files, in inference_files)
    test_files: list[str] = field(default_factory=lambda: ["real_filtered_LARGE_test_HIP15638.npy"])
    # Each entry is a CSV path (resolved via get_inference_file_path) whose rows
    # describe individual .h5 observations to be grouped into cadences.
    # If non-None, takes precedence over test_files during inference and triggers
    # the energy detection preprocessing pipeline.
    inference_files: list[str] | None = field(default_factory=lambda: None)


@dataclass
class TrainingConfig:
    # Reproducibility now lives in ReproducibilityConfig (#279): one root seed shared by both
    # pipelines instead of a training-only seed plus an independent rf.seed.

    num_training_rounds: int = 20
    epochs_per_round: int = 100

    # Posterior-collapse guard (#282): a latent dim counts ACTIVE while its batch-mean KL
    # exceeds kl_epsilon; a round alarms (advisory WARNING, never fatal) when the active
    # fraction drops below min_active_units_fraction or any dim sits under epsilon for
    # `patience` consecutive epochs
    posterior_collapse_kl_epsilon: float = 0.01
    min_active_units_fraction: float = 0.5
    posterior_collapse_patience: int = 5

    num_samples_beta_vae: int = 499200
    num_samples_rf: int = 99840  # NOTE: come back to this later
    train_val_split: float = 0.8

    per_replica_batch_size: int = (
        128  # Throughput-optimal per GPU (see benchmarks/README.md GPU sweep)
    )
    # Gradient-accumulation target. 7680 = lcm(per_replica_batch_size * {4,5,6}), so effective_batch_size
    # is divisible by per_replica_batch_size * num_replicas on 4-, 5-, or 6-GPU hosts (and divides the
    # 399360-sample train split); it makes the defaults valid on every supported GPU count, including
    # the 5-GPU Blackwell cluster where the previous 3072 failed. On a fixed GPU count you can override
    # with a smaller multiple (e.g. --effective-batch-size 3072 on 4 or 6 GPUs) to accumulate over fewer
    # micro-batches; per-GPU throughput is unchanged (set by per_replica_batch_size).
    effective_batch_size: int = 7680
    # Validation / viz / RF-latent-generation batch per GPU. Its global size (64 * num_replicas) must
    # divide the val split (99840), num_samples_rf (99840), and latent_viz_num_cadences_per_type * 4;
    # 64 is paired with latent_viz_num_cadences_per_type=960 (below) so all three hold on 4-, 5-, or
    # 6-GPU hosts. Runs off the training hot path, but 64 (vs the bare divisibility floor of 16 at
    # latent_viz=240) keeps validation / latent-generation out of the small-batch launch-overhead
    # regime (see the encode sweep in benchmarks/README.md) — a net win over a full run's ~2000
    # validations. Raising V requires raising latent_viz in lockstep (V=128 needs latent_viz=1920).
    per_replica_val_batch_size: int = 64

    # Signal injection params
    # TODO: experiment with larger chunk sizes (how to track chunk processing efficiency)
    signal_injection_chunk_size: int = (
        50000  # Maximum cadences to process at once during data generation
    )
    # Tuned to 64 via smoke-scale (n=8192) task-size sweeps on bla0 (96c) + blpc3 (32c):
    # finer tasks load-balance the create_true_double straggler with negligible per-task
    # overhead; 64 was near-optimal on both (256 ran ~2x slower on bla0).
    # TODO: re-confirm at production sample sizes (~500k) before treating as final.
    data_gen_task_size: int = 64  # Cadences per batched worker task (workers write results straight into the round memmap)
    # On-disk dtype of the three round cadence arrays (main/true/false). "float16" halves the
    # round footprint (~294.5 -> ~147 GB at full scale), the gather volume, and the page-cache
    # working set — the lever that keeps overlapped epochs at page-cache speed once two rounds
    # no longer fit in RAM — at a quantization cost of <= 2^-12 on the [0, 1] log-normalized
    # inputs. DEFAULT FLIPPED to "float16" 2026-07-29 after passing the val-metric A/B gate:
    # 4 seeds, every score metric inside the 6-seed control spread (val AUC >= .9979, recalls
    # .9619-.9907 vs control .9449-.9907), AU within the controls' own 6-8 seed variation, no
    # calibration trips. The gather map upcasts to float32 host-side, so the training graph
    # and loss math are unchanged either way; labels + lognorm sidecars stay float32.
    # Recorded in each round's .done manifest and validated on resume/reuse ("float32"
    # restores the historical behavior byte-for-byte).
    round_array_dtype: str = "float16"

    # Round data pipeline params (disk-backed per-round datasets, see round_data.py)
    round_data_dir: str | None = None  # Defaults to get_training_file_path("round_data") at runtime
    overlap_data_generation: bool = (
        True  # Generate round k+1's data in a background process while round k trains
    )
    keep_round_data: bool = (
        False  # Retain each round's on-disk data after its training completes (debugging)
    )
    # TODO: tune to include sufficient info without bottlenecking training
    plot_injection_subsampling_count: int = (
        100000  # Max points per series in injection_stats scatter plots
    )
    # TODO: tune to include sufficient info without bottlenecking training
    plot_injection_outlier_percentile: float = (
        99.0  # Always include injection_stats points beyond this percentile
    )

    # Latent space visualization params
    # Cadences per signal type for the latent-space viz batch (total = 4×). 960 rather than a smaller
    # value so latent_viz*4 (3840) is divisible by per_replica_val_batch_size * num_replicas on 4-, 5-,
    # or 6-GPU hosts (this is the binding constraint that lets per_replica_val_batch_size be 64 instead
    # of 16); it also yields denser latent-space plots. Must stay <= the val split (99840).
    latent_viz_num_cadences_per_type: int = 960
    latent_viz_step_interval: int = 10  # Capture snapshot every N training steps
    latent_viz_umap_fit_max_samples: int = (
        100_000  # Max pooled vectors for UMAP fit (rest are projected via .transform())
    )
    # Note, Python dataclasses don't allow mutable objects (e.g. lists) to be used as defaults,
    # since Python will create that object once when the class is defined, rather than each time
    # a new object of that class is instantiated. This means that all instances of that class
    # would share the same mutable object in memory (i.e. if we modified latent_viz_umap_n_neighbors
    # in one instance, it would affect all other instances -- a dangerous bug).
    # The default_factory parameter takes a callable (lambda function) that's called each time a
    # new instance is created, ensuring each instance gets its own independent list, preventing
    # the shared-state bug. Note that once created, the list behaves identical to any other list
    latent_viz_umap_n_neighbors: list[int] = field(
        # UMAP n_neighbors values to sweep
        # n_neighbors constrains the size of the local neighborhood UMAP will look at when attempting
        # to learn the manifold structure of the data. Lower values lead to better local structure,
        # whereas larger values yield better global structure
        # - 5: fine-grained local structure
        # - 15: UMAP default, good baseline
        # - 50: global topology emphasis
        # (30 was dropped from the default sweep 2026-07: it sat between 15 and 50 without
        # adding a distinct view, and each nn value costs 3 min_dist x 2 level combos)
        default_factory=lambda: [5, 15, 50]
    )
    latent_viz_umap_min_dist: list[float] = field(
        # UMAP min_dist values to sweep
        # min_dist controls how tightly points in the lower-dimensional representation can be packed.
        # Lower values result in clumpier embeddings (useful for observing clusters), whereas larger
        # values preserve more of the broad topological structure
        # - 0.0: maximum cluster tightness
        # - 0.1: UMAP default, slight breathing room
        # - 0.5: spread out, reveals continuous gradients
        default_factory=lambda: [0.0, 0.1, 0.5]
    )
    latent_viz_gif_max_frames: int = 500  # Max frames in output GIF (log-spaced subsampling)
    latent_viz_gif_duration_ms: int = 100  # Milliseconds per frame in output GIF

    # Latent traversal params (decoder-based interpretation of the latent dims; see
    # TrainingPipeline.plot_latent_traversal)
    latent_traversal_every_round: bool = (
        False  # Also render traversal figures at the end of every training round
    )
    latent_traversal_num_steps: int = (
        7  # Steps per latent dim (odd & >= 3 so the center column is the unperturbed decode)
    )
    latent_traversal_max_sigma: float = (
        3.0  # Traversal range in per-dim standard deviations: steps span ±max_sigma (> 0)
    )

    # RF visualization params
    shap_max_samples_summary: int = 5000  # Samples for SHAP summary/dependence computation
    shap_max_samples_interaction: int = (
        1500  # Samples for SHAP interaction values (O(F^2) per sample)
    )
    shap_top_k_features_dependence: int = (
        48  # Number of dependence plot panels (all 6 obs * 8 dims by default)
    )
    rf_decision_boundary_grid_size: int = (
        150  # Grid resolution for decision boundary contour (grid_size x grid_size)
    )
    rf_decision_boundary_max_points: int = (
        5000  # Subsample val points for decision boundary plot legibility
    )

    # Model-quality gate params (issue #139 Gate 1)
    min_val_auc: float = 0.0  # Opt-in floor on the RF's validation ROC-AUC; 0.0 disables the check. When set and unmet after the RF fit, training logs a loud WARNING (reaches the Slack summary) rather than failing the run

    # Curriculum learning params
    snr_base: int = 10
    initial_snr_range: int = 40
    final_snr_range: int = 10
    curriculum_schedule: str = "exponential"  # "linear", "exponential", "step"
    exponential_decay_rate: float = -3.0  # How quickly schedule should progress from easy to hard (must be <0) (more negative = less easy rounds & more hard rounds)
    # TODO: generalize this to receive a step schedule (as a list/dict?) validate that len(list/dict) is divisible by num_training_rounds
    step_easy_rounds: int = 5  # Number of rounds with easy signals
    step_hard_rounds: int = 15  # Number of rounds with challenging signals

    # Adaptive LR params
    base_learning_rate: float = 0.001
    min_learning_rate: float = 1e-6
    min_pct_improvement: float = 0.001  # 0.1% val loss improvement
    patience_threshold: int = 3  # consecutive epochs with no improvement
    reduction_factor: float = 0.2  # 20% LR reduction

    # NOTE: should we try an exponential backoff?
    # Fault tolerance params
    max_retries: int = 3
    retry_delay: int = 60  # seconds


@dataclass
class InferenceConfig:
    """Inference configuration"""

    # TODO: have a startup function that sets encoder_path, rf_path, and config_path using load_dir & load_tag if either/both are None
    encoder_path: str = None
    rf_path: str = None
    config_path: str = None
    # Per-replica encode batch in SNIPPETS (each snippet = num_observations independent
    # encoder inputs, so 256 snippets = 1,536 observation forwards — near the measured
    # ~1024-obs single-GPU encode throughput peak; see benchmarks/README.md). The old 2048
    # default was a units conflation (#298 I4): 2048 SNIPPETS = a 12,288-obs per-replica
    # forward, past the throughput peak and above the bench's documented >~8192-obs int32
    # launch-config abort. Also the cap of _distributed_encode's bucketed batch geometry.
    per_replica_batch_size: int = 256
    classification_threshold: float = 0.99
    # Two-pass cascade (#282). Pass 1 scores EVERY snippet deterministically (z_mean-based
    # features) against this permissive screening threshold, tuned for recall — its only job
    # is "definitely not a candidate". Pass 2 re-scores the survivors with mc_draws seeded
    # latent samples and carries the actual science threshold (classification_threshold
    # above, applied to the MC mean). The two are a cascade, not two ANDed criteria.
    screening_threshold: float = 0.5
    mc_draws: int = 32
    # Reference cloud (#282): pass 2 also MC-scores a seeded uniform reservoir subsample of
    # the pass-1 REJECTS, persisted per run, so the candidate uncertainty plot compares each
    # candidate against the survey population rather than against other candidates. 0
    # disables the cloud (and the plot's background).
    reference_cloud_size: int = 10000
    # Streaming-loop prefetch depth (#298 N2): cadences preprocessed+loaded concurrently
    # ahead of the GPU stage. Depth 1 = the historical behavior (one in-flight cadence);
    # depth 2 overlaps one cadence's disk-bound energy detection with the previous one's
    # decompression-bound extraction and fills the worker pool during per-cadence serial
    # sections, at the cost of one extra in-flight cadence of RAM (stamps + loaded array,
    # up to ~10-20 GB each). Default stays 1 until the post-I1 on-cluster A/B decides the
    # flip — per-cadence outputs are identical at any depth (results are consumed in
    # catalog order and seeding keys on the catalog index).
    prefetch_depth: int = 1

    # NOTE: come back to this later (is this the optimal grouping?)
    # Energy detection preprocessing
    cadence_group_by_cols: list[str] = field(
        default_factory=lambda: ["Target", "Session", "Band", "Cadence ID", "Frequency"]
    )
    cadence_h5_path_col: str = ".h5 path"
    cadence_expected_obs: int = 6  # expected observations per cadence (ABACAD)

    # NOTE: come back to this later (are these params correct?)
    coarse_channel_width: int = 1048576
    # Progress-logging chunk size for energy detection (coarse channels per log line).
    # None -> use manager.n_processes. Parallelism itself comes from the persistent worker
    # pool (one fused task per coarse channel), not from this knob.
    coarse_channel_log_interval: int | None = None
    # Bandpass flattening method for energy detection: "pfb" divides each coarse channel by
    # the instrument's static polyphase-filterbank response (computed once per run); "spline"
    # fits and subtracts a spline per coarse channel per file (the historical, data-driven
    # method).
    bandpass_method: str = "pfb"
    # PFB prototype-filter taps per coarse channel. Instrument-dependent: 12 is the GBT /
    # Breakthrough Listen backend default, and it must match the backend that produced the
    # .h5 files being processed.
    pfb_taps_per_channel: int = 12
    # Opt-in debug artifact: save a raw-vs-flattened overlay plot (a few sampled coarse
    # channels per cadence) under {output_path}/plots/inference/.
    bandpass_debug_plot: bool = False
    spline_order: int = 16
    detection_window_size: int = 256
    detection_step_size: int = 128
    stat_threshold: float = 2048.0
    stamp_width: int = 4096
    # Downsample stamps along frequency at extraction time (by data.downsample_factor), so
    # the per-cadence .npy stores stamp_width // downsample_factor bins (~8x smaller at
    # defaults). Disable to archive raw-resolution stamps; loading handles both layouts.
    store_downsampled_stamps: bool = True

    # NOTE: come back to this later (is overlap search implemented correctly? redo analysis from peter forwards search paper)
    overlap_search: bool = True
    overlap_fraction: float = 0.5

    # NOTE: come back to this later (placeholder for future drop logic — wired but inert)
    discard_side_channels: bool = False
    side_channel_count: int = 0

    # Per-cadence .npy output directory for energy-detection preprocessing. None (default)
    # resolves per CSV to {data_path}/inference/preprocessed/<csv_stem>_<save_tag>/ — tag
    # scoping keeps runs isolated (same-tag retries resume; a new tag starts clean). Set
    # explicitly to share/reuse one directory across runs and CSVs.
    preprocess_output_dir: str | None = None

    # Visualization suite (aetherscan.inference_viz): rendered at the end of a streaming
    # CSV inference run, saved under {output_path}/plots/inference/{save_tag}/ and uploaded
    # to Slack. Every figure is individually exception-guarded — a plot bug can never kill
    # a science run.
    inference_viz_enabled: bool = True
    # Number of top-statistic stamps shown in the stamp gallery (6-obs waterfall grids).
    stamp_gallery_top_k: int = 12
    # Cap on per-candidate figures (candidate_{i}_{tag}.png), highest confidence first.
    max_candidate_plots: int = 50

    # NOTE: come back to this later (is this implemented correctly?)
    # Fault tolerance
    max_retries: int = 3
    retry_delay: int = 60  # seconds


@dataclass
class HFConfig:
    """HuggingFace Hub integration configuration"""

    # Target model repo for weight upload (train) and download (inference).
    repo_id: str = "zachtheyek/aetherscan"
    # Opt-in: publish the final artifacts + a generated model card after training completes
    # (requires HF_TOKEN in the environment; default off = local-only).
    upload_after_training: bool = False
    # Inference pin: HF revision (tag/branch/commit) to download artifacts from. None
    # resolves to the latest release tag (highest semver vX.Y.Z); a release tag is required.
    revision: str | None = None


@dataclass
class CheckpointConfig:
    """Checkpoint configuration"""

    load_dir: str | None = None
    load_tag: str | None = None
    start_round: int = 1
    # Resolved once at runtime by cli.resolve_save_tag (in main(), before init_logger) to
    # {command}_{YYYYMMDD_HHMMSS}. None until then — the pipeline always runs through the CLI,
    # which sets it before any stage reads it.
    save_tag: str | None = None
    # Override the fail-early save-tag dedup guards (local artifact/DB collisions and the
    # HF-side tag check) for an explicitly-provided save_tag.
    force_tag: bool = False

    def infer_start_round(self):
        """Infer start_round from load_tag"""
        if self.load_tag and self.load_tag.startswith("round_"):
            self.start_round = (
                int(self.load_tag.split("_", 1)[1]) + 1
            )  # Start from the round proceeding model checkpoint (round_XX + 1)


class Config:
    """Main configuration class"""

    _instance = None  # Stores singleton instance
    _lock = threading.Lock()  # Ensures thread safety on object initialization

    # __new__ allocates the object in memory (constructor at the object-creation level)
    # __init__ initializes the object's attributes after it's created
    # since __new__ is called before __init__ every time we instantiate a class,
    # by overriding __new__, we can short-circuit object creation entirely, and control whether a
    # new instance is created, or just return the existing instance
    def __new__(cls):
        # Double-checked locking pattern:
        # First check if _instance is None, without lock (for performance)
        if cls._instance is None:
            # If None, acquire the lock to serialize the initialization path,
            # preventing race conditions (2 threads violating singleton semantics)
            with cls._lock:
                # Check if _instance is None again inside the lock
                # (since multiple threads can be calling simultaneously)
                if cls._instance is None:
                    # If still None, only then we construct the singleton instance
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False  # Mark as not initialized (for __init__)
        # Return the same instance for all subsequent constructor calls
        return cls._instance

    def __init__(self):
        """Initialize configuration"""
        # Note, __init__ is triggered every time the class's constructor is called,
        # even if __new__ returned the existing singleton instance
        # Hence, we use the _initialized flag to make sure __init__ only runs once
        if self._initialized:
            return

        self._initialized = True

        self.db = DBConfig()
        self.manager = ManagerConfig()
        self.monitor = MonitorConfig()
        self.logger = LoggerConfig()
        self.reproducibility = ReproducibilityConfig()
        self.beta_vae = BetaVAEConfig()
        self.rf = RandomForestConfig()
        self.gpu = GPUConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.hf = HFConfig()
        self.checkpoint = CheckpointConfig()

        # Paths
        self.data_path = os.environ.get(
            "AETHERSCAN_DATA_PATH", "/datax/scratch/zachy/data/aetherscan"
        )
        self.model_path = os.environ.get(
            "AETHERSCAN_MODEL_PATH", "/datax/scratch/zachy/models/aetherscan"
        )
        self.output_path = os.environ.get(
            "AETHERSCAN_OUTPUT_PATH", "/datax/scratch/zachy/outputs/aetherscan"
        )

        # Coupled-defaults guard: inference.stamp_width and data.width_bin are independent
        # dataclass fields that must agree — energy-detection stamps are cut at stamp_width
        # while every loader/model surface is sized off width_bin. A divergent source-level
        # default edit would otherwise surface only at load time (load_inference_data fails
        # safe, but late and per-file); fail here, at config init. CLI overrides are
        # validated separately in cli.collect_validation_errors.
        if self.inference.stamp_width != self.data.width_bin:
            raise ValueError(
                f"Config defaults diverged: inference.stamp_width "
                f"({self.inference.stamp_width}) must equal data.width_bin "
                f"({self.data.width_bin})"
            )

    def resolved_rf_seed(self) -> int:
        """
        The Random Forest random_state actually used this run (#279): rf.seed when explicitly
        overridden (deprecated --rf-seed), else derived from the root seed via STREAM_RF.
        Always a concrete int — with an unseeded root the derived value is OS entropy, so the
        RF still gets a set random_state (the #279 constraint: never drop one that exists).
        """
        # Deferred import: config must stay importable before/without the rest of the package
        from aetherscan.seeding import STREAM_RF, derive_seed  # noqa: PLC0415

        if self.rf.seed is not None:
            return self.rf.seed
        return derive_seed(self.reproducibility.seed, STREAM_RF)

    @classmethod
    def _reset(cls):
        """
        Teardown hook for thread-safe singleton
        Resets the config instance to None

        WARNING: Only use for testing or restarting the application
        Calling this while the config is active will cause issues.
        Do NOT call this method unless you know what you're doing
        """
        # Acquire lock to prevent race conditions
        with cls._lock:
            # Discard the singleton instance by removing the global reference
            # Guarantees the next constructor call will produce a fresh instance
            # Note, resources held by the old instance will remain alive unless explicitly closed beforehand
            cls._instance = None

    def get_training_file_path(self, filename: str, base_path: str | None = None) -> str:
        """Get full path for training data files. base_path overrides self.data_path (used by
        validate_args, which runs before a --data-path override is applied to the singleton)."""
        return os.path.join(
            base_path if base_path is not None else self.data_path, "training", filename
        )

    def get_test_file_path(self, filename: str, base_path: str | None = None) -> str:
        """Get full path for testing data files. base_path overrides self.data_path (used by
        validate_args, which runs before a --data-path override is applied to the singleton)."""
        return os.path.join(
            base_path if base_path is not None else self.data_path, "testing", filename
        )

    def get_inference_file_path(self, filename: str, base_path: str | None = None) -> str:
        """Get full path for inference CSV files. base_path overrides self.data_path (used by
        validate_args, which runs before a --data-path override is applied to the singleton)."""
        return os.path.join(
            base_path if base_path is not None else self.data_path, "inference", filename
        )

    def get_file_subset(self, filename: str) -> tuple[int | None, int | None]:
        """Get subset parameters for a file (start, end indices)"""
        # Option to define subsets for specific files to manage memory usage
        subset_map = {
            "real_filtered_LARGE_HIP110750.npy": (None, None),  # Shape: (14567, 6, 16, 4096)
            "real_filtered_LARGE_HIP13402.npy": (None, None),  # Shape: (14567, 6, 16, 4096)
            "real_filtered_LARGE_HIP8497.npy": (None, None),  # Shape: (14567, 6, 16, 4096)
            "real_filtered_LARGE_testHIP83043.npy": (None, None),  # Shape: (14567, 6, 16, 4096)
        }
        return subset_map.get(filename, (None, None))

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization"""
        return {
            "paths": {
                "data_path": self.data_path,
                "model_path": self.model_path,
                "output_path": self.output_path,
            },
            "db": {
                "get_connection_timeout": self.db.get_connection_timeout,
                "stop_writer_timeout": self.db.stop_writer_timeout,
                "write_interval": self.db.write_interval,
                "write_buffer_max_size": self.db.write_buffer_max_size,
                "write_retry_delay": self.db.write_retry_delay,
                "flush_timeout": self.db.flush_timeout,
                "bulk_chunk_rows": self.db.bulk_chunk_rows,
                "bulk_queue_max_items": self.db.bulk_queue_max_items,
                "stop_drain_timeout": self.db.stop_drain_timeout,
            },
            "manager": {
                "n_processes": self.manager.n_processes,
                "chunks_per_worker": self.manager.chunks_per_worker,
                "pool_terminate_timeout": self.manager.pool_terminate_timeout,
            },
            "monitor": {
                "get_gpu_timeout": self.monitor.get_gpu_timeout,
                "stop_monitor_timeout": self.monitor.stop_monitor_timeout,
                "monitor_interval": self.monitor.monitor_interval,
                "monitor_retry_delay": self.monitor.monitor_retry_delay,
                "annotate_stages": self.monitor.annotate_stages,
                "dashboard_enabled": self.monitor.dashboard_enabled,
                "dashboard_port": self.monitor.dashboard_port,
                "benchmark_report_enabled": self.monitor.benchmark_report_enabled,
            },
            "logger": {
                "console_level": self.logger.console_level,
                "file_level": self.logger.file_level,
                "slack_level": self.logger.slack_level,
                "slack_enabled": self.logger.slack_enabled,
                "slack_channel": self.logger.slack_channel,
                "slack_username": self.logger.slack_username,
                "slack_timeout": self.logger.slack_timeout,
                "slack_retry_attempts": self.logger.slack_retry_attempts,
                "slack_buffer_size": self.logger.slack_buffer_size,
                "slack_flush_interval": self.logger.slack_flush_interval,
                "slack_broadcast_level": self.logger.slack_broadcast_level,
            },
            "beta_vae": {
                "latent_dim": self.beta_vae.latent_dim,
                "dense_layer_size": self.beta_vae.dense_layer_size,
                "kernel_size": self.beta_vae.kernel_size,
                "beta": self.beta_vae.beta,
                "alpha": self.beta_vae.alpha,
                "mixed_precision": self.beta_vae.mixed_precision,
                "regularization_active": self.beta_vae.regularization_active,
            },
            "reproducibility": {
                "seed": self.reproducibility.seed,
                "tf_deterministic_ops": self.reproducibility.tf_deterministic_ops,
                # Provenance only (#279): the concrete values derived from the root this run.
                # Not a settable field — apply_saved_config skips unknown sub-keys, so a
                # restored config re-derives from the root instead of pinning these.
                "derived_rf_seed": self.resolved_rf_seed(),
            },
            "rf": {
                "n_estimators": self.rf.n_estimators,
                "bootstrap": self.rf.bootstrap,
                "max_features": self.rf.max_features,
                "n_jobs": self.rf.n_jobs,
                # The override field (None = derived from reproducibility.seed; see
                # reproducibility.derived_rf_seed for the value actually used)
                "seed": self.rf.seed,
                "latent_variant": self.rf.latent_variant,
                "active_dims": self.rf.active_dims,
                "z_aug_draws": self.rf.z_aug_draws,
                "active_units_threshold": self.rf.active_units_threshold,
                "selection_max_fpr": self.rf.selection_max_fpr,
                "selection_bootstrap_rounds": self.rf.selection_bootstrap_rounds,
                "max_ece": self.rf.max_ece,
                "calibration_min_isotonic": self.rf.calibration_min_isotonic,
                "calibration_active": self.rf.calibration_active,
                "calibration_method": self.rf.calibration_method,
                "val_selection_fraction": self.rf.val_selection_fraction,
                "val_calibration_fraction": self.rf.val_calibration_fraction,
                "screen_recall_tolerance": self.rf.screen_recall_tolerance,
            },
            "gpu": {
                "num_replicas": self.gpu.num_replicas,
                "per_gpu_memory_limit_mb": self.gpu.per_gpu_memory_limit_mb,
                "nccl_num_packs": self.gpu.nccl_num_packs,
                "use_async_allocator": self.gpu.use_async_allocator,
                "gpu_thread_mode": self.gpu.gpu_thread_mode,
                "gpu_thread_count": self.gpu.gpu_thread_count,
            },
            "data": {
                "num_observations": self.data.num_observations,
                "width_bin": self.data.width_bin,
                "downsample_factor": self.data.downsample_factor,
                "time_bins": self.data.time_bins,
                "freq_resolution": self.data.freq_resolution,
                "time_resolution": self.data.time_resolution,
                "num_target_backgrounds": self.data.num_target_backgrounds,
                "background_load_chunk_size": self.data.background_load_chunk_size,
                "max_chunks_per_file": self.data.max_chunks_per_file,
                "inference_background_load_chunk_size": self.data.inference_background_load_chunk_size,
                "train_files": self.data.train_files,
                "test_files": self.data.test_files,
                "inference_files": self.data.inference_files,
            },
            "training": {
                "num_training_rounds": self.training.num_training_rounds,
                "epochs_per_round": self.training.epochs_per_round,
                "posterior_collapse_kl_epsilon": self.training.posterior_collapse_kl_epsilon,
                "min_active_units_fraction": self.training.min_active_units_fraction,
                "posterior_collapse_patience": self.training.posterior_collapse_patience,
                "num_samples_beta_vae": self.training.num_samples_beta_vae,
                "num_samples_rf": self.training.num_samples_rf,
                "train_val_split": self.training.train_val_split,
                "per_replica_batch_size": self.training.per_replica_batch_size,
                "effective_batch_size": self.training.effective_batch_size,
                "per_replica_val_batch_size": self.training.per_replica_val_batch_size,
                "signal_injection_chunk_size": self.training.signal_injection_chunk_size,
                "data_gen_task_size": self.training.data_gen_task_size,
                "round_array_dtype": self.training.round_array_dtype,
                "round_data_dir": self.training.round_data_dir,
                "overlap_data_generation": self.training.overlap_data_generation,
                "keep_round_data": self.training.keep_round_data,
                "plot_injection_subsampling_count": self.training.plot_injection_subsampling_count,
                "plot_injection_outlier_percentile": self.training.plot_injection_outlier_percentile,
                "latent_viz_num_cadences_per_type": self.training.latent_viz_num_cadences_per_type,
                "latent_viz_step_interval": self.training.latent_viz_step_interval,
                "latent_viz_umap_fit_max_samples": self.training.latent_viz_umap_fit_max_samples,
                "latent_viz_umap_n_neighbors": self.training.latent_viz_umap_n_neighbors,
                "latent_viz_umap_min_dist": self.training.latent_viz_umap_min_dist,
                "latent_viz_gif_max_frames": self.training.latent_viz_gif_max_frames,
                "latent_viz_gif_duration_ms": self.training.latent_viz_gif_duration_ms,
                "latent_traversal_every_round": self.training.latent_traversal_every_round,
                "latent_traversal_num_steps": self.training.latent_traversal_num_steps,
                "latent_traversal_max_sigma": self.training.latent_traversal_max_sigma,
                "shap_max_samples_summary": self.training.shap_max_samples_summary,
                "shap_max_samples_interaction": self.training.shap_max_samples_interaction,
                "shap_top_k_features_dependence": self.training.shap_top_k_features_dependence,
                "rf_decision_boundary_grid_size": self.training.rf_decision_boundary_grid_size,
                "rf_decision_boundary_max_points": self.training.rf_decision_boundary_max_points,
                "min_val_auc": self.training.min_val_auc,
                "snr_base": self.training.snr_base,
                "initial_snr_range": self.training.initial_snr_range,
                "final_snr_range": self.training.final_snr_range,
                "curriculum_schedule": self.training.curriculum_schedule,
                "exponential_decay_rate": self.training.exponential_decay_rate,
                "step_easy_rounds": self.training.step_easy_rounds,
                "step_hard_rounds": self.training.step_hard_rounds,
                "base_learning_rate": self.training.base_learning_rate,
                "min_learning_rate": self.training.min_learning_rate,
                "min_pct_improvement": self.training.min_pct_improvement,
                "patience_threshold": self.training.patience_threshold,
                "reduction_factor": self.training.reduction_factor,
                "max_retries": self.training.max_retries,
                "retry_delay": self.training.retry_delay,
            },
            "inference": {
                "encoder_path": self.inference.encoder_path,
                "rf_path": self.inference.rf_path,
                "config_path": self.inference.config_path,
                "per_replica_batch_size": self.inference.per_replica_batch_size,
                "classification_threshold": self.inference.classification_threshold,
                "screening_threshold": self.inference.screening_threshold,
                "mc_draws": self.inference.mc_draws,
                "reference_cloud_size": self.inference.reference_cloud_size,
                "prefetch_depth": self.inference.prefetch_depth,
                "cadence_group_by_cols": self.inference.cadence_group_by_cols,
                "cadence_h5_path_col": self.inference.cadence_h5_path_col,
                "cadence_expected_obs": self.inference.cadence_expected_obs,
                "coarse_channel_width": self.inference.coarse_channel_width,
                "coarse_channel_log_interval": self.inference.coarse_channel_log_interval,
                "bandpass_method": self.inference.bandpass_method,
                "pfb_taps_per_channel": self.inference.pfb_taps_per_channel,
                "bandpass_debug_plot": self.inference.bandpass_debug_plot,
                "spline_order": self.inference.spline_order,
                "detection_window_size": self.inference.detection_window_size,
                "detection_step_size": self.inference.detection_step_size,
                "stat_threshold": self.inference.stat_threshold,
                "stamp_width": self.inference.stamp_width,
                "store_downsampled_stamps": self.inference.store_downsampled_stamps,
                "overlap_search": self.inference.overlap_search,
                "overlap_fraction": self.inference.overlap_fraction,
                "discard_side_channels": self.inference.discard_side_channels,
                "side_channel_count": self.inference.side_channel_count,
                "preprocess_output_dir": self.inference.preprocess_output_dir,
                "inference_viz_enabled": self.inference.inference_viz_enabled,
                "stamp_gallery_top_k": self.inference.stamp_gallery_top_k,
                "max_candidate_plots": self.inference.max_candidate_plots,
                "max_retries": self.inference.max_retries,
                "retry_delay": self.inference.retry_delay,
            },
            "hf": {
                "repo_id": self.hf.repo_id,
                "upload_after_training": self.hf.upload_after_training,
                "revision": self.hf.revision,
            },
            "checkpoint": {
                "load_dir": self.checkpoint.load_dir,
                "load_tag": self.checkpoint.load_tag,
                "start_round": self.checkpoint.start_round,
                "save_tag": self.checkpoint.save_tag,
                "force_tag": self.checkpoint.force_tag,
            },
        }


def init_config() -> Config:
    """
    Initialize global config instance (call once at startup)
    """
    config = Config()
    return config


# NOTE: suppress None return type for now to reduce pyright errors
# def get_config() -> Config | None:
def get_config() -> Config:
    """
    Get the global config instance
    """
    return Config._instance
