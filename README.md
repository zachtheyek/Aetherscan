# TODO:

- setup environment with conda, uv, docker, or podman
- download pre-commit hooks
- point to CONTRIBUTING.md, LICENSE, and CITATION.cff

---

---

# CLI Help Documentation

## No Command Provided

When running `aetherscan` with no command:

```
usage: [-h] {train,inference,evaluate} ...

Aetherscan Pipeline -- Breakthrough Listen's first end-to-end production-grade
DL pipeline for SETI @ scale

positional arguments:
  {train,inference,evaluate}
                        Command to execute
    train               Execute training pipeline
    inference           Execute inference pipeline
    evaluate            Execute evaluation pipeline

options:
  -h, --help            show this help message and exit
```

## Train Command Help

When running `aetherscan train --help`:

```
usage:  train [-h] [--data-path DATA_PATH] [--model-path MODEL_PATH]
              [--output-path OUTPUT_PATH] [--vae-latent-dim VAE_LATENT_DIM]
              [--vae-dense-layer-size VAE_DENSE_LAYER_SIZE]
              [--vae-kernel-size VAE_KERNEL_SIZE VAE_KERNEL_SIZE]
              [--vae-beta VAE_BETA] [--vae-alpha VAE_ALPHA]
              [--rf-n-estimators RF_N_ESTIMATORS]
              [--rf-bootstrap RF_BOOTSTRAP]
              [--rf-max-features RF_MAX_FEATURES] [--rf-n-jobs RF_N_JOBS]
              [--rf-seed RF_SEED] [--num-observations NUM_OBSERVATIONS]
              [--width-bin WIDTH_BIN] [--downsample-factor DOWNSAMPLE_FACTOR]
              [--time-bins TIME_BINS] [--freq-resolution FREQ_RESOLUTION]
              [--time-resolution TIME_RESOLUTION]
              [--num-target-backgrounds NUM_TARGET_BACKGROUNDS]
              [--background-load-chunk-size BACKGROUND_LOAD_CHUNK_SIZE]
              [--max-chunks-per-file MAX_CHUNKS_PER_FILE]
              [--train-files TRAIN_FILES [TRAIN_FILES ...]]
              [--test-files TEST_FILES [TEST_FILES ...]]
              [--num-training-rounds NUM_TRAINING_ROUNDS]
              [--epochs-per-round EPOCHS_PER_ROUND]
              [--num-samples-vae NUM_SAMPLES_VAE]
              [--num-samples-rf NUM_SAMPLES_RF]
              [--train-val-split TRAIN_VAL_SPLIT]
              [--per-replica-batch-size PER_REPLICA_BATCH_SIZE]
              [--global-batch-size GLOBAL_BATCH_SIZE]
              [--per-replica-val-batch-size PER_REPLICA_VAL_BATCH_SIZE]
              [--signal-injection-chunk-size SIGNAL_INJECTION_CHUNK_SIZE]
              [--snr-base SNR_BASE] [--initial-snr-range INITIAL_SNR_RANGE]
              [--final-snr-range FINAL_SNR_RANGE]
              [--curriculum-schedule CURRICULUM_SCHEDULE]
              [--exponential-decay-rate EXPONENTIAL_DECAY_RATE]
              [--step-easy-rounds STEP_EASY_ROUNDS]
              [--step-hard-rounds STEP_HARD_ROUNDS]
              [--base-learning-rate BASE_LEARNING_RATE]
              [--min-learning-rate MIN_LEARNING_RATE]
              [--min-pct-improvement MIN_PCT_IMPROVEMENT]
              [--patience-threshold PATIENCE_THRESHOLD]
              [--lr-reduction-factor LR_REDUCTION_FACTOR]
              [--max-retries MAX_RETRIES] [--retry-delay RETRY_DELAY]
              [--load-tag LOAD_TAG] [--load-dir LOAD_DIR]
              [--save-tag SAVE_TAG]

options:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        Path to data directory (overrides AETHERSCAN_DATA_PATH
                        environment variable)
  --model-path MODEL_PATH
                        Path to model directory (overrides
                        AETHERSCAN_MODEL_PATH environment variable)
  --output-path OUTPUT_PATH
                        Path to output directory (overrides
                        AETHERSCAN_OUTPUT_PATH environment variable)
  --vae-latent-dim VAE_LATENT_DIM
                        Dimensionality of the VAE latent space (bottleneck
                        size)
  --vae-dense-layer-size VAE_DENSE_LAYER_SIZE
                        Size of dense layer in VAE architecture (should match
                        frequency bins after downsampling)
  --vae-kernel-size VAE_KERNEL_SIZE VAE_KERNEL_SIZE
                        Kernel size for Conv2D layers as two integers (e.g.,
                        --vae-kernel-size 3 3)
  --vae-beta VAE_BETA   Beta coefficient for KL divergence loss term in beta-
                        VAE (controls disentanglement)
  --vae-alpha VAE_ALPHA
                        Alpha coefficient for clustering loss term in VAE
                        (controls cluster separation)
  --rf-n-estimators RF_N_ESTIMATORS
                        Number of decision trees in the random forest ensemble
  --rf-bootstrap RF_BOOTSTRAP
                        Whether to use bootstrap sampling when building trees
                        (enables bagging)
  --rf-max-features RF_MAX_FEATURES
                        Number of features to consider for splits: 'sqrt',
                        'log2', or a float (fraction of features)
  --rf-n-jobs RF_N_JOBS
                        Number of parallel jobs for random forest training (-1
                        uses all CPU cores)
  --rf-seed RF_SEED     Random seed for random forest reproducibility
  --num-observations NUM_OBSERVATIONS
                        Number of observations per cadence snippet (e.g., 6
                        for 3 ON + 3 OFF)
  --width-bin WIDTH_BIN
                        Number of frequency bins per observation (spectral
                        resolution)
  --downsample-factor DOWNSAMPLE_FACTOR
                        Downsampling factor for frequency bins (reduces
                        spectral dimension)
  --time-bins TIME_BINS
                        Number of time bins per observation (temporal
                        resolution)
  --freq-resolution FREQ_RESOLUTION
                        Frequency resolution in Hz (determined by instrument)
  --time-resolution TIME_RESOLUTION
                        Time resolution in seconds (determined by instrument)
  --num-target-backgrounds NUM_TARGET_BACKGROUNDS
                        Number of background (noise-only) cadences to load for
                        training data generation
  --background-load-chunk-size BACKGROUND_LOAD_CHUNK_SIZE
                        Maximum number of background cadences to process at
                        once during loading (memory management)
  --max-chunks-per-file MAX_CHUNKS_PER_FILE
                        Maximum number of chunks to load from a single data
                        file (limits per-file contribution)
  --train-files TRAIN_FILES [TRAIN_FILES ...]
                        Space-separated list of training data file names
                        (e.g., real_filtered_LARGE_HIP110750.npy)
  --test-files TEST_FILES [TEST_FILES ...]
                        Space-separated list of testing data file names (e.g.,
                        real_filtered_LARGE_test_HIP15638.npy)
  --num-training-rounds NUM_TRAINING_ROUNDS
                        Total number of training rounds in curriculum learning
                        schedule
  --epochs-per-round EPOCHS_PER_ROUND
                        Number of epochs to train the VAE per curriculum
                        learning round
  --num-samples-vae NUM_SAMPLES_VAE
                        Number of training samples to generate for beta-VAE
                        per round (must be divisible by 4)
  --num-samples-rf NUM_SAMPLES_RF
                        Number of training samples to generate for random
                        forest (must be divisible by 4)
  --train-val-split TRAIN_VAL_SPLIT
                        Fraction of data to use for training vs validation
                        (e.g., 0.8 = 80% train, 20% val)
  --per-replica-batch-size PER_REPLICA_BATCH_SIZE
                        Batch size per GPU/device replica during training
  --global-batch-size GLOBAL_BATCH_SIZE
                        Effective global batch size for gradient accumulation
                        across all replicas
  --per-replica-val-batch-size PER_REPLICA_VAL_BATCH_SIZE
                        Batch size per GPU/device replica during validation
  --signal-injection-chunk-size SIGNAL_INJECTION_CHUNK_SIZE
                        Maximum cadences to process at once during synthetic
                        signal injection (must be divisible by 4)
  --snr-base SNR_BASE   Base signal-to-noise ratio for curriculum learning
                        (minimum SNR difficulty level)
  --initial-snr-range INITIAL_SNR_RANGE
                        SNR range for initial (easiest) training rounds
                        (signals sampled from snr_base to snr_base +
                        initial_snr_range)
  --final-snr-range FINAL_SNR_RANGE
                        SNR range for final (hardest) training rounds (signals
                        sampled from snr_base to snr_base + final_snr_range).
                        Ignored if only training for 1 round
  --curriculum-schedule CURRICULUM_SCHEDULE
                        Curriculum difficulty progression schedule: 'linear',
                        'exponential', or 'step'
  --exponential-decay-rate EXPONENTIAL_DECAY_RATE
                        Decay rate for exponential curriculum schedule (must
                        be negative; more negative = faster difficulty
                        increase)
  --step-easy-rounds STEP_EASY_ROUNDS
                        Number of rounds with easy signals when using step
                        curriculum schedule
  --step-hard-rounds STEP_HARD_ROUNDS
                        Number of rounds with hard signals when using step
                        curriculum schedule
  --base-learning-rate BASE_LEARNING_RATE
                        Initial learning rate for Adam optimizer
  --min-learning-rate MIN_LEARNING_RATE
                        Learning rate floor for adaptive learning rate
                        reduction
  --min-pct-improvement MIN_PCT_IMPROVEMENT
                        Minimum fractional validation loss improvement to
                        avoid LR reduction (e.g., 0.001 = 0.1%)
  --patience-threshold PATIENCE_THRESHOLD
                        Number of consecutive epochs without minimum
                        improvement before reducing learning rate
  --lr-reduction-factor LR_REDUCTION_FACTOR
                        Multiplicative factor for learning rate reduction
                        (e.g., 0.2 reduces LR by 20%)
  --max-retries MAX_RETRIES
                        Maximum number of retry attempts when training fails
                        due to errors
  --retry-delay RETRY_DELAY
                        Delay in seconds between retry attempts after training
                        failure
  --load-tag LOAD_TAG   Model tag to resume training from. Accepted formats:
                        final_vX, round_XX, YYYYMMDD_HHMMSS
  --load-dir LOAD_DIR   Directory to load model tag from. Argument appended to
                        AETHERSCAN_OUTPUT_PATH
  --save-tag SAVE_TAG   Tag for current pipeline run. Accepted formats:
                        final_vX, round_XX, YYYYMMDD_HHMMSS
```
