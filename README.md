<h1>
<p align="center">
  <span style="font-size: 5em;">Aetherscan</span>
</h1>
  <p align="center">
    Breakthrough Listen's first end-to-end production-grade deep learning pipeline for SETI @ scale
    <br />
    <br />
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg" alt="License"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python"></a>
    <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.16-orange.svg" alt="TensorFlow"></a>
    <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.2-green.svg" alt="CUDA"></a>
  </p>
</p>

---

## Overview

Aetherscan is a machine learning pipeline designed to detect technosignatures (potential signs of extraterrestrial intelligence) in radio telescope data at scale. It implements a **two-stage approach** combining a Beta-VAE for feature extraction with a Random Forest classifier for candidate detection.

The architecture is based on [Ma et al. 2023](https://arxiv.org/abs/2301.12670) ("A deep-learning search for technosignatures from 820 unique stars"), extending the research prototype into a production-ready system capable of processing the Breakthrough Listen archive.

### Key Innovations

- **Curriculum learning** with progressive SNR difficulty schedules
- **Multi-GPU training** via TensorFlow MirroredStrategy with NCCL
- **Gradient accumulation** for large effective batch sizes
- **Custom clustering loss** that leverages the ON/OFF cadence pattern
- **Fault tolerance** with automatic retry and checkpoint recovery
- **Real-time monitoring** with Slack integration

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │  Background  │    │   Signal     │    │  Curriculum  │    │   Multi-GPU  │  │
│   │    Data      │───▶│  Injection   │───▶│   Learning   │───▶│   Training   │  │
│   │  (.npy)      │    │  (setigen)   │    │  (SNR sched) │    │ (MirroredStr)│  │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                                      │          │
│                                                                      ▼          │
│                                                              ┌──────────────┐   │
│                                                              │   Beta-VAE   │   │
│                                                              │  (8-dim z)   │   │
│                                                              └──────────────┘   │
│                                                                      │          │
│                                                                      ▼          │
│                                                              ┌──────────────┐   │
│                                                              │ Random Forest│   │
│                                                              │ (1000 trees) │   │
│                                                              └──────────────┘   │
│                                                                      │          │
│                                                                      ▼          │
│                                                              ┌──────────────┐   │
│                                                              │  Checkpoints │   │
│                                                              │   (.keras)   │   │
│                                                              └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                             INFERENCE PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │  Observation │    │   Preproc    │    │   Encoder    │    │  RF Predict  │  │
│   │    Data      │───▶│  (downsamp)  │───▶│  (latents)   │───▶│ (candidates) │  │
│   │   (.h5)      │    │              │    │              │    │              │  │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                                      │          │
│                                                                      ▼          │
│                                                              ┌──────────────┐   │
│                                                              │   SQLite DB  │   │
│                                                              │ (candidates) │   │
│                                                              └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature                    | Description                                                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Curriculum Learning**    | Progressive difficulty schedules (linear, exponential, step) that start with high-SNR signals and gradually introduce harder examples |
| **Multi-GPU Training**     | Distributed training with MirroredStrategy and NCCL AllReduce for efficient gradient synchronization                                  |
| **Gradient Accumulation**  | Achieves large effective batch sizes (3072) without requiring proportional GPU memory                                                 |
| **Custom Clustering Loss** | Exploits ON/OFF cadence pattern to separate technosignatures from RFI in latent space                                                 |
| **Adaptive Learning Rate** | Patience-based LR reduction when validation loss plateaus                                                                             |
| **Fault Tolerance**        | Automatic retry with exponential backoff; per-round checkpointing for recovery                                                        |
| **Real-time Monitoring**   | Resource tracking (CPU, RAM, GPU) with Slack alerts for errors and milestones                                                         |
| **Async Database**         | Thread-safe SQLite with queue-based writes for concurrent metric logging                                                              |

---

## System Requirements

### Training

| Resource | Minimum                   | Recommended                |
| -------- | ------------------------- | -------------------------- |
| GPU      | 1x NVIDIA GPU (16GB VRAM) | 4x NVIDIA A100 (80GB each) |
| RAM      | 32 GB                     | 64+ GB                     |
| Storage  | 100 GB                    | 500+ GB SSD                |
| CUDA     | 12.2                      | 12.2+                      |

### Inference

| Resource | Minimum                  | Recommended               |
| -------- | ------------------------ | ------------------------- |
| GPU      | 1x NVIDIA GPU (8GB VRAM) | 1x NVIDIA GPU (16GB VRAM) |
| RAM      | 16 GB                    | 32 GB                     |
| Storage  | 50 GB                    | 100+ GB SSD               |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zachtheyek/Aetherscan.git
cd Aetherscan
```

### 2. Create conda environment

```bash
conda env create -f environment.yml
conda activate aetherscan
```

### 3. Install the package

```bash
pip install -e .
```

### 4. Set environment variables

```bash
export AETHERSCAN_DATA_PATH="/path/to/data"
export AETHERSCAN_MODEL_PATH="/path/to/models"
export AETHERSCAN_OUTPUT_PATH="/path/to/outputs"

# Optional: Slack integration
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_CHANNEL="#aetherscan-alerts"
```

---

## Usage

### Training

```bash
# Basic training run
aetherscan train

# Training with custom parameters
aetherscan train \
    --num-training-rounds 20 \
    --epochs-per-round 100 \
    --curriculum-schedule exponential \
    --save-tag my_experiment

# Resume from checkpoint
aetherscan train \
    --load-tag round_10 \
    --save-tag my_experiment_resumed
```

### Inference

```bash
# Run inference on test data
aetherscan inference \
    --test-files data_file.npy \
    --threshold 0.9

# With custom batch size
aetherscan inference \
    --test-files data_file.npy \
    --batch-size 1728 \
    --save-tag inference_run_1
```

### Evaluation

```bash
# Evaluate model performance
aetherscan evaluate
```

---

## Configuration

Aetherscan uses a hierarchical configuration system with dataclass-based configs. Values can be set via:

1. **Defaults** - Defined in `src/aetherscan/config.py`
2. **Environment variables** - For paths and secrets
3. **CLI arguments** - Override at runtime

### Configuration Groups

| Config Class         | Description          | Key Parameters                                                                |
| -------------------- | -------------------- | ----------------------------------------------------------------------------- |
| `BetaVAEConfig`      | VAE architecture     | `latent_dim=8`, `beta=1.5`, `alpha=10.0`                                      |
| `RandomForestConfig` | RF classifier        | `n_estimators=1000`, `max_features='sqrt'`                                    |
| `DataConfig`         | Data processing      | `num_observations=6`, `width_bin=4096`, `downsample_factor=8`                 |
| `TrainingConfig`     | Training hyperparams | `num_training_rounds=20`, `epochs_per_round=100`, `effective_batch_size=3072` |
| `InferenceConfig`    | Inference settings   | `classification_threshold=0.9`, `per_replica_batch_size=1728`                 |
| `CheckpointConfig`   | Model persistence    | `load_tag`, `save_tag`, `start_round`                                         |
| `LoggerConfig`       | Logging & Slack      | `slack_enabled=True`, `slack_broadcast_level='ERROR'`                         |
| `MonitorConfig`      | Resource monitoring  | `monitor_interval=1.0`                                                        |
| `DBConfig`           | Database settings    | `write_interval=5.0`, `write_buffer_max_size=100`                             |
| `ManagerConfig`      | Resource management  | `n_processes=cpu_count()`, `pool_terminate_timeout=10.0`                      |

---

## Project Structure

```
Aetherscan/
├── src/aetherscan/
│   ├── __init__.py           # Package initialization, version
│   ├── main.py               # Entry point, command dispatch
│   ├── cli.py                # Argument parsing, validation
│   ├── config.py             # Configuration dataclasses
│   ├── train.py              # Training orchestration
│   ├── inference.py          # Inference pipeline
│   ├── evaluate.py           # Evaluation metrics
│   ├── preprocessing.py      # Data preprocessing
│   ├── data_generation.py    # Synthetic signal injection
│   ├── models/
│   │   ├── __init__.py       # Model exports
│   │   ├── vae.py            # Beta-VAE architecture
│   │   └── random_forest.py  # RF classifier wrapper
│   ├── db/
│   │   ├── __init__.py       # Database exports
│   │   └── db.py             # SQLite async writer
│   ├── logger/
│   │   ├── __init__.py       # Logger exports
│   │   ├── logger.py         # Logging configuration
│   │   └── slack_handler.py  # Slack integration
│   ├── monitor/
│   │   ├── __init__.py       # Monitor exports
│   │   └── monitor.py        # Resource monitoring
│   └── manager/
│       ├── __init__.py       # Manager exports
│       └── manager.py        # Resource lifecycle management
├── tests/                    # Test suite
├── docs/                     # Documentation
├── environment.yml           # Conda dependencies
├── pyproject.toml            # Package metadata
├── AGENTS.md                 # Claude Code agent guidelines
├── CONTRIBUTING.md           # Contribution guidelines
├── KNOWN_ISSUES.md           # Known issues and workarounds
├── SECURITY.md               # Security policy
├── LICENSE                   # BSD-3-Clause
└── CITATION.cff              # Citation metadata
```

### Module Responsibilities

| Module                    | Purpose                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------ |
| `train.py`                | Orchestrates training: curriculum learning, distributed datasets, gradient accumulation, checkpointing |
| `inference.py`            | Runs trained models on new data, writes candidates to database                                         |
| `models/vae.py`           | Beta-VAE with custom clustering loss (L_same, L_diff)                                                  |
| `models/random_forest.py` | Scikit-learn RF wrapper with save/load                                                                 |
| `data_generation.py`      | Uses setigen for synthetic signal injection with ON/OFF patterns                                       |
| `preprocessing.py`        | Downsampling, normalization, data alignment                                                            |
| `db/db.py`                | Thread-safe SQLite with async queue-based writes                                                       |
| `monitor/monitor.py`      | Background thread for CPU/RAM/GPU metrics                                                              |
| `manager/manager.py`      | Centralized resource lifecycle (pools, shared memory, cleanup)                                         |
| `logger/`                 | Multi-handler logging with Slack integration                                                           |

---

## How It Works

### The ON/OFF Cadence Pattern

Radio telescopes observe potential technosignature targets using an "ON/OFF" cadence:

- **ON**: Telescope points at target star
- **OFF**: Telescope points at nearby reference position

A real technosignature should appear only in ON observations (signal follows target), while RFI appears in both ON and OFF (signal is local interference).

```
Cadence: [ON₁] [OFF₁] [ON₂] [OFF₂] [ON₃] [OFF₃]
Signal:   ██     --     ██     --     ██     --   ← Technosignature (target-dependent)
RFI:      ██     ██     ██     ██     ██     ██   ← Interference (always present)
```

### Custom Clustering Loss

The Beta-VAE uses a custom loss function that exploits this pattern:

```
L_total = L_reconstruction + β·L_KL + α·(L_true + L_false)
```

Where:

- **L_reconstruction**: Standard VAE reconstruction loss
- **L_KL**: KL divergence for latent regularization (β=1.5)
- **L_true**: For real signals, minimize ON-ON distance, maximize ON-OFF distance
- **L_false**: For RFI/noise, minimize all pairwise distances (all observations similar)

```python
# Pseudo-code for clustering loss
def loss_same(a, b):  # Minimize distance
    return mean(sum((a - b)²))

def loss_diff(a, b):  # Maximize distance
    return mean(1 / (sum((a - b)²) + ε))

# True signals: ON observations cluster together, separate from OFF
L_true = loss_same(ON₁, ON₂) + loss_same(ON₂, ON₃) + ...  # ON-ON close
       + loss_diff(ON₁, OFF₁) + loss_diff(ON₁, OFF₂) + ... # ON-OFF far

# False signals: All observations cluster together
L_false = loss_same(all pairs)  # Everything close
```

---

## CLI Reference

### No Command Provided

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

### Train Command Help

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

---

## Citation

If you use Aetherscan in your research, please cite:

```bibtex
@software{aetherscan,
  author = {Yek, Zach and Ma, Peter Xiangyuan and Croft, Steve and Lebofsky, Matt},
  title = {Aetherscan: Breakthrough Listen's first end-to-end production-grade DL pipeline for SETI @ scale},
  url = {https://github.com/zachtheyek/Aetherscan},
  version = {0.1.0},
  year = {2026}
}
```

See also the foundational paper:

```bibtex
@article{ma2023deep,
  title={A deep-learning search for technosignatures from 820 unique stars},
  author={Ma, Peter Xiangyuan and Ng, Cherry and Croft, Steve and others},
  journal={Nature Astronomy},
  year={2023},
  publisher={Nature Publishing Group}
}
```

---

## Links

- [Contributing Guidelines](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Known Issues](KNOWN_ISSUES.md)
- [License](LICENSE)
- [Breakthrough Listen](https://breakthroughlisten.org/)
- [Ma et al. 2023 Paper](https://arxiv.org/abs/2301.12670)

---

## Acknowledgments

This project is part of [Breakthrough Listen](https://breakthroughlisten.org/), the largest scientific research program aimed at finding evidence of civilizations beyond Earth. The infrastructure and research is funded by the [Breakthrough Prize Foundation](https://breakthroughprize.org/).
