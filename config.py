# TODO: make sure config params are used where possible, and remove unaccessed config params
# TODO: make sure pipeline uses the same singleton config param instance (not separate instances)
# TODO: double check to_dict
"""
Configuration module for Aetherscan Pipeline
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DBConfig:
    """SQLite database configuration"""

    get_connection_timeout: float = 60.0  # seconds
    stop_writer_timeout: float = 10.0  # seconds
    write_interval: float = 5.0  # seconds
    write_buffer_max_size: int = 100  # records
    write_retry_delay: float = 1.0  # seconds


@dataclass
class ManagerConfig:
    """Resource manager configuration"""

    pool_terminate_timeout: float = 2.0  # seconds
    # multiprocessing_chunksize: int = 4


@dataclass
class MonitorConfig:
    """Resource monitor configuration"""

    get_gpu_timeout: float = 5.0  # seconds
    stop_monitor_timeout: float = 10.0  # seconds
    monitor_interval: float = 1.0  # seconds
    monitor_retry_delay: float = 1.0  # seconds


@dataclass
class BetaVAEConfig:
    """Beta-VAE model configuration"""

    latent_dim: int = 8  # Bottleneck size
    dense_layer_size: int = 512  # Should match num frequency bins after downsampling
    kernel_size: tuple[int, int] = (3, 3)  # For Conv2D & Conv2DTranspose layers
    beta: float = 1.5  # KL divergence weight
    alpha: float = 10.0  # Clustering loss weight


@dataclass
class RandomForestConfig:
    """Random Forest configuration"""

    n_estimators: int = 1000  # Number of trees
    bootstrap: bool = (
        True  # Whether to use bootstrap sampling when building each tree (True = bagging)
    )
    max_features: str = "sqrt"  # Random feature selection (sqrt, log2, float)
    n_jobs: int = -1  # Number of parallel jobs to run (-1 = use all available CPU cores)
    seed: int = 11


@dataclass
class DataConfig:
    """Data processing configuration"""

    num_observations: int = 6  # Per cadence snippet (3 ON, 3 OFF)
    width_bin: int = 4096  # Frequency bins per observation
    downsample_factor: int = 8  # Frequency bins downsampling factor
    time_bins: int = 16  # Time bins per observation
    freq_resolution: float = 2.7939677238464355  # Hz
    time_resolution: float = 18.25361108  # seconds

    num_target_backgrounds: int = 15000  # Number of background cadences to load
    # Note that max backgrounds per file = max_chunks_per_file * background_load_chunk_size
    background_load_chunk_size: int = (
        200  # Maximum cadences to process at once during background loading
    )
    max_chunks_per_file: int = 25  # Maximum chunks to load from a single file

    # Data files
    train_files: list[str] | None = None
    test_files: list[str] | None = None

    def __post_init__(self):
        """Set default file lists"""
        if self.train_files is None:
            self.train_files = [
                "real_filtered_LARGE_HIP110750.npy",
                "real_filtered_LARGE_HIP13402.npy",
                "real_filtered_LARGE_HIP8497.npy",
            ]
        if self.test_files is None:
            self.test_files = ["real_filtered_LARGE_testHIP83043.npy"]


@dataclass
class TrainingConfig:
    num_training_rounds: int = 20
    epochs_per_round: int = 100

    num_samples_beta_vae: int = 120000
    num_samples_rf: int = 24000
    train_val_split: float = 0.8

    per_replica_batch_size: int = 128
    global_batch_size: int = 2048  # Effective batch size for gradient accumulation
    per_replica_val_batch_size: int = 4096

    signal_injection_chunk_size: int = (
        1000  # Maximum cadences to process at once during data generation
    )

    # Curriculum learning params
    snr_base: int = 10
    initial_snr_range: int = 40
    final_snr_range: int = 10
    curriculum_schedule: str = "exponential"  # "linear", "exponential", "step"
    exponential_decay_rate: int = -3  # How quickly schedule should progress from easy to hard (must be <0) (more negative = less easy rounds & more hard rounds)
    step_easy_rounds: int = 5  # Number of rounds with easy signals
    step_hard_rounds: int = 15  # Number of rounds with challenging signals

    # Adaptive LR params
    base_learning_rate: float = 0.001
    min_learning_rate: float = 1e-6
    min_pct_improvement: float = 0.001  # 0.1% val loss improvement
    patience_threshold: int = 3  # consecutive epochs with no improvement
    reduction_factor: float = 0.2  # 20% LR reduction

    # Fault tolerance params
    max_retries: int = 5
    retry_delay: int = 60  # seconds


# NOTE: come back to this later
@dataclass
class InferenceConfig:
    """Inference configuration"""

    classification_threshold: float = 0.5
    batch_size: int = 4048
    max_drift_rate: float = 10.0  # Hz/s
    # overlap search


class Config:
    """Main configuration class"""

    def __init__(self):
        self.db = DBConfig()
        self.manager = ManagerConfig()
        self.monitor = MonitorConfig()
        self.beta_vae = BetaVAEConfig()
        self.rf = RandomForestConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()

        # Paths
        self.data_path = os.environ.get("SETI_DATA_PATH", "/datax/scratch/zachy/data/aetherscan")
        self.model_path = os.environ.get(
            "SETI_MODEL_PATH", "/datax/scratch/zachy/models/aetherscan"
        )
        self.output_path = os.environ.get(
            "SETI_OUTPUT_PATH", "/datax/scratch/zachy/outputs/aetherscan"
        )

    def get_training_file_path(self, filename: str) -> str:
        """Get full path for training data file"""
        return os.path.join(self.data_path, "training", filename)

    def get_test_file_path(self, filename: str) -> str:
        """Get full path for test data file"""
        return os.path.join(self.data_path, "testing", filename)

    def get_file_subset(self, filename: str) -> tuple[int | None, int | None]:
        """Get subset parameters for a file (start, end indices)"""
        # Option to define subsets for specific files to manage memory usage
        subset_map = {
            "real_filtered_LARGE_HIP110750.npy": (None, 5000),
            "real_filtered_LARGE_HIP13402.npy": (3000, 10000),
            "real_filtered_LARGE_HIP8497.npy": (8000, None),
            "real_filtered_LARGE_testHIP83043.npy": (None, None),
        }
        return subset_map.get(filename, (None, None))

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization"""
        return {
            "db": {
                "get_connection_timeout": self.db.get_connection_timeout,
                "stop_writer_timeout": self.db.stop_writer_timeout,
                "write_interval": self.db.write_interval,
                "write_buffer_max_size": self.db.write_buffer_max_size,
                "write_retry_delay": self.db.write_retry_delay,
            },
            "manager": {
                "pool_terminate_timeout": self.manager.pool_terminate_timeout,
            },
            "monitor": {
                "get_gpu_timeout": self.monitor.get_gpu_timeout,
                "stop_monitor_timeout": self.monitor.stop_monitor_timeout,
                "monitor_interval": self.monitor.monitor_interval,
                "monitor_retry_delay": self.monitor.monitor_retry_delay,
            },
            "beta_vae": {
                "latent_dim": self.beta_vae.latent_dim,
                "dense_layer_size": self.beta_vae.dense_layer_size,
                "kernel_size": self.beta_vae.kernel_size,
                "beta": self.beta_vae.beta,
                "alpha": self.beta_vae.alpha,
            },
            "rf": {
                "n_estimators": self.rf.n_estimators,
                "bootstrap": self.rf.bootstrap,
                "max_features": self.rf.max_features,
                "n_jobs": self.rf.n_jobs,
                "seed": self.rf.seed,
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
                "train_files": self.data.train_files,
                "test_files": self.data.test_files,
            },
            "training": {
                "num_training_rounds": self.training.num_training_rounds,
                "epochs_per_round": self.training.epochs_per_round,
                "num_samples_beta_vae": self.training.num_samples_beta_vae,
                "num_samples_rf": self.training.num_samples_rf,
                "train_val_split": self.training.train_val_split,
                "per_replica_batch_size": self.training.per_replica_batch_size,
                "global_batch_size": self.training.global_batch_size,
                "per_replica_val_batch_size": self.training.per_replica_val_batch_size,
                "signal_injection_chunk_size": self.training.signal_injection_chunk_size,
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
                "classification_threshold": self.inference.classification_threshold,
                "batch_size": self.inference.batch_size,
                "max_drift_rate": self.inference.max_drift_rate,
            },
            "paths": {
                "data_path": self.data_path,
                "model_path": self.model_path,
                "output_path": self.output_path,
            },
        }
