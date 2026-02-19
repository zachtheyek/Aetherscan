# Note, we avoid logging anything in config.py to prevent coupling with the logger module
"""
Configuration module for Aetherscan Pipeline
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import cpu_count


@dataclass
class DBConfig:
    """SQLite database configuration"""

    get_connection_timeout: float = 60.0  # seconds
    stop_writer_timeout: float = 10.0  # seconds
    write_interval: float = 5.0  # seconds
    write_buffer_max_size: int = 100  # records
    write_retry_delay: float = 1.0  # seconds
    flush_timeout: float = 10.0  # seconds


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
    test_files: list[str] = field(default_factory=lambda: ["real_filtered_LARGE_test_HIP15638.npy"])
    # TODO: have a new member called inference_files, whose values take precedence over test_files if not None (should still live inside self.data_path). test_files -> .npy, inference_files -> .h5 (add checks in validate_args & inference_command)


@dataclass
class TrainingConfig:
    num_training_rounds: int = 20
    epochs_per_round: int = 100

    num_samples_beta_vae: int = 499200
    num_samples_rf: int = 99840
    train_val_split: float = 0.8

    per_replica_batch_size: int = 128
    effective_batch_size: int = 3072  # Effective batch size for gradient accumulation
    per_replica_val_batch_size: int = 320

    # Signal injection params
    # TODO: experiment with larger chunk sizes (how to track chunk processing efficiency)
    signal_injection_chunk_size: int = (
        50000  # Maximum cadences to process at once during data generation
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
    latent_viz_num_cadences_per_type: int = (
        250  # Cadences per signal type for viz batch (total = 4×)
    )
    latent_viz_step_interval: int = 10  # Capture snapshot every N training steps
    latent_viz_gif_duration_ms: int = 100  # Milliseconds per frame in output GIF

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
    per_replica_batch_size: int = 2048  # NOTE: come back to this later
    classification_threshold: float = 0.9


@dataclass
class CheckpointConfig:
    """Checkpoint configuration"""

    load_dir: str | None = None
    load_tag: str | None = None
    start_round: int = 1
    save_tag: str = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        self.beta_vae = BetaVAEConfig()
        self.rf = RandomForestConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
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
                "inference_background_load_chunk_size": self.data.inference_background_load_chunk_size,
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
                "effective_batch_size": self.training.effective_batch_size,
                "per_replica_val_batch_size": self.training.per_replica_val_batch_size,
                "signal_injection_chunk_size": self.training.signal_injection_chunk_size,
                "plot_injection_subsampling_count": self.training.plot_injection_subsampling_count,
                "plot_injection_outlier_percentile": self.training.plot_injection_outlier_percentile,
                "latent_viz_num_cadences_per_type": self.training.latent_viz_num_cadences_per_type,
                "latent_viz_step_interval": self.training.latent_viz_step_interval,
                "latent_viz_gif_duration_ms": self.training.latent_viz_gif_duration_ms,
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
            },
            "checkpoint": {
                "load_dir": self.checkpoint.load_dir,
                "load_tag": self.checkpoint.load_tag,
                "start_round": self.checkpoint.start_round,
                "save_tag": self.checkpoint.save_tag,
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
