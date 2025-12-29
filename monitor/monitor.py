# BUG:
# system total CPU usage appears "unnormalized" compared to process + children CPU usage
# process + children CPU & RAM sometimes exceeds system total
# also, split GPU usage & memory into separate plots?
# TODO: add a threshold to config where if RAM usage > threshold, immediately exit & initiate cleanup
"""
Resource monitor for Aetherscan Pipeline
Runs as background thread & records system metrics (CPU, RAM, GPU) to database writer queue
Saves resource utilization plot on exit
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import subprocess
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import tensorflow as tf

matplotlib.use("Agg")  # Non-interactive backend for headless environments

from config import get_config

logger = logging.getLogger(__name__)


def get_process_tree_stats(process: psutil.Process) -> dict[str, float]:
    """
    Get total CPU and RAM usage for a process and all child processes.
    This captures multiprocessing workers spawned by Pool() calls.

    Args:
        process: psutil.Process object of the main process to track

    Returns:
        Dictionary containing:
        - cpu_percent: CPU usage as percentage of total system CPU (0-100)
        - ram_percent: RAM usage as percentage of total system RAM (0-100)
        - ram_bytes: Total RAM usage in bytes (PSS - Proportional Set Size)
        - ram_gb: Total RAM usage in gigabytes
    """
    try:
        # Get all processes in tree (main + children)
        processes = [process]
        with contextlib.suppress(psutil.NoSuchProcess):
            processes.extend(process.children(recursive=True))

        # Aggregate CPU and RAM usage across all processes
        total_cpu = 0.0
        total_ram_bytes = 0

        for proc in processes:
            try:
                # CPU: Get percentage (can be >100% for multi-core usage)
                cpu = proc.cpu_percent(interval=0.0)  # Non-blocking
                total_cpu += cpu

                # RAM: Get PSS (Proportional Set Size)
                # Use PSS instead of RSS to avoid double-counting shared memory across processes.
                # RSS counts shared pages once per process, so summing RSS across a process tree
                # can exceed system total RAM. PSS divides shared pages by # of sharing processes,
                # making it additive and accurate when summing across multiple processes.
                mem_info = proc.memory_full_info()
                total_ram_bytes += mem_info.pss

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process may have died between children() and cpu_percent() or memory_info() calls
                continue

        # Convert CPU to percentage of total system CPU
        num_cores = psutil.cpu_count() or 0
        cpu_percent = total_cpu / num_cores if num_cores > 0 else 0.0

        # Convert RAM to percentage of total system RAM
        total_system_ram = psutil.virtual_memory().total
        ram_percent = (total_ram_bytes / total_system_ram) * 100

        return {
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
            "ram_bytes": total_ram_bytes,
            "ram_gb": total_ram_bytes / 1e9,
        }

    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.warning(f"Error getting process tree stats for PID {process.pid}: {e}")
        return {
            "cpu_percent": 0.0,
            "ram_percent": 0.0,
            "ram_bytes": 0,
            "ram_gb": 0.0,
        }


class ResourceMonitor:
    """Background thread to monitor system resources"""

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
        """Initialize monitor"""
        # Note, __init__ is triggered every time the class's constructor is called,
        # even if __new__ returned the existing singleton instance
        # Hence, we use the _initialized flag to make sure __init__ only runs once
        if self._initialized:
            return

        self._initialized = True

        self.config = get_config()
        if self.config is None:
            raise ValueError("get_config() returned None")

        self.tag = self.config.checkpoint.save_tag
        self.get_gpu_timeout = self.config.monitor.get_gpu_timeout
        self.stop_monitor_timeout = self.config.monitor.stop_monitor_timeout
        self.monitor_interval = self.config.monitor.monitor_interval
        self.monitor_retry_delay = self.config.monitor.monitor_retry_delay

        self.monitor_thread = None
        self.stop_event = threading.Event()  # Thread-safe flag for stopping

        # Get main process ID
        self.process = psutil.Process(os.getpid())

        # Detect GPUs
        self._detect_gpus()

        # Get database instance
        # Late import to avoid circular dependency (db imports from manager)
        from db import get_db  # noqa: PLC0415

        self.db = get_db()
        if self.db is None:
            raise RuntimeError(
                "Database not initialized - resource monitoring data won't be persisted"
            )

        logger.info("Resource monitor initialized")
        logger.info(f"Main process PID: {self.process.pid}")
        logger.info(f"CPU cores: {psutil.cpu_count() or 0}")
        logger.info(f"Total memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        logger.info(f"GPUs detected: {self.num_gpus}")
        if self.num_gpus > 0:
            for name in self.gpu_names:
                logger.info(f"  {name}")
        logger.info(f"Monitor interval: {self.monitor_interval} seconds")

    @classmethod
    def _reset(cls):
        """
        Teardown hook for thread-safe singleton
        Resets the monitor instance to None

        WARNING: Only use for testing or cleanup after shutdown.
        Calling this while the monitor is active will cause issues.
        Should only be called after stop() has completed.
        """
        # Acquire lock to prevent race conditions
        with cls._lock:
            # Discard the singleton instance by removing the global reference
            # Guarantees the next constructor call will produce a fresh instance
            # Note, resources held by the old instance will remain alive unless explicitly closed beforehand
            cls._instance = None
            logger.info("Monitor singleton instance reset")

    def _detect_gpus(self):
        """Detect available GPUs"""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            self.num_gpus = len(gpus)

            # Try to get GPU names using nvidia-smi if available
            if self.num_gpus > 0:
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=self.get_gpu_timeout,
                    )
                    if result.returncode == 0:
                        self.gpu_names = [
                            f"{name.strip()}:{i}"
                            for i, name in enumerate(result.stdout.strip().split("\n"))
                        ]
                    else:
                        self.gpu_names = [f"GPU:{i}" for i in range(self.num_gpus)]
                except Exception:
                    self.gpu_names = [f"GPU:{i}" for i in range(self.num_gpus)]
            else:
                self.gpu_names = []

        except Exception:
            self.num_gpus = 0
            self.gpu_names = []

    def _get_process_tree_stats(self):
        """
        Get total CPU and RAM usage for main process and all child processes.

        Returns:
            tuple: (cpu_percent_total, ram_percent)
        """
        stats = get_process_tree_stats(self.process)
        return stats["cpu_percent"], stats["ram_percent"]

    def _get_gpu_stats(self):
        """Get GPU usage and memory statistics"""
        gpu_utils = []
        gpu_mems = []

        if self.num_gpus > 0:
            try:
                # Get GPU utilization
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.get_gpu_timeout,
                )

                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        parts = line.split(",")
                        util = float(parts[0].strip())
                        mem_used = float(parts[1].strip())
                        mem_total = float(parts[2].strip())
                        mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0

                        gpu_utils.append(util)
                        gpu_mems.append(mem_percent)
                else:
                    gpu_utils = [0.0] * self.num_gpus
                    gpu_mems = [0.0] * self.num_gpus
            except Exception:
                gpu_utils = [0.0] * self.num_gpus
                gpu_mems = [0.0] * self.num_gpus

        return gpu_utils, gpu_mems

    def start(self):
        """Start monitoring in background thread"""
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            return

        self.stop_event.clear()
        # NOTE: should monitor be daemon or non-daemon thread?
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=False)
        self.monitor_thread.start()
        logger.info("Resource monitoring thread started")

    def stop(self):
        """Stop monitoring"""
        if self.monitor_thread is None:
            return

        logger.info("Stopping resource monitoring thread...")
        self.stop_event.set()  # Signal thread to stop

        # Wait for monitoring thread to finish
        self.monitor_thread.join(timeout=self.stop_monitor_timeout)

        if self.monitor_thread.is_alive():
            logger.warning("Resource monitoring thread did not stop cleanly")
        else:
            logger.info("Resource monitoring thread stopped")

    def _monitor_loop(self):
        """Background monitoring loop with database writes"""
        self.start_time = time.time()

        # Keep looping until told to stop
        while not self.stop_event.is_set():
            try:
                current_time = time.time()

                if self.db is None:
                    raise RuntimeError("No database instance detected - cannot run monitoring loop")

                # Get system resources & queue db writes (non-blocking)
                self.db.write_system_resource(
                    "cpu",
                    "system_total",
                    psutil.cpu_percent(interval=0.1),
                    unit="percent",
                    tag=self.tag,
                    timestamp=current_time,
                )
                self.db.write_system_resource(
                    "ram",
                    "system_total",
                    psutil.virtual_memory().percent,
                    unit="percent",
                    tag=self.tag,
                    timestamp=current_time,
                )

                cpu_process, ram_process = self._get_process_tree_stats()
                self.db.write_system_resource(
                    "cpu",
                    "process_tree",
                    cpu_process,
                    unit="percent",
                    tag=self.tag,
                    timestamp=current_time,
                )
                self.db.write_system_resource(
                    "ram",
                    "process_tree",
                    ram_process,
                    unit="percent",
                    tag=self.tag,
                    timestamp=current_time,
                )

                gpu_utils, gpu_mems = self._get_gpu_stats()
                for gpu_idx, (gpu_util, gpu_mem) in enumerate(
                    zip(gpu_utils, gpu_mems, strict=False)
                ):
                    gpu_name = (
                        self.gpu_names[gpu_idx]
                        if gpu_idx < len(self.gpu_names)
                        else f"GPU:{gpu_idx}"
                    )
                    self.db.write_system_resource(
                        "gpu",
                        f"{gpu_name}_utilization",
                        gpu_util,
                        unit="percent",
                        tag=self.tag,
                        timestamp=current_time,
                    )
                    self.db.write_system_resource(
                        "gpu",
                        f"{gpu_name}_memory",
                        gpu_mem,
                        unit="percent",
                        tag=self.tag,
                        timestamp=current_time,
                    )

                # Sleep until next interval (interruptible for faster shutdown)
                self.stop_event.wait(self.monitor_interval)

            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                # Sleep until next interval (interruptible for faster shutdown)
                self.stop_event.wait(self.monitor_retry_delay)

        # Save plot on shutdown
        self._save_plot()

    def _save_plot(self):
        """Generate and save resource utilization plot from database"""
        current_time = time.time()

        # Query resource metrics from database
        if self.db is None:
            raise RuntimeError("No database instance detected - cannot generate resource plot")

        all_resources = self.db.query_system_resource(
            start_time=self.start_time,
            end_time=current_time,
        )

        if not all_resources:
            logger.warning("No resource monitoring data to plot")
            return

        # TODO: potential memory optimization here with array pre-allocation?
        # TODO: or instead of extracting dict -> ndarray|dict, just use dict directly?
        # Organize resources by type and name
        timestamps_dict = {}
        values_dict = {}

        for resource in all_resources:
            key = f"{resource['resource_type']}_{resource['resource_name']}"
            if key not in timestamps_dict:
                timestamps_dict[key] = []
                values_dict[key] = []

            # Timestamps measured relative to start time, in minutes
            timestamps_dict[key].append((resource["timestamp"] - self.start_time) / 60)
            values_dict[key].append(resource["value"])

        del all_resources
        gc.collect()

        # Extract CPU data
        cpu_system_timestamps = np.array(timestamps_dict.get("cpu_system_total", []))
        cpu_system_data = np.array(values_dict.get("cpu_system_total", []))
        cpu_process_timestamps = np.array(timestamps_dict.get("cpu_process_tree", []))
        cpu_process_data = np.array(values_dict.get("cpu_process_tree", []))

        # Extract RAM data
        ram_system_timestamps = np.array(timestamps_dict.get("ram_system_total", []))
        ram_system_data = np.array(values_dict.get("ram_system_total", []))
        ram_process_timestamps = np.array(timestamps_dict.get("ram_process_tree", []))
        ram_process_data = np.array(values_dict.get("ram_process_tree", []))

        # Extract GPU data (organized by GPU)
        gpu_data = {}
        for key in timestamps_dict:
            if key.startswith("gpu_"):
                parts = key.split("_")
                # Format: gpu_<name>_<utilization|memory>
                if len(parts) >= 2:
                    metric_type = parts[-1]  # utilization or memory
                    gpu_name = "_".join(parts[1:-1])  # everything between gpu and metric_type

                    if gpu_name not in gpu_data:
                        gpu_data[gpu_name] = {}

                    timestamps = np.array(timestamps_dict.get(key, []))
                    values = np.array(values_dict.get(key, []))
                    gpu_data[gpu_name][metric_type] = (timestamps, values)

        del timestamps_dict, values_dict
        gc.collect()

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        fig.suptitle("Aetherscan Pipeline: Resource Utilization", fontsize=16, fontweight="bold")

        # CPU plot
        ax_cpu = axes[0]
        if len(cpu_process_data) > 0:
            ax_cpu.plot(
                cpu_process_timestamps,
                cpu_process_data,
                color="#1f77b4",
                linewidth=1.5,
                label="Process + Children",
                alpha=0.8,
            )
            ax_cpu.fill_between(
                cpu_process_timestamps, cpu_process_data, alpha=0.3, color="#1f77b4"
            )

        if len(cpu_system_data) > 0:
            ax_cpu.plot(
                cpu_system_timestamps,
                cpu_system_data,
                color="#ff7f0e",
                linewidth=2.0,
                label="System Total",
                alpha=0.9,
            )

        ax_cpu.set_ylabel("CPU Usage (%)", fontsize=12, fontweight="bold")
        ax_cpu.set_ylim(0, 100)
        ax_cpu.grid(True, alpha=0.3)
        ax_cpu.legend(loc="upper right", fontsize=10)
        ax_cpu.set_title(f"CPU Pressure (n={psutil.cpu_count()} cores)", fontsize=12)

        # RAM plot
        ax_ram = axes[1]
        if len(ram_process_data) > 0:
            ax_ram.plot(
                ram_process_timestamps,
                ram_process_data,
                color="#2ca02c",
                linewidth=1.5,
                label="Process + Children",
                alpha=0.8,
            )
            ax_ram.fill_between(
                ram_process_timestamps, ram_process_data, alpha=0.3, color="#2ca02c"
            )

        if len(ram_system_data) > 0:
            ax_ram.plot(
                ram_system_timestamps,
                ram_system_data,
                color="#d62728",
                linewidth=2.0,
                label="System Total",
                alpha=0.9,
            )

        ax_ram.set_ylabel("RAM Usage (%)", fontsize=12, fontweight="bold")
        ax_ram.set_ylim(0, 100)
        ax_ram.grid(True, alpha=0.3)
        ax_ram.legend(loc="upper right", fontsize=10)
        ax_ram.set_title(
            f"Memory Pressure (total={psutil.virtual_memory().total / (1024**3):.2f} GB)",
            fontsize=12,
        )

        # GPU plot
        ax_gpu = axes[2]
        if gpu_data and self.num_gpus > 0:
            # Create second y-axis
            ax_gpu_mem = ax_gpu.twinx()

            colors = plt.cm.tab10(np.linspace(0, 1, len(gpu_data)))

            for gpu_idx, (gpu_name, metrics) in enumerate(gpu_data.items()):
                color = colors[gpu_idx]

                # Usage (solid line, y1)
                if "utilization" in metrics:
                    timestamps, values = metrics["utilization"]
                    ax_gpu.plot(
                        timestamps,
                        values,
                        label=f"{gpu_name} (Usage)",
                        color=color,
                        linewidth=1.5,
                        alpha=0.9,
                    )

                # Memory (dashed line, y2, dimmer)
                if "memory" in metrics:
                    timestamps, values = metrics["memory"]
                    ax_gpu_mem.plot(
                        timestamps,
                        values,
                        label=f"{gpu_name} (Memory)",
                        color=color,
                        linewidth=1.5,
                        alpha=0.6,
                        linestyle="--",
                    )

            ax_gpu.set_ylabel("GPU Usage (%)", fontsize=12, fontweight="bold")
            ax_gpu_mem.set_ylabel("GPU Memory (%)", fontsize=12, fontweight="bold")
            ax_gpu.set_ylim(0, 100)
            ax_gpu_mem.set_ylim(0, 100)

            # Combine legends
            lines1, labels1 = ax_gpu.get_legend_handles_labels()
            lines2, labels2 = ax_gpu_mem.get_legend_handles_labels()
            ax_gpu.legend(
                lines1 + lines2,
                labels1 + labels2,
                ncol=min(4, len(gpu_data) * 2),
                fontsize=8,
                loc="upper right",
            )
        else:
            ax_gpu.text(
                0.5,
                0.5,
                "No GPUs detected",
                ha="center",
                va="center",
                transform=ax_gpu.transAxes,
                fontsize=14,
            )
            ax_gpu.set_ylabel("GPU Usage (%)", fontsize=12, fontweight="bold")

        ax_gpu.grid(True, alpha=0.3)
        ax_gpu.set_title(f"GPU Pressure (n={self.num_gpus} devices)", fontsize=12)
        ax_gpu.set_xlabel("Time (minutes)", fontsize=12, fontweight="bold")

        # Adjust layout and save
        plt.tight_layout()

        # Save plot
        output_path = os.path.join(
            self.config.output_path, "plots", f"resource_utilization_{self.tag}.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create dir if it doesn't exist

        plt.savefig(output_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        logger.info(f"Resource utilization plot saved to: {output_path}")


def init_monitor() -> ResourceMonitor:
    """
    Initialize global monitor instance (call once at startup)
    """
    monitor = ResourceMonitor()
    monitor.start()

    # Late import to avoid circular dependency (manager imports from monitor)
    from manager import register_monitor  # noqa: PLC0415

    register_monitor(monitor)

    return monitor


def get_monitor() -> ResourceMonitor | None:
    """Get the global monitor instance"""
    monitor = ResourceMonitor._instance

    if monitor is None:
        logger.warning("No monitor instance initialized")

    return monitor


def shutdown_monitor():
    """Shutdown the global monitor instance (call on exit)"""
    monitor = ResourceMonitor._instance

    if monitor is None:
        logger.warning("No monitor instance initialized")
        return

    monitor.stop()
    ResourceMonitor._reset()
