# BUG: sometimes entire sections (e.g. data generation for round X) just don't show up in the resource utilization plot. not sure if data isn't being written to the db properly (haven't verified)?
# TODO: add a threshold to config where if RAM usage > threshold, immediately exit & initiate cleanup. set threshold in monitor config
"""
Resource monitor for Aetherscan Pipeline
Runs as background thread & records system metrics (CPU, RAM, GPU) to database writer queue
Saves resource utilization plot on exit
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import math
import os
import subprocess
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import tensorflow as tf
from matplotlib.lines import Line2D

matplotlib.use("Agg")  # Non-interactive backend for headless environments

from aetherscan.config import get_config
from aetherscan.logger import get_logger

logger = logging.getLogger(__name__)

# Only pipeline_stages spans this shallow (dot-separated components) are overlaid on the
# resource plot — deep spans (per-ON-file energy detection, encode/rf sub-stages, ...) stay
# report-tool-only so the panels don't drown in divider lines
_ANNOTATION_MAX_DEPTH = 2

# Color shared by the stage boundary lines and their text labels (matplotlib named color)
_ANNOTATION_COLOR = "dimgray"

# GPU display names shown in the resource-plot legend ONLY (never the DB resource_name or
# self.gpu_names): a raw name containing one of these substrings collapses to the short alias,
# preserving the ":<idx>" suffix. _sanitize_gpu_display_name returns on the FIRST substring
# match in dict (insertion) order, so if a future key is a substring of another, list the more
# specific (longer) one first. The current two keys don't overlap, so order isn't load-bearing
# yet. Anything outside the whitelist falls back to length-based truncation.
_GPU_DISPLAY_NAME_WHITELIST = {
    "NVIDIA RTX PRO 6000 Blackwell": "PRO 6000",
    "NVIDIA RTX A4000": "A4000",
}

# GPU display names past this length fall back to <cutoff-1> chars + "..." (whitelist misses)
_GPU_NAME_MAX_LEN = 20

# Cap the GPU legend at this many rows per metric group before spilling into another column,
# biasing the busy GPU legend toward fewer columns / more rows so it clears the centered title
_GPU_LEGEND_MAX_ROWS = 6

# Headroom-band layout for the per-panel legends moved outside/above each panel (issue #214):
# each legend's lower edge is pinned at _LEGEND_Y (axes fraction, just above the top spine) so
# it grows upward into the band, and the subplot title is lifted enough to sit above it. The
# title pad must exceed its panel's legend height so the legend's upper edge stays no higher
# than the title: a small pad clears the single-row CPU/RAM legends, a taller one clears the
# multi-row GPU legend (up to _GPU_LEGEND_MAX_ROWS rows).
_TITLE_PAD = 26
_GPU_TITLE_PAD = 80
_LEGEND_Y = 1.005


def select_annotation_spans(rows: list[dict], max_depth: int = _ANNOTATION_MAX_DEPTH) -> list[dict]:
    """
    Filter pipeline_stages rows down to the ones the resource plot overlays: spans whose
    dot-name has at most max_depth components (e.g. "train.round_03" but not
    "train.round_03.epochs"), sorted by start_time. Pure helper, unit-testable without a
    monitor instance.
    """
    spans = [row for row in rows if len(str(row["stage"]).split(".")) <= max_depth]
    spans.sort(key=lambda row: row["start_time"])
    return spans


def _sanitize_gpu_display_name(name: str) -> str:
    """
    Map a raw GPU name to the short label shown in the resource-plot legend ONLY (never the DB
    resource_name or self.gpu_names). Whitelisted product strings (_GPU_DISPLAY_NAME_WHITELIST)
    collapse to a short alias; anything else falls back to the previous length-based truncation
    (name parts longer than _GPU_NAME_MAX_LEN chars become _GPU_NAME_MAX_LEN-1 chars + "...").
    The ":<idx>" suffix appended by _detect_gpus() is preserved in both cases. Pure helper,
    unit-testable.

    e.g. "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition:2" -> "PRO 6000:2",
    "NVIDIA RTX A4000:0" -> "A4000:0", "GPU:1" -> "GPU:1" (short, unchanged).
    """
    name_part, sep, idx_part = name.rpartition(":")
    if not sep:
        name_part = name

    for needle, alias in _GPU_DISPLAY_NAME_WHITELIST.items():
        if needle in name_part:
            return f"{alias}:{idx_part}" if sep else alias

    # Fallback: hard cutoff, truncating to _GPU_NAME_MAX_LEN-1 chars + "..." keeping the
    # ":<idx>" suffix when present (fires with or without a colon).
    if len(name_part) > _GPU_NAME_MAX_LEN:
        truncated = f"{name_part[: _GPU_NAME_MAX_LEN - 1]}..."
        return f"{truncated}:{idx_part}" if sep else truncated
    return name


def _grouped_legend_entries(
    handles1: list,
    labels1: list,
    handles2: list,
    labels2: list,
    *,
    max_rows: int,
) -> tuple[list, list, int]:
    """
    Arrange two metric groups (GPU usage + GPU memory) into a column-major legend grid that
    keeps each group in its own column block and caps the row count at `max_rows`, adding a
    column block to a group only once it exceeds `max_rows` entries. Matplotlib fills legends
    column-major, so each group is padded with invisible handles up to a whole number of
    columns — usage entries then occupy their own column(s) and memory entries always start a
    fresh column (e.g. 5 GPUs at max_rows>=5 -> columns of 5 | 5). Returns (handles, labels,
    ncol). Pure helper.
    """
    n = max(len(handles1), len(handles2))
    cols_per_group = math.ceil(n / max_rows) if n else 1
    nrows = math.ceil(n / cols_per_group) if n else 0
    slots = nrows * cols_per_group

    handles = list(handles1) + [Line2D([], [], alpha=0)] * (slots - len(handles1))
    labels = list(labels1) + [""] * (slots - len(labels1))
    handles += list(handles2) + [Line2D([], [], alpha=0)] * (slots - len(handles2))
    labels += list(labels2) + [""] * (slots - len(labels2))

    return handles, labels, 2 * cols_per_group


def _legend_above_panel(
    ax, handles: list, labels: list, *, ncol: int, fontsize: int, y_anchor: float = _LEGEND_Y
) -> None:
    """
    Place `ax`'s legend outside the panel, in the headroom band between the panel and its title:
    the legend's lower-centre is pinned to (0.5, `y_anchor` in axes fraction) so it is centred
    horizontally under the (also-centred) title and grows upward into the reserved band. The
    title pad must exceed the legend height so the two stay vertically stacked without overlap.
    No-op when there are no labelled artists.
    """
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=ncol,
        fontsize=fontsize,
        frameon=True,
        borderaxespad=0.0,
    )


def _draw_stage_boundaries(
    axes: list, spans: list[dict], start_time: float, current_time: float
) -> None:
    """
    Draw the pipeline-stage overlay: one solid `dimgray` vertical line at each span's right
    edge (its end time, in minutes since `start_time`) on EVERY axis in `axes` (the panels
    share an x-axis), and label each stage once — anchored just left of its line and rotated
    30 deg from horizontal, sitting just BELOW the x-axis of `axes[0]` (in the inter-panel band,
    not inside the plot). Pure drawing helper (no DB or instance state) so it's exercisable with
    synthetic spans in tests and the render harness. `axes` must be non-empty; `axes[0]` is the
    top (CPU) panel whose x-axis the labels hang under.
    """
    label_ax = axes[0]
    for span in spans:
        end_min = (min(span["end_time"], current_time) - start_time) / 60
        for ax in axes:
            ax.axvline(end_min, color=_ANNOTATION_COLOR, linewidth=1.0, alpha=0.7, zorder=2)
        # Leaf name only ("round_03", not "train.round_03"), anchored just left of the line and
        # angled so long names stay legible without overrunning into the neighbouring stage
        label = str(span["stage"]).split(".")[-1]
        label_ax.annotate(
            label,
            # x in data coords (the line); y in axes fraction just BELOW the bottom spine
            # (negative) so the label sits under the top panel's x-axis, in the inter-panel
            # band, rather than inside the plot. y-fraction (not data) survives any y-limit change.
            xy=(end_min, -0.02),
            xycoords=label_ax.get_xaxis_transform(),
            xytext=(-3, 0),
            textcoords="offset points",
            rotation=30,
            rotation_mode="anchor",
            ha="right",
            va="top",
            fontsize=7,
            color=_ANNOTATION_COLOR,
            clip_on=False,  # label lives outside the axes (below the x-axis) — must not be clipped
            zorder=3,
        )


def get_process_tree_stats(
    process: psutil.Process, proc_cache: dict[int, psutil.Process] | None = None
) -> dict[str, float]:
    """
    Sum CPU and RAM usage across `process` and its descendants (the multiprocessing pool
    workers it spawned), returning {cpu_percent, ram_percent, ram_bytes, ram_gb}.

    `proc_cache` maps PID -> psutil.Process and should persist across calls when CPU accuracy
    matters: cpu_percent(interval=0) measures against the baseline recorded by the previous
    call on the *same* Process object, so a freshly created object always reads 0.0 (issue
    #12's undercount). With a persistent cache, each PID is accurate from its second sample
    onward (a new PID contributes one 0.0 reading when first seen), and PIDs that leave the
    tree are evicted. Without a cache (None), every call reads 0.0 CPU per process — fine for
    RAM-only callers.

    cpu_percent is normalized against the system core count (0-100). ram_bytes uses PSS
    (Proportional Set Size) rather than RSS so shared pages aren't double-counted across the
    process tree — summing RSS would let the total exceed system RAM. Dead children that vanish
    mid-iteration are silently skipped.
    """
    if proc_cache is None:
        proc_cache = {}

    try:
        # Get all processes in tree (main + children)
        processes = [process]
        with contextlib.suppress(psutil.NoSuchProcess):
            processes.extend(process.children(recursive=True))

        # Aggregate CPU and RAM usage across all processes, measuring through the cached
        # Process object for any PID seen on a previous call — children() returns brand-new
        # objects every call, which would reset the cpu_percent baseline each interval
        total_cpu = 0.0
        total_ram_bytes = 0

        for proc in processes:
            cached = proc_cache.setdefault(proc.pid, proc)
            try:
                # CPU: Get percentage (can be >100% for multi-core usage)
                cpu = cached.cpu_percent(interval=0.0)  # Non-blocking
                total_cpu += cpu

                # RAM: Get PSS (Proportional Set Size)
                # Use PSS instead of RSS to avoid double-counting shared memory across processes.
                # RSS counts shared pages once per process, so summing RSS across a process tree
                # can exceed system total RAM. PSS divides shared pages by # of sharing processes,
                # making it additive and accurate when summing across multiple processes.
                mem_info = cached.memory_full_info()
                total_ram_bytes += mem_info.pss

            except psutil.NoSuchProcess:
                # Process died between children() and the stat calls — drop it immediately
                proc_cache.pop(proc.pid, None)
                continue
            except (psutil.AccessDenied, psutil.ZombieProcess):
                # Still exists (just unreadable/unreaped) — keep the cache entry; it gets
                # evicted below once the PID leaves the tree
                continue

        # Evict cached entries for PIDs no longer in the tree so dead pool workers don't
        # accumulate across a long run
        tree_pids = {proc.pid for proc in processes}
        for pid in set(proc_cache) - tree_pids:
            del proc_cache[pid]

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

        # PID -> psutil.Process cache reused across monitoring intervals so per-child
        # cpu_percent(interval=0) has a baseline from the previous sample (issue #12)
        self._proc_cache: dict[int, psutil.Process] = {}

        # Detect GPUs
        self._detect_gpus()

        # Get database instance
        # Late import to avoid circular dependency (db imports from manager)
        from aetherscan.db import get_db  # noqa: PLC0415

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
        """Convenience wrapper returning (cpu_percent_total, ram_percent) from
        get_process_tree_stats() against the monitor's own root process, threading the
        persistent Process cache through so per-child CPU readings are accurate."""
        stats = get_process_tree_stats(self.process, self._proc_cache)
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
                    resource_type="cpu",
                    resource_name="system_total",
                    value=psutil.cpu_percent(interval=0.1),
                    unit="percent",
                    tag=self.tag,
                    timestamp=current_time,
                )
                self.db.write_system_resource(
                    resource_type="ram",
                    resource_name="system_total",
                    value=psutil.virtual_memory().percent,
                    unit="percent",
                    tag=self.tag,
                    timestamp=current_time,
                )

                cpu_process, ram_process = self._get_process_tree_stats()
                self.db.write_system_resource(
                    resource_type="cpu",
                    resource_name="process_tree",
                    value=cpu_process,
                    unit="percent",
                    tag=self.tag,
                    timestamp=current_time,
                )
                self.db.write_system_resource(
                    resource_type="ram",
                    resource_name="process_tree",
                    value=ram_process,
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
                        resource_type="gpu",
                        resource_name=f"{gpu_name}_utilization",
                        value=gpu_util,
                        unit="percent",
                        tag=self.tag,
                        timestamp=current_time,
                    )
                    self.db.write_system_resource(
                        resource_type="gpu",
                        resource_name=f"{gpu_name}_memory",
                        value=gpu_mem,
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

    def _annotate_stage_spans(self, axes: list, current_time: float) -> None:
        """
        Overlay this run's top-level pipeline_stages spans (depth <= 2 dot-names) as `dimgray`
        vertical boundary lines at each span's right edge on every panel in `axes`, labelled
        once (left of the line, angled 30 deg) on `axes[0]` (the CPU panel). X units match the
        panels: minutes since monitor start. Flushes and queries the DB exactly once even
        though the lines span all three panels, so spans recorded moments before shutdown
        (final_save, viz) make it onto the plot.
        """
        # The writer thread outlives the monitor (manager cleanup order: monitor before
        # db), so a flush here is safe; a timeout just means the newest spans are missing
        self.db.flush()

        rows = self.db.query_pipeline_stages(
            tag=self.tag,
            start_time=self.start_time,
            end_time=current_time,
        )
        spans = select_annotation_spans(rows)
        if not spans:
            logger.info("No top-level pipeline stage spans to annotate")
            return

        _draw_stage_boundaries(axes, spans, self.start_time, current_time)
        logger.info(
            f"Annotated {len(spans)} pipeline stage boundary line(s) across {len(axes)} panel(s)"
        )

    def _save_plot(self):
        """Generate and save resource utilization plot from database"""
        current_time = time.time()

        # Query resource metrics from database
        if self.db is None:
            raise RuntimeError("No database instance detected - cannot generate resource plot")

        all_resources = self.db.query_system_resource(
            tag=self.tag,
            start_time=self.start_time,
            end_time=current_time,
        )

        if not all_resources:
            logger.warning("No resource monitoring data to plot")
            return

        # TODO: potential memory optimization here with array pre-allocation? or instead of extracting dict -> ndarray|dict, just use dict directly? is the potential improvement worth the effort?
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

        # Create figure with 3 subplots. Extra height over the classic 3-panel figure buys the
        # headroom band above each panel that now holds the external legend (issue #214).
        fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

        # Late import to avoid circular dependency (db imports from manager)
        from aetherscan.db import get_system_metadata  # noqa: PLC0415

        metadata_json = get_system_metadata()
        machine_name = json.loads(metadata_json).get("machine_name")

        fig.suptitle(
            f"Aetherscan Pipeline: Resource Utilization ({self.tag}, {machine_name})",
            fontsize=16,
            fontweight="bold",
        )

        # CPU plot
        ax_cpu = axes[0]
        if len(cpu_process_data) > 0:
            ax_cpu.plot(
                cpu_process_timestamps,
                cpu_process_data,
                color="#1f77b4",
                linewidth=1.5,
                label="Aetherscan",
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
        ax_cpu.set_title(
            f"CPU Pressure (n={psutil.cpu_count()} cores)", fontsize=12, pad=_TITLE_PAD
        )
        _legend_above_panel(ax_cpu, *ax_cpu.get_legend_handles_labels(), ncol=2, fontsize=10)

        # RAM plot
        ax_ram = axes[1]
        if len(ram_process_data) > 0:
            ax_ram.plot(
                ram_process_timestamps,
                ram_process_data,
                color="#2ca02c",
                linewidth=1.5,
                label="Aetherscan",
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
        ax_ram.set_title(
            f"Memory Pressure (total={psutil.virtual_memory().total / (1024**3):.2f} GB)",
            fontsize=12,
            pad=_TITLE_PAD,
        )
        _legend_above_panel(ax_ram, *ax_ram.get_legend_handles_labels(), ncol=2, fontsize=10)

        # GPU plot
        ax_gpu = axes[2]
        if gpu_data and self.num_gpus > 0:
            # Create second y-axis
            ax_gpu_mem = ax_gpu.twinx()

            colors = plt.cm.tab10(np.linspace(0, 1, len(gpu_data)))

            for gpu_idx, (gpu_name, metrics) in enumerate(gpu_data.items()):
                color = colors[gpu_idx]

                # Shorten the GPU name for the legend only (whitelist -> short alias, else the
                # length-based truncation), preserving the ":<idx>" suffix. The DB
                # resource_name and self.gpu_names are untouched.
                display_name = _sanitize_gpu_display_name(gpu_name)

                # Usage (solid line, y1)
                if "utilization" in metrics:
                    timestamps, values = metrics["utilization"]
                    ax_gpu.plot(
                        timestamps,
                        values,
                        label=f"{display_name} (Usage)",
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
                        label=f"{display_name} (Memory)",
                        color=color,
                        linewidth=1.5,
                        alpha=0.6,
                        linestyle="--",
                    )

            ax_gpu.set_ylabel("GPU Usage (%)", fontsize=12, fontweight="bold")
            ax_gpu_mem.set_ylabel("GPU Memory (%)", fontsize=12, fontweight="bold")
            ax_gpu.set_ylim(0, 100)
            ax_gpu_mem.set_ylim(0, 100)

            # Combine both y-axes' legends, grouping usage and memory into their own column
            # blocks and biasing toward more rows / fewer columns so the busy legend stays
            # narrow enough to clear the centered title once moved above the panel.
            lines1, labels1 = ax_gpu.get_legend_handles_labels()
            lines2, labels2 = ax_gpu_mem.get_legend_handles_labels()
            handles, labels, ncol = _grouped_legend_entries(
                lines1, labels1, lines2, labels2, max_rows=_GPU_LEGEND_MAX_ROWS
            )
            _legend_above_panel(ax_gpu, handles, labels, ncol=ncol, fontsize=8)
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
        ax_gpu.set_title(
            f"GPU Pressure (n={self.num_gpus} devices)", fontsize=12, pad=_GPU_TITLE_PAD
        )
        ax_gpu.set_xlabel("Time (minutes)", fontsize=12, fontweight="bold")

        # Overlay top-level pipeline stage boundaries as dimgray divider lines on all three
        # panels (labelled once on the CPU panel) so resource plateaus are attributable at a
        # glance ("this plateau = round 3 data gen"). Done after all panels exist so the lines
        # land on every shared-x axis. Fully exception-guarded: a broken overlay must never
        # cost the resource plot.
        if self.config.monitor.annotate_stages:
            try:
                self._annotate_stage_spans([ax_cpu, ax_ram, ax_gpu], current_time)
            except Exception as e:
                logger.error(f"Failed to annotate pipeline stages on resource plot: {e}")

        # Reserve the headroom band above each panel for the external legends and lifted titles
        # (per-axes bbox_to_anchor handles the right-alignment). `top` leaves room above the CPU
        # panel for its lifted title to clear the suptitle; `hspace` sizes the inter-panel band
        # that holds the (tallest) GPU legend. tight_layout is intentionally not used: it doesn't
        # account for the out-of-axes legends and warns on the twinx GPU panel;
        # savefig(bbox_inches="tight") still trims the final canvas.
        fig.subplots_adjust(left=0.07, right=0.91, top=0.91, bottom=0.05, hspace=0.5)

        # Save plot
        output_path = os.path.join(
            self.config.output_path, "plots", f"resource_utilization_{self.tag}.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create dir if it doesn't exist

        plt.savefig(output_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        logger.info(f"Resource utilization plot saved to: {output_path}")

        # Upload to Slack
        logger_instance = get_logger()
        if logger_instance:
            logger_instance.upload_image_to_slack(
                output_path,
                title=f"Resource Utilization - {self.tag}",
            )


def init_monitor() -> ResourceMonitor:
    """
    Initialize global monitor instance (call once at startup)
    """
    monitor = ResourceMonitor()
    monitor.start()

    # Late import to avoid circular dependency (manager imports from monitor)
    from aetherscan.manager import register_monitor  # noqa: PLC0415

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
