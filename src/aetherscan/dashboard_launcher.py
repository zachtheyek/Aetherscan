"""
Auto-launch the live monitoring dashboard (utils/dashboard.py) alongside a train/inference run.

main.py calls launch_dashboard() once at startup (after the DB is initialized). It spawns a
headless Streamlit server as a detached subprocess pointed at this run's DB + tag, logs the
SSH-forward instructions, and registers an atexit hook to tear it down. Fully guarded: a missing
streamlit, a missing dashboard.py, or any spawn failure degrades to a warning — the dashboard is
optional observability and must never fail the pipeline. Opt out with --no-dashboard
(config.monitor.dashboard_enabled = False).
"""

from __future__ import annotations

import atexit
import importlib.util
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path

from aetherscan.config import get_config
from aetherscan.db import get_db

logger = logging.getLogger(__name__)

# utils/dashboard.py sits at the repo root; this module is src/aetherscan/dashboard_launcher.py
_DASHBOARD_SCRIPT = Path(__file__).resolve().parents[2] / "utils" / "dashboard.py"


def build_dashboard_command(
    python_exe: str,
    dashboard_script: str,
    db_path: str,
    tag: str,
    plots_dir: str,
    port: int,
) -> list[str]:
    """Construct the `python -m streamlit run ...` argv. Streamlit's own flags precede `--`;
    everything after `--` is forwarded to dashboard.py's argparse. Pure/testable."""
    return [
        python_exe,
        "-m",
        "streamlit",
        "run",
        dashboard_script,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--",
        "--db-path",
        db_path,
        "--tag",
        tag,
        "--plots-dir",
        plots_dir,
    ]


def _terminate(proc: subprocess.Popen) -> None:
    """Best-effort teardown at interpreter exit (no logging — runs during shutdown)."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


def launch_dashboard() -> subprocess.Popen | None:
    """Spawn the headless Streamlit dashboard for this run, or return None (disabled / failed)."""
    config = get_config()
    if config is None or not config.monitor.dashboard_enabled:
        return None

    if not _DASHBOARD_SCRIPT.is_file():
        logger.warning(f"Dashboard not launched: {_DASHBOARD_SCRIPT} not found")
        return None

    if importlib.util.find_spec("streamlit") is None:
        logger.warning(
            "Dashboard skipped: streamlit not installed. Rebuild the NGC container (.sif) or the "
            "conda env after the streamlit/plotly deps landed, or run with --no-dashboard."
        )
        return None

    db = get_db()
    db_path = (
        db.db_path if db is not None else os.path.join(config.output_path, "db", "aetherscan.db")
    )
    tag = config.checkpoint.save_tag
    plots_dir = os.path.join(config.output_path, "plots")
    port = config.monitor.dashboard_port

    cmd = build_dashboard_command(
        sys.executable, str(_DASHBOARD_SCRIPT), db_path, tag, plots_dir, port
    )

    try:
        # start_new_session detaches the server from the pipeline's process group so a SIGINT to
        # the pipeline doesn't also kill the dashboard mid-write; atexit tears it down cleanly.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, ValueError) as e:
        logger.warning(f"Dashboard failed to launch (is streamlit installed?): {e}")
        return None

    atexit.register(_terminate, proc)
    host = socket.gethostname()
    logger.info(
        f"Live dashboard for '{tag}' on port {port}. Reach it with: "
        f"ssh -L {port}:localhost:{port} {host}  then open http://localhost:{port}"
    )
    return proc
