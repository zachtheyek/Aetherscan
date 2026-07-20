"""
Auto-launch the live monitoring dashboard (aetherscan/dashboard.py) alongside a train/inference run.

main.py calls launch_dashboard() once at startup (after the DB is initialized). It spawns a
headless Streamlit server as a detached subprocess pointed at this run's DB + tag, logs the
SSH-forward instructions, and registers atexit + SIGTERM/SIGINT teardown so the server is reaped
whether the pipeline exits gracefully or is signal-killed. Fully guarded: a missing streamlit, a
missing dashboard.py, a port already in use, or any spawn failure degrades to a warning — the
dashboard is optional observability and must never fail the pipeline. Opt out with --no-dashboard
(config.monitor.dashboard_enabled = False).
"""

from __future__ import annotations

import atexit
import contextlib
import importlib.util
import logging
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

from aetherscan.config import get_config
from aetherscan.db import get_db

logger = logging.getLogger(__name__)

# dashboard.py ships alongside this module inside the package, so it resolves identically for a
# source checkout, the container, and a pip install (streamlit runs it by file path).
_DASHBOARD_SCRIPT = Path(__file__).resolve().parent / "dashboard.py"


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


def _port_in_use(port: int) -> bool:
    """True if `port` can't be bound on localhost — i.e. a concurrent run's dashboard already holds
    it. Best-effort probe; either way the caller only ever warns, never fails the pipeline."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
    except OSError:
        return True
    return False


def _terminate(proc: subprocess.Popen) -> None:
    """Best-effort teardown (no logging — may run during interpreter shutdown or in a signal
    handler). The dashboard is spawned with start_new_session=True, so it leads its own process
    group; signal the whole group (killpg) to reap Streamlit AND any grandchildren, not just the
    direct child."""
    if proc.poll() is not None:
        return

    def _signal_group(sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            # group already gone, or getpgid raced the exit — fall back to the direct child
            with contextlib.suppress(Exception):
                proc.send_signal(sig)

    try:
        _signal_group(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _signal_group(signal.SIGKILL)
    except Exception:
        pass


def _install_signal_teardown(proc: subprocess.Popen) -> None:
    """Reap the dashboard on SIGTERM/SIGINT too — atexit hooks do NOT run on a signal-kill, and
    the new-session detachment means a process-group signal to the pipeline never reaches the
    dashboard. Each handler tears down the dashboard, restores the previous disposition, and
    re-raises the signal so the pipeline's own shutdown proceeds exactly as before (no logging in
    the handler — that can deadlock). Signal handlers can only be installed from the main thread."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(sig)
        except (ValueError, OSError):
            continue  # not the main thread / unsupported — atexit still covers graceful exits

        def _handler(signum, frame, _prev=prev):
            _terminate(proc)
            signal.signal(signum, _prev if callable(_prev) else signal.SIG_DFL)
            os.kill(os.getpid(), signum)  # re-deliver so the pipeline shuts down as it would have

        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _handler)


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
            "Dashboard skipped: streamlit not installed. Install the extra "
            "(pip install 'aetherscan[dashboard]'), rebuild the NGC container (.sif) / conda env "
            "after the streamlit/plotly deps landed, or run with --no-dashboard."
        )
        return None

    db = get_db()
    db_path = (
        db.db_path if db is not None else os.path.join(config.output_path, "db", "aetherscan.db")
    )
    tag = config.checkpoint.save_tag
    plots_dir = os.path.join(config.output_path, "plots")
    port = config.monitor.dashboard_port

    if _port_in_use(port):
        # A concurrent run on this node already holds the port. Don't spawn a Streamlit that would
        # exit immediately (stderr is DEVNULL'd) and, worse, don't log an SSH-forward line that
        # would tunnel the user to the OTHER run's dashboard — a wrong-tag/wrong-DB footgun.
        logger.warning(
            f"Dashboard skipped: port {port} is already in use (another run's dashboard?). "
            f"Free the port or pass --dashboard-port to pick another."
        )
        return None

    cmd = build_dashboard_command(
        sys.executable, str(_DASHBOARD_SCRIPT), db_path, tag, plots_dir, port
    )

    try:
        # start_new_session puts the server in its own session/process group so a Ctrl-C sent to
        # the pipeline's whole foreground group doesn't kill it mid-write; teardown is instead
        # driven deterministically by our atexit hook (graceful exit) and SIGTERM/SIGINT handlers
        # (signal-kill), both of which killpg the new group to reap Streamlit + any grandchildren.
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
    _install_signal_teardown(proc)
    host = socket.gethostname()
    logger.info(
        f"Live dashboard for '{tag}' on port {port}. Reach it with: "
        f"ssh -L {port}:localhost:{port} {host}  then open http://localhost:{port}"
    )
    return proc
