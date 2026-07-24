"""
Console entry point (`aetherscan-dashboard`) for manual runs of the live dashboard.

main.py auto-launches the dashboard alongside a run (dashboard_launcher.py); this shim covers the
ad-hoc case — inspecting a saved DB after the fact — replacing the verbose manual incantation from
dashboard.py's docstring with:

    aetherscan-dashboard --db-path /path/to/aetherscan.db --tag train_20260101_120000

Everything on the command line is forwarded verbatim to dashboard.py's argparse (--db-path, --tag,
--plots-dir, --refresh). Streamlit must OWN the process (`streamlit run <file>`) for the `st.*`
calls to render, so this execs `python -m streamlit run` in place — a plain
`python -m aetherscan.dashboard` would call `st.*` outside a ScriptRunContext and render nothing.
In a source checkout / the container (no installed entry point), the same shim runs as
`PYTHONPATH=src python -m aetherscan.dashboard_cli <args>`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Same resolution as dashboard_launcher._DASHBOARD_SCRIPT — not imported from there, so this shim
# stays stdlib-only and the console script works (and fails fast) without the pipeline deps.
_DASHBOARD_SCRIPT = Path(__file__).resolve().parent / "dashboard.py"


def build_exec_argv(argv: list[str]) -> list[str]:
    """`python -m streamlit run <packaged dashboard.py> -- <argv>` — build_dashboard_command's
    shape, with the caller's args forwarded verbatim to dashboard.py's argparse. Pure/testable."""
    return [sys.executable, "-m", "streamlit", "run", str(_DASHBOARD_SCRIPT), "--", *argv]


def main(argv: list[str] | None = None) -> None:
    """Replace this process with the Streamlit server (os.execv never returns on success)."""
    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit(
            "aetherscan-dashboard: streamlit is not installed. "
            "Install the dashboard extra: pip install 'aetherscan[dashboard]'"
        )
    # Parity with dashboard_launcher._DASHBOARD_SCRIPT.is_file(): fail with a clear message
    # instead of a cryptic os.execv OSError / Streamlit error if the packaged script is absent.
    if not _DASHBOARD_SCRIPT.is_file():
        raise SystemExit(f"aetherscan-dashboard: dashboard script not found: {_DASHBOARD_SCRIPT}")
    os.execv(sys.executable, build_exec_argv(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
