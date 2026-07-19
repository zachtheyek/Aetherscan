"""Unit tests for aetherscan.dashboard_launcher.build_dashboard_command — the pure argv builder
for the auto-launched Streamlit dashboard (the spawn itself is I/O and not unit-tested)."""

from __future__ import annotations

from aetherscan.dashboard_launcher import build_dashboard_command


def test_build_dashboard_command_structure():
    cmd = build_dashboard_command(
        "/usr/bin/python",
        "/repo/utils/dashboard.py",
        "/out/db/aetherscan.db",
        "final_v1",
        "/out/plots",
        8501,
    )
    # `python -m streamlit run <script>` up front
    assert cmd[:5] == ["/usr/bin/python", "-m", "streamlit", "run", "/repo/utils/dashboard.py"]

    # streamlit's own flags precede `--`; dashboard.py's argparse args follow it
    sep = cmd.index("--")
    streamlit_flags = cmd[5:sep]
    script_args = cmd[sep + 1 :]

    assert "--server.headless" in streamlit_flags
    assert streamlit_flags[streamlit_flags.index("--server.port") + 1] == "8501"
    assert script_args == [
        "--db-path",
        "/out/db/aetherscan.db",
        "--tag",
        "final_v1",
        "--plots-dir",
        "/out/plots",
    ]


def test_port_is_stringified():
    cmd = build_dashboard_command("py", "d.py", "db", "t", "p", 9000)
    assert "9000" in cmd and 9000 not in cmd  # streamlit needs a string port
