"""Unit tests for aetherscan.dashboard_launcher: the pure argv builder plus the safety-critical
guard paths of launch_dashboard (the dashboard is optional observability and must never fail or
hang the pipeline). The actual Streamlit spawn is I/O and stays untested; every guard around it is
mocked here."""

from __future__ import annotations

import os
import signal
import subprocess
from unittest import mock

from aetherscan.dashboard_launcher import (
    _install_signal_teardown,
    _terminate,
    build_dashboard_command,
    launch_dashboard,
)


def _fake_config(enabled=True, port=8501, output_path="/out", save_tag="t1"):
    cfg = mock.MagicMock()
    cfg.monitor.dashboard_enabled = enabled
    cfg.monitor.dashboard_port = port
    cfg.output_path = output_path
    cfg.checkpoint.save_tag = save_tag
    return cfg


def test_build_dashboard_command_structure():
    cmd = build_dashboard_command(
        "/usr/bin/python",
        "/pkg/aetherscan/dashboard.py",
        "/out/db/aetherscan.db",
        "final_v1",
        "/out/plots",
        8501,
    )
    # `python -m streamlit run <script>` up front
    assert cmd[:5] == ["/usr/bin/python", "-m", "streamlit", "run", "/pkg/aetherscan/dashboard.py"]

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


_MOD = "aetherscan.dashboard_launcher"


def test_launch_returns_none_when_config_missing():
    with mock.patch(f"{_MOD}.get_config", return_value=None):
        assert launch_dashboard() is None


def test_launch_returns_none_when_disabled():
    with mock.patch(f"{_MOD}.get_config", return_value=_fake_config(enabled=False)):
        assert launch_dashboard() is None


def test_launch_returns_none_when_script_missing():
    with (
        mock.patch(f"{_MOD}.get_config", return_value=_fake_config()),
        mock.patch(f"{_MOD}._DASHBOARD_SCRIPT") as script,
        # Truthy spec: in a streamlit-absent env the later find_spec guard also returns None,
        # which would mask a deleted is_file guard — pin the None to the guard under test.
        mock.patch(f"{_MOD}.importlib.util.find_spec", return_value=object()),
    ):
        script.is_file.return_value = False
        assert launch_dashboard() is None


def test_launch_returns_none_when_streamlit_absent():
    with (
        mock.patch(f"{_MOD}.get_config", return_value=_fake_config()),
        mock.patch(f"{_MOD}._DASHBOARD_SCRIPT") as script,
        mock.patch(f"{_MOD}.importlib.util.find_spec", return_value=None),
    ):
        script.is_file.return_value = True
        assert launch_dashboard() is None


def test_launch_returns_none_when_port_in_use():
    with (
        mock.patch(f"{_MOD}.get_config", return_value=_fake_config()),
        mock.patch(f"{_MOD}._DASHBOARD_SCRIPT") as script,
        mock.patch(f"{_MOD}.importlib.util.find_spec", return_value=object()),
        mock.patch(f"{_MOD}.get_db", return_value=None),
        mock.patch(f"{_MOD}._port_in_use", return_value=True),
        mock.patch(f"{_MOD}.subprocess.Popen") as popen,
    ):
        script.is_file.return_value = True
        assert launch_dashboard() is None
        popen.assert_not_called()  # never spawn a doomed Streamlit on a taken port


def test_launch_returns_none_when_spawn_raises():
    # A Popen failure (OSError/ValueError) must degrade to None, never propagate to the pipeline.
    with (
        mock.patch(f"{_MOD}.get_config", return_value=_fake_config()),
        mock.patch(f"{_MOD}._DASHBOARD_SCRIPT") as script,
        mock.patch(f"{_MOD}.importlib.util.find_spec", return_value=object()),
        mock.patch(f"{_MOD}.get_db", return_value=None),
        mock.patch(f"{_MOD}._port_in_use", return_value=False),
        mock.patch(f"{_MOD}.subprocess.Popen", side_effect=OSError("boom")),
    ):
        script.is_file.return_value = True
        assert launch_dashboard() is None


def test_launch_spawns_and_registers_teardown():
    proc = mock.MagicMock()
    with (
        mock.patch(f"{_MOD}.get_config", return_value=_fake_config(port=8600)),
        mock.patch(f"{_MOD}._DASHBOARD_SCRIPT") as script,
        mock.patch(f"{_MOD}.importlib.util.find_spec", return_value=object()),
        mock.patch(f"{_MOD}.get_db", return_value=None),
        mock.patch(f"{_MOD}._port_in_use", return_value=False),
        mock.patch(f"{_MOD}.subprocess.Popen", return_value=proc) as popen,
        mock.patch(f"{_MOD}.atexit.register") as atexit_reg,
        mock.patch(f"{_MOD}._install_signal_teardown") as sig_teardown,
    ):
        script.is_file.return_value = True
        script.__str__ = lambda self: "/pkg/aetherscan/dashboard.py"
        result = launch_dashboard()
        assert result is proc
        # detached so a filling pipe / group signal can't hang or kill the pipeline
        kwargs = popen.call_args.kwargs
        assert kwargs["stdout"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["start_new_session"] is True
        atexit_reg.assert_called_once_with(_terminate, proc)
        sig_teardown.assert_called_once_with(proc)


def test_signal_teardown_restores_sig_ign_disposition():
    # A prior SIG_IGN disposition must be restored as SIG_IGN, not collapsed to SIG_DFL —
    # the handler's re-delivery would otherwise terminate a process that was ignoring the
    # signal (latent: the real pipeline always installs a callable handler first).
    proc = mock.MagicMock()
    proc.poll.return_value = 0  # already exited -> _terminate is a no-op
    originals = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        _install_signal_teardown(proc)
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        with mock.patch(f"{_MOD}.os.kill") as kill:
            handler(signal.SIGTERM, None)
            kill.assert_called_once_with(os.getpid(), signal.SIGTERM)
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    finally:
        for sig, prev in originals.items():
            signal.signal(sig, prev)


def test_terminate_noop_when_already_dead():
    proc = mock.MagicMock()
    proc.poll.return_value = 0  # already exited
    with mock.patch(f"{_MOD}.os.killpg") as killpg:
        _terminate(proc)
        killpg.assert_not_called()


def test_terminate_escalates_to_sigkill_on_timeout():
    proc = mock.MagicMock()
    proc.poll.return_value = None  # still running
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=5)
    with (
        mock.patch(f"{_MOD}.os.getpgid", return_value=4321),
        mock.patch(f"{_MOD}.os.killpg") as killpg,
    ):
        _terminate(proc)
        sigs = [c.args[1] for c in killpg.call_args_list]
        assert sigs == [signal.SIGTERM, signal.SIGKILL]  # graceful then forced, whole group
