"""Unit tests for aetherscan.dashboard_cli: the `aetherscan-dashboard` console entry point that
re-execs `python -m streamlit run dashboard.py -- <args>` for manual dashboard runs. The exec
itself is mocked — only the argv construction and the streamlit-missing guard are under test."""

from __future__ import annotations

import sys
from unittest import mock

import pytest

from aetherscan.dashboard_cli import _DASHBOARD_SCRIPT, build_exec_argv, main

_MOD = "aetherscan.dashboard_cli"


def test_build_exec_argv_shape():
    argv = build_exec_argv(["--db-path", "/out/db/aetherscan.db", "--tag", "final_v1"])
    # `python -m streamlit run <packaged dashboard.py>` up front — streamlit must own the process
    assert argv[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert argv[4] == str(_DASHBOARD_SCRIPT)
    assert argv[4].endswith("dashboard.py")
    # everything after `--` is the caller's args, forwarded verbatim to dashboard.py's argparse
    assert argv[5] == "--"
    assert argv[6:] == ["--db-path", "/out/db/aetherscan.db", "--tag", "final_v1"]


def test_build_exec_argv_no_args():
    # bare `aetherscan-dashboard` — trailing `--` with nothing after it is valid for streamlit
    assert build_exec_argv([])[-1] == "--"


def test_main_execs_streamlit():
    with (
        mock.patch(f"{_MOD}.importlib.util.find_spec", return_value=object()),
        mock.patch(f"{_MOD}.os.execv") as execv,
    ):
        main(["--db-path", "/x/aetherscan.db"])
        execv.assert_called_once_with(
            sys.executable, build_exec_argv(["--db-path", "/x/aetherscan.db"])
        )


def test_main_defaults_to_sys_argv():
    with (
        mock.patch(f"{_MOD}.importlib.util.find_spec", return_value=object()),
        mock.patch(f"{_MOD}.os.execv") as execv,
        mock.patch.object(sys, "argv", ["aetherscan-dashboard", "--tag", "t1"]),
    ):
        main()
        assert execv.call_args.args[1][-2:] == ["--tag", "t1"]  # argv[0] not forwarded


def test_main_exits_when_streamlit_absent():
    # A bare install still gets the console script; it must fail fast with a hint, never exec
    with (
        mock.patch(f"{_MOD}.importlib.util.find_spec", return_value=None),
        mock.patch(f"{_MOD}.os.execv") as execv,
    ):
        with pytest.raises(SystemExit, match="streamlit is not installed"):
            main([])
        execv.assert_not_called()
