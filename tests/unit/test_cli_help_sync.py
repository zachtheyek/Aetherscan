"""Guards against drift between cli.py's --help output and the README CLI Reference blocks.

The README's `## CLI Reference` fenced blocks are generated from cli.py by
utils/print_cli_help.py and must be regenerated whenever a CLI flag changes (see
CONTRIBUTING.md). This test drives that util's per-subcommand output under the same pinned
terminal width the docs were generated with and asserts each README block still matches
byte-for-byte, so a forgotten regeneration fails CI instead of silently rotting the docs.
"""

from __future__ import annotations

import importlib.util
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_README = _REPO_ROOT / "README.md"

# print_cli_help.py is a standalone script under utils/, not an importable package, so load it
# straight from its path — this drives the exact code a maintainer runs to regenerate the docs.
_spec = importlib.util.spec_from_file_location(
    "print_cli_help", _REPO_ROOT / "utils" / "print_cli_help.py"
)
_print_cli_help = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_print_cli_help)

# The subparser targets print_cli_help.py accepts; each maps to one README CLI Reference block.
_TARGETS = ("top", "train", "inference")


def _util_help(target: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """Capture print_cli_help.py's stdout for one target under a pinned 80-column width."""
    # argparse reads COLUMNS when it builds the HelpFormatter, so line wrapping depends on it.
    # Pin it to reproduce exactly the width the README was generated with (print_cli_help.py
    # defaults COLUMNS to "80"), independent of the ambient terminal.
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr(sys, "argv", ["print_cli_help.py", target])
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_cli_help.main()
    return buf.getvalue()


def _readme_block(target: str) -> str:
    """Extract the fenced block whose intro line cites `print_cli_help.py <target>`."""
    lines = _README.read_text().splitlines()
    anchor = re.compile(rf"print_cli_help\.py {re.escape(target)}\b")
    intro = next(n for n, line in enumerate(lines) if anchor.search(line))
    open_fence = next(n for n in range(intro + 1, len(lines)) if lines[n].strip() == "```")
    close_fence = next(n for n in range(open_fence + 1, len(lines)) if lines[n].strip() == "```")
    return "\n".join(lines[open_fence + 1 : close_fence]) + "\n"


@pytest.mark.parametrize("target", _TARGETS)
def test_readme_cli_reference_matches_help(target: str, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _util_help(target, monkeypatch) == _readme_block(target), (
        f"README CLI Reference for '{target}' is stale; regenerate with "
        f"`PYTHONPATH=src python utils/print_cli_help.py {target}` (or `all`)."
    )
