#!/usr/bin/env python3
"""
Print canonical argparse --help output for Aetherscan's top-level + train +
inference subparsers, with clean prog names so the result can be pasted
verbatim into README.md's `## CLI Reference` section.

This exists because running `python -m aetherscan.main {train,inference} --help`
emits a less clean prog name (e.g. `main.py train`) than the README expects
(`train`), and because pinning COLUMNS makes line-wrapping deterministic across
hosts.

Usage:
    PYTHONPATH=src python utils/print_cli_help.py [top|train|inference|all]

Default target is "all", which emits all three blocks separated by blank lines.

The script only imports aetherscan.config and aetherscan.cli, both pure-stdlib
modules, so it does NOT require TensorFlow or the conda env to run.
"""

from __future__ import annotations

import argparse
import os
import sys

# Pin terminal width so argparse wraps deterministically regardless of where
# this runs (CI, container, local terminal). Must be set before any argparse
# parser is constructed — argparse reads COLUMNS at HelpFormatter init time.
os.environ.setdefault("COLUMNS", "80")

from aetherscan.cli import setup_argument_parser  # noqa: E402
from aetherscan.config import init_config  # noqa: E402


def _emit(parser: argparse.ArgumentParser, prog: str) -> None:
    """Print help for `parser` with `parser.prog` overridden to `prog`."""
    saved = parser.prog
    parser.prog = prog
    try:
        parser.print_help()
    finally:
        parser.prog = saved


def main() -> None:
    init_config()
    parser = setup_argument_parser()

    subparsers: dict[str, argparse.ArgumentParser] = {
        name: action.choices[name]
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
        for name in action.choices
    }

    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    valid = {"top", "train", "inference", "all"}
    if target not in valid:
        sys.stderr.write(f"Unknown target: {target!r}. Use one of: {sorted(valid)}.\n")
        sys.exit(1)

    emit_top = target in ("top", "all")
    emit_train = target in ("train", "all")
    emit_inf = target in ("inference", "all")

    first = True
    if emit_top:
        _emit(parser, "")
        first = False
    if emit_train:
        if not first:
            print()
        _emit(subparsers["train"], "train")
        first = False
    if emit_inf:
        if not first:
            print()
        _emit(subparsers["inference"], "inference")


if __name__ == "__main__":
    main()
