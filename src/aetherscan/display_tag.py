"""
Presentation/filename "display tag" derivation.

A run's DB tag is ``{command}_{datetime}`` (e.g. ``inf_20260731_182011``; see
``cli.resolve_save_tag``). Two runs of the same command that start in the same second on two
different machines therefore resolve to the *same* tag, and their on-disk artifacts collide the
moment they share a directory (e.g. weights copied bla0 -> blpc3). The **display tag** inserts the
local machine name — ``{command}_{machine}_{datetime}`` (e.g. ``inf_blpc3_20260731_182011``) — and
is used for every artifact FILENAME/path and every plot title + Slack message, so those no longer
collide.

The display tag is a DERIVED, presentation-only string: nothing that keys a DB row changes. The tag
written to and queried from the database (and ``config.checkpoint.save_tag`` itself) stays the plain
``{command}_{datetime}``.

This module is intentionally dependency-free (stdlib only, no ``aetherscan`` imports) so it is
import-safe from every consumer — train, inference, viz, monitor, logger, HF upload — with no risk
of an import cycle, and so ``display_tag`` is unit-testable in isolation. The machine name comes
from the one accessor ``aetherscan.db.get_machine_name``; callers pass it in.
"""

from __future__ import annotations

import re

# Fully-resolved run-tag prefixes — mirrors cli._SAVE_TAG_PREFIXES / cli._FULL_TAG_PATTERN. Kept
# local (a 4-tuple duplicate) so this module stays dependency-free; the set changes ~never.
_RUN_TAG_PREFIXES = ("test", "train", "inf", "bench")
# The datetime stamp resolve_save_tag appends: %Y%m%d_%H%M%S.
_DATETIME_RE = re.compile(r"\d{8}_\d{6}")
# Filename-safe machine-name characters; any run of anything else collapses to a single '-'. A
# real RFC-1123 hostname already satisfies this, so blpc3/bla0 are untouched — this only hardens a
# pathological hostname (a path separator, space, or other filesystem-hostile character) from
# breaking a path. (It deliberately preserves `_`/digits, so it is NOT a guard against a hostname
# that itself looks like a run-tag component — only run tags with a {test,train,inf,bench} prefix
# are ever machine-scoped, so that collision can't arise here.)
_MACHINE_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_machine_name(machine_name: str) -> str:
    """Collapse any non-filename-safe characters in a machine name to '-' (identity on the
    RFC-1123 hostnames this pipeline actually runs on)."""
    return _MACHINE_SANITIZE_RE.sub("-", machine_name)


def display_tag(tag: str, machine_name: str) -> str:
    """Return the machine-scoped display tag for ``tag``.

    A fully-resolved run tag ``{command}_{datetime}`` (command in {test, train, inf, bench},
    datetime ``YYYYMMDD_HHMMSS``) becomes ``{command}_{machine_name}_{datetime}`` — the command
    prefix is the part before the first underscore, the datetime is the rest.

    Anything that is not a run tag is returned UNCHANGED: a ``round_XX`` per-round checkpoint tag,
    the conventional ``final`` alias, a falsy value (``None`` / ``""`` — an as-yet-unresolved tag),
    or any malformed string. This selectivity is load-bearing — splicing the machine name into
    ``round_05`` would both break the ``round_(\\d+)`` checkpoint cleanup regex and produce a name
    no reader reconstructs, and ``final`` has no datetime to place the machine before. So filename
    choke points can call this unconditionally: it is the identity on the non-run-tag flavors and
    only rewrites genuine run tags.
    """
    if not tag:
        return tag
    command, sep, datetime_part = tag.partition("_")
    if not sep or command not in _RUN_TAG_PREFIXES or not _DATETIME_RE.fullmatch(datetime_part):
        return tag
    return f"{command}_{_sanitize_machine_name(machine_name)}_{datetime_part}"
