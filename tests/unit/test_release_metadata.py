"""Release-metadata coupling: CITATION.cff must track pyproject.toml's version.

The CD guard enforces tag == pyproject version and the wheel smoke enforces
__version__ == guarded version, but CITATION.cff was on the honor system — a release PR
that bumps one file and not the other ships silently wrong citation metadata (#389 review).

The invariant has two phases:
- On a release commit (no ``.devN`` suffix) the two versions must be identical.
- Between releases pyproject carries the next ``X.Y.(Z+1).devN`` pre-release while
  CITATION.cff stays at the last *published* version — so equality is wrong there;
  instead CITATION must not already claim the unreleased number.

Plain text parsing on the CITATION side (no yaml dependency); tomllib is stdlib on 3.11+,
with a line-parse fallback for 3.10 (the field is static, single, and quoted).
"""

from __future__ import annotations

import os
import re

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _pyproject_version() -> str:
    path = os.path.join(_REPO_ROOT, "pyproject.toml")
    try:
        import tomllib  # noqa: PLC0415 — stdlib on 3.11+; the fallback covers 3.10

        with open(path, "rb") as f:
            return tomllib.load(f)["project"]["version"]
    except ModuleNotFoundError:
        with open(path, encoding="utf-8") as f:
            for line in f:
                match = re.match(r'^version\s*=\s*"([^"]+)"\s*$', line)
                if match:
                    return match.group(1)
        raise AssertionError("no version line found in pyproject.toml") from None


def _citation_version() -> str:
    with open(os.path.join(_REPO_ROOT, "CITATION.cff"), encoding="utf-8") as f:
        for line in f:
            # 'cff-version:' must not match; anchor to line start on the bare field.
            match = re.match(r"^version:\s*(\S+)\s*$", line)
            if match:
                return match.group(1)
    raise AssertionError("no version field found in CITATION.cff")


def test_citation_version_tracks_pyproject():
    py_version = _pyproject_version()
    cff_version = _citation_version()
    if ".dev" in py_version:
        next_release = py_version.split(".dev")[0]
        assert cff_version != next_release, (
            f"CITATION.cff ({cff_version}) already claims the unreleased {next_release} — "
            f"it must only be bumped in the release PR that drops the .dev suffix"
        )
    else:
        assert cff_version == py_version, (
            f"CITATION.cff version ({cff_version}) != pyproject.toml version ({py_version}) "
            f"— a release PR must bump both"
        )
