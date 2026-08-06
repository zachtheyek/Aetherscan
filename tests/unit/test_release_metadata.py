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


def _citation_field(name: str) -> str:
    with open(os.path.join(_REPO_ROOT, "CITATION.cff"), encoding="utf-8") as f:
        for line in f:
            # Anchor to line start so e.g. 'cff-version:' can't match 'version:'. Strip
            # optional YAML quotes — most other scalars in the file are quoted, so a quoted
            # version is a plausible future edit that must not read as a version mismatch.
            match = re.match(rf"^{re.escape(name)}:\s*(\S+)\s*$", line)
            if match:
                return match.group(1).strip("\"'")
    raise AssertionError(f"no {name} field found in CITATION.cff")


def test_citation_version_tracks_pyproject():
    py_version = _pyproject_version()
    cff_version = _citation_field("version")
    # Only .devN is recognized as the between-releases marker — deliberate coupling to
    # docs/RELEASE.md's versioning policy, which documents no other pre-release form.
    if ".dev" in py_version:
        next_release = py_version.split(".dev")[0]
        assert cff_version != next_release, (
            f"CITATION.cff ({cff_version}) already claims the unreleased {next_release} — "
            f"it must only be bumped in the release PR that drops the .dev suffix"
        )
        # ...and it must not be AHEAD of the next release either (plain X.Y.Z numbering
        # per the release policy, so an int-tuple compare suffices).
        assert tuple(map(int, cff_version.split("."))) < tuple(map(int, next_release.split("."))), (
            f"CITATION.cff ({cff_version}) is ahead of the next pre-release "
            f"({py_version}) — it must lag pyproject between releases"
        )
    else:
        assert cff_version == py_version, (
            f"CITATION.cff version ({cff_version}) != pyproject.toml version ({py_version}) "
            f"— a release PR must bump both"
        )


def test_citation_release_date_is_valid_and_not_future():
    """Runbook step 3 bumps `date-released` alongside `version`; guard the format and the
    obvious wrongness (a date after today means the release PR predicted and nobody
    re-confirmed at tag time)."""
    import datetime  # noqa: PLC0415

    date_released = _citation_field("date-released")
    parsed = datetime.date.fromisoformat(date_released)  # raises on non-ISO dates
    # +1 day of latitude: the maintainer's timezone (GMT+8) is ahead of CI's UTC clock, so
    # a release dated "today" locally is legitimately tomorrow from UTC's view.
    assert parsed <= datetime.date.today() + datetime.timedelta(days=1), (
        f"CITATION.cff date-released ({date_released}) is in the future — re-confirm it at "
        f"release time"
    )
