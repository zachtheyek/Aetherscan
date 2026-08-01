"""Unit tests for aetherscan.display_tag.display_tag — the pure derivation of the
presentation/filename "display tag" `{command}_{machine}_{datetime}` from a run's DB tag
`{command}_{datetime}`. Dependency-free (no TF / DB import needed to exercise the function)."""

from __future__ import annotations

import pytest

from aetherscan.display_tag import _RUN_TAG_PREFIXES, display_tag


class TestRunTagRewrite:
    """A fully-resolved run tag gains the machine token between command and datetime."""

    def test_matches_the_documented_example(self):
        assert display_tag("inf_20260731_182011", "blpc3") == "inf_blpc3_20260731_182011"

    @pytest.mark.parametrize("prefix", _RUN_TAG_PREFIXES)
    def test_every_command_prefix_round_trips(self, prefix):
        # command prefix preserved, machine inserted, datetime (the rest, incl. its own
        # underscore) preserved verbatim.
        tag = f"{prefix}_20260101_000000"
        assert display_tag(tag, "bla0") == f"{prefix}_bla0_20260101_000000"

    def test_command_and_datetime_are_split_on_the_first_underscore(self):
        # The command is the part before the first underscore; the datetime is the rest.
        out = display_tag("train_20260731_182011", "m")
        assert out.split("_", 1) == ["train", "m_20260731_182011"]

    def test_distinct_machines_do_not_collide(self):
        tag = "inf_20260731_182011"
        assert display_tag(tag, "bla0") != display_tag(tag, "blpc3")


class TestNonRunTagsPassThrough:
    """Only a genuine run tag is rewritten; every other shape is returned unchanged so the
    machine name is never spliced into a name no reader reconstructs."""

    @pytest.mark.parametrize(
        "value",
        [
            "round_01",  # per-round checkpoint tag (breaks the round_(\d+) cleanup regex if rewritten)
            "round_99",
            "final",  # the conventional load alias (no datetime to place the machine before)
            "test_v30",  # a legacy hand-numbered tag (no YYYYMMDD_HHMMSS)
            "train",  # a bare command prefix (unresolved)
            "prod_20260101_000000",  # datetime shape but an unknown command prefix
            "inf_2026_182011",  # malformed datetime (too short)
            "inf_20260731_1820",  # malformed datetime (too short)
            "garbage",
            "",
            None,
        ],
    )
    def test_passes_through_unchanged(self, value):
        assert display_tag(value, "blpc3") == value

    def test_a_second_underscore_group_that_is_not_a_datetime_is_left_alone(self):
        # partition() splits on the FIRST underscore, so a non-datetime remainder is rejected
        # wholesale rather than partially rewritten.
        assert display_tag("test_v30_extra", "m") == "test_v30_extra"
