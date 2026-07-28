"""Unit tests for aetherscan.monitor.get_process_tree_stats(): the PID -> Process cache
that fixes issue #12's CPU undercount (a freshly created psutil.Process reads 0.0 on its
first cpu_percent(interval=0) call, and children() manufactures fresh objects every call).

Importing aetherscan.monitor pulls TensorFlow at collection (monitor.py imports tf for GPU
detection) — same as most unit modules. The tests themselves drive the pure function with
fake Process objects; no monitor instance, GPU, or cluster data involved.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import psutil
import pytest
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

from aetherscan.monitor import get_process_tree_stats
from aetherscan.monitor.monitor import (
    _draw_stage_boundaries,
    _grouped_legend_entries,
    _sanitize_gpu_display_name,
)


class FakeProcess:
    """Mimics the psutil.Process surface get_process_tree_stats() touches.

    cpu_percent() reproduces psutil's documented behavior for interval=0: the first call on
    a given object has no baseline and returns 0.0; subsequent calls return `cpu`.
    """

    def __init__(self, pid: int, cpu: float = 0.0, pss: int = 0):
        self.pid = pid
        self.cpu_calls = 0
        self._cpu = cpu
        self._pss = pss
        self.dead = False
        self.access_denied = False

    def cpu_percent(self, interval: float = 0.0) -> float:
        if self.dead:
            raise psutil.NoSuchProcess(self.pid)
        if self.access_denied:
            raise psutil.AccessDenied(self.pid)
        self.cpu_calls += 1
        return 0.0 if self.cpu_calls == 1 else self._cpu

    def memory_full_info(self):
        if self.dead:
            raise psutil.NoSuchProcess(self.pid)
        return SimpleNamespace(pss=self._pss)


class FakeRoot(FakeProcess):
    """Root process whose children() manufactures brand-new FakeProcess objects on every
    call — exactly what psutil.Process.children(recursive=True) does, and the reason the
    cache is needed. `spawned` records every manufactured child for identity assertions."""

    def __init__(self, pid: int = 1, cpu: float = 0.0, pss: int = 0):
        super().__init__(pid, cpu, pss)
        self.child_specs: list[tuple[int, float, int]] = []  # (pid, cpu, pss)
        self.spawned: list[FakeProcess] = []

    def children(self, recursive: bool = False) -> list[FakeProcess]:
        procs = [FakeProcess(pid, cpu, pss) for pid, cpu, pss in self.child_specs]
        self.spawned.extend(procs)
        return procs


@pytest.fixture(autouse=True)
def _system_totals(monkeypatch):
    """Pin core count (4) and total RAM (10_000 bytes) so normalization is deterministic."""
    monkeypatch.setattr(psutil, "cpu_count", lambda: 4)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(total=10_000, percent=0.0)
    )


class TestProcessCache:
    def test_cached_object_reused_for_stable_pid(self):
        root = FakeRoot()
        root.child_specs = [(2, 100.0, 0)]
        cache: dict[int, FakeProcess] = {}

        get_process_tree_stats(root, cache)
        first_seen = root.spawned[0]
        get_process_tree_stats(root, cache)

        # Second call manufactured a fresh object for pid 2, but measurement went through
        # the cached first-seen object (fresh one untouched)
        assert cache[2] is first_seen
        assert first_seen.cpu_calls == 2
        assert root.spawned[1].cpu_calls == 0

    def test_cpu_accurate_from_second_sample(self):
        root = FakeRoot(cpu=40.0)
        root.child_specs = [(2, 100.0, 0), (3, 100.0, 0)]
        cache: dict[int, FakeProcess] = {}

        first = get_process_tree_stats(root, cache)
        second = get_process_tree_stats(root, cache)

        # First sample: every object is new -> all 0.0 (the one-time cost the fix accepts)
        assert first["cpu_percent"] == 0.0
        # Second sample: (40 + 100 + 100) / 4 cores
        assert second["cpu_percent"] == pytest.approx(60.0)

    def test_new_pid_contributes_one_zero_reading(self):
        root = FakeRoot()
        root.child_specs = [(2, 100.0, 0)]
        cache: dict[int, FakeProcess] = {}

        get_process_tree_stats(root, cache)
        root.child_specs.append((3, 60.0, 0))

        # pid 2 is warm (100.0); pid 3 is new -> 0.0 this interval
        second = get_process_tree_stats(root, cache)
        assert second["cpu_percent"] == pytest.approx(100.0 / 4)

        # Next interval both are warm
        third = get_process_tree_stats(root, cache)
        assert third["cpu_percent"] == pytest.approx((100.0 + 60.0) / 4)

    def test_departed_pids_evicted_from_cache(self):
        root = FakeRoot()
        root.child_specs = [(2, 0.0, 0), (3, 0.0, 0)]
        cache: dict[int, FakeProcess] = {}

        get_process_tree_stats(root, cache)
        assert set(cache) == {1, 2, 3}

        root.child_specs = [(2, 0.0, 0)]
        get_process_tree_stats(root, cache)
        assert set(cache) == {1, 2}

    def test_nosuchprocess_evicts_and_skips(self):
        root = FakeRoot(cpu=40.0)
        root.child_specs = [(2, 100.0, 0)]
        cache: dict[int, FakeProcess] = {}

        get_process_tree_stats(root, cache)
        # pid 2 dies between children() enumeration and the stat calls
        cache[2].dead = True

        stats = get_process_tree_stats(root, cache)
        assert 2 not in cache
        assert stats["cpu_percent"] == pytest.approx(40.0 / 4)  # root still counted

    def test_access_denied_skipped_but_kept_in_cache(self):
        root = FakeRoot(cpu=40.0)
        root.child_specs = [(2, 100.0, 0)]
        cache: dict[int, FakeProcess] = {}

        get_process_tree_stats(root, cache)
        cache[2].access_denied = True

        stats = get_process_tree_stats(root, cache)
        assert stats["cpu_percent"] == pytest.approx(40.0 / 4)
        assert 2 in cache  # process still exists; only this sample is skipped


class TestAggregation:
    def test_ram_sums_pss_across_tree(self):
        root = FakeRoot(pss=1_000)
        root.child_specs = [(2, 0.0, 2_000), (3, 0.0, 3_000)]

        stats = get_process_tree_stats(root, {})
        assert stats["ram_bytes"] == 6_000
        assert stats["ram_gb"] == pytest.approx(6_000 / 1e9)
        assert stats["ram_percent"] == pytest.approx(60.0)  # of the pinned 10_000 total

    def test_no_cache_still_aggregates_ram(self):
        # RAM-only callers (manager._get_memory_usage) pass no cache and build a fresh root
        # Process object per call; PSS aggregation must work regardless, while CPU stays at
        # the no-baseline 0.0
        for _ in range(2):
            root = FakeRoot(cpu=40.0, pss=1_000)
            root.child_specs = [(2, 100.0, 2_000)]
            stats = get_process_tree_stats(root)
            assert stats["cpu_percent"] == 0.0
            assert stats["ram_bytes"] == 3_000

    def test_unreadable_root_returns_zeros(self):
        # AccessDenied from children() escapes the NoSuchProcess suppression and lands in
        # the outer guard, which degrades to all-zero stats instead of raising
        root = FakeRoot()

        def _raise(recursive: bool = False):
            raise psutil.AccessDenied(root.pid)

        root.children = _raise

        stats = get_process_tree_stats(root, {})
        assert stats == {
            "cpu_percent": 0.0,
            "ram_percent": 0.0,
            "ram_bytes": 0,
            "ram_gb": 0.0,
        }


class TestSanitizeGpuDisplayName:
    """The resource-plot legend name sanitization (issue #214): whitelist hits collapse to a
    short alias, misses fall back to the pre-existing 20-char truncation, and the ":<idx>"
    suffix is preserved throughout. Plot-legend only — DB resource_name / gpu_names untouched."""

    def test_a4000_whitelist_hit(self):
        assert _sanitize_gpu_display_name("NVIDIA RTX A4000:0") == "A4000:0"

    def test_pro6000_whitelist_hit_preserves_maxq_and_idx(self):
        raw = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition:2"
        assert _sanitize_gpu_display_name(raw) == "PRO 6000:2"

    def test_non_whitelisted_long_name_truncates_to_19_chars_plus_ellipsis(self):
        raw = "NVIDIA GeForce RTX 3090 Ti Founders:4"
        # Name part is 35 chars (> 20) -> first 19 chars + "..." + preserved ":4"
        expected = f"{'NVIDIA GeForce RTX 3090 Ti Founders'[:19]}...:4"
        assert _sanitize_gpu_display_name(raw) == expected == "NVIDIA GeForce RTX ...:4"

    def test_short_non_whitelisted_name_unchanged(self):
        assert _sanitize_gpu_display_name("GPU:1") == "GPU:1"

    def test_long_non_whitelisted_name_without_colon_still_truncates(self):
        # No ":<idx>" suffix — the truncation fallback must still fire (the suffixless case).
        raw = "NVIDIA GeForce RTX 3090 Ti Founders"
        assert _sanitize_gpu_display_name(raw) == f"{raw[:19]}..."


class TestDrawStageBoundaries:
    """The boundary-line/label overlay (issue #214): a divider line on every panel at each
    span's right edge, with the leaf stage name labelled once on the CPU (first) panel."""

    @staticmethod
    def _boundary_patches(fig):
        """The figure-level ConnectionPatch artists _draw_stage_boundaries adds (#280)."""
        return [artist for artist in fig.artists if isinstance(artist, ConnectionPatch)]

    def test_one_figure_line_per_span_labels_on_cpu_only(self):
        fig, axes = plt.subplots(3, 1)
        try:
            spans = [
                {"stage": "load_backgrounds", "start_time": 0, "end_time": 600},
                {"stage": "train.round_01", "start_time": 600, "end_time": 1200},
                {"stage": "train.round_02", "start_time": 1200, "end_time": 1800},
            ]
            _draw_stage_boundaries(list(axes), spans, start_time=0, current_time=1800)

            # ONE figure-level line per span (#280) — the panels themselves stay untouched
            boundaries = self._boundary_patches(fig)
            assert len(boundaries) == len(spans)
            for ax in axes:
                assert len(ax.lines) == 0

            # Labels only on the CPU (first) panel, one per span, leaf name only
            assert [t.get_text() for t in axes[0].texts] == [
                "load_backgrounds",
                "round_01",
                "round_02",
            ]
            assert len(axes[1].texts) == 0
            assert len(axes[2].texts) == 0

            # Boundaries sit at each span's end time in minutes since start (600s -> 10min, ...)
            xs = [patch.xy1[0] for patch in boundaries]
            assert xs == pytest.approx([10.0, 20.0, 30.0])

            # Labels are dimgray and angled 30 deg from horizontal — byte-identical to the
            # pre-#280 annotations (only the lines changed)
            assert axes[0].texts[0].get_color() == "dimgray"
            assert axes[0].texts[0].get_rotation() == pytest.approx(30.0)
        finally:
            plt.close(fig)

    def test_boundary_line_style_and_span(self):
        # #280 acceptance: dashed, semi-transparent, dimgray, spanning from the top of the
        # first panel (y=1 in axes fraction) to the bottom of the last (y=0)
        fig, axes = plt.subplots(3, 1)
        try:
            spans = [{"stage": "round_01", "start_time": 0, "end_time": 600}]
            _draw_stage_boundaries(list(axes), spans, start_time=0, current_time=1800)
            [boundary] = self._boundary_patches(fig)
            assert boundary.get_linestyle() == "--"
            assert boundary.get_alpha() is not None and boundary.get_alpha() < 1.0
            assert boundary.get_edgecolor()[:3] == mcolors.to_rgb("dimgray")
            assert boundary.xy1 == (10.0, 1.0)
            assert boundary.xy2 == (10.0, 0.0)
            # Endpoint transforms anchor to the FIRST and LAST axes respectively
            assert boundary.coords1 is axes[0].get_xaxis_transform()
            assert boundary.coords2 is axes[-1].get_xaxis_transform()
        finally:
            plt.close(fig)

    def test_end_time_clamped_to_current_time(self):
        # A span still open at teardown (end_time in the future) is clamped to current_time
        fig, axes = plt.subplots(3, 1)
        try:
            spans = [{"stage": "final_save", "start_time": 0, "end_time": 9_999}]
            _draw_stage_boundaries(list(axes), spans, start_time=0, current_time=1800)
            # min(9999, 1800) = 1800s -> 30 min
            [boundary] = self._boundary_patches(fig)
            assert boundary.xy1[0] == pytest.approx(30.0)
        finally:
            plt.close(fig)


class TestGroupedLegendEntries:
    """The column-major legend grid helper (issue #217): two GPU-metric groups (usage + memory)
    each occupy their own column block, padded with invisible handles so the second group always
    starts a fresh column, and a group grows a second column only once it exceeds `max_rows`.
    Pure helper — handles pass straight through, so plain sentinels stand in for Line2D artists."""

    @pytest.mark.parametrize(
        ("n_usage", "n_memory", "max_rows", "expected_ncol", "expected_slots"),
        [
            # No wrap (n <= max_rows -> cols_per_group=1, ncol=2): the two groups sit in one
            # column each, and the memory group starts at the column boundary (slots).
            pytest.param(3, 3, 5, 2, 3, id="no_wrap_equal"),
            # No wrap with a shorter memory group: it is padded up to a whole column (slots).
            pytest.param(3, 2, 5, 2, 3, id="no_wrap_padded"),
            # The docstring's worked example: 5 GPUs at max_rows>=5 -> columns of 5 | 5.
            pytest.param(5, 5, 5, 2, 5, id="docstring_5x5"),
            # Wrap (n > max_rows -> cols_per_group=2, ncol=4): each group spans two columns.
            pytest.param(6, 6, 5, 4, 6, id="wrap_equal"),
            # Wrap with a shorter memory group: padded up to two whole columns (slots).
            pytest.param(6, 4, 5, 4, 6, id="wrap_padded"),
        ],
    )
    def test_grid_shape_and_column_major_padding(
        self, n_usage, n_memory, max_rows, expected_ncol, expected_slots
    ):
        # Unique sentinels so real entries are identifiable by identity and padding entries
        # (Line2D with alpha=0) are unambiguously distinguishable from them.
        usage_handles = [object() for _ in range(n_usage)]
        usage_labels = [f"usage{i}" for i in range(n_usage)]
        memory_handles = [object() for _ in range(n_memory)]
        memory_labels = [f"mem{i}" for i in range(n_memory)]

        handles, labels, ncol = _grouped_legend_entries(
            usage_handles, usage_labels, memory_handles, memory_labels, max_rows=max_rows
        )

        assert ncol == expected_ncol
        # Both groups are padded to `slots`, so the output is exactly two column blocks.
        assert len(handles) == len(labels) == 2 * expected_slots

        def _assert_group(offset, real_handles, real_labels):
            # Real entries land first, unchanged and in order...
            assert handles[offset : offset + len(real_handles)] == real_handles
            assert labels[offset : offset + len(real_labels)] == real_labels
            # ...then invisible Line2D handles + "" labels pad the block up to `slots`.
            for i in range(offset + len(real_handles), offset + expected_slots):
                assert isinstance(handles[i], Line2D)
                assert handles[i].get_alpha() == 0
                assert labels[i] == ""

        # Group 1 (usage) fills the first block; group 2 (memory) starts a fresh column block
        # at index `slots` — the column-major invariant.
        _assert_group(0, usage_handles, usage_labels)
        _assert_group(expected_slots, memory_handles, memory_labels)
