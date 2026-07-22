"""Unit tests for aetherscan.monitor.get_process_tree_stats(): the PID -> Process cache
that fixes issue #12's CPU undercount (a freshly created psutil.Process reads 0.0 on its
first cpu_percent(interval=0) call, and children() manufactures fresh objects every call).

Importing aetherscan.monitor pulls TensorFlow at collection (monitor.py imports tf for GPU
detection) — same as most unit modules. The tests themselves drive the pure function with
fake Process objects; no monitor instance, GPU, or cluster data involved.
"""

from __future__ import annotations

from types import SimpleNamespace

import psutil
import pytest

from aetherscan.monitor import get_process_tree_stats


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
