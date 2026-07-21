# NOTE: come back to this later

"""Unit tests for aetherscan.manager: pool / shared-memory tracking and cleanup idempotence.

ResourceManager registers atexit + signal handlers on construction; the conftest autouse
fixture snapshots/restores signal handlers and unregisters the atexit hook after each test,
so per-test instances don't accumulate process-global state.
"""

from __future__ import annotations

import atexit
import multiprocessing
import signal
import sys
import time
from multiprocessing.shared_memory import SharedMemory

import pytest

from aetherscan.manager import get_manager, init_manager
from aetherscan.manager.manager import ManagedProcess, ResourceManager


@pytest.fixture
def manager():
    return init_manager()


class TestSingletonSemantics:
    def test_init_returns_singleton(self, manager):
        assert ResourceManager() is manager
        assert get_manager() is manager

    def test_reset_produces_fresh_instance(self, manager):
        ResourceManager._reset()
        try:
            assert init_manager() is not manager
        finally:
            # The conftest teardown only unregisters the *current* instance's atexit hook;
            # drop the first instance's hook here so it can't fire at interpreter exit.
            atexit.unregister(manager.cleanup_all)


class TestPoolTracking:
    def test_create_pool_tracks_and_works(self, manager):
        pool = manager.create_pool(n_processes=2, name="test-pool")
        assert manager.stats.pools_active == 1
        # abs is importable-by-reference in spawned workers — keeps the test light.
        assert pool.map(abs, [-2, -3, 4]) == [2, 3, 4]

    def test_close_pool_updates_stats(self, manager):
        pool = manager.create_pool(n_processes=2, name="test-pool")
        manager.close_pool(pool)
        assert manager.stats.pools_active == 0
        assert manager.stats.pools_closed == 1
        assert manager._pools == []

    def test_close_pool_twice_is_safe(self, manager):
        pool = manager.create_pool(n_processes=2, name="test-pool")
        manager.close_pool(pool)
        manager.close_pool(pool)  # logs a warning; must not raise or double-count
        assert manager.stats.pools_closed == 1

    def test_managed_pool_close_is_idempotent(self, manager):
        manager.create_pool(n_processes=2, name="test-pool")
        managed = manager._pools[0]
        managed.close(timeout=10.0)
        assert managed.closed is True
        managed.close(timeout=10.0)  # second call short-circuits on the closed flag
        assert managed.closed is True

    def test_multiple_pools_tracked_independently(self, manager):
        pool_a = manager.create_pool(n_processes=1, name="a")
        pool_b = manager.create_pool(n_processes=1, name="b")
        assert manager.stats.pools_active == 2
        manager.close_pool(pool_a)
        assert manager.stats.pools_active == 1
        assert manager._pools[0].pool is pool_b
        manager.close_pool(pool_b)
        assert manager.stats.pools_active == 0


class TestSharedMemoryTracking:
    def test_create_tracks_and_is_usable(self, manager):
        shm = manager.create_shared_memory(size=1024, name="test-shm")
        assert manager.stats.shared_memories_active == 1
        shm.buf[:4] = b"seti"
        # Another handle attached by name sees the same bytes (i.e. it's real POSIX shm).
        other = SharedMemory(name=shm.name)
        try:
            assert bytes(other.buf[:4]) == b"seti"
        finally:
            other.close()

    def test_close_unlinks_and_updates_stats(self, manager):
        shm = manager.create_shared_memory(size=1024, name="test-shm")
        name = shm.name
        manager.close_shared_memory(shm)
        assert manager.stats.shared_memories_active == 0
        assert manager.stats.shared_memories_cleaned == 1
        # The segment must be gone from the system namespace.
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)

    def test_close_twice_is_safe(self, manager):
        shm = manager.create_shared_memory(size=1024, name="test-shm")
        manager.close_shared_memory(shm)
        manager.close_shared_memory(shm)  # logs a warning; must not raise or double-count
        assert manager.stats.shared_memories_cleaned == 1


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="cleanup_all reports process-tree memory via PSS, which psutil only exposes on Linux",
)
class TestCleanupAll:
    def test_cleanup_all_closes_everything(self, manager):
        manager.create_pool(n_processes=1, name="pool")
        shm = manager.create_shared_memory(size=512, name="shm")
        name = shm.name

        manager.cleanup_all()

        assert manager.stats.pools_active == 0
        assert manager.stats.pools_closed == 1
        assert manager.stats.shared_memories_active == 0
        assert manager.stats.shared_memories_cleaned == 1
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)

    def test_cleanup_all_is_idempotent(self, manager):
        manager.create_pool(n_processes=1, name="pool")
        manager.cleanup_all()
        closed = manager.stats.pools_closed
        manager.cleanup_all()  # second call must be a guarded no-op
        assert manager.stats.pools_closed == closed

    def test_cleanup_time_recorded(self, manager):
        manager.create_shared_memory(size=512, name="shm")
        manager.cleanup_all()
        assert manager.stats.cleanup_time_seconds >= 0.0


def _sleep_target(ready_event):
    """Process target that just idles (module-level so spawn-based platforms can pickle it)."""
    ready_event.set()
    time.sleep(60)


def _sigterm_ignoring_target(ready_event):
    """Process target that ignores SIGTERM, forcing the kill escalation path."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready_event.set()
    time.sleep(60)


def _start_process(target):
    ready = multiprocessing.Event()
    process = multiprocessing.Process(target=target, args=(ready,))
    process.start()
    assert ready.wait(timeout=30)  # don't race the target's signal-handler setup
    return process


class _FakeSubtreeChild:
    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True


def _fake_psutil_process_cls(children):
    """psutil.Process stand-in whose children(recursive=True) returns the given fakes."""

    class _FakePsutilProcess:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive=False):
            return children

    return _FakePsutilProcess


class TestProcessTracking:
    def test_register_and_close_process(self, manager):
        process = _start_process(_sleep_target)
        manager.register_process(process, name="test-process")
        assert manager.stats.processes_active == 1

        manager.close_process(process)
        assert not process.is_alive()
        assert manager.stats.processes_active == 0
        assert manager.stats.processes_closed == 1
        assert manager._processes == []

    def test_close_process_twice_is_safe(self, manager):
        process = _start_process(_sleep_target)
        manager.register_process(process, name="test-process")
        manager.close_process(process)
        manager.close_process(process)  # logs a warning; must not raise or double-count
        assert manager.stats.processes_closed == 1

    def test_close_escalates_to_kill_when_sigterm_ignored(self, manager):
        process = _start_process(_sigterm_ignoring_target)
        manager.register_process(process, name="stubborn-process")
        managed = manager._processes[0]
        managed.close(timeout=1.0)  # SIGTERM is ignored -> join times out -> SIGKILL
        assert managed.closed is True
        process.join(timeout=10)
        assert not process.is_alive()

    def test_managed_process_close_on_dead_process_is_clean(self, manager):
        process = _start_process(_sleep_target)
        manager.register_process(process, name="test-process")
        process.terminate()
        process.join(timeout=10)
        # Closing an already-dead process must succeed without escalation
        manager.close_process(process)
        assert manager.stats.processes_closed == 1

    def test_close_on_dead_process_reaps_recorded_children(self, manager, monkeypatch):
        # A process that died before its own SIGTERM handler could reap its pool must still
        # get a subtree sweep on close() — not a bare closed=True (issue #141).
        process = _start_process(_sleep_target)
        manager.register_process(process, name="test-process")
        process.terminate()
        process.join(timeout=10)

        children = [_FakeSubtreeChild(), _FakeSubtreeChild()]
        monkeypatch.setattr(
            "aetherscan.manager.manager.psutil.Process", _fake_psutil_process_cls(children)
        )
        manager._processes[0].close(timeout=1.0)
        assert all(child.killed for child in children)

    def test_close_exception_fallback_reaps_subtree_before_kill(self, monkeypatch):
        # If terminate() raises, the fallback must sweep the subtree before SIGKILLing the
        # process — a bare kill() would orphan its children (issue #141).
        kill_calls = []

        class _ExplodingProcess:
            pid = 424242

            def is_alive(self):
                return True

            def terminate(self):
                raise RuntimeError("terminate exploded")

            def kill(self):
                kill_calls.append("kill")

        children = [_FakeSubtreeChild()]
        monkeypatch.setattr(
            "aetherscan.manager.manager.psutil.Process", _fake_psutil_process_cls(children)
        )
        managed = ManagedProcess(
            process=_ExplodingProcess(), name="exploding", created_at=time.time()
        )
        managed.close(timeout=0.1)
        assert managed.closed is True
        assert children[0].killed
        assert kill_calls == ["kill"]
