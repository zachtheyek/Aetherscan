# NOTE: come back to this later

"""Unit tests for aetherscan.manager: pool / shared-memory tracking and cleanup idempotence.

ResourceManager registers atexit + signal handlers on construction; the conftest autouse
fixture snapshots/restores signal handlers and unregisters the atexit hook after each test,
so per-test instances don't accumulate process-global state.
"""

from __future__ import annotations

import atexit
import sys
from multiprocessing.shared_memory import SharedMemory

import pytest

from aetherscan.manager import get_manager, init_manager
from aetherscan.manager.manager import ResourceManager


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
