"""
Resource manager for Aetherscan Pipeline
Centralizes orchestration of all system resources -- including multiprocessing pools, shared memory,
and background threads (e.g. database, monitor, logger)
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory

import psutil

from config import get_config
from logger import get_logger
from monitor import get_process_tree_stats

logger = logging.getLogger(__name__)


@dataclass
class ResourceStats:
    """Statistics about managed resources"""

    pools_active: int = 0
    pools_closed: int = 0
    shared_memories_active: int = 0
    shared_memories_cleaned: int = 0
    total_memory_freed_gb: float = 0.0
    cleanup_time_seconds: float = 0.0


@dataclass
class ManagedPool:
    """Wrapper for tracked multiprocessing Pool"""

    pool: Pool
    name: str
    created_at: float
    process_count: int
    closed: bool = False

    def close(self, timeout):
        """Close the pool with terminate-or-fail policy"""
        if self.closed:
            return

        try:
            logger.info(f"Forcefully terminating pool '{self.name}'")

            # NOTE: does terminating instead of closing lead to any issues with corrupted db writes?
            # Forcefully kill all running processes & clear internal job queues with .terminate()
            # Then, wait for parent to finish handling dead processes,
            # and exit & close the pool with .join()
            # Note, less forceful alternative is to use self.pool.close()
            # Which stops accepting new jobs, but lets running processes finish current job queue
            # Don't use both! Better to "terminate or fail". Inconsistent states are worse than leaks
            self.pool.terminate()

            # For .join(), we wrap the call in a daemon thread with timeout
            # so it doesn't block on exit or hang indefinitely
            join_thread = threading.Thread(target=self.pool.join, daemon=True)
            join_thread.start()
            join_thread.join(timeout=timeout)

            # If timeout exceeded, force-kill survivors
            if join_thread.is_alive():
                logger.warning(f"Pool '{self.name}' join timeout exceeded")
                logger.info("Force-killing remaining worker processes with SIGKILL")
                self._force_kill_workers()
                # NOTE: 0.2 seems arbitrary & can potentially add latency to shutdown. is there a more precise wall we can use?
                time.sleep(0.2)  # Brief wait for SIGKILL to take effect

            # Verify cleanup
            if self._check_alive():
                logger.error(f"Pool '{self.name}' workers still alive after SIGKILL!")
                # At this point, we proceed anyway instead of raising
                # The OS should clean up on process exit
                # NOTE: is it safe to just ignore workers that survive SIGKILL? will self.closed = True lead to issues down the line?

            self.closed = True
            logger.info(f"Pool '{self.name}' closed ({self.process_count} processes)")

        except Exception as e:
            logger.warning(f"Error terminating pool '{self.name}': {e}")
            # Still try to force-kill on error
            # If it fails, let the OS handle on process exit
            # NOTE: is it safe to just ignore workers that survive SIGKILL? will self.closed = True lead to issues down the line?
            # NOTE: should we add time.sleep(0.2) & _check_alive() after _force_kill_workers()? would logging an error on SIGKILL fail be useful?
            with contextlib.suppress(Exception):
                self._force_kill_workers()
            self.closed = True

    def _check_alive(self):
        """Check if any worker processes are still alive"""
        try:
            return any(p.is_alive() for p in getattr(self.pool, "_pool", []))
        except Exception as e:
            logger.info(f"_check_alive(): unable to inspect pool '{self.name}': {e}")
            logger.info(f"Assuming pool '{self.name}' is gone and no remaining workers alive")
            # NOTE: is there a more precise way to differentiate pool already destroyed vs internal state corrupted?
            return False  # pool already destroyed or internal state corrupted

    def _force_kill_workers(self):
        """Force-kill all worker processes with SIGKILL"""
        killed_any = False

        # Method 1: Via pool._pool attribute
        try:
            for worker in getattr(self.pool, "_pool", []):
                if worker.is_alive():
                    pid = worker.pid
                    try:
                        process = psutil.Process(pid)
                        process.kill()  # SIGKILL - cannot be ignored
                        logger.info(f"Force-killed worker PID {pid}")
                        killed_any = True
                    except psutil.NoSuchProcess:
                        pass  # Already dead
                    except Exception as e:
                        logger.warning(f"Failed to kill worker PID {pid}: {e}")
        except Exception as e:
            logger.warning(f"Error accessing pool._pool: {e}")

            # Method 2: Fallback - find & kill all child processes via psutil
            # Useful for when getattr(self.pool, "_pool", []) raises an Exception, of if _pool is
            # in an inconsistent state during cleanup
            # Note, method 2 kills ALL child processes, not just the workers belonging to a specific
            # Pool. This is acceptable in our current implementation, since pool closures either
            # happen sequentially (PreProc Pool is opened & closed before DataGen, etc.), or
            # indiscriminately during cleanup (atexit, SIGINT, SIGTERM, etc.).
            # However, in the case where we have multiple long-lived pools or workers, and SIGTERM
            # is escalated to SIGKILL, and pool._pool attribution fails, that innocent workers may
            # be killed by accident. If this becomes an issue in the future, simply comment out
            # method 2 and let the OS clean up on process exit
            # Note, could alternatively try tracking worker PIDs by Pool proactively, though this
            # requires architectural changes that aren't worth the time rn

            # logger.info("Attempting fallback: killing ALL child processes directly")
            # try:
            #     current = psutil.Process(os.getpid())
            #     children = current.children(recursive=True)
            #
            #     if not children:
            #         logger.info("No child processes found")
            #         return
            #
            #     for child in children:
            #         try:
            #             if child.is_running():
            #                 child.kill()
            #                 logger.info(f"Force-killed child process PID {child.pid}")
            #                 killed_any = True
            #         except psutil.NoSuchProcess:
            #             pass  # Already terminated
            #         except Exception as child_error:
            #             logger.warning(f"Failed to kill child PID {child.pid}: {child_error}")
            # except Exception as fallback_error:
            #     logger.error(f"Fallback force-kill failed: {fallback_error}")

        if not killed_any:
            logger.info("No workers force-killed")


@dataclass
class ManagedSharedMemory:
    """Wrapper for tracked shared memory"""

    shm: SharedMemory
    name: str  # NOTE: this should just be self.shm.name?
    created_at: float
    size_gb: float
    closed: bool = False

    def close(self):
        """Close and unlink shared memory"""
        if self.closed:
            return

        try:
            logger.info(f"Closing shared memory '{self.shm.name}'")
            # Detach current process's reference to shared memory object
            # Importantly, shm.close() doesn't remove the shared memory from the system
            # It just closes the file descriptor / memory mapping in the current process
            # Other processes with references to this shared memory can continue using it
            self.shm.close()
        except Exception as e:
            logger.warning(f"Error closing '{self.shm.name}': {e}")

        try:
            logger.info(f"Unlinking shared memory '{self.shm.name}'")
            # Remove shared memory object from system namespace
            # Once all processes close their handles, the OS reclaims the memory
            self.shm.unlink()
        except FileNotFoundError:
            pass  # Already unlinked by another process
        except Exception as e:
            logger.warning(f"Error unlinking '{self.shm.name}': {e}")

        # Verify after attempting both close() and unlink()
        if not self._check_unlinked():
            raise RuntimeError(f"Shared memory '{self.shm.name}' still exists after cleanup")

        self.closed = True
        logger.info(f"Shared memory '{self.shm.name}' cleaned ({self.size_gb:.2f} GB)")

    def _check_unlinked(self):
        """Check if shared memory can still be reattached to by name"""
        try:
            test_shm = SharedMemory(name=self.name)
            test_shm.close()
            return False  # Reattached successfully, shared memory object still exists
        except FileNotFoundError:
            return True  # Successfully unlinked
        except Exception as e:
            logger.info(f"_check_unlinked(): unable to inspect shared memory'{self.name}': {e}")
            logger.info(f"Assuming shared memory '{self.name}' didn't close properly")
            return False  # Assume unexpected failure = close() didn't run properly


class ResourceManager:
    """
    Resource manager for centralized tracking & cleanup
    Handles multiprocessing pools, shared memory, and background threads
    """

    _instance = None  # Stores singleton instance
    _lock = threading.Lock()  # Ensures thread safety on object initialization

    # __new__ allocates the object in memory (constructor at the object-creation level)
    # __init__ initializes the object's attributes after it's created
    # since __new__ is called before __init__ every time we instantiate a class,
    # by overriding __new__, we can short-circuit object creation entirely, and control whether a
    # new instance is created, or just return the existing instance
    def __new__(cls):
        # Double-checked locking pattern:
        # First check if _instance is None, without lock (for performance)
        if cls._instance is None:
            # If None, acquire the lock to serialize the initialization path,
            # preventing race conditions (2 threads violating singleton semantics)
            with cls._lock:
                # Check if _instance is None again inside the lock
                # (since multiple threads can be calling simultaneously)
                if cls._instance is None:
                    # If still None, only then we construct the singleton instance
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False  # Mark as not initialized (for __init__)
        # Return the same instance for all subsequent constructor calls
        return cls._instance

    def __init__(self):
        """Initialize manager"""
        # Note, __init__ is triggered every time the class's constructor is called,
        # even if __new__ returned the existing singleton instance
        # Hence, we use the _initialized flag to make sure __init__ only runs once
        if self._initialized:
            return

        self._initialized = True
        self.config = get_config()
        if self.config is None:
            raise ValueError("get_config() returned None")

        self._main_process_pid = os.getpid()
        self._cleanup_executed = False
        self._cleanup_lock = threading.Lock()

        # NOTE: should these be strong or weak references? import weakref ...
        # Track resources
        self._pools: list[ManagedPool] = []
        self._shared_memories: list[ManagedSharedMemory] = []

        # Track threads
        self._db = None
        self._logger = None
        self._monitor = None

        # Track statistics
        self.stats = ResourceStats()

        # Register cleanup handlers
        self._register_cleanup_handlers()

        logger.info(f"ResourceManager initialized (PID: {self._main_process_pid})")
        logger.info(f"  Pools active: {self.stats.pools_active}")
        logger.info(f"  Shared memories active: {self.stats.shared_memories_active}")

    @classmethod
    def _reset(cls):
        """
        Teardown hook for thread-safe singleton
        Resets the manager instance to None

        WARNING: Only use for testing or restarting the application
        Calling this while the manager is active will cause issues.
        Do NOT call this method unless you know what you're doing
        """
        # Acquire lock to prevent race conditions
        with cls._lock:
            # Discard the singleton instance by removing the global reference
            # Guarantees the next constructor call will produce a fresh instance
            # Note, resources held by the old instance will remain alive unless explicitly closed beforehand
            cls._instance = None
            logger.info("Manager singleton instance reset")
            logger.info(
                "Note that resource cleanup has not been triggered, and will still run as expected on exit."
            )

    def _register_cleanup_handlers(self):
        """Register atexit and signal handlers for cleanup"""
        # Register cleanup handler to fire on system exit
        atexit.register(self.cleanup_all)

        # Register signal handlers to fire on interruptions and terminations
        signal.signal(signal.SIGINT, self._signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler)  # kill/docker stop

        logger.info("Cleanup handlers registered")

    def _signal_handler(self, signum, frame):
        """Handle SIGINT and SIGTERM gracefully"""
        # Ignore signals from worker processes entirely
        if os.getpid() != self._main_process_pid:
            return

        with contextlib.suppress(Exception):
            logger.info(f"Received signal {signum}, initiating cleanup...")

        self.cleanup_all()
        # Properly terminate with sys.exit() after handling the signal inside main process
        # Note, sys.exit() triggers the following cleanup handlers (in order):
        # 1. finally blocks in active try/except statements
        # 2. context managers (__exit__ methods)
        # 3. atexit registered functions
        # sys.exit(0): exit with successful termination status
        # sys.exit(1): exit with failed termination status
        sys.exit(0)

    def cleanup_all(self):
        """
        Unified cleanup of all resources.
        Strict order:
            1. Pools
            2. SharedMemory
            3. Monitor
            4. DB
            5. Logger
        """
        # Skip if not main process
        if os.getpid() != self._main_process_pid:
            logger.info(f"Skipping cleanup in worker process (PID: {os.getpid()})")
            return

        # Idempotent cleanup (skip if already cleaned)
        # Guarantees cleanup routine only runs once, even if multiple threads call it concurrently
        # First, acquire a mutual-exclusion lock. Only one thread can enter this block at a time
        # (prevents race conditions)
        with self._cleanup_lock:
            # Once inside the lock, the thread checks whether cleanup already occured
            if self._cleanup_executed:
                # If so, we return from the function immediately and do nothing
                return
            # Otherwise, the current thread marks cleanup as complete and executes the rest of the routine
            self._cleanup_executed = True

        logger.info("=" * 60)
        logger.info("Starting ResourceManager cleanup...")

        # Log initial stats
        start_time = time.time()
        initial_memory = self._get_memory_usage()

        # Close managed pools
        logger.info("Closing multiprocessing pools...")
        for managed in self._pools:
            if not managed.closed:
                self._close_managed_pool(managed)

        # Close managed shared memory
        logger.info("Closing shared memory...")
        for managed in self._shared_memories:
            if not managed.closed:
                self._close_managed_shared_memory(managed)

        # Shutdown monitor
        if self._monitor:
            logger.info("Shutting down monitor...")
            try:
                # Late import to avoid circular dependency (monitor imports from manager)
                from monitor import shutdown_monitor  # noqa: PLC0415

                shutdown_monitor()
            except Exception as e:
                with contextlib.suppress(Exception):
                    logger.error(f"Error during monitor shutdown: {e}")

        # Shutdown database
        if self._db:
            logger.info("Shutting down database...")
            try:
                # Late import to avoid circular dependency (db imports from manager)
                from db import shutdown_db  # noqa: PLC0415

                shutdown_db()
            except Exception as e:
                with contextlib.suppress(Exception):
                    logger.error(f"Error during database shutdown: {e}")

        # BUG: sometimes total_memory_freed_gb < 0
        # Log final stats
        final_memory = self._get_memory_usage()
        self.stats.total_memory_freed_gb = initial_memory - final_memory

        end_time = time.time()
        self.stats.cleanup_time_seconds = end_time - start_time

        logger.info("=" * 60)
        logger.info("ResourceManager cleanup complete:")
        logger.info(f"  Pools closed: {self.stats.pools_closed}")
        logger.info(f"  Shared memories cleaned: {self.stats.shared_memories_cleaned}")
        logger.info(f"  Memory freed: {self.stats.total_memory_freed_gb:.2f} GB")
        logger.info(f"  Cleanup time: {self.stats.cleanup_time_seconds:.2f} seconds")
        logger.info("=" * 60)

        # Shutdown logger
        if self._logger:
            with contextlib.suppress(Exception):
                # Late import to avoid circular dependency (logger imports from manager)
                from logger import shutdown_logger  # noqa: PLC0415

                shutdown_logger()
                # Note, we can't log after stopping the listener thread, so no final message here

    def _get_memory_usage(self) -> float:
        """Get process tree's current memory usage (in Gb)"""
        process = psutil.Process(self._main_process_pid)
        stats = get_process_tree_stats(process)
        return stats["ram_gb"]

    def create_pool(
        self,
        n_processes: int,
        name: str = "unnamed",
        initializer: Callable | None = None,
        initargs: tuple = (),
    ) -> Pool:
        """
        Create and register a managed pool

        Args:
            n_processes: Number of worker processes
            name: Descriptive name for debugging
            initializer: Worker initialization function
            initargs: Arguments for initializer

        Returns:
            Multiprocessing Pool instance
        """
        pool = Pool(processes=n_processes, initializer=initializer, initargs=initargs)

        managed = ManagedPool(
            pool=pool, name=name, created_at=time.time(), process_count=n_processes
        )

        self._pools.append(managed)
        self.stats.pools_active += 1

        logger.info(f"Created pool '{name}' with {n_processes} processes")
        logger.info(f"  Current total active: {self.stats.pools_active}")
        return pool

    def close_pool(self, pool: Pool):
        """Explicitly close a specific pool by reference"""
        for managed in self._pools:
            if managed.pool is pool and not managed.closed:
                self._close_managed_pool(managed)
                return
        logger.warning("ResourceManager.close_pool(): Pool not found in managed pools")

    def _close_managed_pool(self, managed: ManagedPool):
        """Internal method to close a ManagedPool"""
        managed.close(timeout=self.config.manager.pool_terminate_timeout)
        # Remove from tracking list to allow garbage collection of Pool object and its file descriptors
        self._pools.remove(managed)
        self.stats.pools_active -= 1
        self.stats.pools_closed += 1

    def create_shared_memory(self, size: int, name: str = "unnamed") -> SharedMemory:
        """
        Create and register a managed shared memory

        Args:
            size: Size in bytes
            name: Descriptive name for debugging

        Returns:
            SharedMemory instance
        """
        shm = SharedMemory(create=True, size=size)

        managed = ManagedSharedMemory(
            shm=shm, name=name, created_at=time.time(), size_gb=size / 1e9
        )

        self._shared_memories.append(managed)
        self.stats.shared_memories_active += 1

        logger.info(f"Created shared memory '{name}' ({managed.size_gb:.2f} GB)")
        logger.info(f"  Current total active: {self.stats.shared_memories_active}")
        return shm

    def close_shared_memory(self, shm: SharedMemory):
        """Explicitly close specific shared memory by reference"""
        for managed in self._shared_memories:
            if managed.shm is shm and not managed.closed:
                self._close_managed_shared_memory(managed)
                return
        logger.warning(
            "ResourceManager.close_shared_memory(): SharedMemory not found in managed memories"
        )

    def _close_managed_shared_memory(self, managed: ManagedSharedMemory):
        """Internal method to close a ManagedSharedMemory"""
        managed.close()
        # Remove from tracking list to allow garbage collection of SharedMemory object and its file descriptors
        self._shared_memories.remove(managed)
        self.stats.shared_memories_active -= 1
        self.stats.shared_memories_cleaned += 1

    def set_db(self, db):
        """Set database"""
        self._db = db

    def set_logger(self, logger):
        """Set logger"""
        self._logger = logger

    def set_monitor(self, monitor):
        """Set monitor"""
        self._monitor = monitor

    def get_stats(self) -> ResourceStats:
        """Get current resource statistics"""
        return self.stats


def init_manager() -> ResourceManager:
    """
    Initialize global manager instance (call once at startup)
    """
    manager = ResourceManager()
    return manager


# NOTE: suppress None return type for now to reduce pyright errors
# def get_manager() -> ResourceManager | None:
def get_manager() -> ResourceManager:
    """Get the global manager instance"""
    manager = ResourceManager._instance

    if manager is None:
        logger.warning("No manager instance initialized")

    return manager


def register_db(db):
    """Register database instance to resource manager"""
    manager = ResourceManager._instance

    if manager is None:
        logger.warning("Cannot register db, no manager instance initialized")
        return

    manager.set_db(db)
    logger.info("Registered database")


def register_logger():
    """Register logger instance to resource manager"""
    manager = ResourceManager._instance

    if manager is None:
        logger.warning("Cannot register logger, no manager instance initialized")
        return

    # Note, unlike other modules, due to dependency chains,
    # register_logger() is not baked into init_logger()
    # Hence, we need to get the logger instance separately with get_logger()
    logger_instance = get_logger()
    if logger_instance is None:
        logger.warning("Cannot register logger, no logger instance initialized")
        return

    manager.set_logger(logger_instance)
    logger.info("Registered logger")


def register_monitor(monitor):
    """Register monitor instance to resource manager"""
    manager = ResourceManager._instance

    if manager is None:
        logger.warning("Cannot register monitor, no manager instance initialized")
        return

    manager.set_monitor(monitor)
    logger.info("Registered monitor")
