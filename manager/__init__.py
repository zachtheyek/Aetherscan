"""
Resource manager package for Aetherscan pipeline
"""

from .manager import (
    get_manager,
    init_manager,
    register_db,
    register_logger,
    register_monitor,
    shutdown_manager,
)

__all__ = [
    "init_manager",
    "get_manager",
    "register_db",
    "register_logger",
    "register_monitor",
    "shutdown_manager",
]
