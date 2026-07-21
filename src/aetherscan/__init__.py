"""
Aetherscan: Breakthrough Listen's first end-to-end production-grade DL pipeline for SETI @ scale.

Provides tools for technosignature detection using signal processing and deep learning techniques
applied to radio astronomy data. Import submodules directly (aetherscan.config, aetherscan.models,
aetherscan.data_generation, aetherscan.preprocessing, etc.) — the top-level package intentionally
exposes only the version string.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# Version is single-sourced from pyproject.toml's [project].version via the installed
# distribution's metadata. Source-tree runs (PYTHONPATH=src, the NGC container) have no
# installed distribution — they fall back to a dev sentinel, which also keeps the
# version-coupled HF weight resolution (hf_hub.version_default_revision) inactive there.
try:
    __version__ = version("aetherscan")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
__author__ = "Zach Yek"

# TODO: determine which components are necessary to expose for public API
# # Core configuration
# from aetherscan.config import Config
#
# # Models
# from aetherscan.models import RandomForestModel, Sampling, create_beta_vae_model
#
# # Data processing
# from aetherscan.data_generation import DataGenerator
# from aetherscan.preprocessing import DataPreprocessor
#
# __all__ = [
#   "Config",
#   "RandomForestModel",
#   "Sampling",
#   "create_beta_vae_model",
#   "DataGenerator",
#   "DataPreprocessor",
# ]

# Minimal public API - import submodules explicitly as needed
__all__ = []
