"""
Aetherscan: Breakthrough Listen's first end-to-end production-grade DL pipeline for SETI @ scale

This package provides tools for technosignature detection using signal processing and deep learning
techniques applied to radio astronomy data.

For most use cases, import specific modules as needed:
    from aetherscan.config import Config
    from aetherscan.models import RandomForestModel, create_beta_vae_model
    from aetherscan.data_generation import DataGenerator
    from aetherscan.preprocessing import DataPreprocessor
    etc.
"""

__version__ = "0.1.0"
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
