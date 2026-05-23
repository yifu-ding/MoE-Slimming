"""
Slim model building utilities.

This module provides functions and classes for building slim models
by pruning attention heads and expert dimensions.
"""
from .build import build_real_slim_model
from .utils import skip_moe_mlp_forward

__all__ = [
    "build_real_slim_model",
    "skip_moe_mlp_forward",
]
