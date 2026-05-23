"""
Expert masking utilities.

This module provides utilities for applying gradual masks to MoE expert modules.
"""
from .expert import (
    mask_expert, 
    zero_experts, 
    forward_with_mask,
    forward_with_alpha,
    ori_moe_mlp_forward,
    _patch_block_alpha_if_needed,
)

__all__ = [
    "mask_expert",
    "zero_experts",
    "forward_with_mask",
    "forward_with_alpha",
    "ori_moe_mlp_forward",
    "_patch_block_alpha_if_needed",
]
