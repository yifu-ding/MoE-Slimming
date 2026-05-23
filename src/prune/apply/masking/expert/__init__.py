"""
Expert masking utilities for MoE models.
"""
from .fake_prune_wrapper import wrap_moe_experts_with_fake_prune_mask as mask_expert
from .apply_zero_to_weight import apply_zero_to_weight as zero_experts
from .forward_utils import forward_with_mask, forward_with_alpha, ori_moe_mlp_forward, _patch_block_alpha_if_needed
from .replace_pruned_layers import replace_pruned_layers

__all__ = [
    "mask_expert",
    "zero_experts",
    "forward_with_mask",
    "forward_with_alpha",
    "ori_moe_mlp_forward",
    "_patch_block_alpha_if_needed",
    "replace_pruned_layers",
]
