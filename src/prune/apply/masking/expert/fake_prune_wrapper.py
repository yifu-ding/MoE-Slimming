"""
Fake prune dynamic masked expert wrapper.

Main entry point for applying gradual masks to MoE expert modules.
This module provides a high-level interface that orchestrates the masking
process across all layers and experts of a model.

NOTE: This is FAKE PRUNING - only masking, not actually removing parameters.
"""
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from src.base.shared_utils.safe_isinstance import (
    _get_moe_block, 
    _get_experts, 
    _get_moe_intermediate_size,
    _get_num_hidden_layers
)

__all__ = [
    'wrap_moe_experts_with_fake_prune_mask'
]

class FakePruneMaskedMoEExpert(nn.Module):
    """
    Wrap an MoE expert module and apply gradual masks on hidden and intermediate dimensions.

    This wrapper applies gradual masks to:
    - Hidden dimension: masks input and output of the expert
    - Intermediate dimension: masks the intermediate activation

    The mask strength is controlled by a gradual schedule, allowing the model
    to gradually adapt to the pruning during training.

    Attributes:
        expert: The wrapped expert module
        inter_dim: Intermediate dimension size
        act: Activation function
    """
    
    def __init__(
        self,
        expert: nn.Module,
        inter_dim: int,
        intermediate_mask: torch.Tensor,
        act: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.expert = expert
        
        try:
            device = next(expert.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        
        self.intermediate_mask = intermediate_mask if intermediate_mask is not None else torch.ones(inter_dim, dtype=torch.float32, device=device)
        
        if act is not None:
            self.act = act
        else:
            self.act = getattr(expert, "act_fn", nn.SiLU())

    def _get_masks(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        intermediate_mask = self.intermediate_mask.to(dtype=x.dtype, device=x.device)
        return intermediate_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gradual masking.

        Args:
            x: Input tensor of shape [*, hidden_dim]

        Returns:
            Output tensor of shape [*, hidden_dim]
        """
        original_dtype = x.dtype
        
        mi = self._get_masks(x)
        out = self.expert.down_proj(x)
        out = out * mi
        
        if out.dtype != original_dtype:
            out = out.to(original_dtype)
        
        return out
    
    def __getattr__(self, name: str):
        """
        Delegate attribute access to the wrapped expert module.

        This allows the wrapper to transparently forward attributes
        from the wrapped module (e.g., gate_proj, up_proj, down_proj).
        """
        if name in {
            "expert", "intermediate_mask",
            "act", "_cache_step", "_cache_inter_mask",
            "_parameters", "_buffers", "_modules",
            "_non_persistent_buffers_set", "_backward_hooks",
            "_is_full_backward_hook", "_forward_hooks", "_forward_pre_hooks",
            "_state_dict_hooks", "_load_state_dict_pre_hooks"
        }:
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.expert, name)



def wrap_moe_experts_with_fake_prune_mask(
    model: nn.Module,
    intermediate_masks: List[torch.Tensor],
) -> nn.Module:
   
    num_layers = _get_num_hidden_layers(model)
    layer_indices = range(num_layers)

    mask_idx = 0
    layer_to_mask_idx = {}
    for layer_idx in layer_indices:
        moe_block = _get_moe_block(model, layer_idx)
        experts = _get_experts(moe_block)
        if experts is not None:
            layer_to_mask_idx[layer_idx] = mask_idx
            mask_idx += 1

    for layer_idx in layer_indices:
        moe_block = _get_moe_block(model, layer_idx)
        experts = _get_experts(moe_block)
        if experts is None:
            continue
        
        if layer_idx not in layer_to_mask_idx:
            continue
        
        mask_layer_idx = layer_to_mask_idx[layer_idx]
          
        for expert_idx, expert in enumerate(experts):
            intermediate_mask = intermediate_masks[mask_layer_idx][expert_idx]
            
            wrapped = FakePruneMaskedMoEExpert(
                expert=expert,
                inter_dim=_get_moe_intermediate_size(model),
                intermediate_mask=intermediate_mask,
                act=None,
            )
            experts[expert_idx] = wrapped

    return model
