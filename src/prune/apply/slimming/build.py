"""
Main entry point for building slim models.

This module provides a unified interface for pruning expert modules.
It orchestrates the pruning process across different components of the model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Union

import torch
import torch.nn as nn

from .expert_slim import slim_moe_inter

__all__ = [
    "build_real_slim_model",
]


@torch.no_grad()
def build_real_slim_model(
    model: nn.Module,
    mask: Union[torch.Tensor, List[List[torch.Tensor]], Dict[str, Any]],
    shrink_gate: bool = False,
    add_hooks: bool = True,
    verbose: bool = False,
) -> nn.Module:
 
    qcfg = getattr(getattr(model, "config", None), "quantization_config", None)

    if isinstance(mask, dict):
        inter_masks = mask["intermediate_masks"]
    else:
        inter_masks = mask
        
    slim_moe_inter(
        model=model,
        inter_masks=inter_masks,
        qcfg=qcfg,
        shrink_gate=shrink_gate,
        add_hooks=add_hooks,
        verbose=verbose,
    )
    return model
