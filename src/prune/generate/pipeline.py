import torch
from time import time
import numpy as np
from typing import Dict, Any, Optional

from src.prune.generate.stages import (
    prepare_scores, 
    adjust_masks,
    init_mask_for_I,
)
from src.base.shared_utils import _print

__all__ = [
    "generate_masks", 
]

def generate_masks(
    scores_dir: str,
    mask_dir: Optional[str] = None,
    prune_kwargs: Dict[str, Any] = None,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, Any]:

    if mask_dir is not None:
        masks = torch.load(mask_dir, map_location=device)
        if verbose:
            _print(f"[Mask Loading] ✅ Loaded masks from {mask_dir}, skip mask generate pipeline. ")
        
        if isinstance(masks, dict):
            pass
        elif isinstance(masks, torch.Tensor):
            if masks.ndim == 2:
                masks = None
            else:
                _print(f"[Mask Loading] Intermediate masks shape: {masks.shape}")
                masks = {
                    "intermediate_masks": masks,
                }        
        return masks
    
    prune_ratio = prune_kwargs.get("prune_ratio", 0.0)
    mask_method_kwargs = prune_kwargs.get("mask_method_kwargs", {})
    adjust_masks_kwargs = prune_kwargs.get("adjust_masks_kwargs", {})
    
    intermediate_scores, expertwise_scores, L, E, I, \
        loss_based_importance_kwargs = prepare_scores(
        scores_dir=scores_dir,
        prune_ratio=prune_ratio,
        mask_method_kwargs=mask_method_kwargs,
        device=device,
        verbose=verbose,
    )
    
    if verbose:
        _print("[Step 1-2] ✅ Prepare scores and layerwise_keep_plan")
    
    result = {}
    layerwise_keep_plan = loss_based_importance_kwargs["layerwise_keep_plan"]
    result["layerwise_keep_plan"] = layerwise_keep_plan
    
    start_time = time()

    mask_result = init_mask_for_I(
        intermediate_scores=intermediate_scores,
        expertwise_scores=expertwise_scores,
        layerwise_keep_plan=layerwise_keep_plan,
        intra_layer_method=mask_method_kwargs["intra_layer_method"],
        L=L,
        E=E,
        I=I,
        verbose=verbose,
    )
    result.update(mask_result)
    result["layerwise_inter_prune_ratio"] = None
    
    if verbose:
        _print(f"[Prune Inter] ✅ Building masks for I")

    align_inter = adjust_masks_kwargs.get("align_inter", 0)
    min_per_expert = adjust_masks_kwargs.get("min_per_expert", 0)

    if align_inter > 0 or min_per_expert > 0:
        K_E_inter = result["K_E_inter"]
        intermediate_masks = result["intermediate_masks"]
        
        if K_E_inter is not None:
            intermediate_masks, K_E_inter = adjust_masks(
                scores=intermediate_scores,
                masks=intermediate_masks,
                K_E=K_E_inter,
                L=L,
                E=E,
                I=I,
                align=align_inter,
                min_per_expert=min_per_expert,
                verbose=verbose,
            )
        
    intra_end_time = time()
    if verbose:
        _print(f"Intra-layer mask building time (including adjust): {((intra_end_time - start_time) * 1000):.2f} ms")
  
    return result
