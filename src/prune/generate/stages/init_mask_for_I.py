import torch
from typing import Optional, Tuple

from src.prune.generate.planners import intra_layer_planner
from src.base.shared_utils import _print

def init_mask_for_I(
    intermediate_scores: torch.Tensor,  # [L, E, I]
    expertwise_scores: Optional[torch.Tensor] = None,  # [L, E] or None
    layerwise_keep_plan: Optional[torch.Tensor] = None,  # [L] or None
    intra_layer_method: str = "uniform",
    L: int = None,
    E: int = None,
    I: int = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if expertwise_scores is not None:
        weighted_scores = torch.zeros_like(intermediate_scores)
        for layer_idx in range(L):
            expert_weights = expertwise_scores[layer_idx]
            weighted_scores[layer_idx] = intermediate_scores[layer_idx] * expert_weights[:, None]
        scores_to_use = weighted_scores
    else:
        scores_to_use = intermediate_scores
    
    intermediate_masks_float, K_E = intra_layer_planner(
        scores=scores_to_use,
        expertwise_scores=expertwise_scores,
        keep_ratio=layerwise_keep_plan,
        method=intra_layer_method,
        L=L,
        E=E,
        I=I,
        verbose=verbose,
    )
    
    intermediate_masks = intermediate_masks_float.bool()

    if verbose:
        inter_keep_ratio_actual = float(intermediate_masks.sum().item()) / float(L * E * I)
        _print(
            f"[I ratio planning] method: {intra_layer_method}, keep_ratio: {inter_keep_ratio_actual:.4f}, "
        )
    
    return {
        "intermediate_masks": intermediate_masks,
        "K_E_inter": K_E,
    }
