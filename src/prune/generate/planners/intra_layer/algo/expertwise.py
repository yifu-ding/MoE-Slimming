import torch
import math
from src.base.shared_utils import _print

def build_masks_expertwise(scores,   # scores[layer][expert] = 1D torch.Tensor[D] per-expert scores; tensor or dict
                              keep_ratio=None,   # tensor with shape (L, E) or float
                              top_k=None, 
                              L=None, 
                              E=None, 
                              I=None,
                              verbose=False,
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(scores, torch.Tensor):
        scores = scores.to(dtype=torch.float32)
        device = scores.device
    else:
        device = None

    masks = torch.zeros((L, E, I), dtype=torch.float32)
    if device is not None:
        masks = masks.to(device)

    K_E = torch.zeros(L, E, dtype=torch.int64)

    if isinstance(keep_ratio, (float, int)):  
        keep_ratio = torch.ones(L, dtype=torch.float32) * keep_ratio
        keep_ratio = keep_ratio.unsqueeze(1).repeat(1, E)  # [L, E]
    elif isinstance(keep_ratio, (list, tuple)):
        keep_ratio = torch.tensor(keep_ratio, dtype=torch.float32)  # [L]
        keep_ratio = keep_ratio.unsqueeze(1).repeat(1, E)  # [L, E]
    elif keep_ratio.ndim == 1 and keep_ratio.shape[0] == E:
        keep_ratio = keep_ratio.unsqueeze(0).repeat(L, 1)  # [L, E]
    elif keep_ratio.ndim == 1 and keep_ratio.shape[0] == L:
        keep_ratio = keep_ratio.unsqueeze(1).repeat(1, E)  # [L, E]
    
    assert keep_ratio.shape == (L, E), "keep_ratio must be a tensor with shape (L, E)"
    keep_ratio = keep_ratio.clamp(min=0.0, max=1.0)
    
    if verbose:
        _print(f"[build_masks_expertwise] keep_ratio: {keep_ratio}")
    
    for lid in range(L):
        layer_scores = scores[lid]
        for eid in range(E):
            expert_scores = layer_scores[eid]
            if expert_scores.dtype != torch.float32:
                expert_scores = expert_scores.float()
            if device is not None and expert_scores.device != device:
                expert_scores = expert_scores.to(device)
            D = expert_scores.numel()
            if top_k is not None:
                k = min(top_k, D)
            else:
                k = max(1, int(math.ceil(D * keep_ratio[lid][eid])))
            idx = torch.topk(expert_scores, k=k, largest=True).indices
            masks[lid, eid] = torch.zeros(I, dtype=torch.float32).to(scores.device)
            masks[lid, eid].index_fill_(0, idx, 1.0)
            K_E[lid, eid] = k

    return masks, K_E
