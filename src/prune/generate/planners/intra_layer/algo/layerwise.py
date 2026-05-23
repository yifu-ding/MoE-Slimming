from collections import defaultdict
import torch
import math
import numpy as np
from src.base.shared_utils import _print

def build_masks_layerwise(scores,   # scores[layer][expert] = 1D torch.Tensor[D] per-expert scores; tensor or dict
                             keep_ratio=None, 
                             top_k=None, 
                             L=None, 
                             E=None, 
                             I=None,
                             verbose=False,
                             ) -> tuple[torch.Tensor, torch.Tensor]:

    masks = torch.zeros((L, E, I), dtype=torch.float32).to(scores.device)
    K_E = torch.zeros(L, E, dtype=torch.int64)

    if isinstance(keep_ratio, (float, int)):
        keep_ratio = torch.tensor([keep_ratio for _ in range(L)]).clamp(max=1.0)

    assert len(keep_ratio) == L, "keep_ratio must be a list or tensor with length L"

    if verbose:
        _print(f"Building masks layerwise keep_ratio={keep_ratio}")
        
    for lid in range(L):
        sizes = torch.tensor([I for _ in range(E)], dtype=torch.long)
        D = int(sizes.sum().item())
        if top_k is not None:
            k = min(top_k, D)
        else:
            k = max(1, int(math.ceil(D * keep_ratio[lid])))
        flat = torch.cat([scores[lid][e].reshape(-1) for e in range(E)], dim=0)

        top_idx = torch.topk(flat, k=k, largest=True).indices

        device = flat.device
        cum = torch.cat([torch.tensor([0], dtype=torch.long), torch.cumsum(sizes, dim=0)]).to(device)
        owner = torch.bucketize(top_idx, cum[1:], right=True).to(device)
        local = top_idx - cum[:-1][owner]

        for e in range(E):
            masks[lid, e] = torch.zeros(I, dtype=torch.float32).to(device)

        for e in range(E):
            selected = local[owner == e]
            if selected.numel() > 0:
                masks[lid, e].index_fill_(0, selected, 1.0)
                K_E[lid, e] = selected.numel()
    
    return masks, K_E
