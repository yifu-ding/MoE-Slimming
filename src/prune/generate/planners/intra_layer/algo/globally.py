import torch
from src.base.shared_utils import _print

def build_masks_globally(
    scores,
    global_keep_ratio: float = 0.5,
    L=None, 
    E=None, 
    I=None,
    verbose=False,
) -> tuple[torch.Tensor, torch.Tensor]:
    layers = list(range(L))
    D_layer = {lid: E * I for lid in layers}                   # total channels per layer

    assert isinstance(global_keep_ratio, (float, int)), "global_keep_ratio must be a float or int"
  
    sum_dims = sum(D_layer.values())
    assert sum_dims > 0, "No channels found."
    
    wE = torch.zeros((L, E), dtype=torch.float32)
    for lid in range(L):
        for eid in range(E):
            wE[lid, eid] = sum(scores[lid][eid])
    wE = wE / wE.sum()  # [L, E] normalized per-layer proportion

    K_total = int(round(global_keep_ratio * sum_dims))   # total channels to keep
    K_E = torch.floor(wE * K_total).to(torch.int64)  # [L, E] integer counts

    if verbose:
        _print(f"[build_masks_globally] K_E: {K_E}")
        
    masks = torch.zeros((L, E, I), dtype=torch.float32).to(scores.device)

    for lid in len(scores):
        layer_scores = scores[lid]
        for eid in range(E):
            expert_scores = layer_scores[eid]
            top_idx = torch.topk(expert_scores, k=K_E[lid][eid], largest=True).indices  # [k]
            masks[lid, eid] = torch.zeros(I, dtype=torch.float32).to(scores.device)
            masks[lid, eid].index_fill_(0, top_idx, 1.0)

    return masks, K_E
