from typing import Tuple
import torch
import numpy as np
from .utils import _largest_remainder_alloc


def adjust_with_caps_min(
    K_E: torch.Tensor,                 # [L, E] int64, initial number of channels per layer per expert
    mat: dict,                          # [L, E, I] float32, score per layer per expert per channel
    D_layer: dict,                     # {layer_idx: total channels cap per layer}
    cap_per_expert: int = 768,
    min_per_expert: int = min,
    layers: list[int] | None = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    assert K_E.dtype == torch.int64
    L, E = K_E.shape
    if layers is None:
        layers = sorted(D_layer.keys())
    assert len(layers) == L

    caps_e = torch.full((E,), int(cap_per_expert), dtype=torch.int64, device=K_E.device)

    target_total = int(K_E.sum().item())

    K = torch.clamp(K_E.clone(), max=caps_e)

    freed_total = 0
    for i in range(L):
        mask = (K[i] > 0) & (K[i] < min_per_expert)
        freed_total += int(K[i][mask].sum().item())
        K[i][mask] = 0

    layer_caps_vec = torch.tensor([int(D_layer[l]) for l in layers], dtype=torch.int64, device=K.device)
    layer_used_vec = K.sum(dim=1)
    layer_headroom = torch.clamp(layer_caps_vec - layer_used_vec, min=torch.zeros_like(layer_caps_vec))
    
    total_headroom = int(layer_headroom.sum().item())
    if total_headroom == 0:
        I = cap_per_expert
        masks = torch.zeros((L, E, I), dtype=torch.bool, device=K.device)
        for lid in range(L):
            for eid in range(E):
                score_vec = mat[lid][eid].to(device=K.device)
                topk_idx = torch.topk(score_vec, k=K[lid][eid], largest=True).indices
                masks[lid][eid].index_fill_(0, topk_idx, True)
        
        info = {
            "target_total": int(target_total),
            "curr_total": int(K.sum().item()),
            "diff": 0,
        }
        return K, masks, info

    L_add = int(freed_total)
    layer_add = _largest_remainder_alloc(
        ideals=(layer_headroom.double() * (L_add / max(total_headroom, 1))).to(torch.float64),
        caps=layer_headroom,
        target=L_add,
    )

    for i in range(L):
        add_i = int(layer_add[i].item())
        if add_i <= 0:
            continue
        row = K[i]
        active = (row >= min_per_expert)
        if active.any():
            headroom_e = torch.clamp(caps_e - row, min=0)
            headroom_active = torch.where(active, headroom_e, torch.zeros_like(headroom_e))
            total_hr = int(headroom_active.sum().item())
            if total_hr > 0:
                give = _largest_remainder_alloc(
                    ideals=(headroom_active.double() * (add_i / total_hr)).to(torch.float64),
                    caps=headroom_active,
                    target=add_i,
                )
                row = row + give
                add_i -= int(give.sum().item())

        if add_i >= min_per_expert:
            zeros = (row == 0)
            if zeros.any():
                headroom_e = torch.clamp(caps_e - row, min=0)
                max_new = min(
                    add_i // min_per_expert,
                    int((headroom_e[zeros] >= min_per_expert).sum().item())
                )
                if max_new > 0:
                    idx_zero = torch.where(zeros & (headroom_e >= min_per_expert))[0][:max_new]
                    row[idx_zero] = min_per_expert
                    add_i -= max_new * min_per_expert

        if add_i > 0:
            active = (row >= min_per_expert)
            if active.any():
                headroom_e = torch.clamp(caps_e - row, min=0)
                headroom_active = torch.where(active, headroom_e, torch.zeros_like(headroom_e))
                total_hr = int(headroom_active.sum().item())
                if total_hr > 0:
                    give = _largest_remainder_alloc(
                        ideals=(headroom_active.double() * (add_i / total_hr)).to(torch.float64),
                        caps=headroom_active,
                        target=add_i,
                    )
                    row = row + give
                    add_i -= int(give.sum().item())

        K[i] = row
        s = int(K[i].sum().item()); cap = int(D_layer[layers[i]])
        if s > cap:
            surplus = s - cap
            reducible = K[i].clone()
            real_cut = reducible.double() * (surplus / max(int(reducible.sum().item()), 1))
            base_cut = torch.floor(real_cut).to(torch.int64)
            base_cut = torch.minimum(base_cut, reducible)
            K[i] = K[i] - base_cut
            rem = surplus - int(base_cut.sum().item())
            if rem > 0:
                frac = (real_cut - base_cut.double()).cpu().numpy()
                order = np.argsort(-frac)
                j = 0
                while rem > 0 and j < E:
                    idx = int(order[j])
                    if K[i, idx] > 0:
                        K[i, idx] -= 1
                        rem -= 1
                    j += 1

    K = torch.clamp(K, max=caps_e)
    K[(K > 0) & (K < min_per_expert)] = 0

    curr_total = int(K.sum().item())
    if curr_total != target_total:
        diff = target_total - curr_total
        if diff > 0:
            layer_headroom = torch.clamp(
                torch.tensor([int(D_layer[l]) for l in layers], dtype=torch.int64, device=K.device) - K.sum(1),
                min=torch.zeros_like(K.sum(1))
            )
            total_headroom = int(layer_headroom.sum().item())
            if total_headroom > 0:
                layer_add = _largest_remainder_alloc(
                    (layer_headroom.double() * (diff / total_headroom)).to(torch.float64),
                    caps=layer_headroom, target=diff
                )
                for i in range(L):
                    ai = int(layer_add[i].item())
                    if ai <= 0: continue
                    row = K[i]
                    active = (row >= min_per_expert)
                    if active.any():
                        headroom_e = torch.clamp(caps_e - row, min=torch.zeros_like(row))
                        give = _largest_remainder_alloc(
                            (torch.where(active, headroom_e, torch.zeros_like(headroom_e)).double()
                             * (ai / max(int(headroom_e[active].sum().item()), 1))).to(torch.float64),
                            caps=torch.where(active, headroom_e, torch.zeros_like(headroom_e)),
                            target=ai
                        )
                        K[i] = row + give
        else:
            surplus = -diff
            flat_idx = torch.nonzero(K >= min_per_expert, as_tuple=False)
            if flat_idx.numel() > 0:
                order = torch.argsort(K.flatten(), descending=True)
                j = 0
                while surplus > 0 and j < order.numel():
                    r = order[j] // E
                    c = order[j] % E
                    if K.view(-1)[order[j]] > min_per_expert:
                        K[r, c] -= 1
                        surplus -= 1
                    j += 1

    I = cap_per_expert
    masks = torch.zeros((L, E, I), dtype=torch.bool, device=K.device)
    for lid in range(L):
        for eid in range(E):
            score_vec = mat[lid][eid].to(device=K.device)
            topk_idx = torch.topk(score_vec, k=K[lid][eid], largest=True).indices
            masks[lid][eid].index_fill_(0, topk_idx, True)
    
    info = {
        "method": "evict_min",
        "align": "none",
        "min_per_expert": int(min_per_expert),
        "target_total": int(target_total),
        "curr_total": int(K.sum().item()),
        "diff": int(K.sum().item() - target_total),
    }
    
    return K, masks, info
