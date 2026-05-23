import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from .utils import _largest_remainder_alloc
from time import time
from src.base.shared_utils import _print

@torch.no_grad()
def adjust_with_align(
    K_E: torch.Tensor,                      # [L, E] int64
    mat: dict,                   # [L, E, I] float32
    D_layer: Dict[int, int],               # {layer_idx: layer_cap}
    cap_per_expert: int = 768,
    align: int = 64,                       # take 64 or 128
    min_per_expert: int = 0,               # if need to start lower bound, suggest giving as a multiple of 64 or 128
    layers: Optional[List[int]] = None,
    prefer_keep_active: bool = False,       # when evicting, try not to drop active slots to 0
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    align each expert channel count to multiples of align, while keeping totals stable.
    satisfy:
      0 <= K[l,e] <= cap_per_expert
      sum_e K[l,e] <= D_layer[l]
      K[l,e] == 0 or K[l,e] >= min_per_expert (if enabled) 
      K[l,e] % align == 0
    return:
      K_aligned, info dictionary: {'target_total','final_total','added','removed','align','unmet_diff'}
    """
    start_time = time()
    
    assert K_E.dtype == torch.int64
    L, E = K_E.shape
    if layers is None:
        layers = sorted(D_layer.keys())
    assert len(layers) == L
    device = K_E.device

    caps_e = torch.full((E,), int(cap_per_expert), dtype=torch.int64, device=device)
    layer_caps_vec = torch.tensor([int(D_layer[l]) for l in layers], dtype=torch.int64, device=device)

    target_total = int(K_E.sum().item())

    if min_per_expert > 0:
        if min_per_expert % align != 0:
            min_per_expert = ((min_per_expert + align - 1) // align) * align

    K = torch.clamp(K_E.clone(), min=torch.zeros_like(caps_e), max=caps_e)

    base = (K // align) * align

    if min_per_expert > 0:
        mask_low = (base > 0) & (base < min_per_expert)
        base[mask_low] = 0

    for i in range(L):
        row = base[i]
        cap = int(layer_caps_vec[i].item())
        s = int(row.sum().item())
        if s <= cap:
            continue
        over = s - cap
        over_chunks = (over + align - 1) // align
        candidates = torch.nonzero(row > 0, as_tuple=False).flatten()
        if prefer_keep_active and min_per_expert > 0:
            key = row[candidates] - min_per_expert
        else:
            key = row[candidates]
        order = candidates[torch.argsort(key)]  # from small to large
        j = 0
        while over_chunks > 0 and j < order.numel():
            idx = int(order[j].item())
            if row[idx] >= align:
                new_val = row[idx] - align
                if min_per_expert > 0 and new_val > 0 and new_val < min_per_expert:
                    over_chunks -= int((row[idx] + align - 1) // align)
                    row[idx] = 0
                else:
                    row[idx] = new_val
                    over_chunks -= 1
            j += 1
        base[i] = row

    base_total = int(base.sum().item())
    base_time = time()

    diff = target_total - base_total
    def round_to_chunks(x: int, step: int) -> int:
        return int(np.round(x / step))

    need_chunks = diff // align
    if diff % align != 0:
        cand1 = need_chunks
        cand2 = need_chunks + 1 if diff > 0 else need_chunks - 1
        e1 = abs(diff - cand1 * align)
        e2 = abs(diff - cand2 * align)
        if e2 < e1:
            need_chunks = cand2

    added_chunks = 0
    removed_chunks = 0

    def layer_headroom_chunks(curr: torch.Tensor) -> torch.Tensor:
        used = curr.sum(dim=1)
        hr_val = torch.clamp(layer_caps_vec - used, min=torch.zeros_like(layer_caps_vec))
        return hr_val // align

    def slot_headroom_chunks(curr_row: torch.Tensor) -> torch.Tensor:
        hr_e = torch.clamp(caps_e - curr_row, min=torch.zeros_like(caps_e))
        return hr_e // align

    frac = (K % align).to(torch.int64)
    desire = frac
    
    compute_slot_headroom_time = time() 

    if need_chunks > 0:
        remain = need_chunks
        while remain > 0:
            layer_hr = layer_headroom_chunks(base)
            if int(layer_hr.sum().item()) == 0:
                break
            proportional = (layer_hr.double() * (remain / max(int(layer_hr.sum().item()), 1))).to(torch.float64)
            layer_add = _largest_remainder_alloc(proportional, caps=layer_hr, target=remain).to(torch.int64)

            for i in range(L):
                c_i = int(layer_add[i].item())
                if c_i <= 0:
                    continue

                row = base[i]
                row_hr_chunks = slot_headroom_chunks(row)

                active = row > 0
                can_new = torch.zeros_like(row, dtype=torch.bool)
                if min_per_expert > 0:
                    need_min_chunks = min_per_expert // align
                    can_new = (row == 0) & (K[i] >= min_per_expert) & (row_hr_chunks >= need_min_chunks)

                cand = torch.where(active | can_new, torch.ones_like(row, dtype=torch.bool), torch.zeros_like(row, dtype=torch.bool))
                if cand.any():
                    scores = torch.where(cand, desire[i], torch.full_like(row, -1))
                    order = torch.argsort(scores, descending=True)

                    j = 0
                    while c_i > 0 and j < order.numel():
                        idx = int(order[j].item())
                        if not cand[idx]:
                            j += 1
                            continue
                        if row_hr_chunks[idx] <= 0:
                            j += 1
                            continue

                        if row[idx] == 0 and min_per_expert > 0:
                            need_chunks_here = min(min_per_expert // align, int(row_hr_chunks[idx].item()))
                            give = min(need_chunks_here, c_i)
                            if give == 0:
                                j += 1
                                continue
                            row[idx] += give * align
                            c_i -= give
                            added_chunks += give
                        else:
                            row[idx] += align
                            c_i -= 1
                            added_chunks += 1
                        row_hr_chunks = slot_headroom_chunks(row)
                        j += 1

                if c_i > 0:
                    row_hr_chunks = slot_headroom_chunks(row)
                    if int(row_hr_chunks.sum().item()) > 0:
                        order = torch.argsort(row_hr_chunks, descending=True)
                        j = 0
                        while c_i > 0 and j < order.numel():
                            idx = int(order[j].item())
                            if row_hr_chunks[idx] <= 0:
                                j += 1
                                continue
                            row[idx] += align
                            c_i -= 1
                            added_chunks += 1
                            row_hr_chunks[idx] -= 1
                            j += 1

                base[i] = row
                remain = int(remain - layer_add[i].item() + c_i)

    if need_chunks < 0:
        need = -need_chunks
        remain = need
        while remain > 0:
            any_pos = torch.nonzero(base > 0, as_tuple=False)
            if any_pos.numel() == 0:
                break

            scores = []
            indices = []
            for i in range(L):
                row = base[i]
                cand = torch.nonzero(row > 0, as_tuple=False).flatten()
                for idx in cand.tolist():
                    v = int(row[idx])
                    penalty = 0
                    if min_per_expert > 0:
                        if v - align > 0 and v - align < min_per_expert:
                            penalty = 2
                    score = penalty * 10 + int(align - (desire[i, idx].item()))
                    scores.append(score)
                    indices.append((i, idx))
            if len(indices) == 0:
                break
            order = np.argsort(np.array(scores))

            j = 0
            while remain > 0 and j < len(order):
                i, idx = indices[order[j]]
                v = int(base[i, idx].item())
                if v <= 0:
                    j += 1
                    continue
                new_v = v - align
                if min_per_expert > 0 and new_v > 0 and new_v < min_per_expert:
                    base[i, idx] = 0
                    removed = v // align
                    removed_chunks += removed
                    remain -= removed
                else:
                    base[i, idx] = new_v
                    removed_chunks += 1
                    remain -= 1
                j += 1
    
    allocate_chunks_time = time()
    
    K_final = torch.clamp(base, min=torch.zeros_like(caps_e), max=caps_e)
    if min_per_expert > 0:
        K_final[(K_final > 0) & (K_final < min_per_expert)] = 0
    for i in range(L):
        row = K_final[i]
        cap = int(layer_caps_vec[i].item())
        s = int(row.sum().item())
        if s > cap:
            over = s - cap
            over_chunks = (over + align - 1) // align
            cand = torch.nonzero(row > 0, as_tuple=False).flatten()
            order = cand[torch.argsort(row[cand])]
            j = 0
            while over_chunks > 0 and j < order.numel():
                idx = int(order[j].item())
                v = int(row[idx].item())
                if v <= 0:
                    j += 1
                    continue
                nv = v - align
                if min_per_expert > 0 and nv > 0 and nv < min_per_expert:
                    row[idx] = 0
                    over_chunks -= int((v + align - 1) // align)
                else:
                    row[idx] = nv
                    over_chunks -= 1
                j += 1
            K_final[i] = row

    final_total = int(K_final.sum().item())
    total_time = time() 
    
    info = {
        "method": "align",
        "align": int(align),
        "min_per_expert": int(min_per_expert),
        "target_total": int(target_total),
        "final_total": int(final_total),
        "added": int(added_chunks * align),
        "removed": int(removed_chunks * align),
        "unmet_diff": int(final_total - target_total),
    }
    
    if verbose:
        _print("[Time Summary] align time summary ")
        _print(f"    base time: {(base_time - start_time) * 1000:.2f}ms")
        _print(f"    compute_slot_headroom time: {(compute_slot_headroom_time - base_time) * 1000:.2f}ms")
        _print(f"    allocate_chunks time: {(allocate_chunks_time - compute_slot_headroom_time) * 1000:.2f}ms")
        _print(f"    finalize_clamp_and_cap time: {(total_time - allocate_chunks_time) * 1000:.2f}ms")
        _print(f"    total time: {(total_time - start_time) * 1000:.2f}ms")
    
    I = cap_per_expert
    masks = torch.zeros((L, E, I), dtype=torch.bool, device=K_final.device)
    for lid in range(L):
        for eid in range(E):
            score_vec = mat[lid][eid].to(device=K_final.device)
            topk_idx = torch.topk(score_vec, k=K_final[lid][eid], largest=True).indices
            masks[lid][eid].index_fill_(0, topk_idx, True)
    
    return K_final, masks, info
