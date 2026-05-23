import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

def _largest_remainder_alloc(ideals: torch.Tensor, caps: torch.Tensor, target: int) -> torch.Tensor:
    """
    standard largest remainder method to allocate integers, constrained by caps.
    ideals: float64 non-negative, sum to expected value
    caps: int64 non-negative, element-wise upper bound
    target: total integers to allocate
    return:
      int64, sum equals min(target, sum(caps)), and stays close to ideals.
    """
    assert ideals.dtype in (torch.float64, torch.float32)
    assert caps.dtype == torch.int64
    n = ideals.numel()
    base = torch.floor(ideals).to(torch.int64)
    base = torch.minimum(base, caps)
    s = int(base.sum().item())
    remain = min(target, int(caps.sum().item())) - s
    if remain <= 0:
        return base

    frac = (ideals - base.to(ideals.dtype))
    mask = (base < caps)
    scores = torch.where(mask, frac, torch.full_like(frac, -1e9))
    order = torch.argsort(scores, descending=True)
    res = base.clone()
    j = 0
    while remain > 0 and j < n:
        idx = int(order[j].item())
        if res[idx] < caps[idx]:
            res[idx] += 1
            remain -= 1
        j += 1
    return res

