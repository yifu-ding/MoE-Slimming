import math
import torch
from typing import Dict, Any, Tuple, List
from time import time
from src.base.shared_utils import _print

def prepare_layer_info(
    scores: Dict[int, torch.Tensor],
    clamp_non_negative: bool = True,
) -> Tuple[Dict[int, Any], int, float]:
    """
    Preprocess per-layer saliency:
      - flatten to 1D
      - optional clamp to non-negative
      - sort descending
      - compute prefix sums

    Returns:
      layer_info: dict[layer_idx] -> {
          "size": int,                # layer channel count D_l
          "sorted": Tensor[D_l],      # sorted descending
          "prefix": Tensor[D_l],      # prefix sum
          "total_saliency": float,    # last prefix element
      }
      total_channels: total channel count across layers N
      total_saliency: total saliency sum (for global coverage)
    """
    layer_info: Dict[int, Any] = {}
    total_channels = 0
    total_saliency = 0.0

    for l in range(len(scores)):
        v = scores[l]
        vals = v.reshape(-1).float()
        if clamp_non_negative:
            vals = vals.clamp_min(0.0)

        d_l = vals.numel()
        total_channels += d_l

        if d_l == 0:
            layer_info[l] = {
                "size": 0,
                "sorted": None,
                "prefix": None,
                "total_saliency": 0.0,
            }
            continue

        sorted_vals, _ = torch.sort(vals, descending=True)
        prefix = torch.cumsum(sorted_vals, dim=0)
        total = prefix[-1].item() if d_l > 0 else 0.0

        total_saliency += total

        layer_info[l] = {
            "size": d_l,
            "sorted": sorted_vals,
            "prefix": prefix,
            "total_saliency": total,
        }

    return layer_info, total_channels, float(total_saliency)


def eval_prune_ratio_for_s(
    layer_info: Dict[int, Any],
    total_channels: int,
    s_list: List[float],
) -> Tuple[float, Dict[int, int]]:
    """
    Given per-layer target coverage ratio s, compute per-layer kept channels
    k_l(s) and global prune ratio p(s).

    Idea:
      - For each layer:
          If total_saliency <= 0: fall back to uniform saliency, keep ceil(s * D_l)
          Else: find smallest k where prefix[k-1] >= s * total_saliency
      - Sum K(s) = sum_l k_l(s)
      - Global prune ratio p(s) = 1 - K(s) / N
    """
    keep_counts: Dict[int, int] = {}
    if total_channels == 0:
        return 0.0, keep_counts  # Edge case

    K = 0  # Global kept channel count
    for l, info in layer_info.items():
        s = s_list[l].item()
        d_l = info["size"]
        if d_l == 0:
            keep_counts[l] = 0
            continue

        total = info["total_saliency"]
        prefix = info["prefix"]

        if total <= 0.0:
            if s <= 0.0:
                k_l = 0
            else:
                k_l = int(math.ceil(s * d_l))
        else:
            target = float(s) * total
                if target <= 0.0:
                    k_l = 0
                else:
                    t = torch.tensor(target, device=prefix.device)
                idx = torch.searchsorted(prefix, t, right=False)
                k_l = int(idx.item()) + 1   # idx is 0-based

        # Clamp to valid range
        if k_l > d_l:
            k_l = d_l
        if k_l < 0:
            k_l = 0

        keep_counts[l] = k_l
        K += k_l

    prune_ratio = 1.0 - float(K) / float(total_channels)
    return prune_ratio, keep_counts


def binary_search_s_for_target_prune(
    layer_info: Dict[int, Any],
    total_channels: int,
    p_target: float,
    layerwise_loss: torch.Tensor = None, 
    max_iter: int = 32,
    tol: float = 1e-3,  # tolerance
) -> Tuple[float, Dict[int, int], float]:
    """
    Binary search s in [0, 1] so global prune ratio p(s) ≈ p_target.
    Returns:
      s_star: chosen s
      keep_counts: dict[layer] -> k_l(s_star)
      p_actual: achieved global prune ratio
    """
    p_target = float(p_target)
    if p_target < 0.0 or p_target > 1.0:
        raise ValueError(f"p_target must be in [0,1], got {p_target}")

    low, high = 0.0, 1.0

    best_s = 0.0
    best_keep = {}
    best_p = None
    best_err = float("inf")

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        s_list = [mid] * len(layer_info)
        s_list = torch.tensor(s_list)
        if layerwise_loss is not None:
            s_list = layerwise_loss * s_list.to(layerwise_loss.device)
        p_mid, keep_mid = eval_prune_ratio_for_s(layer_info, total_channels, s_list=s_list)

        err = abs(p_mid - p_target)
        if err < best_err:
            best_err = err
            best_s = s_list
            best_keep = keep_mid
            best_p = p_mid

        if err < tol:
            break

        if p_mid > p_target:
            low = mid
        else:
            high = mid

    if best_p is None:
        best_p, best_keep = eval_prune_ratio_for_s(layer_info, total_channels, best_s)

    return best_s, best_keep, best_p


def coverage_binary_search_keep_plan(
    scores: Dict[int, torch.Tensor],
    p_target: float,
    layerwise_loss: torch.Tensor = None, 
    max_iter: int = 32,
    tol: float = None,  # tolerance; for 1408 channels, 1 channel ~= 7e-4
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Main entry:
      Inputs:
        scores: dict[layer_idx] -> Tensor, per-layer saliency (any shape; will flatten)
        p_target: target global prune ratio (e.g., 0.5 = prune half)
      Output:
        dict with:
          - "s_star": per-layer saliency coverage lower bound
          - "keep_ratio_per_layer": dict[layer] -> keep ratio
          - "keep_count_per_layer": dict[layer] -> kept channels
          - "global_prune_ratio": achieved global prune ratio
          - "global_saliency_coverage": global saliency coverage ratio
          - "saliency_coverage_per_layer": dict[layer] -> per-layer coverage ratio
    """
    st_time = time()
    if verbose:
        _print("start to compute global prune plan", flush=True)
    
    start_time = time()
    layer_info, total_channels, total_saliency = prepare_layer_info(scores)
    prepare_time = time() - start_time
    
    if tol is None:
        tol = 1 / layer_info[0]["size"]
    if verbose:
        _print(f"tolerance={tol:.6f}", flush=True)
    if total_channels == 0:
        raise ValueError("No channels found in scores. Check your input.")

    start_time = time()
    s_star, keep_counts, p_actual = binary_search_s_for_target_prune(
        layer_info=layer_info,
        total_channels=total_channels,
        p_target=p_target,
        layerwise_loss=layerwise_loss,
        max_iter=max_iter,
        tol=tol,
    )
    binary_search_time = time() - start_time

    keep_ratio_per_layer = torch.zeros(len(scores), dtype=torch.float32)
    saliency_coverage_per_layer = torch.zeros(len(scores), dtype=torch.float32)
    kept_saliency_total = 0.0

    for l, info in layer_info.items():
        d_l = info["size"]
        k_l = keep_counts.get(l, 0)
        if d_l > 0:
            keep_ratio_per_layer[l] = float(k_l) / float(d_l)  # keep ratio
        else:
            raise ValueError(f"No channels found in layer {l}")

        total_l = info["total_saliency"]
        prefix = info["prefix"]

        if total_l > 0.0 and k_l > 0:
            kept_l = prefix[k_l - 1].item()
            cov_l = kept_l / (total_l + 1e-12)
        else:
            kept_l = 0.0
            cov_l = 0.0

        kept_saliency_total += kept_l
        saliency_coverage_per_layer[l] = float(cov_l)

    if total_saliency > 0.0:
        global_saliency_coverage = kept_saliency_total / (total_saliency + 1e-12)
    else:
        global_saliency_coverage = 0.0

    if verbose:
        _print("[Time Summary] layerwise_planning time summary")
        _print(f"\t layerwise_planning: prepare_layer_info time: {prepare_time * 1000:.2f}ms", flush=True)
        _print(f"\t layerwise_planning: binary_search_time: {binary_search_time * 1000:.2f}ms", flush=True)
        _print(f"\t layerwise_planning: compute global prune plan done in {(time() - st_time) * 1000:.2f}ms", flush=True)
        
        
    result = {
        "s_star": s_star.tolist(),
        "keep_ratio_per_layer": keep_ratio_per_layer,
        "keep_count_per_layer": keep_counts,
        "global_prune_ratio": float(p_actual),
        "global_saliency_coverage": float(global_saliency_coverage),
        "saliency_coverage_per_layer": saliency_coverage_per_layer,
    }
    
    keep_ratio_per_layer = result['keep_ratio_per_layer']
    diff = abs(result['global_prune_ratio'] - p_target)
    try:
        assert diff < tol, f"coverage_binary_search_keep_plan: actual prune diff={diff} is not close to tol={tol}"
    except:
        import ipdb; ipdb.set_trace()
    
    if verbose:
        _print("Probe of coverage_binary_search_keep_plan:")
        for k, v in result.items():
            _print(f"\t {k}: {v}", flush=True)
            
    return keep_ratio_per_layer
