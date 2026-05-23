import csv
import os
import re
import torch
from typing import Dict, Tuple, Any
from src.base.shared_utils.dict_to_tensor import dict_to_tensor
from src.prune.generate.planners import inter_layer_planner
from src.base.shared_utils import _print

__all__ = [
    "load_expert_evict_loss",
    "load_channel_scores",
    "load_layerwise_loss",
    "prepare_scores"
]

def load_expert_evict_loss(path: str, L: int, E: int) -> torch.Tensor:
    """
    Load loss data from CSV as expertwise_scores.

    Args:
        path: CSV file path; must include layer_idx, expert_idx, delta_nll
        L: number of layers
        E: experts per layer

    Returns:
        expertwise_scores: [L, E] tensor
    """
    tensor = torch.zeros((L, E), dtype=torch.float32)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"{path} header is empty.")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if "delta_nll" not in reader.fieldnames:
            raise ValueError(f"{path} must contain delta_nll column.")
        for row in reader:
            lid_raw = row.get("layer_idx")
            eid_raw = row.get("expert_idx")
            val_raw = row.get("delta_nll")
            if lid_raw is None or eid_raw is None or val_raw is None:
                continue
            try:
                lid = int(float(lid_raw.strip()))
                eid = int(float(eid_raw.strip()))
                delta = float(val_raw)
            except (TypeError, ValueError):
                continue
            if not (0 <= lid < L and 0 <= eid < E):
                continue
            delta = max(0.0, delta)
            if delta > tensor[lid, eid]:
                tensor[lid, eid] = delta
    return tensor


def load_channel_scores(scores_dir: str, device, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    if verbose:
        _print(f"[Score Loading] Loading channel scores from {scores_dir}")
    expert_scores = torch.load(os.path.join(scores_dir, "expert_scores.pth"), map_location=device)
    return expert_scores

def load_layerwise_loss(scores_dir: str, inter_layer_method: str, device: str, verbose: bool = True) -> Dict[str, Any]:
    m = re.match(r"loss_smooth_(\d+)", inter_layer_method)
    smooth_times = int(m.group(1)) if m else 0
    loss_based_kwargs = {"layerwise_loss": None, "smooth_times": 0}
    if 'loss' in inter_layer_method:
        layerwise_loss = torch.load(os.path.join(scores_dir, "layerwise_loss.pth"), map_location=device)
        if verbose: 
            _print(f"[Score Loading] Loading layerwise_loss (shape: {layerwise_loss.shape}), smooth_times={smooth_times}")
        loss_based_kwargs = {"layerwise_loss": layerwise_loss, "smooth_times": smooth_times}

    return loss_based_kwargs


def prepare_scores(
    scores_dir: str,
    mask_method_kwargs: Dict[str, Any],
    prune_ratio: float,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    """
    Prepare scores based on mask_method_kwargs.

    Args:
        scores_dir: scores directory path
        mask_method_kwargs: includes intra_expert_metric, intra_layer_method, etc.
        device: target device
        verbose: whether to print details

    Returns:
        intermediate_scores: [L, E, I] tensor
        expertwise_scores: [L, E] tensor
        L: number of layers
        E: experts per layer
        I: intermediate size
    """
    
    expert_scores = load_channel_scores(scores_dir, device, verbose)
    
    intra_expert_metric = mask_method_kwargs["intra_expert_metric"]
    intra_layer_method = mask_method_kwargs["intra_layer_method"]
    
    intermediate_scores = expert_scores[intra_expert_metric]
    intermediate_scores = dict_to_tensor(intermediate_scores)
    L, E, I = intermediate_scores.shape
    
    if verbose:
        _print(f"[Score Loading] Using intermediate metric: {intra_expert_metric}")
        _print(f"[Score Loading] Intermediate shape: {intermediate_scores.shape}")
    
    if intra_layer_method == "attr_coverage":
        expertwise_scores = expert_scores["expert_out_token_contrib"]
        expertwise_scores = dict_to_tensor(expertwise_scores)
        expertwise_scores = -expertwise_scores
    elif 'loss' in intra_layer_method:  
        loss_file_path = os.path.join(os.path.dirname(os.path.dirname(scores_dir)), "loss_csv_files", "single_expert_nll_results.csv")
        if not os.path.exists(loss_file_path):
            raise FileNotFoundError(f"loss file {loss_file_path} not found")
        if verbose:
            _print(f"Loading loss from {loss_file_path}")
        expertwise_scores = load_expert_evict_loss(path=loss_file_path, L=L, E=E)
        expertwise_scores = expertwise_scores.to(device=device)
    elif "usage" in intra_layer_method:  # usage, usage_coverage
        gate_scores = torch.load(os.path.join(scores_dir, "gate_scores.pth"), map_location=device)
        expertwise_scores = gate_scores["usage"]
        expertwise_scores = dict_to_tensor(expertwise_scores)
        expertwise_scores = expertwise_scores.to(device=device)
    elif "router" in intra_layer_method:  # router, router_coverage
        gate_scores = torch.load(os.path.join(scores_dir, "gate_scores.pth"), map_location=device)
        expertwise_scores = gate_scores["out"]
        expertwise_scores = dict_to_tensor(expertwise_scores).squeeze()
        expertwise_scores = expertwise_scores.to(device=device)
    else: # uniform, uniform_coverage, channel_ranking
        expertwise_scores = torch.ones((L, E), dtype=torch.float32, device=device)

    inter_layer_method = mask_method_kwargs["inter_layer_method"]
    loss_based_kwargs = load_layerwise_loss(scores_dir, inter_layer_method, device, verbose)
    
    layerwise_keep_plan = inter_layer_planner(
        intermediate_scores,
        p_target=prune_ratio,
        method=inter_layer_method,
        L=L,
        loss_based_importance_kwargs=loss_based_kwargs,
        tol=0.1,
        verbose=verbose,
    )
    loss_based_kwargs["layerwise_keep_plan"] = layerwise_keep_plan

    return intermediate_scores, expertwise_scores, L, E, I, loss_based_kwargs
