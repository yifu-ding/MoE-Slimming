import torch
from src.base.shared_utils import _print

def smooth_layerwise_loss(layerwise_loss: torch.Tensor, smooth_times=0) -> torch.Tensor:
    if layerwise_loss is None:
        return None

    for _ in range(smooth_times):
        layerwise_loss = torch.sqrt(layerwise_loss) # [L]
        _std = layerwise_loss.std()
        _mean = layerwise_loss.mean()
        layerwise_loss = layerwise_loss.clamp(min=_mean - _std, max=_mean + _std)

    final_mean = layerwise_loss.mean()
    if final_mean > 0:
        layerwise_loss = layerwise_loss / final_mean
    return layerwise_loss

def loss_based_importance_keep_plan(layerwise_loss, p_target, L: int, tol: float = 1e-5, smooth_times=0, verbose=False):
    assert layerwise_loss.shape == (L,), "layerwise_loss must have shape (L,)"
    layerwise_loss = smooth_layerwise_loss(layerwise_loss, smooth_times=smooth_times)
    loss_sum = sum(layerwise_loss)
    total_keep_ratio = (1 - p_target) * L
    layerwise_keep_ratio = total_keep_ratio * (layerwise_loss / loss_sum)
    try:
        assert abs(sum(layerwise_keep_ratio)/L - (1-p_target)) < tol, "loss_based_importance_keep_plan: layerwise_keep_ratio.mean() - p_target is not close to 0"
    except:
        _print(f"\t loss_based_importance_keep_plan: layerwise_keep_ratio.mean() is not close to 1-p_target, diff={abs(sum(layerwise_keep_ratio)/L - (1-p_target))}")
        return None
    
    try:
        assert all(layerwise_keep_ratio >= 0) and all(layerwise_keep_ratio <= 1), "loss_based_importance_keep_plan: layerwise_keep_ratio is not in [0, 1]"
    except:
        _print(f"\t loss_based_importance_keep_plan: layerwise_keep_ratio is not in [0, 1], {layerwise_keep_ratio}")
        return None
    
    if verbose:
        _print(f"\t loss_based_importance_keep_plan: {layerwise_keep_ratio}")
    return layerwise_keep_ratio