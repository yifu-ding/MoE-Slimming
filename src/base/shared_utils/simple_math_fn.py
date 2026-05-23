import torch
from torch.nn import functional as F

__all__ = [
    "_layer_norm",
    "angle_loss",
    "as_long_1d",
    "_round_down_to_multiple",
    "_to_cpu_long_1d",
    '_round_f',
]

def _round_f(x: float, nd: int = 6) -> float:
    return float(round(float(x), nd))

def _layer_norm(x, dim=None):
    if x is None:
        return None
    # x = x - x.min()
    if dim is None:
        dim = tuple(range(x.dim()))
    x = x / x.mean(dim=dim, keepdim=True)
    return x

def angle_loss(pred, y, eps=1e-6):
    # pred, y: shape [N, D]
    p = pred.float()
    t = y.float()
    # cosine
    cos = F.cosine_similarity(p, t, dim=-1, eps=eps).clamp(-1+1e-6, 1-1e-6)
    # angle in radians
    theta = torch.acos(cos)   # [N]

    return theta


def as_long_1d(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Convert input to a 1D long tensor, removing duplicates and sorting.

    Args:
        x: Input tensor or array-like
        device: Target device for the output tensor

    Returns:
        1D long tensor with unique, sorted values
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=device, dtype=torch.long)
    x = x.to(device=device, dtype=torch.long).view(-1)
    if x.numel() == 0:
        return x
    x = torch.unique(x)
    x, _ = torch.sort(x)
    return x


def _round_down_to_multiple(x: int, m: int) -> int:
    x = int(x)
    m = int(m)
    if m <= 1:
        return x
    return (x // m) * m


def _to_cpu_long_1d(idx: torch.Tensor) -> torch.Tensor:
    if idx.dtype != torch.long:
        idx = idx.to(torch.long)
    if idx.dim() != 1:
        idx = idx.view(-1)
    return idx.detach().cpu()


