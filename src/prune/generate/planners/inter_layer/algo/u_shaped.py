import numpy as np

def quadratic_u_schedule(
    L: int,
    keep_global: float = 0.5,   # Global mean keep ratio
    beta: float = 0.35,         # Curvature/contrast; larger -> higher ends (0.2 ~ 0.6)
    center: float = 0.5,        # U-shape center (0~1)
    rmin: float | None = None,  # Per-layer lower bound (e.g., 0.2); None = no bound
    rmax: float | None = None,  # Per-layer upper bound (e.g., 0.8); None = no bound
    verbose: bool = False,
) -> np.ndarray:
    """
    Return per-layer keep_ratio with shape (L,):
      r[l] ≈ keep_global + beta * std_normalized((x-center)^2)
    Create a U-shape, clamp by rmin/rmax
    """
    x = np.linspace(0.0, 1.0, L)
    base = (x - center) ** 2
    base = (base - base.mean()) / (base.std() + 1e-12)

    r = keep_global + beta * keep_global * base
    if rmin is not None:
        r = np.maximum(r, rmin)
    if rmax is not None:
        r = np.minimum(r, rmax)

    if verbose:
        print(f"\t beta={beta}, center={center}, rmax={r.max()}, rmin={r.min()}")
    return r


def u_shaped_keep_plan(p_target: float, beta: float = 0.02, rmin: float = 0.10, center: float = 0.5, L: int = None, tol: float = 1e-5, verbose: bool = False):  
    layerwise_scores = quadratic_u_schedule(L, keep_global=1-p_target, beta=beta, rmin=rmin, center=center, verbose=verbose)
    total_keep_ratio = (1-p_target) * L 
    total_scores = sum(layerwise_scores)
    layerwise_keep_ratio = [total_keep_ratio * (layerwise_scores[lid] / total_scores) for lid in range(L)]
    
    assert abs(sum(layerwise_keep_ratio)/L - p_target) < tol, "u_shaped_keep_plan: layerwise_keep_ratio.mean() - p_target is not close to 0"

    if verbose:
        print(f"\t p_target={p_target}, beta={beta}, rmin={rmin}, center={center}, L={L}")
        print(f"\t u_shaped_keep_plan: {layerwise_keep_ratio}")
    return layerwise_keep_ratio
