from src.base.shared_utils import _print

from .algo.coverage import coverage_binary_search_keep_plan
from .algo.u_shaped import u_shaped_keep_plan
from .algo.uniform import uniform_keep_plan
from .algo.loss_based import loss_based_importance_keep_plan, smooth_layerwise_loss
from time import time

__all__ = [
    "supported_inter_layer_methods", 
    "inter_layer_planner",
]

supported_inter_layer_methods = ["uniform", "loss_coverage", "uniform_coverage", "loss", "loss_smooth_1", "loss_smooth_2", "u_shaped", "raw_loss_coverage"]


def inter_layer_planner(scores, p_target: float, method: str = 'uniform', L: int = None, 
               u_shaped_kwargs: dict = { "beta": 0.02, "rmin": 0.10, "center": 0.5}, 
               coverage_kwargs: dict = { "tol": None, "max_iter": 32}, 
               loss_based_importance_kwargs: dict = { "layerwise_loss": None, "smooth_times": 0}, 
               tol: float = 1e-5,
               verbose: bool = False):
    if method == 'uniform':
        return uniform_keep_plan(p_target, L, verbose)

    elif method == 'u_shaped':
        return u_shaped_keep_plan(p_target, **u_shaped_kwargs, L=L, tol=tol, verbose=verbose)

    elif method == 'uniform_coverage':
        return coverage_binary_search_keep_plan(scores, p_target, **coverage_kwargs, verbose=verbose)

    elif method == "loss" or (isinstance(method, str) and method.startswith("loss_smooth_")):
        layerwise_loss = loss_based_importance_kwargs["layerwise_loss"]
        smooth_times = loss_based_importance_kwargs["smooth_times"]
        return loss_based_importance_keep_plan(layerwise_loss, p_target, L, tol=tol, smooth_times=smooth_times, verbose=verbose)

    elif method == "loss_coverage" or method == "raw_loss_coverage":
        layerwise_loss = loss_based_importance_kwargs["layerwise_loss"]
        start_time = time()
        if method == "raw_loss_coverage":
            smooth_times = 0
        else:
            smooth_times = 4
        coverage_kwargs['layerwise_loss'] = smooth_layerwise_loss(layerwise_loss, smooth_times=smooth_times)  # Default uses smooth_times=2
        if verbose:
            _print(f"layerwise loss before smooth: {layerwise_loss}")
            _print(f"layerwise loss after smooth: {coverage_kwargs['layerwise_loss']}")
        smooth_time = time() - start_time
        if verbose:
            _print(f"smooth_layerwise_loss time: {smooth_time * 1000:.2f}ms")
        return coverage_binary_search_keep_plan(scores, p_target, **coverage_kwargs, verbose=verbose)        

    else:
        raise ValueError(f"Invalid method: {method}")
