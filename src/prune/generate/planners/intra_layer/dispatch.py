from .algo.layerwise import build_masks_layerwise
from .algo.expertwise import build_masks_expertwise
from .algo.globally import build_masks_globally
from .algo.loss_based import loss_based_expert_keep_plan
from .algo.coverage import build_masks_coverage

__all__ = [
    "build_masks",
    "supported_intra_layer_methods",
    "supported_intra_expert_metrics",
]

supported_intra_layer_methods = ['uniform', 'channel_ranking', 'loss', 'global', 'usage', 'router', 'attr_coverage', "loss_coverage", "usage_coverage", "router_coverage"]
supported_intra_expert_metrics = ["activation", "wa", "grad"]

def build_masks(scores, expertwise_scores=None, keep_ratio=None, top_k=None, method='uniform', L=None, E=None, I=None, verbose=False, **kwargs):
    if method == 'uniform':
        return build_masks_expertwise(scores, keep_ratio=keep_ratio, top_k=top_k, L=L, E=E, I=I, verbose=verbose)
    elif method == 'channel_ranking':
        return build_masks_layerwise(scores, keep_ratio=keep_ratio, top_k=top_k, L=L, E=E, I=I, verbose=verbose)
    elif method == 'loss':
        expert_wise_keep_plan = loss_based_expert_keep_plan(keep_ratio=keep_ratio, L=L, E=E, loss_result=expertwise_scores, verbose=verbose)
        return build_masks_expertwise(scores, keep_ratio=expert_wise_keep_plan, top_k=top_k, L=L, E=E, I=I, verbose=verbose)
    elif method == 'global':
        return build_masks_globally(scores, global_keep_ratio=keep_ratio, L=L, E=E, I=I, verbose=verbose)
    elif method in ['usage', 'attr', 'router']:
        expert_wise_keep_plan = loss_based_expert_keep_plan(keep_ratio=keep_ratio, L=L, E=E, loss_result=expertwise_scores, verbose=verbose)
        return build_masks_expertwise(scores, keep_ratio=expert_wise_keep_plan, top_k=top_k, L=L, E=E, I=I, verbose=verbose)
    elif 'coverage' in method:
        assert expertwise_scores is not None
        return build_masks_coverage(scores, expertwise_scores, keep_ratio=keep_ratio, L=L, E=E, I=I, verbose=verbose)
    else:
        raise ValueError(f"Invalid method: {method}")
