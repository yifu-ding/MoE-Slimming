from src.base.shared_utils import _print
def uniform_keep_plan(p_target: float, L: int, verbose: bool = False):
    if verbose:
        _print(f"\t uniform_prune_plan: p_target={p_target}, L={L}")
    return [1-p_target for _ in range(L)]