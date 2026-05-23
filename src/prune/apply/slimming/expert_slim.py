"""
Expert module pruning for MoE (Mixture of Experts).

Provides functions to prune MoE expert modules, including intermediate dimension pruning.
"""
import types
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from src.base.shared_utils import _print
from src.base.shared_utils.dict_to_tensor import dict_to_tensor
from src.base.shared_utils.safe_isinstance import (
    _is_ds_model,
    _is_moe_block,
    _get_num_hidden_layers, 
)
from src.prune.apply.slimming.utils import (
    skip_moe_mlp_forward,
    make_gate_mask_hook,
    _dequant_weight_to_16bit,
    _build_slim_linear_16bit,
    _to_linear4bit,
    _n_params_linear,
    _is_load_in_4bit,
    _extract_layer_expert_id,
)

try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = Any  # type: ignore
    _HAS_BNB = False


@torch.no_grad()
def slim_moe_inter(
    model: nn.Module,
    inter_masks: Union[torch.Tensor, List[List[torch.Tensor]]],
    qcfg: BitsAndBytesConfig | None = None,
    shrink_gate: bool = False,
    add_hooks: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Prune MoE intermediate dims (inter) and optionally prune hidden input dims for gate_proj/up_proj.

    Args:
        model: The model to prune
        inter_masks: [L, E, I] bool tensor (keep-mask) for intermediate dimensions
        hidden_keep_masks: [L, E, H] bool tensor (keep-mask) for hidden dimensions, optional
        qcfg: Quantization config (optional)
        shrink_gate: Whether to physically remove pruned experts and shrink router
        add_hooks: Whether to add hooks to mask inactive experts (if not shrink_gate)
        verbose: Whether to _print pruning statistics

    Returns:
        Dictionary with pruning statistics

    Note:
        If hidden_keep_masks is provided:
          - gate/up input columns are reduced to kept hidden dims
          - forward keeps full hidden input by wrapping with GatherLinear(keep_idx, small_linear)
          - down_proj is NOT hidden-pruned (it maps inter -> hidden, hidden stays full)
    """
    if verbose:
        _print(f"[real slim] Detect quantization config: {qcfg}")

    if qcfg is not None and _is_load_in_4bit(qcfg) and not _HAS_BNB:
        raise RuntimeError("load_in_4bit=True but bitsandbytes not installed.")

    inter_t = dict_to_tensor(inter_masks, dtype=torch.bool)

    params_removed = 0
    params_kept = 0
    n_inter_before = 0
    n_inter_after = 0

    gate_hook_handles = []
    gate_hook_added = 0
    shrink_gate_cnt = 0
    inactive_experts = 0

    skipped_layers = 1 if _is_ds_model(model) else 0
    L = _get_num_hidden_layers(model)

    pbar = tqdm(total=L, desc="Slimming Moe Blocks", leave=True)

    for name, module in model.named_modules():
        lid, _ = _extract_layer_expert_id(name)
        if not _is_moe_block(module):
            continue

        lid -= skipped_layers
        if lid < 0 or lid >= L:
            continue

        old_num_experts = len(module.experts)
        layer_active_expert = torch.ones(old_num_experts, dtype=torch.bool)

        keep_eids: List[int] = []
        new_experts: List[nn.Module] = []

        for eid, expert in enumerate(list[Any](module.experts)):
            gate = expert.gate_proj
            up = expert.up_proj
            down = expert.down_proj

            dtype = gate.weight.dtype
            device = gate.weight.device

            inter = int(up.out_features)
            assert int(gate.out_features) == inter and int(down.in_features) == inter
            assert int(gate.in_features) == int(up.in_features) == int(down.out_features)

            m_inter = inter_t[lid, eid].to(device=device, dtype=torch.bool)
            inter_keep_total = int(m_inter.sum().item()) 
            n_inter_before += inter
            n_inter_after += inter_keep_total

            if inter_keep_total == 0:
                layer_active_expert[eid] = False
                params_removed += int(inter * int(gate.in_features) * 2 + int(down.out_features) * inter)
                expert.forward = types.MethodType(skip_moe_mlp_forward, expert)
                continue

            W_gate, b_gate = _dequant_weight_to_16bit(gate)
            W_up, b_up = _dequant_weight_to_16bit(up)
            W_down, b_down = _dequant_weight_to_16bit(down)

            W_gate_new = W_gate[m_inter, :]
            b_gate_new = None if b_gate is None else b_gate[m_inter]
            W_up_new = W_up[m_inter, :]
            b_up_new = None if b_up is None else b_up[m_inter]
            W_down_new = W_down[:, m_inter]
            b_down_new = b_down

            small_gate = _build_slim_linear_16bit(W_gate_new, b_gate_new, device=device, dtype=torch.float16)
            small_up = _build_slim_linear_16bit(W_up_new, b_up_new, device=device, dtype=torch.float16)
            small_down = _build_slim_linear_16bit(W_down_new, b_down_new, device=device, dtype=torch.float16)

            if qcfg is not None and _is_load_in_4bit(qcfg):
                small_gate = _to_linear4bit(small_gate, qcfg, device=device, linear_type=type(gate))
                small_up = _to_linear4bit(small_up, qcfg, device=device, linear_type=type(up))
                small_down = _to_linear4bit(small_down, qcfg, device=device, linear_type=type(down))
                is_load_in_4bit = True
            else:
                small_gate = small_gate.to(dtype)
                small_up = small_up.to(dtype)
                small_down = small_down.to(dtype)
                is_load_in_4bit = False

            before = int(W_gate.numel()) + int(W_up.numel()) + int(W_down.numel())
            if b_gate is not None:
                before += int(b_gate.numel())
            if b_up is not None:
                before += int(b_up.numel())
            if b_down is not None:
                before += int(b_down.numel())

            after = (
                _n_params_linear(small_gate, is_load_in_4bit)
                + _n_params_linear(small_up, is_load_in_4bit)
                + _n_params_linear(small_down, is_load_in_4bit)
            )
            params_removed += int(before - after)
            params_kept += int(after)

            expert.gate_proj = small_gate
            expert.up_proj = small_up
            expert.down_proj = small_down

            if shrink_gate:
                keep_eids.append(eid)
                new_experts.append(expert)

        if shrink_gate:
            if len(new_experts) == 0:
                raise RuntimeError(
                    f"All experts in layer {lid} were fully pruned. Adjust masks to keep at least one expert."
                )

            if len(new_experts) != old_num_experts:
                shrink_gate_cnt += (old_num_experts - len(new_experts))
                module.experts = nn.ModuleList(new_experts)

                router_gate = module.gate
                W_router, b_router = _dequant_weight_to_16bit(router_gate)  # [E_old, H]

                keep_mask = torch.zeros(old_num_experts, device=W_router.device, dtype=torch.bool)
                keep_mask[torch.tensor(keep_eids, device=W_router.device, dtype=torch.long)] = True

                W_router_new = W_router[keep_mask, :]
                b_router_new = None if b_router is None else b_router[keep_mask]

                if isinstance(router_gate, nn.Linear):
                    small_router_gate = _build_slim_linear_16bit(
                        W_router_new, b_router_new, device=W_router.device, dtype=torch.float16
                    )
                    if qcfg is not None and _is_load_in_4bit(qcfg):
                        small_router_gate = _to_linear4bit(
                            small_router_gate, qcfg, device=W_router.device, linear_type=type(router_gate)
                        )
                    else:
                        small_router_gate = small_router_gate.to(router_gate.weight.dtype)
                    module.gate = small_router_gate
                elif hasattr(router_gate, "__class__") and "MoEGate" in router_gate.__class__.__name__:
                    module.gate.weight = nn.Parameter(W_router_new)

                module.num_experts = len(new_experts)

                topk = None
                if hasattr(module, "top_k"):
                    module.top_k = min(int(module.top_k), int(module.num_experts))
                    topk = module.top_k
                elif hasattr(module.gate, "top_k"):
                    module.gate.top_k = min(int(module.gate.top_k), int(module.num_experts))
                    topk = module.gate.top_k

                if hasattr(module, "config") and hasattr(module.config, "num_experts"):
                    module.config.num_experts = module.num_experts
                if hasattr(module, "config") and hasattr(module.config, "num_experts_per_tok"):
                    module.config.num_experts_per_tok = topk if topk is not None else module.config.num_experts_per_tok

                n_active = int(layer_active_expert.sum().item())
                inactive_experts += int(layer_active_expert.numel()) - n_active

        else:
            # hook path: mask inactive experts on router
            if hasattr(module, "gate") and isinstance(module.gate, nn.Linear) and add_hooks:
                top_k = int(getattr(module, "top_k", 1))
                n_active = int(layer_active_expert.sum().item())
                inactive_experts += int(layer_active_expert.numel()) - n_active
                if n_active < top_k:
                    raise RuntimeError(
                        f"Layer {lid}: active experts ({n_active}) < top_k ({top_k}). Routing invalid."
                    )
                h = module.gate.register_forward_hook(make_gate_mask_hook(layer_active_expert))
                gate_hook_handles.append(h)
                gate_hook_added += 1

        pbar.update(1)

    pbar.close()

    if verbose:
        total_params = params_removed + params_kept
        inter_prune_ratio = (1 - n_inter_after / max(n_inter_before, 1)) * 100.0
        real_prune_ratio = (1 - params_kept / max(total_params, 1)) * 100.0
        if shrink_gate:
            _print(f"[Gate Shrunk] ✅ shrinked gate dims: {shrink_gate_cnt}, {inactive_experts} experts has been removed. ")
            assert inactive_experts == shrink_gate_cnt
        else:
            _print(f"[Gate Hooked] ✅ {gate_hook_added} hooks added to mask {inactive_experts} inactive experts, but gate is not modified. ")
            
        _print(f"[real slim] pruned_inter_ratio%: {inter_prune_ratio:.4f}")
        _print(f"[real slim] pruned_param_ratio%: {real_prune_ratio:.4f} (kept: {params_kept:,} / total: {total_params:,})")

    return {
        "n_inter_before": int(n_inter_before),
        "n_inter_after": int(n_inter_after),
        "params_removed": int(params_removed),
        "params_kept": int(params_kept),
        "shrink_gate": int(shrink_gate_cnt),
        "gate_hook_added": int(gate_hook_added),
    }
