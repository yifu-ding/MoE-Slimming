import torch.nn as nn
from typing import Any, List, Tuple
from peft import PeftModelForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

__all__ = ['_is_moe_block', 
           '_is_mlp', 
           '_get_num_experts', 
           '_get_moe_intermediate_size', 
           '_get_num_hidden_layers',
           "_get_num_hidden_size", 
           '_is_ds_model',
           '_is_qwen_model', 
           '_is_qwen2_model',
           '_is_qwen3_model',
           '_get_model_layer',
           '_get_attn_module',
           '_get_moe_block',
           '_get_experts',
           '_get_attn_num_heads',
           '_get_attn_num_kv_heads',
           '_get_attn_head_dim',
           '_get_mlp_block',
           "_get_topk", 
           "_get_router_gate", 
           "_collect_attn_modules",
           "_infer_deepseekv2_dims", 
           ]

def _infer_deepseekv2_dims(attn: nn.Module) -> Tuple[int, int, int, int, int]:
    """
    Returns:
        num_heads,
        qk_head_dim,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim
    """
    # Prefer explicit attributes used by forward.
    qk_head_dim = getattr(attn, "qk_head_dim", None)
    qk_nope = getattr(attn, "qk_nope_head_dim", None)
    qk_rope = getattr(attn, "qk_rope_head_dim", None)
    v_head_dim = getattr(attn, "v_head_dim", None)

    if qk_head_dim is None and (qk_nope is not None) and (qk_rope is not None):
        qk_head_dim = int(qk_nope) + int(qk_rope)

    if qk_head_dim is None or qk_nope is None or qk_rope is None or v_head_dim is None:
        raise RuntimeError(
            "Cannot infer DeepSeek-V2 attention dims. "
            "Need attn.{qk_head_dim,qk_nope_head_dim,qk_rope_head_dim,v_head_dim}."
        )

    qk_head_dim = int(qk_head_dim)
    qk_nope = int(qk_nope)
    qk_rope = int(qk_rope)
    v_head_dim = int(v_head_dim)

    # Infer num_heads from q projection output features.
    # q path could be q_proj or q_b_proj depending on LoRA rank.
    if getattr(attn, "q_lora_rank", None) is None:
        q_lin = getattr(attn, "q_proj", None)
    else:
        q_lin = getattr(attn, "q_b_proj", None)

    if q_lin is None or not hasattr(q_lin, "out_features"):
        raise RuntimeError("Cannot find q_proj/q_b_proj to infer num_heads.")
    q_out = int(q_lin.out_features)
    if q_out % qk_head_dim != 0:
        raise RuntimeError(f"q_out_features={q_out} not divisible by qk_head_dim={qk_head_dim}.")
    num_heads = q_out // qk_head_dim

    return int(num_heads), qk_head_dim, qk_nope, qk_rope, v_head_dim


def _collect_attn_modules(model: nn.Module) -> List[nn.Module]:
    attn_modules: List[nn.Module] = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for blk in model.model.layers:
            if hasattr(blk, "self_attn"):
                attn_modules.append(blk.self_attn)
            elif hasattr(blk, "attention"):
                attn_modules.append(blk.attention)
    else:
        tmp = []
        for n, m in model.named_modules():
            # DeepSeekV2Attention has q_proj and o_proj and kv_b_proj
            if hasattr(m, "q_proj") and hasattr(m, "o_proj") and hasattr(m, "kv_b_proj"):
                tmp.append((n, m))
        tmp.sort(key=lambda x: x[0])
        attn_modules = [m for _, m in tmp]
    return attn_modules


def _is_deepseek_moe_block(module):
    if hasattr(module, '__class__') and "DeepseekMoE" in module.__class__.__name__:
        return True
    elif hasattr(module, '__class__') and "DeepseekV3MoE" in module.__class__.__name__:
        return True
    elif hasattr(module, '__class__') and "DeepseekV2MoE" in module.__class__.__name__:
        return True
    else:
        return False
    
def _is_moe_block(module):
    if _is_deepseek_moe_block(module):
        return True
    return isinstance(module, (Qwen3MoeSparseMoeBlock, Qwen2MoeSparseMoeBlock))
        
def _is_mlp(module):
    if _is_deepseek_mlp(module):
        return True
    if hasattr(module, '__class__') and module.__class__.__name__ == 'Qwen3MoeMLP':
        return True
    elif hasattr(module, '__class__') and module.__class__.__name__ == 'Qwen2MoeMLP':
        return True
    else:
        return False
    

def _is_deepseek_mlp(module):
    if hasattr(module, '__class__') and module.__class__.__name__ == 'DeepseekV3MLP':
        return True
    elif hasattr(module, '__class__') and module.__class__.__name__ == 'DeepseekV2MLP':
        return True
    elif hasattr(module, '__class__') and module.__class__.__name__ == 'DeepseekMLP':
        return True
    else:
        return False

def _get_num_experts(model):
    if isinstance(model, PeftModelForCausalLM):
        return _get_num_experts(model.model)
    if _is_ds_model(model):
        if hasattr(model, "config") and hasattr(model.config, "n_routed_experts"):
            return model.config.n_routed_experts
    elif _is_qwen_model(model):
        return model.config.num_experts
    else:
        raise ValueError(f"Unsupported model: {model.__class__.__name__}")
    
def _get_moe_intermediate_size(model):
    if isinstance(model, PeftModelForCausalLM):
        return _get_moe_intermediate_size(model.model)
    return model.config.moe_intermediate_size

def _get_num_hidden_layers(model):
    if isinstance(model, PeftModelForCausalLM):
        return _get_num_hidden_layers(model.model)
    return model.config.num_hidden_layers

def _get_num_hidden_size(model):
    if isinstance(model, PeftModelForCausalLM):
        return _get_num_hidden_size(model.model)
    return model.config.hidden_size


def _get_cfg(model: Any):
    return getattr(model, "config", None)

def _cfg_str(cfg: Any, key: str, default: str = "") -> str:
    v = getattr(cfg, key, default)
    return str(v) if v is not None else default


def _is_qwen2_model(model: Any) -> bool:
    """
    Detect Qwen2 (including Qwen2-MoE) models.
    """
    cfg = _get_cfg(model)
    if cfg is not None:
        mt = _cfg_str(cfg, "model_type", "").lower()
        if "qwen2" in mt:
            return True
    return False

def _is_qwen3_model(model: Any) -> bool:
    """
    Detect Qwen3 models.
    """
    cfg = _get_cfg(model)
    if cfg is not None:
        mt = _cfg_str(cfg, "model_type", "").lower()
        if "qwen3" in mt:
            return True
    return False


def _is_qwen_model(model: Any) -> bool:
    """
    Detect Qwen models (Qwen2 or Qwen3, including MoE variants).
    """
    return _is_qwen2_model(model) or _is_qwen3_model(model)


def _is_ds_model(model: Any) -> bool:
    """
    Detect DeepSeek models (DeepSeek-V2, DeepSeek-V3, including MoE variants).
    """
    cfg = _get_cfg(model)
    if cfg is not None:
        mt = _cfg_str(cfg, "model_type", "").lower()
        # Check for deepseek_v2, deepseek_v3, deepseekv2, deepseekv3, etc.
        if "deepseek" in mt:
            return True
    return False


def _get_model_layer(model, layer_idx: int = None):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers if layer_idx is None else model.model.layers[layer_idx]
    elif hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        return model.model.model.layers if layer_idx is None else model.model.model.layers[layer_idx]
    elif hasattr(model, "layers"):
        return model.layers if layer_idx is None else model.layers[layer_idx]
    raise AttributeError("Cannot locate layers on model.")


def _get_attn_module(model, layer_idx: int):
    layer = _get_model_layer(model, layer_idx)
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    elif hasattr(layer, "attention"):
        return layer.attention
    raise AttributeError(f"Cannot find attention module on layer {layer_idx}.")


def _get_moe_block(model, layer_idx: int):
    layer = _get_model_layer(model, layer_idx)
    if hasattr(layer, "mlp"):
        return layer.mlp
    elif hasattr(layer, "moe"):
        return layer.moe
    raise AttributeError(f"Cannot find MoE block on layer {layer_idx}.")

def _get_mlp_block(block):
    """
    Unify access to MoE MLP module for _is_moe_block().
    """
    if hasattr(block, "mlp"):
        return block.mlp
    if hasattr(block, "feed_forward"):
        return block.feed_forward
    if hasattr(block, "ffn"):
        return block.ffn
    raise AttributeError("Cannot locate MLP/FFN module on block. Please update _get_mlp_block().")


def _get_experts(moe_block):
    if hasattr(moe_block, "experts"):
        return moe_block.experts
    return None
    # raise AttributeError(f"Cannot find experts in type {type(moe_block).__name__}.")


def _get_attn_num_heads(attn) -> int:
    for k in ["num_heads", "num_attention_heads", "n_heads"]:
        if hasattr(attn, k):
            return int(getattr(attn, k))
    config = _get_cfg(attn)
    if config is not None:
        for k in ["num_attention_heads", "num_heads", "n_heads"]:
            if hasattr(config, k):
                return int(getattr(config, k))
    raise AttributeError("Cannot find num_heads on attention module.")


def _get_attn_num_kv_heads(attn) -> int:
    for k in ["num_key_value_heads", "num_kv_heads", "n_kv_heads"]:
        if hasattr(attn, k):
            return int(getattr(attn, k))
    config = _get_cfg(attn)
    if config is not None:
        for k in ["num_key_value_heads", "num_kv_heads", "n_kv_heads"]:
            if hasattr(config, k):
                return int(getattr(config, k))
    raise AttributeError("Cannot find num_key_value_heads on attention module.")


def _get_attn_head_dim(attn) -> int:
    for k in ["head_dim", "head_size"]:
        if hasattr(attn, k):
            return int(getattr(attn, k))
    config = _get_cfg(attn)
    if config is not None:
        for k in ["head_dim", "head_size"]:
            if hasattr(config, k):
                return int(getattr(config, k))
    
    raise AttributeError("Cannot find head_dim on attention module.")


def _get_topk(model):
    if hasattr(model, "config") and hasattr(model.config, "num_experts_per_tok"):
        return model.config.num_experts_per_tok
    else:
        raise ValueError(f"Cannot find topk on model: {model.__class__.__name__}")
    

def _get_router_gate(block: nn.Module) -> nn.Module | None:
    if hasattr(block, "mlp") and isinstance(getattr(block, "mlp"), nn.Module):
        mlp = getattr(block, "mlp")
        if hasattr(mlp, "gate") and isinstance(getattr(mlp, "gate"), nn.Module):
            return getattr(mlp, "gate")
    if hasattr(block, "moe") and isinstance(getattr(block, "moe"), nn.Module):
        moe = getattr(block, "moe")
        if hasattr(moe, "gate") and isinstance(getattr(moe, "gate"), nn.Module):
            return getattr(moe, "gate")
    if hasattr(block, "gate") and isinstance(getattr(block, "gate"), nn.Module):
        return getattr(block, "gate")
    for _, m in block.named_modules():
        if hasattr(m, "gate") and isinstance(getattr(m, "gate"), nn.Module):
            return getattr(m, "gate")
    return None