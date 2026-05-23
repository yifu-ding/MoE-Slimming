import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List

try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

from src.base.shared_utils.safe_isinstance import _is_ds_model
    
__all__ = [
    "skip_moe_mlp_forward",
    "GatherLinear",
    "unique_sorted_long_1d",
    "complement_idx",
    "set_if_has",
    "heads_to_rows",
    "make_gate_mask_hook",
    "resolve_hidden_drop_for_layer",
    "_dequant_weight_to_16bit",
    "_build_slim_linear_16bit",
    "_to_linear4bit",
    "_extract_layer_expert_id",
    "_as_tensor_1d",
    "_is_load_in_4bit",
    "_round_down_to_multiple",
    "_topk_idx",
    "_n_params_linear",
    "_infer_num_heads_kv_heads_head_dim",
    "_get_attn_projs",
]

def skip_moe_mlp_forward(self, x):
    y = x.new_zeros(x.shape)
    return y


class GatherLinear(nn.Module):
    """
    Linear layer with input gathering.

    Selects specific input dimensions before applying linear transformation.
    x: [..., full_in]
    x_sel = x[..., idx_in], then linear(len(idx_in) -> out)
    """
    def __init__(self, idx_in: torch.Tensor, linear: nn.Module):
        super().__init__()
        assert idx_in.dtype == torch.long
        self.register_buffer("idx_in", idx_in, persistent=True)
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sel = x.index_select(-1, self.idx_in)
        return self.linear(x_sel)


def unique_sorted_long_1d(x, device: torch.device) -> torch.Tensor:
    """
    Convert input to a unique, sorted 1D long tensor.

    Args:
        x: Input tensor or list of tensors
        device: Target device

    Returns:
        1D long tensor with unique, sorted values
    """
    if isinstance(x, list):
        x = torch.stack(x)
    x = x.to(device=device, dtype=torch.long).view(-1)
    if x.numel() == 0:
        return x
    x = torch.unique(x)
    x, _ = torch.sort(x)
    return x


def complement_idx(drop_idx: torch.Tensor, total: int, device: torch.device) -> torch.Tensor:
    """
    Return keep indices in [0, total-1] that are NOT in drop_idx.

    Args:
        drop_idx: 1D long tensor of indices to drop (may be empty)
        total: Total number of dimensions
        device: Target device

    Returns:
        1D long tensor of keep indices

    Raises:
        ValueError: If drop_idx is invalid or drops all dimensions
    """
    if total <= 0:
        raise ValueError(f"total must be > 0, got {total}")

    drop_idx = unique_sorted_long_1d(drop_idx, device=device)
    if drop_idx.numel() == 0:
        return torch.arange(total, device=device, dtype=torch.long)

    if int(drop_idx.min()) < 0 or int(drop_idx.max()) >= total:
        raise ValueError(
            f"drop_idx out of range: min={int(drop_idx.min())}, max={int(drop_idx.max())}, total={total}"
        )

    keep_mask = torch.ones((total,), device=device, dtype=torch.bool)
    keep_mask[drop_idx] = False
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).view(-1).to(dtype=torch.long)

    if keep_idx.numel() == 0:
        raise ValueError(f"Invalid drop_idx: it drops all {total} dims.")

    return keep_idx


def set_if_has(obj, names: List[str], value: int) -> None:
    """
    Set attribute on object if it exists.

    Args:
        obj: Object to set attribute on
        names: List of attribute names to try
        value: Value to set
    """
    for n in names:
        if hasattr(obj, n):
            setattr(obj, n, int(value))


def heads_to_rows(head_idx: torch.Tensor, head_dim: int, device: torch.device) -> torch.Tensor:
    """
    Convert head indices to row indices in weight matrix.

    Args:
        head_idx: 1D tensor of head indices
        head_dim: Dimension of each head
        device: Target device

    Returns:
        1D tensor of row indices
    """
    head_idx = head_idx.to(device=device, dtype=torch.long).view(-1)
    if head_idx.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    rows = []
    hd = int(head_dim)
    for h in head_idx.tolist():
        start = int(h) * hd
        rows.extend(range(start, start + hd))
    return torch.tensor(rows, device=device, dtype=torch.long)


def make_gate_mask_hook(active_expert_mask_cpu: torch.Tensor):
    """
    Create a forward hook that masks inactive experts in gate output.

    Args:
        active_expert_mask_cpu: [E] bool tensor on CPU

    Returns:
        Hook function
    """
    active_expert_mask_cpu = active_expert_mask_cpu.to(torch.bool).cpu()

    def hook(module: nn.Module, inputs, output: torch.Tensor):
        mask = active_expert_mask_cpu
        if mask.device != output.device:
            mask = mask.to(device=output.device)
        neg = torch.finfo(output.dtype).min
        return output.masked_fill(~mask.view(1, -1), neg)

    return hook


def resolve_hidden_drop_for_layer(
    drop_hidden_layer: Optional[List[torch.Tensor]],
    active_eids: List[int],
    eid: int,
) -> Optional[torch.Tensor]:
    """
    Resolve hidden drop indices for a specific expert.

    Supports two formats for drop_hidden_layer:
      - full: len == E_old, indexed by original eid
      - trimmed: len == len(active_eids), indexed by order of active experts

    Args:
        drop_hidden_layer: List of drop indices per expert
        active_eids: List of active expert IDs
        eid: Expert ID to resolve

    Returns:
        Drop indices tensor for the expert, or None
    """
    if drop_hidden_layer is None:
        return None
    if len(drop_hidden_layer) == 0:
        return torch.empty((0,), dtype=torch.long)
    if len(drop_hidden_layer) == len(active_eids):
        pos = active_eids.index(eid)
        return drop_hidden_layer[pos]
    return drop_hidden_layer[eid]



@torch.no_grad()
def _dequant_weight_to_16bit(linear: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(linear, (bnb.nn.Linear4bit)) and isinstance(linear.weight, (bnb.nn.Params4bit)):
        W = bnb.functional.dequantize_4bit(linear.weight.data, linear.weight.quant_state)
        bias = None
    else:
        W = linear.weight.data
        bias = linear.bias if hasattr(linear, "bias") else None
    return W, bias

@torch.no_grad()
def _build_slim_linear_16bit(
    W_new: torch.Tensor,                 # [out, in]
    bias_new: Optional[torch.Tensor],    # [out] or None
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Linear:
    device = device or W_new.device
    out_features, in_features = W_new.shape
    lin = nn.Linear(in_features, out_features, bias=bias_new is not None, 
                    dtype=dtype, device=device)
    lin.weight.data.copy_(W_new.to(dtype=dtype, device=device))
    if bias_new is not None:
        lin.bias.data.copy_(bias_new.to(dtype=dtype, device=device))
    return lin

@torch.no_grad  
def _to_linear4bit(
    linear: nn.Linear,
    qcfg: BitsAndBytesConfig,
    device: str = "cuda",
    linear_type: type = bnb.nn.Linear4bit,
):
    if isinstance(qcfg, dict):
        quant_type = qcfg.get("bnb_4bit_quant_type", "nf4")
        use_double = qcfg.get("bnb_4bit_use_double_quant", False)
        comp_dtype = qcfg.get("bnb_4bit_compute_dtype", torch.bfloat16)
    else:
        quant_type = getattr(qcfg, "bnb_4bit_quant_type", "nf4")
        use_double = getattr(qcfg, "bnb_4bit_use_double_quant", False)
        comp_dtype = getattr(qcfg, "bnb_4bit_compute_dtype", torch.bfloat16)

    if linear_type is bnb.nn.Linear4bit:
        qlin = bnb.nn.Linear4bit(
            linear.in_features,
            linear.out_features,
            bias=(linear.bias is not None),
            quant_type=quant_type,
            compute_dtype=comp_dtype,
            compress_statistics=bool(use_double),
        )
    else:
        raise ValueError(f"Unsupported linear type: {linear_type}")

    sd = {k: v.to(comp_dtype) for k, v in linear.state_dict().items()}
    qlin.load_state_dict(sd)
    qlin = qlin.to(device)
    return qlin

def _extract_layer_expert_id(name: str):
    p = name.split(".")
    L, E = None, None
    try:
        L = int(p[p.index("layers")+1])
        E = int(p[p.index("experts")+1])
        return L, E
    except:
        if L is not None:
            return L, None
        else:
            return None, None
        
        
def _as_tensor_1d(x, device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.to(device=device)
    else:
        t = torch.tensor(x, device=device)
    return t

def _is_load_in_4bit(qcfg) -> bool:
    if qcfg is None:
        return False
    if getattr(qcfg, "load_in_4bit", False):
        return True
    if isinstance(qcfg, dict) and qcfg.get("load_in_4bit", False):
        return True
    return False


def _n_params_linear(mod: nn.Module, load_in_4bit: bool) -> int:
    if not hasattr(mod, "weight"):
        return 0
    w = mod.weight
    n = int(w.numel() * 2) if load_in_4bit else int(w.numel())
    b = getattr(mod, "bias", None)
    if b is not None:
        n += int(b.numel())
    return n


def _topk_idx(scores_1d: torch.Tensor, k: int) -> torch.Tensor:
    assert scores_1d.dim() == 1
    k = int(k)
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores_1d.device)
    k = min(k, scores_1d.numel())
    _, idx = torch.topk(scores_1d, k=k, largest=True, sorted=True)
    return idx.to(torch.long)


def _round_down_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return int(x)
    return int((x // m) * m)


def _get_attn_projs(attn: nn.Module) -> Dict[str, nn.Module]:
    out = {}
    for k in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "Wqkv", "wo"]:
        if hasattr(attn, k):
            out[k] = getattr(attn, k)
    if "Wqkv" in out and "qkv_proj" not in out:
        out["qkv_proj"] = out["Wqkv"]
    if "wo" in out and "o_proj" not in out:
        out["o_proj"] = out["wo"]
    return out




def _infer_num_heads_kv_heads_head_dim(attn: nn.Module, hidden_size: int) -> Tuple[int, int, int]:
    """
    Best-effort inference that stays correct after real-slim pruning.
    Priority:
      1) explicit attn attrs
      2) weight shapes (q_proj/k_proj/v_proj or qkv_proj)
      3) fallback to hidden_size heuristics
    Assumes fused qkv layout is [Q | K | V] along output dim.
    """
    projs = _get_attn_projs(attn)

    num_heads = getattr(attn, "num_heads", None) or getattr(attn, "num_attention_heads", None) or getattr(attn, "n_heads", None)
    num_kv_heads = getattr(attn, "num_key_value_heads", None) or getattr(attn, "num_kv_heads", None) or getattr(attn, "n_kv_heads", None)
    head_dim = getattr(attn, "head_dim", None) or getattr(attn, "head_size", None)

    if "q_proj" in projs and hasattr(projs["q_proj"], "out_features"):
        q_out = int(projs["q_proj"].out_features)
        if head_dim is None and num_heads is not None:
            hd = q_out // int(num_heads)
            if hd * int(num_heads) != q_out:
                raise RuntimeError(f"Cannot infer head_dim from q_out={q_out} and num_heads={num_heads}.")
            head_dim = hd

        if head_dim is not None and num_heads is None:
            if q_out % int(head_dim) != 0:
                raise RuntimeError(f"q_out={q_out} not divisible by head_dim={head_dim}.")
            num_heads = q_out // int(head_dim)

    if head_dim is not None and num_kv_heads is None and "k_proj" in projs and hasattr(projs["k_proj"], "out_features"):
        k_out = int(projs["k_proj"].out_features)
        if k_out % int(head_dim) != 0:
            raise RuntimeError(f"k_out={k_out} not divisible by head_dim={head_dim}.")
        num_kv_heads = k_out // int(head_dim)

    if ("qkv_proj" in projs) and hasattr(projs["qkv_proj"], "out_features"):
        qkv_out = int(projs["qkv_proj"].out_features)

        if head_dim is None and (num_heads is not None) and (num_kv_heads is not None):
            denom = int(num_heads) + 2 * int(num_kv_heads)
            if qkv_out % denom != 0:
                raise RuntimeError(
                    f"Cannot infer head_dim from qkv_out={qkv_out}, num_heads={num_heads}, num_kv_heads={num_kv_heads}."
                )
            head_dim = qkv_out // denom

        if head_dim is not None:
            total = qkv_out // int(head_dim)
            if total * int(head_dim) != qkv_out:
                raise RuntimeError(f"qkv_out={qkv_out} not divisible by head_dim={head_dim}.")

            if num_heads is not None and num_kv_heads is None:
                rem = total - int(num_heads)
                if rem % 2 != 0 or rem < 0:
                    raise RuntimeError(f"Invalid fused dims: total={total}, num_heads={num_heads}.")
                num_kv_heads = rem // 2

            elif num_heads is None and num_kv_heads is not None:
                num_heads = total - 2 * int(num_kv_heads)
                if num_heads <= 0:
                    raise RuntimeError(f"Invalid fused dims: total={total}, num_kv_heads={num_kv_heads} -> num_heads={num_heads}.")

            elif num_heads is None and num_kv_heads is None:
                if total % 3 != 0:
                    raise RuntimeError(f"Cannot infer fused heads from total={total} without attrs.")
                num_heads = total // 3
                num_kv_heads = int(num_heads)

    if num_heads is None:
        raise RuntimeError("Cannot infer num_heads from attn attrs or projection shapes.")
    if head_dim is None:
        head_dim = hidden_size // int(num_heads)
        if int(num_heads) * int(head_dim) != int(hidden_size):
            raise RuntimeError(f"Cannot infer head_dim: hidden_size={hidden_size} not divisible by num_heads={num_heads}.")
    if num_kv_heads is None:
        num_kv_heads = int(num_heads)

    return int(num_heads), int(num_kv_heads), int(head_dim)
