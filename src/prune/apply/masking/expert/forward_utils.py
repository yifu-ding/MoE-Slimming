import torch
import types

__all__ = [
    "forward_with_mask",
    "forward_with_alpha",
    "ori_moe_mlp_forward",
    "_patch_block_alpha_if_needed",
]


def _patch_block_alpha_if_needed(block, E: int, args) -> None:
    if not (hasattr(args, "alpha") and args.alpha is not None):
        return
    if not hasattr(block, "mlp"):
        return
    mlp = block.mlp
    if not hasattr(mlp, "experts"):
        return
    for eid in range(E):
        expert = mlp.experts[eid]
        expert.forward = types.MethodType(forward_with_alpha(alpha=args.alpha), expert)

def forward_with_alpha(alpha=0.9):
    def _forward_with_alpha(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        down_proj = self.down_proj(x) * alpha 
        return down_proj
    return _forward_with_alpha

def ori_moe_mlp_forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def _mask_ste(m_hard, m_soft):
    return (m_hard - m_soft).detach() + m_soft

def _generate_hard_mask(scores: torch.Tensor, k: int, total_dim: int, dtype):
    k = int(k)
    if k <= 0:
        return torch.zeros((total_dim,), dtype=dtype, device=scores.device)
    if k >= total_dim:
        return torch.ones((total_dim,), dtype=dtype, device=scores.device)
    topk = torch.topk(scores, k=k, dim=-1).indices
    m = torch.zeros((total_dim,), dtype=dtype, device=scores.device)
    m.scatter_(0, topk, 1)
    return m

def forward_with_mask(self, x):
    g = torch.sigmoid(self.mask_logits / self.tau.clamp_min(1e-4))
    m_soft = g.to(self.down_proj.weight.dtype)
    m_hard = _generate_hard_mask(g, k=int(self.dim_to_keep.item()),
                                total_dim=self.intermediate_size,
                                dtype=self.down_proj.weight.dtype)
    mask = _mask_ste(m_hard, m_soft)

    a = self.act_fn(self.gate_proj(x))
    b = self.up_proj(x)
    h = (a * b) * mask
    y = self.down_proj(h)
    return y
