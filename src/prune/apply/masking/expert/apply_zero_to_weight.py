import torch
from src.base.shared_utils.safe_isinstance import _get_mlp_block, _get_experts

@torch.no_grad()
def apply_zero_to_weight(block, inter_mask: torch.Tensor):
    experts = _get_experts(_get_mlp_block(block))
    E = len(experts)

    for eid in range(E):
        e = experts[eid]
        keep_I = inter_mask[eid]

        drop_I = ~keep_I

        for proj_name in ["gate_proj", "up_proj"]:
            proj = getattr(e, proj_name)
            W = proj.weight

            if drop_I.any():
                W[drop_I, :] = 0
                if getattr(proj, "bias", None) is not None:
                    proj.bias[drop_I] = 0

        down = e.down_proj
        Wd = down.weight
        if drop_I.any():
            Wd[:, drop_I] = 0
