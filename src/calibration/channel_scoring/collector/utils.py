import torch
import torch.nn as nn


def weight_rms(weight: torch.Tensor, channel_dim: int = 0) -> torch.Tensor:
    x = weight.detach()
    reduce_dims = [d for d in range(x.ndim) if d != channel_dim]
    x2 = x.pow(2).sum(dim=reduce_dims)
    return x2.sqrt()

def channel_rms(act: torch.Tensor) -> torch.Tensor:
    x = act.detach()
    dims = tuple(range(x.dim() - 1))
    x2 = x.pow(2).sum(dim=dims)
    return x2.sqrt()  

def wa_score(weight: torch.Tensor, activation: torch.Tensor, sum_dim=0) -> torch.Tensor:
    return (weight.abs() * activation.unsqueeze(0)).sum(dim=sum_dim)


def snip_score(
    weight: torch.Tensor,
    grad: torch.Tensor,
    channel_dim: int = 0,
) -> torch.Tensor:
    score = (weight * grad).abs()
    reduce_dims = [d for d in range(score.ndim) if d != channel_dim]
    return score.sum(dim=reduce_dims)


def token_contrib(
    g: torch.Tensor,
    z: torch.Tensor,
    trim_head: float = 0.01,  # Clip top p% by absolute value
    trim_tail: float = 0.00,  # Not used anymore, kept for compatibility
) -> torch.Tensor:
    """
    g, z: [..., I]
    Returns: [I], clipped mean contribution for each channel

    What it does:
    - c = g * z
    - Compute (1 - trim_head) quantile q_high for each channel by absolute value
    - Clip each channel with sign to [-q_high, q_high]
    - Then take mean over token dimension
    """
    assert g.shape == z.shape, "g and z must have the same shape"
    I = z.size(-1)
    orig_dtype = z.dtype

    gz = g * z
    gz_flat = gz.view(-1, I).to(torch.float32)

    N = gz_flat.size(0)
    if N <= 2 or trim_head <= 0.0:
        return gz_flat.mean(dim=0).to(orig_dtype)

    trim_head = float(max(0.0, min(trim_head, 0.49)))
    abs_gz = gz_flat.abs()
    q_high = torch.quantile(abs_gz, 1.0 - trim_head, dim=0, keepdim=True)

    q_high = torch.clamp(q_high, min=1e-25)
    clipped = torch.clamp(gz_flat, min=-q_high, max=q_high)
    contrib_mean = clipped.mean(dim=0)

    return contrib_mean.to(orig_dtype)


def channel_saliency(act: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    s = (act * grad).abs().detach()
    dims = tuple[int, ...](range(s.dim() - 1))
    return s.mean(dim=dims)    

def safe_add_with_ema(target, ema, value, key=None):
    if isinstance(value, torch.Tensor):
        value = value.detach()

    def ema_update(old, new):
        if isinstance(new, torch.Tensor):
            if old is None:
                return new.clone()
            old.mul_(ema).add_(new, alpha=1.0 - ema)
            return old
        else:
            return new if old is None else old * ema + new * (1.0 - ema)

    if key is None:
        return ema_update(target, value)

    assert isinstance(target, nn.Module), f"target must be nn.Module, got {type(target)}"
    old = getattr(target, key, None)
    setattr(target, key, ema_update(old, value))



def compute_token_contrib_I(down_input: torch.Tensor = None, 
                            down_grad: torch.Tensor = None, 
                            up_output: torch.Tensor = None,
                            up_out_grad: torch.Tensor = None, 
                            gate_output: torch.Tensor = None,
                            gate_grad: torch.Tensor = None):
    down_token_contrib = token_contrib(down_grad, down_input)
    up_token_contrib = token_contrib(up_out_grad, up_output)
    gate_token_contrib = token_contrib(gate_grad, gate_output)
    return (down_token_contrib + up_token_contrib + gate_token_contrib) / 3.0

def compute_grad_I(down_grad: torch.Tensor = None, 
                    up_out_grad: torch.Tensor = None, 
                    gate_grad: torch.Tensor = None):

    down_grad = channel_rms(down_grad)
    up_out_grad = channel_rms(up_out_grad)
    gate_grad = channel_rms(gate_grad)
    grad_mean = (down_grad + up_out_grad + gate_grad) / 3.0
    return grad_mean, down_grad, up_out_grad, gate_grad
        
def compute_saliency_I(down_input: torch.Tensor = None, 
                        down_grad: torch.Tensor = None,
                        up_output: torch.Tensor = None, 
                        up_out_grad: torch.Tensor = None,
                        gate_output: torch.Tensor = None,
                        gate_grad: torch.Tensor = None):
    down_sal = channel_saliency(down_input, down_grad)
    up_sal = channel_saliency(up_output, up_out_grad)
    gate_sal = channel_saliency(gate_output, gate_grad)
    assert down_sal.shape == up_sal.shape == gate_sal.shape
    return (down_sal + up_sal + gate_sal) / 3.0
            
def compute_activation_I(down_input: torch.Tensor = None, 
                          up_output: torch.Tensor = None,
                          gate_output: torch.Tensor = None):
    down_act = channel_rms(down_input)
    up_act = channel_rms(up_output)
    gate_act = channel_rms(gate_output)
    assert down_act.shape == up_act.shape == gate_act.shape
    return (down_act + up_act + gate_act) / 3.0, down_act

def compute_wa_I(W_down: torch.Tensor = None, 
                 W_up: torch.Tensor = None, 
                 W_gate: torch.Tensor = None, 
                 down_ch_act: torch.Tensor = None, 
                 up_input: torch.Tensor = None, 
                 gate_input: torch.Tensor = None):
    assert W_down.size(1) == W_up.size(0) == W_gate.size(0)
    up_input = channel_rms(up_input)
    gate_input = channel_rms(gate_input)
    wa_down = wa_score(W_down, down_ch_act, sum_dim=0)
    wa_up = wa_score(W_up, up_input, sum_dim=1)
    wa_gate = wa_score(W_gate, gate_input, sum_dim=1)
    return (wa_down + wa_up + wa_gate) / 3.0


def compute_wg_I(W_down: torch.Tensor = None, 
                 W_up: torch.Tensor = None, 
                 W_gate: torch.Tensor = None, 
                 W_down_grad: torch.Tensor = None, 
                 W_up_grad: torch.Tensor = None, 
                 W_gate_grad: torch.Tensor = None):
    assert W_down.size(1) == W_up.size(0) == W_gate.size(0)
    W_down_grad = W_down_grad.detach()
    W_up_grad = W_up_grad.detach()
    W_gate_grad = W_gate_grad.detach()
    snip_score_down = snip_score(W_down, W_down_grad, channel_dim=1)
    snip_score_up = snip_score(W_up, W_up_grad, channel_dim=0)
    snip_score_gate = snip_score(W_gate, W_gate_grad, channel_dim=0)
    return (snip_score_down + snip_score_up + snip_score_gate) / 3.0


def masked_mean_bs(x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """
    x: [B, S, H]
    attn_mask: [B, S] (1 for valid token, 0 for pad)
    return: [H]
    """
    mask = attn_mask.to(x.device).float()[:, :, None]
    x_masked = x * mask
    denom = mask.sum().clamp_min(1.0)
    return x_masked.sum(dim=(0, 1)) / denom
