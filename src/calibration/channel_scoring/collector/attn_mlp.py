import torch
from .utils import *
        
def collect_scores_attn_mlp(cnt_block, 
                            ema: float = 0.9, 
                            attn_mask: torch.Tensor = None) -> None:

    for expert in cnt_block.mlp.experts:
        down_input    = expert.down_proj.saved_input
        down_output   = expert.down_proj.saved_output
        down_grad     = expert.down_proj.saved_grad_in  
        down_out_grad = expert.down_proj.saved_grad_out
        expert.down_proj.saved_input    = None
        expert.down_proj.saved_output   = None
        expert.down_proj.saved_grad_in  = None
        expert.down_proj.saved_grad_out = None

        up_input    = expert.up_proj.saved_input
        up_output   = expert.up_proj.saved_output
        up_out_grad = expert.up_proj.saved_grad_out
        expert.up_proj.saved_input    = None
        expert.up_proj.saved_output   = None
        expert.up_proj.saved_grad_in  = None
        expert.up_proj.saved_grad_out = None

        gate_input      = expert.gate_proj.saved_input
        gate_output     = expert.gate_proj.saved_output
        gate_grad       = expert.gate_proj.saved_grad_out
        expert.gate_proj.saved_input    = None
        expert.gate_proj.saved_output   = None
        expert.gate_proj.saved_grad_in  = None
        expert.gate_proj.saved_grad_out = None
        
        W_down, W_up, W_gate = expert.down_proj.weight, expert.up_proj.weight, expert.gate_proj.weight
        
        with torch.no_grad():
            w_norm_mean = (weight_rms(W_down, channel_dim=1) + weight_rms(W_up, channel_dim=0) + weight_rms(W_gate, channel_dim=0)) / 3.0
            safe_add_with_ema(expert, ema, w_norm_mean, "weight")
            
            if down_input is not None:
                wg_mean = compute_wg_I(W_down, W_up, W_gate, W_down.grad, W_up.grad, W_gate.grad)
                safe_add_with_ema(expert, ema, wg_mean, "wg")
            
                token_contrib_I = compute_token_contrib_I(down_input, down_grad, up_output, up_out_grad, gate_output, gate_grad)
                safe_add_with_ema(expert, ema, token_contrib_I, "token_contrib")
                    
                grad_I, down_grad, up_out_grad, gate_grad = compute_grad_I(down_grad, up_out_grad, gate_grad)
                safe_add_with_ema(expert, ema, grad_I, "grad")
                    
                saliency_I = compute_saliency_I(down_input, down_grad, up_output, up_out_grad, gate_output, gate_grad)
                safe_add_with_ema(expert, ema, saliency_I, "saliency")

                act_mean, down_ch_act = compute_activation_I(down_input, up_output, gate_output)
                safe_add_with_ema(expert, ema, act_mean, "activation")
                
                wa_mean = compute_wa_I(W_down, W_up, W_gate, down_ch_act, up_input, gate_input)
                safe_add_with_ema(expert, ema, wa_mean, "wa")
                
                total_tokens = float(attn_mask.sum().item())
                usage = float(down_output.shape[0]) / max(total_tokens, 1.0)
                expert_out_token_contrib = token_contrib(down_out_grad, down_output).sum() * usage
                safe_add_with_ema(expert, ema, expert_out_token_contrib, "expert_out_token_contrib")
                safe_add_with_ema(expert, ema, usage, "usage")
