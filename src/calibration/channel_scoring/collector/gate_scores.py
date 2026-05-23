from torch.nn import functional as F

def collect_gate_scores(cnt_mlp, ema=0.9):
    gate = cnt_mlp.gate
    gate_output = gate.saved_output
    gate.saved_output = None
    gate_grad = gate.saved_grad_out
    gate.saved_grad_out = None
    
    if gate_output is not None and gate_grad is not None:
        gate_saliency = (gate_output * gate_grad).abs().detach()
        gate_saliency = gate_saliency.mean(dim=0)
        
        if hasattr(gate, 'saliency') and gate.saliency is not None:
            gate.saliency.mul_(ema).add_(gate_saliency, alpha=1.0 - ema)
        else:
            gate.saliency = gate_saliency

        gate_grad = gate_grad.abs().detach().mean(dim=0)
        if hasattr(gate, 'gate_grad') and gate.gate_grad is not None:
            gate.gate_grad.mul_(ema).add_(gate_grad, alpha=1.0 - ema)
        else:
            gate.gate_grad = gate_grad

        gate_output = gate_output.abs().detach().mean(dim=0)
        if hasattr(gate, 'gate_output') and gate.gate_output is not None:
            gate.gate_output.mul_(ema).add_(gate_output, alpha=1.0 - ema)
        else:
            gate.gate_output = gate_output

    elif gate.saved_input is not None:
        gate_input = gate.saved_input
        gate = cnt_mlp.gate
        _w = gate.weight
        _b = gate.bias if hasattr(gate, 'bias') else None
        logits = F.linear(gate_input, _w, _b)
        logits = logits.abs().detach().mean(dim=(0,1))
        if hasattr(gate, 'gate_output') and gate.gate_output is not None:
            gate.gate_output.mul_(ema).add_(logits, alpha=1.0 - ema)
        else:
            gate.gate_output = logits