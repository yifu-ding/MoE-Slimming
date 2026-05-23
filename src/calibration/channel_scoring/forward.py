import torch
from torch import nn
from tqdm import tqdm

from src.base.shared_utils.hook_manager import *
from src.calibration.channel_scoring.collector import *
from src.base.shared_utils import to_device_dtype, angle_loss

__all__ = [     
    "block_forward",
]

def add_hooks(model: nn.Module, block: nn.Module, layer_idx: int) -> list:
    hooks = []
    hooks.extend(add_block_hook_for_model(model, layer_idx=layer_idx))
    hooks.extend(add_down_proj_hook(block.mlp))
    hooks.extend(add_up_proj_hook(block.mlp))
    hooks.extend(add_gate_proj_hook(block.mlp))
    hooks.extend(add_gate_hook(block.mlp))
    return hooks


def clear_hooks(block: nn.Module):
    mlp = block.mlp
    for expert in mlp.experts:
        expert.up_proj.saved_input = None
        expert.up_proj.saved_output = None
        expert.up_proj.saved_grad_in = None
        expert.up_proj.saved_grad_out = None
        expert.gate_proj.saved_input = None
        expert.gate_proj.saved_output = None
        expert.gate_proj.saved_grad_in = None
        expert.gate_proj.saved_grad_out = None
        expert.down_proj.saved_input = None
        expert.down_proj.saved_output = None
        expert.down_proj.saved_grad_in = None
        expert.down_proj.saved_grad_out = None
    mlp.gate.saved_input = None
    mlp.gate.saved_grad_in = None
    mlp.gate.saved_output = None
    mlp.gate.saved_grad_out = None
    mlp.saved_input = None
    mlp.saved_output = None

    
def block_forward(model: nn.Module, 
                  cnt_block: nn.Module, 
                  layer_idx: int, 
                  dataset: list, 
                  tokenizer, 
                  max_seq_length: int,
                  saliency_ema: float,
                  batch_size: int = 128, 
                  calib_batches: int = 10,
                  loss_fn: str = "rel_l2",
                  device: str = "cuda", 
                  dtype=torch.float32,
                  verbose: bool = False):

    model.eval().to(device)
    hooks = add_hooks(model, cnt_block, layer_idx)
    
    total_loss = 0.0

    for n_batches, batch_idx in enumerate(
            tqdm(range(0, len(dataset), batch_size),
                desc=f"Calibrating L{layer_idx}",
                total=calib_batches, 
                disable=not verbose, 
                leave=False),
            start=1,
    ):
        if n_batches > calib_batches:
            break

        raw = dataset[batch_idx : batch_idx + batch_size]
        batch_texts = []
        for x in raw:
            if x is None:
                continue
            if not isinstance(x, str):
                x = str(x)
            x = x.strip()
            if x:
                batch_texts.append(x)

        if not batch_texts:
            continue

        enc = tokenizer(
            batch_texts,
            max_length=max_seq_length,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        attn_mask = enc["attention_mask"].to(device, non_blocking=True)

        with torch.no_grad():  
            teacher_block = model.model.layers[layer_idx]
            try:
                _ = model(**inputs, use_cache=False)
            except ValueError:
                in_args = teacher_block.saved_input_args
                in_kwargs = teacher_block.saved_input_kwargs
                y = teacher_block.saved_output

                teacher_block.saved_input_args = None
                teacher_block.saved_input_kwargs = None
                teacher_block.saved_output = None
                
        in_args = to_device_dtype(in_args, device=device)
        in_kwargs = to_device_dtype(in_kwargs, device=device)
       
        y = y[0] if isinstance(y, (tuple, list)) else y
        y = to_device_dtype(y, device=device)
       
        m = attn_mask.to(device) 

        def set_training_true(module):
            module.training = True
            for _child in getattr(module, 'children', lambda: [])():
                set_training_true(_child)
        set_training_true(cnt_block)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out = cnt_block(*in_args, **in_kwargs)
            pred = out[0] if isinstance(out, (tuple, list)) else out

            if loss_fn == "l2":
                l_l2 = (pred.float() - y.float()).pow(2).mean(dim=-1)  # [N]
                loss_vec = l_l2 * m
            if loss_fn == "rel_l2":
                pred_f = pred.float().view(-1, pred.size(-1))  # [N, D]
                y_f    = y.float().view(-1, y.size(-1))        # [N, D]
                m_f    = m.view(-1).float()                    # [N]

                diff2 = (pred_f - y_f).pow(2).sum(dim=-1)      # [N]
                base2 = y_f.pow(2).sum(dim=-1)                 # [N]

                eps = 1e-6
                l_rel = diff2 / (base2 + eps)                # [N] relative L2^2
                loss_vec = l_rel * m_f
            elif loss_fn == "cosine":
                l_cos = angle_loss(pred, y)
                loss_vec = l_cos * m

            loss_sum = loss_vec.sum()
        
        cnt_block.zero_grad(set_to_none=True)   
        loss_sum.backward()
        total_loss += loss_sum.detach().float().item()

        collect_scores_attn_mlp(cnt_block, 
                                ema=saliency_ema, 
                                attn_mask=attn_mask)
        collect_gate_scores(cnt_block.mlp, ema=saliency_ema)
        
    clear_hooks(cnt_block)
    remove_hooks(hooks)
    hooks = []
    
    return total_loss / (calib_batches * batch_size)