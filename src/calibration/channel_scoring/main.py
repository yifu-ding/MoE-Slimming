import os
import torch
import argparse
from copy import deepcopy
from tqdm import tqdm

from src.base.models import get_model
from src.base.datasets import load_datasets
from src.base.shared_utils import format_name, _layer_norm, _print
from src.base.shared_utils.safe_isinstance import (
    _get_num_experts, 
    _get_moe_intermediate_size,
    _get_num_hidden_layers,
    _is_moe_block,
)

from src.prune.apply.masking.expert.forward_utils import _patch_block_alpha_if_needed
from src.calibration.channel_scoring.forward import block_forward

def main(args, model, tokenizer, output_dir, calib_dataset, verbose=False):
    
    model.eval()
    torch.set_float32_matmul_precision("high")

    E = _get_num_experts(model)
    I = _get_moe_intermediate_size(model)
    L = _get_num_hidden_layers(model)
    
    _print(f"num_experts: {E}, num_intermediate_size: {I}, num_hidden_layers: {L}")
    
    calib_dataset = load_datasets(calib_dataset, tokenizer, max_samples=args.calib_batches * args.batch_size)

    gate_scores = {"saliency": {}, "out": {}, "grad": {}, "usage": {}}
    expert_scores = {"activation": {}, "saliency": {}, "wa": {}, "grad": {}, "token_contrib": {}, "expert_out_token_contrib": {}, "wg": {}, "weight": {}}

    layerwise_loss = []
    
    for layer_idx in tqdm(range(L), desc="Processing"):
        teacher_block = model.model.layers[layer_idx]
        copied_block = deepcopy(teacher_block).to(device=args.device, dtype=args.dtype)
        copied_mlp = copied_block.mlp
        if not _is_moe_block(copied_mlp):
            continue

        if not hasattr(args, 'alpha'):
            args.alpha = 0.9
        _patch_block_alpha_if_needed(copied_block, E=E, args=args)

        total_loss = block_forward(model, 
                      copied_block, 
                      layer_idx,
                      calib_dataset, 
                      tokenizer, 
                      max_seq_length=args.max_seq_length,
                      saliency_ema=0.9,
                      batch_size=args.batch_size, 
                      calib_batches=args.calib_batches,  
                      device=args.device,
                      dtype=args.dtype,
                      verbose=verbose)
        
        layerwise_loss.append(total_loss)
        gate_saliency = getattr(copied_mlp.gate, 'saliency', None)
        gate_saliency = _layer_norm(gate_saliency) 
        gate_scores["saliency"][layer_idx] = gate_saliency[:, None] if gate_saliency is not None else None
        gate_output = getattr(copied_mlp.gate, 'gate_output', None)
        gate_output = torch.softmax(gate_output, dim=0)[:, None]  if gate_output is not None else None
        gate_scores["out"][layer_idx] = gate_output
        gate_grad = getattr(copied_mlp.gate, 'gate_grad', None)
        gate_grad = _layer_norm(gate_grad)
        gate_scores["grad"][layer_idx] = gate_grad[:, None] if gate_grad is not None else None
        
        copied_mlp.gate.saliency = None
        copied_mlp.gate.gate_output = None
        copied_mlp.gate.gate_input = None
        copied_mlp.gate.gate_grad_in = None
        copied_mlp.gate.gate_grad_out = None

        def _get_score_for_expert(mode, expert):    
            if mode == "expert_out_token_contrib" or mode == "usage":
                if not hasattr(expert, mode) or getattr(expert, mode) is None:
                    return torch.zeros((1,), dtype=torch.float32, device=args.device)
            elif "H_" in mode: 
                if not hasattr(expert, mode) or getattr(expert, mode) is None:
                    return torch.zeros((H, ), dtype=torch.float32, device=args.device)
            else:
                if not hasattr(expert, mode) or getattr(expert, mode) is None:
                    return torch.randn(I, dtype=torch.float32, device=args.device) * 1e-25
            return getattr(expert, mode).clone().detach() if isinstance(getattr(expert, mode), torch.Tensor) else getattr(expert, mode)
        
        expert_scores["activation"][layer_idx] = torch.zeros((E, I), dtype=torch.float32, device=args.device)
        expert_scores["saliency"][layer_idx] = torch.zeros((E, I), dtype=torch.float32, device=args.device)
        expert_scores["wa"][layer_idx] = torch.zeros((E, I), dtype=torch.float32, device=args.device)
        expert_scores["grad"][layer_idx] = torch.zeros((E, I), dtype=torch.float32, device=args.device)
        expert_scores["weight"][layer_idx] = torch.zeros((E, I), dtype=torch.float32, device=args.device)
        expert_scores["token_contrib"][layer_idx] = torch.zeros((E, I), dtype=torch.float32, device=args.device)
        expert_scores["wg"][layer_idx] = torch.zeros((E, I), dtype=torch.float32, device=args.device)
        expert_scores["expert_out_token_contrib"][layer_idx] = torch.zeros((E,), dtype=torch.float32, device=args.device)
        gate_scores["usage"][layer_idx] = torch.zeros((E,), dtype=torch.float32, device=args.device)

        for eid in range(E):
            expert = copied_mlp.experts[eid]
            activation = _get_score_for_expert("activation", expert)
            saliency = _get_score_for_expert("saliency", expert)
            wa = _get_score_for_expert("wa", expert)
            wg = _get_score_for_expert("wg", expert)
            grad = _get_score_for_expert("grad", expert)
            weight = _get_score_for_expert("weight", expert)
            token_contrib = _get_score_for_expert("token_contrib", expert)
            expert_out_token_contrib = _get_score_for_expert("expert_out_token_contrib", expert)
            usage = _get_score_for_expert("usage", expert)

            expert_scores["activation"][layer_idx][eid] = activation
            expert_scores["saliency"][layer_idx][eid] = saliency
            expert_scores["wa"][layer_idx][eid] = wa
            expert_scores["wg"][layer_idx][eid] = wg
            expert_scores["grad"][layer_idx][eid] = grad
            expert_scores["weight"][layer_idx][eid] = weight
            expert_scores["token_contrib"][layer_idx][eid] = token_contrib
            expert_scores["expert_out_token_contrib"][layer_idx][eid] = expert_out_token_contrib
            gate_scores["usage"][layer_idx][eid] = usage

        expert_scores["activation"][layer_idx] = _layer_norm(expert_scores["activation"][layer_idx])
        expert_scores["saliency"][layer_idx] = _layer_norm(expert_scores["saliency"][layer_idx])
        expert_scores["wa"][layer_idx] = _layer_norm(expert_scores["wa"][layer_idx])
        expert_scores["wg"][layer_idx] = _layer_norm(expert_scores["wg"][layer_idx])
        expert_scores["grad"][layer_idx] = _layer_norm(expert_scores["grad"][layer_idx])
        expert_scores["weight"][layer_idx] = _layer_norm(expert_scores["weight"][layer_idx])
        expert_scores["token_contrib"][layer_idx] = _layer_norm(expert_scores["token_contrib"][layer_idx])
        expert_scores["expert_out_token_contrib"][layer_idx] = expert_scores["expert_out_token_contrib"][layer_idx]
       
    torch.save(gate_scores, os.path.join(output_dir, "gate_scores.pth"))
    torch.save(expert_scores, os.path.join(output_dir, "expert_scores.pth"))
    layerwise_loss = torch.tensor(layerwise_loss, dtype=torch.float32, device=args.device)
    torch.save(layerwise_loss, os.path.join(output_dir, "layerwise_loss.pth"))
    
    _print(f"gate_scores.pth: for gate importance score, keys: {gate_scores.keys()}")
    _print(f"expert_scores.pth: for intermediate dim score, keys: {expert_scores.keys()}")
    _print(f"layerwise_loss.pth: for layerwise loss")
    _print("=" * 80)
    _print(f"✅ Scores saved to {output_dir}")
    _print("Usage: ")
    _print(f"\t 1. Set scores_dir={output_dir} in `configs/train/model.yaml` for mask generation")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-name-or-path", type=str, required=True)
    p.add_argument("--calib-datasets", type=str, nargs="+", required=True)
    p.add_argument("--calib-batches", type=int, default=200)  # 200 for full calibration
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16", "fp32", "fp16", "bf16"])
    p.add_argument("--trust-remote-code", action="store_true", default=False)
    p.add_argument("--output-dir", type=str, default="/path/to/output/dir")
    p.add_argument("--verbose", action="store_true", default=True)

    p.add_argument("--load-in-4bit", action="store_true", default=False)
    p.add_argument("--load-in-8bit", action="store_true", default=False)
    p.add_argument("--test-only", action="store_true", default=False)
    args = p.parse_args()
    
    dtype_map = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16
    }
    args.dtype = dtype_map.get(args.dtype, torch.bfloat16)
    
    model, tokenizer = get_model(args)
    _print(args)
    
    for calib_dataset in args.calib_datasets:
        model_name = format_name(args.model_name_or_path)
        output_dir = os.path.join(args.output_dir, model_name, format_name(calib_dataset), "scores")
        os.makedirs(output_dir, exist_ok=True)
        main(args, model, tokenizer, output_dir, calib_dataset, verbose=args.verbose)
