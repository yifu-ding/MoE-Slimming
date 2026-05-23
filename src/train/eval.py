import os
import json
import torch

# Base modules
from src.base.models import get_model
from src.base.shared_utils import log_memory_usage, eval_dispatch, test_throughput, _print
from src.base.argparser import parse_args
# Training utilities
from src.train.utils import load_adapter_with_remap

# Pruning modules
from src.prune.generate import generate_masks
from src.prune.apply.slimming import build_real_slim_model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main(args, model, tokenizer):
    
    model.eval()
    torch.set_float32_matmul_precision("high")

    total_params_initial = count_params(model)
    prune_ratio = args.prune_kwargs['prune_ratio']
    
    _print(f"[Step 0] Initial model parameters: {total_params_initial:,}")
    
    if args.resume_path:
        _print(f"\n[Step 1] Load LoRA adapter from: {args.resume_path}")
        
        model = load_adapter_with_remap(model, args.resume_path)
        
        cnt_step = "unknown"
        state_file_pt = os.path.join(args.resume_path, "training_state.pt")
        state_file_json = os.path.join(args.resume_path, "trainer_state.json")

        if os.path.exists(state_file_pt):
            state_payload = torch.load(state_file_pt, map_location=args.device)
            cnt_step = int(state_payload.get("global_step", 0))
        elif os.path.exists(state_file_json):
            try:
                with open(state_file_json, "r", encoding="utf-8") as f:
                    state_payload = json.load(f)
                if isinstance(state_payload, dict):
                    global_step = state_payload.get("global_step")
                    if global_step is None and isinstance(state_payload.get("log_history"), list):
                        for record in reversed(state_payload["log_history"]):
                            if "step" in record:
                                global_step = record["step"]
                                break
                    if global_step is not None:
                        cnt_step = int(global_step)
            except (json.JSONDecodeError, ValueError):
                pass

        _print(f"[Step 1] ✅ LoRA adapter loaded (training step: {cnt_step})")
    else:
        _print(f"\n[Step 1] Skip LoRA loading (no resume_path)")
    
    if args.resume_path:
        model = model.merge_and_unload()
        total_params_after_merge = count_params(model)
        _print(f"[Step 2] ✅ LoRA merged to base model")
        _print(f"  - Merge parameters: {total_params_after_merge:,}")
    else:
        _print(f"\n[Step 2] Skip LoRA merge (no resume_path)")
        total_params_after_merge = total_params_initial
    
    if prune_ratio > 0:
        mask_result = generate_masks(
            scores_dir=args.scores_dir,
            mask_dir=args.mask_dir,
            prune_kwargs=args.prune_kwargs,
            device=args.device,
            verbose=True
        )
        _print(f"[Step 3] ✅ Masks generated")
    else:
        _print(f"\n[Step 3] Skip mask generation (prune_ratio=0)")
    
    if prune_ratio > 0:
        model = build_real_slim_model(
            model,
            mask=mask_result,                 # [L, E, I] bool
            shrink_gate=args.shrink_gate,
            add_hooks=False,
            verbose=True
        )
        _print(f"[Step 4] ✅ Real slim model built")
        
    total_params_after_slim = count_params(model)
    _print("\n" + "=" * 80)
    _print("Parameter statistics:")
    _print(f"  - Initial parameters: {total_params_initial:,}")
    if args.resume_path:
        _print(f"  - LoRA merge parameters: {total_params_after_merge:,}")
    _print(f"  - Final parameters: {total_params_after_slim:,}")
    if args.resume_path and total_params_after_merge != total_params_initial:
        _print(f"  - LoRA merge parameters change: {total_params_after_merge - total_params_initial:+,} ({100 * (total_params_after_merge - total_params_initial) / total_params_initial:+.2f}%)")
    overall_pruned = total_params_after_merge - total_params_after_slim
    _print(f"  - Pruned pruning parameters: {overall_pruned:,}")
    overall_prune_ratio = 100 * overall_pruned / total_params_after_merge
    _print("-" * 80)
    _print(f"  - Overall pruning ratio (relative to merged): {overall_prune_ratio:.2f}%")
    _print("=" * 80)
  
    if args.test_speed and (args.real_slim or prune_ratio == 0):
        _print(f"\n[Step 5] Test throughput...")
        res = test_throughput(
            model, 
            tokenizer, 
            args.device, 
            batch_size=4, 
            prompt_len=512, 
            gen_len=128
        )
        _print(f"[Step 5] ✅ Throughput = {res}")
        log_memory_usage(tag=f"real_slim: {args.test_speed}")

    _print(f"\n[Step 6] Start evaluation...")
    results = eval_dispatch(args, model, tokenizer, verbose=True)
    _print(f"[Step 6] ✅ Evaluation results: {results}")
    

if __name__ == "__main__":
    args = parse_args()

    model, tokenizer = get_model(args)
    main(args, model, tokenizer)
