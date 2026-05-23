import os

# Base modules
from src.base.argparser import parse_args
from src.base.models import get_model, adapt_hf_model
from src.base.datasets.data_preprocessor import build_datasets
from src.base.shared_utils import seed_all

# Training modules
from src.train.callbacks import *
from src.train.trainer import *
from src.train.utils import (
    DataCollatorForCausalLM,
    prepare_model_for_training,
    build_training_args,
    resolve_resume_checkpoint,
    load_lora_adapter_with_compatibility,
)
from src.base.shared_utils import _print

# Pruning modules
from src.prune.apply.masking import mask_expert
from src.prune.generate import generate_masks

def main(args):
    seed_all(args.seed)

    model, tokenizer = get_model(args, device_map=None)
    
    mask_result = generate_masks(
        scores_dir=args.scores_dir,
        mask_dir=args.mask_dir,
        prune_kwargs=args.prune_kwargs,
        device=args.device,
        verbose=True
    )
   
    intermediate_masks = mask_result['intermediate_masks']
    
    _print("\n[Step 1] Process LoRA...")
    
    if args.resume_path:
        model = load_lora_adapter_with_compatibility(model, args.resume_path, adapter_name="default", verbose=True)
    else:
        _print(f"[peft] initialize expert LoRA, enable_gate_lora: {args.enable_gate_lora}, enable_attn_lora: {args.enable_attn_lora}")
        model = adapt_hf_model(args, model, enable_gate_lora=args.enable_gate_lora, enable_attn_lora=args.enable_attn_lora)
   
    _print("\n[Step 3] Add experts gradual mask (intermediate)...")
    mask_expert(
        model, 
        intermediate_masks=intermediate_masks
    )
    _print(f"  - ✅ Experts gradual mask (intermediate) added")

    model = prepare_model_for_training(model, args)
    
    target_device = args.device
    _print(f"[Device] Ensuring all model parameters are on {target_device}...")
    model = model.to(target_device)
    
    devices = {p.device for p in model.parameters()}
    devices.update({b.device for b in model.buffers()})
    _print(f"[Device] Model parameters/buffers devices: {devices}")
    
    _print("\n[Step 4] ✅ Prepare model for training completed")

    train_dataset, eval_dataset = build_datasets(args, tokenizer)
    training_args = build_training_args(args)
  
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        max_len=int(args.max_seq_length),
        pad_to_multiple_of=8,
    )
    _print("\n[Step 5] ✅ Build dataset and training parameters completed")

    trainer = SlimTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            SavePruningArtifactsCallback(masks=mask_result, verbose=True)
        ],
    )
    _print("\n[Step 6] ✅ Build Trainer completed")

    _print("\n[Step 7] Start training...")
    os.makedirs(training_args.output_dir, exist_ok=True)
    resume_ckpt = resolve_resume_checkpoint(args)
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if training_args.eval_strategy != "no":
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    _print("=" * 80)
    _print(f"✅ Trainer completed, checkpoint saved to {training_args.output_dir}")
    _print("Usage: ")
    _print(f"\t 1. Resume training, set the following parameters in config files, e.g. `configs/train/xxx_model_dataset.yaml`:")
    _print(f"\t    Set `resume_path: {training_args.output_dir}`")
    _print(f"\t    Set `mask_dir: {training_args.output_dir}/checkpoint-xxx/masks.pth`")
    _print(f"\t    Set `resume_training: true`")
    _print("\t 2. Eval, set the following parameters in running scripts, e.g. `scripts/eval/xxx_model/xxx.sh`:")
    _print(f"\t    export RESUME_PATH={training_args.output_dir}")
    _print(f"\t    export MASK_DIR={training_args.output_dir}/checkpoint-xxx/masks.pth\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)

