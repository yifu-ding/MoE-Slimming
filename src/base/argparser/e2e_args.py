from dataclasses import dataclass, field
import argparse
import torch
from typing import Optional, Dict, List, Tuple
import yaml
import numpy as np
import random
import os
import time
from src.base.shared_utils import _print

@dataclass
class E2EArguments:  # Prune & Train Arguments

    # W&B
    use_wandb: bool = field(default=False, metadata={"help": "Whether to use wandb."})
    wandb_project: str = field(default="slimmoe_kd", metadata={"help": "Wandb project name."})
    wandb_log_interval: int = field(default=1, metadata={"help": "Wandb log interval."})
    wandb_name: str = field(default="default", metadata={"help": "Wandb run name."})
    
    # Model and device
    model_name_or_path: str = field(default="Qwen/Qwen3-30B-A3B", metadata={"help": "Model to load"})
    load_in_4bit: Optional[bool] = field(default=None, metadata={"help": "Whether to load model in 4bit."})
    load_in_8bit: Optional[bool] = field(default=None, metadata={"help": "Whether to load model in 8bit."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Whether to trust remote code."})

    debug_mode: bool = field(default=False, metadata={"help": "Whether to use debug mode. "})
    device: str = field(default="cuda", metadata={"help": "Device"})
    dtype: str = field(default="bf16", metadata={"help": "bf16/fp16/fp32"})
    num_workers: int = field(default=0, metadata={"help": "Number of workers for dataloader."})
    seed: int = field(default=42, metadata={"help": "Seed for evaluation."})
    test_only: bool = field(default=False, metadata={"help": "Whether to test only, no training."})  # Used by get_model()
    
    # Calibration settings
    calib_datasets: List[str] = field(default_factory=list, metadata={"help": "Datasets names for calib."})
    calib_batches: int = field(default=10, metadata={"help": "Number of batches to calibrate, collect scores."})

    # lora parameters
    dora_rank: int = field(default=32, metadata={"help": "DoRA / LoRA rank."})
    lora_alpha: int = field(default=128, metadata={"help": "Scaling factor for LoRA / DoRA."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Dropout for LoRA / DoRA."})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA / DoRA. Defaults to gate_proj, up_proj, down_proj."}
    )
    gate_lr_mult: float = field(default=0.01, metadata={"help": "Multiplier for gate learning rate."})
    enable_gate_lora: bool = field(default=False, metadata={"help": "Whether to enable gate LoRA."})
    enable_attn_lora: bool = field(default=False, metadata={"help": "Whether to enable attn LoRA."})
    attn_lora_rank: int = field(default=16, metadata={"help": "Rank for attn LoRA."})
    attn_lora_alpha: int = field(default=64, metadata={"help": "Scaling factor for attn LoRA."})
    attn_lr_mult: float = field(default=0.5, metadata={"help": "Multiplier for attn learning rate."})
    gate_lora_rank: int = field(default=4, metadata={"help": "Rank for gate LoRA."})
    gate_lora_alpha: int = field(default=16, metadata={"help": "Scaling factor for gate LoRA."})

    # router auxiliary loss
    train_router_aux_loss: bool = field(default=False, metadata={"help": "Whether to train router auxiliary loss."})
    router_aux_loss_coef: float = field(default=3e-2, metadata={"help": "Coefficient for router auxiliary loss."})
  
    # Pruning settings
    prune_kwargs: Dict = field(default_factory=dict, metadata={"help": "Keyword arguments for pruning."})
    real_slim: bool = field(default=False, metadata={"help": "Whether to use real slim."})  # Apply actual pruning to the model
    shrink_gate: bool = field(default=False, metadata={"help": "Whether to shrink gate."})  # Prune gate channels

    # Training settings
    n_epochs: int = field(default=1, metadata={"help": "Number of training epochs."})
    batch_size: int = field(default=32, metadata={"help": "Global batch size."})
    lr: float = field(default=4e-4, metadata={"help": "Learning rate for main parameters."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay used by the optimizer."})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Warmup ratio for LR scheduler."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "HF Trainer scheduler type name."})
    grad_accum_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps."})
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer name."})
    max_grad_norm: float = field(default=0.5, metadata={"help": "Max gradient norm."})
    use_ademamix: bool = field(default=False, metadata={"help": "Whether to use AdEMAMix."})  # Can be unstable; use AdamW instead

    # Training schedule
    log_every_n_steps: int = field(default=50, metadata={"help": "Log every N training steps."})
    eval_every_n_steps: int = field(default=50, metadata={"help": "Run eval every N training steps."})
    save_every_n_steps: int = field(default=500, metadata={"help": "Save checkpoint every N training steps."})
    max_steps: Optional[int] = field(default=None, metadata={"help": "Maximum number of training steps."})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Whether to use gradient checkpointing."})
    gradient_checkpointing_kwargs: Dict = field(default_factory=lambda: {"use_reentrant": False}, metadata={"help": "Keyword arguments for gradient checkpointing."})
    overwrite_output_dir: bool = field(default=True, metadata={"help": "Whether to overwrite the output directory."})
    eval_strategy: Optional[str] = field(default=None, metadata={"help": "HF Trainer evaluation strategy override."})
    save_strategy: Optional[str] = field(default=None, metadata={"help": "HF Trainer save strategy override."})
    save_total_limit: int = field(default=3, metadata={"help": "Maximum checkpoints to keep when saving."})
    dataloader_pin_memory: bool = field(default=True, metadata={"help": "Whether the dataloader should pin memory."})
    group_by_length: bool = field(default=False, metadata={"help": "Group sequences of similar length in HF Trainer."})
    report_to: Optional[List[str]] = field(default=None, metadata={"help": "HF Trainer report_to integrations. None derives from wandb flag."})
    ddp_find_unused_parameters: bool = field(default=True, metadata={"help": "Pass-through flag to HF Trainer for DDP unused params detection."})
    max_seq_length: int = field(default=4096, metadata={"help": "Max sequence length."})
    max_samples: int = field(default=None, metadata={"help": "Max number of samples to load."})

    # Evaluation
    eval_sample_limit: int = field(default=-1, metadata={"help": "Max number of eval samples per run."})
    eval_split: str = field(default="test", metadata={"help": "Evaluation split name."})
    per_device_eval_batch_size: int = field(default=16, metadata={"help": "Batch size for evaluation."})
    output_dir: Optional[str] = field(default=None, metadata={"help": "Directory to save eval results."})
    eval_task_names: str = field(default="", metadata={"help": "Comma-separated task names to evaluate."})
    test_speed: bool = field(default=False, metadata={"help": "Whether to test speed."})
    num_fewshot: Optional[int] = field(default=None, metadata={"help": "Number of few-shot samples to evaluate."})

    # Resume and save
    resume_path: Optional[str] = field(default=None, metadata={"help": "Path to resume checkpoint."})
    resume_training: bool = field(default=False, metadata={"help": "Whether to resume training."})
    scores_dir: List[str] = field(default_factory=list, metadata={"help": "Directory that stores importance/saliency scores."})
    mask_dir: Optional[str] = field(default=None, metadata={"help": "Directory that stores masks."})

    attn_implementation: str = "flash_attention_2"
    dtype: str = "bfloat16"
    use_vllm_fast_inference: bool = False

    def seed_all(self, seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def add_time_stamp_to_output_dir(self, time_stamp):
        if self.output_dir is not None:
            self.output_dir = self.output_dir + "_" + time_stamp
            _print(f"⚠️ Output directory: {self.output_dir}")

    def add_time_stamp_to_wandb_name(self, time_stamp):
        if self.wandb_name is not None and self.use_wandb:
            self.wandb_name = self.wandb_name + "_" + time_stamp
            _print(f"⚠️ Wandb name: {self.wandb_name}")
            
    # Process dtype, seed, eval_max_len, output+time_stamp, wandb_name+time_stamp
    def post_init(self):
        # Convert dtype to torch.dtype
        if self.dtype == "bf16":
            self.dtype = torch.bfloat16
        elif self.dtype == "fp16":
            self.dtype = torch.float16
        elif self.dtype == "fp32":
            self.dtype = torch.float32
        else:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        self.seed_all(self.seed)

        time_stamp = os.environ.get("TIME_STAMP", time.strftime("%Y%m%d_%H%M%S"))
        self.add_time_stamp_to_output_dir(time_stamp)
        self.add_time_stamp_to_wandb_name(time_stamp)


##############################
# Parse args from YAML
##############################
def parse_args_from_yaml():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    cli_args = parser.parse_args()

    with open(cli_args.config, "r") as f:
        cfg = yaml.safe_load(f)

    args = E2EArguments(**cfg)
    args.post_init()
    
    return args

##############################
# entry
##############################
def parse_args():
    args = parse_args_from_yaml()
    _print(f"all args: {args}")
    
    return args
