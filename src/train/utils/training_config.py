import inspect
import os

import torch
from transformers import TrainingArguments


TRAINING_ARGUMENTS_SUPPORTED_KEYS = set(
    inspect.signature(TrainingArguments.__init__).parameters.keys()
)


def build_training_args(args) -> TrainingArguments:
    run_name = None
    if getattr(args, "use_wandb", False):
        os.environ.setdefault("WANDB_PROJECT", getattr(args, "wandb_project", "slimmoe_kd"))
        run_name = getattr(args, "wandb_name", None)
        if run_name:
            os.environ.setdefault("WANDB_NAME", run_name)

    eval_strategy = getattr(args, "eval_strategy", None)
    if eval_strategy is None:
        eval_strategy = "steps" if getattr(args, "eval_every_n_steps", 0) > 0 else "no"
    save_strategy = getattr(args, "save_strategy", None)
    if save_strategy is None:
        save_strategy = "steps" if getattr(args, "save_every_n_steps", 0) > 0 else "no"
    default_output = os.path.join(
        "outputs",
        os.path.basename(args.model_name_or_path).replace("/", "_"),
    )
    output_dir = args.output_dir or default_output
    training_kwargs = dict(
        output_dir=output_dir,
        overwrite_output_dir=getattr(args, "overwrite_output_dir", True),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.n_epochs,
        learning_rate=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.01),
        warmup_ratio=getattr(args, "warmup_ratio", 0.05),
        max_grad_norm=getattr(args, "max_grad_norm", 0.5),
        lr_scheduler_type=getattr(args, "lr_scheduler_type", "cosine"),
        logging_steps=args.log_every_n_steps,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=args.eval_every_n_steps if eval_strategy == "steps" else None,
        save_steps=args.save_every_n_steps if save_strategy == "steps" else None,
        save_total_limit=getattr(args, "save_total_limit", 3),
        max_steps=args.max_steps if args.max_steps is not None else -1,
        bf16=args.dtype == torch.bfloat16,
        fp16=args.dtype == torch.float16,
        gradient_checkpointing=getattr(args, "gradient_checkpointing", True),
        gradient_checkpointing_kwargs=getattr(args, "gradient_checkpointing_kwargs", {"use_reentrant": False}),
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=getattr(args, "dataloader_pin_memory", True),
        dataloader_drop_last=True, 
        group_by_length=getattr(args, "group_by_length", False), # Don't group by length, as dataset is already grouped by length
        report_to=getattr(args, "report_to", None) or (["wandb"] if getattr(args, "use_wandb", False) else ["none"]),
        run_name=run_name,
        ddp_find_unused_parameters=getattr(args, "ddp_find_unused_parameters", True),
        seed=args.seed,
        ddp_broadcast_buffers=False,
        use_ademamix=getattr(args, "use_ademamix", False),
        gate_lr_mult=getattr(args, "gate_lr_mult", 0.1),
        gate_learning_rate=getattr(args, "gate_learning_rate", None),
        gate_param_keywords=getattr(args, "gate_param_keywords", None),
        enable_gate_lora=getattr(args, "enable_gate_lora", False),
        optim=getattr(args, "optim", "adamw"),
        eval_on_start=True, 
    )
    accepted_kwargs = {}
    extra_kwargs = {}
    for key, value in training_kwargs.items():
        target = (
            accepted_kwargs
            if key in TRAINING_ARGUMENTS_SUPPORTED_KEYS
            else extra_kwargs
        )
        target[key] = value

    training_args = TrainingArguments(**accepted_kwargs)
    for key, value in extra_kwargs.items():
        setattr(training_args, key, value)
    return training_args
