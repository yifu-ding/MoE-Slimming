import torch
from torch.optim import Optimizer

def _create_optimizer(self):
    if self.optimizer is not None:
        return self.optimizer

    base_lr = self.args.learning_rate

    # Option A: use multiplier (recommended, simple)
    gate_lr_mult = getattr(self.args, "gate_lr_mult", 0.2)
    gate_lr = base_lr * gate_lr_mult

    # Option B: override absolute gate lr if provided
    gate_lr_override = getattr(self.args, "gate_learning_rate", None)
    if gate_lr_override is not None:
        gate_lr = float(gate_lr_override)

    # Attention learning rate multiplier
    attn_lr_mult = getattr(self.args, "attn_lr_mult", 0.5)
    attn_lr = base_lr * attn_lr_mult

    # Option B: override absolute attn lr if provided
    attn_lr_override = getattr(self.args, "attn_learning_rate", None)
    if attn_lr_override is not None:
        attn_lr = float(attn_lr_override)

    def is_no_decay(name: str) -> bool:
        return any(nd in name for nd in ["bias", "LayerNorm.weight", "norm.weight"])

    def is_gate_param(name: str) -> bool:
        parts = name.lower().split(".")
        return "gate" in parts

    def is_attn_param(name: str) -> bool:
        """Check if parameter belongs to attention module."""
        name_lower = name.lower()
        if ".self_attn." not in name_lower and ".attention." not in name_lower:
            return False
        return any(attn_key in name_lower for attn_key in [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "query", "key", "value", "out",
            "query_proj", "key_proj", "value_proj", "out_proj",
        ])

    group_gate_decay = []
    group_gate_no_decay = []
    group_attn_decay = []
    group_attn_no_decay = []
    group_other_decay = []
    group_other_no_decay = []

    for n, p in self.model.named_parameters():
        if not p.requires_grad:
            continue

        if is_gate_param(n):
            if is_no_decay(n):
                group_gate_no_decay.append(p)
            else:
                group_gate_decay.append(p)
        elif is_attn_param(n):
            if is_no_decay(n):
                group_attn_no_decay.append(p)
            else:
                group_attn_decay.append(p)
        else:
            if is_no_decay(n):
                group_other_no_decay.append(p)
            else:
                group_other_decay.append(p)

    optimizer_grouped_parameters = []
    if group_other_decay:
        optimizer_grouped_parameters.append({
            "params": group_other_decay,
            "weight_decay": self.args.weight_decay,
            "lr": base_lr,
        })
    if group_other_no_decay:
        optimizer_grouped_parameters.append({
            "params": group_other_no_decay,
            "weight_decay": 0.0,
            "lr": base_lr,
        })
    if group_gate_decay:
        optimizer_grouped_parameters.append({
            "params": group_gate_decay,
            "weight_decay": self.args.weight_decay,
            "lr": gate_lr,
        })
    if group_gate_no_decay:
        optimizer_grouped_parameters.append({
            "params": group_gate_no_decay,
            "weight_decay": 0.0,
            "lr": gate_lr,
        })
    if group_attn_decay:
        optimizer_grouped_parameters.append({
            "params": group_attn_decay,
            "weight_decay": self.args.weight_decay,
            "lr": attn_lr,
        })
    if group_attn_no_decay:
        optimizer_grouped_parameters.append({
            "params": group_attn_no_decay,
            "weight_decay": 0.0,
            "lr": attn_lr,
        })

    print(f"[Optim] base_lr={base_lr} gate_lr={gate_lr} gate_lr_mult={gate_lr_mult} attn_lr={attn_lr} attn_lr_mult={attn_lr_mult}")
    print(f"[Optim] #params other(decay/no_decay)={len(group_other_decay)}/{len(group_other_no_decay)}, "
            f"gate(decay/no_decay)={len(group_gate_decay)}/{len(group_gate_no_decay)}, "
            f"attn(decay/no_decay)={len(group_attn_decay)}/{len(group_attn_no_decay)}")

    # self.optimizer = super().create_optimizer()
    if self.optimizer_cls_and_kwargs is not None:
        optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
    else:
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)

    # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
    # e.g. for GaLore optimizer.
    if "params" in optimizer_kwargs:
        optimizer_grouped_parameters = optimizer_kwargs.pop("params")

    # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
    # e.g. for LOMO optimizer.
    if "model" in optimizer_kwargs:
        optimizer_grouped_parameters = optimizer_kwargs.pop("model")

    # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
    # to avoid arguments conflicts.
    if "optimizer_dict" in optimizer_kwargs:
        optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

    self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    
    print("[Optim] optimizer_cls: ", optimizer_cls)
