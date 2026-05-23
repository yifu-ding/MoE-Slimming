import torch.nn as nn
from peft import PeftModel, LoraConfig, get_peft_model
from src.base.shared_utils import _print

def _collect_gate_linear_module_names(model):
    """
    Collect gate linear module leaf names by exact match on name segments.

    Rule:
    - Only nn.Linear modules (for Qwen-like models).
    - Only modules whose leaf name is exactly "gate" (plus optional extra leaf names).
    - No substring matching, so "gate_proj" will NOT be included unless explicitly allowed.
    
    Note:
    - For DeepSeek models, gate is nn.Parameter, not nn.Linear, so this function will NOT detect it.
    - DeepSeek gate parameters are small (~n_experts * hidden_size) and should be trained directly.
    - Adding LoRA to nn.Parameter is not supported by PEFT and not recommended.
    """
    allowed_leaf = {"gate"}  # gate only
    hits = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        leaf = name.split(".")[-1].lower()
        if leaf in allowed_leaf:
            hits.append(name)

    # For PEFT, leaf names are usually sufficient and more robust.
    leaf_names = sorted(set(n.split(".")[-1] for n in hits))
    return leaf_names


def _collect_attn_linear_module_names(model):
    """
    Collect attention linear module leaf names (target_modules) by exact leaf match.

    Rule:
    - Only nn.Linear modules.
    - Only modules under attention scope (".self_attn." or ".attention.").
    - Only leaf names in allowed set (exact match).
    - Return leaf names only (no prefixes), for PEFT target_modules.
    
    Supported architectures:
    1. Standard Transformer: q_proj, k_proj, v_proj, o_proj
    2. DeepSeek-V2 MLA architecture: 
       - Q: q_proj (or q_a_proj + q_b_proj)
       - KV: kv_a_proj_with_mqa + kv_b_proj (low-rank projection)
       - O: o_proj
    """
    # standard + DeepSeek-V2 MLA attention projection
    allowed_leaf = {
        # Standard Transformer
        "q_proj", "k_proj", "v_proj", "o_proj",
        "query", "key", "value", "out",
        "query_proj", "key_proj", "value_proj", "out_proj",
        # DeepSeek-V2 MLA architecture
        "q_a_proj", "q_b_proj",       # Q low-rank decomposition (if using q_lora_rank)
        "kv_a_proj_with_mqa",          # KV compression projection (first stage, reduce to latent space)
        "kv_b_proj",                   # KV decompression projection (second stage, decode from latent space)
        # Note: q_a_layernorm and kv_a_layernorm are LayerNorm, not Linear, will be automatically filtered
    }

    hits = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        lname = name.lower()
        # Restrict to attention scope only
        if ".self_attn." not in lname and ".attention." not in lname:
            continue

        leaf = name.split(".")[-1].lower()
        if leaf in allowed_leaf:
            hits.append(name)

    leaf_names = sorted(set(n.split(".")[-1] for n in hits))
    
    # print found attention projection layers
    if leaf_names:
        _print(f"[Attention LoRA] Found {len(leaf_names)} types of attention projection layers: {leaf_names}")
    else:
        _print("[WARNING] No attention projection layers found!")
        _print("[INFO] Checking for any Linear modules in attention scope...")
        # print first few attention related modules for debugging
        attn_modules = []
        for name, module in model.named_modules():
            lname = name.lower()
            if ".self_attn." in lname or ".attention." in lname:
                if isinstance(module, nn.Linear):
                    attn_modules.append(f"{name} (leaf: {name.split('.')[-1]})")
        if attn_modules:
            _print(f"[DEBUG] Found {len(attn_modules)} Linear modules in attention scope:")
            for m in attn_modules[:10]:
                _print(f"  {m}")
    
    return leaf_names



def _build_lora_target_modules(args, model, enable_gate_lora=False, enable_attn_lora=False):
    """
    Build target_modules list.
    - Start from args.lora_target_modules (your existing expert targets).
    - Optionally add gate targets if args.enable_gate_lora is True.
    - Optionally add attention targets if args.enable_attn_lora is True.
    """
    # Normalize base targets.
    base = getattr(args, "lora_target_modules", None)
    if base is None:
        base_targets = []
    elif isinstance(base, (list, tuple)):
        base_targets = list(base)
    else:
        # Allow comma-separated string.
        base_targets = [x.strip() for x in str(base).split(",") if x.strip()]

    targets = list(base_targets)

    if enable_gate_lora: 
        explicit = getattr(args, "gate_lora_target_modules", None)
        if explicit:
            if isinstance(explicit, (list, tuple)):
                gate_targets = list(explicit)
            else:
                gate_targets = [x.strip() for x in str(explicit).split(",") if x.strip()]
        else:
            # Auto detect by scanning module names.
            gate_targets = _collect_gate_linear_module_names(
                model,
            )

        # Merge unique, keep stable order.
        seen = set()
        merged = []
        for t in targets + gate_targets:
            if t not in seen:
                merged.append(t)
                seen.add(t)
        targets = merged

    if enable_attn_lora:
        attn_targets = _collect_attn_linear_module_names(model)

        seen = set(targets)
        for t in attn_targets:
            if t not in seen:
                targets.append(t)
                seen.add(t)
        
    return targets


def adapt_hf_model(args, model, enable_gate_lora=False, enable_attn_lora=False):
    # If already PeftModel, return directly.
    if isinstance(model, PeftModel):
        return model

    if getattr(model, "_has_lora_adapter", False):
        return model

    # Build targets: expert targets
    target_modules = _build_lora_target_modules(args, model, 
                                                enable_gate_lora=enable_gate_lora, 
                                                enable_attn_lora=enable_attn_lora)

    # Detect model dtype before adding LoRA
    import torch
    model_dtype = None
    for name, param in model.named_parameters():
        if param.dtype in [torch.float16, torch.bfloat16, torch.float32]:
            model_dtype = param.dtype
            break
    
    if model_dtype is None:
        model_dtype = torch.float32  # fallback
    
    _print(f"[LoRA] Detected model dtype: {model_dtype}")

    peft_config = LoraConfig(
        r=args.dora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        use_dora=True,
    )

    model = get_peft_model(model, peft_config)
    model._has_lora_adapter = True
    
    # Ensure all LoRA parameters match the base model dtype
    for name, param in model.named_parameters():
        if ".lora_" in name or "lora_magnitude_vector" in name:
            if param.dtype != model_dtype:
                param.data = param.data.to(model_dtype)

    # Optional: _print out what we actually targeted.
    _print(f"[LoRA] target_modules = {target_modules}")
    model.print_trainable_parameters()
    _print(f"[Load Model] ✅ adapt_hf_model finish, enable_gate_lora: {enable_gate_lora}, enable_attn_lora: {enable_attn_lora}")
    return model


