"""
LoRA loading utility functions

Handle compatibility loading of checkpoints saved after gradual mask wrapping
"""

import os
import re
import json
import torch
from typing import Dict, Any, Optional
from safetensors import safe_open
from peft import PeftModel, LoraConfig, get_peft_model, set_peft_model_state_dict
from src.base.shared_utils import _print

# ==================== Regular Expression Definitions ====================

# Match patterns like .q_proj.proj. (HeadOutWithMask wrapper)
_PROJ_STRIP_RE = re.compile(
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.proj\."
)

# Match duplicate patterns like .q_proj.q_proj.
_DUP_LEAF_RE = re.compile(
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.\1\."
)

# Match .experts.0.expert. pattern (MaskedMoEExpert wrapper)
_EXPERT_SEG_RE = re.compile(r"\.experts\.(\d+)\.expert\.")


# ==================== Key Name Conversion Functions ====================

def _strip_proj_segment(k: str) -> str:
    """Remove .proj. wrapper layer from attention projection"""
    return _PROJ_STRIP_RE.sub(r".\1.", k)


def _strip_dup_leaf(k: str) -> str:
    """Remove duplicate module names (e.g. .q_proj.q_proj.)"""
    return _DUP_LEAF_RE.sub(r".\1.", k)


def _strip_expert_segment(k: str) -> str:
    """Remove .expert. wrapper layer from expert"""
    return _EXPERT_SEG_RE.sub(r".experts.\1.", k)


def _normalize_magnitude_vector_key(k: str) -> str:
    """
    Normalize DoRA magnitude vector key names:
      - ...lora_magnitude_vector           -> ...lora_magnitude_vector.weight
      - ...lora_magnitude_vector.weight    -> ...lora_magnitude_vector.weight (keep)
      - ...lora_magnitude_vector.default.weight -> ...lora_magnitude_vector.weight (remove .default.)
    
    Note: Keep .weight suffix, set_peft_model_state_dict will add .default. in the middle
    """
    if ".lora_magnitude_vector" not in k:
        return k

    # If already in correct format, keep unchanged
    if k.endswith(".lora_magnitude_vector.weight"):
        return k
    
    # If has .default.weight, remove .default. and keep .weight
    if ".lora_magnitude_vector.default.weight" in k:
        return k.replace(".lora_magnitude_vector.default.weight", ".lora_magnitude_vector.weight")
    
    # If just lora_magnitude_vector, add .weight
    if k.endswith(".lora_magnitude_vector"):
        return k + ".weight"
    
    return k


def _normalize_ab_default(k: str) -> str:
    """
    Normalize A/B key names:
      - ...lora_A.weight -> ...lora_A.weight (keep)
      - ...lora_B.weight -> ...lora_B.weight (keep)
      - ...lora_A.default.weight -> ...lora_A.weight (remove .default.)
      - ...lora_B.default.weight -> ...lora_B.weight (remove .default.)
    
    Note: Keep .weight suffix, set_peft_model_state_dict will add .default. in the middle
    """
    for tag in ["lora_A", "lora_B"]:
        # If has .default.weight, remove .default. and keep .weight
        pattern = f".{tag}.default.weight"
        replacement = f".{tag}.weight"
        if pattern in k:
            k = k.replace(pattern, replacement)
    return k


def _maybe_add_base_model_prefix(k: str) -> str:
    """If key name doesn't have base_model. prefix, add it"""
    if k.startswith("base_model."):
        return k
    return "base_model." + k


def _is_adapter_key(k: str) -> bool:
    """Check if it's a LoRA adapter parameter"""
    return ("lora_" in k) or ("lora_magnitude_vector" in k)


def remap_adapter_key(k: str) -> str:
    """
    Complete key name remapping process
    
    Handles:
    1. Remove extra layers introduced by gradual mask wrappers
    2. Normalize PEFT version differences
    3. Add necessary prefixes
    """
    k = _strip_proj_segment(k)
    k = _strip_dup_leaf(k)
    k = _strip_expert_segment(k)
    k = _normalize_ab_default(k)
    k = _normalize_magnitude_vector_key(k)
    k = _maybe_add_base_model_prefix(k)
    return k


def convert_checkpoint_keys_for_unwrapped_model(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert all key names in checkpoint, remove extra layers introduced by gradual mask wrappers
    
    Args:
        state_dict: Original state dict
        
    Returns:
        Converted state dict
    """
    new_state_dict = {}
    converted_count = 0
    
    for key, value in state_dict.items():
        new_key = remap_adapter_key(key)
        new_state_dict[new_key] = value
        if new_key != key:
            converted_count += 1
    
    if converted_count > 0:
        _print(f"[LoRA Loading] Key name conversion: converted {converted_count}/{len(state_dict)} keys")
    
    return new_state_dict


# ==================== LoRA Loading Functions ====================

def load_lora_adapter_with_compatibility(
    model,
    adapter_dir: str,
    adapter_name: str = "default",
    verbose: bool = True,
    strict: bool = False,
) -> PeftModel:
    """
    Compatibility loading of LoRA adapter
    
    Automatically handles:
    1. Checkpoints saved after gradual mask wrapping
    2. Key name format differences across PEFT versions
    3. Only load LoRA parameters, ignore base model weights
    
    Args:
        model: Base model
        adapter_dir: Adapter checkpoint directory
        adapter_name: Adapter name
        verbose: Whether to print detailed information
        strict: Whether to strictly check all keys match
        
    Returns:
        PeftModel with LoRA loaded
    """
    if verbose:
        _print(f"[LoRA Loading] Loading adapter from {adapter_dir}")
    
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, 'r') as f:
        adapter_config = json.load(f)
    
    # Load weight file
    adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(adapter_dir, "adapter_model.bin")
        if not os.path.exists(adapter_file):
            raise FileNotFoundError(f"Adapter weight file not found: {adapter_dir}")
    
    if adapter_file.endswith(".safetensors"):
        with safe_open(adapter_file, framework="pt", device="cpu") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
    else:
        state_dict = torch.load(adapter_file, map_location="cpu")
    
    if verbose:
        _print(f"[LoRA Loading] Weight file loaded, {len(state_dict)} keys total")
    
    needs_conversion = any('.expert.' in k or '.proj.proj.' in k or '.k_proj.proj.' in k 
                          for k in state_dict.keys())
    
    if needs_conversion:
        if verbose:
            _print(f"[LoRA Loading] Detected wrapped key names, converting...")
        state_dict = convert_checkpoint_keys_for_unwrapped_model(state_dict)
    
    # Keep only LoRA-related weights
    lora_state_dict = {k: v for k, v in state_dict.items() if _is_adapter_key(k)}
    
    if verbose:
        _print(f"[LoRA Loading] After filtering, kept {len(lora_state_dict)} LoRA parameters (original {len(state_dict)})")
    
    peft_config = LoraConfig(**{k: v for k, v in adapter_config.items() 
                                if k not in ['base_model_name_or_path', 'inference_mode']})
    peft_config.inference_mode = False
    
    model = get_peft_model(model, peft_config)
    if verbose:
        _print(f"[LoRA Loading] PEFT model structure initialized")
    
    incompatible = set_peft_model_state_dict(model, lora_state_dict, adapter_name=adapter_name)
    
    lora_missing_keys = [k for k in incompatible.missing_keys if _is_adapter_key(k)]
    lora_unexpected_keys = [k for k in incompatible.unexpected_keys if _is_adapter_key(k)]
    
    # Report loading status
    loaded_lora_keys = len(lora_state_dict) - len(lora_missing_keys)
    success_rate = (loaded_lora_keys / len(lora_state_dict) * 100) if len(lora_state_dict) > 0 else 0
    
    if verbose:
        if lora_missing_keys:
            _print(f"[LoRA Loading] ⚠️  Warning: {len(lora_missing_keys)} LoRA parameters not found")
            if len(lora_missing_keys) <= 10:
                for k in lora_missing_keys:
                    _print(f"  - Missing: {k}")
            else:
                _print(f"  - First 10 missing keys: {lora_missing_keys[:10]}")
        
        if lora_unexpected_keys:
            _print(f"[LoRA Loading] Warning: {len(lora_unexpected_keys)} additional LoRA keys")
            if len(lora_unexpected_keys) <= 5:
                for k in lora_unexpected_keys:
                    _print(f"  - Extra: {k}")
        
        _print(f"[LoRA Loading] ✅ Loading complete: {loaded_lora_keys}/{len(lora_state_dict)} LoRA parameters loaded successfully ({success_rate:.1f}%)")
    
    if strict and lora_missing_keys:
        raise RuntimeError(f"Strict mode: {len(lora_missing_keys)} LoRA parameters missing")
    
    return model


def load_adapter_with_remap(
    base_model,
    adapter_dir: str,
    adapter_name: str = "default",
    verbose: bool = True
) -> PeftModel:
    """
    Load and remap LoRA adapter (compatible with old interface)
    
    This is an alias for load_lora_adapter_with_compatibility, maintained for backward compatibility
    """
    return load_lora_adapter_with_compatibility(
        base_model,
        adapter_dir,
        adapter_name=adapter_name,
        verbose=verbose,
        strict=False
    )

