from __future__ import annotations
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from alignment import get_kbit_device_map
from src.base.shared_utils import _print

def _get_max_memory(num_gpus: int, per_gpu: str = "95GiB"):
    return {i: per_gpu for i in range(num_gpus)}
        
def load_hf_model(model_name_or_path, **model_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    _print("[Load Model] load_hf_model finish!")
    return base_model, tokenizer

def set_chat_template(args, tokenizer):
    # 1) user-provided template string
    tpl = getattr(args, "chat_template", None)
    if tpl:
        tokenizer.chat_template = tpl
        return tokenizer

    # 2) user-provided template file
    path = getattr(args, "chat_template_path", None)
    if path:
        with open(path, "r") as f:
            tokenizer.chat_template = f.read()
        return tokenizer

    # 3) tokenizer already has one -> keep
    if getattr(tokenizer, "chat_template", None):
        return tokenizer

    # 4) optional fallback default
    # tokenizer.chat_template = DEFAULT_TEMPLATE
    return tokenizer


def set_tokenizer(args, tokenizer, test_only=False):
    # Padding: train right, inference left
    tokenizer.padding_side = "left" if test_only else "right"

    # Truncation: usually always keep the rightmost context for causal LM
    if getattr(args, "truncation_side", None) is not None:
        tokenizer.truncation_side = args.truncation_side
    else:
        tokenizer.truncation_side = "left"

    # Pad token: use eos if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 4096

    tokenizer = set_chat_template(args, tokenizer)
    return tokenizer



def build_quant_config(model_name_or_path: str, dtype, load_in_4bit: bool, load_in_8bit: bool):
    """
    Decide what quantization_config to pass into from_pretrained.

    Rules:
    1) If the model repo already defines config.quantization_config (e.g., FineGrainedFP8Config),
       do NOT pass BitsAndBytesConfig by default, otherwise Transformers will raise.
    2) Only build BitsAndBytesConfig when the model has no built-in quantization_config.
    3) Optional override via env FORCE_BNB=1 (use at your own risk).
    """
    if not (load_in_4bit or load_in_8bit):
        return None

    # Read config first to detect built-in quantization
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    built_in_qcfg = getattr(cfg, "quantization_config", None)

    force_bnb = bool(int(os.environ.get("FORCE_BNB", "0")))

    # If model already has a quantization_config (e.g., FineGrainedFP8Config), do not override.
    if built_in_qcfg is not None and not force_bnb:
        qname = built_in_qcfg.__class__.__name__
        _print(f"[quant] Detected built-in quantization_config={qname}; skip BitsAndBytesConfig.")
        return None

    # Otherwise, apply BitsAndBytes quantization
    _print("[quant] Using BitsAndBytesConfig.")
    return BitsAndBytesConfig(
        load_in_4bit=bool(load_in_4bit),
        load_in_8bit=bool(load_in_8bit),
        bnb_4bit_quant_type="nf4",  # or "fp4"
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def get_model(args, max_memory = None, device_map = None):
       
    dtype = getattr(args, "dtype", torch.bfloat16)

    revision = getattr(args, "model_revision", None)
    trust_remote_code = getattr(args, "trust_remote_code", False)
    attn_implementation = getattr(args, "attn_implementation", None)
    load_in_4bit = getattr(args, "load_in_4bit", False)
    load_in_8bit = getattr(args, "load_in_8bit", False)
    gradient_checkpointing = getattr(args, "gradient_checkpointing", False)

    quant_config = build_quant_config(
        args.model_name_or_path,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
            
    # use auto to let model switch to multiple GPUs, use when testing, use get_kbit_device_map() when training
    num_gpus = int(getattr(args, "num_gpus", torch.cuda.device_count()))
    per_gpu_mem = getattr(args, "per_gpu_mem", "95GiB")
    max_memory = _get_max_memory(num_gpus, per_gpu=per_gpu_mem) if max_memory is None else max_memory
    
    if num_gpus > 1 and args.test_only:  # use auto when testing, use get_kbit_device_map() when training
        device_map = 'auto'
    else:
        device_map = get_kbit_device_map() if device_map is None else device_map  # get current cuda

    model_kwargs = dict(
        revision=revision,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,                # note: transformers parameter name is usually torch_dtype
        use_cache=False if gradient_checkpointing else True,
        device_map=device_map, # use when training on multiple GPUs
        max_memory=max_memory,
    )
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    

    model, tokenizer = load_hf_model(args.model_name_or_path, **model_kwargs)

    tokenizer = set_tokenizer(args, tokenizer, test_only=args.test_only)
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
