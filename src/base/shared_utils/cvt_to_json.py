import torch
from typing import Any

def convert_to_json_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON serializable types"""
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.item())
        else:
            return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # try to convert to string, if failed return type name
        try:
            return str(obj)
        except:
            return type(obj).__name__