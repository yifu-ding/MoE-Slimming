import torch
from torch.nn import functional as F


def to_device_dtype(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device_dtype(v, device) for v in obj)
    if isinstance(obj, dict):
        return {k: to_device_dtype(v, device) for k, v in obj.items()}
    return obj

def format_name(name):
    return name.replace("/", "_")
