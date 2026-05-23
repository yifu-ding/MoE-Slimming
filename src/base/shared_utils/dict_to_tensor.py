import torch
from typing import Any, Iterable


def dict_to_tensor(data: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:

    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)

    def _order_dict(d: dict) -> Iterable[Any]:
        if all(isinstance(k, int) for k in d.keys()):
            keys = sorted(d.keys())
        else:
            keys = sorted(d.keys(), key=str)
        return [d[k] for k in keys]

    def _stack(values: Any) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(dtype=dtype)
        if isinstance(values, dict):
            if len(values) == 0:
                raise ValueError("dict_to_tensor: empty dict encountered")
            seq = _order_dict(values)
        elif isinstance(values, (list, tuple)):
            if len(values) == 0:
                raise ValueError("dict_to_tensor: empty list/tuple encountered")
            seq = list(values)
        else:
            return torch.as_tensor(values).to(dtype=dtype)

        tensors = [_stack(v) for v in seq]
        ref = tensors[0]
        tensors = [t.to(dtype=ref.dtype, device=ref.device) for t in tensors]
        return torch.stack(tensors, dim=0).to(dtype=dtype)

    tensor = _stack(data)
    if tensor.dim() == 0:
        return tensor.unsqueeze(0).to(dtype=dtype)
    return tensor.to(dtype=dtype)