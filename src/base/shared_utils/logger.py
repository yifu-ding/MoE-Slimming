
import torch
import torch.distributed as dist

def _print(_string: str, flush=False):
    if is_main_process():
        print(_string, flush=flush)

def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def log_memory_usage(tag="default", print_to_console=True):
    if print_to_console:
        torch.cuda.synchronize()
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        p = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[{tag}] allocated={a:.2f}GB  reserved={r:.2f}GB  peak={p:.2f}GB")
        return {"allocated": a, "reserved": r, "peak": p}
