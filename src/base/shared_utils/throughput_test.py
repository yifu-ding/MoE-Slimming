import torch
import time
from tqdm import trange
from src.base.shared_utils.logger import _print

def make_synthetic_batch(tokenizer, batch_size, prompt_len, device):
    # decoder-only inference needs left padding; if not set, force to left
    if getattr(tokenizer, "padding_side", None) != "left":
        _print(f"[throughput] padding_side={getattr(tokenizer, 'padding_side', None)}, set to 'left' for decoder-only")
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch.manual_seed(0)
    token_id = tokenizer.eos_token_id or 0
    input_ids = torch.full(
        (batch_size, prompt_len),
        fill_value=token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

@torch.no_grad()
def warmup(model, batch, gen_len):
    _ = model.generate(
        **batch,
        max_new_tokens=gen_len,
        do_sample=False,
        use_cache=True,
    )
    torch.cuda.synchronize()

@torch.no_grad()
def benchmark_throughput(model, batch, gen_len, n_warmup=2, n_round=5):
    for _ in trange(n_warmup, desc="Throughput Warmup", leave=False):
        warmup(model, batch, gen_len)

    total_time = 0.0
    total_tokens = 0

    for _ in trange(n_round, desc="Throughput Test", leave=False):
        torch.cuda.synchronize()
        start = time.time()

        outputs = model.generate(
            **batch,
            max_new_tokens=gen_len,
            do_sample=False,  
            use_cache=True,    
        )

        torch.cuda.synchronize()
        end = time.time()

        elapsed = end - start
        total_time += elapsed

        # outputs.shape: [B, L_in + L_out]
        B = outputs.shape[0]
        total_tokens += B * gen_len

    avg_time = total_time / n_round
    avg_tokens = total_tokens / n_round

    throughput = avg_tokens / avg_time
    time_per_token_ms = avg_time / avg_tokens * 1000

    return {
        "avg_time_s": avg_time,
        "tokens_per_s": throughput,
        "ms_per_token": time_per_token_ms,
    }


def test_throughput(model, tokenizer, device, batch_size=4, prompt_len=512, gen_len=128,  n_warmup=2, n_round=5):
    model.eval().to(device)
    batch = make_synthetic_batch(tokenizer, batch_size, prompt_len, device)
    res = benchmark_throughput(model, batch, gen_len,  n_warmup=n_warmup, n_round=n_round)
    return res
