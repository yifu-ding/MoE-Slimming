from src.base.shared_utils import _print

def patch_qwen3_moe_load_balancing_loss():
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    import transformers.models.qwen3_moe.modeling_qwen3_moe as m

    def _lb_loss_all_ranks(router_logits, top_k, attention_mask=None):
        """
        DDP-safe load-balancing loss.
        Key points:
        1) Compute per-layer loss, do NOT torch.cat across layers.
        2) Infer E from router_logits_tensor.shape[-1], ignore passed num_experts.
        3) Do one all_reduce per layer with a packed stats tensor.
        """

        def _one(router_logits_tensor):
            # Infer real expert count for this tensor.
            E = int(router_logits_tensor.shape[-1])
            tk = int(top_k)
            if tk > E:
                tk = E
            if tk <= 0 or E <= 0:
                return torch.zeros((), device=router_logits_tensor.device, dtype=router_logits_tensor.dtype)

            logits = router_logits_tensor.float()
            probs = F.softmax(logits, dim=-1)

            # probs: [B, S, E] or [T, E]
            if probs.dim() == 3:
                probs2 = probs.reshape(-1, E)  # [T, E]
                am = attention_mask
                if am is not None:
                    am = am.reshape(-1).to(device=probs2.device, dtype=probs2.dtype)
            elif probs.dim() == 2:
                probs2 = probs  # [T, E]
                am = attention_mask
                if am is not None:
                    am = am.reshape(-1).to(device=probs2.device, dtype=probs2.dtype)
            else:
                raise RuntimeError(f"Unsupported router_logits shape: {tuple(probs.shape)}")

            T = probs2.shape[0]

            # If mask length mismatch, fall back to all-ones mask.
            if am is not None and am.numel() != T:
                am = torch.ones(T, device=probs2.device, dtype=probs2.dtype)

            # Top-k selection.
            _, top_idx = torch.topk(probs2, k=tk, dim=-1)  # [T, tk]
            expert_mask = F.one_hot(top_idx, num_classes=E).sum(dim=1).float()  # [T, E]

            if am is not None:
                amf = am.unsqueeze(-1)  # [T, 1]
                expert_mask = expert_mask * amf
                probs2_masked = probs2 * amf
                token_count = amf.sum()  # scalar
            else:
                probs2_masked = probs2
                token_count = torch.tensor(float(T), device=probs2.device, dtype=probs2.dtype)

            expert_mask_sum = expert_mask.sum(dim=0)              # [E]
            probs_sum = probs2_masked.sum(dim=0)                  # [E]
            token_count_sum = token_count.to(dtype=probs2.dtype)  # scalar

            token_count_sum = token_count_sum.clamp(min=1.0)

            # All-reduce packed stats, outside autograd.
            if dist.is_available() and dist.is_initialized():
                stats = torch.empty(2 * E + 1, device=probs2.device, dtype=torch.float32)
                stats[:E] = expert_mask_sum.to(torch.float32)
                stats[E:2 * E] = probs_sum.to(torch.float32)
                stats[2 * E] = token_count_sum.to(torch.float32)

                stats = stats.detach()
                with torch.no_grad():
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

                expert_mask_sum = stats[:E]
                probs_sum = stats[E:2 * E]
                token_count_sum = stats[2 * E].clamp(min=1.0)

            tokens_per_expert = expert_mask_sum / (token_count_sum * float(tk))  # [E]
            router_prob_per_expert = probs_sum / token_count_sum                  # [E]
            loss = (tokens_per_expert * router_prob_per_expert).sum() * float(E)

            return loss.to(dtype=router_logits_tensor.dtype, device=router_logits_tensor.device)

        # router_logits can be a tensor, or a list/tuple of per-layer tensors.
        if isinstance(router_logits, (tuple, list)):
            # Determine fallback E for None entries.
            ref_E = []
            for x in router_logits:
                ref_E.append(None if x is None else int(x.shape[-1]))

            if all(e is None for e in ref_E):
                device = attention_mask.device if attention_mask is not None else "cpu"
                return torch.tensor(0.0, device=device)

            first_E = next(e for e in ref_E if e is not None)
            ref_E = [e if e is not None else first_E for e in ref_E]

            losses = []
            for i, x in enumerate(router_logits):
                if x is None:
                    E_i = ref_E[i]
                    device = attention_mask.device if attention_mask is not None else "cuda"
                    dummy = torch.zeros(1, E_i, device=device, dtype=torch.float32)
                    losses.append(_one(dummy))
                else:
                    losses.append(_one(x))

            return torch.stack(losses).mean()

        return _one(router_logits)

    def patched_load_balancing_loss_func(*args, **kwargs):
        # Qwen3 forward calls: load_balancing_loss_func(gate_logits, num_experts, top_k, attention_mask)
        router_logits = kwargs.get("gate_logits", kwargs.get("router_logits", args[0] if len(args) > 0 else None))
        top_k = kwargs.get("top_k", args[2] if len(args) > 2 else kwargs.get("num_experts_per_tok", None))
        attention_mask = kwargs.get("attention_mask", args[3] if len(args) > 3 else None)

        if router_logits is None or top_k is None:
            raise RuntimeError("load_balancing_loss_func: missing required args (router_logits/top_k)")

        return _lb_loss_all_ranks(router_logits, top_k, attention_mask)

    m.load_balancing_loss_func = patched_load_balancing_loss_func
    _print("[patch] qwen3_moe load_balancing_loss_func hard-replaced (per-layer, DDP safe, E inferred from logits).")

def patch_qwen2_moe_load_balancing_loss():
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    import transformers.models.qwen2_moe.modeling_qwen2_moe as m

    def _lb_loss_all_ranks(router_logits, top_k, attention_mask=None):
        """
        DDP-safe load-balancing loss.
        Critical: infer E from router_logits_tensor.shape[-1] (the real expert count),
        do not trust the passed num_experts which may be stale after pruning.
        """

        def _one(router_logits_tensor):
            # Infer real expert count for this tensor.
            E = int(router_logits_tensor.shape[-1])
            tk = int(top_k)
            if tk > E:
                tk = E

            logits = router_logits_tensor.float()
            probs = F.softmax(logits, dim=-1)

            # probs: [B, S, E] or [T, E]
            if probs.dim() == 3:
                probs2 = probs.reshape(-1, E)  # [B*S, E]
                am = attention_mask
                if am is not None:
                    am = am.reshape(-1).to(device=probs2.device, dtype=probs2.dtype)
            elif probs.dim() == 2:
                probs2 = probs  # [T, E]
                am = attention_mask
                if am is not None:
                    am = am.reshape(-1).to(device=probs2.device, dtype=probs2.dtype)
            else:
                raise RuntimeError(f"Unsupported router_logits shape: {tuple(probs.shape)}")

            T = probs2.shape[0]

            # If mask length mismatch, fall back to all-ones mask to keep control flow aligned.
            if am is not None and am.numel() != T:
                am = torch.ones(T, device=probs2.device, dtype=probs2.dtype)

            # Top-k selection.
            _, top_idx = torch.topk(probs2, k=tk, dim=-1)  # [T, tk]
            expert_mask = F.one_hot(top_idx, num_classes=E).sum(dim=1).float()  # [T, E]

            if am is not None:
                amf = am.unsqueeze(-1)  # [T, 1]
                expert_mask = expert_mask * amf
                probs2_masked = probs2 * amf
                token_count = amf.sum()  # scalar
            else:
                probs2_masked = probs2
                token_count = torch.tensor(float(T), device=probs2.device, dtype=probs2.dtype)

            expert_mask_sum = expert_mask.sum(dim=0)            # [E]
            probs_sum = probs2_masked.sum(dim=0)                # [E]
            token_count_sum = token_count.to(dtype=probs2.dtype)  # scalar

            # Single all_reduce for stability. Must not be in autograd graph.
            token_count_sum = token_count_sum.clamp(min=1.0)
            if dist.is_available() and dist.is_initialized():
                stats = torch.empty(2 * E + 1, device=probs2.device, dtype=torch.float32)
                stats[:E] = expert_mask_sum.to(torch.float32)
                stats[E:2 * E] = probs_sum.to(torch.float32)
                stats[2 * E] = token_count_sum.to(torch.float32)

                stats = stats.detach()
                with torch.no_grad():
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

                expert_mask_sum = stats[:E]
                probs_sum = stats[E:2 * E]
                token_count_sum = stats[2 * E].clamp(min=1.0)
            else:
                # already clamped
                pass

            tokens_per_expert = expert_mask_sum / (token_count_sum * float(tk))  # [E]
            router_prob_per_expert = probs_sum / token_count_sum                  # [E]
            loss = (tokens_per_expert * router_prob_per_expert).sum() * float(E)

            return loss.to(dtype=router_logits_tensor.dtype, device=router_logits_tensor.device)

        # Handle tuple/list of per-layer router logits.
        if isinstance(router_logits, (tuple, list)):
            # Precompute a reference E for each position to construct dummy tensors if needed.
            ref_E = []
            for x in router_logits:
                if x is None:
                    ref_E.append(None)
                else:
                    ref_E.append(int(x.shape[-1]))

            # Fallback: if everything is None, return 0.
            if all(e is None for e in ref_E):
                device = attention_mask.device if attention_mask is not None else "cpu"
                return torch.tensor(0.0, device=device)

            # Fill missing E with the first non-None E to keep shapes deterministic.
            first_E = next(e for e in ref_E if e is not None)
            ref_E = [e if e is not None else first_E for e in ref_E]

            losses = []
            for i, x in enumerate(router_logits):
                if x is None:
                    # Dummy logits with correct E for this slot.
                    E_i = ref_E[i]
                    device = attention_mask.device if attention_mask is not None else "cuda"
                    dummy = torch.zeros(1, E_i, device=device, dtype=torch.float32)
                    losses.append(_one(dummy))
                else:
                    losses.append(_one(x))
            return torch.stack(losses).mean()

        return _one(router_logits)

    def patched_load_balancing_loss_func(*args, **kwargs):
        router_logits = kwargs.get("router_logits", args[0] if len(args) > 0 else None)
        top_k = kwargs.get("top_k", args[2] if len(args) > 2 else kwargs.get("num_experts_per_tok", None))
        attention_mask = kwargs.get("attention_mask", args[3] if len(args) > 3 else None)

        if router_logits is None or top_k is None:
            raise RuntimeError("load_balancing_loss_func: missing required args (router_logits/top_k)")

        # Important: ignore num_experts argument entirely, infer from router_logits.
        return _lb_loss_all_ranks(router_logits, top_k, attention_mask)

    m.load_balancing_loss_func = patched_load_balancing_loss_func
    _print("[patch] qwen2_moe load_balancing_loss_func hard-replaced (DDP safe, E inferred from logits).")
