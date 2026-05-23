import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

@dataclass
class DataCollatorForCausalLM:
    tokenizer: Any
    max_len: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    _printed_once: bool = False

    def _aligned_max_len(self, L: int) -> int:
        if self.max_len is None:
            return L
        if self.pad_to_multiple_of is None:
            return min(L, self.max_len)
        m = int(self.pad_to_multiple_of)
        aligned_max = (self.max_len // m) * m
        return min(L, aligned_max)

    def _maybe_truncate_2d(self, x: torch.Tensor, tgt: int) -> torch.Tensor:
        if not torch.is_tensor(x) or x.dim() < 2:
            return x
        if x.size(1) <= tgt:
            return x

        trunc_side = getattr(self.tokenizer, "truncation_side", "right")
        if trunc_side == "right":
            return x[:, :tgt].contiguous()   # discard right side
        else:
            return x[:, -tgt:].contiguous()  # discard left side, keep right side


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        has_labels = "labels" in features[0]

        if has_labels:
            labels_list = [f["labels"] for f in features]
            features_wo_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        else:
            labels_list = None
            features_wo_labels = features

        batch = self.tokenizer.pad(
            features_wo_labels,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Build labels
        if labels_list is not None:
            labels_tensor = [
                lab if torch.is_tensor(lab) else torch.tensor(lab, dtype=torch.long)
                for lab in labels_list
            ]
            max_len = batch["input_ids"].shape[1]
            padded_labels = torch.full((len(labels_tensor), max_len), -100, dtype=torch.long)

            pad_side = getattr(self.tokenizer, "padding_side", "right")
            for i, lab in enumerate(labels_tensor):
                L = min(lab.numel(), max_len)
                if pad_side == "right":
                    padded_labels[i, :L] = lab[:L]
                else:
                    padded_labels[i, -L:] = lab[-L:]

            # Never compute loss on pad positions
            if "attention_mask" in batch:
                padded_labels = padded_labels.clone()
                padded_labels[batch["attention_mask"] == 0] = -100

            batch["labels"] = padded_labels

        # Optional truncate to max_len (must keep alignment consistent)
        if self.max_len is not None:
            tgt = self._aligned_max_len(batch["input_ids"].shape[1])
            for k in [
                "input_ids",
                "attention_mask",
                "labels",
                "position_ids",
                "token_type_ids",
                "decoder_input_ids",
                "decoder_attention_mask",
            ]:
                if k in batch and batch[k] is not None:
                    batch[k] = self._maybe_truncate_2d(batch[k], tgt)

        # before return batch
        if (not self._printed_once) and ("labels" in batch):
            self._printed_once = True

            bad = ((batch["attention_mask"] == 0) & (batch["labels"] != -100)).sum().item()
            print(f"[SANITY] pad positions with non -100 labels: {bad} (should be 0)")

            mask = batch["labels"] != -100
            mismatch = (batch["labels"][mask] != batch["input_ids"][mask]).sum().item()
            print(f"[SANITY] label-input mismatch on supervised positions: {mismatch} (should be 0)" )

            # optional: show one sample alignment
            i = 0
            L = batch["input_ids"].shape[1]
            print("[SANITY] padding_side:", getattr(self.tokenizer, "padding_side", None),
                  "truncation_side:", getattr(self.tokenizer, "truncation_side", None),
                  "seq_len:", L,
                  "nonpad_tokens:", int(batch["attention_mask"][i].sum().item()))

        return batch



@dataclass
class DataCollatorForCausalLMKeepLabels:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1) Extract labels (if exists), avoid tokenizer.pad() processing it
        labels_list = None
        if "labels" in features[0]:
            labels_list = [f.pop("labels") for f in features]
        
        # 2) Use tokenizer.pad to process input_ids/attention_mask, etc.
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # 3) Manually pad labels (SFT: prompt already -100)
        if labels_list is not None:
            labels_tensor = [torch.tensor(lab, dtype=torch.long) for lab in labels_list]
            max_len = batch["input_ids"].shape[1]

            padded_labels = torch.full(
                (len(labels_tensor), max_len),
                fill_value=-100,
                dtype=torch.long,
            )
            for i, lab in enumerate(labels_tensor):
                L = min(lab.numel(), max_len)
                padded_labels[i, :L] = lab[:L]
            batch["labels"] = padded_labels

        return batch


@dataclass
class ClampMaxLenCollator:
    base_collator: Callable[[Any], Dict[str, Any]]
    max_len: int
    pad_to_multiple_of: Optional[int] = None  # e.g. 8, 16, 64. Optional

    def _target_len(self, L: int) -> int:
        if self.pad_to_multiple_of is None:
            return min(L, self.max_len)
        m = int(self.pad_to_multiple_of)
        # clamp to the largest multiple of max_len that is less than or equal to max_len
        aligned_max = (self.max_len // m) * m
        return min(L, aligned_max)

    # def _clamp_2d(self, x: torch.Tensor) -> torch.Tensor:
    #     if not torch.is_tensor(x):
    #         return x
    #     if x.dim() >= 2:
    #         tgt = self._target_len(x.size(1))
    #         if x.size(1) > tgt:
    #             x = x[:, :tgt].contiguous()
    #     return x
    
    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x) or x.dim() < 2:
            return x
        tgt = self._target_len(x.size(1))  # default by dim=1
        if x.size(1) > tgt:
            return x[:, :tgt].contiguous()
        # fallback: length in the last dimension
        if x.size(-1) > tgt:
            return x[..., :tgt].contiguous()
        return x


    def __call__(self, features: Any) -> Dict[str, Any]:
        batch = self.base_collator(features)

        # The key with the longest length is most common in training
        for k in [
            "input_ids",
            "attention_mask",
            "labels",
            "position_ids",
            "token_type_ids",
            "decoder_input_ids",
            "decoder_attention_mask",
        ]:
            if k in batch and batch[k] is not None:
                batch[k] = self._clamp(batch[k])

        return batch
