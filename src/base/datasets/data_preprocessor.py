from typing import List

import torch
from torch.utils.data import Dataset

from src.base.datasets.load_data import load_datasets

__all__ = ["build_datasets"]

def build_messages_from_sample(sample, add_system: bool = True):
    """
    Convert different instruction sample formats into chat messages list.
    """
    messages = []

    if add_system:
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant for math and code problems."
        })

    # Alpaca format
    if "instruction" in sample and "output" in sample:
        if sample.get("input"):
            user_text = f"Instruction: {sample['instruction']}\nInput: {sample['input']}"
        else:
            user_text = sample["instruction"]
        assistant_text = sample["output"]

    # GSM8K format
    elif "question" in sample and "answer" in sample:
        user_text = sample["question"]
        assistant_text = sample["answer"]

    # HumanEval format
    elif "prompt" in sample:
        user_text = sample["prompt"]
        assistant_text = sample.get("solution", sample.get("canonical_solution", ""))

    else:
        user_text = str(sample.get("input", ""))
        assistant_text = str(sample.get("output", ""))

    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": assistant_text})
    return messages


class TokenizedTextDataset(Dataset):
    """Convert raw samples to variable-length LM inputs, with collator handling padding."""

    def __init__(
        self,
        samples: List,
        tokenizer,
        max_length: int,
        use_chat_template: bool = True,
        add_system: bool = True,
        add_lm_labels: bool = False,
    ):
        self.add_lm_labels = add_lm_labels
        texts: List[str] = []

        for s in samples:
            if isinstance(s, str):
                t = s.strip()
                if not t:
                    continue
                texts.append(t)
                continue

            if isinstance(s, dict):
                if use_chat_template:
                    messages = build_messages_from_sample(s, add_system=add_system)
                    t = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    t = t.strip()
                    if not t:
                        continue
                    texts.append(t)
                else:
                    user = str(s.get("instruction", s.get("question", s.get("prompt", s.get("input", "")))))
                    ans = str(s.get("output", s.get("answer", s.get("solution", ""))))
                    t = (user + "\n" + ans).strip()
                    if not t:
                        continue
                    texts.append(t)
                continue

            t = str(s).strip()
            if t:
                texts.append(t)

        if not texts:
            raise ValueError("no valid text samples found")

        encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=False,
        )

        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.length = len(self.input_ids)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_mask[idx], dtype=torch.long)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.add_lm_labels:
            item["labels"] = input_ids.clone()

        return item


def format_alpaca_example(ex):
    instruction = ex.get("instruction", "")
    inp = ex.get("input", "")
    output = ex.get("output", "")

    if inp.strip():
        user = f"Instruction:\n{instruction}\n\nInput:\n{inp}"
    else:
        user = f"Instruction:\n{instruction}"

    return {"user": user, "assistant": output}

def tokenize_sft(ex, tokenizer, max_length=2048):
    ex = format_alpaca_example(ex)

    # Chat template route
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # To create mask, need to know prompt token boundaries
        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["user"]}],
            tokenize=False,
            add_generation_prompt=True,  # Make assistant start token appear
        )

        full_ids = tokenizer(full, truncation=True, max_length=max_length)["input_ids"]
        prompt_ids = tokenizer(prompt_only, truncation=True, max_length=max_length)["input_ids"]

        labels = full_ids.copy()
        # mask prompt part
        n_prompt = min(len(prompt_ids), len(labels))
        for i in range(n_prompt):
            labels[i] = -100

        return {"input_ids": full_ids, "labels": labels}

    # Fallback: no chat template
    prompt = ex["user"] + "\n\nResponse:\n"
    full_text = prompt + ex["assistant"]
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
    full_ids = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]

    labels = full_ids.copy()
    n_prompt = min(len(prompt_ids), len(labels))
    for i in range(n_prompt):
        labels[i] = -100
    return {"input_ids": full_ids, "labels": labels}


def build_datasets(args, tokenizer):
    _data = load_datasets(
        args.calib_datasets[0],
        tokenizer,
        max_length=args.max_seq_length,
        max_samples=args.max_samples,
    )
    if len(_data) == 2:
        train_dataset, eval_dataset = _data

        train_tok = train_dataset.map(
            tokenize_sft, 
            fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_seq_length},
            remove_columns=train_dataset.column_names
        )
        eval_tok = eval_dataset.map(
            tokenize_sft, 
            fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_seq_length},
            remove_columns=eval_dataset.column_names
        )
    else:
        train_tok = _data

        eval_max_samples = None
        if args.eval_sample_limit and args.eval_sample_limit > 0:
            eval_max_samples = args.eval_sample_limit * args.per_device_eval_batch_size

        eval_samples = load_datasets(
            "c4",
            tokenizer,
            max_samples=eval_max_samples,
            max_length=args.max_seq_length,
        )

        eval_tok = TokenizedTextDataset(
            eval_samples,
            tokenizer,
            max_length=args.max_seq_length,
            use_chat_template=False,  
            add_system=False,
            add_lm_labels=True, 
        )
    print(f"eval_dataset (c4) sample_num: {len(eval_tok)}")
    return train_tok, eval_tok

