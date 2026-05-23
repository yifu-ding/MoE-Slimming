from typing import List, Callable, Dict, Tuple
from datasets import load_dataset, Dataset
from typing import List, Optional
from datasets import load_dataset

from src.base.shared_utils import _print

def _load_gsm8k(split: str = "train", 
                max_samples: int | None = None,
                max_length: Optional[int] = None) -> List[str]:
    ds: Dataset = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    if max_samples is not None:
        ds = ds.select(range(max_samples))
    texts: List[str] = []
    for ex in ds:
        question = ex["question"]
        full_answer = ex["answer"] 

        text = (
            "Question:\n"
            + question
            + "\n\nSolution:\n"
            + full_answer
        )
        texts.append(text)

    return texts

def _load_c4_validation(max_samples: int | None = None,
                        max_length: Optional[int] = None) -> List[str]:

    ds = load_dataset(
        "allenai/c4",
        data_files={
            "validation": [
                f"en/c4-validation.{i:05d}-of-00008.json.gz"
                for i in range(8)
            ]
        },
        split="validation",
        download_mode="reuse_dataset_if_exists",
        cache_dir="./.hf_cache",
    )
   
    texts = [
        x.strip()
        for x in ds["text"]
        if x is not None and x.strip() != ""
    ]
    if max_samples is not None:
        texts = texts[:max_samples]
    return texts


def load_opencode_calib(
    max_samples: Optional[int] = None,
    max_length: Optional[int] = None,
) -> List[str]:
    total_shards = 30
    shards_to_load = min(5, total_shards)  # 
    shard_base_url = (
        "https://huggingface.co/datasets/nvidia/OpenCodeReasoning"
        "/resolve/main/split_0/train-{shard:05d}-of-00030.parquet"
    )
    shard_urls = [
        shard_base_url.format(shard=shard_idx)
        for shard_idx in range(shards_to_load)
    ]

    ds = load_dataset(
        "parquet",
        data_files={"train": shard_urls},
        split="train",
    )

    texts: List[str] = []
    for s in ds:
        q = s.get("question", "")
        r = s.get("reasoning", "")
        sol = s.get("solution", "")

        text = (
            "Problem:\n"
            + q
            + "\n\nReasoning:\n"
            + r
            + "\n\nSolution:\n"
            + sol
        )
        text = text.strip()
        if text:
            texts.append(text)

    if max_samples is not None and len(texts) > max_samples:
        texts = texts[:max_samples]

    return texts



def _load_alpaca_full(
    max_samples: Optional[int] = None,
    max_length: Optional[int] = None, 
) -> Tuple[List[str], List[str]]:
    """Load the full Alpaca 52k dataset as a list of prompt-formatted texts.

    Uses the `text` field from tatsu-lab/alpaca, which already contains
    the instruction, input, and output formatted with the original Alpaca
    prompt template.
    """
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.train_test_split(test_size=0.01, seed=42)  # 1% as validation set
    train_ds = ds["train"]
    eval_ds = ds["test"]

    return (train_ds, eval_ds)



CalibLoader = Callable[..., List[str]]

CALIB_DATASET_REGISTRY: Dict[str, CalibLoader] = {
    "c4":        lambda tokenizer, max_samples=None, max_length=None: _load_c4_validation(max_samples, max_length),
    "alpaca":    lambda tokenizer, max_samples=None, max_length=None: _load_alpaca_full(max_samples, max_length),
    "openai/gsm8k": lambda tokenizer, max_samples=None, max_length=None: _load_gsm8k(max_samples=max_samples, max_length=max_length),
    "nvidia/OpenCodeReasoning": lambda tokenizer, max_samples=None, max_length=None: load_opencode_calib(max_samples, max_length),
}


def load_datasets(dataset_id: str,
                     tokenizer,
                     max_samples: int | None = None,
                     max_length: int | None = None) -> List[str]:
    if dataset_id not in CALIB_DATASET_REGISTRY:
        raise ValueError(f"Unknown calib dataset_id: {dataset_id}")
    _print(f"Loading calib dataset: {dataset_id}")
    return CALIB_DATASET_REGISTRY[dataset_id](tokenizer, max_samples=max_samples, max_length=max_length)
